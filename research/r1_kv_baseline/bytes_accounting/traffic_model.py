"""KV 流量模型：封装 ``cache_path`` 的 ``bytes_breakdown``（metrics v1.1）。

本模块不另起记账口径。单步 bytes/token 与 D(L_in, L_out) 积分都先把
真实 cache 填到目标长度，再读四项分解。C0 只接受 ``fp16`` / ``C0``，禁止
原生 HF attention 路径（会少算 meta）。

``n_elem`` 按协议 §3.3 取 **每 token 的 K+V 标量个数**
（``2 * n_kv * d``），使 ``N * n_elem`` 等于该步读取的元素总数；
FP16 无 meta 时 ``b_eff = 16``。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache_path"
if str(_CACHE_PATH) not in sys.path:
    sys.path.insert(0, str(_CACHE_PATH))

from attention_with_cache import AttentionWithCache  # noqa: E402
from kv_cache import BytesBreakdown  # noqa: E402
from paged_cache import DEFAULT_PAGE_SIZE, DEFAULT_PTE_BYTES  # noqa: E402

# 禁止把 kivi_eval 的原生 HF C0 别名当流量入口
_FORBIDDEN_C0 = frozenset({"hf", "baseline", "native"})

_FORMAT_ALIASES: dict[str, str] = {
    "fp16": "fp16",
    "c0": "fp16",
    "int8": "int8",
    "c1": "int8",
    "int4": "int4",
    "c2": "int4",
    "int4_bdr": "int4_bdr",
    "int4bdr": "int4_bdr",
    "bdr": "int4_bdr",
    "c3": "int4_bdr",
    "kivi2": "kivi2",
    "kivi_2": "kivi2",
    "kivi_2bit": "kivi2",
    "c4": "kivi2",
    "kivi4": "kivi4",
    "kivi_4": "kivi4",
    "kivi_4bit": "kivi4",
    "c5": "kivi4",
}

_PROTOCOL_ID: dict[str, str] = {
    "fp16": "C0",
    "int8": "C1",
    "int4": "C2",
    "int4_bdr": "C3",
    "kivi2": "C4",
    "kivi4": "C5",
}

LAYOUTS: tuple[str, str] = ("contiguous", "paged")
FORMATS: tuple[str, ...] = ("fp16", "int8", "int4", "int4_bdr", "kivi2", "kivi4")

__all__ = [
    "FORMATS",
    "LAYOUTS",
    "LLAMA_31_8B",
    "M4_SYNTH",
    "LayerGeometry",
    "PressureTraffic",
    "StepTraffic",
    "breakdown_total",
    "canonical_kv_format",
    "effective_bits",
    "measure_decode_pressure",
    "measure_step",
    "n_elem_per_token",
    "protocol_format_id",
]


@dataclass(frozen=True)
class LayerGeometry:
    """单层 KV 几何；``num_layers`` 只做全模乘法，不改变 ``b_eff``。

    ``num_kv_heads`` 是实际 K/V head 数（GQA 不用 query head）。
    """

    num_kv_heads: int
    head_dim: int
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.num_kv_heads <= 0 or self.head_dim <= 0 or self.num_layers <= 0:
            raise ValueError(
                f"几何须为正：heads={self.num_kv_heads}, dim={self.head_dim}, "
                f"layers={self.num_layers}"
            )


# M4 paged_layout 合成几何（对拍 REPORT，不是 8B）
M4_SYNTH = LayerGeometry(num_kv_heads=8, head_dim=64, num_layers=1)
# 协议主 Pareto：Llama-3.1-8B-Instruct（GQA）
LLAMA_31_8B = LayerGeometry(num_kv_heads=8, head_dim=128, num_layers=32)


@dataclass(frozen=True)
class StepTraffic:
    """单层 decode step 读长度为 ``n`` 的 KV 时的流量（可乘到全模）。"""

    n: int
    kv_format: str
    protocol_id: str
    layout: str
    num_layers: int
    per_layer: BytesBreakdown
    all_layers: BytesBreakdown
    bytes_per_token: int
    b_eff: float
    n_elem: int

    def as_dict(self) -> dict[str, int | float | str]:
        """便于实验脚本落 JSON。"""
        b = self.all_layers
        return {
            "n": self.n,
            "kv_format": self.kv_format,
            "protocol_id": self.protocol_id,
            "layout": self.layout,
            "num_layers": self.num_layers,
            "payload": b.payload,
            "scale": b.scale,
            "zp": b.zp,
            "page": b.page,
            "bytes_per_token": self.bytes_per_token,
            "b_eff": self.b_eff,
            "n_elem": self.n_elem,
        }


@dataclass(frozen=True)
class PressureTraffic:
    """压力点 D(L_in, L_out)：全程 KV 读 / L_out，加末步单步。"""

    l_in: int
    l_out: int
    kv_format: str
    protocol_id: str
    layout: str
    num_layers: int
    total_kv_read: int
    mean_bytes_per_token: float
    last_step: StepTraffic

    def as_dict(self) -> dict[str, int | float | str]:
        """便于实验脚本落 JSON。"""
        out: dict[str, int | float | str] = {
            "l_in": self.l_in,
            "l_out": self.l_out,
            "kv_format": self.kv_format,
            "protocol_id": self.protocol_id,
            "layout": self.layout,
            "num_layers": self.num_layers,
            "total_kv_read": self.total_kv_read,
            "mean_bytes_per_token": self.mean_bytes_per_token,
        }
        for key, value in self.last_step.as_dict().items():
            out[f"last_{key}"] = value
        return out


def canonical_kv_format(kv_format: str) -> str:
    """把 C0–C5 / 别名规范成 ``fp16``…``kivi4``。

    参数
        kv_format: 协议 ID 或 ``get_codec`` 别名。

    返回
        规范短名。

    异常
        原生 HF C0 别名，或未知格式。
    """
    key = kv_format.strip().lower().replace("-", "_").replace("+", "_")
    if key in _FORBIDDEN_C0:
        raise ValueError(f"C0 流量必须走 FP16 codec，禁止 {kv_format!r}（原生 HF 会少算 meta）")
    if key not in _FORMAT_ALIASES:
        raise ValueError(
            f"未知 kv_format={kv_format!r}；支持 fp16/int8/int4/int4_bdr/kivi2/kivi4 或 C0–C5"
        )
    return _FORMAT_ALIASES[key]


def protocol_format_id(kv_format: str) -> str:
    """规范短名或别名 → ``C0``…``C5``。"""
    return _PROTOCOL_ID[canonical_kv_format(kv_format)]


def n_elem_per_token(geometry: LayerGeometry) -> int:
    """每 token 的 K+V 标量个数：``2 * n_kv * d``。"""
    return 2 * geometry.num_kv_heads * geometry.head_dim


def breakdown_total(b: BytesBreakdown) -> int:
    """四项之和（payload + scale + zp + page）。"""
    return b.payload + b.scale + b.zp + b.page


def effective_bits(b: BytesBreakdown, n: int, n_elem: int) -> float:
    """协议 §3.3 的 ``b_eff``（bit / 元素），含元数据与页表。"""
    if n <= 0 or n_elem <= 0:
        raise ValueError(f"n 与 n_elem 须为正，得到 n={n}, n_elem={n_elem}")
    return 8.0 * breakdown_total(b) / (n * n_elem)


def _scale_breakdown(b: BytesBreakdown, n_layers: int) -> BytesBreakdown:
    if n_layers == 1:
        return b
    return BytesBreakdown(
        payload=b.payload * n_layers,
        scale=b.scale * n_layers,
        zp=b.zp * n_layers,
        page=b.page * n_layers,
    )


def _make_attn(
    geometry: LayerGeometry,
    kv_format: str,
    layout: str,
    *,
    page_size: int,
    pte_bytes: int,
    device: torch.device | None,
) -> AttentionWithCache:
    if layout not in LAYOUTS:
        raise ValueError(f"layout 须为 contiguous / paged，得到 {layout!r}")
    return AttentionWithCache(
        kv_format,
        num_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        layout=layout,
        page_size=page_size,
        pte_bytes=pte_bytes,
        device=device,
    )


def _append_zeros(attn: AttentionWithCache, n_tokens: int) -> None:
    """写入占位 K/V；载荷/meta 形状与数值无关。"""
    if n_tokens <= 0:
        raise ValueError(f"n_tokens 须为正，得到 {n_tokens}")
    zeros = torch.zeros(
        n_tokens,
        attn.num_heads,
        attn.head_dim,
        device=attn.device,
        dtype=torch.float32,
    )
    attn.cache.append(zeros, zeros)


def _pack_step(
    geometry: LayerGeometry,
    *,
    n: int,
    kv_format: str,
    layout: str,
    per_layer: BytesBreakdown,
) -> StepTraffic:
    n_elem = n_elem_per_token(geometry)
    all_layers = _scale_breakdown(per_layer, geometry.num_layers)
    return StepTraffic(
        n=n,
        kv_format=kv_format,
        protocol_id=_PROTOCOL_ID[kv_format],
        layout=layout,
        num_layers=geometry.num_layers,
        per_layer=per_layer,
        all_layers=all_layers,
        bytes_per_token=breakdown_total(all_layers),
        b_eff=effective_bits(per_layer, n, n_elem),
        n_elem=n_elem,
    )


def measure_step(
    geometry: LayerGeometry,
    *,
    kv_format: str,
    n: int,
    layout: str = "contiguous",
    page_size: int = DEFAULT_PAGE_SIZE,
    pte_bytes: int = DEFAULT_PTE_BYTES,
    device: torch.device | None = None,
) -> StepTraffic:
    """把 cache 填到长度 ``n``，读取 ``bytes_breakdown``。

    参数
        geometry: KV head / head_dim / 层数。
        kv_format: C0–C5 或其 ``get_codec`` 别名；C0 必须是 FP16 codec。
        n: 已缓存 token 数（生成第 n+1 个 token 时读取）。
        layout: ``contiguous`` 或 ``paged``。
        page_size / pte_bytes: 仅 paged；默认与 metrics §8.1 一致。
        device: 默认 CPU。

    返回
        单步流量；``bytes_per_token`` 为全模四项之和（``num_layers=1`` 时即单层）。
    """
    if n <= 0:
        raise ValueError(f"n 须为正，得到 {n}")
    name = canonical_kv_format(kv_format)
    attn = _make_attn(
        geometry,
        name,
        layout,
        page_size=page_size,
        pte_bytes=pte_bytes,
        device=device,
    )
    _append_zeros(attn, n)
    return _pack_step(
        geometry, n=n, kv_format=name, layout=layout, per_layer=attn.bytes_breakdown()
    )


def measure_decode_pressure(
    geometry: LayerGeometry,
    *,
    kv_format: str,
    l_in: int,
    l_out: int,
    layout: str = "contiguous",
    page_size: int = DEFAULT_PAGE_SIZE,
    pte_bytes: int = DEFAULT_PTE_BYTES,
    device: torch.device | None = None,
) -> PressureTraffic:
    """压力点 D(L_in, L_out)。

    预填 L_in 后，对 t = 0..L_out-1 读取 N = L_in+t 的 KV，
    再追加 1 token（末步不再追加）。全程合计 / L_out 为均值 bytes/token；
    末步 N = L_in + L_out - 1。
    """
    if l_in <= 0 or l_out <= 0:
        raise ValueError(f"L_in / L_out 须为正，得到 D({l_in},{l_out})")
    name = canonical_kv_format(kv_format)
    attn = _make_attn(
        geometry,
        name,
        layout,
        page_size=page_size,
        pte_bytes=pte_bytes,
        device=device,
    )
    _append_zeros(attn, l_in)
    total = 0
    last: StepTraffic | None = None
    for t in range(l_out):
        n = l_in + t
        step = _pack_step(
            geometry,
            n=n,
            kv_format=name,
            layout=layout,
            per_layer=attn.bytes_breakdown(),
        )
        total += step.bytes_per_token
        last = step
        if t < l_out - 1:
            _append_zeros(attn, 1)
    assert last is not None
    return PressureTraffic(
        l_in=l_in,
        l_out=l_out,
        kv_format=name,
        protocol_id=_PROTOCOL_ID[name],
        layout=layout,
        num_layers=geometry.num_layers,
        total_kv_read=total,
        mean_bytes_per_token=total / l_out,
        last_step=last,
    )
