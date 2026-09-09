"""敏感性 误差—流量敏感性：层 / 头 / token 位置，以及 decode 累积初趋势。

张量误差相对 **未量化 float** golden（与 ``codec_compare`` 同口径）。
整模 PPL 层消融相对 C0 codec，由实验脚本负责。
流量一律走 ``traffic_model``（C0 = FP16 codec）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache_path"
if str(_CACHE_PATH) not in sys.path:
    sys.path.insert(0, str(_CACHE_PATH))

from attention_with_cache import (  # noqa: E402
    AttentionWithCache,
    scaled_dot_product_attention,
)
from kv_cache import BytesBreakdown  # noqa: E402

from .traffic_model import (  # noqa: E402
    FORMATS,
    LLAMA_31_8B,
    QWEN25_05B,
    LayerGeometry,
    MixedLayerTraffic,
    breakdown_total,
    canonical_kv_format,
    measure_layer_mix,
    measure_step,
    protocol_format_id,
)

__all__ = [
    "FORMATS",
    "LLAMA_31_8B",
    "QWEN25_05B",
    "DecodePoint",
    "HeadRow",
    "PositionRow",
    "TensorError",
    "decode_accumulation",
    "default_target_formats",
    "geometry_by_name",
    "head_sensitivity",
    "last_query_scores",
    "layer_mix_traffic",
    "make_qkv",
    "position_sensitivity",
    "reconstruct_kv",
    "reconstruct_kv_split",
    "tensor_error",
]

_TARGET_FORMATS: tuple[str, ...] = ("int8", "int4", "int4_bdr", "kivi2", "kivi4")


from .tensor_metrics import TensorError, tensor_error


@dataclass(frozen=True)
class HeadRow:
    """单个 KV head 的 last-query 误差。"""

    kv_format: str
    protocol_id: str
    head: int
    n: int
    attn: TensorError
    scores: TensorError
    kv: TensorError
    bytes_per_token: int

    def as_dict(self) -> dict[str, int | float | str]:
        """便于落 JSON。"""
        out: dict[str, int | float | str] = {
            "kv_format": self.kv_format,
            "protocol_id": self.protocol_id,
            "head": self.head,
            "n": self.n,
            "bytes_per_token": self.bytes_per_token,
        }
        for prefix, err in (("attn", self.attn), ("scores", self.scores), ("kv", self.kv)):
            for key, value in err.as_dict().items():
                out[f"{prefix}_{key}"] = value
        return out


@dataclass(frozen=True)
class PositionRow:
    """近窗 / 历史 / 全序列三种切分下的 last-query 误差。"""

    kv_format: str
    protocol_id: str
    region: str
    n: int
    recent_len: int
    attn: TensorError
    kv: TensorError
    bytes_per_token: int

    def as_dict(self) -> dict[str, int | float | str]:
        """便于落 JSON。"""
        out: dict[str, int | float | str] = {
            "kv_format": self.kv_format,
            "protocol_id": self.protocol_id,
            "region": self.region,
            "n": self.n,
            "recent_len": self.recent_len,
            "bytes_per_token": self.bytes_per_token,
        }
        for prefix, err in (("attn", self.attn), ("kv", self.kv)):
            for key, value in err.as_dict().items():
                out[f"{prefix}_{key}"] = value
        return out


@dataclass(frozen=True)
class DecodePoint:
    """decode 第 ``step`` 步（从 1 计）相对 float golden 的误差。"""

    kv_format: str
    protocol_id: str
    step: int
    cache_len: int
    attn: TensorError
    bytes_per_token: int

    def as_dict(self) -> dict[str, int | float | str]:
        """便于落 JSON。"""
        out: dict[str, int | float | str] = {
            "kv_format": self.kv_format,
            "protocol_id": self.protocol_id,
            "step": self.step,
            "cache_len": self.cache_len,
            "bytes_per_token": self.bytes_per_token,
        }
        out.update({f"attn_{k}": v for k, v in self.attn.as_dict().items()})
        return out


def make_qkv(
    num_heads: int,
    head_dim: int,
    seq: int,
    *,
    seed: int,
    device: torch.device | None = None,
    distribution: str = "gaussian",
    q_len: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """合成 Q/K/V；默认 last-query（``q_len=1``）。

    ``outlier``：K/V 末维前 2 通道 ×20（与 ``codec_compare`` 一致）。
    """
    if distribution not in {"gaussian", "outlier"}:
        raise ValueError(f"distribution 须为 gaussian / outlier，得到 {distribution!r}")
    device = device or torch.device("cpu")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    q = torch.randn(1, num_heads, q_len, head_dim, generator=g)
    k = torch.randn(1, num_heads, seq, head_dim, generator=g)
    v = torch.randn(1, num_heads, seq, head_dim, generator=g)
    if distribution == "outlier":
        k[..., : min(2, head_dim)] *= 20.0
        v[..., : min(2, head_dim)] *= 20.0
    return q.to(device), k.to(device), v.to(device)


def last_query_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Last-query scores，形状 ``(1, H, 1, S)``。"""
    scale = q.shape[-1] ** -0.5
    return torch.matmul(q, k.transpose(-2, -1)) * scale


def _to_cache_layout(x: torch.Tensor) -> torch.Tensor:
    """将单批次张量从 ``(1, H, S, D)`` 转为缓存布局 ``(S, H, D)``。"""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"期望 (1, H, S, D)，得到 {tuple(x.shape)}")
    return x.squeeze(0).transpose(0, 1).contiguous()


def _from_cache_layout(x: torch.Tensor) -> torch.Tensor:
    """将缓存布局 ``(S, H, D)`` 还原为单批次张量 ``(1, H, S, D)``。"""
    return x.transpose(0, 1).unsqueeze(0)


def reconstruct_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_format: str,
    *,
    layout: str = "contiguous",
    group_size: int = 32,
    residual_length: int = 128,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, BytesBreakdown]:
    """真实 cache-path round-trip；输入 ``(1, H, S, D)``。"""
    name = canonical_kv_format(kv_format)
    if k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1:
        raise ValueError(f"期望 k/v 为 (1, H, S, D)，得到 k={tuple(k.shape)} v={tuple(v.shape)}")
    _, heads, _seq, dim = k.shape
    attn = AttentionWithCache(
        name,
        num_heads=heads,
        head_dim=dim,
        layout=layout,
        device=device or k.device,
        group_size=group_size,
        residual_length=residual_length,
        seed=seed,
    )
    attn.cache.append(_to_cache_layout(k), _to_cache_layout(v))
    k2, v2 = attn.cache.load()
    return _from_cache_layout(k2), _from_cache_layout(v2), attn.bytes_breakdown()


def reconstruct_kv_split(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    recent_len: int,
    history_format: str,
    recent_format: str,
    layout: str = "contiguous",
    group_size: int = 32,
    residual_length: int = 128,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, BytesBreakdown]:
    """历史与近窗用不同格式 round-trip 后再拼接。"""
    if recent_len < 0:
        raise ValueError(f"recent_len 须为非负，得到 {recent_len}")
    seq = int(k.shape[2])
    cut = max(0, seq - recent_len)
    parts_k: list[torch.Tensor] = []
    parts_v: list[torch.Tensor] = []
    payload = scale = zp = page = 0

    def _add(chunk_k: torch.Tensor, chunk_v: torch.Tensor, fmt: str) -> None:
        nonlocal payload, scale, zp, page
        if chunk_k.shape[2] <= 0:
            return
        rk, rv, br = reconstruct_kv(
            chunk_k,
            chunk_v,
            fmt,
            layout=layout,
            group_size=group_size,
            residual_length=residual_length,
            seed=seed,
            device=device,
        )
        parts_k.append(rk)
        parts_v.append(rv)
        payload += br.payload
        scale += br.scale
        zp += br.zp
        page += br.page

    if cut > 0:
        _add(k[:, :, :cut, :], v[:, :, :cut, :], history_format)
    if cut < seq:
        _add(k[:, :, cut:, :], v[:, :, cut:, :], recent_format)
    if not parts_k:
        raise ValueError("空序列，无法 round-trip")
    return (
        torch.cat(parts_k, dim=2),
        torch.cat(parts_v, dim=2),
        BytesBreakdown(payload=payload, scale=scale, zp=zp, page=page),
    )


def layer_mix_traffic(
    geometry: LayerGeometry,
    layer_formats: list[str] | tuple[str, ...],
    *,
    n: int,
    layout: str = "contiguous",
) -> MixedLayerTraffic:
    """``measure_layer_mix`` 的薄封装。"""
    return measure_layer_mix(geometry, layer_formats, n=n, layout=layout)


def head_sensitivity(
    geometry: LayerGeometry,
    *,
    kv_format: str,
    n: int,
    seed: int = 0,
    layout: str = "contiguous",
    distribution: str = "gaussian",
    group_size: int = 32,
    residual_length: int = 128,
    device: torch.device | None = None,
) -> list[HeadRow]:
    """Last-query 下每个 KV head 的 attention / score / KV 重建误差。"""
    name = canonical_kv_format(kv_format)
    q, k, v = make_qkv(
        geometry.num_kv_heads,
        geometry.head_dim,
        n,
        seed=seed,
        device=device,
        distribution=distribution,
        q_len=1,
    )
    k2, v2, br = reconstruct_kv(
        k,
        v,
        name,
        layout=layout,
        group_size=group_size,
        residual_length=residual_length,
        seed=seed,
        device=device,
    )
    gold_out = scaled_dot_product_attention(q, k, v)
    pred_out = scaled_dot_product_attention(q, k2, v2)
    gold_sc = last_query_scores(q, k)
    pred_sc = last_query_scores(q, k2)
    traffic = measure_step(geometry, kv_format=name, n=n, layout=layout, device=device)
    rows: list[HeadRow] = []
    for h in range(geometry.num_kv_heads):
        rows.append(
            HeadRow(
                kv_format=name,
                protocol_id=protocol_format_id(name),
                head=h,
                n=n,
                attn=tensor_error(pred_out[:, h], gold_out[:, h]),
                scores=tensor_error(pred_sc[:, h], gold_sc[:, h]),
                kv=tensor_error(
                    torch.cat((k2[:, h], v2[:, h]), dim=-1),
                    torch.cat((k[:, h], v[:, h]), dim=-1),
                ),
                bytes_per_token=traffic.bytes_per_token,
            )
        )
    return rows


def position_sensitivity(
    geometry: LayerGeometry,
    *,
    kv_format: str,
    n: int,
    recent_len: int = 128,
    seed: int = 0,
    layout: str = "contiguous",
    distribution: str = "gaussian",
    group_size: int = 32,
    residual_length: int = 128,
    device: torch.device | None = None,
) -> list[PositionRow]:
    """三种切分：全部量化、只压历史、只压近窗。C0 近窗对齐 KIVI ``R=128``。"""
    name = canonical_kv_format(kv_format)
    q, k, v = make_qkv(
        geometry.num_kv_heads,
        geometry.head_dim,
        n,
        seed=seed,
        device=device,
        distribution=distribution,
        q_len=1,
    )
    gold_out = scaled_dot_product_attention(q, k, v)
    rows: list[PositionRow] = []

    def _row(region: str, k2: torch.Tensor, v2: torch.Tensor, br: BytesBreakdown) -> PositionRow:
        pred_out = scaled_dot_product_attention(q, k2, v2)
        return PositionRow(
            kv_format=name,
            protocol_id=protocol_format_id(name),
            region=region,
            n=n,
            recent_len=recent_len,
            attn=tensor_error(pred_out, gold_out),
            kv=tensor_error(torch.cat((k2, v2), dim=-1), torch.cat((k, v), dim=-1)),
            bytes_per_token=breakdown_total(br) * geometry.num_layers,
        )

    k_all, v_all, br_all = reconstruct_kv(
        k,
        v,
        name,
        layout=layout,
        group_size=group_size,
        residual_length=residual_length,
        seed=seed,
        device=device,
    )
    rows.append(_row("all", k_all, v_all, br_all))
    for region, hist_fmt, rec_fmt in (("history", name, "fp16"), ("recent", "fp16", name)):
        k2, v2, br = reconstruct_kv_split(
            k,
            v,
            recent_len=recent_len,
            history_format=hist_fmt,
            recent_format=rec_fmt,
            layout=layout,
            group_size=group_size,
            residual_length=residual_length,
            seed=seed,
            device=device,
        )
        rows.append(_row(region, k2, v2, br))
    return rows


def decode_accumulation(
    geometry: LayerGeometry,
    *,
    kv_format: str,
    prefill: int,
    n_decode: int,
    checkpoints: tuple[int, ...] = (1, 16, 64, 128),
    seed: int = 0,
    layout: str = "contiguous",
    distribution: str = "gaussian",
    group_size: int = 32,
    residual_length: int = 128,
    device: torch.device | None = None,
) -> list[DecodePoint]:
    """Prefill 后逐步 decode，在检查点记录相对 float golden 的 attention 误差。"""
    name = canonical_kv_format(kv_format)
    if prefill <= 0 or n_decode <= 0:
        raise ValueError(f"prefill / n_decode 须为正，得到 {prefill}, {n_decode}")
    device = device or torch.device("cpu")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    def _tok(seq: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(1, geometry.num_kv_heads, seq, geometry.head_dim, generator=g)
        k = torch.randn(1, geometry.num_kv_heads, seq, geometry.head_dim, generator=g)
        v = torch.randn(1, geometry.num_kv_heads, seq, geometry.head_dim, generator=g)
        if distribution == "outlier":
            k[..., : min(2, geometry.head_dim)] *= 20.0
            v[..., : min(2, geometry.head_dim)] *= 20.0
        return q.to(device), k.to(device), v.to(device)

    attn = AttentionWithCache(
        name,
        num_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        layout=layout,
        device=device,
        group_size=group_size,
        residual_length=residual_length,
        seed=seed,
    )
    q0, k0, v0 = _tok(prefill)
    attn.cache.append(_to_cache_layout(k0), _to_cache_layout(v0))
    gold_k = [_to_cache_layout(k0).float()]
    gold_v = [_to_cache_layout(v0).float()]

    want = {int(x) for x in checkpoints if 1 <= int(x) <= n_decode}
    rows: list[DecodePoint] = []
    for t in range(1, n_decode + 1):
        qt, kt, vt = _tok(1)
        pred = attn.decode_step(qt, kt, vt)
        gold_k.append(_to_cache_layout(kt).float())
        gold_v.append(_to_cache_layout(vt).float())
        gk = _from_cache_layout(torch.cat(gold_k, dim=0))
        gv = _from_cache_layout(torch.cat(gold_v, dim=0))
        gold = scaled_dot_product_attention(qt, gk, gv)
        if t in want:
            br = attn.bytes_breakdown()
            rows.append(
                DecodePoint(
                    kv_format=name,
                    protocol_id=protocol_format_id(name),
                    step=t,
                    cache_len=prefill + t,
                    attn=tensor_error(pred, gold),
                    bytes_per_token=breakdown_total(br) * geometry.num_layers,
                )
            )
    return rows


def default_target_formats() -> tuple[str, ...]:
    """C1–C5（不含 C0）。"""
    return _TARGET_FORMATS


def geometry_by_name(name: str) -> LayerGeometry:
    """``qwen05b`` / ``llama8b``，或 ``H,D,L``（如 ``2,64,24``）。"""
    key = name.strip().lower()
    if key in {"qwen05b", "qwen2.5-0.5b", "qwen25_05b", "dev"}:
        return QWEN25_05B
    if key in {"llama8b", "llama3.1-8b", "llama_31_8b", "pareto"}:
        return LLAMA_31_8B
    parts = [p.strip() for p in key.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 3:
        h, d, layers = (int(x) for x in parts)
        return LayerGeometry(num_kv_heads=h, head_dim=d, num_layers=layers)
    raise ValueError(f"未知 geometry={name!r}；用 qwen05b / llama8b / H,D,L")
