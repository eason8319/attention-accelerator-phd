"""物理低比特打包：nibble/bit 装填、DMA 整段对齐，不重算量化统计量。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACK_LAYOUT_PATH = (
    ROOT / "research" / "r2_streaming_attention" / "experiments" / "configs" / "pack_layout.json"
)


@dataclass(frozen=True)
class PackLayout:
    """步骤 3 冻结的打包布局；字段与 ``pack_layout.json`` 对应。"""

    layout_id: str
    dma_align_bytes: int
    int4_qmin: int
    int4_qmax: int
    nibble_order: str
    signed_encoding: str
    bit_order: str
    group_size: int
    page_tokens: int
    c5_key_open_group_max_tokens: int


def load_pack_layout(path: Path | str | None = None) -> PackLayout:
    """读取共享打包布局；拒绝未约定的 nibble/符号编码。"""
    import json

    raw = json.loads((path or PACK_LAYOUT_PATH).read_text(encoding="utf-8"))
    signed = raw["int4_symmetric"]
    unsigned = raw["kivi_unsigned"]
    if signed["nibble_order"] != "low_even":
        raise ValueError(f"不支持的 nibble_order={signed['nibble_order']!r}")
    if signed["signed_encoding"] != "twos_complement":
        raise ValueError(f"不支持的 signed_encoding={signed['signed_encoding']!r}")
    if unsigned["bit_order"] != "low_first":
        raise ValueError(f"不支持的 bit_order={unsigned['bit_order']!r}")
    return PackLayout(
        layout_id=raw["layout_id"],
        dma_align_bytes=int(raw["dma"]["align_bytes"]),
        int4_qmin=int(signed["qmin"]),
        int4_qmax=int(signed["qmax"]),
        nibble_order=signed["nibble_order"],
        signed_encoding=signed["signed_encoding"],
        bit_order=unsigned["bit_order"],
        group_size=int(raw["group"]["size"]),
        page_tokens=int(raw["page_tokens"]),
        c5_key_open_group_max_tokens=int(raw["append"]["c5_key_open_group_max_tokens"]),
    )


def align_up(n_bytes: int, align: int) -> int:
    """空缓冲保持 0；非空按 ``align`` 向上取整。"""
    if n_bytes < 0:
        raise ValueError(f"字节数不能为负，得到 {n_bytes}")
    if align <= 0:
        raise ValueError(f"对齐量子须为正，得到 {align}")
    if n_bytes == 0:
        return 0
    return ((n_bytes + align - 1) // align) * align


def tensor_nbytes(x: torch.Tensor) -> int:
    """返回张量逻辑元素字节；视图持有的底层存储须另行核验。"""
    return int(x.numel() * x.element_size())


def pack_unsigned_bits(q: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    """把非负整数网格按 ``bits`` 位、低位先填打包成 ``uint8``。

    参数
        q: 任意形状的整数张量，值域 ``[0, 2**bits-1]``
        bits: 2 或 4
    返回
        ``(packed, pad_values)``；``packed`` 为一维 ``uint8``
    """
    if bits not in (2, 4):
        raise ValueError(f"仅支持 2/4-bit 打包，得到 {bits}")
    per_byte = 8 // bits
    max_q = (1 << bits) - 1
    flat = q.reshape(-1).to(torch.int64)
    if torch.any(flat < 0) or torch.any(flat > max_q):
        raise ValueError(f"无符号 {bits}-bit 网格超出 [0,{max_q}]")
    n = int(flat.numel())
    pad = (per_byte - (n % per_byte)) % per_byte
    if pad:
        flat = torch.cat(
            [flat, torch.zeros(pad, dtype=torch.int64, device=flat.device)],
            dim=0,
        )
    grouped = flat.view(-1, per_byte)
    acc = torch.zeros(grouped.shape[0], dtype=torch.int64, device=grouped.device)
    for lane in range(per_byte):
        acc = acc | (grouped[:, lane] << (lane * bits))
    return acc.to(torch.uint8), pad


def unpack_unsigned_bits(
    packed: torch.Tensor,
    bits: int,
    numel: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """与 ``pack_unsigned_bits`` 对称，返回长度为 ``numel`` 的 int64 网格。"""
    if bits not in (2, 4):
        raise ValueError(f"仅支持 2/4-bit 解包，得到 {bits}")
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed 须为 uint8，得到 {packed.dtype}")
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    acc = packed.reshape(-1).to(torch.int64)
    expected = (numel + per_byte - 1) // per_byte
    if int(acc.numel()) != expected:
        raise ValueError(f"packed 长度 {acc.numel()} 与 numel={numel} 不匹配（期望 {expected}）")
    parts = [(acc >> (lane * bits)) & mask for lane in range(per_byte)]
    flat = torch.stack(parts, dim=1).reshape(-1)[:numel]
    if device is not None:
        flat = flat.to(device)
    return flat


def to_signed_int4_nibbles(q: torch.Tensor, *, qmin: int, qmax: int) -> torch.Tensor:
    """把有符号 INT4 网格变成 4-bit 补码 nibbles（0..15）。"""
    flat = q.reshape(-1).to(torch.int32)
    if torch.any(flat < qmin) or torch.any(flat > qmax):
        raise ValueError(f"有符号 INT4 网格超出 [{qmin},{qmax}]")
    return (flat & 0xF).to(torch.int64)


def from_signed_int4_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """4-bit 补码 sign-extend 为 int8；拒绝 C2/C3 不使用的 -8 码点。"""
    x = nibbles.to(torch.int16)
    if torch.any(x == 8):
        raise ValueError("解包得到 nibble 8（即 -8），C2/C3 对称网格不允许该码点")
    signed = torch.where(x >= 8, x - 16, x)
    return signed.to(torch.int8)


def pack_signed_int4(q: torch.Tensor, *, qmin: int = -7, qmax: int = 7) -> tuple[torch.Tensor, int]:
    """C2/C3：有符号 INT4 按低半字节=偶数下标打包。"""
    nibbles = to_signed_int4_nibbles(q, qmin=qmin, qmax=qmax)
    return pack_unsigned_bits(nibbles, bits=4)


def unpack_signed_int4(
    packed: torch.Tensor,
    numel: int,
    *,
    qmin: int = -7,
    qmax: int = 7,
    device: torch.device | None = None,
) -> torch.Tensor:
    """与 ``pack_signed_int4`` 对称，返回 int8 网格。"""
    nibbles = unpack_unsigned_bits(packed, 4, numel, device=device)
    q = from_signed_int4_nibbles(nibbles)
    if torch.any(q.to(torch.int32) < qmin) or torch.any(q.to(torch.int32) > qmax):
        raise ValueError(f"解包后 INT4 超出 [{qmin},{qmax}]")
    return q


def pack_fp16_bytes(x: torch.Tensor) -> torch.Tensor:
    """把 float 张量按 float16 位型展成 ``uint8``。"""
    raw = x.to(torch.float16).contiguous()
    return raw.view(torch.uint8).reshape(-1).clone()


def unpack_fp16_bytes(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """从 float16 位型恢复 float32，长度为 ``numel``。"""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed 须为 uint8，得到 {packed.dtype}")
    if int(packed.numel()) != numel * 2:
        raise ValueError(f"FP16 packed 长度 {packed.numel()} 不是 numel*2={numel * 2}")
    return packed.contiguous().view(torch.float16).reshape(numel).float()


def pack_int8_bytes(q: torch.Tensor) -> torch.Tensor:
    """INT8 位型恒等打包。"""
    raw = q.to(torch.int8).contiguous()
    return raw.view(torch.uint8).reshape(-1).clone()


def unpack_int8_bytes(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """与 ``pack_int8_bytes`` 对称。"""
    if int(packed.numel()) != numel:
        raise ValueError(f"INT8 packed 长度 {packed.numel()} 不是 numel={numel}")
    return packed.contiguous().view(torch.int8).reshape(numel)
