"""KV Cache 编解码器（C0 FP16 / C1 INT8 / C2 INT4 / C3 INT4+BDR）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from rotation import BlockDiagonalRotation


@dataclass
class EncodedKV:
    """encode 产出：载荷与可选 scale / zero-point。"""

    payload: torch.Tensor
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None


class KVCodec(ABC):
    """写路径 encode、读路径 decode。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """编解码器短名，如 ``fp16`` / ``int8`` / ``int4``。"""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> EncodedKV:
        """float → 可存储形态。"""

    @abstractmethod
    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        """存储形态 → float。"""

    def bytes_payload(self, encoded: EncodedKV) -> int:
        """载荷字节数（按 ``payload`` 实际 dtype 存储计）。"""
        return int(encoded.payload.numel() * encoded.payload.element_size())

    def bytes_metadata(self, encoded: EncodedKV) -> int:
        """scale / zero-point 字节数；缺省为 0。"""
        total = 0
        if encoded.scale is not None:
            total += int(encoded.scale.numel() * encoded.scale.element_size())
        if encoded.zero_point is not None:
            total += int(encoded.zero_point.numel() * encoded.zero_point.element_size())
        return total


# ---------------------------------------------------------------------------
# INT 量化（encode / decode 共用）
# ---------------------------------------------------------------------------


def _encode_uniform_int(
    x: torch.Tensor,
    bits: int,
    *,
    symmetric: bool = True,
    group_size: int | None = None,
) -> EncodedKV:
    """沿最后一维均匀量化；``group_size`` 为 None 时整段共享一个 scale。"""
    if bits not in (4, 8):
        raise ValueError(f"仅支持 4/8 bit，得到 {bits}")

    x_f = x.float()
    axis = x_f.ndim - 1
    dim = x_f.shape[axis]

    if group_size is None:
        work = x_f
        reduce_dims = (axis,)
    else:
        if dim % group_size != 0:
            raise ValueError(f"末维长度 {dim} 不能被 group_size {group_size} 整除")
        n_groups = dim // group_size
        lead = x_f.shape[:axis]
        work = x_f.reshape(*lead, n_groups, group_size)
        reduce_dims = (axis + 1,)

    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    qmin = -qmax if symmetric else 0

    if symmetric:
        max_abs = work.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        scale = max_abs / qmax
        q = torch.round(work / scale).clamp(qmin, qmax)
        zero_point = None
    else:
        xmin = work.amin(dim=reduce_dims, keepdim=True)
        xmax = work.amax(dim=reduce_dims, keepdim=True)
        scale = ((xmax - xmin) / qmax).clamp(min=1e-8)
        zero_point = torch.round(-xmin / scale).clamp(qmin, qmax)
        q = torch.round(work / scale + zero_point).clamp(qmin, qmax)

    # INT4 暂用整型存网格值（未 nibble-pack）；非对称 INT8 用 uint8
    q = q.reshape_as(x_f)
    if bits == 8 and not symmetric:
        payload = q.to(torch.uint8)
        zp_dtype = torch.uint8
    else:
        payload = q.to(torch.int8)
        zp_dtype = torch.int8

    scale = scale.reshape(*scale.shape[:axis], -1).to(torch.float16)
    if zero_point is not None:
        zero_point = zero_point.reshape(*zero_point.shape[:axis], -1).to(zp_dtype)

    return EncodedKV(payload=payload, scale=scale, zero_point=zero_point)


def _decode_uniform_int(
    encoded: EncodedKV,
    *,
    group_size: int | None = None,
) -> torch.Tensor:
    """与 ``_encode_uniform_int`` 对称的反量化，返回 float32。"""
    if encoded.scale is None:
        raise ValueError("INT decode 需要 scale")

    q = encoded.payload.float()
    scale = encoded.scale.float()
    zp = None if encoded.zero_point is None else encoded.zero_point.float()

    if group_size is None:
        if zp is None:
            return q * scale
        return (q - zp) * scale

    axis = q.ndim - 1
    dim = q.shape[axis]
    if dim % group_size != 0:
        raise ValueError(f"末维长度 {dim} 不能被 group_size {group_size} 整除")
    n_groups = dim // group_size
    lead = q.shape[:axis]
    q_g = q.reshape(*lead, n_groups, group_size)
    scale_g = scale.reshape(*lead, n_groups, 1)
    if zp is None:
        out = q_g * scale_g
    else:
        out = (q_g - zp.reshape(*lead, n_groups, 1)) * scale_g
    return out.reshape_as(q)


# ---------------------------------------------------------------------------
# C0 / C1 / C2 / C3
# ---------------------------------------------------------------------------


class FP16Codec(KVCodec):
    """C0：FP16 存储，不量化。"""

    @property
    def name(self) -> str:
        return "fp16"

    def encode(self, x: torch.Tensor) -> EncodedKV:
        return EncodedKV(payload=x.to(torch.float16))

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        return encoded.payload.float()


class Int8Codec(KVCodec):
    """C1：均匀 token-wise INT8。"""

    def __init__(self, symmetric: bool = True) -> None:
        self.symmetric = symmetric

    @property
    def name(self) -> str:
        return "int8"

    def encode(self, x: torch.Tensor) -> EncodedKV:
        return _encode_uniform_int(x, bits=8, symmetric=self.symmetric)

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        return _decode_uniform_int(encoded)


class Int4Codec(KVCodec):
    """C2：均匀 token-wise INT4。"""

    def __init__(self, symmetric: bool = True, group_size: int = 32) -> None:
        self.symmetric = symmetric
        self.group_size = group_size

    @property
    def name(self) -> str:
        return "int4"

    def encode(self, x: torch.Tensor) -> EncodedKV:
        return _encode_uniform_int(
            x, bits=4, symmetric=self.symmetric, group_size=self.group_size
        )

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        return _decode_uniform_int(encoded, group_size=self.group_size)

    def bytes_payload(self, encoded: EncodedKV) -> int:
        """按协议 INT4 = 0.5 B/元素；当前以 int8 暂存网格，尚未 nibble-pack。"""
        return (encoded.payload.numel() + 1) // 2


class Int4BdrCodec(KVCodec):
    """C3：块对角旋转（BDR）+ 均匀 INT4（SAW-INT4 思想）。

    写路径：沿末维 ``rotate → INT4 encode``；读路径：``INT4 decode → inverse-rotate``。
    旋转矩阵视为片上常驻，不计入 ``bytes_metadata``（与协议一致）。
    ``dim``（head_dim）可在构造时给定，或在首次 ``encode``/``decode`` 时从张量末维推断。
    """

    def __init__(
        self,
        *,
        symmetric: bool = True,
        group_size: int = 32,
        block_size: int = 32,
        seed: int = 0,
        dim: int | None = None,
    ) -> None:
        self.symmetric = symmetric
        self.group_size = group_size
        self.block_size = block_size
        self.seed = seed
        self._rot: BlockDiagonalRotation | None = None
        if dim is not None:
            self._ensure_rotation(dim)

    @property
    def name(self) -> str:
        return "int4_bdr"

    def _ensure_rotation(self, dim: int) -> BlockDiagonalRotation:
        if self._rot is None:
            self._rot = BlockDiagonalRotation(
                dim, block_size=self.block_size, seed=self.seed
            )
        elif self._rot.dim != dim:
            raise ValueError(
                f"INT4+BDR 末维须与构造时一致：期望 {self._rot.dim}，得到 {dim}"
            )
        return self._rot

    def encode(self, x: torch.Tensor) -> EncodedKV:
        if x.shape[-1] % self.block_size != 0:
            raise ValueError(
                f"INT4+BDR 要求末维能被 block_size={self.block_size} 整除，得到 {x.shape[-1]}"
            )
        if x.shape[-1] % self.group_size != 0:
            raise ValueError(
                f"INT4+BDR 要求末维能被 group_size={self.group_size} 整除，得到 {x.shape[-1]}"
            )
        rot = self._ensure_rotation(x.shape[-1])
        x_rot = rot.rotate(x.float(), axis=-1)
        return _encode_uniform_int(
            x_rot, bits=4, symmetric=self.symmetric, group_size=self.group_size
        )

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        y = _decode_uniform_int(encoded, group_size=self.group_size)
        rot = self._ensure_rotation(y.shape[-1])
        return rot.inverse(y, axis=-1)

    def bytes_payload(self, encoded: EncodedKV) -> int:
        """与均匀 INT4 相同：按 0.5 B/元素记账。"""
        return (encoded.payload.numel() + 1) // 2


def get_codec(format_id: str, **kwargs) -> KVCodec:
    """按格式别名返回编解码器。

    支持：``fp16``/``C0``，``int8``/``C1``，``int4``/``C2``，
    ``int4_bdr``/``int4+bdr``/``C3``。
    """
    key = format_id.strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
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
    }
    if key not in aliases:
        raise ValueError(
            f"未知 format_id={format_id!r}；支持 fp16/int8/int4/int4_bdr 或 C0–C3"
        )
    name = aliases[key]
    if name == "fp16":
        return FP16Codec()
    if name == "int8":
        return Int8Codec(**{k: v for k, v in kwargs.items() if k in {"symmetric"}})
    if name == "int4":
        return Int4Codec(
            **{k: v for k, v in kwargs.items() if k in {"symmetric", "group_size"}}
        )
    bdr_keys = {"symmetric", "group_size", "block_size", "seed", "dim"}
    return Int4BdrCodec(**{k: v for k, v in kwargs.items() if k in bdr_keys})
