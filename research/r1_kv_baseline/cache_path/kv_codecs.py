"""KV Cache 编解码器（C0–C3 + KIVI 风格 K/V 量化核）。"""

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
# KIVI 风格非对称量化核（对齐 jy-yuan/KIVI quant/new_pack.py）
# ---------------------------------------------------------------------------
# 约定输入形状 ``(T, H, D)``（与 ContiguousKVCache 一致）。
# Key：沿 token 维按 group_size 分组，对每个 channel 做 min-max（per-channel）。
# Value：沿 channel 维按 group_size 分组，对每个 token 做 min-max（per-token）。
# 反量化：``x̂ = q * scale + mn``；``EncodedKV.zero_point`` 存 float16 的 ``mn``。
# 载荷暂用 uint8 存网格值（未 bit-pack）；bytes_payload 按名义比特记账。


def _kivi_qmax(bits: int) -> int:
    if bits not in (2, 4):
        raise ValueError(f"KIVI 核仅支持 2/4 bit，得到 {bits}")
    return (1 << bits) - 1


def _bytes_lowbit_payload(numel: int, bits: int) -> int:
    """名义低比特载荷字节数（未 pack 时的记账口径）。"""
    return (numel * bits + 7) // 8


def encode_kivi_key(
    x: torch.Tensor,
    *,
    bits: int = 2,
    group_size: int = 32,
) -> EncodedKV:
    """KIVI Key：per-channel 分组量化。

    将 token 维 ``T`` 划为 ``T // group_size`` 组；组内沿 token 维求
    min/max，每个 ``(group, head, channel)`` 共享一套 ``scale`` / ``mn``。

    参数
        x: ``(T, H, D)``，要求 ``T % group_size == 0``。
        bits: 2 或 4。
        group_size: 每组 token 数（协议默认 32）。
    """
    if x.ndim != 3:
        raise ValueError(f"KIVI Key 期望 (T, H, D)，得到 {tuple(x.shape)}")
    t, h, d = x.shape
    if t <= 0 or t % group_size != 0:
        raise ValueError(
            f"KIVI Key 要求 T>0 且能被 group_size={group_size} 整除，得到 T={t}"
        )
    qmax = _kivi_qmax(bits)
    n_groups = t // group_size
    work = x.float().reshape(n_groups, group_size, h, d)
    mn = work.amin(dim=1, keepdim=True)
    mx = work.amax(dim=1, keepdim=True)
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    q = ((work - mn) / scale).clamp(0, qmax).round()
    payload = q.reshape(t, h, d).to(torch.uint8)
    # scale/mn: (n_groups, 1, H, D) → (n_groups, H, D)
    scale_out = scale.squeeze(1).to(torch.float16)
    mn_out = mn.squeeze(1).to(torch.float16)
    return EncodedKV(payload=payload, scale=scale_out, zero_point=mn_out)


def decode_kivi_key(
    encoded: EncodedKV,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """与 ``encode_kivi_key`` 对称的反量化，返回 float32 ``(T, H, D)``。"""
    if encoded.scale is None or encoded.zero_point is None:
        raise ValueError("KIVI Key decode 需要 scale 与 mn（zero_point）")
    q = encoded.payload.float()
    t, h, d = q.shape
    if t % group_size != 0:
        raise ValueError(f"payload T={t} 不能被 group_size={group_size} 整除")
    n_groups = t // group_size
    scale = encoded.scale.float()
    mn = encoded.zero_point.float()
    if scale.shape != (n_groups, h, d) or mn.shape != (n_groups, h, d):
        raise ValueError(
            f"scale/mn 形状须为 {(n_groups, h, d)}，得到 scale={tuple(scale.shape)}, "
            f"mn={tuple(mn.shape)}"
        )
    q_g = q.reshape(n_groups, group_size, h, d)
    out = q_g * scale.unsqueeze(1) + mn.unsqueeze(1)
    return out.reshape(t, h, d)


def encode_kivi_value(
    x: torch.Tensor,
    *,
    bits: int = 2,
    group_size: int = 32,
) -> EncodedKV:
    """KIVI Value：per-token 分组量化。

    将 channel 维 ``D`` 划为 ``D // group_size`` 组；组内沿 channel 维求
    min/max，每个 ``(token, head, group)`` 共享一套 ``scale`` / ``mn``。

    参数
        x: ``(T, H, D)``，要求 ``D % group_size == 0``。
        bits: 2 或 4。
        group_size: 每组 channel 数（协议默认 32）。
    """
    if x.ndim != 3:
        raise ValueError(f"KIVI Value 期望 (T, H, D)，得到 {tuple(x.shape)}")
    t, h, d = x.shape
    if d <= 0 or d % group_size != 0:
        raise ValueError(
            f"KIVI Value 要求 D>0 且能被 group_size={group_size} 整除，得到 D={d}"
        )
    qmax = _kivi_qmax(bits)
    n_groups = d // group_size
    work = x.float().reshape(t, h, n_groups, group_size)
    mn = work.amin(dim=-1, keepdim=True)
    mx = work.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    q = ((work - mn) / scale).clamp(0, qmax).round()
    payload = q.reshape(t, h, d).to(torch.uint8)
    # scale/mn: (T, H, n_groups, 1) → (T, H, n_groups)
    scale_out = scale.squeeze(-1).to(torch.float16)
    mn_out = mn.squeeze(-1).to(torch.float16)
    return EncodedKV(payload=payload, scale=scale_out, zero_point=mn_out)


def decode_kivi_value(
    encoded: EncodedKV,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """与 ``encode_kivi_value`` 对称的反量化，返回 float32 ``(T, H, D)``。"""
    if encoded.scale is None or encoded.zero_point is None:
        raise ValueError("KIVI Value decode 需要 scale 与 mn（zero_point）")
    q = encoded.payload.float()
    t, h, d = q.shape
    if d % group_size != 0:
        raise ValueError(f"payload D={d} 不能被 group_size={group_size} 整除")
    n_groups = d // group_size
    scale = encoded.scale.float()
    mn = encoded.zero_point.float()
    if scale.shape != (t, h, n_groups) or mn.shape != (t, h, n_groups):
        raise ValueError(
            f"scale/mn 形状须为 {(t, h, n_groups)}，得到 scale={tuple(scale.shape)}, "
            f"mn={tuple(mn.shape)}"
        )
    q_g = q.reshape(t, h, n_groups, group_size)
    out = q_g * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return out.reshape(t, h, d)


class KiviKeyCodec(KVCodec):
    """KIVI Key 编解码器（per-channel，非对称 2/4-bit）。

    仅量化完整 group；残差窗由 ``KiviKVCache`` 维护。
    """

    def __init__(self, bits: int = 2, group_size: int = 32) -> None:
        _kivi_qmax(bits)
        self.bits = bits
        self.group_size = group_size

    @property
    def name(self) -> str:
        return f"kivi_key_{self.bits}bit"

    def encode(self, x: torch.Tensor) -> EncodedKV:
        return encode_kivi_key(x, bits=self.bits, group_size=self.group_size)

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        return decode_kivi_key(encoded, group_size=self.group_size)

    def bytes_payload(self, encoded: EncodedKV) -> int:
        return _bytes_lowbit_payload(encoded.payload.numel(), self.bits)


class KiviValueCodec(KVCodec):
    """KIVI Value 编解码器（per-token，非对称 2/4-bit）。

    仅量化历史段；残差窗由 ``KiviKVCache`` 维护。
    """

    def __init__(self, bits: int = 2, group_size: int = 32) -> None:
        _kivi_qmax(bits)
        self.bits = bits
        self.group_size = group_size

    @property
    def name(self) -> str:
        return f"kivi_value_{self.bits}bit"

    def encode(self, x: torch.Tensor) -> EncodedKV:
        return encode_kivi_value(x, bits=self.bits, group_size=self.group_size)

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        return decode_kivi_value(encoded, group_size=self.group_size)

    def bytes_payload(self, encoded: EncodedKV) -> int:
        return _bytes_lowbit_payload(encoded.payload.numel(), self.bits)


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


@dataclass(frozen=True)
class KiviFormat:
    """C4/C5 格式句柄：KIVI 非对称量化 + 残差窗参数。

    不能当作单张量 ``KVCodec`` 使用（K/V 粒度不同且含残差窗）。
    通过 ``make_cache`` 构造 ``KiviKVCache``。
    """

    bits: int
    group_size: int = 32
    residual_length: int = 128
    k_bits: int | None = None
    v_bits: int | None = None

    def __post_init__(self) -> None:
        _kivi_qmax(self.bits)
        if self.k_bits is not None:
            _kivi_qmax(self.k_bits)
        if self.v_bits is not None:
            _kivi_qmax(self.v_bits)
        if self.residual_length <= 0:
            raise ValueError(f"residual_length 须为正，得到 {self.residual_length}")
        if self.residual_length % self.group_size != 0:
            raise ValueError(
                f"residual_length={self.residual_length} 须能被 "
                f"group_size={self.group_size} 整除"
            )

    @property
    def name(self) -> str:
        """短名：``kivi2`` / ``kivi4``（与 ``get_codec`` 别名一致）。"""
        return f"kivi{self.bits}"

    @property
    def format_id(self) -> str:
        """协议对照谱 ID：2-bit→C4，4-bit→C5。"""
        if self.bits == 2:
            return "C4"
        if self.bits == 4:
            return "C5"
        return f"kivi{self.bits}"

    def make_cache(
        self,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None = None,
    ):
        """构造带残差窗的 ``KiviKVCache``（延迟导入以避免循环依赖）。"""
        from kv_cache import KiviKVCache

        return KiviKVCache(
            num_heads=num_heads,
            head_dim=head_dim,
            bits=self.bits,
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            group_size=self.group_size,
            residual_length=self.residual_length,
            device=device,
        )


def get_codec(format_id: str, **kwargs) -> KVCodec | KiviFormat:
    """按格式别名返回编解码器或 KIVI 格式句柄。

    支持：
    - ``fp16``/``C0``，``int8``/``C1``，``int4``/``C2``，``int4_bdr``/``C3`` → ``KVCodec``
    - ``kivi2``/``C4``，``kivi4``/``C5`` → ``KiviFormat``（再 ``.make_cache(...)``）
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
        "kivi2": "kivi2",
        "kivi_2": "kivi2",
        "kivi_2bit": "kivi2",
        "c4": "kivi2",
        "kivi4": "kivi4",
        "kivi_4": "kivi4",
        "kivi_4bit": "kivi4",
        "c5": "kivi4",
    }
    if key not in aliases:
        raise ValueError(
            f"未知 format_id={format_id!r}；支持 fp16/int8/int4/int4_bdr/kivi2/kivi4 "
            f"或 C0–C5"
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
    if name in {"kivi2", "kivi4"}:
        bits = 2 if name == "kivi2" else 4
        kivi_keys = {"group_size", "residual_length", "k_bits", "v_bits", "bits"}
        params = {k: v for k, v in kwargs.items() if k in kivi_keys}
        # 别名已锁定默认 bits；仅当显式传入且冲突时拒绝
        if "bits" in params and params["bits"] != bits:
            raise ValueError(
                f"{format_id!r} 对应 bits={bits}，与传入 bits={params['bits']} 冲突"
            )
        params["bits"] = bits
        return KiviFormat(**params)
    bdr_keys = {"symmetric", "group_size", "block_size", "seed", "dim"}
    return Int4BdrCodec(**{k: v for k, v in kwargs.items() if k in bdr_keys})
