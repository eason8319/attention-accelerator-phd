"""INT、FP8、MXFP4 的假量化 / 反量化工具。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch


class QuantGranularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


@dataclass(frozen=True)
class QuantConfig:
    bits: int
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR
    symmetric: bool = True
    group_size: int = 32
    axis: int = -1


# ---------------------------------------------------------------------------
# INT 假量化
# ---------------------------------------------------------------------------


def _quantize_axis(x: torch.Tensor, axis: int) -> tuple[torch.Tensor, int]:
    if axis < 0:
        axis = x.ndim + axis
    return x, axis


def _reduce_for_granularity(
    x: torch.Tensor,
    cfg: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, axis = _quantize_axis(x, cfg.axis)
    if cfg.granularity == QuantGranularity.PER_TENSOR:
        reduce_dims = tuple(range(x.ndim))
    elif cfg.granularity == QuantGranularity.PER_CHANNEL:
        reduce_dims = tuple(d for d in range(x.ndim) if d != axis)
    elif cfg.granularity == QuantGranularity.PER_GROUP:
        if x.shape[axis] % cfg.group_size != 0:
            raise ValueError(f"轴长度 {x.shape[axis]} 不能被 group_size {cfg.group_size} 整除")
        # 将 axis 重塑为 (..., n_groups, group_size)
        n_groups = x.shape[axis] // cfg.group_size
        lead = x.shape[:axis]
        trail = x.shape[axis + 1 :]
        grouped = x.reshape(*lead, n_groups, cfg.group_size, *trail)
        reduce_dims = (axis + 1,)  # 在组内归约
        return grouped, reduce_dims
    else:
        raise ValueError(f"未知粒度 {cfg.granularity}")

    return x, reduce_dims


def _broadcast_params(
    x: torch.Tensor,
    params: torch.Tensor,
    cfg: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 scale/zero-point 张量广播到与 x 匹配。"""
    if cfg.granularity == QuantGranularity.PER_TENSOR:
        return params, params

    if cfg.granularity == QuantGranularity.PER_CHANNEL:
        shape = [1] * x.ndim
        axis = cfg.axis if cfg.axis >= 0 else x.ndim + cfg.axis
        shape[axis] = -1
        return params.view(*shape), params.view(*shape)

    # per_group：params 形状为 (..., n_groups, 1, ...)
    return params, params


def int_fake_quant(
    x: torch.Tensor,
    cfg: QuantConfig,
) -> torch.Tensor:
    """对称或非对称 INT 假量化-反量化。"""
    if cfg.bits not in (4, 8):
        raise ValueError("INT 假量化仅支持 4 或 8 bit")

    work, reduce_dims = _reduce_for_granularity(x, cfg)
    qmax = (1 << (cfg.bits - 1)) - 1 if cfg.symmetric else (1 << cfg.bits) - 1
    qmin = -qmax if cfg.symmetric else 0

    if cfg.symmetric:
        max_abs = work.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        scale = max_abs / qmax
        q = torch.round(work / scale).clamp(qmin, qmax)
        dequant = q * scale
    else:
        xmin = work.amin(dim=reduce_dims, keepdim=True)
        xmax = work.amax(dim=reduce_dims, keepdim=True)
        scale = ((xmax - xmin) / qmax).clamp(min=1e-8)
        zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        q = torch.round(work / scale + zero_point).clamp(qmin, qmax)
        dequant = (q - zero_point) * scale

    if cfg.granularity == QuantGranularity.PER_GROUP:
        dequant = dequant.reshape_as(x)
    return dequant.to(dtype=x.dtype)


def int4_fake_quant(
    x: torch.Tensor,
    *,
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR,
    symmetric: bool = True,
    group_size: int = 32,
    axis: int = -1,
) -> torch.Tensor:
    cfg = QuantConfig(4, granularity, symmetric, group_size, axis)
    return int_fake_quant(x, cfg)


def int8_fake_quant(
    x: torch.Tensor,
    *,
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR,
    symmetric: bool = True,
    group_size: int = 32,
    axis: int = -1,
) -> torch.Tensor:
    cfg = QuantConfig(8, granularity, symmetric, group_size, axis)
    return int_fake_quant(x, cfg)


# ---------------------------------------------------------------------------
# FP8 假量化（E4M3 / E5M2）
# ---------------------------------------------------------------------------

FP8Format = Literal["e4m3", "e5m2"]

_FP8_MAX = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}


def _torch_fp8_dtype(fmt: FP8Format) -> torch.dtype:
    if fmt == "e4m3":
        return torch.float8_e4m3fn
    return torch.float8_e5m2


def fp8_fake_quant(
    x: torch.Tensor,
    fmt: FP8Format = "e4m3",
    *,
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR,
    axis: int = -1,
) -> torch.Tensor:
    """使用 torch FP8 dtype 的 per-tensor 或 per-channel FP8 假量化。"""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch FP8 dtype 需要 PyTorch >= 2.1")

    fp8_dtype = _torch_fp8_dtype(fmt)
    work, reduce_dims = _reduce_for_granularity(
        x, cfg := QuantConfig(8, granularity, True, 32, axis)
    )
    if cfg.granularity == QuantGranularity.PER_GROUP:
        raise ValueError("FP8 假量化仅支持 per_tensor 或 per_channel")

    max_abs = work.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    scale = max_abs / _FP8_MAX[fmt]
    scaled = (work / scale).to(fp8_dtype).float() * scale

    if cfg.granularity == QuantGranularity.PER_GROUP:
        return scaled.reshape_as(x).to(dtype=x.dtype)
    return scaled.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# MXFP4（OCP microscaling，块大小 32，共享 E8M0 scale）
# ---------------------------------------------------------------------------

# E2M1 正值（符号单独处理）；max = 6.0
_MXFP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
_MXFP4_MAX = 6.0


def _shared_block_scale(block: torch.Tensor) -> torch.Tensor:
    """E8M0 风格共享指数：由块内最大绝对值得到的 2 的幂 scale。"""
    max_abs = block.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # floor log2，使缩放后数值 <= _MXFP4_MAX
    exp = torch.floor(torch.log2(max_abs / _MXFP4_MAX))
    return torch.pow(2.0, exp)


def _quantize_e2m1(values: torch.Tensor) -> torch.Tensor:
    """将 |x| 最近邻量化到 E2M1 档位，保留符号。"""
    levels = _MXFP4_E2M1_LEVELS.to(device=values.device, dtype=values.dtype)
    abs_v = values.abs().unsqueeze(-1)
    idx = (abs_v - levels).abs().argmin(dim=-1)
    q_abs = levels[idx]
    return torch.copysign(q_abs, values)


def mxfp4_fake_quant(x: torch.Tensor, block_size: int = 32, axis: int = -1) -> torch.Tensor:
    """MXFP4 假量化：块级共享 2 的幂 scale + E2M1 元素。"""
    if block_size != 32:
        raise ValueError("MXFP4 按 OCP 规范使用 block_size=32")

    axis = axis if axis >= 0 else x.ndim + axis
    if x.shape[axis] % block_size != 0:
        raise ValueError(f"轴长度 {x.shape[axis]} 须能被 {block_size} 整除")

    n_groups = x.shape[axis] // block_size
    lead = x.shape[:axis]
    trail = x.shape[axis + 1 :]
    grouped = x.reshape(*lead, n_groups, block_size, *trail).float()

    scale = _shared_block_scale(grouped)
    normalized = grouped / scale
    quantized = _quantize_e2m1(normalized)
    dequant = (quantized * scale).reshape_as(x)
    return dequant.to(dtype=x.dtype)


def relative_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """平均相对 L2 误差。"""
    ref_f = ref.float().reshape(-1)
    approx_f = approx.float().reshape(-1)
    denom = ref_f.norm().clamp(min=1e-8)
    return ((ref_f - approx_f).norm() / denom).item()


def mse(ref: torch.Tensor, approx: torch.Tensor) -> float:
    return ((ref.float() - approx.float()) ** 2).mean().item()
