"""只在观测边界使用 float64，不改变模型及量化张量。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

METRIC_VERSION = "tensor-error-f64-v1"
COSINE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TensorError:
    """零参考范数的相对 L2、任一零范数的余弦以 None 表示未定义。"""

    rel_l2: float | None
    cosine: float | None
    max_abs: float

    def as_dict(self) -> dict[str, float | None]:
        """导出标准 JSON 兼容字段，不用 NaN 表示未定义。"""
        return asdict(self)


def tensor_error(pred: torch.Tensor, gold: torch.Tensor) -> TensorError:
    """以同一个 float64 参考范数计算相对 L2 与余弦；异常值报错。"""
    if pred.shape != gold.shape or pred.device != gold.device:
        raise ValueError("预测与参考的形状和设备必须一致")
    if pred.is_complex() or gold.is_complex():
        raise ValueError("误差指标只接受实数张量")
    p = pred.detach().to(dtype=torch.float64).reshape(-1)
    g = gold.detach().to(dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(p).all() and torch.isfinite(g).all()):
        raise ValueError("误差指标输入包含非有限值")
    p_norm = torch.linalg.vector_norm(p)
    g_norm = torch.linalg.vector_norm(g)
    diff = p - g
    diff_norm = torch.linalg.vector_norm(diff)
    if not all(math.isfinite(float(v)) for v in (p_norm, g_norm, diff_norm)):
        raise ArithmeticError("float64 范数归约溢出")
    rel_l2 = float(diff_norm / g_norm) if float(g_norm) != 0.0 else None
    cosine = None
    if float(p_norm) != 0.0 and float(g_norm) != 0.0:
        cosine = float(torch.dot(p, g) / (p_norm * g_norm))
        if not math.isfinite(cosine) or abs(cosine) > 1.0 + COSINE_TOLERANCE:
            raise ArithmeticError(f"余弦越界或非有限: {cosine!r}")
    if rel_l2 is not None and not math.isfinite(rel_l2):
        raise ArithmeticError("相对 L2 非有限")
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    return TensorError(rel_l2=rel_l2, cosine=cosine, max_abs=max_abs)
