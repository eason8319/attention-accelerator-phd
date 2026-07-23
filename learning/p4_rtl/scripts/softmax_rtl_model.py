"""与 ``softmax_unit/online_softmax.sv`` 逐位一致的 online softmax（单行）。

状态：运行 max ``m``（Q6.10）与运行 sum ``l``（UQ8.24）。
使用 ``exp_approx_rtl`` 计算 ``exp(·)``。不计算 ``O`` / PV（P4 范围）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exp_rtl_model import exp_approx_rtl  # noqa: E402
from fixedpoint import Q6_10, fixed_to_float, float_to_fixed  # noqa: E402

# l 将 UQ0.24 的 exp 输出累加到 UQ8.24（小数位相同，8 位整数）
L_FRAC = 24
L_MAX = (1 << 32) - 1  # 32 位无符号饱和
ALPHA_ONE = (1 << 24) - 1  # UQ0.24 编码的 ~1.0（与 exp(0) 饱和一致）


def _q610_sub(a: int, b: int) -> int:
    """饱和 Q6.10 减法。"""
    return int(np.clip(int(a) - int(b), Q6_10.min_int, Q6_10.max_int))


def _mul_alpha_l(alpha_uq024: int, l_uq824: int) -> int:
    """(alpha * l)，alpha 为 UQ0.24，l 为 UQ8.24 → UQ8.24。"""
    prod = (int(alpha_uq024) * int(l_uq824)) >> L_FRAC
    return int(min(prod, L_MAX))


def online_softmax_rtl(
    scores_q610: np.ndarray | list[int],
    block_size: int,
) -> tuple[int, int]:
    """处理一行 Q6.10 score；返回 ``(m_raw Q6.10, l_raw UQ8.24)``。"""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    scores = [int(v) for v in np.asarray(scores_q610, dtype=np.int64).reshape(-1)]
    if not scores:
        raise ValueError("empty score row")

    m_valid = False
    m = 0
    l = 0

    n = len(scores)
    for start in range(0, n, block_size):
        blk = scores[start : start + block_size]
        m_blk = max(blk)
        m_new = max(m, m_blk) if m_valid else m_blk

        if m_valid:
            alpha = int(exp_approx_rtl(_q610_sub(m, m_new)))
            l = _mul_alpha_l(alpha, l)
        # 否则：首块，l 保持 0，alpha 未使用

        sum_e = 0
        for s in blk:
            sum_e += int(exp_approx_rtl(_q610_sub(s, m_new)))
        l = int(min(l + sum_e, L_MAX))
        m = int(m_new)
        m_valid = True

    return m, l


def online_softmax_rtl_from_float(
    scores: np.ndarray,
    block_size: int,
) -> tuple[float, float]:
    raw = float_to_fixed(scores, Q6_10)
    m_raw, l_raw = online_softmax_rtl(raw, block_size)
    return float(fixed_to_float(m_raw, Q6_10)), float(l_raw) / float(1 << L_FRAC)


def online_softmax_float_ref(scores: np.ndarray) -> tuple[float, float]:
    """用 math.exp 的真实 online softmax 统计量（单块 = 整行）。"""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = float(np.max(s))
    l = float(np.sum(np.exp(s - m)))
    return m, l


def _self_test() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, 1.5, size=64).astype(np.float64)
    scores = np.clip(scores, Q6_10.min_val, Q6_10.max_val)

    m_ref, l_ref = online_softmax_float_ref(scores)
    raw = float_to_fixed(scores, Q6_10)

    m0, l0 = online_softmax_rtl(raw, 8)
    for bs in (1, 4, 16, 32):
        m1, l1 = online_softmax_rtl(raw, bs)
        assert m1 == m0, (bs, m1, m0)
        # 定点 + 近似 exp：l 仅近似与 block 无关
        rel = abs(l1 - l0) / max(l0, 1)
        assert rel < 1e-2, (bs, l1, l0, rel)

    m_f, l_f = online_softmax_rtl_from_float(scores, 8)
    assert abs(m_f - m_ref) < 0.01, (m_f, m_ref)
    rel_l = abs(l_f - l_ref) / max(l_ref, 1e-12)
    assert rel_l < 0.05, (l_f, l_ref, rel_l)

    print(f"softmax_rtl_model self-test PASSED (m_err={abs(m_f - m_ref):.3e}, l_rel={rel_l:.3e})")


if __name__ == "__main__":
    _self_test()
