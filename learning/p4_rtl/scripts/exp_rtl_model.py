"""与 ``exp_unit/exp_approx.sv`` 逐位一致的定点 exp 模型。

流水线运算（纯整数）::

1. ``y = x * log2(e)``，``x`` 为 Q6.10，``log2(e)`` 为 Q2.14 → 乘积 Q8.24
2. ``yi = floor(y)``，``yf ∈ [0,1)`` 表示为 UQ0.24
3. 对 ``2^{yf}`` 做 16 段 PWL（截距 UQ2.22，斜率 UQ2.14）
4. ``exp ≈ 2^{yi} * pwl`` → UQ0.24，四舍五入并饱和

gen/compare 使用的公开 API::

    exp_approx_rtl(x_raw: int | ndarray) -> int | ndarray   # Q6.10 raw -> UQ0.24 raw
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fixedpoint import Q6_10, UQ0_24, fixed_to_float, float_to_fixed  # noqa: E402

# 与 exp_approx.sv 中的 SV 参数一致
N_SEG = 16
LOG2E_Q214 = 23637  # round(log2(e) * 2^14)

# PWL：截距 UQ2.22，斜率 UQ2.14 — 在 2^f 端点处精确
_INTERCEPT_Q222 = np.array(
    [
        4194304,
        4380002,
        4573921,
        4776426,
        4987896,
        5208729,
        5439339,
        5680159,
        5931642,
        6194258,
        6468501,
        6754886,
        7053950,
        7366255,
        7692387,
        8032959,
    ],
    dtype=np.int64,
)
_SLOPE_Q214 = np.array(
    [
        11606,
        12120,
        12657,
        13217,
        13802,
        14413,
        15051,
        15718,
        16414,
        17140,
        17899,
        18692,
        19519,
        20383,
        21286,
        22228,
    ],
    dtype=np.int64,
)

OUT_MAX = (1 << 24) - 1


def _exp_one(x_raw: int) -> int:
    """单样本：Q6.10 raw（int16 范围）-> UQ0.24 raw。"""
    # 输入饱和到 int16 / Q6.10 范围
    x_raw = int(np.clip(x_raw, Q6_10.min_int, Q6_10.max_int))

    # Q6.10 * Q2.14 -> Q8.24（有符号 32 位乘积）
    y = np.int64(x_raw) * np.int64(LOG2E_Q214)

    # 算术右移取 floor（补码）
    yi = int(y >> 24)
    yf = int(y - (np.int64(yi) << 24))  # UQ0.24，范围 [0, 2^24)

    seg = (yf >> 20) & 0xF
    local = yf & ((1 << 20) - 1)  # 段内偏移

    intercept = int(_INTERCEPT_Q222[seg])
    slope = int(_SLOPE_Q214[seg])
    # slope*local / 2^16 将 UQ2.14 * UQ0.20 对齐为 UQ2.22 加项
    pwl = intercept + ((slope * local) >> 16)  # UQ2.22

    # 2^yi * pwl -> UQ0.24：wide = pwl << 2（UQ2.24），再 >> (-yi)
    wide = np.int64(pwl) << 2
    if yi >= 0:
        out = wide << yi
    else:
        sh = -yi
        if sh >= 62:
            out = np.int64(0)
        else:
            # 近似 round-to-nearest-even：移位前加半 LSB
            out = (wide + (np.int64(1) << (sh - 1))) >> sh

    if out < 0:
        out = np.int64(0)
    if out > OUT_MAX:
        out = np.int64(OUT_MAX)
    return int(out)


def exp_approx_rtl(x_raw: np.ndarray | int) -> np.ndarray | int:
    """逐位 RTL 模型。接受 Q6.10 raw int，返回 UQ0.24 raw。"""
    if np.isscalar(x_raw) or (isinstance(x_raw, np.ndarray) and x_raw.ndim == 0):
        return _exp_one(int(x_raw))
    arr = np.asarray(x_raw, dtype=np.int64).reshape(-1)
    return np.array([_exp_one(int(v)) for v in arr], dtype=np.int64)


def exp_approx_rtl_from_float(x: np.ndarray | float) -> np.ndarray:
    """浮点 -> Q6.10 量化 -> RTL 模型 -> UQ0.24 反量化浮点。"""
    raw_in = float_to_fixed(x, Q6_10)
    raw_out = exp_approx_rtl(raw_in)
    return fixed_to_float(raw_out, UQ0_24)


def dump_sv_params() -> str:
    """输出 ROM 表的 SystemVerilog 参数片段。"""
    lines = [
        f"  localparam int LOG2E_Q214 = {LOG2E_Q214};",
        "  // intercept UQ2.22",
        "  localparam logic [23:0] INTERCEPT [0:15] = '{",
    ]
    lines.append("    " + ", ".join(f"24'd{v}" for v in _INTERCEPT_Q222) + "")
    lines.append("  };")
    lines.append("  // slope UQ2.14")
    lines.append("  localparam logic [15:0] SLOPE [0:15] = '{")
    lines.append("    " + ", ".join(f"16'd{v}" for v in _SLOPE_Q214))
    lines.append("  };")
    return "\n".join(lines)


def _self_test() -> None:
    from exp_approx import REF_FLOOR, REL_BUDGET, TARGET_X_MIN, sample_attention_exp_inputs
    from fixedpoint import rel_err

    # x=0 -> 接近 1.0（UQ0.24 饱和最大值）
    assert exp_approx_rtl(0) == OUT_MAX

    # 粗网格上近似单调
    xs = np.linspace(-8.0, 0.0, 801)
    raw = float_to_fixed(xs, Q6_10)
    out = exp_approx_rtl(raw)
    assert np.all(np.diff(out.astype(np.int64)) >= -1)  # 舍入可能导致微小非单调

    # 目标域上相对 true exp 的误差
    x = sample_attention_exp_inputs(seq=256, heads=2, block_size=64)
    x = x[x >= TARGET_X_MIN]
    ref = np.exp(x)
    dut = exp_approx_rtl_from_float(x)
    mask = ref >= REF_FLOOR
    re = rel_err(ref[mask], dut[mask])
    assert float(re.max()) < REL_BUDGET, float(re.max())
    print(f"exp_rtl_model self-test PASSED (max_rel={float(re.max()):.3e})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dump-sv", action="store_true")
    args = p.parse_args()
    if args.dump_sv:
        print(dump_sv_params())
    if args.self_test or not args.dump_sv:
        _self_test()
