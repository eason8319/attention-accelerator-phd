#!/usr/bin/env python3
"""比对 exp_approx DUT 输出与逐位 golden（可选 true-exp 预算）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from exp_approx import REF_FLOOR, REL_BUDGET, TARGET_X_MIN  # noqa: E402
from fixedpoint import Q6_10, UQ0_24, fixed_to_float, read_vec, rel_err  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vec", type=Path, required=True, help="向量目录")
    p.add_argument("--lsb-tol", type=int, default=0, help="允许的 |dut-expected| LSB 数")
    args = p.parse_args()

    x_raw = read_vec(args.vec / "x.txt")
    expected = read_vec(args.vec / "expected.txt")
    dut = read_vec(args.vec / "dut_out.txt")

    if len(dut) != len(expected):
        raise SystemExit(f"length mismatch: dut={len(dut)} expected={len(expected)} x={len(x_raw)}")
    if len(x_raw) != len(expected):
        raise SystemExit(f"x/expected length mismatch: {len(x_raw)} vs {len(expected)}")

    diff = np.abs(dut.astype(np.int64) - expected.astype(np.int64))
    n_mismatch = int(np.sum(diff > args.lsb_tol))
    max_lsb = int(diff.max()) if len(diff) else 0

    print(f"bit-exact check: mismatches={n_mismatch}/{len(dut)} max_lsb_err={max_lsb}")

    # 同时报告相对 true exp 的误差（验收预算）
    x = fixed_to_float(x_raw, Q6_10)
    y_dut = fixed_to_float(dut, UQ0_24)
    ref = np.exp(x)
    mask = (x >= TARGET_X_MIN) & (ref >= REF_FLOOR)
    if np.any(mask):
        re = rel_err(ref[mask], y_dut[mask])
        print(
            f"vs true exp (x>={TARGET_X_MIN:g}, ref>={REF_FLOOR:g}): "
            f"max_rel={float(re.max()):.3e} p99={float(np.percentile(re, 99)):.3e} "
            f"budget={REL_BUDGET:g} pass={float(re.max()) < REL_BUDGET}"
        )

    if n_mismatch:
        bad = np.where(diff > args.lsb_tol)[0][:8]
        for i in bad:
            print(
                f"  mismatch[{i}]: x_raw={int(x_raw[i])} "
                f"expected={int(expected[i])} dut={int(dut[i])}"
            )
        raise SystemExit("compare_exp FAILED: bit mismatch with RTL golden")

    print("compare_exp PASSED")


if __name__ == "__main__":
    main()
