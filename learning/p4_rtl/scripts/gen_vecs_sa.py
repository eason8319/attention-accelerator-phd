#!/usr/bin/env python3
"""生成 systolic array 测试向量（INT8 A、W → INT32 C）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from fixedpoint import write_vec  # noqa: E402
from systolic_rtl_model import LAT, N, gemm_int8, systolic_ws_sim  # noqa: E402


def _write_matrix(path: Path, mat: np.ndarray) -> None:
    """行主序展平整数文本。"""
    write_vec(path, mat.astype(np.int64).reshape(-1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--m", type=int, default=8, help="A/C 行数")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    a = rng.integers(-128, 128, size=(args.m, N), dtype=np.int8)
    w = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
    # 包含一行角点
    a[0] = np.array([-128, -1, 0, 127], dtype=np.int8)

    c_ref = gemm_int8(a, w)
    c_sim = systolic_ws_sim(a, w)
    assert np.array_equal(c_ref, c_sim), "cycle model != numpy"

    args.out.mkdir(parents=True, exist_ok=True)
    _write_matrix(args.out / "a.txt", a)
    _write_matrix(args.out / "w.txt", w)
    _write_matrix(args.out / "expected_c.txt", c_ref)

    (args.out / "meta.txt").write_text(
        "\n".join(
            [
                f"m {args.m}",
                f"n {N}",
                f"k {N}",
                "dtype_a int8",
                "dtype_w int8",
                "dtype_c int32",
                f"seed {args.seed}",
                f"lat {LAT}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"gen_vecs_sa: M={args.m} N={N} -> {args.out}")


if __name__ == "__main__":
    main()
