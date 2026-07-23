#!/usr/bin/env python3
"""比对 systolic DUT 的 C 矩阵与 numpy INT32 golden（逐位一致）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from fixedpoint import read_vec  # noqa: E402
from systolic_rtl_model import N, gemm_int8  # noqa: E402


def _read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(" ")
        meta[k] = v.strip()
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vec", type=Path, required=True)
    args = p.parse_args()

    meta = _read_meta(args.vec / "meta.txt")
    m = int(meta["m"])
    n = int(meta["n"])
    assert n == N

    a = read_vec(args.vec / "a.txt").astype(np.int8).reshape(m, n)
    w = read_vec(args.vec / "w.txt").astype(np.int8).reshape(n, n)
    expected = read_vec(args.vec / "expected_c.txt").astype(np.int32).reshape(m, n)
    dut = read_vec(args.vec / "dut_c.txt").astype(np.int32).reshape(m, n)

    ref = gemm_int8(a, w)
    assert np.array_equal(expected, ref)

    if dut.shape != expected.shape:
        raise SystemExit(f"shape mismatch dut={dut.shape} exp={expected.shape}")

    mism = int(np.sum(dut != expected))
    max_abs = int(np.max(np.abs(dut.astype(np.int64) - expected.astype(np.int64))))
    print(f"bit-exact: mismatches={mism}/{dut.size} max_abs_err={max_abs}")
    if mism:
        bad = np.argwhere(dut != expected)[:8]
        for i, j in bad:
            print(f"  C[{i},{j}] dut={int(dut[i, j])} exp={int(expected[i, j])}")
        raise SystemExit("compare_sa FAILED")
    print("compare_sa PASSED")


if __name__ == "__main__":
    main()
