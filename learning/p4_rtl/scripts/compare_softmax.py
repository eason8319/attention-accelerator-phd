#!/usr/bin/env python3
"""比对 online_softmax DUT 输出与逐位 golden。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from fixedpoint import Q6_10, fixed_to_float, read_vec  # noqa: E402
from softmax_rtl_model import L_FRAC, online_softmax_float_ref, online_softmax_rtl  # noqa: E402


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

    scores = read_vec(args.vec / "scores.txt")
    exp_m = int(read_vec(args.vec / "expected_m.txt")[0])
    exp_l = int(read_vec(args.vec / "expected_l.txt")[0])
    dut_m = int(read_vec(args.vec / "dut_m.txt")[0])
    dut_l = int(read_vec(args.vec / "dut_l.txt")[0])
    meta = _read_meta(args.vec / "meta.txt")
    block_size = int(meta["block_size"])

    print(f"bit-exact: m dut={dut_m} exp={exp_m} | l dut={dut_l} exp={exp_l}")

    # 重算 golden 并检验近似 block_size 无关性
    m0, l0 = online_softmax_rtl(scores, block_size)
    assert m0 == exp_m and l0 == exp_l
    for bs in (1, 3, 7, block_size, len(scores)):
        m1, l1 = online_softmax_rtl(scores, bs)
        if m1 != m0:
            raise SystemExit(f"block independence m failed at bs={bs}")
        rel = abs(l1 - l0) / max(l0, 1)
        if rel >= 1e-2:
            raise SystemExit(f"block independence l failed at bs={bs}: rel={rel:.3e}")

    if dut_m != exp_m or dut_l != exp_l:
        raise SystemExit("compare_softmax FAILED: mismatch vs RTL golden")

    # 软检查：与 math.exp 参考对比
    s_f = fixed_to_float(scores, Q6_10)
    m_ref, l_ref = online_softmax_float_ref(s_f)
    m_f = float(fixed_to_float(dut_m, Q6_10))
    l_f = float(dut_l) / float(1 << L_FRAC)
    l_rel = abs(l_f - l_ref) / max(abs(l_ref), 1e-12)
    print(f"vs math.exp: m_err={abs(m_f - m_ref):.3e} l_rel={l_rel:.3e} (informational)")
    print("compare_softmax PASSED")


if __name__ == "__main__":
    main()
