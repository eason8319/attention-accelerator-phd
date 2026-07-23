#!/usr/bin/env python3
"""从带插桩的 P1 score 行生成 online_softmax 测试向量。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from fixedpoint import Q6_10, float_to_fixed, write_vec  # noqa: E402
from online_softmax_golden import sample_attention_score_rows  # noqa: E402
from softmax_rtl_model import online_softmax_rtl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--row-index", type=int, default=0, help="使用第几条采样行")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="使用长度为 --seq 的 RNG 行，而非 P1 attention score",
    )
    args = p.parse_args()

    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        row = rng.normal(0.0, 1.5, size=args.seq).astype(np.float64)
    else:
        rows = sample_attention_score_rows(seq=args.seq, seed=args.seed, max_rows=8)
        if not rows:
            raise SystemExit("no score rows sampled")
        row = rows[args.row_index % len(rows)]
    # 适配 Q6.10 动态范围
    row = np.clip(row, Q6_10.min_val, Q6_10.max_val)

    scores_raw = float_to_fixed(row, Q6_10)
    m_raw, l_raw = online_softmax_rtl(scores_raw, args.block_size)

    # 跨 block_size：m 精确，l 在 1% 内
    m2, l2 = online_softmax_rtl(scores_raw, max(1, args.block_size // 2 or 1))
    assert m2 == m_raw
    assert abs(l2 - l_raw) / max(l_raw, 1) < 1e-2

    args.out.mkdir(parents=True, exist_ok=True)
    write_vec(args.out / "scores.txt", scores_raw)
    write_vec(args.out / "expected_m.txt", np.array([m_raw], dtype=np.int64))
    write_vec(args.out / "expected_l.txt", np.array([l_raw], dtype=np.int64))

    meta = [
        f"n {len(scores_raw)}",
        f"block_size {args.block_size}",
        f"fmt_score {Q6_10}",
        "fmt_l UQ8.24",
        f"seed {args.seed}",
        f"row_index {args.row_index}",
        f"seq {args.seq}",
    ]
    (args.out / "meta.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    print(
        f"gen_vecs_softmax: n={len(scores_raw)} block={args.block_size} "
        f"m={m_raw} l={l_raw} -> {args.out}"
    )


if __name__ == "__main__":
    main()
