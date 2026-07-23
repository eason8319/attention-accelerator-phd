#!/usr/bin/env python3
"""生成 exp_approx 测试向量（Q6.10 输入，UQ0.24 期望输出）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from exp_approx import sample_attention_exp_inputs  # noqa: E402
from exp_rtl_model import exp_approx_rtl  # noqa: E402
from fixedpoint import Q6_10, UQ0_24, float_to_fixed, write_vec  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="输出向量目录")
    p.add_argument("--max-samples", type=int, default=8192, help="向量长度上限")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 真实 attention score + 若干手工角点
    x_real = sample_attention_exp_inputs(seq=256, heads=2, block_size=64, seed=args.seed)
    # 为仿真速度子采样，同时覆盖分布
    rng = np.random.default_rng(args.seed)
    if x_real.size > args.max_samples - 16:
        idx = rng.choice(x_real.size, size=args.max_samples - 16, replace=False)
        x_real = x_real[idx]

    corners = np.array(
        [0.0, -0.001, -0.5, -1.0, -2.0, -4.0, -8.0, -10.0, -16.0, -31.0],
        dtype=np.float64,
    )
    x = np.concatenate([corners, x_real])
    x = np.minimum(x, 0.0)

    x_raw = float_to_fixed(x, Q6_10)
    y_raw = exp_approx_rtl(x_raw)

    write_vec(args.out / "x.txt", x_raw)
    write_vec(args.out / "expected.txt", y_raw)

    meta = args.out / "meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"n {len(x_raw)}",
                f"fmt_in {Q6_10}",
                f"fmt_out {UQ0_24}",
                "latency 3",
                f"seed {args.seed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"gen_vecs_exp: wrote {len(x_raw)} samples to {args.out}")


if __name__ == "__main__":
    main()
