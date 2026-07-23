#!/usr/bin/env python3
"""P5 一键入口：单元测试 → 双端退化演示 → tile 搜索 → SCALE-Sim 交叉校验。

用法（仓库根目录或 ``learning/p5_tile_sim``）::

    conda activate p5-tile-sim
    python learning/p5_tile_sim/run_p5.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hw_config import HwConfig, default_hw_config
from search import run_sweep
from simulator import TileConfig, simulate
from validate_vs_scalesim import main as validate_main
from workload import llama7b_attention


def run_pytest() -> int:
    print("=== 1/4 pytest ===")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(HERE)],
        cwd=str(HERE),
    )
    return int(proc.returncode)


def demo_dual_end_degradation() -> None:
    """演示 tile 过小（DMA 暴露）与 tile 过大（SRAM / 无 DB）两端退化。"""
    print("=== 2/4 dual-end degradation demo ===")
    wl = llama7b_attention("prefill", 4096)

    # 限制带宽，使极小 tile 无法把 DMA 藏在 compute 之后。
    hw_bw = HwConfig(dram_bandwidth_bytes_per_s=32e9)
    small = simulate(wl, TileConfig(16, 16), hw=hw_bw)
    medium = simulate(wl, TileConfig(128, 128), hw=hw_bw)
    print(
        f"  small 16×16 @32GB/s: util={small.pe_util:.4f} "
        f"dma_frac={small.dma_cycle_fraction:.3f} "
        f"db={small.double_buffer_enabled}"
    )
    print(
        f"  medium 128×128 @32GB/s: util={medium.pe_util:.4f} "
        f"dma_frac={medium.dma_cycle_fraction:.3f} "
        f"db={medium.double_buffer_enabled}"
    )
    assert small.dma_cycle_fraction > medium.dma_cycle_fraction
    assert small.pe_util < medium.pe_util

    # 过大 tile：单缓冲能放下，双缓冲放不下 → 失去 double buffering。
    wl_long = llama7b_attention("prefill", 32_768)
    too_big_db = simulate(wl_long, TileConfig(2048, 32_768), hw=default_hw_config())
    print(
        f"  large 2048×32768 @16MiB: feasible={too_big_db.feasible} "
        f"fp={too_big_db.footprint_bytes / 1e6:.2f}MB "
        f"db_ok={too_big_db.double_buffer_ok} "
        f"(single-buffer only → no overlap)"
    )
    assert too_big_db.feasible and not too_big_db.double_buffer_ok

    # 真正超出 SRAM。
    hw_tiny = HwConfig(
        sram_bytes=256 * 1024,
        sram_q_bytes=64 * 1024,
        sram_kv_bytes=64 * 1024,
        sram_o_bytes=64 * 1024,
        sram_stats_bytes=64 * 1024,
    )
    over = simulate(wl, TileConfig(1024, 1024), hw=hw_tiny)
    print(
        f"  over-SRAM 1024×1024 @256KiB: feasible={over.feasible} "
        f"fp={over.footprint_bytes / 1024:.1f}KiB reason={over.reason!r}"
    )
    assert not over.feasible
    print("  dual-end demo OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P5 一键运行入口")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="跳过 pytest",
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        default=[4096, 32768],
        help="tile 搜索用的序列长度",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "outputs",
        help="输出目录",
    )
    args = parser.parse_args(argv)

    if not args.skip_tests:
        rc = run_pytest()
        if rc != 0:
            print("pytest failed; aborting")
            return rc

    demo_dual_end_degradation()

    print("=== 3/4 tile search + Pareto ===")
    run_sweep(
        modes=("prefill", "decode"),
        seq_lens=tuple(args.seq),
        out_dir=args.out,
    )

    print("=== 4/4 validate vs SCALE-Sim ===")
    rc = validate_main(["--out-dir", str(args.out)])
    if rc != 0:
        print("cross-check failed")
        return rc

    print()
    print("P5 run complete.")
    print(f"  outputs: {args.out}")
    print("  report:  learning/p5_tile_sim/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
