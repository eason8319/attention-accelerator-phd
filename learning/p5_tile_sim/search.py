"""对 FlashAttention (Br, Bc) tile 做网格搜索，并绘制延迟/流量 Pareto。

输出所有评估点的 CSV，以及标出非支配前沿的 PNG 散点图（P5 验收）。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from hw_config import HwConfig, default_hw_config
from simulator import SimResult, TileConfig, simulate
from workload import Workload, llama7b_attention

# 默认 2 的幂 tile 候选（搜索时再受 seq_len / SRAM 约束）。
DEFAULT_TILE_POWERS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class SearchRow:
    """一条已评估（或被拒绝）的 tile 配置。"""

    mode: str
    seq_len: int
    br: int
    bc: int
    feasible: bool
    double_buffer_enabled: bool
    footprint_bytes: float
    latency_cycles: float
    dram_traffic_bytes: float
    pe_util: float
    dma_cycle_fraction: float
    on_pareto: bool
    reason: str

    @classmethod
    def from_sim(
        cls,
        workload: Workload,
        result: SimResult,
        *,
        on_pareto: bool = False,
    ) -> SearchRow:
        return cls(
            mode=workload.mode,
            seq_len=workload.seq_len,
            br=result.br,
            bc=result.bc,
            feasible=result.feasible,
            double_buffer_enabled=result.double_buffer_enabled,
            footprint_bytes=result.footprint_bytes,
            latency_cycles=result.latency_cycles,
            dram_traffic_bytes=result.dram_traffic_bytes,
            pe_util=result.pe_util,
            dma_cycle_fraction=result.dma_cycle_fraction,
            on_pareto=on_pareto,
            reason=result.reason,
        )


def tile_candidates(
    seq_len: int,
    *,
    powers: Sequence[int] = DEFAULT_TILE_POWERS,
    include_seq: bool = True,
) -> list[int]:
    """不超过 ``seq_len`` 的 2 的幂尺寸（可选是否包含 ``seq_len`` 本身）。"""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    vals = [p for p in powers if p <= seq_len]
    if include_seq and seq_len not in vals:
        vals.append(seq_len)
    if not vals:
        vals = [seq_len]
    return sorted(set(vals))


def pareto_minimize_two(
    points: Sequence[SimResult],
) -> list[SimResult]:
    """最小化 (latency_cycles, dram_traffic_bytes) 的非支配集。"""
    feasible = [p for p in points if p.feasible and math.isfinite(p.latency_cycles)]
    if not feasible:
        return []
    # 先按延迟升序，再按流量升序。
    ordered = sorted(
        feasible,
        key=lambda r: (r.latency_cycles, r.dram_traffic_bytes, r.br, r.bc),
    )
    front: list[SimResult] = []
    best_traffic = math.inf
    for r in ordered:
        if r.dram_traffic_bytes < best_traffic:
            front.append(r)
            best_traffic = r.dram_traffic_bytes
    return front


def grid_search(
    workload: Workload,
    hw: HwConfig | None = None,
    *,
    br_values: Sequence[int] | None = None,
    bc_values: Sequence[int] | None = None,
    use_double_buffer: bool = True,
) -> tuple[list[SimResult], list[SimResult]]:
    """扫描 (Br, Bc)；返回 (全部结果, 可行点的 Pareto 前沿)。"""
    hw = default_hw_config() if hw is None else hw

    if workload.mode == "decode":
        brs: list[int] = [1]
    else:
        brs = list(br_values) if br_values is not None else tile_candidates(workload.n_q)

    bcs = list(bc_values) if bc_values is not None else tile_candidates(workload.n_kv)

    results: list[SimResult] = []
    seen: set[tuple[int, int]] = set()
    for br in brs:
        for bc in bcs:
            key = (1 if workload.mode == "decode" else br, bc)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                simulate(
                    workload,
                    TileConfig(br=br, bc=bc),
                    hw=hw,
                    use_double_buffer=use_double_buffer,
                )
            )

    front = pareto_minimize_two(results)
    return results, front


def best_latency(results: Sequence[SimResult]) -> SimResult | None:
    feas = [r for r in results if r.feasible]
    if not feas:
        return None
    return min(feas, key=lambda r: (r.latency_cycles, r.dram_traffic_bytes, r.br, r.bc))


def best_traffic(results: Sequence[SimResult]) -> SimResult | None:
    feas = [r for r in results if r.feasible]
    if not feas:
        return None
    return min(feas, key=lambda r: (r.dram_traffic_bytes, r.latency_cycles, r.br, r.bc))


def results_to_rows(
    workload: Workload,
    results: Sequence[SimResult],
    front: Sequence[SimResult],
) -> list[SearchRow]:
    front_keys = {(r.br, r.bc) for r in front}
    return [SearchRow.from_sim(workload, r, on_pareto=(r.br, r.bc) in front_keys) for r in results]


CSV_FIELDS = [
    "mode",
    "seq_len",
    "br",
    "bc",
    "feasible",
    "double_buffer_enabled",
    "footprint_bytes",
    "latency_cycles",
    "dram_traffic_bytes",
    "pe_util",
    "dma_cycle_fraction",
    "on_pareto",
    "reason",
]


def write_search_csv(rows: Iterable[SearchRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_pareto(
    rows: Sequence[SearchRow],
    path: Path,
    *,
    title: str,
) -> None:
    """散点图：延迟 vs 流量；高亮 Pareto 前沿。"""
    import matplotlib.pyplot as plt

    feas = [r for r in rows if r.feasible]
    if not feas:
        raise ValueError("no feasible points to plot")

    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.dram_traffic_bytes / 1e9 for r in feas]
    ys = [r.latency_cycles for r in feas]
    front = sorted(
        (r for r in feas if r.on_pareto),
        key=lambda r: r.dram_traffic_bytes,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(xs, ys, s=28, alpha=0.45, c="#4c78a8", label="feasible")
    if front:
        fx = [r.dram_traffic_bytes / 1e9 for r in front]
        fy = [r.latency_cycles for r in front]
        ax.plot(fx, fy, "-o", color="#e45756", markersize=6, label="Pareto")
        for r in front:
            ax.annotate(
                f"({r.br},{r.bc})",
                (r.dram_traffic_bytes / 1e9, r.latency_cycles),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_xlabel("DRAM traffic (GB)")
    ax.set_ylabel("Latency (cycles)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_sweep(
    *,
    modes: Sequence[str] = ("prefill", "decode"),
    seq_lens: Sequence[int] = (4096, 32768),
    hw: HwConfig | None = None,
    out_dir: Path | None = None,
) -> list[SearchRow]:
    """对每个 mode×seq 做网格搜索；写出 CSV 与 Pareto PNG。"""
    hw = default_hw_config() if hw is None else hw
    out = Path(__file__).resolve().parent / "outputs" if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[SearchRow] = []
    for mode in modes:
        for seq in seq_lens:
            wl = llama7b_attention(mode, seq)  # type: ignore[arg-type]
            results, front = grid_search(wl, hw)
            rows = results_to_rows(wl, results, front)
            all_rows.extend(rows)

            stem = f"{mode}_s{seq}"
            write_search_csv(rows, out / f"search_{stem}.csv")
            plot_pareto(
                rows,
                out / f"pareto_{stem}.png",
                title=f"Pareto: {mode}, seq={seq}",
            )

            lat = best_latency(results)
            traf = best_traffic(results)
            print(
                f"[{mode} seq={seq}] feasible="
                f"{sum(1 for r in results if r.feasible)}/{len(results)} "
                f"pareto={len(front)}"
            )
            if lat is not None:
                print(
                    f"  best latency: Br={lat.br} Bc={lat.bc} "
                    f"cycles={lat.latency_cycles:.3e} "
                    f"traffic={lat.dram_traffic_bytes:.3e} "
                    f"db={lat.double_buffer_enabled}"
                )
            if traf is not None:
                print(
                    f"  best traffic: Br={traf.br} Bc={traf.bc} "
                    f"cycles={traf.latency_cycles:.3e} "
                    f"traffic={traf.dram_traffic_bytes:.3e} "
                    f"db={traf.double_buffer_enabled}"
                )

    write_search_csv(all_rows, out / "search_results.csv")
    print(f"Wrote {out / 'search_results.csv'} and per-config CSV/PNG under {out}")
    return all_rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P5 tile 网格搜索与 Pareto 图")
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        default=[4096, 32768],
        help="要扫描的序列长度",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["prefill", "decode"],
        choices=["prefill", "decode"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认：learning/p5_tile_sim/outputs）",
    )
    args = parser.parse_args(argv)
    run_sweep(modes=args.modes, seq_lens=args.seq, out_dir=args.out)


if __name__ == "__main__":
    main()
