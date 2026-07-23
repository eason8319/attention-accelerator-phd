"""Tile 网格搜索与 Pareto 提取测试。"""

from __future__ import annotations

from pathlib import Path

from hw_config import HwConfig
from search import (
    best_latency,
    best_traffic,
    grid_search,
    pareto_minimize_two,
    plot_pareto,
    results_to_rows,
    run_sweep,
    tile_candidates,
    write_search_csv,
)
from simulator import SimResult
from workload import llama7b_attention


def _fake(
    br: int,
    bc: int,
    lat: float,
    traf: float,
    *,
    feasible: bool = True,
) -> SimResult:
    return SimResult(
        feasible=feasible,
        br=br,
        bc=bc,
        footprint_bytes=1.0,
        double_buffer_ok=True,
        double_buffer_enabled=True,
        latency_cycles=lat,
        latency_serial_cycles=lat,
        latency_db_cycles=lat,
        dram_traffic_bytes=traf,
        dma_load_bytes=traf,
        dma_store_bytes=0.0,
        compute_cycles_total=1.0,
        dma_load_cycles_total=1.0,
        dma_store_cycles_total=0.0,
        mac_work=1.0,
        pe_util=0.5,
        reason="ok",
    )


def test_tile_candidates_powers_and_seq() -> None:
    assert tile_candidates(100) == [16, 32, 64, 100]
    assert tile_candidates(64) == [16, 32, 64]


def test_pareto_front_filters_dominated() -> None:
    pts = [
        _fake(16, 16, lat=100, traf=50),
        _fake(32, 32, lat=80, traf=60),  # 延迟更好，流量更差
        _fake(64, 64, lat=90, traf=40),  # 中等延迟；排序后目前最佳流量
        _fake(8, 8, lat=120, traf=30),  # 最差延迟、最佳流量 → 在前沿上
        _fake(128, 128, lat=70, traf=70),  # 最佳延迟
        _fake(1, 1, lat=200, traf=10, feasible=False),
    ]
    front = pareto_minimize_two(pts)
    keys = {(r.br, r.bc) for r in front}
    # 按延迟排序：(128,70), (32,60), (64,40), (16,50), (8,30)
    # 流量改善时保留：(128,70), (32,60), (64,40), (8,30)
    # (16,50) 被 (64,40) 支配（更低或相等延迟下已有更好流量）…
    # (64,40) 之后 best_traffic=40；(16,50) 的 50>40 跳过；(8,30) 保留。
    assert keys == {(128, 128), (32, 32), (64, 64), (8, 8)}


def test_grid_search_decode_br_is_one() -> None:
    wl = llama7b_attention("decode", 512)
    results, front = grid_search(
        wl,
        br_values=[16, 32, 64],
        bc_values=[32, 64],
    )
    assert results
    assert all(r.br == 1 for r in results)
    assert front
    assert all(r.feasible for r in front)


def test_grid_search_prefill_has_feasible_and_optima() -> None:
    wl = llama7b_attention("prefill", 512)
    results, front = grid_search(wl, br_values=[16, 32, 64], bc_values=[16, 32, 64])
    feas = [r for r in results if r.feasible]
    assert feas
    assert front
    assert best_latency(results) is not None
    assert best_traffic(results) is not None
    # Pareto 点必须均可运行。
    assert all(r.feasible for r in front)


def test_csv_and_plot(tmp_path: Path) -> None:
    wl = llama7b_attention("prefill", 256)
    results, front = grid_search(wl, br_values=[16, 32], bc_values=[16, 32])
    rows = results_to_rows(wl, results, front)
    csv_path = tmp_path / "search.csv"
    png_path = tmp_path / "pareto.png"
    write_search_csv(rows, csv_path)
    plot_pareto(rows, png_path, title="test")
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0


def test_run_sweep_writes_outputs(tmp_path: Path) -> None:
    # 默认 SRAM 仍能放下小 tile；保持 sweep 很快。
    hw = HwConfig()
    rows = run_sweep(
        modes=("prefill",),
        seq_lens=(256,),
        hw=hw,
        out_dir=tmp_path,
    )
    assert rows
    assert (tmp_path / "search_results.csv").is_file()
    assert (tmp_path / "search_prefill_s256.csv").is_file()
    assert (tmp_path / "pareto_prefill_s256.png").is_file()
