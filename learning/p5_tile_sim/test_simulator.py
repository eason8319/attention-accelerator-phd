"""Tile 级 FlashAttention 仿真器测试。"""

from __future__ import annotations

import math

import pytest
from hw_config import HwConfig
from simulator import TileConfig, simulate
from workload import llama7b_attention


def test_infeasible_when_tile_exceeds_sram() -> None:
    hw = HwConfig(
        sram_bytes=64 * 1024,
        sram_q_bytes=16 * 1024,
        sram_kv_bytes=16 * 1024,
        sram_o_bytes=16 * 1024,
        sram_stats_bytes=16 * 1024,
    )
    wl = llama7b_attention("prefill", 4096)
    # 极大 tile → footprint ≫ 64 KiB
    res = simulate(wl, TileConfig(br=512, bc=512), hw=hw)
    assert not res.feasible
    assert math.isinf(res.latency_cycles)
    assert "footprint" in res.reason


def test_double_buffer_requires_2x_footprint() -> None:
    wl = llama7b_attention("prefill", 32_768)
    hw = HwConfig()  # 16 MiB
    # INT8 等宽时 footprint = 2*d*(br+bc) ≈ 256*(br+bc)。
    # 选约 9 MiB 单缓冲，使 2× 超过 16 MiB。
    br, bc = 2048, 32_768
    fp = wl.tile_footprint_bytes(br, bc)
    assert fp <= hw.sram_bytes
    assert 2 * fp > hw.sram_bytes
    res = simulate(wl, TileConfig(br=br, bc=bc), hw=hw)
    assert res.feasible
    assert not res.double_buffer_ok
    assert not res.double_buffer_enabled
    assert res.latency_cycles == res.latency_serial_cycles


def test_db_latency_le_serial_when_enabled() -> None:
    wl = llama7b_attention("prefill", 2048)
    res = simulate(wl, TileConfig(br=64, bc=128))
    assert res.feasible and res.double_buffer_enabled
    assert res.latency_db_cycles <= res.latency_serial_cycles
    assert res.latency_cycles == res.latency_db_cycles


def test_decode_forces_br_one() -> None:
    wl = llama7b_attention("decode", 4096)
    res = simulate(wl, TileConfig(br=128, bc=256))
    assert res.feasible
    assert res.br == 1


def test_traffic_scales_with_seq_for_qk_kv_rescan() -> None:
    """外层 Q / 内层 KV：KV 流量 ∝ num_q_tiles * N ≈ N²/Br。"""
    short = simulate(llama7b_attention("prefill", 1024), TileConfig(64, 64))
    long = simulate(llama7b_attention("prefill", 2048), TileConfig(64, 64))
    assert short.feasible and long.feasible
    # 更长序列应严格搬动更多 DRAM 字节。
    assert long.dram_traffic_bytes > short.dram_traffic_bytes
    # 该数据流下序列×2 时约 ~4×（Q tile 数与 KV 长度均 ×2）。
    ratio = long.dram_traffic_bytes / short.dram_traffic_bytes
    assert 3.0 < ratio < 5.0


def test_small_tile_dma_not_hidden() -> None:
    """带宽受限时，极小 tile 使 DMA 暴露 → PE 利用率更低。"""
    # ~32 GB/s ≪ 默认 1 TB/s，使 load 时间相对 compute 可见。
    hw = HwConfig(dram_bandwidth_bytes_per_s=32e9)
    wl = llama7b_attention("prefill", 4096)
    small = simulate(wl, TileConfig(br=16, bc=16), hw=hw)
    medium = simulate(wl, TileConfig(br=128, bc=128), hw=hw)
    assert small.feasible and medium.feasible
    assert small.double_buffer_enabled and medium.double_buffer_enabled
    assert small.dma_cycle_fraction > medium.dma_cycle_fraction
    assert small.pe_util < medium.pe_util
    # 小 tile 上 DB 仍优于串行，但无法完全隐藏 DMA。
    assert small.latency_db_cycles < small.latency_serial_cycles
    assert small.latency_db_cycles > small.compute_cycles_total


def test_mac_work_matches_qk_plus_pv() -> None:
    wl = llama7b_attention("prefill", 256)
    res = simulate(wl, TileConfig(32, 32))
    expected = wl.batch * wl.heads * 2 * wl.n_q * wl.n_kv * wl.head_dim
    assert res.mac_work == pytest.approx(expected)


def test_force_serial_flag() -> None:
    wl = llama7b_attention("prefill", 1024)
    res = simulate(wl, TileConfig(32, 64), use_double_buffer=False)
    assert res.feasible and res.double_buffer_ok
    assert not res.double_buffer_enabled
    assert res.latency_cycles == res.latency_serial_cycles
