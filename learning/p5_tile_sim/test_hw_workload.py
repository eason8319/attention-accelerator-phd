"""hw_config 与 workload 的冒烟 / 单元检查（尚不依赖仿真器）。"""

from __future__ import annotations

import math

import pytest
from hw_config import HwConfig, default_hw_config
from workload import DEFAULT_SEQ_LENS, ElementBytes, Workload, llama7b_attention


def test_default_hw_aligns_with_p3() -> None:
    hw = default_hw_config()
    assert hw.pe_rows == 32 and hw.pe_cols == 32
    assert hw.macs_per_cycle == 1024
    assert hw.clock_hz == 1.0e9
    assert hw.sram_bytes == 16 * 1024 * 1024
    assert (
        hw.sram_q_bytes + hw.sram_kv_bytes + hw.sram_o_bytes + hw.sram_stats_bytes == hw.sram_bytes
    )
    assert hw.dram_bandwidth_bytes_per_s == 1.0e12
    assert hw.dram_bytes_per_cycle == pytest.approx(1000.0)
    # 32×32 × 2 ops × 1 GHz ≈ 2.048 TOPS（P3 笔记）。
    assert hw.array_peak_ops_per_s == pytest.approx(2.048e12)


def test_dma_cycles() -> None:
    hw = HwConfig()
    assert hw.dma_cycles(0) == 0.0
    assert hw.dma_cycles(1000.0) == pytest.approx(1.0)


def test_sram_partition_overflow_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds sram_bytes"):
        HwConfig(
            sram_bytes=1024,
            sram_q_bytes=512,
            sram_kv_bytes=512,
            sram_o_bytes=512,
            sram_stats_bytes=64,
        )


def test_workload_prefill_decode_shapes() -> None:
    pref = llama7b_attention("prefill", 4096)
    dec = llama7b_attention("decode", 4096)
    assert pref.n_q == 4096 and pref.n_kv == 4096
    assert dec.n_q == 1 and dec.n_kv == 4096
    assert pref.hidden == pref.heads * pref.head_dim


def test_mixed_precision_byte_hook() -> None:
    wl = llama7b_attention("prefill", 1024).with_bytes(q=0.5, k=0.5, v=1.0, o=2.0)
    assert wl.bytes == ElementBytes(q=0.5, k=0.5, v=1.0, o=2.0)
    # br=64, bc=128, d=128
    # 64*128*0.5 + 128*128*(0.5+1) + 64*128*2
    expected = 64 * 128 * 0.5 + 128 * 128 * 1.5 + 64 * 128 * 2.0
    assert wl.tile_footprint_bytes(64, 128) == pytest.approx(expected)


def test_tile_counts() -> None:
    wl = llama7b_attention("prefill", 1000)
    assert wl.num_q_tiles(256) == math.ceil(1000 / 256)
    assert wl.num_kv_tiles(128) == math.ceil(1000 / 128)
    dec = llama7b_attention("decode", 1000)
    assert dec.num_q_tiles(16) == 1


def test_default_seq_lens_match_p3() -> None:
    assert DEFAULT_SEQ_LENS == (4096, 32768, 131072)


def test_invalid_mode_and_bytes() -> None:
    with pytest.raises(ValueError):
        Workload(mode="train", seq_len=128)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ElementBytes(q=0.0)
