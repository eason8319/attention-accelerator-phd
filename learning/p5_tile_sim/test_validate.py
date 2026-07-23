"""SCALE-Sim 趋势交叉校验测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from hw_config import default_hw_config
from validate_vs_scalesim import (
    DEFAULT_SCALESIM_CSV,
    check_decode_more_memory_bound,
    check_latency_scales_with_s,
    check_traffic_scales_with_s,
    check_util_gap,
    load_scalesim_attention,
    run_checks,
    run_p5_attention,
    write_report,
)


@pytest.mark.skipif(
    not DEFAULT_SCALESIM_CSV.is_file(),
    reason="缺少 P3 scalesim_results.csv",
)
def test_load_scalesim_has_ws_attention() -> None:
    data = load_scalesim_attention(DEFAULT_SCALESIM_CSV, dataflow="ws")
    assert ("prefill", 4096) in data
    assert ("decode", 4096) in data
    pref = data[("prefill", 4096)]
    dec = data[("decode", 4096)]
    assert pref.util_pct > 50
    assert dec.util_pct < 5
    assert pref.util_pct / dec.util_pct > 30


@pytest.mark.skipif(
    not DEFAULT_SCALESIM_CSV.is_file(),
    reason="缺少 P3 scalesim_results.csv",
)
def test_all_trend_checks_pass(tmp_path: Path) -> None:
    hw = default_hw_config()
    scalesim = load_scalesim_attention(DEFAULT_SCALESIM_CSV)
    p5 = run_p5_attention(hw)
    checks = run_checks(scalesim, p5)
    assert checks
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]

    out = tmp_path / "cross_check_vs_scalesim.md"
    write_report(
        out,
        scalesim=scalesim,
        p5=p5,
        checks=checks,
        scalesim_csv=DEFAULT_SCALESIM_CSV,
        hw=hw,
    )
    text = out.read_text(encoding="utf-8")
    assert "**结论**：PASS" in text
    assert "decode_util_ll_prefill" in text


@pytest.mark.skipif(
    not DEFAULT_SCALESIM_CSV.is_file(),
    reason="缺少 P3 scalesim_results.csv",
)
def test_individual_check_helpers() -> None:
    hw = default_hw_config()
    scalesim = load_scalesim_attention(DEFAULT_SCALESIM_CSV)
    p5 = run_p5_attention(hw)
    assert check_util_gap(scalesim, p5).passed
    assert check_traffic_scales_with_s(scalesim, p5, mode="prefill").passed
    assert check_traffic_scales_with_s(scalesim, p5, mode="decode").passed
    assert check_latency_scales_with_s(scalesim, p5, mode="prefill").passed
    assert check_latency_scales_with_s(scalesim, p5, mode="decode").passed
    assert check_decode_more_memory_bound(scalesim, p5).passed
