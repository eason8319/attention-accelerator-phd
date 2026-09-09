"""将 P5 tile 仿真器趋势与 P3 SCALE-Sim 结果交叉校验。

只比较**相对**趋势（利用率差距、流量随序列长度、decode 访存压力）。
绝对周期不必对齐：SCALE-Sim 无跨 tile 复用/overlap；P5 增加 double buffering
与融合 FA 流量模型。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from hw_config import HwConfig, default_hw_config
from simulator import TileConfig, simulate
from workload import llama7b_attention

LEARNING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALESIM_CSV = LEARNING_ROOT / "p3_arch_eval" / "outputs" / "scalesim_results.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs"

# 与 P3 ≤256 固定 attention tile 对齐的代表尺寸。
PREFILL_TILE = TileConfig(br=256, bc=256)
DECODE_TILE = TileConfig(br=1, bc=256)

ATTENTION_GEMMS = ("QK_T", "PV")
SEQ_LENS = (4096, 32768, 131072)


@dataclass(frozen=True)
class ScaleSimAgg:
    mode: str
    seq_len: int
    dataflow: str
    util_pct: float
    total_cycles: float
    dram_words: float
    sram_words: float

    @property
    def dram_share(self) -> float:
        denom = self.dram_words + self.sram_words
        return self.dram_words / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def load_scalesim_attention(
    path: Path,
    *,
    dataflow: str = "ws",
) -> dict[tuple[str, int], ScaleSimAgg]:
    """聚合 SCALE-Sim CSV 中 QK_T+PV 行（周期/流量求和；利用率取均值）。"""
    if not path.is_file():
        raise FileNotFoundError(f"SCALE-Sim CSV not found: {path}")

    buckets: dict[tuple[str, int], list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataflow"] != dataflow:
                continue
            if row["gemm"] not in ATTENTION_GEMMS:
                continue
            key = (row["mode"], int(row["seq_len"]))
            buckets.setdefault(key, []).append(row)

    out: dict[tuple[str, int], ScaleSimAgg] = {}
    for (mode, seq), rows in sorted(buckets.items()):
        utils = [float(r["overall_util_pct"]) for r in rows]
        out[(mode, seq)] = ScaleSimAgg(
            mode=mode,
            seq_len=seq,
            dataflow=dataflow,
            util_pct=sum(utils) / len(utils),
            total_cycles=sum(float(r["total_cycles"]) for r in rows),
            dram_words=sum(float(r["dram_traffic_words"]) for r in rows),
            sram_words=sum(float(r["sram_traffic_words"]) for r in rows),
        )
    return out


def run_p5_attention(
    hw: HwConfig,
    seq_lens: Sequence[int] = SEQ_LENS,
) -> dict[tuple[str, int], object]:
    """在各序列长度上仿真融合 FA（prefill/decode）。"""
    results: dict[tuple[str, int], object] = {}
    for seq in seq_lens:
        pref = simulate(
            llama7b_attention("prefill", seq),
            PREFILL_TILE,
            hw=hw,
        )
        dec = simulate(
            llama7b_attention("decode", seq),
            DECODE_TILE,
            hw=hw,
        )
        results[("prefill", seq)] = pref
        results[("decode", seq)] = dec
    return results


def _ratio(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return a / b


def check_util_gap(
    scalesim: dict[tuple[str, int], ScaleSimAgg],
    p5: dict[tuple[str, int], object],
    seq: int = 4096,
) -> CheckResult:
    ss_p = scalesim[("prefill", seq)]
    ss_d = scalesim[("decode", seq)]
    p5_p = p5[("prefill", seq)]
    p5_d = p5[("decode", seq)]

    ss_ratio = _ratio(ss_p.util_pct, ss_d.util_pct)
    p5_ratio = _ratio(p5_p.pe_util, p5_d.pe_util)  # type: ignore[attr-defined]

    # 同向且数量级差距（SCALE-Sim WS ≈ 69×）。
    passed = ss_ratio > 10 and p5_ratio > 5
    detail = (
        f"SCALE-Sim WS util prefill/decode @S={seq}: "
        f"{ss_p.util_pct:.3f}% / {ss_d.util_pct:.3f}% = {ss_ratio:.1f}×; "
        f"P5 pe_util: {p5_p.pe_util:.4f} / {p5_d.pe_util:.4f} = {p5_ratio:.1f}×"  # type: ignore[attr-defined]
    )
    return CheckResult("decode_util_ll_prefill", passed, detail)


def check_traffic_scales_with_s(
    scalesim: dict[tuple[str, int], ScaleSimAgg],
    p5: dict[tuple[str, int], object],
    *,
    mode: str,
    s_lo: int = 4096,
    s_hi: int = 32768,
) -> CheckResult:
    ss_lo = scalesim[(mode, s_lo)]
    ss_hi = scalesim[(mode, s_hi)]
    p5_lo = p5[(mode, s_lo)]
    p5_hi = p5[(mode, s_hi)]

    ss_r = _ratio(ss_hi.dram_words, ss_lo.dram_words)
    p5_r = _ratio(
        p5_hi.dram_traffic_bytes,  # type: ignore[attr-defined]
        p5_lo.dram_traffic_bytes,  # type: ignore[attr-defined]
    )
    expected = (s_hi / s_lo) ** (2 if mode == "prefill" else 1)

    # 同号增长，且相对解析期望落在宽松倍数内。
    passed = (
        ss_r > 1.5
        and p5_r > 1.5
        and 0.25 * expected <= ss_r <= 4.0 * expected
        and 0.25 * expected <= p5_r <= 4.0 * expected
    )
    detail = (
        f"{mode} DRAM traffic S={s_hi}/S={s_lo}: "
        f"SCALE-Sim {ss_r:.2f}×, P5 {p5_r:.2f}× "
        f"(analytic ~{expected:.0f}× for {'N²' if mode == 'prefill' else 'N'})"
    )
    return CheckResult(f"traffic_scales_{mode}", passed, detail)


def check_latency_scales_with_s(
    scalesim: dict[tuple[str, int], ScaleSimAgg],
    p5: dict[tuple[str, int], object],
    *,
    mode: str,
    s_lo: int = 4096,
    s_hi: int = 32768,
) -> CheckResult:
    ss_r = _ratio(
        scalesim[(mode, s_hi)].total_cycles,
        scalesim[(mode, s_lo)].total_cycles,
    )
    p5_r = _ratio(
        p5[(mode, s_hi)].latency_cycles,  # type: ignore[attr-defined]
        p5[(mode, s_lo)].latency_cycles,  # type: ignore[attr-defined]
    )
    expected = (s_hi / s_lo) ** (2 if mode == "prefill" else 1)
    passed = ss_r > 1.5 and p5_r > 1.5 and abs(ss_r - p5_r) / max(ss_r, p5_r) < 0.5
    # 另要求接近解析量级（宽松）。
    passed = passed and 0.25 * expected <= p5_r <= 4.0 * expected
    detail = (
        f"{mode} latency/cycles S={s_hi}/S={s_lo}: "
        f"SCALE-Sim {ss_r:.2f}×, P5 {p5_r:.2f}× (expect ~{expected:.0f}×)"
    )
    return CheckResult(f"latency_scales_{mode}", passed, detail)


def check_decode_more_memory_bound(
    scalesim: dict[tuple[str, int], ScaleSimAgg],
    p5: dict[tuple[str, int], object],
    seq: int = 4096,
) -> CheckResult:
    ss_p = scalesim[("prefill", seq)]
    ss_d = scalesim[("decode", seq)]
    p5_p = p5[("prefill", seq)]
    p5_d = p5[("decode", seq)]

    ss_ok = ss_d.dram_share > ss_p.dram_share
    p5_ok = p5_d.dma_cycle_fraction > p5_p.dma_cycle_fraction  # type: ignore[attr-defined]
    passed = ss_ok and p5_ok
    detail = (
        f"@S={seq} SCALE-Sim DRAM/(SRAM+DRAM): "
        f"prefill {ss_p.dram_share:.3f} vs decode {ss_d.dram_share:.3f}; "
        f"P5 dma_cycle_fraction: "
        f"prefill {p5_p.dma_cycle_fraction:.4f} vs decode {p5_d.dma_cycle_fraction:.4f}"  # type: ignore[attr-defined]
    )
    return CheckResult("decode_more_memory_pressure", passed, detail)


def run_checks(
    scalesim: dict[tuple[str, int], ScaleSimAgg],
    p5: dict[tuple[str, int], object],
) -> list[CheckResult]:
    return [
        check_util_gap(scalesim, p5),
        check_traffic_scales_with_s(scalesim, p5, mode="prefill"),
        check_traffic_scales_with_s(scalesim, p5, mode="decode"),
        check_latency_scales_with_s(scalesim, p5, mode="prefill"),
        check_latency_scales_with_s(scalesim, p5, mode="decode"),
        check_decode_more_memory_bound(scalesim, p5),
    ]


def write_results(
    path: Path,
    *,
    scalesim: dict[tuple[str, int], ScaleSimAgg],
    p5: dict[tuple[str, int], object],
    checks: Sequence[CheckResult],
    scalesim_csv: Path,
    hw: HwConfig,
) -> None:
    """导出实际检查状态和逐点数据，不写固定叙述或结论。"""
    if path.suffix.lower() != ".json":
        raise ValueError("结果导出必须使用 .json，禁止覆盖报告")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"mode": mode, "seq_len": seq,
             "scalesim": asdict(scalesim[(mode, seq)]),
             "p5": asdict(p5[(mode, seq)])}
            for mode, seq in sorted(scalesim)]
    data = {"source_csv": str(scalesim_csv), "hardware": asdict(hw),
            "checks": [asdict(c) for c in checks], "rows": rows,
            "all_checks_passed": bool(checks) and all(c.passed for c in checks)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="趋势交叉校验：P5 tile_sim vs P3 SCALE-Sim")
    parser.add_argument(
        "--scalesim-csv",
        type=Path,
        default=DEFAULT_SCALESIM_CSV,
        help="P3 scalesim_results.csv 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="写入 cross_check_vs_scalesim_results.json 的目录",
    )
    parser.add_argument(
        "--dataflow",
        default="ws",
        choices=["ws", "os"],
        help="要聚合的 SCALE-Sim dataflow 列",
    )
    args = parser.parse_args(argv)

    hw = default_hw_config()
    scalesim = load_scalesim_attention(args.scalesim_csv, dataflow=args.dataflow)
    p5 = run_p5_attention(hw)
    checks = run_checks(scalesim, p5)

    out_json = args.out_dir / "cross_check_vs_scalesim_results.json"
    write_results(
        out_json,
        scalesim=scalesim,
        p5=p5,
        checks=checks,
        scalesim_csv=args.scalesim_csv,
        hw=hw,
    )

    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        print(f"[{flag}] {c.name}: {c.detail}")
    print(f"Wrote {out_json}")

    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
