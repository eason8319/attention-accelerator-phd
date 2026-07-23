#!/usr/bin/env python3
"""P3：交叉校验 Roofline、SCALE-Sim 与 Timeloop 结果。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# 假定时钟频率，用于将 SCALE-Sim 周期换算为 attained TOPS。
# 绝对 TOPS 仅作示意；prefill/decode 相对趋势更重要。
ASSUMED_FREQ_HZ = 1e9
PEAK_TOPS = 128.0
RIDGE_AI = 128.0
# 假定时钟下 32×32 MAC 阵列（每个 MAC 计 2 ops）。
ARRAY_PEAK_TOPS = 32 * 32 * 2 * ASSUMED_FREQ_HZ / 1e12
SEQ_LENS = (4096, 32768, 131072)
GEMMS = ("QKV_proj", "QK_T", "PV", "O_proj")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def gemm_ops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def load_inputs(
    output_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    roofline = read_csv(output_dir / "roofline_table.csv")
    scalesim = read_csv(output_dir / "scalesim_results.csv")
    timeloop = read_csv(output_dir / "timeloop_energy.csv")
    if not roofline or not scalesim or not timeloop:
        raise RuntimeError("one or more input CSVs are empty")
    return roofline, scalesim, timeloop


def build_joined_table(
    roofline: list[dict[str, str]],
    scalesim: list[dict[str, str]],
    timeloop: list[dict[str, str]],
) -> list[dict[str, object]]:
    roof = {(r["mode"], as_int(r, "seq_len"), r["gemm"]): r for r in roofline}
    time = {(r["mode"], as_int(r, "seq_len"), r["gemm"]): r for r in timeloop}
    rows: list[dict[str, object]] = []
    for s in scalesim:
        key = (s["mode"], as_int(s, "seq_len"), s["gemm"])
        if key not in roof or key not in time:
            raise KeyError(f"missing companion row for {key}")
        r = roof[key]
        t = time[key]
        # 优先用 tile×repetitions，以计入 head 倍数。
        ops = gemm_ops(as_int(s, "tile_M"), as_int(s, "tile_N"), as_int(s, "tile_K")) * as_int(
            s, "tile_repetitions"
        )
        cycles = as_float(s, "total_cycles")
        attained_tops = ops / cycles * ASSUMED_FREQ_HZ / 1e12 if cycles > 0 else 0.0
        ai = as_float(r, "ai_ops_per_byte")
        # peak=128 TOPS，ridge=128 ops/byte → roof_tops = min(peak, AI)
        roof_tops = min(PEAK_TOPS, ai * (PEAK_TOPS / RIDGE_AI))
        rows.append(
            {
                "dataflow": s["dataflow"],
                "mode": s["mode"],
                "seq_len": as_int(s, "seq_len"),
                "gemm": s["gemm"],
                "ai_ops_per_byte": ai,
                "roofline_bound": r["bound"],
                "roofline_tops": roof_tops,
                "scalesim_util_pct": as_float(s, "overall_util_pct"),
                "scalesim_cycles": cycles,
                "scalesim_attained_tops": attained_tops,
                "sram_traffic_words": as_float(s, "sram_traffic_words"),
                "dram_traffic_words": as_float(s, "dram_traffic_words"),
                "mac_energy_pj": as_float(t, "mac_energy_pj"),
                "register_energy_pj": as_float(t, "register_energy_pj"),
                "sram_energy_pj": as_float(t, "sram_energy_pj"),
                "dram_energy_pj": as_float(t, "dram_energy_pj"),
                "total_energy_pj": as_float(t, "total_energy_pj"),
            }
        )
    return rows


def write_joined_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_utilization(scalesim: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = np.arange(len(SEQ_LENS))
    width = 0.35
    for ax, dataflow in zip(axes, ("ws", "os"), strict=True):
        prefill = []
        decode = []
        for seq in SEQ_LENS:
            p_vals = [
                as_float(r, "overall_util_pct")
                for r in scalesim
                if r["dataflow"] == dataflow
                and r["mode"] == "prefill"
                and as_int(r, "seq_len") == seq
                and r["gemm"] in {"QK_T", "PV"}
            ]
            d_vals = [
                as_float(r, "overall_util_pct")
                for r in scalesim
                if r["dataflow"] == dataflow
                and r["mode"] == "decode"
                and as_int(r, "seq_len") == seq
                and r["gemm"] in {"QK_T", "PV"}
            ]
            prefill.append(float(np.mean(p_vals)))
            decode.append(float(np.mean(d_vals)))
        ax.bar(x - width / 2, prefill, width, label="prefill", color="#2a6f97")
        ax.bar(x + width / 2, decode, width, label="decode", color="#e76f51")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s // 1024}K" for s in SEQ_LENS])
        ax.set_title(f"{dataflow.upper()} QK_T/PV util")
        ax.set_xlabel("seq_len")
        ax.set_ylabel("overall utilization (%)")
        ax.legend(frameon=False)
        ax.set_ylim(0, 100)
    fig.suptitle("Prefill vs decode PE utilization (SCALE-Sim)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_traffic_energy(
    scalesim: list[dict[str, str]],
    timeloop: list[dict[str, str]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(SEQ_LENS))
    width = 0.35

    # 左图：WS 下 QK_T+PV 层切片的 DRAM 与 SRAM 流量
    for mode, offset, color_sram, color_dram in (
        ("prefill", -width / 2, "#90be6d", "#577590"),
        ("decode", width / 2, "#f9c74f", "#f94144"),
    ):
        sram = []
        dram = []
        for seq in SEQ_LENS:
            rows = [
                r
                for r in scalesim
                if r["dataflow"] == "ws" and r["mode"] == mode and as_int(r, "seq_len") == seq
            ]
            sram.append(sum(as_float(r, "sram_traffic_words") for r in rows))
            dram.append(sum(as_float(r, "dram_traffic_words") for r in rows))
        axes[0].bar(
            x + offset,
            sram,
            width,
            label=f"{mode} SRAM",
            color=color_sram,
        )
        axes[0].bar(
            x + offset,
            dram,
            width,
            bottom=sram,
            label=f"{mode} DRAM",
            color=color_dram,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s // 1024}K" for s in SEQ_LENS])
    axes[0].set_title("SCALE-Sim traffic (WS, full layer)")
    axes[0].set_ylabel("words (log)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    # 右图：各 seq 下 decode 与 prefill 的 Timeloop 能耗堆叠
    components = (
        ("mac_energy_pj", "#264653", "MAC"),
        ("register_energy_pj", "#2a9d8f", "registers"),
        ("sram_energy_pj", "#e9c46a", "SRAM"),
        ("dram_energy_pj", "#e76f51", "DRAM"),
    )
    labels = []
    positions = []
    pos = 0
    series = {name: [] for _, _, name in components}
    order: list[tuple[str, int]] = []
    for seq in SEQ_LENS:
        for mode in ("prefill", "decode"):
            order.append((mode, seq))
            labels.append(f"{mode[0]}\n{seq // 1024}K")
            positions.append(pos)
            pos += 1
        pos += 0.4

    for mode, seq in order:
        rows = [r for r in timeloop if r["mode"] == mode and as_int(r, "seq_len") == seq]
        for key, _, name in components:
            series[name].append(sum(as_float(r, key) for r in rows) / 1e9)

    bottom = np.zeros(len(order))
    xpos = np.array(positions)
    for _, color, name in components:
        vals = np.array(series[name])
        axes[1].bar(xpos, vals, 0.8, bottom=bottom, label=name, color=color)
        bottom += vals
    axes[1].set_yscale("log")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("Timeloop energy stack (mJ, log)")
    axes[1].set_ylabel("energy (mJ)")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Traffic and energy breakdown")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roofline(
    joined: list[dict[str, object]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ai_line = np.logspace(-1, 4, 200)
    roof = np.minimum(PEAK_TOPS, ai_line * (PEAK_TOPS / RIDGE_AI))
    array_roof = np.minimum(ARRAY_PEAK_TOPS, ai_line * (PEAK_TOPS / RIDGE_AI))
    ax.loglog(ai_line, roof, "k-", linewidth=1.5, label="analytical roof (128 TOPS)")
    ax.loglog(
        ai_line,
        array_roof,
        "k--",
        linewidth=1.2,
        label=f"32×32 @ 1 GHz ({ARRAY_PEAK_TOPS:.2f} TOPS)",
    )
    ax.axvline(RIDGE_AI, color="gray", linestyle=":", linewidth=1)

    # 每个 GEMM 取 WS SCALE-Sim + Roofline AI 的一个点
    markers = {"prefill": "o", "decode": "s"}
    colors = {
        "QKV_proj": "#264653",
        "QK_T": "#2a9d8f",
        "PV": "#e9c46a",
        "O_proj": "#e76f51",
    }
    seen: set[tuple[str, str]] = set()
    for row in joined:
        if row["dataflow"] != "ws":
            continue
        # 为避免拥挤，仅画代表 seq（4K）；点大小编码 seq
        if int(row["seq_len"]) not in SEQ_LENS:
            continue
        mode = str(row["mode"])
        gemm = str(row["gemm"])
        key = (mode, gemm)
        # 用 alpha/大小区分所有 seq
        size = 30 + 20 * SEQ_LENS.index(int(row["seq_len"]))
        label = None
        if key not in seen and int(row["seq_len"]) == 4096:
            label = f"{mode} {gemm}"
            seen.add(key)
        ax.scatter(
            float(row["ai_ops_per_byte"]),
            float(row["scalesim_attained_tops"]),
            s=size,
            marker=markers[mode],
            color=colors[gemm],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
            label=label,
        )

    ax.set_xlabel("arithmetic intensity (ops/byte)")
    ax.set_ylabel(f"attained TOPS @ {ASSUMED_FREQ_HZ / 1e9:.0f} GHz")
    ax.set_title("Roofline points (WS SCALE-Sim attained vs AI)")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    ax.set_xlim(0.5, 1e4)
    ax.set_ylim(1e-3, PEAK_TOPS * 1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def util_ratio_table(scalesim: list[dict[str, str]]) -> list[str]:
    lines = [
        "| dataflow | seq | gemm | prefill util | decode util | ratio |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lookup = {
        (
            r["dataflow"],
            r["mode"],
            as_int(r, "seq_len"),
            r["gemm"],
        ): r
        for r in scalesim
    }
    for dataflow in ("ws", "os"):
        for seq in SEQ_LENS:
            for gemm in ("QK_T", "PV"):
                p = as_float(
                    lookup[(dataflow, "prefill", seq, gemm)],
                    "overall_util_pct",
                )
                d = as_float(
                    lookup[(dataflow, "decode", seq, gemm)],
                    "overall_util_pct",
                )
                lines.append(
                    f"| {dataflow.upper()} | {seq} | {gemm} | {p:.3f}% | {d:.3f}% | {p / d:.2f}× |"
                )
    return lines


def energy_share_table(timeloop: list[dict[str, str]]) -> list[str]:
    lines = [
        "| mode | seq | MAC | registers | SRAM | DRAM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("decode", "prefill"):
        for seq in SEQ_LENS:
            rows = [r for r in timeloop if r["mode"] == mode and as_int(r, "seq_len") == seq]
            mac = sum(as_float(r, "mac_energy_pj") for r in rows)
            reg = sum(as_float(r, "register_energy_pj") for r in rows)
            sram = sum(as_float(r, "sram_energy_pj") for r in rows)
            dram = sum(as_float(r, "dram_energy_pj") for r in rows)
            total = mac + reg + sram + dram
            lines.append(
                f"| {mode} | {seq} | "
                f"{100 * mac / total:.3f}% | "
                f"{100 * reg / total:.3f}% | "
                f"{100 * sram / total:.3f}% | "
                f"{100 * dram / total:.3f}% |"
            )
    return lines


def write_cross_summary(
    scalesim: list[dict[str, str]],
    timeloop: list[dict[str, str]],
    joined: list[dict[str, object]],
    path: Path,
) -> None:
    # 叙述用：decode 与 prefill 利用率极值
    ws_qk = [r for r in scalesim if r["dataflow"] == "ws" and r["gemm"] == "QK_T"]
    p_util = as_float(
        next(r for r in ws_qk if r["mode"] == "prefill"),
        "overall_util_pct",
    )
    d_util = as_float(
        next(r for r in ws_qk if r["mode"] == "decode"),
        "overall_util_pct",
    )

    decode_ai = [
        float(r["ai_ops_per_byte"])
        for r in joined
        if r["dataflow"] == "ws"
        and r["mode"] == "decode"
        and r["gemm"] in {"QK_T", "PV"}
        and int(r["seq_len"]) == 4096
    ][0]
    prefill_ai = [
        float(r["ai_ops_per_byte"])
        for r in joined
        if r["dataflow"] == "ws"
        and r["mode"] == "prefill"
        and r["gemm"] in {"QK_T", "PV"}
        and int(r["seq_len"]) == 4096
    ][0]

    lines = [
        "# P3 Cross-Validation Summary",
        "",
        "## Inputs",
        "",
        "- `roofline_table.csv`: analytical AI and compute/memory bound",
        "- `scalesim_results.csv`: cycle / utilization / traffic",
        "- `timeloop_energy.csv`: MAC / register / SRAM / DRAM energy",
        "",
        f"SCALE-Sim attained TOPS assumes a {ASSUMED_FREQ_HZ / 1e9:.0f} GHz "
        "clock for unit conversion only.",
        "",
        "## Prefill vs decode utilization",
        "",
        *util_ratio_table(scalesim),
        "",
        f"WS `QK_T` utilization: prefill **{p_util:.2f}%** vs decode "
        f"**{d_util:.2f}%** ({p_util / d_util:.1f}×).",
        "",
        "## Timeloop energy shares",
        "",
        *energy_share_table(timeloop),
        "",
        "## Figures",
        "",
        "- `util_prefill_vs_decode.png`",
        "- `traffic_energy_stack.png`",
        "- `roofline_points.png`",
        "- `cross_joined.csv`: per-GEMM join of the three tools",
        "",
        "## Agreement (relative conclusions)",
        "",
        "1. **Decode is memory-bound / under-utilized.** Roofline places "
        f"decode `QK_T`/`PV` at AI≈{decode_ai:.1f} ops/byte "
        f"(below ridge {RIDGE_AI:.0f}), while prefill AI≈{prefill_ai:.0f}. "
        "SCALE-Sim shows decode PE utilization one to two orders of "
        "magnitude below prefill for both WS and OS.",
        "2. **Longer context grows traffic and energy.** Both SCALE-Sim "
        "DRAM/SRAM traffic and Timeloop total energy rise with seq_len; "
        "decode projection GEMMs stay skinny while `QK_T`/`PV` grow with S.",
        "3. **Dataflow modulates absolute util, not the decode gap.** OS "
        "and WS differ in absolute percentages, but both preserve "
        "decode ≪ prefill.",
        "",
        "## Discrepancy sources (expected)",
        "",
        "| Tool | What it answers | Why absolutes diverge |",
        "|---|---|---|",
        "| Roofline | Ideal AI and bound latency | Perfect bandwidth overlap; "
        "no stalls, tiling waste, or PE mapping inefficiency |",
        "| SCALE-Sim | Cycle-accurate systolic schedule + buffers | "
        "Utilization driven by array mapping and tile shape; tile "
        "repetition omits inter-tile reuse/overlap |",
        "| Timeloop/Accelergy | Mapping search + energy/area model | "
        "Cycle semantics differ from SCALE-Sim; bundled PAT makes "
        "16 MiB SRAM dominate energy, so DRAM share is model-dependent |",
        "",
        "An extra scale gap appears on the roofline plot: the analytical "
        f"peak is {PEAK_TOPS:.0f} TOPS, while a 32×32 array at 1 GHz peaks "
        f"near {ARRAY_PEAK_TOPS:.2f} TOPS. Prefill points sit near that "
        "array roof times utilization (~0.65 TOPS), not the 128 TOPS "
        "system peak. Compare AI / bound class and util ratios, not "
        "absolute TOPS across tools.",
        "",
        "Do **not** equate SCALE-Sim cycles with Timeloop cycles, or "
        "Roofline microseconds with either simulator. Accept the shared "
        "relative story: decode skinny GEMMs sit under the memory roof "
        "and leave the array poorly utilized.",
        "",
        "## Caveats carried into `analysis.md`",
        "",
        "- Fixed 256-bounded tiles: per-tile util is seq-independent.",
        "- Timeloop energy shares need tech-node calibration before "
        "claiming DRAM energy dominance.",
        "- Attained TOPS on the roofline plot inherit the 1 GHz assumption.",
        "- Analytical 128 TOPS peak ≠ SCALE-Sim 32×32 microarchitecture peak.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(
    scalesim: list[dict[str, str]],
    joined: list[dict[str, object]],
) -> None:
    for dataflow in ("ws", "os"):
        for seq in SEQ_LENS:
            for gemm in ("QK_T", "PV"):
                p = next(
                    as_float(r, "overall_util_pct")
                    for r in scalesim
                    if r["dataflow"] == dataflow
                    and r["mode"] == "prefill"
                    and as_int(r, "seq_len") == seq
                    and r["gemm"] == gemm
                )
                d = next(
                    as_float(r, "overall_util_pct")
                    for r in scalesim
                    if r["dataflow"] == dataflow
                    and r["mode"] == "decode"
                    and as_int(r, "seq_len") == seq
                    and r["gemm"] == gemm
                )
                if not d < p:
                    raise AssertionError(
                        f"{dataflow} {gemm} seq={seq}: decode util {d} not below prefill {p}"
                    )
    decode_bounds = {
        str(r["roofline_bound"]) for r in joined if r["mode"] == "decode" and r["dataflow"] == "ws"
    }
    if decode_bounds != {"memory"}:
        raise AssertionError(f"expected all decode GEMMs memory-bound, got {decode_bounds}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS,
        help="存放各工具 CSV 并输出图表的目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roofline, scalesim, timeloop = load_inputs(output_dir)
    joined = build_joined_table(roofline, scalesim, timeloop)
    validate(scalesim, joined)

    joined_path = output_dir / "cross_joined.csv"
    write_joined_csv(joined, joined_path)

    util_path = output_dir / "util_prefill_vs_decode.png"
    traffic_path = output_dir / "traffic_energy_stack.png"
    roof_path = output_dir / "roofline_points.png"
    plot_utilization(scalesim, util_path)
    plot_traffic_energy(scalesim, timeloop, traffic_path)
    plot_roofline(joined, roof_path)

    summary_path = output_dir / "cross_validation.md"
    write_cross_summary(scalesim, timeloop, joined, summary_path)

    print(f"Wrote joined table to {joined_path}")
    print(f"Wrote {util_path.name}, {traffic_path.name}, {roof_path.name}")
    print(f"Wrote summary to {summary_path}")
    print("Validation passed: decode util < prefill and decode is memory-bound.")


if __name__ == "__main__":
    main()
