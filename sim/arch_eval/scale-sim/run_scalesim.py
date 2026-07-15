#!/usr/bin/env python3
"""Run a bounded-tile SCALE-Sim sweep for attention GEMMs.

SCALE-Sim explicitly materializes operand and demand matrices. Monolithic
32K/128K prefill GEMMs are therefore impractical. This runner simulates one
representative instance of every exact tile shape and multiplies cycles and
access counts by the exact number of tiles. Inter-tile reuse and overlap are
intentionally not modeled.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scalesim.scale_sim import scalesim


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT.parent / "outputs"
SEQ_LENS = (4_096, 32_768, 131_072)
DATAFLOWS = ("ws", "os")
TILE_LIMIT = 256
HEADS = 32
HEAD_DIM = 128
HIDDEN = 4_096


@dataclass(frozen=True)
class Workload:
    mode: str
    seq_len: int
    gemm: str
    full_m: int
    full_n: int
    full_k: int
    multiplicity: int = 1

    @property
    def tile_m(self) -> int:
        return min(self.full_m, TILE_LIMIT)

    @property
    def tile_n(self) -> int:
        return min(self.full_n, TILE_LIMIT)

    @property
    def tile_k(self) -> int:
        return min(self.full_k, TILE_LIMIT)

    @property
    def repetitions(self) -> int:
        return (
            self.multiplicity
            * math.ceil(self.full_m / self.tile_m)
            * math.ceil(self.full_n / self.tile_n)
            * math.ceil(self.full_k / self.tile_k)
        )

    @property
    def layer_name(self) -> str:
        return f"{self.mode}_s{self.seq_len}_{self.gemm}"


def build_workloads() -> list[Workload]:
    workloads: list[Workload] = []
    for mode in ("prefill", "decode"):
        for seq_len in SEQ_LENS:
            tokens = seq_len if mode == "prefill" else 1
            attention_m = seq_len if mode == "prefill" else 1
            workloads.extend(
                (
                    Workload(
                        mode, seq_len, "QKV_proj",
                        tokens, 3 * HIDDEN, HIDDEN,
                    ),
                    Workload(
                        mode, seq_len, "QK_T",
                        attention_m, seq_len, HEAD_DIM, HEADS,
                    ),
                    Workload(
                        mode, seq_len, "PV",
                        attention_m, HEAD_DIM, seq_len, HEADS,
                    ),
                    Workload(
                        mode, seq_len, "O_proj",
                        tokens, HIDDEN, HIDDEN,
                    ),
                )
            )
    return workloads


def write_config(dataflow: str) -> Path:
    path = ROOT / f"arch_32x32_{dataflow}.cfg"
    path.write_text(
        f"""[general]
run_name = attention_32x32_{dataflow}

[architecture_presets]
ArrayHeight: 32
ArrayWidth: 32
IfmapSramSzkB: 6144
FilterSramSzkB: 6144
OfmapSramSzkB: 4096
IfmapOffset: 0
FilterOffset: 1000000000
OfmapOffset: 2000000000
Bandwidth: 32
Dataflow: {dataflow}
MemoryBanks: 1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 32
IfmapSRAMBankNum: 1
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 32
FilterSRAMBankNum: 1
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport: false
SparseRep: ellpack_block
OptimizedMapping: false
BlockSize: 8
RandomNumberGeneratorSeed: 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
""",
        encoding="utf-8",
    )
    return path


def write_topology(workloads: Iterable[Workload]) -> Path:
    path = ROOT / "attention_tiles.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Layer name", "M", "N", "K", "Sparsity", ""))
        for item in workloads:
            writer.writerow(
                (
                    item.layer_name,
                    item.tile_m,
                    item.tile_n,
                    item.tile_k,
                    "1:1",
                    "",
                )
            )
    return path


def write_layout_stub() -> Path:
    # SCALE-Sim 3.0.0 requires a layout path even when custom layouts are off.
    path = ROOT / "default_layout.csv"
    path.write_text("Layer name,\n", encoding="utf-8")
    return path


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items() if key}


def read_report(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [normalize_row(row) for row in csv.DictReader(handle)]


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def run_one(
    dataflow: str,
    config: Path,
    topology: Path,
    layout: Path,
) -> Path:
    raw_root = OUTPUT_ROOT / "scalesim_raw" / dataflow
    raw_root.mkdir(parents=True, exist_ok=True)
    simulator = scalesim(
        save_disk_space=True,
        verbose=False,
        config=str(config),
        topology=str(topology),
        layout=str(layout),
        input_type_gemm=True,
    )
    simulator.run_scale(top_path=str(raw_root))
    return raw_root / f"attention_32x32_{dataflow}"


def collect(
    dataflow: str,
    raw_dir: Path,
    workloads: list[Workload],
) -> list[dict[str, object]]:
    compute = read_report(raw_dir / "COMPUTE_REPORT.csv")
    detail = read_report(raw_dir / "DETAILED_ACCESS_REPORT.csv")
    if not (len(compute) == len(detail) == len(workloads)):
        raise RuntimeError("SCALE-Sim report row count does not match topology")

    rows: list[dict[str, object]] = []
    for item, comp, access in zip(workloads, compute, detail, strict=True):
        repeats = item.repetitions
        row: dict[str, object] = {
            "dataflow": dataflow,
            "mode": item.mode,
            "seq_len": item.seq_len,
            "gemm": item.gemm,
            "full_M": item.full_m,
            "full_N": item.full_n,
            "full_K": item.full_k,
            "tile_M": item.tile_m,
            "tile_N": item.tile_n,
            "tile_K": item.tile_k,
            "tile_repetitions": repeats,
            "total_cycles": (
                as_int(comp, "Total Cycles (incl. prefetch)") * repeats
            ),
            "compute_cycles": as_int(comp, "Total Cycles") * repeats,
            "stall_cycles": as_int(comp, "Stall Cycles") * repeats,
            "overall_util_pct": as_float(comp, "Overall Util %"),
            "mapping_efficiency_pct": as_float(
                comp, "Mapping Efficiency %"
            ),
            "compute_util_pct": as_float(comp, "Compute Util %"),
        }
        for level, operand, action in (
            ("sram", "ifmap", "Reads"),
            ("sram", "filter", "Reads"),
            ("sram", "ofmap", "Writes"),
            ("dram", "ifmap", "Reads"),
            ("dram", "filter", "Reads"),
            ("dram", "ofmap", "Writes"),
        ):
            report_operand = (
                operand.upper()
                if operand in {"ifmap", "ofmap"}
                else operand.title()
            )
            report_key = f"{level.upper()} {report_operand} {action}"
            row[f"{level}_{operand}_{action.lower()}"] = (
                as_int(access, report_key) * repeats
            )
        row["sram_traffic_words"] = sum(
            int(row[key])
            for key in (
                "sram_ifmap_reads",
                "sram_filter_reads",
                "sram_ofmap_writes",
            )
        )
        row["dram_traffic_words"] = sum(
            int(row[key])
            for key in (
                "dram_ifmap_reads",
                "dram_filter_reads",
                "dram_ofmap_writes",
            )
        )
        rows.append(row)
    return rows


def write_results(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_ROOT / "scalesim_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_ROOT / "scalesim_summary.md"
    lines = [
        "# SCALE-Sim 32×32 Attention Sweep",
        "",
        "## Method",
        "",
        "- SCALE-Sim version: 3.0.0",
        "- Array: 32×32; dataflows: WS and OS",
        "- SRAM: 6 MiB IFMAP + 6 MiB filter + 4 MiB OFMAP",
        "- Workloads: prefill/decode × 4K/32K/128K × four GEMMs",
        (
            "- Exact fixed tiles bounded by 256 per M/N/K dimension are "
            "simulated once and multiplied by their repetition count."
        ),
        (
            "- Aggregate cycles/traffic exclude inter-tile reuse and overlap; "
            "use them for trend comparison, not absolute end-to-end latency."
        ),
        "",
        "## Utilization",
        "",
        "| dataflow | seq | gemm | prefill util | decode util | ratio |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lookup = {
        (
            str(row["dataflow"]),
            str(row["mode"]),
            int(row["seq_len"]),
            str(row["gemm"]),
        ): row
        for row in rows
    }
    for dataflow in DATAFLOWS:
        for seq_len in SEQ_LENS:
            for gemm in ("QK_T", "PV"):
                prefill = lookup[(dataflow, "prefill", seq_len, gemm)]
                decode = lookup[(dataflow, "decode", seq_len, gemm)]
                p_util = float(prefill["overall_util_pct"])
                d_util = float(decode["overall_util_pct"])
                ratio = p_util / d_util if d_util else math.inf
                lines.append(
                    f"| {dataflow.upper()} | {seq_len} | {gemm} | "
                    f"{p_util:.3f}% | {d_util:.3f}% | {ratio:.2f}× |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Decode preserves only one query row per head, so both "
                "dataflows show severe PE under-utilization. WS and OS map "
                "different GEMM dimensions spatially/temporally, which "
                "explains their different absolute utilization."
            ),
            (
                "Because all sequence lengths use the same fixed tile shape, "
                "per-tile utilization is sequence-length independent; "
                "sequence length changes tile count, aggregate cycles and "
                "traffic."
            ),
            "",
            "Raw reports are under `outputs/scalesim_raw/`; aggregated cycle, "
            "utilization and SRAM/DRAM traffic are in `scalesim_results.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def validate(rows: list[dict[str, object]]) -> None:
    lookup = {
        (
            str(row["dataflow"]),
            str(row["mode"]),
            int(row["seq_len"]),
            str(row["gemm"]),
        ): row
        for row in rows
    }
    for dataflow in DATAFLOWS:
        for seq_len in SEQ_LENS:
            for gemm in ("QK_T", "PV"):
                prefill = float(
                    lookup[(dataflow, "prefill", seq_len, gemm)][
                        "overall_util_pct"
                    ]
                )
                decode = float(
                    lookup[(dataflow, "decode", seq_len, gemm)][
                        "overall_util_pct"
                    ]
                )
                if not decode < prefill:
                    raise AssertionError(
                        f"expected {dataflow.upper()} decode {gemm} "
                        f"utilization below prefill at seq={seq_len}: "
                        f"{decode} >= {prefill}"
                    )


def main() -> None:
    workloads = build_workloads()
    topology = write_topology(workloads)
    layout = write_layout_stub()
    all_rows: list[dict[str, object]] = []
    for dataflow in DATAFLOWS:
        config = write_config(dataflow)
        raw_dir = run_one(dataflow, config, topology, layout)
        all_rows.extend(collect(dataflow, raw_dir, workloads))

    validate(all_rows)
    result_path = write_results(all_rows)
    summary_path = write_summary(all_rows)
    print(f"Wrote {len(all_rows)} aggregated rows to {result_path}")
    print(f"Wrote summary to {summary_path}")
    print("Validation passed: WS/OS decode QK_T/PV utilization < prefill.")


if __name__ == "__main__":
    main()
