#!/usr/bin/env python3
"""Run Timeloop/Accelergy on the same bounded tiles as SCALE-Sim."""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCH_EVAL = ROOT.parent
OUTPUTS = ARCH_EVAL / "outputs"
SCALESIM_RESULTS = OUTPUTS / "scalesim_results.csv"
IMAGE = "timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64"


@dataclass(frozen=True)
class TileResult:
    cycles: int
    computes: int
    mac_fj_per_compute: float
    register_fj_per_compute: float
    sram_fj_per_compute: float
    dram_fj_per_compute: float


def docker_run(entrypoint: str, workdir: str, *args: str) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        entrypoint,
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        workdir,
        IMAGE,
        *args,
    ]
    subprocess.run(command, check=True)


def load_workloads() -> list[dict[str, str]]:
    with SCALESIM_RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # WS and OS rows have identical GEMM/tile metadata; retain one copy.
    selected = [row for row in rows if row["dataflow"] == "ws"]
    if len(selected) != 24:
        raise RuntimeError(
            f"expected 24 SCALE-Sim workloads, found {len(selected)}"
        )
    return selected


def shape_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        int(row["tile_M"]),
        int(row["tile_N"]),
        int(row["tile_K"]),
    )


def write_problem(directory: Path, m: int, n: int, k: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # A GEMM MxK @ KxN is represented as a 1x1 convolution:
    # P=M, C=K, K=N; all other dimensions are one.
    (directory / "problem.yaml").write_text(
        f"""problem:
  shape: cnn_layer
  R: 1
  S: 1
  P: {m}
  Q: 1
  C: {k}
  K: {n}
  N: 1
  Wstride: 1
  Hstride: 1
  Wdilation: 1
  Hdilation: 1
""",
        encoding="utf-8",
    )


def parse_stats(path: Path) -> TileResult:
    text = path.read_text(encoding="utf-8")
    summary = text.split("Summary Stats", maxsplit=1)[-1]
    cycles_match = re.search(r"Cycles:\s+(\d+)", summary)
    computes_match = re.search(r"Computes =\s+(\d+)", summary)
    if not cycles_match or not computes_match:
        raise RuntimeError(f"could not parse summary from {path}")

    values: dict[str, float] = {}
    for name in ("__ARITH__", "Registers", "GlobalBuffer", "DRAM"):
        match = re.search(
            rf"^\s*{re.escape(name)}\s*=\s*([0-9.eE+-]+)",
            summary,
            flags=re.MULTILINE,
        )
        if not match:
            raise RuntimeError(f"missing {name} energy in {path}")
        values[name] = float(match.group(1))

    return TileResult(
        cycles=int(cycles_match.group(1)),
        computes=int(computes_match.group(1)),
        mac_fj_per_compute=values["__ARITH__"],
        register_fj_per_compute=values["Registers"],
        sram_fj_per_compute=values["GlobalBuffer"],
        dram_fj_per_compute=values["DRAM"],
    )


def run_tiles(
    workloads: list[dict[str, str]],
) -> dict[tuple[int, int, int], TileResult]:
    results: dict[tuple[int, int, int], TileResult] = {}
    for m, n, k in sorted({shape_key(row) for row in workloads}):
        directory = ROOT / "generated" / f"m{m}_n{n}_k{k}"
        write_problem(directory, m, n, k)
        docker_run(
            "timeloop-mapper",
            f"/workspace/generated/{directory.name}",
            "../../arch.yaml",
            "problem.yaml",
            "../../mapper.yaml",
            "../../constraints.yaml",
        )
        results[(m, n, k)] = parse_stats(
            directory / "timeloop-mapper.stats.txt"
        )
    return results


def energy_pj(fj_per_compute: float, computes: int, repeats: int) -> float:
    return fj_per_compute * computes * repeats / 1000.0


def build_energy_rows(
    workloads: list[dict[str, str]],
    tile_results: dict[tuple[int, int, int], TileResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for workload in workloads:
        tile = tile_results[shape_key(workload)]
        repeats = int(workload["tile_repetitions"])
        components = {
            "mac_energy_pj": energy_pj(
                tile.mac_fj_per_compute, tile.computes, repeats
            ),
            "register_energy_pj": energy_pj(
                tile.register_fj_per_compute, tile.computes, repeats
            ),
            "sram_energy_pj": energy_pj(
                tile.sram_fj_per_compute, tile.computes, repeats
            ),
            "dram_energy_pj": energy_pj(
                tile.dram_fj_per_compute, tile.computes, repeats
            ),
        }
        total = sum(components.values())
        rows.append(
            {
                "mode": workload["mode"],
                "seq_len": int(workload["seq_len"]),
                "gemm": workload["gemm"],
                "full_M": int(workload["full_M"]),
                "full_N": int(workload["full_N"]),
                "full_K": int(workload["full_K"]),
                "tile_M": int(workload["tile_M"]),
                "tile_N": int(workload["tile_N"]),
                "tile_K": int(workload["tile_K"]),
                "tile_repetitions": repeats,
                "cycles": tile.cycles * repeats,
                **components,
                "total_energy_pj": total,
                "mac_pct": 100 * components["mac_energy_pj"] / total,
                "register_pct": (
                    100 * components["register_energy_pj"] / total
                ),
                "sram_pct": 100 * components["sram_energy_pj"] / total,
                "dram_pct": 100 * components["dram_energy_pj"] / total,
            }
        )
    return rows


def write_energy_csv(rows: list[dict[str, object]]) -> Path:
    path = OUTPUTS / "timeloop_energy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_accelergy() -> None:
    (ROOT / "accelergy_outputs").mkdir(exist_ok=True)
    docker_run(
        "accelergy",
        "/workspace",
        "arch_accelergy.yaml",
        "-o",
        "accelergy_outputs",
    )


def parse_area() -> list[dict[str, object]]:
    path = ROOT / "accelergy_outputs" / "ART_summary.yaml"
    text = path.read_text(encoding="utf-8")
    specs = (
        ("Register", 1024),
        ("mac", 1024),
        ("DRAM", 1),
        ("GlobalBuffer", 1),
    )
    rows: list[dict[str, object]] = []
    for component, instances in specs:
        match = re.search(
            rf"name:\s+\S*{component}\s*\n\s*area:\s*([0-9.eE+-]+)",
            text,
        )
        if not match:
            raise RuntimeError(f"missing {component} area in {path}")
        area_um2 = float(match.group(1))
        rows.append(
            {
                "component": component,
                "instances": instances,
                "area_per_instance_um2": area_um2,
                "total_area_mm2": area_um2 * instances / 1e6,
            }
        )
    out = OUTPUTS / "timeloop_area.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(
    energy_rows: list[dict[str, object]],
    area_rows: list[dict[str, object]],
) -> Path:
    path = OUTPUTS / "timeloop_summary.md"
    lines = [
        "# Timeloop + Accelergy Summary",
        "",
        "## Method",
        "",
        "- Official image: `timeloopaccelergy/"
        "timeloop-accelergy-pytorch:latest-amd64`",
        "- Architecture: 32×32 INT8 MACs, 16 MiB global SRAM, DRAM",
        "- Mapper: energy then delay; fixed C×K spatial mapping",
        "- Workloads use the same bounded tiles and repetition counts as SCALE-Sim",
        "- Energy source: Timeloop PAT; area source: Accelergy/CACTI/Aladdin",
        "",
        "Official 2020 ISPASS tutorial exercise 00 passes in this image. "
        "The repository's newer v0.4 `example_designs` do not parse unchanged "
        "because their schema is newer than the bundled front-end; this "
        "project therefore uses the compatible legacy schema.",
        "",
        "## Per-layer energy breakdown",
        "",
        "| mode | seq | MAC | registers | SRAM | DRAM | total (mJ) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in energy_rows:
        grouped.setdefault(
            (str(row["mode"]), int(row["seq_len"])), []
        ).append(row)
    for (mode, seq_len), rows in sorted(grouped.items()):
        component_keys = (
            "mac_energy_pj",
            "register_energy_pj",
            "sram_energy_pj",
            "dram_energy_pj",
        )
        values = {
            key: sum(float(row[key]) for row in rows)
            for key in component_keys
        }
        total = sum(values.values())
        lines.append(
            f"| {mode} | {seq_len} | "
            f"{100 * values['mac_energy_pj'] / total:.3f}% | "
            f"{100 * values['register_energy_pj'] / total:.3f}% | "
            f"{100 * values['sram_energy_pj'] / total:.3f}% | "
            f"{100 * values['dram_energy_pj'] / total:.3f}% | "
            f"{total / 1e9:.6g} |"
        )
    total_area = sum(float(row["total_area_mm2"]) for row in area_rows)
    lines.extend(
        [
            "",
            "## Area estimate",
            "",
            "| component | instances | total area (mm²) |",
            "|---|---:|---:|",
        ]
    )
    for row in area_rows:
        lines.append(
            f"| {row['component']} | {row['instances']} | "
            f"{float(row['total_area_mm2']):.6g} |"
        )
    lines.extend(
        [
            f"| **Total** | — | **{total_area:.6g}** |",
            "",
            "## Interpretation",
            "",
            "Under the bundled 45 nm PAT model, the 16 MiB global SRAM—not "
            "DRAM—dominates dynamic energy. This does not support a literal "
            "\"DRAM energy dominates\" claim; the robust conclusion from "
            "Roofline/SCALE-Sim is bandwidth pressure and low decode PE "
            "utilization. Absolute energy shares require technology and "
            "memory-model calibration before publication.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    workloads = load_workloads()
    tile_results = run_tiles(workloads)
    energy_rows = build_energy_rows(workloads, tile_results)
    energy_path = write_energy_csv(energy_rows)
    run_accelergy()
    area_rows = parse_area()
    summary_path = write_summary(energy_rows, area_rows)
    print(f"Wrote {len(energy_rows)} energy rows to {energy_path}")
    print(f"Wrote area estimates to {OUTPUTS / 'timeloop_area.csv'}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
