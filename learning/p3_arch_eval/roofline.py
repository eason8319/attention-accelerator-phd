#!/usr/bin/env python3
"""LLaMA-7B 规模单层 attention 的解析 Roofline 模型。

模型刻意采用简单、可复现的流量假设：每个 GEMM 从 HBM 各读一次 A、B，写一次 C。
这是理论基线，而非考虑 cache 或 tiling 的仿真器。
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEQ_LENS = (4_096, 32_768, 131_072)


@dataclass(frozen=True)
class HardwareConfig:
    peak_ops_per_s: float = 128e12
    memory_bytes_per_s: float = 1e12
    sram_bytes: int = 16 * 1024 * 1024
    element_bytes: int = 1

    @property
    def ridge_ai(self) -> float:
        return self.peak_ops_per_s / self.memory_bytes_per_s


@dataclass(frozen=True)
class ModelConfig:
    batch: int = 1
    hidden: int = 4_096
    heads: int = 32
    head_dim: int = 128

    def validate(self) -> None:
        if self.hidden != self.heads * self.head_dim:
            raise ValueError(
                "hidden must equal heads * head_dim, got "
                f"{self.hidden} != {self.heads} * {self.head_dim}"
            )


@dataclass(frozen=True)
class Gemm:
    name: str
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class RooflineRow:
    mode: str
    seq_len: int
    gemm: Gemm
    ops: int
    bytes_moved: int
    ai: float
    compute_s: float
    memory_s: float
    bound_s: float
    bound: str
    ridge_ai: float


def attention_gemms(mode: str, seq_len: int, model: ModelConfig) -> tuple[Gemm, ...]:
    """返回单层 attention 的 GEMM 序列。

    按 head 的 QK^T/PV GEMM 通过将 batch 与 head 数并入 M 表示为一个 batched GEMM。
    这样保留总算量与总流量，同时暴露 decode 时 M=head 数的瘦矩阵形状。
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if mode not in {"prefill", "decode"}:
        raise ValueError(f"unsupported mode: {mode}")

    tokens = model.batch * (seq_len if mode == "prefill" else 1)
    attention_rows = model.batch * model.heads * (seq_len if mode == "prefill" else 1)

    return (
        Gemm("QKV_proj", tokens, 3 * model.hidden, model.hidden),
        Gemm("QK_T", attention_rows, seq_len, model.head_dim),
        Gemm("PV", attention_rows, model.head_dim, seq_len),
        Gemm("O_proj", tokens, model.hidden, model.hidden),
    )


def evaluate_gemm(
    mode: str,
    seq_len: int,
    gemm: Gemm,
    hardware: HardwareConfig,
) -> RooflineRow:
    """用 ops=2MNK、bytes=(MK+KN+MN)*element_bytes 评估单个 GEMM。"""
    ops = 2 * gemm.m * gemm.n * gemm.k
    bytes_moved = (gemm.m * gemm.k + gemm.k * gemm.n + gemm.m * gemm.n) * hardware.element_bytes
    ai = ops / bytes_moved
    compute_s = ops / hardware.peak_ops_per_s
    memory_s = bytes_moved / hardware.memory_bytes_per_s
    bound_s = max(compute_s, memory_s)
    bound = "compute" if compute_s >= memory_s else "memory"
    return RooflineRow(
        mode=mode,
        seq_len=seq_len,
        gemm=gemm,
        ops=ops,
        bytes_moved=bytes_moved,
        ai=ai,
        compute_s=compute_s,
        memory_s=memory_s,
        bound_s=bound_s,
        bound=bound,
        ridge_ai=hardware.ridge_ai,
    )


def evaluate(
    seq_lens: Iterable[int],
    model: ModelConfig,
    hardware: HardwareConfig,
) -> list[RooflineRow]:
    model.validate()
    rows: list[RooflineRow] = []
    for mode in ("prefill", "decode"):
        for seq_len in seq_lens:
            rows.extend(
                evaluate_gemm(mode, seq_len, gemm, hardware)
                for gemm in attention_gemms(mode, seq_len, model)
            )
    return rows


def write_csv(rows: Sequence[RooflineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "mode",
        "seq_len",
        "gemm",
        "M",
        "N",
        "K",
        "flops",
        "bytes",
        "ai_ops_per_byte",
        "t_compute_s",
        "t_memory_s",
        "t_bound_s",
        "bound",
        "ridge_ai",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.mode,
                    "seq_len": row.seq_len,
                    "gemm": row.gemm.name,
                    "M": row.gemm.m,
                    "N": row.gemm.n,
                    "K": row.gemm.k,
                    "flops": f"{row.ops:.6e}",
                    "bytes": f"{row.bytes_moved:.6e}",
                    "ai_ops_per_byte": f"{row.ai:.6f}",
                    "t_compute_s": f"{row.compute_s:.6e}",
                    "t_memory_s": f"{row.memory_s:.6e}",
                    "t_bound_s": f"{row.bound_s:.6e}",
                    "bound": row.bound,
                    "ridge_ai": f"{row.ridge_ai:.6f}",
                }
            )


def grouped_rows(
    rows: Sequence[RooflineRow],
) -> dict[tuple[str, int], list[RooflineRow]]:
    groups: dict[tuple[str, int], list[RooflineRow]] = {}
    for row in rows:
        groups.setdefault((row.mode, row.seq_len), []).append(row)
    return groups


def write_markdown(
    rows: Sequence[RooflineRow],
    hardware: HardwareConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Roofline table (LLaMA-7B-scale attention layer)",
        "",
        "## Assumption",
        "",
        "Each GEMM reads A/B once from HBM and writes C once; ops count uses $2MNK$.",
        "",
        "## Hardware",
        "",
        f"- Peak compute: **{hardware.peak_ops_per_s / 1e12:g} TOPS** INT8",
        f"- HBM bandwidth: **{hardware.memory_bytes_per_s / 1e12:g} TB/s**",
        f"- SRAM (reference): **{hardware.sram_bytes / 1024**2:g} MB**",
        f"- Element size: **{hardware.element_bytes} byte** (INT8)",
        f"- Ridge AI: **{hardware.ridge_ai:g} ops/byte**",
        "",
        "## Per-GEMM",
        "",
        "| mode | seq | gemm | M | N | K | AI | bound | t_bound (us) |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.mode} | {row.seq_len} | {row.gemm.name} | "
            f"{row.gemm.m} | {row.gemm.n} | {row.gemm.k} | "
            f"{row.ai:.4g} | {row.bound} | {row.bound_s * 1e6:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Layer summary (sum of per-GEMM bound latencies)",
            "",
            "| mode | seq | AI | bound | t_layer (us) |",
            "|---|---:|---:|---|---:|",
        ]
    )
    groups = grouped_rows(rows)
    for (mode, seq_len), group in sorted(groups.items()):
        total_ops = sum(row.ops for row in group)
        total_bytes = sum(row.bytes_moved for row in group)
        ai = total_ops / total_bytes
        bound = "compute" if ai >= hardware.ridge_ai else "memory"
        latency_us = sum(row.bound_s for row in group) * 1e6
        lines.append(f"| {mode} | {seq_len} | {ai:.4g} | {bound} | {latency_us:.3f} |")

    lines.extend(
        [
            "",
            "## Sanity check (decode vs prefill on QK_T / PV)",
            "",
            "Decode `QK_T`/`PV` AI should be below prefill and below the "
            "ridge, hence memory-bound.",
            "",
        ]
    )
    lookup = {(row.mode, row.seq_len, row.gemm.name): row for row in rows}
    seq_lens = sorted({row.seq_len for row in rows})
    for seq_len in seq_lens:
        for name in ("QK_T", "PV"):
            prefill = lookup[("prefill", seq_len, name)]
            decode = lookup[("decode", seq_len, name)]
            lines.append(
                f"- seq={seq_len}: {name} prefill AI={prefill.ai:.4g} "
                f"({prefill.bound}) vs decode AI={decode.ai:.4g} "
                f"({decode.bound}); ratio={prefill.ai / decode.ai:.2f}x"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_expected_trend(rows: Sequence[RooflineRow]) -> None:
    """若 P3 Roofline 核心结论未复现则显式失败。"""
    lookup = {(row.mode, row.seq_len, row.gemm.name): row for row in rows}
    for seq_len in sorted({row.seq_len for row in rows}):
        for name in ("QK_T", "PV"):
            prefill = lookup[("prefill", seq_len, name)]
            decode = lookup[("decode", seq_len, name)]
            if decode.bound != "memory":
                raise AssertionError(f"expected decode {name} at seq={seq_len} to be memory-bound")
            if decode.ai >= prefill.ai:
                raise AssertionError(f"expected decode {name} AI to be below prefill AI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
    )
    parser.add_argument("--peak-tops", type=float, default=128.0)
    parser.add_argument("--bandwidth-tbps", type=float, default=1.0)
    parser.add_argument("--sram-mib", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hardware = HardwareConfig(
        peak_ops_per_s=args.peak_tops * 1e12,
        memory_bytes_per_s=args.bandwidth_tbps * 1e12,
        sram_bytes=int(args.sram_mib * 1024**2),
    )
    model = ModelConfig()
    rows = evaluate(args.seq_lens, model, hardware)
    validate_expected_trend(rows)

    csv_path = args.output_dir / "roofline_table.csv"
    md_path = args.output_dir / "roofline_table.md"
    write_csv(rows, csv_path)
    write_markdown(rows, hardware, md_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote summary to {md_path}")
    print("Sanity check passed: decode QK_T/PV are memory-bound and have lower AI than prefill.")


if __name__ == "__main__":
    main()
