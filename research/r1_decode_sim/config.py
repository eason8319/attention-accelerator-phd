"""解码模拟器的显式假设，仅建模 attention 周期。"""

from __future__ import annotations

import math
from dataclasses import dataclass


def positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Hardware:
    pe_rows: int = 32
    pe_cols: int = 32
    clock_hz: float = 1e9
    sram_bytes: int = 16 * 1024**2
    hbm_bytes_per_second: float = 1e12
    macs_per_pe_per_cycle: float = 1.0
    softmax_scores_per_cycle: float = 32.0
    dequant_elements_per_cycle: float = 256.0
    rotation_macs_per_cycle: float = 256.0
    dma_setup_cycles: float = 0.0

    def __post_init__(self) -> None:
        for name in ("pe_rows", "pe_cols", "sram_bytes"):
            positive_int(name, getattr(self, name))
        for name in (
            "clock_hz",
            "hbm_bytes_per_second",
            "macs_per_pe_per_cycle",
            "softmax_scores_per_cycle",
            "dequant_elements_per_cycle",
            "rotation_macs_per_cycle",
            "dma_setup_cycles",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise ValueError(f"invalid hardware parameter {name}={value!r}")
            minimum_ok = value >= 0 if name == "dma_setup_cycles" else value > 0
            if not math.isfinite(value) or not minimum_ok:
                raise ValueError(f"invalid hardware parameter {name}={value!r}")

    @property
    def peak_macs_per_cycle(self) -> float:
        return self.pe_rows * self.pe_cols * self.macs_per_pe_per_cycle

    def dma_cycles(self, nbytes: int) -> float:
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("DMA byte count must be a nonnegative integer")
        return (
            self.dma_setup_cycles + nbytes * self.clock_hz / self.hbm_bytes_per_second
            if nbytes
            else 0.0
        )


@dataclass(frozen=True)
class Mapping:
    query_tile: int = 32
    kv_tile: int = 512
    buffering: str = "auto"
    head_mapping: str = "query_rows"

    def __post_init__(self) -> None:
        positive_int("query_tile", self.query_tile)
        positive_int("kv_tile", self.kv_tile)
        if self.buffering not in ("auto", "single", "double"):
            raise ValueError("buffering must be auto, single or double")
        if self.head_mapping not in ("query_rows", "head_folded"):
            raise ValueError("head_mapping must be query_rows or head_folded")


@dataclass(frozen=True)
class AttentionWorkload:
    """单批次负载；prefill 使用稠密矩形，不跳过因果掩码之外的分块。

    query_heads 与输入中的 KV 头数分别定义。Q/O 和解码后的 KV 使用 FP16；
    分数、O 累加器及 softmax 统计量使用 FP32。这些物理 SRAM 位宽不采用
    精度与流量输入中包含元数据的有效比特数。
    """

    mode: str = "decode"
    query_heads: int = 32

    def __post_init__(self) -> None:
        if self.mode not in ("decode", "prefill"):
            raise ValueError("mode must be decode or prefill")
        positive_int("query_heads", self.query_heads)
