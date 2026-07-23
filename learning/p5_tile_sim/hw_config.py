"""P5 tile 级性能模型的硬件抽象。

默认值与 P3（``learning/p3_arch_eval``）对齐：32×32 PE 阵列 @ 1 GHz、
16 MiB 片上 SRAM、1 TB/s HBM。Softmax 吞吐为可调粗粒度常数（非周期精确 RTL）。
"""

from __future__ import annotations

from dataclasses import dataclass

# 与 arch_32x32_ws.cfg 中 SCALE-Sim WS 分区一致（6 + 6 + 4 MiB）。
# 从 OFMAP 预算划出小块 stats 区，总和仍为 16 MiB。
_DEFAULT_SRAM_TOTAL = 16 * 1024 * 1024
_DEFAULT_SRAM_Q = 6 * 1024 * 1024
_DEFAULT_SRAM_KV = 6 * 1024 * 1024
_DEFAULT_SRAM_STATS = 64 * 1024
_DEFAULT_SRAM_O = 4 * 1024 * 1024 - _DEFAULT_SRAM_STATS


@dataclass(frozen=True)
class HwConfig:
    """tile 仿真器使用的参数化加速器。"""

    pe_rows: int = 32
    pe_cols: int = 32
    clock_hz: float = 1.0e9

    sram_bytes: int = _DEFAULT_SRAM_TOTAL
    # 逻辑 buffer 预算（供参考，也可做分 buffer 检查）。
    sram_q_bytes: int = _DEFAULT_SRAM_Q
    sram_kv_bytes: int = _DEFAULT_SRAM_KV
    sram_o_bytes: int = _DEFAULT_SRAM_O
    # 在线 softmax 统计 m/ℓ；从上方 4 MiB OFMAP 预算中划出。
    sram_stats_bytes: int = _DEFAULT_SRAM_STATS

    # 系统 HBM 峰值（同 P3 Roofline）；非 SCALE-Sim 的 word Bandwidth。
    dram_bandwidth_bytes_per_s: float = 1.0e12

    # 每周期处理的 score 数（exp / rescale 路径）。默认约一行 PE。
    softmax_elems_per_cycle: float = 32.0

    def __post_init__(self) -> None:
        if self.pe_rows <= 0 or self.pe_cols <= 0:
            raise ValueError("pe_rows and pe_cols must be positive")
        if self.clock_hz <= 0:
            raise ValueError("clock_hz must be positive")
        if self.sram_bytes <= 0:
            raise ValueError("sram_bytes must be positive")
        parts = self.sram_q_bytes + self.sram_kv_bytes + self.sram_o_bytes + self.sram_stats_bytes
        if parts > self.sram_bytes:
            raise ValueError(
                f"sum of logical SRAM partitions ({parts}) exceeds sram_bytes ({self.sram_bytes})"
            )
        if self.dram_bandwidth_bytes_per_s <= 0:
            raise ValueError("dram_bandwidth_bytes_per_s must be positive")
        if self.softmax_elems_per_cycle <= 0:
            raise ValueError("softmax_elems_per_cycle must be positive")

    @property
    def macs_per_cycle(self) -> int:
        """PE 阵列 INT8 MAC 吞吐（每 PE 每周期一个 MAC）。"""
        return self.pe_rows * self.pe_cols

    @property
    def dram_bytes_per_cycle(self) -> float:
        return self.dram_bandwidth_bytes_per_s / self.clock_hz

    @property
    def array_peak_ops_per_s(self) -> float:
        """阵列峰值：每 MAC 计 2 ops（mul+add），与 P3 笔记一致。"""
        return self.macs_per_cycle * 2 * self.clock_hz

    def dma_cycles(self, nbytes: float) -> float:
        """在 modeled DRAM 接口上搬运 ``nbytes`` 所需的周期数。"""
        if nbytes <= 0:
            return 0.0
        return nbytes / self.dram_bytes_per_cycle


def default_hw_config() -> HwConfig:
    """与 P3 对齐的默认配置，用于交叉校验 SCALE-Sim 趋势。"""
    return HwConfig()
