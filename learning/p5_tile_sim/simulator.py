"""Tile 级 FlashAttention 性能模型（P5）。

建模 SRAM 合法性、DMA load / compute / DMA store 事件，以及
:math:`B_r \\times B_c` tiling 下的串行与 double-buffer 调度。

数据流（阅读笔记 §4）：外层 Q tile、内层 KV tile；
对单个 Q tile 的 KV 块，running :math:`(m,\\ell,O)` 更新串行进行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hw_config import HwConfig, default_hw_config
from workload import Workload


@dataclass(frozen=True)
class TileConfig:
    """FlashAttention 在 Q（:math:`B_r`）与 KV（:math:`B_c`）方向的 tile 尺寸。"""

    br: int
    bc: int

    def __post_init__(self) -> None:
        if self.br <= 0 or self.bc <= 0:
            raise ValueError("br and bc must be positive")


@dataclass(frozen=True)
class SimResult:
    """单个 (hw, workload, tile) 三元组的仿真结果。"""

    feasible: bool
    br: int
    bc: int
    footprint_bytes: float
    double_buffer_ok: bool
    double_buffer_enabled: bool

    latency_cycles: float
    latency_serial_cycles: float
    latency_db_cycles: float

    dram_traffic_bytes: float
    dma_load_bytes: float
    dma_store_bytes: float

    compute_cycles_total: float
    dma_load_cycles_total: float
    dma_store_cycles_total: float

    mac_work: float
    pe_util: float
    reason: str = ""

    @property
    def dma_cycle_fraction(self) -> float:
        """串行关键路径中归因于 DMA（load+store）的比例。"""
        dma = self.dma_load_cycles_total + self.dma_store_cycles_total
        denom = dma + self.compute_cycles_total
        return dma / denom if denom > 0 else 0.0


def _ceil_div(num: float, den: float) -> float:
    if den <= 0:
        raise ValueError("denominator must be positive")
    if num <= 0:
        return 0.0
    return math.ceil(num / den)


def _extent(total: int, tile: int, index: int) -> int:
    start = index * tile
    return max(0, min(tile, total - start))


def _spatial_macs_per_cycle(hw: HwConfig, br: int, bc: int) -> float:
    """将 Br×Bc tile 映射到 PE 阵列后的有效 MAC/周期。

    粗粒度 WS 类规则：Q tile 方向最多 ``pe_rows``、KV tile 方向最多 ``pe_cols`` 活跃。
    Prefill tile ≥ 阵列时填满 mesh；decode ``Br=1`` 仅一行 PE 活跃（约 1/pe_rows 利用率）。
    """
    used_r = min(br, hw.pe_rows)
    used_c = min(bc, hw.pe_cols)
    return float(max(1, used_r * used_c))


def _compute_tile_cycles(hw: HwConfig, br: int, bc: int, head_dim: int) -> float:
    """单个 FA tile 的 QKᵀ + softmax + PV（粗粒度解析模型）。"""
    macs = _spatial_macs_per_cycle(hw, br, bc)
    qk = _ceil_div(br * bc * head_dim, macs)
    sm = _ceil_div(br * bc, hw.softmax_elems_per_cycle)
    pv = _ceil_div(br * bc * head_dim, macs)
    return qk + sm + pv


def _schedule_serial(events: list[tuple[float, float, float]]) -> float:
    return sum(load + comp + store for load, comp, store in events)


def _schedule_double_buffer(events: list[tuple[float, float, float]]) -> float:
    """tile i+1 的 load 与 tile i 的 compute 重叠（store 在 compute 之后）。"""
    if not events:
        return 0.0
    t = events[0][0]
    for i, (_load, comp, store) in enumerate(events):
        next_load = events[i + 1][0] if i + 1 < len(events) else 0.0
        t += max(comp, next_load)
        t += store
    return t


def simulate(
    workload: Workload,
    tile: TileConfig,
    hw: HwConfig | None = None,
    *,
    use_double_buffer: bool = True,
) -> SimResult:
    """对候选 tile 配置仿真单层 attention。

    head 与 batch 按串行建模（保守延迟）。流量与 MAC 工作量按 ``batch * heads`` 缩放。
    """
    hw = default_hw_config() if hw is None else hw

    br = tile.br
    bc = tile.bc
    # Decode 仅一行 query；将 Br 钳为 1。
    if workload.mode == "decode":
        br = 1

    footprint = workload.tile_footprint_bytes(br, bc)
    db_ok = (2.0 * footprint) <= hw.sram_bytes
    feasible = footprint <= hw.sram_bytes

    if not feasible:
        return SimResult(
            feasible=False,
            br=br,
            bc=bc,
            footprint_bytes=footprint,
            double_buffer_ok=False,
            double_buffer_enabled=False,
            latency_cycles=math.inf,
            latency_serial_cycles=math.inf,
            latency_db_cycles=math.inf,
            dram_traffic_bytes=0.0,
            dma_load_bytes=0.0,
            dma_store_bytes=0.0,
            compute_cycles_total=0.0,
            dma_load_cycles_total=0.0,
            dma_store_cycles_total=0.0,
            mac_work=0.0,
            pe_util=0.0,
            reason="footprint exceeds SRAM",
        )

    n_q = workload.n_q
    n_kv = workload.n_kv
    d = workload.head_dim
    b = workload.bytes
    n_q_tiles = int(math.ceil(n_q / br))
    n_kv_tiles = int(math.ceil(n_kv / bc))

    # 单 head 事件列表：(load_cycles, compute_cycles, store_cycles)。
    events: list[tuple[float, float, float]] = []
    load_bytes = 0.0
    store_bytes = 0.0
    compute_cycles = 0.0
    load_cycles = 0.0
    store_cycles = 0.0

    for qi in range(n_q_tiles):
        br_eff = _extent(n_q, br, qi)
        if br_eff <= 0:
            continue
        for kj in range(n_kv_tiles):
            bc_eff = _extent(n_kv, bc, kj)
            if bc_eff <= 0:
                continue

            chunk = 0.0
            if kj == 0:
                # 开始扫 KV 时，该 Q tile 只 load 一次。
                chunk += br_eff * d * b.q
            chunk += bc_eff * d * (b.k + b.v)

            store = 0.0
            if kj == n_kv_tiles - 1:
                store = br_eff * d * b.o

            load_c = hw.dma_cycles(chunk)
            store_c = hw.dma_cycles(store)
            comp_c = _compute_tile_cycles(hw, br_eff, bc_eff, d)

            events.append((load_c, comp_c, store_c))
            load_bytes += chunk
            store_bytes += store
            compute_cycles += comp_c
            load_cycles += load_c
            store_cycles += store_c

    serial = _schedule_serial(events)
    db_lat = _schedule_double_buffer(events) if db_ok else serial
    enable_db = bool(use_double_buffer and db_ok)
    latency_one_head = db_lat if enable_db else serial

    scale = workload.batch * workload.heads
    latency_serial = serial * scale
    latency_db = db_lat * scale
    latency = latency_one_head * scale

    # QKᵀ 与 PV：每个 head 各贡献 n_q * n_kv * d 次 MAC。
    mac_work = scale * (2.0 * n_q * n_kv * d)
    pe_util = mac_work / (latency * hw.macs_per_cycle) if latency > 0 else 0.0

    return SimResult(
        feasible=True,
        br=br,
        bc=bc,
        footprint_bytes=footprint,
        double_buffer_ok=db_ok,
        double_buffer_enabled=enable_db,
        latency_cycles=latency,
        latency_serial_cycles=latency_serial,
        latency_db_cycles=latency_db,
        dram_traffic_bytes=(load_bytes + store_bytes) * scale,
        dma_load_bytes=load_bytes * scale,
        dma_store_bytes=store_bytes * scale,
        compute_cycles_total=compute_cycles * scale,
        dma_load_cycles_total=load_cycles * scale,
        dma_store_cycles_total=store_cycles * scale,
        mac_work=mac_work,
        pe_util=pe_util,
        reason="ok" if enable_db else ("serial (SRAM too tight for DB)" if feasible else ""),
    )
