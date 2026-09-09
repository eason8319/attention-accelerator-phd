"""解码模拟器的分块级 attention 周期模型，不导入模型运行时或学习实验。

各层串行执行。GQA 查询头共享每次载入的 KV 分块；默认 PE 映射按 P5
的约定串行执行查询头。可选 head_folded 映射假设并行头通道具有独立操作数，
不同于常规的单个 GEMM。DMA 与串行计算流水线只能在合法 SRAM 缓冲条件下
重叠；不估计能耗或完整解码器时延。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import AttentionWorkload, Hardware, Mapping, positive_int
from .inputs import FORMATS, LAYOUTS, PROTOCOL_IDS, ByteBreakdown, PressureInput, StepInput

MODEL_VERSION = "r1-attention-tile-v1"
LIMITATIONS = (
    "attention_only_excludes_projections_mlp_weights_cache_append_and_sampling",
    "nominal_packed_KV_bytes_not_current_torch_storage_or_measured_HBM_time",
    "metadata_and_KIVI_residual_work_apportioned_across_tiles_not_physical_page_trace",
    "no_bank_conflicts_array_skew_interconnect_or_page_pointer_stalls",
    "auxiliary_throughputs_and_FP16_MAC_rate_are_uncalibrated_assumptions",
    "prefill_is_dense_rectangle_no_causal_tile_skipping_no_measured_prefill_traffic",
)


@dataclass(frozen=True)
class TileEvent:
    index: int
    kv_tokens: int
    read_bytes: int
    dma_start: float
    dma_end: float
    compute_start: float
    compute_end: float


def schedule_tiles(
    loads: tuple[float, ...], computes: tuple[float, ...], slots: int
) -> tuple[tuple[float, float, float, float], ...]:
    """使用一个 DMA 引擎和一个计算引擎，复用缓冲前等待释放。

    返回时间不含 Q 载入和最终 O 写回。双槽缓冲中，分块 i 等待分块 i-2
    计算完成后复用；单槽缓冲则等待分块 i-1 完成。
    """
    if type(slots) is not int or slots not in (1, 2) or len(loads) != len(computes):
        raise ValueError("schedule requires matching events and one or two buffers")
    for value in loads + computes:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("event duration must be finite and nonnegative")
    events = []
    dma_end = compute_end = 0.0
    for i, (load, compute) in enumerate(zip(loads, computes, strict=True)):
        release = events[i - slots][3] if i >= slots else 0.0
        start = max(dma_end, release)
        dma_end = start + load
        compute_start = max(dma_end, compute_end)
        compute_end = compute_start + compute
        events.append((start, dma_end, compute_start, compute_end))
    return tuple(events)


def _portion(total: int, start: int, end: int, length: int) -> int:
    """将总量划分为整数分量；保证字节守恒，不表示物理打包轨迹。"""
    return total * end // length - total * start // length


def _validate(step: StepInput, workload: AttentionWorkload) -> None:
    for name in ("num_layers", "num_kv_heads", "head_dim"):
        positive_int(name, getattr(step.geometry, name))
    positive_int("n", step.n)
    if workload.query_heads % step.geometry.num_kv_heads:
        raise ValueError("query_heads must be a multiple of KV heads for GQA")
    if step.kv_format not in FORMATS:
        raise ValueError("unsupported KV format")
    if step.protocol_id != PROTOCOL_IDS[FORMATS.index(step.kv_format)]:
        raise ValueError("protocol_id disagrees with KV format")
    if step.layout not in LAYOUTS:
        raise ValueError("unsupported layout")
    if step.kv_format == "int4_bdr" and step.geometry.head_dim % 32:
        raise ValueError("BDR model requires head_dim divisible by block size 32")
    for value in asdict(step.all_layers).values():
        if type(value) is not int or value < 0 or value % step.geometry.num_layers:
            raise ValueError("invalid whole-model/per-layer byte count")
    if (step.layout == "contiguous" and step.all_layers.page != 0) or (
        step.layout == "paged" and step.all_layers.page <= 0
    ):
        raise ValueError("page metadata disagrees with layout")
    if step.bytes_per_token <= 0 or not math.isfinite(step.b_eff):
        raise ValueError("invalid KV traffic")
    expected = step.per_layer.total * 8 / (step.n * step.geometry.n_elem)
    if not math.isclose(step.b_eff, expected, rel_tol=1e-12):
        raise ValueError("effective bits disagree with traffic")


def _quantized_elements(step: StepInput) -> int:
    if step.kv_format == "fp16":
        return 0
    if step.kv_format.startswith("kivi"):
        # 残差窗固定为 R=128；K 与 V 的 FP16 残差长度分别计算。
        residual_tokens = step.n % 128 + min(step.n, 128)
        return (2 * step.n - residual_tokens) * step.geometry.num_kv_heads * step.geometry.head_dim
    return step.n * step.geometry.n_elem


def _tile_compute(
    br: int,
    bc: int,
    step: StepInput,
    workload: AttentionWorkload,
    mapping: Mapping,
    hw: Hardware,
    quantized: int,
) -> tuple[float, float, float, float, float]:
    heads = workload.query_heads
    rows, serial_heads = (br, heads) if mapping.head_mapping == "query_rows" else (br * heads, 1)
    row_waves = math.ceil(rows / hw.pe_rows)
    d = step.geometry.head_dim
    # 分别计算 QK^T 和 PV 的矩阵形状，包含边界分块的占用。
    qk = serial_heads * row_waves * math.ceil(bc / hw.pe_cols) * d / hw.macs_per_pe_per_cycle
    pv = serial_heads * row_waves * math.ceil(d / hw.pe_cols) * bc / hw.macs_per_pe_per_cycle
    softmax = math.ceil(br * bc * heads / hw.softmax_scores_per_cycle)
    dequant = math.ceil(quantized / hw.dequant_elements_per_cycle)
    # 按结构化稠密 32×32 分块计入逆旋转，不假设旋转零开销。
    rotation_macs = bc * step.geometry.n_elem * 32 if step.kv_format == "int4_bdr" else 0
    rotation = math.ceil(rotation_macs / hw.rotation_macs_per_cycle)
    return qk, pv, float(softmax), float(dequant), float(rotation)


def _footprint(br: int, bc: int, packed: int, step: StepInput, heads: int) -> dict[str, int]:
    d = step.geometry.head_dim
    return {
        "q_fp16": br * heads * d * 2,
        "output_accumulator_fp32": br * heads * d * 4,
        "softmax_stats_fp32": br * heads * 2 * 4,
        "score_tile_fp32": br * heads * bc * 4,
        "packed_kv_per_buffer": packed,
        "decoded_kv_fp16_per_buffer": bc * step.geometry.n_elem * 2,
        "rotation_scratch_fp32": bc * step.geometry.n_elem * 4
        if step.kv_format == "int4_bdr"
        else 0,
        "rotation_matrix_fp32": d * d * 4 if step.kv_format == "int4_bdr" else 0,
    }


def _buffer_size(parts: dict[str, int], slots: int) -> int:
    return sum(
        value * (slots if name.endswith("_per_buffer") else 1) for name, value in parts.items()
    )


@dataclass(frozen=True)
class SimulationResult:
    model_version: str
    mode: str
    model_id: str
    n: int
    protocol_id: str
    layout: str
    num_layers: int
    query_heads: int
    kv_heads: int
    mapping: Mapping
    feasible: bool
    reason: str
    buffer_slots: int
    effective_query_tile: int
    kv_tiles_per_query_tile: int
    query_tiles: int
    sram_single_bytes: int
    sram_double_bytes: int
    sram_breakdown: dict[str, int]
    kv_read: ByteBreakdown
    q_read_bytes: int
    output_write_bytes: int
    attention_macs: int
    latency_cycles: float | None
    serial_cycles: float | None
    double_buffer_cycles: float | None
    component_cycles: dict[str, float]
    clock_hz: float
    peak_macs_per_cycle: float
    hbm_bytes_per_second: float
    first_query_tile_trace: tuple[TileEvent, ...]
    input_source: dict
    limitations: tuple[str, ...] = LIMITATIONS

    @property
    def hbm_bytes(self) -> int:
        return self.kv_read.total + self.q_read_bytes + self.output_write_bytes

    @property
    def attention_latency_seconds(self) -> float | None:
        return self.latency_cycles / self.clock_hz if self.latency_cycles is not None else None

    @property
    def pe_utilization(self) -> float | None:
        return (
            self.attention_macs / (self.latency_cycles * self.peak_macs_per_cycle)
            if self.latency_cycles is not None
            else None
        )

    @property
    def hbm_utilization(self) -> float | None:
        seconds = self.attention_latency_seconds
        return self.hbm_bytes / (seconds * self.hbm_bytes_per_second) if seconds else None

    @property
    def arithmetic_intensity_ops_per_byte(self) -> float:
        """仅统计 attention 矩阵乘运算，每 MAC 计 2 次运算，不含辅助操作。"""
        return 2 * self.attention_macs / self.hbm_bytes

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "hbm_bytes": self.hbm_bytes,
            "attention_latency_seconds": self.attention_latency_seconds,
            "pe_utilization": self.pe_utilization,
            "hbm_utilization": self.hbm_utilization,
            "arithmetic_intensity_ops_per_byte": self.arithmetic_intensity_ops_per_byte,
        }


def simulate(
    step: StepInput,
    mapping: Mapping | None = None,
    hw: Hardware | None = None,
    workload: AttentionWorkload | None = None,
    *,
    trace: bool = False,
) -> SimulationResult:
    """使用一条已校验的 KV 读取输入模拟 attention。

    decode 恰好读取一次输入中的 KV 总量。prefill 是派生的稠密负载，
    每个 Q 分块都重读 KV，并非实测的 prefill 缓存路径结果。完整 Q 分块
    具有相同形状，计数时不展开平方规模的事件列表。Q 分块和层之间不重叠。
    """
    mapping, hw, workload = mapping or Mapping(), hw or Hardware(), workload or AttentionWorkload()
    _validate(step, workload)
    n, layers = step.n, step.geometry.num_layers
    nq = 1 if workload.mode == "decode" else n
    br = min(nq, mapping.query_tile)
    extents = [(start, min(start + mapping.kv_tile, n)) for start in range(0, n, mapping.kv_tile)]
    parts = step.per_layer
    tile_bytes = tuple(
        sum(_portion(v, a, b, n) for v in asdict(parts).values()) for a, b in extents
    )
    quantized = _quantized_elements(step)
    max_bc = min(n, mapping.kv_tile)
    memory = _footprint(br, max_bc, max(tile_bytes), step, workload.query_heads)
    single_size, double_size = _buffer_size(memory, 1), _buffer_size(memory, 2)
    slots = (
        (2 if double_size <= hw.sram_bytes else 1)
        if mapping.buffering == "auto"
        else (2 if mapping.buffering == "double" else 1)
    )
    feasible = (double_size if slots == 2 else single_size) <= hw.sram_bytes
    query_tiles = math.ceil(nq / br)
    kv = ByteBreakdown(
        *(getattr(step.all_layers, p) * query_tiles for p in ("payload", "scale", "zp", "page"))
    )
    q_bytes = nq * workload.query_heads * step.geometry.head_dim * 2 * layers
    macs = 2 * nq * n * workload.query_heads * step.geometry.head_dim * layers
    components = {
        name: 0.0
        for name in ("q_dma", "kv_dma", "output_dma", "qk", "pv", "softmax", "dequant", "rotation")
    }
    serial_total = double_total = 0.0
    trace_events = ()
    if feasible:
        full_count, tail = divmod(nq, br)
        query_groups = [(br, full_count)] + ([(tail, 1)] if tail else [])
        for query_rows, repeats in query_groups:
            computations = tuple(
                _tile_compute(
                    query_rows, b - a, step, workload, mapping, hw, _portion(quantized, a, b, n)
                )
                for a, b in extents
            )
            loads = tuple(hw.dma_cycles(value) for value in tile_bytes)
            computes = tuple(sum(values) for values in computations)
            serial_events = schedule_tiles(loads, computes, 1)
            double_events = (
                schedule_tiles(loads, computes, 2) if double_size <= hw.sram_bytes else ()
            )
            io_bytes = query_rows * workload.query_heads * step.geometry.head_dim * 2
            q_dma = output_dma = hw.dma_cycles(io_bytes)
            serial_total += (q_dma + serial_events[-1][3] + output_dma) * repeats * layers
            if double_events:
                double_total += (q_dma + double_events[-1][3] + output_dma) * repeats * layers
            components["q_dma"] += q_dma * repeats * layers
            components["output_dma"] += output_dma * repeats * layers
            components["kv_dma"] += sum(loads) * repeats * layers
            for i, name in enumerate(("qk", "pv", "softmax", "dequant", "rotation")):
                components[name] += sum(values[i] for values in computations) * repeats * layers
            if trace and not trace_events:
                events = double_events if slots == 2 else serial_events
                trace_events = tuple(
                    TileEvent(
                        i, extents[i][1] - extents[i][0], tile_bytes[i], *(t + q_dma for t in times)
                    )
                    for i, times in enumerate(events)
                )
    latency = (double_total if slots == 2 else serial_total) if feasible else None
    return SimulationResult(
        MODEL_VERSION,
        workload.mode,
        step.model_id,
        n,
        step.protocol_id,
        step.layout,
        layers,
        workload.query_heads,
        step.geometry.num_kv_heads,
        mapping,
        feasible,
        "ok" if feasible else "requested buffer footprint exceeds SRAM",
        slots,
        br,
        len(extents),
        query_tiles,
        single_size,
        double_size,
        memory,
        kv,
        q_bytes,
        q_bytes,
        macs,
        latency,
        serial_total if feasible else None,
        double_total if feasible and double_size <= hw.sram_bytes else None,
        components,
        hw.clock_hz,
        hw.peak_macs_per_cycle,
        hw.hbm_bytes_per_second,
        trace_events,
        {**asdict(step.source), "pointer": step.source_pointer},
    )


def simulate_pressure(
    pressure: PressureInput,
    mapping: Mapping | None = None,
    hw: Hardware | None = None,
    workload: AttentionWorkload | None = None,
) -> dict:
    """给出精确的总流量、峰值性能下界和末步模拟结果。

    总字节数不能确定逐步重叠或反量化停顿，因此不从平均流量或末步结果
    推算平均生成时延。
    """
    hw, workload = hw or Hardware(), workload or AttentionWorkload()
    if workload.mode != "decode":
        raise ValueError("pressure aggregates are decode-only")
    for name in ("l_in", "l_out", "total_kv_read"):
        positive_int(name, getattr(pressure, name))
    if pressure.last_step.n != pressure.l_in + pressure.l_out - 1:
        raise ValueError("pressure last-step N mismatch")
    last = simulate(pressure.last_step, mapping, hw, workload)
    layers = pressure.last_step.geometry.num_layers
    if (
        pressure.total_kv_read % layers
        or pressure.total_kv_read < pressure.last_step.bytes_per_token
    ):
        raise ValueError("invalid pressure aggregate bytes")
    sum_n = pressure.l_out * (2 * pressure.l_in + pressure.l_out - 1) // 2
    macs = 2 * sum_n * workload.query_heads * pressure.last_step.geometry.head_dim * layers
    q_o = pressure.l_out * workload.query_heads * pressure.last_step.geometry.head_dim * 4 * layers
    hbm_cycles = (pressure.total_kv_read + q_o) * hw.clock_hz / hw.hbm_bytes_per_second
    lower = max(hbm_cycles, macs / hw.peak_macs_per_cycle)
    return {
        "l_in": pressure.l_in,
        "l_out": pressure.l_out,
        "protocol_id": pressure.last_step.protocol_id,
        "layout": pressure.last_step.layout,
        "total_kv_read": pressure.total_kv_read,
        "mean_kv_bytes_per_token": pressure.mean_bytes_per_token,
        "last_kv_bytes_per_token": pressure.last_step.bytes_per_token,
        "total_attention_macs": macs,
        "total_q_and_output_bytes": q_o,
        "kv_only_hbm_lower_bound_seconds": pressure.total_kv_read / hw.hbm_bytes_per_second,
        "mean_attention_lower_bound_cycles": lower / pressure.l_out,
        "mean_attention_latency_cycles": None,
        "generation_latency_status": "requires_per_step_traffic_trace",
        "last_step": last.as_dict(),
    }
