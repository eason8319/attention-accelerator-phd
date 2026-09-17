"""公平映射下的占用、动作计数、周期调度与动态能耗模型。

不物化完整高精度 KV tile。1 GHz 与 Accelergy 单位能量是声明模型，不是综合频率
或硅片测量。C3 按混合基线计入写侧旋转、读侧 K 逆旋转、输出逆旋转和短块补零。
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from research.r2_streaming_attention.access_events import (
    expected_page_counts,
    kivi_pool_tokens,
)
from research.r2_streaming_attention.physical_pack import align_up

ROOT = Path(__file__).resolve().parents[2]
MAPPING_COST_PATH = (
    ROOT / "research" / "r2_streaming_attention" / "experiments" / "configs" / "mapping_cost.json"
)
PROTOCOL_PATH = (
    ROOT
    / "research"
    / "r2_streaming_attention"
    / "experiments"
    / "configs"
    / "evaluation_protocol.json"
)

GROUP = 32
BDR_BLOCK = 32


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 须为正整数，得到 {value!r}")
    return value


def _nonneg_int(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} 须为非负整数，得到 {value!r}")
    return value


def ceil_div(n: int, d: int) -> int:
    """正除法向上取整；n=0 时为 0。"""
    if d <= 0:
        raise ValueError(f"除数须为正，得到 {d}")
    if n < 0:
        raise ValueError(f"被除数不能为负，得到 {n}")
    if n == 0:
        return 0
    return (n + d - 1) // d


@dataclass(frozen=True)
class Geometry:
    """单层注意力几何；合成 GQA 须标明来源。"""

    name: str
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    gqa_ratio: int
    source: str

    @property
    def rotation_macs_per_vector(self) -> int:
        """BDR 逆旋转一个 head 向量的 MAC：每 32 维一块稠密 :math:`32\\times 32`。"""
        if self.head_dim % BDR_BLOCK:
            raise ValueError("C3 旋转要求 head_dim 能被 32 整除")
        return self.head_dim * BDR_BLOCK


@dataclass(frozen=True)
class Hardware:
    """步骤 6 冻结的阵列、SRAM 端口、DMA 与单位能量。"""

    mac_budget: int
    pe_rows: int
    pe_cols: int
    frequency_hz: float
    bandwidth_tb_s: float
    sram_mib: int
    dequant_elements_cycle: int
    dma_engines: int
    dma_queue_depth: int
    dma_setup_cycles: int
    dma_align_bytes: int
    sram_banks: int
    sram_rw_ports: int
    sram_width_bytes: int
    sram_latency_cycles: int
    operand_tokens: int
    operand_ports: int
    hbm2_pJ_per_bit: float
    mac_fp16_pJ: float
    sram_pJ_per_bit: float
    dequant_pJ_per_element: float
    softmax_pJ_per_score: float
    broadcast_pJ_per_byte: float
    quant_pJ_per_element: float
    pack_pJ_per_byte: float
    control_pJ_per_event: float
    convert_pJ_per_element: float
    execution: dict

    @property
    def sram_bytes(self) -> int:
        return self.sram_mib * 1024 * 1024

    @property
    def bytes_per_cycle(self) -> float:
        return self.bandwidth_tb_s * 1e12 / self.frequency_hz

    def dma_cycles(self, physical_bytes: int, n_transactions: int) -> float:
        """单 DMA 引擎：启动开销加带宽时间；多事务串行。"""
        if physical_bytes < 0 or n_transactions < 0:
            raise ValueError("DMA 字节与事务数不能为负")
        if physical_bytes == 0:
            return 0.0
        engines = max(1, min(self.dma_engines, n_transactions or 1))
        setup = self.dma_setup_cycles * ceil_div(n_transactions, engines)
        # 一个 SRAM rw 端口专供 DMA，其余端口供计算；预取不免费占用计算端口。
        xfer = physical_bytes / min(self.bytes_per_cycle, self.sram_banks * self.sram_width_bytes)
        return setup + xfer


@dataclass(frozen=True)
class Mapping:
    """一次公平映射选择。"""

    tile: int
    buffering: str
    kv_pass: str
    kv_split: int
    gqa_mode: str
    parallel_kv_heads: int
    group_buffers: int
    merge_tokens: int
    batch: int
    batch_parallel: bool = False

    def __post_init__(self) -> None:
        _positive_int("tile", self.tile)
        _positive_int("kv_split", self.kv_split)
        _positive_int("parallel_kv_heads", self.parallel_kv_heads)
        _positive_int("group_buffers", self.group_buffers)
        _positive_int("merge_tokens", self.merge_tokens)
        _positive_int("batch", self.batch)
        if self.buffering not in {"single", "double"}:
            raise ValueError(self.buffering)
        if self.kv_pass not in {"separate", "fused_tile"}:
            raise ValueError(self.kv_pass)
        if self.gqa_mode not in {"gqa_shared", "gqa_expand"}:
            raise ValueError(self.gqa_mode)


@dataclass(frozen=True)
class Occupancy:
    """与步骤 4 占用字段对齐的解析驻留；allocated 不是分配器实测。"""

    n_tokens: int
    payload_packed: int
    residual_fp16: int
    tail_fp16: int
    scale: int
    offset: int
    pte: int
    tags: int
    hbm_payload_allocated: int
    hbm_metadata_allocated: int
    n_hbm_pages: int
    n_pte: int

    @property
    def alignment_waste(self) -> int:
        return (
            self.hbm_payload_allocated
            + self.hbm_metadata_allocated
            - self.payload_packed
            - self.scale
            - self.offset
        )

    @property
    def sram_fp16(self) -> int:
        return self.residual_fp16 + self.tail_fp16

    @property
    def logical_payload_and_meta(self) -> int:
        return self.payload_packed + self.scale + self.offset

    @property
    def hbm_allocated(self) -> int:
        return self.hbm_payload_allocated + self.hbm_metadata_allocated


@dataclass
class StepResult:
    """单层单步 attention-only 映射结果。"""

    feasible: bool
    reason: str
    geometry: str
    format_id: str
    layout: str
    n_tokens: int
    mapping: dict[str, object]
    occupancy: dict[str, int]
    actions: dict[str, int | float]
    sram_breakdown: dict[str, int]
    sram_peak_bytes: int
    component_cycles: dict[str, float]
    latency_cycles: float | None
    serial_cycles: float | None
    roofline_cycles: float
    energy_dynamic_pJ: dict[str, float]
    energy_total_dynamic_pJ: float | None
    pe_utilization: float | None
    schedule: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "attention_only",
        "reference_1ghz_not_synthesized",
        "accelergy_style_dynamic_energy",
        "leakage_not_in_headline",
        "c3_hybrid_retains_key_inverse",
    )


def load_mapping_cost(path: Path | str | None = None) -> dict:
    """读取步骤 6 共享配置。"""
    return json.loads((path or MAPPING_COST_PATH).read_text(encoding="utf-8"))


def load_protocol(path: Path | str | None = None) -> dict:
    """读取评测协议，供几何与开发轴使用。"""
    return json.loads((path or PROTOCOL_PATH).read_text(encoding="utf-8"))


def hardware_from_config(
    raw: dict,
    *,
    bandwidth_tb_s: float,
    sram_mib: int,
    dequant_elements_cycle: int,
    operand_ports: int | None = None,
    dma_setup_cycles: int | None = None,
    energy_overrides: dict[str, float] | None = None,
) -> Hardware:
    """用协议扫描点覆盖带宽/SRAM/解量化，其余保持冻结。"""
    energy = dict(raw["energy"])
    if energy_overrides:
        energy.update(energy_overrides)
    dma = raw["dma"]
    sram = raw["sram"]
    ports = raw["operand"]["ports_default"] if operand_ports is None else operand_ports
    if ports not in raw["operand"]["ports_allowed"]:
        raise ValueError(f"operand_ports={ports} 不在允许集合")
    setup = dma["setup_cycles"] if dma_setup_cycles is None else dma_setup_cycles
    if raw["pe_rows"] * raw["pe_cols"] != raw["mac_budget"]:
        raise ValueError("mac_budget 必须等于 pe_rows×pe_cols")
    return Hardware(
        mac_budget=int(raw["mac_budget"]),
        pe_rows=int(raw["pe_rows"]),
        pe_cols=int(raw["pe_cols"]),
        frequency_hz=float(raw["reference_frequency_hz"]),
        bandwidth_tb_s=float(bandwidth_tb_s),
        sram_mib=int(sram_mib),
        dequant_elements_cycle=int(dequant_elements_cycle),
        dma_engines=int(dma["engines"]),
        dma_queue_depth=int(dma["queue_depth"]),
        dma_setup_cycles=int(setup),
        dma_align_bytes=int(dma["align_bytes"]),
        sram_banks=int(sram["banks"]),
        sram_rw_ports=int(sram["rw_ports"]),
        sram_width_bytes=int(sram["width_bits"]) // 8,
        sram_latency_cycles=int(sram["latency_cycles"]),
        operand_tokens=int(raw["operand"]["tokens"]),
        operand_ports=int(ports),
        hbm2_pJ_per_bit=float(energy["hbm2_pJ_per_bit"]),
        mac_fp16_pJ=float(energy["mac_fp16_pJ"]),
        sram_pJ_per_bit=float(energy["sram_pJ_per_bit"]),
        dequant_pJ_per_element=float(energy["dequant_pJ_per_element"]),
        softmax_pJ_per_score=float(energy["softmax_pJ_per_score"]),
        broadcast_pJ_per_byte=float(energy["broadcast_pJ_per_byte"]),
        quant_pJ_per_element=float(energy["quant_pJ_per_element"]),
        pack_pJ_per_byte=float(energy["pack_pJ_per_byte"]),
        control_pJ_per_event=float(energy["control_pJ_per_event"]),
        convert_pJ_per_element=float(energy["convert_pJ_per_element"]),
        execution=dict(raw["execution"]),
    )


def geometry_spec(name: str, *, gqa_ratio: int | None = None) -> Geometry:
    """协议几何；合成共享比保持 8 个 KV head。"""
    proto = load_protocol()
    models = {row["id"]: row["geometry"] for row in proto["models"]["formal"]}
    if name == "llama":
        g = models["meta-llama/Llama-3.1-8B-Instruct"]
        ratio = gqa_ratio or int(g["gqa_ratio"])
        if ratio != int(g["gqa_ratio"]):
            kv = int(g["kv_heads"])
            return Geometry(
                name=f"synthetic_{ratio * kv}Q_{kv}KV_d{g['head_dim']}",
                layers=int(g["layers"]),
                q_heads=ratio * kv,
                kv_heads=kv,
                head_dim=int(g["head_dim"]),
                gqa_ratio=ratio,
                source="synthetic_ratio_axis_keeps_8KV",
            )
        return Geometry(
            "llama",
            int(g["layers"]),
            int(g["q_heads"]),
            int(g["kv_heads"]),
            int(g["head_dim"]),
            int(g["gqa_ratio"]),
            "protocol Llama-3.1-8B-Instruct",
        )
    if name == "qwen":
        g = models["Qwen/Qwen2.5-7B-Instruct"]
        if gqa_ratio not in {None, int(g["gqa_ratio"])}:
            raise ValueError("qwen 几何轴使用原生 GQA=7；合成比请用 llama 轴")
        return Geometry(
            "qwen",
            int(g["layers"]),
            int(g["q_heads"]),
            int(g["kv_heads"]),
            int(g["head_dim"]),
            int(g["gqa_ratio"]),
            "protocol Qwen2.5-7B-Instruct geometry; not a full model run",
        )
    raise ValueError(name)


def _fp16(tokens: int, kv_heads: int, dim: int) -> int:
    return _nonneg_int("tokens", tokens) * kv_heads * dim * 2


def _int4_payload(tokens: int, kv_heads: int, dim: int) -> int:
    return _nonneg_int("tokens", tokens) * kv_heads * dim // 2


def _c2_scale(tokens: int, kv_heads: int, dim: int) -> int:
    return tokens * kv_heads * (dim // GROUP) * 2


def _c5_key_meta(k_quant: int, kv_heads: int, dim: int) -> int:
    return (k_quant // GROUP) * kv_heads * dim * 4


def _c5_value_meta(v_quant: int, kv_heads: int, dim: int) -> int:
    return v_quant * kv_heads * (dim // GROUP) * 4


def _uniform_committed(n_tokens: int, layout: str, page: int) -> tuple[int, int]:
    if layout == "contiguous":
        return n_tokens, 0
    committed = (n_tokens // page) * page
    return committed, n_tokens - committed


def _page_alloc(
    n_pages: int,
    payload_per_page: int,
    meta_per_page: int,
    *,
    align: int,
    colocated: bool,
) -> tuple[int, int]:
    if n_pages == 0:
        return 0, 0
    if colocated:
        return n_pages * align_up(payload_per_page + meta_per_page, align), 0
    pay = n_pages * align_up(payload_per_page, align)
    meta = n_pages * align_up(meta_per_page, align) if meta_per_page else 0
    return pay, meta


def layout_occupancy(
    format_id: str,
    layout: str,
    n_tokens: int,
    geom: Geometry,
    *,
    page_tokens: int = 16,
    pte_bytes: int = 8,
    tag_bytes: int = 1,
    residual_length: int = 128,
    dma_align: int = 32,
    metadata_placement: str = "separate",
) -> Occupancy:
    """解析占用；须与 ``PackedPhysicalCache.occupancy`` 在同设置下一致。"""
    key = format_id.strip().upper()
    if layout not in {"contiguous", "paged"}:
        raise ValueError(layout)
    colocated = metadata_placement == "colocated"
    heads, dim = geom.kv_heads, geom.head_dim
    if key == "C5":
        pools = kivi_pool_tokens(n_tokens, residual_length)
        payload = _int4_payload(pools["k_quant"] + pools["v_quant"], heads, dim)
        meta = _c5_key_meta(pools["k_quant"], heads, dim) + _c5_value_meta(
            pools["v_quant"], heads, dim
        )
        scale = meta // 2
        offset = meta - scale
        residual = _fp16(pools["k_res"] + pools["v_res"], heads, dim)
        tail = 0
        counts = expected_page_counts(
            key, layout, n_tokens, page_tokens=page_tokens, residual_length=residual_length
        )
        n_pte = sum(counts.values()) if layout == "paged" else 0
        hbm_pay = hbm_meta = n_hbm = 0
        if layout == "contiguous":
            k_pay = _int4_payload(pools["k_quant"], heads, dim)
            v_pay = _int4_payload(pools["v_quant"], heads, dim)
            k_meta = _c5_key_meta(pools["k_quant"], heads, dim)
            v_meta = _c5_value_meta(pools["v_quant"], heads, dim)
            if colocated:
                hbm_pay = align_up(k_pay + k_meta, dma_align) + align_up(v_pay + v_meta, dma_align)
            else:
                hbm_pay = align_up(k_pay, dma_align) + align_up(v_pay, dma_align)
                hbm_meta = align_up(k_meta, dma_align) + align_up(v_meta, dma_align)
            n_hbm = int(k_pay > 0) + int(v_pay > 0)
        else:
            hbm_pay, hbm_meta, n_hbm = _c5_paged_hbm(
                pools, heads, dim, page_tokens, dma_align, colocated
            )
        return Occupancy(
            n_tokens=n_tokens,
            payload_packed=payload,
            residual_fp16=residual,
            tail_fp16=tail,
            scale=scale,
            offset=offset,
            pte=n_pte * pte_bytes,
            tags=n_pte * tag_bytes,
            hbm_payload_allocated=hbm_pay,
            hbm_metadata_allocated=hbm_meta,
            n_hbm_pages=n_hbm,
            n_pte=n_pte,
        )

    committed, tail_tokens = _uniform_committed(n_tokens, layout, page_tokens)
    if key == "C0":
        payload = 2 * _fp16(committed, heads, dim)
        scale = offset = 0
        page_pay = _fp16(page_tokens, heads, dim)
        page_meta = 0
    elif key in {"C2", "C3"}:
        payload = 2 * _int4_payload(committed, heads, dim)
        scale = 2 * _c2_scale(committed, heads, dim)
        offset = 0
        page_pay = _int4_payload(page_tokens, heads, dim)
        page_meta = _c2_scale(page_tokens, heads, dim)
    else:
        raise ValueError(key)
    tail = 2 * _fp16(tail_tokens, heads, dim)
    counts = expected_page_counts(
        key, layout, n_tokens, page_tokens=page_tokens, residual_length=residual_length
    )
    n_pte = sum(counts.values()) if layout == "paged" else 0
    if layout == "contiguous":
        side_pay = payload // 2
        side_meta = scale // 2
        if colocated:
            hbm_pay = 2 * align_up(side_pay + side_meta, dma_align)
            hbm_meta = 0
        else:
            hbm_pay = 2 * align_up(side_pay, dma_align)
            hbm_meta = 2 * align_up(side_meta, dma_align) if side_meta else 0
        n_hbm = 2 if committed else 0
    else:
        full = committed // page_tokens
        hbm_pay, hbm_meta = _page_alloc(
            2 * full, page_pay, page_meta, align=dma_align, colocated=colocated
        )
        n_hbm = 2 * full
    return Occupancy(
        n_tokens=n_tokens,
        payload_packed=payload,
        residual_fp16=0,
        tail_fp16=tail,
        scale=scale,
        offset=offset,
        pte=n_pte * pte_bytes,
        tags=n_pte * tag_bytes,
        hbm_payload_allocated=hbm_pay,
        hbm_metadata_allocated=hbm_meta,
        n_hbm_pages=n_hbm,
        n_pte=n_pte,
    )


def _c5_paged_hbm(
    pools: dict[str, int],
    heads: int,
    dim: int,
    page: int,
    align: int,
    colocated: bool,
) -> tuple[int, int, int]:
    """C5 量化池按页 DMA：Key 的 scale/min 只挂在每组第一页。"""
    hbm_pay = hbm_meta = n_hbm = 0
    k_page_pay = _int4_payload(page, heads, dim)
    k_meta_first = _c5_key_meta(GROUP, heads, dim)
    k_full, k_last = divmod(pools["k_quant"], page)
    for index in range(k_full):
        meta = k_meta_first if index % (GROUP // page) == 0 else 0
        pay_a, meta_a = _page_alloc(1, k_page_pay, meta, align=align, colocated=colocated)
        hbm_pay += pay_a
        hbm_meta += meta_a
        n_hbm += 1
    if k_last:
        pay_a, meta_a = _page_alloc(
            1, _int4_payload(k_last, heads, dim), 0, align=align, colocated=colocated
        )
        hbm_pay += pay_a
        hbm_meta += meta_a
        n_hbm += 1
    v_full, v_last = divmod(pools["v_quant"], page)
    v_page_pay = k_page_pay
    v_page_meta = _c5_value_meta(page, heads, dim)
    pay_a, meta_a = _page_alloc(v_full, v_page_pay, v_page_meta, align=align, colocated=colocated)
    hbm_pay += pay_a
    hbm_meta += meta_a
    n_hbm += v_full
    if v_last:
        pay_a, meta_a = _page_alloc(
            1,
            _int4_payload(v_last, heads, dim),
            _c5_value_meta(v_last, heads, dim),
            align=align,
            colocated=colocated,
        )
        hbm_pay += pay_a
        hbm_meta += meta_a
        n_hbm += 1
    return hbm_pay, hbm_meta, n_hbm


def _transfer(payload, meta, align, colocated):
    sizes = [payload + meta] if colocated else [payload, meta]
    return sum(align_up(x, align) for x in sizes if x), sum(x > 0 for x in sizes)


def cache_read_actions(
    fmt: str,
    layout: str,
    start: int,
    end: int,
    geom: Geometry,
    hw: Hardware,
    *,
    n_tokens: int,
    metadata_placement: str = "separate",
    side_filter: str | None = None,
) -> dict[str, int]:
    """分页读取相交整页；连续地址按范围 DMA。返回动作，不展开高精度 KV。"""
    row = dict(
        logical=0,
        physical=0,
        transactions=0,
        sram_read=0,
        pte=0,
        quantized_elements=0,
        k_quant_tokens=0,
        v_tail_tokens=0,
    )
    if fmt == "C5":
        pools = kivi_pool_tokens(n_tokens, 128)
        lengths = {"k": pools["k_quant"], "v": pools["v_quant"]}
    else:
        committed, _ = _uniform_committed(n_tokens, layout, 16)
        lengths = {"k": committed, "v": committed}
    for side in [side_filter] if side_filter else ("k", "v"):
        quant_end = lengths[side]
        a, b = start, min(end, quant_end)
        if b > a:
            spans = (
                [(p * 16, min((p + 1) * 16, quant_end)) for p in range(a // 16, ceil_div(b, 16))]
                if layout == "paged"
                else [(a, b)]
            )
            seen_groups = set()
            for lo, hi in spans:
                count = hi - lo
                payload = (_fp16 if fmt == "C0" else _int4_payload)(
                    count, geom.kv_heads, geom.head_dim
                )
                if fmt == "C0":
                    meta = 0
                elif fmt == "C5" and side == "k":
                    groups = set(range(lo // GROUP, ceil_div(hi, GROUP))) - seen_groups
                    seen_groups.update(groups)
                    meta = len(groups) * geom.kv_heads * geom.head_dim * 4
                elif fmt == "C5":
                    meta = _c5_value_meta(count, geom.kv_heads, geom.head_dim)
                else:
                    meta = _c2_scale(count, geom.kv_heads, geom.head_dim)
                companion_meta = (
                    fmt == "C5"
                    and side == "k"
                    and layout == "paged"
                    and lo // 16 % 2 == 1
                    and meta > 0
                )
                # 从 group 第二页开始时，元数据在另一页，不能伪称一次共址读取。
                physical, tx = _transfer(
                    payload,
                    meta,
                    hw.dma_align_bytes,
                    metadata_placement == "colocated" and not companion_meta,
                )
                row["logical"] += payload + meta
                row["physical"] += physical
                row["transactions"] += tx
                row["pte"] += int(layout == "paged")
            if fmt != "C0":
                row["quantized_elements"] += (b - a) * geom.kv_heads * geom.head_dim
                if side == "k":
                    row["k_quant_tokens"] += b - a
        a, b = max(start, quant_end), end
        if b > a:
            row["sram_read"] += _fp16(b - a, geom.kv_heads, geom.head_dim)
            if layout == "paged":
                row["pte"] += ceil_div(b - quant_end, 16) - (a - quant_end) // 16
            if side == "v":
                row["v_tail_tokens"] += b - a
    return row


def append_actions(
    fmt: str,
    layout: str,
    n_tokens: int,
    geom: Geometry,
    hw: Hardware,
    *,
    metadata_placement: str = "separate",
) -> dict[str, int]:
    """一个 token 的缓存动作；与 tokenwise PackedPhysicalCache 事件独立对拍。"""
    row = dict(
        logical=0,
        physical=0,
        transactions=0,
        rmw_logical=0,
        rmw_physical=0,
        rmw_transactions=0,
        sram_read=0,
        sram_write=0,
        pte=0,
        quantized_elements=0,
        pack_bytes=0,
        rotation_macs=0,
    )
    if not n_tokens:
        return row
    heads, dim = geom.kv_heads, geom.head_dim

    def transfer(payload, meta, *, rmw=False):
        physical, tx = _transfer(
            payload, meta, hw.dma_align_bytes, metadata_placement == "colocated"
        )
        prefix = "rmw_" if rmw else ""
        for name, value in (
            ("logical", payload + meta),
            ("physical", physical),
            ("transactions", tx),
        ):
            row[prefix + name] += value
        if not rmw and fmt != "C0":
            row["pack_bytes"] += payload

    if fmt == "C5":
        row["sram_write"] = 2 * _fp16(1, heads, dim)
        if layout == "paged":
            row["pte"] += int((n_tokens - 1) % 16 == 0)
            row["pte"] += int(n_tokens > 128 or (n_tokens - 1) % 16 == 0)
        if n_tokens % 128 == 0:
            row["sram_read"] += _fp16(128, heads, dim)
            row["quantized_elements"] += 128 * heads * dim
            if layout == "paged":
                for page in range(8):
                    transfer(
                        _int4_payload(16, heads, dim),
                        _c5_key_meta(32, heads, dim) if page % 2 == 0 else 0,
                    )
                row["pte"] += 8
            else:
                transfer(_int4_payload(128, heads, dim), _c5_key_meta(128, heads, dim))
        if n_tokens > 128:
            row["sram_read"] += _fp16(129, heads, dim)
            row["sram_write"] += _fp16(128, heads, dim)
            row["quantized_elements"] += heads * dim
            old = (n_tokens - 129) % 16 if layout == "paged" else 0
            if old:
                transfer(_int4_payload(old, heads, dim), _c5_value_meta(old, heads, dim), rmw=True)
            elif layout == "paged":
                row["pte"] += 1
            transfer(_int4_payload(old + 1, heads, dim), _c5_value_meta(old + 1, heads, dim))
    else:
        count = 1 if layout == "contiguous" else (16 if n_tokens % 16 == 0 else 0)
        if layout == "paged":
            row["sram_write"] = 2 * _fp16(1, heads, dim)
            row["sram_read"] = 2 * _fp16(count, heads, dim)
            row["pte"] = 2 if n_tokens % 16 in {0, 1} else 0
        for _side in ("k", "v"):
            if fmt == "C0":
                transfer(_fp16(count, heads, dim), 0)
            else:
                transfer(_int4_payload(count, heads, dim), _c2_scale(count, heads, dim))
        if fmt != "C0":
            row["quantized_elements"] = 2 * count * heads * dim
        if fmt == "C3":
            row["rotation_macs"] = 2 * count * heads * geom.rotation_macs_per_vector
    return row


def _blocks(n_tokens, mapping):
    length = ceil_div(n_tokens, mapping.kv_split)
    splits = []
    for split in range(mapping.kv_split):
        lo, hi = split * length, min((split + 1) * length, n_tokens)
        blocks = []
        for seg, start in enumerate(range(lo, hi, mapping.merge_tokens)):
            end = min(start + mapping.merge_tokens, hi)
            for a in range(start, end, mapping.tile):
                b = min(a + mapping.tile, end)
                blocks.append(
                    dict(start=a, end=b, split=split, local_merge=int(seg > 0 and b == end))
                )
        if blocks:
            splits.append(blocks)
    return [
        items[i]
        for i in range(max(map(len, splits), default=0))
        for items in splits
        if i < len(items)
    ]


def _active_lanes(mapping, geom, hw):
    return min(mapping.kv_split, hw.operand_ports) if mapping.kv_split > 1 else 1


def _matrix_cycles(m, n, k, rows, cols, startup):
    if not m or not n or not k:
        return 0.0
    return float(ceil_div(m, rows) * ceil_div(n, cols) * (k + startup))


def _qk_pv_cycles(geom, mapping, tile, hw, *, lanes=1, qk_pad=0):
    """Query 映射 PE 行，QK token/PV channel 映射列；尾 PE 空闲。"""
    heads = min(mapping.parallel_kv_heads, hw.operand_ports, geom.kv_heads)
    parts = lanes if lanes > 1 else heads
    rows = hw.pe_rows // parts
    if rows < 1:
        raise ValueError("活动通路超过 PE 行预算")
    qr = geom.gqa_ratio if mapping.gqa_mode == "gqa_shared" else 1
    groups = geom.kv_heads if mapping.gqa_mode == "gqa_shared" else geom.q_heads
    waves = ceil_div(groups, heads)
    startup = hw.execution["array_startup_cycles"]
    qk = waves * _matrix_cycles(qr, tile + qk_pad, geom.head_dim, rows, hw.pe_cols, startup)
    pv = waves * _matrix_cycles(qr, geom.head_dim, tile, rows, hw.pe_cols, startup)
    return qk, pv, heads


def _rotation_cycles(tokens, geom, hw, *, lanes=1, repeat=1, min_rows=1):
    """逐 head、至多 16 个向量的 32×32 分块旋转，不默认全阵列利用。"""
    total = 0.0
    for start in range(0, tokens, hw.operand_tokens):
        count = max(min_rows, min(hw.operand_tokens, tokens - start))
        total += _matrix_cycles(
            count,
            BDR_BLOCK,
            BDR_BLOCK,
            hw.pe_rows // lanes,
            hw.pe_cols,
            hw.execution["array_startup_cycles"],
        )
    return total * geom.kv_heads * (geom.head_dim // BDR_BLOCK) * repeat


def schedule_tasks(
    tasks: list[dict], *, slots: int, lanes: int, fused: bool, queue_depth: int = 4
) -> dict:
    """依赖图：共享 DMA、分区计算、slot 释放；并行服务时间不相加作墙钟。"""
    nodes, last, history = [], [None] * lanes, [[] for _ in range(lanes)]
    dma_nodes = []

    def add(resource, duration, deps):
        node = len(nodes)
        if resource == "dma":
            if len(dma_nodes) >= queue_depth:
                deps = [*deps, dma_nodes[-queue_depth]]
            dma_nodes.append(node)
        nodes.append(
            dict(
                resource=resource,
                duration=duration,
                deps=list(dict.fromkeys(x for x in deps if x is not None)),
                successors=[],
            )
        )
        return node

    for task in tasks:
        lane = task["lane"]
        release = history[lane][-slots] if len(history[lane]) >= slots else None
        if fused:
            load = add("dma", task["load_k"] + task["load_v"], [release])
            done = add(f"lane{lane}", task["compute_k"] + task["compute_v"], [load, last[lane]])
        else:
            load = add("dma", task["load_k"], [release])
            key = add(f"lane{lane}", task["compute_k"], [load, last[lane]])
            value = add("dma", task["load_v"], [key])
            done = add(f"lane{lane}", task["compute_v"], [value])
        last[lane] = done
        history[lane].append(done)
    pending = [len(n["deps"]) for n in nodes]
    release_times = [0.0] * len(nodes)
    for i, node in enumerate(nodes):
        for dep in node["deps"]:
            nodes[dep]["successors"].append(i)
    ready = [(0.0, i) for i, count in enumerate(pending) if not count]
    heapq.heapify(ready)
    free, service, stalls, finish = {}, {}, {}, [0.0] * len(nodes)
    while ready:
        release, i = heapq.heappop(ready)
        node = nodes[i]
        resource = node["resource"]
        start = max(release, free.get(resource, 0.0))
        end = start + node["duration"]
        free[resource] = finish[i] = end
        service[resource] = service.get(resource, 0.0) + node["duration"]
        stalls[resource] = stalls.get(resource, 0.0) + start - release
        for child in node["successors"]:
            release_times[child] = max(release_times[child], end)
            pending[child] -= 1
            if not pending[child]:
                heapq.heappush(ready, (release_times[child], child))
    return dict(
        latency=max(finish, default=0.0),
        service_cycles=service,
        resource_wait_cycles=stalls,
        tasks=len(tasks),
        active_lanes=lanes,
        buffer_slots_per_lane=slots,
        dma_queue_depth=queue_depth,
        ordering="dependency_release_then_node_id",
    )


def _sram_service(logical_read, logical_write, hw, *, lanes=1):
    width = hw.sram_width_bytes
    read = align_up(int(logical_read), width) if logical_read else 0
    write = align_up(int(logical_write), width) if logical_write else 0
    bw = hw.sram_banks * (hw.sram_rw_ports - hw.execution["sram_dma_ports"]) * width / lanes
    if bw <= 0:
        raise ValueError("须为 DMA 与计算各保留至少一个 SRAM 端口")
    cycles = (read + write) / bw + hw.sram_latency_cycles * (int(read > 0) + int(write > 0))
    return read, write, cycles


def sram_footprint(
    geom: Geometry,
    mapping: Mapping,
    occ: Occupancy,
    hw: Hardware,
    *,
    format_id: str,
    capacity_tokens: int | None = None,
) -> dict[str, int]:
    """物理缓存和有界工作缓冲的峰值；容量专用入口可预约大 tile。"""
    lanes = _active_lanes(mapping, geom, hw)
    heads = min(mapping.parallel_kv_heads, hw.operand_ports, geom.kv_heads)
    slots = 2 if mapping.buffering == "double" else 1
    sides = 2 if mapping.kv_pass == "fused_tile" else 1
    tokens = min(mapping.tile, mapping.merge_tokens, max(occ.n_tokens, 1))
    if capacity_tokens is not None:
        tokens = min(capacity_tokens, max(occ.n_tokens, 1))
    staged = min(max(occ.n_tokens, 1), align_up(tokens + (30 if mapping.kv_split > 1 else 0), 16))
    pay = (_fp16 if format_id == "C0" else _int4_payload)(staged, geom.kv_heads, geom.head_dim)
    meta = (
        0
        if format_id == "C0"
        else max(
            _c2_scale(staged, geom.kv_heads, geom.head_dim),
            _c5_key_meta(align_up(staged, 32), geom.kv_heads, geom.head_dim)
            if format_id == "C5"
            else 0,
            _c5_value_meta(staged, geom.kv_heads, geom.head_dim) if format_id == "C5" else 0,
        )
    )
    return {
        "q_fp16": geom.q_heads * geom.head_dim * 2,
        "partial_o_fp32": 2 * geom.q_heads * geom.head_dim * 4 * mapping.kv_split,
        "softmax_stats_fp32": 2 * geom.q_heads * 8 * mapping.kv_split,
        "score_tile_fp32": geom.q_heads * tokens * 4 * lanes,
        "operand_fp32": mapping.group_buffers
        * hw.operand_tokens
        * geom.head_dim
        * 4
        * heads
        * lanes,
        "packed_group_buffer": pay * sides * slots * lanes,
        "scale_min_buffer": meta * sides * slots * lanes,
        "pte_tags": occ.pte + occ.tags,
        "residual_or_tail_fp16": max(
            occ.sram_fp16,
            (2 * _fp16(16, geom.kv_heads, geom.head_dim) if occ.n_pte else 0)
            if format_id != "C5"
            else _fp16(257, geom.kv_heads, geom.head_dim),
        ),
        "quantization_scratch": (
            32 * geom.kv_heads * geom.head_dim * 4 if format_id == "C5" else 32 * 4 * geom.kv_heads
        )
        if format_id != "C0"
        else 0,
        "dma_queue": hw.dma_queue_depth * 32,
        "operand_address_queues": hw.operand_ports * hw.dma_queue_depth * 16,
        "rotation_matrix_fp32": geom.head_dim * BDR_BLOCK * 4 if format_id == "C3" else 0,
    }


def simulate_step(
    format_id: str,
    layout: str,
    n_tokens: int,
    geom: Geometry,
    mapping: Mapping,
    hw: Hardware,
    *,
    metadata_placement: str = "separate",
) -> StepResult:
    """带 shape、端口、缓冲依赖的单层一步声明模型。"""
    _nonneg_int("n_tokens", n_tokens)
    if mapping.batch_parallel:
        raise ValueError("当前 batch 仅时间复用")
    occ = layout_occupancy(format_id, layout, n_tokens, geom, metadata_placement=metadata_placement)
    parts = sram_footprint(geom, mapping, occ, hw, format_id=format_id)
    peak = sum(parts.values())

    def result(feasible, reason, actions, components, latency, serial, roofline, energy, schedule):
        return StepResult(
            feasible,
            reason,
            geom.name,
            format_id,
            layout,
            n_tokens,
            asdict(mapping),
            asdict(occ),
            actions,
            parts,
            peak,
            components,
            latency,
            serial,
            roofline,
            energy,
            sum(energy.values()) if feasible else None,
            actions.get("total_mac_equivalents", 0) / (latency * hw.mac_budget) if latency else 0.0,
            schedule=schedule,
        )

    if mapping.parallel_kv_heads > 1 and mapping.kv_split > 1:
        return result(
            False,
            "refuse_combined_head_and_kv_split_without_added_mac",
            {},
            {},
            None,
            None,
            0,
            {},
            {},
        )
    if peak > hw.sram_bytes:
        return result(False, "sram_capacity", {}, {}, None, None, 0, {}, {})
    if not n_tokens:
        return result(True, "empty", {}, {}, 0.0, 0.0, 0.0, {}, {})
    lanes = _active_lanes(mapping, geom, hw)
    budget_per_lane = (hw.pe_rows // lanes) * hw.pe_cols
    repeat = geom.gqa_ratio if mapping.gqa_mode == "gqa_expand" else 1
    blocks = _blocks(n_tokens, mapping)
    rates = hw.execution
    actions = dict(
        hbm_read_logical_bytes=0,
        hbm_read_physical_bytes=0,
        hbm_read_transactions=0,
        sram_read_bytes=0,
        sram_write_bytes=0,
        sram_read_physical_bytes=0,
        sram_write_physical_bytes=0,
        pte_lookups=0,
        tag_reads=0,
        quantized_elements=0,
        qk_macs=0,
        pv_macs=0,
        c3_key_inverse_macs=0,
        c3_tail_rotation_macs=0,
        c3_pad_macs=0,
        partial_o_rescale=0,
        merge_macs=0,
        softmax_scores=0,
        broadcast_bytes=0,
        convert_elements=0,
        control_events=0,
        n_tiles=len(blocks),
        n_segments=0,
    )
    components = dict(
        dma=0.0,
        dequant=0.0,
        rotation_read=0.0,
        qk=0.0,
        pv=0.0,
        softmax=0.0,
        rescale=0.0,
        merge_local=0.0,
        sram=0.0,
        broadcast=0.0,
        control=0.0,
        conversion=0.0,
    )
    tasks = []

    def memory(rd, wr, *, dma_write=False, lane_count=lanes):
        rp, wp, cycles = _sram_service(rd, 0 if dma_write else wr, hw, lanes=lane_count)
        if dma_write:
            wp = align_up(wr, hw.sram_width_bytes) if wr else 0
        actions["sram_read_bytes"] += rd
        actions["sram_write_bytes"] += wr
        actions["sram_read_physical_bytes"] += rp
        actions["sram_write_physical_bytes"] += wp
        return cycles

    for block in blocks:
        a, b = block["start"], block["end"]
        count = b - a
        reads = {
            side: cache_read_actions(
                format_id,
                layout,
                a,
                b,
                geom,
                hw,
                n_tokens=n_tokens,
                metadata_placement=metadata_placement,
                side_filter=side,
            )
            for side in ("k", "v")
        }
        read = {key: sum(item[key] for item in reads.values()) for key in reads["k"]}
        for dst, src in (
            ("hbm_read_logical_bytes", "logical"),
            ("hbm_read_physical_bytes", "physical"),
            ("hbm_read_transactions", "transactions"),
            ("quantized_elements", "quantized_elements"),
            ("pte_lookups", "pte"),
            ("tag_reads", "pte"),
        ):
            actions[dst] += read[src] * repeat
        inv_pad = qk_pad = 0
        if format_id == "C3":
            for lo in range(a, b, hw.operand_tokens):
                size = min(b, lo + hw.operand_tokens) - lo
                qk_pad += max(0, 4 - size)
                quant_end = n_tokens if layout == "contiguous" else n_tokens // 16 * 16
                quant_size = max(0, min(b, lo + hw.operand_tokens, quant_end) - lo)
                inv_pad += max(0, 4 - quant_size) if quant_size else 0
        qk, pv, _ = _qk_pv_cycles(geom, mapping, count, hw, lanes=lanes, qk_pad=qk_pad)
        inv = (
            (read["k_quant_tokens"] + inv_pad)
            * geom.kv_heads
            * geom.rotation_macs_per_vector
            * repeat
            if format_id == "C3"
            else 0
        )
        tail_rot = (
            read["v_tail_tokens"] * geom.kv_heads * geom.rotation_macs_per_vector * repeat
            if format_id == "C3"
            else 0
        )
        actions["c3_key_inverse_macs"] += (
            inv - inv_pad * geom.kv_heads * geom.rotation_macs_per_vector * repeat
        )
        actions["c3_tail_rotation_macs"] += tail_rot
        actions["c3_pad_macs"] += (
            inv_pad * geom.kv_heads * geom.rotation_macs_per_vector * repeat
            + qk_pad * geom.q_heads * geom.head_dim
        )
        actions["qk_macs"] += geom.q_heads * count * geom.head_dim
        actions["pv_macs"] += geom.q_heads * count * geom.head_dim
        rescale = geom.q_heads * (geom.head_dim + 1)
        merge = block["local_merge"] * 2 * geom.q_heads * (geom.head_dim + 1)
        actions["partial_o_rescale"] += rescale
        actions["merge_macs"] += merge
        scores = geom.q_heads * (count + 1 + 2 * block["local_merge"])
        actions["softmax_scores"] += scores
        side_costs = {}
        for side in ("k", "v"):
            item = reads[side]
            decoded = count * geom.kv_heads * geom.head_dim * repeat
            conversion = decoded - item["quantized_elements"] * repeat
            control = (
                item["pte"] * repeat * 2
                + ceil_div(count, hw.operand_tokens) * geom.kv_heads * repeat
            )
            broadcast = (
                count * geom.kv_heads * geom.head_dim * 4 * (geom.gqa_ratio - 1)
                if repeat == 1
                else 0
            )
            actions["convert_elements"] += conversion
            actions["control_events"] += control
            actions["broadcast_bytes"] += broadcast
            rot_macs = inv if side == "k" else tail_rot
            sram_cycles = memory(
                item["physical"] * repeat, item["physical"] * repeat, dma_write=True
            )
            sram_cycles += memory(item["sram_read"] * repeat, 0)
            sram_cycles += memory(decoded * 4, decoded * 4)
            sram_cycles += memory(geom.q_heads * count * 4, geom.q_heads * count * 4)
            sram_cycles += memory(item["pte"] * repeat * 9, 0)
            if side == "k":
                sram_cycles += memory(
                    geom.q_heads * geom.head_dim * 2 * ceil_div(count, hw.operand_tokens), 0
                )
            else:
                state = geom.q_heads * (geom.head_dim * 4 + 8)
                sram_cycles += memory(
                    state * (1 + 2 * block["local_merge"]), state * (1 + block["local_merge"])
                )
            if rot_macs:
                # 每个最多 16 行的旋转组读取一次块对角权重。
                rot_vectors = ceil_div(rot_macs, geom.rotation_macs_per_vector)
                sram_cycles += memory(
                    ceil_div(rot_vectors, hw.operand_tokens) * geom.head_dim * BDR_BLOCK * 4, 0
                )
            rotation_cycles = 0.0
            if format_id == "C3":
                rotation_cycles = _rotation_cycles(
                    item["k_quant_tokens"] if side == "k" else item["v_tail_tokens"],
                    geom,
                    hw,
                    lanes=lanes,
                    repeat=repeat,
                    min_rows=4 if side == "k" else 1,
                )
            service = dict(
                dequant=math.ceil(
                    item["quantized_elements"] * repeat * lanes / hw.dequant_elements_cycle
                ),
                rotation_read=rotation_cycles,
                sram=sram_cycles,
                broadcast=broadcast * lanes / rates["broadcast_bytes_cycle"],
                control=control * lanes / rates["control_events_cycle"],
                conversion=conversion * lanes / rates["convert_elements_cycle"],
            )
            side_costs[side] = sum(service.values())
            for key, value in service.items():
                components[key] += value
        sm = scores * lanes / rates["softmax_scores_cycle"]
        rescale_c, merge_c = (
            math.ceil(rescale / budget_per_lane),
            math.ceil(merge / budget_per_lane),
        )
        load_k, load_v = (
            hw.dma_cycles(reads[s]["physical"] * repeat, reads[s]["transactions"] * repeat)
            for s in ("k", "v")
        )
        tasks.append(
            dict(
                lane=block["split"] % lanes,
                load_k=load_k,
                load_v=load_v,
                compute_k=qk + sm + rescale_c + side_costs["k"],
                compute_v=pv + merge_c + side_costs["v"],
            )
        )
        for key, value in dict(
            dma=load_k + load_v, qk=qk, pv=pv, softmax=sm, rescale=rescale_c, merge_local=merge_c
        ).items():
            components[key] += value
    nonempty_splits = len({block["split"] for block in blocks})
    actions["n_segments"] = sum(block["local_merge"] for block in blocks) + nonempty_splits
    final_merge = 2 * geom.q_heads * (geom.head_dim + 1) * (nonempty_splits - 1)
    actions["merge_macs"] += final_merge
    write = append_actions(
        format_id, layout, n_tokens, geom, hw, metadata_placement=metadata_placement
    )
    actions.update(
        hbm_write_logical_bytes=write["logical"],
        hbm_write_physical_bytes=write["physical"],
        hbm_write_transactions=write["transactions"],
        hbm_rmw_read_logical_bytes=write["rmw_logical"],
        hbm_rmw_read_physical_bytes=write["rmw_physical"],
        hbm_rmw_read_transactions=write["rmw_transactions"],
        append_sram_read_bytes=write["sram_read"],
        append_sram_write_bytes=write["sram_write"],
        pte_writes=write["pte"],
        tag_writes=write["pte"],
        write_quantized_elements=write["quantized_elements"],
        pack_bytes=write["pack_bytes"],
        c3_write_rotation_macs=write["rotation_macs"],
        c3_output_inverse_macs=geom.q_heads * geom.rotation_macs_per_vector
        if format_id == "C3"
        else 0,
        normalization_elements=geom.q_heads * geom.head_dim,
    )
    qio = geom.q_heads * geom.head_dim * 2
    actions["q_read_hbm_bytes"] = actions["o_write_hbm_bytes"] = align_up(qio, hw.dma_align_bytes)
    actions["r1_logical_read_bytes"] = occ.logical_payload_and_meta + occ.sram_fp16
    actions["alignment_waste_bytes"] = (
        actions["hbm_read_physical_bytes"]
        - actions["hbm_read_logical_bytes"]
        + write["physical"]
        - write["logical"]
        + write["rmw_physical"]
        - write["rmw_logical"]
    )
    prefix_sram = memory(write["sram_read"], write["sram_write"], lane_count=1)
    prefix_sram += memory(
        write["quantized_elements"] * 4, write["quantized_elements"] * 2, lane_count=1
    )
    prefix_sram += memory(
        write["physical"] + write["rmw_physical"],
        write["physical"] + write["rmw_physical"],
        lane_count=1,
    )
    prefix_sram += memory(0, write["pte"] * 9, lane_count=1)
    prefix_sram += memory(qio, qio, lane_count=1)
    suffix_sram = memory(
        geom.q_heads * (geom.head_dim * 4 + 8) * (1 + 2 * (nonempty_splits - 1)), qio, lane_count=1
    )
    write_ctrl = 1 + 2 * write["pte"] + write["transactions"] + write["rmw_transactions"]
    actions["control_events"] += write_ctrl
    actions["convert_elements"] += qio
    # 归并指数和写/输出旋转矩阵访问也独立入账。
    final_merge_scores = 2 * geom.q_heads * (nonempty_splits - 1)
    actions["softmax_scores"] += final_merge_scores
    if format_id == "C3":
        prefix_sram += memory(
            geom.head_dim * BDR_BLOCK * 4 * 2 * geom.kv_heads if write["rotation_macs"] else 0,
            0,
            lane_count=1,
        )
        suffix_sram += memory(geom.head_dim * BDR_BLOCK * 4, 0, lane_count=1)
    written_tokens = (
        write["quantized_elements"] // (2 * geom.kv_heads * geom.head_dim)
        if format_id == "C3"
        else 0
    )
    outside_components = dict(
        q_load=hw.dma_cycles(actions["q_read_hbm_bytes"], 1),
        write_dma=hw.dma_cycles(
            write["physical"] + write["rmw_physical"],
            write["transactions"] + write["rmw_transactions"],
        ),
        write_quant=write["quantized_elements"] / rates["quant_elements_cycle"],
        write_pack=write["pack_bytes"] / rates["pack_bytes_cycle"],
        write_rotation=2 * _rotation_cycles(written_tokens, geom, hw) if format_id == "C3" else 0,
        write_control=write_ctrl / rates["control_events_cycle"],
        prefix_sram=prefix_sram,
        q_conversion=qio / 2 / rates["convert_elements_cycle"],
        merge_final=math.ceil(final_merge / hw.mac_budget)
        + final_merge_scores / rates["softmax_scores_cycle"],
        output_inverse=(
            _matrix_cycles(
                geom.q_heads,
                BDR_BLOCK,
                BDR_BLOCK,
                hw.pe_rows,
                hw.pe_cols,
                rates["array_startup_cycles"],
            )
            * (geom.head_dim // BDR_BLOCK)
        )
        if format_id == "C3"
        else 0,
        normalize=actions["normalization_elements"] / rates["softmax_scores_cycle"],
        suffix_sram=suffix_sram,
        o_conversion=qio / 2 / rates["convert_elements_cycle"],
        o_store=hw.dma_cycles(actions["o_write_hbm_bytes"], 1),
    )
    components.update(outside_components)
    sched = schedule_tasks(
        tasks,
        slots=2 if mapping.buffering == "double" else 1,
        lanes=lanes,
        fused=mapping.kv_pass == "fused_tile",
        queue_depth=hw.dma_queue_depth,
    )
    outside = sum(outside_components.values())
    latency = outside + sched["latency"]
    serial = sum(components.values())
    macs = sum(
        actions[k]
        for k in (
            "qk_macs",
            "pv_macs",
            "c3_key_inverse_macs",
            "c3_tail_rotation_macs",
            "c3_pad_macs",
            "c3_write_rotation_macs",
            "c3_output_inverse_macs",
            "partial_o_rescale",
            "merge_macs",
        )
    )
    actions["total_mac_equivalents"] = macs
    hbm = sum(
        actions[k]
        for k in (
            "hbm_read_physical_bytes",
            "hbm_write_physical_bytes",
            "hbm_rmw_read_physical_bytes",
            "q_read_hbm_bytes",
            "o_write_hbm_bytes",
        )
    )
    actions["hbm_total_physical_bytes"] = hbm
    actions["hbm_kv_physical_bytes"] = hbm - 2 * align_up(qio, hw.dma_align_bytes)
    energy = dict(
        hbm=hbm * 8 * hw.hbm2_pJ_per_bit,
        sram=(actions["sram_read_physical_bytes"] + actions["sram_write_physical_bytes"])
        * 8
        * hw.sram_pJ_per_bit,
        mac=macs * hw.mac_fp16_pJ,
        dequant=actions["quantized_elements"] * hw.dequant_pJ_per_element,
        quant=write["quantized_elements"] * hw.quant_pJ_per_element,
        pack=write["pack_bytes"] * hw.pack_pJ_per_byte,
        softmax=(actions["softmax_scores"] + actions["normalization_elements"])
        * hw.softmax_pJ_per_score,
        broadcast=actions["broadcast_bytes"] * hw.broadcast_pJ_per_byte,
        control=actions["control_events"] * hw.control_pJ_per_event,
        conversion=actions["convert_elements"] * hw.convert_pJ_per_element,
    )
    roofline = max(macs / hw.mac_budget, hbm / hw.bytes_per_cycle)
    sched.update(
        mac_budget=hw.mac_budget,
        macs_per_lane=budget_per_lane,
        physical_compute_partitions=max(
            lanes, min(mapping.parallel_kv_heads, hw.operand_ports, geom.kv_heads)
        ),
        operand_ports=hw.operand_ports,
        sram_banks=hw.sram_banks,
        sram_width_bytes=hw.sram_width_bytes,
        outside_cycles=outside,
        component_semantics="service_cycles_sum_is_serial_not_parallel_wall_time",
    )
    batch = mapping.batch
    actions = {k: v * batch for k, v in actions.items()}
    components = {k: v * batch for k, v in components.items()}
    energy = {k: v * batch for k, v in energy.items()}
    sched["batch_serial"] = batch
    return result(
        True,
        "ok",
        actions,
        components,
        latency * batch,
        serial * batch,
        roofline * batch,
        energy,
        sched,
    )


def default_mapping(raw: dict | None = None, **overrides: object) -> Mapping:
    """读取开发缺省，仅覆盖显式指定的映射字段。"""
    proto = load_protocol()["architecture"]["development_default"]
    cost = raw or load_mapping_cost()
    values = dict(
        tile=int(proto["tile"]),
        buffering=str(proto["buffering"]),
        kv_pass="separate",
        kv_split=int(proto["kv_split"]),
        gqa_mode="gqa_shared",
        parallel_kv_heads=1,
        group_buffers=int(cost["held_defaults"]["group_buffers"]),
        merge_tokens=int(cost["held_defaults"]["merge_tokens"]),
        batch=int(proto["batch"]),
        batch_parallel=False,
    )
    values.update(overrides)
    return Mapping(**values)


def search_optimized(
    format_id: str, layout: str, n_tokens: int, geom: Geometry, hw: Hardware, raw: dict
) -> dict:
    """预定开发空间；capacity-only tile 由独立入口检查。"""
    space = raw["search_space"]
    candidates = []
    for tile in space["tile"]:
        for buffering in space["buffering"]:
            for kv_pass in space["kv_pass"]:
                for split in space["kv_split"]:
                    for parallel in space["parallel_kv_heads"]:
                        if parallel > hw.operand_ports or (split > 1 and parallel > 1):
                            continue
                        mapping = default_mapping(
                            raw,
                            tile=tile,
                            buffering=buffering,
                            kv_pass=kv_pass,
                            kv_split=split,
                            parallel_kv_heads=parallel,
                        )
                        candidates.append(
                            simulate_step(format_id, layout, n_tokens, geom, mapping, hw)
                        )
    feasible = [r for r in candidates if r.feasible]
    best = (
        min(feasible, key=lambda r: (r.latency_cycles, r.energy_total_dynamic_pJ))
        if feasible
        else None
    )
    return dict(
        n_candidates=len(candidates),
        n_feasible=len(feasible),
        best=step_to_dict(best),
        candidates=[
            dict(
                mapping=r.mapping,
                feasible=r.feasible,
                reason=r.reason,
                latency_cycles=r.latency_cycles,
                energy_dynamic_pJ=r.energy_total_dynamic_pJ,
                sram_peak_bytes=r.sram_peak_bytes,
            )
            for r in candidates
        ],
        infeasible_reasons=sorted({r.reason for r in candidates if not r.feasible}),
    )


def step_to_dict(row: StepResult | None) -> dict | None:
    """保留完整动作和调度元数据供机器核验。"""
    return asdict(row) if row is not None else None


def percentiles(values: list[float]) -> dict[str, float]:
    """线性插值百分位；相关生成步骤不视为独立统计样本。"""
    if not values:
        raise ValueError("百分位需要非空样本")
    ordered = sorted(values)

    def at(p):
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] * (1 - (index - lo)) + ordered[hi] * (index - lo)

    return dict(
        mean=sum(ordered) / len(ordered),
        p50=at(0.5),
        p95=at(0.95),
        p99=at(0.99),
        max=ordered[-1],
        min=ordered[0],
        n=len(ordered),
    )


def simulate_pressure(
    format_id: str,
    layout: str,
    start: int,
    steps: int,
    geom: Geometry,
    mapping: Mapping,
    hw: Hardware,
) -> dict:
    """start 为首步写入后的缓存长度；保存每步动作、占用与分解。"""
    rows = [
        simulate_step(format_id, layout, start + offset, geom, mapping, hw)
        for offset in range(steps)
    ]
    ok = all(r.feasible for r in rows)
    return dict(
        feasible=ok,
        start=start,
        steps=steps,
        length_semantics="post_append_attention_length",
        cycles=percentiles([r.latency_cycles for r in rows]) if ok else None,
        energy_dynamic_pJ=percentiles([r.energy_total_dynamic_pJ for r in rows]) if ok else None,
        hbm_bytes=percentiles([r.actions["hbm_total_physical_bytes"] for r in rows])
        if ok
        else None,
        write_hbm_bytes=percentiles([r.actions["hbm_write_physical_bytes"] for r in rows])
        if ok
        else None,
        n_infeasible=sum(not r.feasible for r in rows),
        trajectory=[step_to_dict(r) for r in rows],
    )
