"""经过校验的逐步缓存计数，以及串行 decode 生成模拟。

采集计数描述名义打包读取量，不代表实测时延或物理 DMA 事务。
不能由流量汇总推断逐步轨迹。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import AttentionWorkload, Hardware, Mapping
from .inputs import (
    FORMATS,
    LAYOUTS,
    MODEL_ID,
    PROTOCOL_IDS,
    InputBundle,
    PressureInput,
    SourceFile,
    StepInput,
    _array,
    _fixed,
    _grid,
    _object,
    _parse_step,
    _read_json,
    _require,
)
from .simulator import simulate, simulate_pressure

EVIDENCE_KIND = "cache_path_nominal_read_measurement"


@dataclass(frozen=True)
class PressureTrace:
    pressure: PressureInput
    steps: tuple[StepInput, ...]
    source: SourceFile
    evidence_kind: str


def _validate_trajectory(pressure: PressureInput, steps: tuple[StepInput, ...]) -> None:
    where = "pressure trajectory"
    _require(len(steps) == pressure.l_out and len(steps) > 0, where, "wrong step count")
    last = pressure.last_step
    for i, step in enumerate(steps):
        _require(step.n == pressure.l_in + i, where, "nonconsecutive step N")
        _require(
            (step.model_id, step.protocol_id, step.kv_format, step.layout, step.geometry)
            == (last.model_id, last.protocol_id, last.kv_format, last.layout, last.geometry),
            where,
            "mixed model, format, layout or geometry",
        )
    _require(
        steps[-1].n == last.n
        and steps[-1].all_layers == last.all_layers
        and steps[-1].b_eff == last.b_eff,
        where,
        "last step disagrees with aggregate source",
    )
    _require(
        sum(s.bytes_per_token for s in steps) == pressure.total_kv_read,
        where,
        "summed trace bytes disagree with aggregate source",
    )


def load_pressure_traces(
    path: str | Path,
    inputs: InputBundle,
    project_root: str | Path | None = None,
) -> tuple[PressureTrace, ...]:
    """加载全部 12 条实测 16384+1024 轨迹，并精确对齐流量输入。

    要求采集完整、步序连续、四项字节分解一致，且首步、末步和全程总量准确。
    每步保留输入哈希及 JSON 定位信息；不导入 torch 依赖。
    """
    root = (
        Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    ).resolve()
    data, source = _read_json(root, Path(path))
    _fixed(
        data,
        {
            "schema_version": 1,
            "complete": True,
            "evidence_kind": EVIDENCE_KIND,
            "model_id": MODEL_ID,
            "page_size": inputs.page_size,
            "pte_bytes": inputs.pte_bytes,
            "c0_path": "fp16_codec",
            "b_pad_included": False,
            "l_in": 16384,
            "l_out": 1024,
        },
        source.path,
    )
    _fixed(
        _object(data.get("geometry"), source.path + "/geometry"),
        asdict(inputs.geometry),
        source.path,
    )
    traces = []
    for i, group in enumerate(_array(data.get("pressures"), source.path + "/pressures")):
        pointer = f"/pressures/{i}"
        group = _object(group, source.path + pointer)
        cid, layout = group.get("protocol_id"), group.get("layout")
        _require(cid in PROTOCOL_IDS and layout in LAYOUTS, pointer, "unknown format or layout")
        _fixed(group, {"kv_format": FORMATS[PROTOCOL_IDS.index(cid)]}, pointer)
        pressure = inputs.pressure(data["l_in"], data["l_out"], cid, layout)
        steps = tuple(
            _parse_step(row, inputs.geometry, source, f"{pointer}/steps/{j}")
            for j, row in enumerate(_array(group.get("steps"), pointer + "/steps"))
        )
        _validate_trajectory(pressure, steps)
        first = inputs.step(pressure.l_in, cid, layout)
        _require(
            steps[0].all_layers == first.all_layers and steps[0].b_eff == first.b_eff,
            pointer,
            "first step disagrees with traffic/PPL window source",
        )
        traces.append(PressureTrace(pressure, steps, source, data["evidence_kind"]))
    _grid([t.pressure.key for t in traces], {p.key for p in inputs.pressures}, source.path)
    return tuple(sorted(traces, key=lambda t: t.pressure.key))


def simulate_pressure_trace(
    trajectory: PressureTrace,
    mapping: Mapping | None = None,
    hw: Hardware | None = None,
    workload: AttentionWorkload | None = None,
) -> dict:
    """逐步模拟 decode，串行 token 与层数各累计一次。

    任一步不可行时，整段生成时延不可用。所有步骤都保存为紧凑机器记录，
    包括不可行步骤。合成数据调用者必须明确保留合成 evidence_kind/source。
    """
    _validate_trajectory(trajectory.pressure, trajectory.steps)
    mapping, hw, workload = mapping or Mapping(), hw or Hardware(), workload or AttentionWorkload()
    summary = simulate_pressure(trajectory.pressure, mapping, hw, workload)
    rows = []
    for i, step in enumerate(trajectory.steps):
        result = simulate(step, mapping, hw, workload)
        rows.append(
            {
                "step_index": i,
                "n": step.n,
                "feasible": result.feasible,
                "reason": result.reason,
                "buffer_slots": result.buffer_slots,
                "kv_read_bytes": result.kv_read.total,
                "hbm_bytes": result.hbm_bytes,
                "attention_macs": result.attention_macs,
                "sram_single_bytes": result.sram_single_bytes,
                "sram_double_bytes": result.sram_double_bytes,
                "latency_cycles": result.latency_cycles,
                "serial_cycles": result.serial_cycles,
                "double_buffer_cycles": result.double_buffer_cycles,
                "component_cycles": result.component_cycles,
                "pe_utilization": result.pe_utilization,
                "hbm_utilization": result.hbm_utilization,
                "input_pointer": step.source_pointer,
            }
        )
    feasible = all(r["feasible"] for r in rows)
    total = math.fsum(r["latency_cycles"] for r in rows) if feasible else None
    summary.update(
        mapping=asdict(mapping),
        trace_source=asdict(trajectory.source),
        evidence_kind=trajectory.evidence_kind,
        feasible=feasible,
        feasible_steps=sum(r["feasible"] for r in rows),
        infeasible_steps=sum(not r["feasible"] for r in rows),
        summed_trace_kv_read=sum(r["kv_read_bytes"] for r in rows),
        generation_latency_status="simulated_from_complete_per_step_trace"
        if feasible
        else "infeasible_steps",
        total_attention_latency_cycles=total,
        total_attention_latency_seconds=total / hw.clock_hz if feasible else None,
        mean_attention_latency_cycles=total / len(rows) if feasible else None,
        min_attention_latency_cycles=min(r["latency_cycles"] for r in rows) if feasible else None,
        max_attention_latency_cycles=max(r["latency_cycles"] for r in rows) if feasible else None,
        steps=rows,
    )
    return summary
