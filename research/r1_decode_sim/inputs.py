"""读取已锁定的精度与流量输入，不导入模型或学习实验代码。

本模块只适配输入，不建模时延。KV 每元素字节数是包含元数据的 K+V 平均值，
既不是独立的 K/V 位宽，也不是解码后的 SRAM 占用。仅读取四个明确指定的
L*/ppl_summary.json；历史根目录中的 4K 副本不计作额外测量。
所有输入路径均相对于 project_root 解析。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LENGTHS = (4096, 8192, 16384, 32768)
FORMATS = ("fp16", "int8", "int4", "int4_bdr", "kivi2", "kivi4")
PROTOCOL_IDS = tuple(f"C{i}" for i in range(6))
LAYOUTS = ("contiguous", "paged")
TRAFFIC_PATH = Path("research/r1_kv_baseline/experiments/kv_pareto/results/summary.json")
PPL_DIR = Path("research/r1_kv_baseline/experiments/wikitext_ppl/results/ppl")
_PARTS = ("payload", "scale", "zp", "page")


class InputValidationError(ValueError):
    """输入证据缺失、不一致、失败或违反协议时抛出的异常。"""


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class Geometry:
    num_kv_heads: int
    head_dim: int
    num_layers: int

    @property
    def n_elem(self) -> int:
        """单层中每个缓存 token 的 K+V 标量数，使用 GQA 的 KV 头数。"""
        return 2 * self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class ByteBreakdown:
    payload: int
    scale: int
    zp: int
    page: int

    @property
    def total(self) -> int:
        return self.payload + self.scale + self.zp + self.page


@dataclass(frozen=True)
class StepInput:
    model_id: str
    n: int
    protocol_id: str
    kv_format: str
    layout: str
    geometry: Geometry
    all_layers: ByteBreakdown
    b_eff: float
    source: SourceFile
    source_pointer: str

    @property
    def key(self) -> tuple[int, str, str]:
        return self.n, self.protocol_id, self.layout

    @property
    def bytes_per_token(self) -> int:
        """全模型 KV 读取字节数，已包含元数据。"""
        return self.all_layers.total

    @property
    def per_layer(self) -> ByteBreakdown:
        return ByteBreakdown(
            *(getattr(self.all_layers, p) // self.geometry.num_layers for p in _PARTS)
        )

    @property
    def kv_bytes_per_element(self) -> float:
        """K+V 合计的有效每元素字节数，仅适用于当前长度、格式和布局。

        乘以 N * geometry.n_elem 可得到单层字节数。不得再次加入元数据，
        也不能将此平均位宽用作逐分块或独立 K/V 的精确打包规则。
        """
        return self.b_eff / 8


@dataclass(frozen=True)
class PressureInput:
    l_in: int
    l_out: int
    total_kv_read: int
    last_step: StepInput

    @property
    def key(self) -> tuple[int, int, str, str]:
        return self.l_in, self.l_out, self.last_step.protocol_id, self.last_step.layout

    @property
    def mean_bytes_per_token(self) -> float:
        """整个生成过程的全模型平均流量，与末步流量分别记录。"""
        return self.total_kv_read / self.l_out

    @property
    def per_layer_total_kv_read(self) -> int:
        return self.total_kv_read // self.last_step.geometry.num_layers

    @property
    def per_layer_mean_bytes_per_token(self) -> float:
        return self.per_layer_total_kv_read / self.l_out


@dataclass(frozen=True)
class PplInput:
    model_id: str
    window: int
    protocol_id: str
    kv_format: str
    layout: str
    dataset: str
    split: str
    stride: int
    n_tokens: int
    n_windows: int
    seq_len: int
    nll_sum: float
    mean_nll: float
    ppl: float
    delta_vs_c0: float
    source: SourceFile
    source_pointer: str

    @property
    def key(self) -> tuple[str, int, str]:
        return self.model_id, self.window, self.protocol_id


@dataclass(frozen=True)
class StepWithQuality:
    traffic: StepInput
    quality: PplInput

    @property
    def quality_scope(self) -> str:
        return (
            "same_layout_window_ppl"
            if self.traffic.layout == self.quality.layout
            else "contiguous_reference_only"
        )


@dataclass(frozen=True)
class InputBundle:
    geometry: Geometry
    page_size: int
    pte_bytes: int
    steps: tuple[StepInput, ...]
    pressures: tuple[PressureInput, ...]
    quality: tuple[PplInput, ...]
    sources: tuple[SourceFile, ...]

    def step(self, n: int, protocol_id: str, layout: str) -> StepInput:
        for item in self.steps:
            if item.key == (n, protocol_id, layout):
                return item
        raise KeyError((n, protocol_id, layout))

    def pressure(self, l_in: int, l_out: int, protocol_id: str, layout: str) -> PressureInput:
        for item in self.pressures:
            if item.key == (l_in, l_out, protocol_id, layout):
                return item
        raise KeyError((l_in, l_out, protocol_id, layout))

    def ppl(self, model_id: str, window: int, protocol_id: str) -> PplInput:
        for item in self.quality:
            if item.key == (model_id, window, protocol_id):
                return item
        raise KeyError((model_id, window, protocol_id))

    def step_with_quality(self, n: int, protocol_id: str, layout: str) -> StepWithQuality:
        """关联窗口级 PPL，并明确保留其原始评测布局。

        这不是逐步 decode 的 PPL 测量。压力汇总没有对应的长生成质量测量，
        因此不会自动关联精度值。
        """
        traffic = self.step(n, protocol_id, layout)
        return StepWithQuality(traffic, self.ppl(traffic.model_id, n, protocol_id))


def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise InputValidationError(f"{where}: {message}")


def _object(value: Any, where: str) -> dict:
    _require(isinstance(value, dict), where, "expected a JSON object")
    return value


def _array(value: Any, where: str) -> list:
    _require(isinstance(value, list), where, "expected a JSON array")
    return value


def _integer(value: Any, where: str, minimum: int = 1) -> int:
    _require(type(value) is int and value >= minimum, where, f"expected integer >= {minimum}")
    return value


def _number(value: Any, where: str, minimum: float = 0) -> float:
    _require(type(value) in (int, float), where, "expected a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise InputValidationError(f"{where}: numeric overflow") from exc
    _require(math.isfinite(number) and number >= minimum, where, "invalid numeric value")
    return number


def _close(actual: float, expected: float, where: str) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
        where,
        f"inconsistent value {actual!r}; expected {expected!r}",
    )


def _fixed(record: dict, expected: dict, where: str) -> None:
    for name, value in expected.items():
        actual = record.get(name)
        _require(
            type(actual) is type(value) and actual == value,
            f"{where}/{name}",
            f"expected {value!r}, got {actual!r}",
        )


def _grid(keys: list, expected: set, where: str) -> None:
    _require(len(keys) == len(set(keys)), where, "duplicate row identifiers")
    _require(set(keys) == expected, where, "incomplete or unexpected grid identifiers")


def _read_json(root: Path, relative: Path) -> tuple[dict, SourceFile]:
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), str(relative), "input path leaves project_root")

    def no_duplicates(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            _require(key not in result, str(relative), f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise InputValidationError(f"{relative}: non-finite JSON constant {value}")

    try:
        raw = path.read_bytes()
        data = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"{relative}: cannot read input: {exc}") from exc
    return _object(data, str(relative)), SourceFile(
        path.relative_to(root).as_posix(), hashlib.sha256(raw).hexdigest(), len(raw)
    )


def _parse_step(
    row: dict, geometry: Geometry, source: SourceFile, pointer: str, prefix: str = ""
) -> StepInput:
    where = source.path + pointer
    row = _object(row, where)
    data = {k.removeprefix(prefix): v for k, v in row.items() if k.startswith(prefix)}
    n = _integer(data.get("n"), where + "/n")
    cid = data.get("protocol_id")
    _require(cid in PROTOCOL_IDS, where, "unknown protocol_id")
    fmt = FORMATS[PROTOCOL_IDS.index(cid)]
    _fixed(
        data,
        {"kv_format": fmt, "num_layers": geometry.num_layers, "n_elem": geometry.n_elem},
        where,
    )
    layout = data.get("layout")
    _require(layout in LAYOUTS, where, "unknown layout")
    parts = ByteBreakdown(*(_integer(data.get(p), where + "/" + p, 0) for p in _PARTS))
    total = _integer(data.get("bytes_per_token"), where + "/bytes_per_token")
    _require(total == parts.total, where, "byte total must equal payload + scale + zp + page")
    for name in _PARTS:
        value = getattr(parts, name)
        _require(value % geometry.num_layers == 0, where, "fractional per-layer byte count")
        per_layer = _integer(data.get("per_layer_" + name), where + "/per_layer_" + name, 0)
        _require(per_layer * geometry.num_layers == value, where, "per-layer bytes disagree")
    _require(
        parts.page == 0 if layout == "contiguous" else parts.page > 0,
        where,
        "invalid page metadata",
    )
    bits = _number(data.get("b_eff"), where + "/b_eff")
    _close(bits, 8 * total / (geometry.num_layers * n * geometry.n_elem), where + "/b_eff")
    return StepInput(MODEL_ID, n, cid, fmt, layout, geometry, parts, bits, source, pointer)


def _parse_traffic(data: dict, source: SourceFile) -> tuple[Geometry, tuple, tuple]:
    where = source.path
    _fixed(
        data,
        {
            "model_id": MODEL_ID,
            "page_size": 16,
            "pte_bytes": 8,
            "c0_path": "fp16_codec",
            "b_pad_included": False,
        },
        where,
    )
    geo = _object(data.get("geometry"), where + "/geometry")
    _fixed(geo, {"num_kv_heads": 8, "head_dim": 128, "num_layers": 32}, where + "/geometry")
    geometry = Geometry(geo["num_kv_heads"], geo["head_dim"], geo["num_layers"])
    for name, values in (("lengths", LENGTHS), ("formats", FORMATS), ("layouts", LAYOUTS)):
        array = _array(data.get(name), where + "/" + name)
        _require(all(type(v) is type(values[0]) for v in array), where, f"invalid {name} types")
        _grid(array, set(values), where + "/" + name)
    pressure = _object(data.get("pressure"), where + "/pressure")
    _fixed(pressure, {"l_in": 16384, "l_out": 1024}, where + "/pressure")
    steps = tuple(
        _parse_step(row, geometry, source, f"/steps/{i}")
        for i, row in enumerate(_array(data.get("steps"), where + "/steps"))
    )
    _grid([s.key for s in steps], set(product(LENGTHS, PROTOCOL_IDS, LAYOUTS)), where + "/steps")
    pressures = []
    for i, row in enumerate(_array(data.get("pressures"), where + "/pressures")):
        pointer = f"/pressures/{i}"
        loc = where + pointer
        row = _object(row, loc)
        _fixed(row, {"l_in": 16384, "l_out": 1024, "num_layers": geometry.num_layers}, loc)
        last = _parse_step(row, geometry, source, pointer, "last_")
        _fixed(
            row,
            {"protocol_id": last.protocol_id, "kv_format": last.kv_format, "layout": last.layout},
            loc,
        )
        _require(last.n == row["l_in"] + row["l_out"] - 1, loc, "wrong last-step N")
        total = _integer(row.get("total_kv_read"), loc + "/total_kv_read")
        _require(total % geometry.num_layers == 0, loc, "fractional per-layer pressure bytes")
        _close(
            _number(row.get("mean_bytes_per_token"), loc),
            total / row["l_out"],
            loc + "/mean_bytes_per_token",
        )
        pressures.append(PressureInput(row["l_in"], row["l_out"], total, last))
    _grid(
        [p.key for p in pressures],
        set(product((16384,), (1024,), PROTOCOL_IDS, LAYOUTS)),
        where + "/pressures",
    )
    return (
        geometry,
        tuple(sorted(steps, key=lambda s: s.key)),
        tuple(sorted(pressures, key=lambda p: p.key)),
    )


def _parse_ppl(data: dict, source: SourceFile, window: int) -> tuple[PplInput, ...]:
    where = source.path
    _fixed(
        data,
        {
            "model_id": MODEL_ID,
            "max_length": window,
            "layout": "contiguous",
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
            "split": "test",
            "stage": "B",
            "stride": 512,
            "limit_tokens": None,
            "group_size": 32,
            "residual_length": 128,
            "c0_is_codec": True,
            "ok": True,
        },
        where,
    )
    results = []
    for i, row in enumerate(_array(data.get("rows"), where + "/rows")):
        pointer = f"/rows/{i}"
        loc = where + pointer
        row = _object(row, loc)
        _fixed(row, {"ok": True}, loc)
        fmt = row.get("resolved")
        _require(fmt in FORMATS, loc, "unknown resolved KV format")
        cid = PROTOCOL_IDS[FORMATS.index(fmt)]
        # 历史 4K C0 行的 cid 标为 fp16，但实际运行入口是 c0。
        _require(
            row.get("cid") in (("C0", "fp16") if cid == "C0" else (cid,)),
            loc,
            "cid disagrees with resolved format",
        )
        _require(
            row.get("kv_format") in (("c0", "fp16_codec") if cid == "C0" else (fmt,)),
            loc,
            "wrong codec entry (native HF fp16 is not C0)",
        )
        for field in ("model_id", "max_length", "stride", "layout"):
            if field in row:
                _fixed(row, {field: data[field]}, loc)
        tokens = _integer(row.get("n_tokens"), loc + "/n_tokens")
        n_windows = _integer(row.get("n_windows"), loc + "/n_windows")
        seq_len = _integer(row.get("seq_len"), loc + "/seq_len")
        _require(tokens == seq_len - 1, loc, "full-corpus scored token coverage disagrees")
        nll = _number(row.get("nll_sum"), loc + "/nll_sum")
        mean = _number(row.get("mean_nll"), loc + "/mean_nll")
        ppl = _number(row.get("ppl"), loc + "/ppl", 1)
        _close(mean, nll / tokens, loc + "/mean_nll")
        _close(math.log(ppl), mean, loc + "/ppl")
        delta = _number(row.get("delta_vs_c0"), loc + "/delta_vs_c0", -math.inf)
        results.append(
            PplInput(
                MODEL_ID,
                window,
                cid,
                fmt,
                data["layout"],
                data["dataset"],
                data["split"],
                data["stride"],
                tokens,
                n_windows,
                seq_len,
                nll,
                mean,
                ppl,
                delta,
                source,
                pointer,
            )
        )
    _grid(
        [r.key for r in results],
        set(product((MODEL_ID,), (window,), PROTOCOL_IDS)),
        where + "/rows",
    )
    baseline = next(r for r in results if r.protocol_id == "C0")
    for result in results:
        _require(
            (result.n_tokens, result.n_windows, result.seq_len)
            == (baseline.n_tokens, baseline.n_windows, baseline.seq_len),
            where,
            "unequal PPL coverage across formats",
        )
        _close(
            result.delta_vs_c0,
            result.ppl - baseline.ppl,
            where + result.source_pointer + "/delta_vs_c0",
        )
    return tuple(results)


def load_inputs(
    project_root: str | Path | None = None,
    *,
    traffic_path: str | Path = TRAFFIC_PATH,
    ppl_dir: str | Path = PPL_DIR,
) -> InputBundle:
    """加载并校验已锁定的完整 48/12/24 输入网格，不跳过无效行。

    本地研究结果缺失时，抛出包含路径的 InputValidationError。
    不访问网络、不使用 GPU、不构建缓存，也不修改原始输入。
    SHA-256 标识本次读取的准确字节，不代表历史执行时的源码。
    """
    root = (
        Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    ).resolve()
    traffic, source = _read_json(root, Path(traffic_path))
    geometry, steps, pressures = _parse_traffic(traffic, source)
    sources = [source]
    quality = []
    for window in LENGTHS:
        data, source = _read_json(root, Path(ppl_dir) / f"L{window}" / "ppl_summary.json")
        quality.extend(_parse_ppl(data, source, window))
        sources.append(source)
    _require(
        len({(q.n_tokens, q.seq_len) for q in quality}) == 1,
        "PPL",
        "unequal corpus coverage across windows",
    )
    return InputBundle(
        geometry,
        traffic["page_size"],
        traffic["pte_bytes"],
        steps,
        pressures,
        tuple(sorted(quality, key=lambda q: q.key)),
        tuple(sources),
    )
