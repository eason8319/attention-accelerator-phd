"""分页与残差路径的访存事件、事务量子和页数公式。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from research.r2_streaming_attention.physical_pack import align_up

ROOT = Path(__file__).resolve().parents[2]
PAGE_ACCESS_PATH = (
    ROOT / "research" / "r2_streaming_attention" / "experiments" / "configs" / "page_access.json"
)


@dataclass(frozen=True)
class PageAccessConfig:
    """步骤 4 冻结的分页、残差与事务参数。"""

    layout_id: str
    page_tokens: int
    pte_bytes: int
    tag_bytes: int
    residual_length: int
    dma_align_bytes: int
    default_metadata_placement: str
    packed_history_level: str
    residual_level: str
    tail_level: str
    pte_level: str
    tag_level: str


def load_page_access(path: Path | str | None = None) -> PageAccessConfig:
    """读取分页访存配置；拒绝未约定的元数据放置或页长。"""
    raw = json.loads((path or PAGE_ACCESS_PATH).read_text(encoding="utf-8"))
    if int(raw["page_tokens"]) <= 0:
        raise ValueError("page_tokens 须为正")
    if int(raw["pte_bytes"]) <= 0 or int(raw["tag_bytes"]) <= 0:
        raise ValueError("pte_bytes 与 tag_bytes 须为正")
    allowed = set(raw["metadata_placement"]["allowed"])
    default = raw["metadata_placement"]["default"]
    if default not in allowed:
        raise ValueError(f"default 元数据放置 {default!r} 不在 {sorted(allowed)}")
    group = 32
    page = int(raw["page_tokens"])
    residual = int(raw["residual_length"])
    if group % page != 0 or residual % page != 0 or residual % group != 0:
        raise ValueError("须满足 page | group、page | residual、group | residual")
    residency = raw["residency"]
    return PageAccessConfig(
        layout_id=raw["layout_id"],
        page_tokens=page,
        pte_bytes=int(raw["pte_bytes"]),
        tag_bytes=int(raw["tag_bytes"]),
        residual_length=residual,
        dma_align_bytes=int(raw["dma"]["align_bytes"]),
        default_metadata_placement=default,
        packed_history_level=residency["packed_history"],
        residual_level=residency["residual_fp16"],
        tail_level=residency["tail_fp16"],
        pte_level=residency["pte"],
        tag_level=residency["tags"],
    )


def n_pages_for_tokens(n_tokens: int, page_size: int) -> int:
    """空池为 0，否则 ``ceil(n / page_size)``。"""
    if n_tokens < 0:
        raise ValueError(f"token 数不能为负，得到 {n_tokens}")
    if n_tokens == 0:
        return 0
    return math.ceil(n_tokens / page_size)


def kivi_pool_tokens(n_tokens: int, residual_length: int) -> dict[str, int]:
    """R1 §8.4 的四池 token 数。"""
    if n_tokens < 0:
        raise ValueError(f"token 数不能为负，得到 {n_tokens}")
    return {
        "k_quant": (n_tokens // residual_length) * residual_length,
        "k_res": n_tokens % residual_length,
        "v_quant": max(n_tokens - residual_length, 0),
        "v_res": min(n_tokens, residual_length),
    }


def expected_page_counts(
    format_id: str,
    cache_layout: str,
    n_tokens: int,
    *,
    page_tokens: int,
    residual_length: int,
) -> dict[str, int]:
    """由公式给出的已分配页数；contiguous 全部为 0。"""
    key = format_id.strip().upper()
    if cache_layout == "contiguous":
        if key == "C5":
            return {"k_quant": 0, "k_res": 0, "v_quant": 0, "v_res": 0}
        return {"k": 0, "v": 0}
    if cache_layout != "paged":
        raise ValueError(cache_layout)
    if key == "C5":
        pools = kivi_pool_tokens(n_tokens, residual_length)
        return {name: n_pages_for_tokens(tokens, page_tokens) for name, tokens in pools.items()}
    n_pages = n_pages_for_tokens(n_tokens, page_tokens)
    return {"k": n_pages, "v": n_pages}


def physical_bytes(logical: int, *, level: str, dma_align: int) -> int:
    """HBM 事务按 DMA 量子取整；SRAM 事件记逻辑字节，不声称 bank 对齐。"""
    if logical < 0:
        raise ValueError(f"逻辑字节不能为负，得到 {logical}")
    if level == "hbm":
        return align_up(logical, dma_align)
    if level == "control":
        return 0
    return logical


@dataclass
class AccessEvent:
    """一次分层访存或控制事件。"""

    action: str
    field: str
    side: str
    pool: str
    level: str
    tokens: int
    logical_bytes: int
    physical_bytes: int
    seq_len: int


@dataclass
class EventLog:
    """追加期间的事件序列；完整列表默认不写入实验结果。"""

    dma_align_bytes: int
    events: list[AccessEvent] = field(default_factory=list)

    def record(
        self,
        *,
        action: str,
        field: str,
        side: str,
        pool: str,
        level: str,
        tokens: int,
        logical_bytes: int,
        seq_len: int,
    ) -> None:
        """追加一条事件；物理字节由层级与 DMA 量子导出。"""
        if logical_bytes < 0 or tokens < 0:
            raise ValueError("tokens 与 logical_bytes 不能为负")
        self.events.append(
            AccessEvent(
                action=action,
                field=field,
                side=side,
                pool=pool,
                level=level,
                tokens=tokens,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes(
                    logical_bytes, level=level, dma_align=self.dma_align_bytes
                ),
                seq_len=seq_len,
            )
        )

    def totals(self) -> dict:
        """可 JSON 化的累计；按 action/field/level 分解。"""
        by_action: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "logical_bytes": 0, "physical_bytes": 0, "tokens": 0}
        )
        by_field: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "logical_bytes": 0, "physical_bytes": 0}
        )
        by_level: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "logical_bytes": 0, "physical_bytes": 0}
        )
        for event in self.events:
            for bucket, key in (
                (by_action, event.action),
                (by_field, event.field),
                (by_level, event.level),
            ):
                slot = bucket[key]
                slot["n"] += 1
                slot["logical_bytes"] += event.logical_bytes
                slot["physical_bytes"] += event.physical_bytes
                if bucket is by_action:
                    slot["tokens"] += event.tokens
        return {
            "n_events": len(self.events),
            "logical_bytes": sum(e.logical_bytes for e in self.events if e.level != "control"),
            "physical_bytes": sum(e.physical_bytes for e in self.events),
            "control_covered_bytes": sum(e.logical_bytes for e in self.events if e.level == "control"),
            "hbm_logical_bytes": sum(e.logical_bytes for e in self.events if e.level == "hbm"),
            "hbm_physical_bytes": sum(e.physical_bytes for e in self.events if e.level == "hbm"),
            "sram_logical_bytes": sum(e.logical_bytes for e in self.events if e.level == "sram"),
            "by_action": dict(by_action),
            "by_field": dict(by_field),
            "by_level": dict(by_level),
        }

    def dump_events(self) -> list[dict]:
        """逐步轨迹需要时才物化事件列表。"""
        return [asdict(event) for event in self.events]
