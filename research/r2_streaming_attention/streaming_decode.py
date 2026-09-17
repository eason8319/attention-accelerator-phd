"""从物理页供给单个 KV head 的小块操作数，不构造全头 KV tile。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research.r2_streaming_attention.paged_kv import PackedPhysicalCache


@dataclass
class OperandSpan:
    """连续逻辑 token 与其物理页、元数据或残差视图。"""

    start: int
    end: int
    page: object | None
    residual: torch.Tensor | None
    metadata: object | None


class PackedOperandReader:
    """单次最多十六个 token、一个 KV head；K 和 V 分两遍读取。"""

    operand_tokens = 16

    def __init__(self, cache: PackedPhysicalCache, *, c3_inverse_min_rows: int = 1):
        if c3_inverse_min_rows not in {1, 4}:
            raise ValueError("C3 原域逆旋转的最小行数须为 1 或 4")
        self.cache = cache
        self.c3_inverse_min_rows = c3_inverse_min_rows
        self.inverse_padded_token_heads = 0
        self.inverse_padding_max_bytes = 0
        self.spans = {side: self._spans(side) for side in ("k", "v")}
        self.max_operand_bytes = 0
        self.max_operand_tokens = 0
        self.read_calls = 0
        self.record_events = False

    def _spans(self, side: str) -> list[OperandSpan]:
        cache = self.cache
        if cache.format_id == "C5":
            pages = getattr(cache, f"_{side}_quant")
            residuals = getattr(cache, f"_{side}_res")
        else:
            pages = getattr(cache, f"_{side}_pages")
            tail = getattr(cache, f"_{side}_tail")
            residuals = [] if tail is None else [tail]
        spans = []
        cursor = 0
        metadata = None
        for page in pages:
            if page.scale is not None:
                metadata = page
            spans.append(OperandSpan(cursor, cursor + page.n_tokens, page, None, metadata))
            cursor += page.n_tokens
        for residual in residuals:
            spans.append(OperandSpan(cursor, cursor + len(residual), None, residual, None))
            cursor += len(residual)
        if cursor != len(cache):
            raise RuntimeError("操作数地址映射未覆盖完整缓存")
        return spans

    def _page_operand(self, span: OperandSpan, start: int, end: int, head: int):
        cache = self.cache
        page = span.page
        dim, heads = cache.head_dim, cache.num_heads
        a, b = start - span.start, end - span.start
        if cache.format_id == "C0":
            # reshape 只创建字节视图，选中单头后才产生高精度操作数。
            raw = page.payload[: page.n_tokens * heads * dim * 2].view(torch.float16)
            return raw.reshape(page.n_tokens, heads, dim)[a:b, head].float()
        if cache.format_id == "C1":
            raw = page.payload[: page.n_tokens * heads * dim].view(torch.int8)
            grid = raw.reshape(page.n_tokens, heads, dim)[a:b, head].float()
        else:
            raw = page.payload[: page.n_tokens * heads * dim // 2]
            packed = raw.reshape(page.n_tokens, heads, dim // 2)[a:b, head]
            lanes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            if cache.format_id in {"C2", "C3"}:
                if bool((lanes == 8).any()):
                    raise ValueError("C2/C3 载荷包含禁止的 -8 码点")
                grid = torch.where(lanes >= 8, lanes.to(torch.int16) - 16, lanes).float()
            else:
                grid = lanes.float()
        meta = span.metadata
        if meta is None or meta.scale is None:
            raise RuntimeError("量化操作数缺少尺度")
        if cache.format_id == "C5" and self._side == "k":
            if cache.cache_layout == "paged":
                scale = meta.scale[0, head].float()
                offset = meta.offset[0, head].float()
            else:
                groups = torch.arange(a, b, device=grid.device) // cache.pack_layout.group_size
                scale = meta.scale[groups, head].float()
                offset = meta.offset[groups, head].float()
            return grid * scale + offset
        group = dim if cache.format_id == "C1" else cache.pack_layout.group_size
        scale = meta.scale[a:b, head].float().reshape(b - a, dim // group, 1)
        grouped = grid.reshape(b - a, dim // group, group)
        if meta.offset is None:
            return (grouped * scale).reshape(b - a, dim)
        offset = meta.offset[a:b, head].float().reshape(b - a, dim // group, 1)
        if cache.format_id == "C5":
            return (grouped * scale + offset).reshape(b - a, dim)
        return ((grouped - offset) * scale).reshape(b - a, dim)

    def read(self, side: str, start: int, end: int, head: int, *, domain: str):
        """读取单侧、单头操作数；尾页与残差维持已有 FP16 存储语义。"""
        if side not in self.spans or not (0 <= start < end <= len(self.cache)):
            raise ValueError("无效的操作数区间")
        if end - start > self.operand_tokens or not 0 <= head < self.cache.num_heads:
            raise ValueError("操作数超过单头十六 token 供给上界")
        if domain not in {"original", "rotated"}:
            raise ValueError("未知坐标域")
        if domain == "rotated" and self.cache.format_id != "C3":
            raise ValueError("仅 C3 具有旋转坐标域")
        self._side = side
        pieces = []
        for span in self.spans[side]:
            a, b = max(start, span.start), min(end, span.end)
            if a >= b:
                continue
            if span.residual is not None:
                part = span.residual[a - span.start : b - span.start, head].float()
                if self.cache.format_id == "C3" and domain == "rotated":
                    part = self.cache._codec._rot.rotate(part)
            else:
                part = self._page_operand(span, a, b, head)
                if self.cache.format_id == "C3" and domain == "original":
                    count = len(part)
                    if count < self.c3_inverse_min_rows:
                        padded = part.new_zeros(self.c3_inverse_min_rows, self.cache.head_dim)
                        padded[:count] = part
                        self.inverse_padded_token_heads += self.c3_inverse_min_rows - count
                        self.inverse_padding_max_bytes = max(
                            self.inverse_padding_max_bytes, padded.numel() * padded.element_size()
                        )
                        part = self.cache._codec._rot.inverse(padded)[:count]
                        del padded
                    else:
                        part = self.cache._codec._rot.inverse(part)
            if self.record_events:
                count, dim = b - a, self.cache.head_dim
                residual = span.residual is not None
                bits = (
                    16
                    if residual or self.cache.format_id == "C0"
                    else (8 if self.cache.format_id == "C1" else 4)
                )
                meta_bytes = 0
                if not residual and bits != 16:
                    if self.cache.format_id == "C5" and side == "k":
                        groups = (b - 1) // 32 - a // 32 + 1
                        meta_bytes = groups * dim * 4
                    else:
                        meta_bytes = count * (1 if bits == 8 else dim // 32) * 2
                        if span.metadata.offset is not None:
                            meta_bytes *= 2
                pool = (
                    f"{side}_res"
                    if residual and self.cache.format_id == "C5"
                    else (
                        "tail"
                        if residual
                        else (f"{side}_quant" if self.cache.format_id == "C5" else "uniform")
                    )
                )
                self.cache._record_stream_read(
                    kind="fp16" if residual else "packed",
                    side=side,
                    pool=pool,
                    field="fp16_residual" if residual else "payload",
                    tokens=count,
                    logical_bytes=count * dim * bits // 8,
                    page=span.page,
                    meta_bytes=meta_bytes,
                )
            pieces.append(part)
        operand = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if operand.shape != (end - start, self.cache.head_dim):
            raise RuntimeError("操作数解码形状不符")
        if not bool(torch.isfinite(operand).all()):
            raise RuntimeError("解码操作数包含非有限值")
        self.max_operand_bytes = max(
            self.max_operand_bytes, operand.numel() * operand.element_size()
        )
        self.max_operand_tokens = max(self.max_operand_tokens, end - start)
        self.read_calls += 1
        return operand
