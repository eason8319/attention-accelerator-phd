"""Paged KV Cache：按 ``metrics.md`` §8 切页存储（均匀 C0–C3 与 KIVI C4/C5）。

存储布局与页表记账；读侧仍 ``load`` 全量，不实现 page-wise partial attention。
"""

from __future__ import annotations

import math

import torch
from kv_cache import BytesBreakdown, _validate_kv_append, scale_zp_nbytes
from kv_codecs import EncodedKV, KiviKeyCodec, KiviValueCodec, KVCodec

# 与 protocols/metrics.md §8.1 默认值一致
DEFAULT_PAGE_SIZE = 16
DEFAULT_PTE_BYTES = 8


def n_pages_for_tokens(n_tokens: int, page_size: int) -> int:
    """空池 0，否则 ``ceil(n / page_size)``。"""
    if n_tokens <= 0:
        return 0
    return math.ceil(n_tokens / page_size)


def _fp16_nbytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def _repage_fp16(x: torch.Tensor, page_size: int) -> list[torch.Tensor]:
    """沿 token 维切成不超过 ``page_size`` 的 FP16 页；空则 ``[]``。"""
    if x.shape[0] == 0:
        return []
    return [x[i : i + page_size].contiguous() for i in range(0, x.shape[0], page_size)]


def _fp16_len(pages: list[torch.Tensor]) -> int:
    return sum(int(p.shape[0]) for p in pages)


def _append_fp16_pages(
    pages: list[torch.Tensor],
    x: torch.Tensor,
    page_size: int,
) -> None:
    """把 ``x`` 追加进 FP16 页列表，先填满末页再开新页。"""
    if x.shape[0] == 0:
        return
    if pages and pages[-1].shape[0] < page_size:
        need = page_size - pages[-1].shape[0]
        take = min(need, x.shape[0])
        pages[-1] = torch.cat([pages[-1], x[:take]], dim=0)
        x = x[take:]
    for i in range(0, x.shape[0], page_size):
        pages.append(x[i : i + page_size].contiguous())


def _slice_encoded_t(encoded: EncodedKV, start: int, end: int) -> EncodedKV:
    """沿 token 维切片；用于 Value（scale/zp 与 payload 同长）。"""
    scale = None if encoded.scale is None else encoded.scale[start:end]
    zp = None if encoded.zero_point is None else encoded.zero_point[start:end]
    return EncodedKV(payload=encoded.payload[start:end], scale=scale, zero_point=zp)


def _cat_encoded_t(left: EncodedKV, right: EncodedKV) -> EncodedKV:
    """沿 token 维拼接两段 EncodedKV（Value 页合并）。"""

    def _cat_opt(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
        if a is None and b is None:
            return None
        if a is None or b is None:
            raise ValueError("scale/zp 不能只在一侧存在")
        return torch.cat([a, b], dim=0)

    return EncodedKV(
        payload=torch.cat([left.payload, right.payload], dim=0),
        scale=_cat_opt(left.scale, right.scale),
        zero_point=_cat_opt(left.zero_point, right.zero_point),
    )


def _split_kivi_key_pages(
    encoded: EncodedKV,
    *,
    page_size: int,
    group_size: int,
) -> list[EncodedKV]:
    """把一次 Key encode 切成页；每 group 的 scale/zp 只挂在该 group 的第一页。"""
    if encoded.scale is None or encoded.zero_point is None:
        raise ValueError("KIVI Key 切页需要 scale 与 mn")
    t = encoded.payload.shape[0]
    if t % group_size != 0:
        raise ValueError(f"Key payload T={t} 不能被 group_size={group_size} 整除")
    if group_size % page_size != 0:
        raise ValueError(f"group_size={group_size} 不能被 page_size={page_size} 整除")
    pages_per_group = group_size // page_size
    n_groups = t // group_size
    pages: list[EncodedKV] = []
    for g in range(n_groups):
        t0 = g * group_size
        scale_g = encoded.scale[g : g + 1]
        zp_g = encoded.zero_point[g : g + 1]
        for p in range(pages_per_group):
            s = t0 + p * page_size
            sl = encoded.payload[s : s + page_size]
            if p == 0:
                pages.append(EncodedKV(payload=sl, scale=scale_g, zero_point=zp_g))
            else:
                pages.append(EncodedKV(payload=sl))
    return pages


def _decode_kivi_key_pages(
    pages: list[EncodedKV],
    codec: KiviKeyCodec,
    *,
    page_size: int,
    group_size: int,
) -> torch.Tensor | None:
    """按 group 重组两页再 decode。"""
    if not pages:
        return None
    pages_per_group = group_size // page_size
    if len(pages) % pages_per_group != 0:
        raise RuntimeError(
            f"Key 量化页数 {len(pages)} 不能被 pages_per_group={pages_per_group} 整除"
        )
    parts: list[torch.Tensor] = []
    for i in range(0, len(pages), pages_per_group):
        group = pages[i : i + pages_per_group]
        payload = torch.cat([p.payload for p in group], dim=0)
        meta = group[0]
        if meta.scale is None or meta.zero_point is None:
            raise RuntimeError("Key group 第一页缺少 scale/mn")
        parts.append(
            codec.decode(EncodedKV(payload=payload, scale=meta.scale, zero_point=meta.zero_point))
        )
    return torch.cat(parts, dim=0)


def _split_along_t(encoded: EncodedKV, page_size: int) -> list[EncodedKV]:
    """沿 token 维按页切开（Value 量化：每页自带对应 scale/zp 切片）。"""
    t = encoded.payload.shape[0]
    return [_slice_encoded_t(encoded, s, min(s + page_size, t)) for s in range(0, t, page_size)]


def _extend_encoded_pages(
    pages: list[EncodedKV],
    incoming: EncodedKV,
    page_size: int,
) -> None:
    """把新量化段并入页列表：先填满末尾不满页，再按 ``page_size`` 切开。"""
    new_pages = _split_along_t(incoming, page_size)
    for nxt in new_pages:
        if pages and pages[-1].payload.shape[0] < page_size:
            room = page_size - pages[-1].payload.shape[0]
            take = min(room, nxt.payload.shape[0])
            pages[-1] = _cat_encoded_t(pages[-1], _slice_encoded_t(nxt, 0, take))
            if nxt.payload.shape[0] > take:
                pages.append(_slice_encoded_t(nxt, take, nxt.payload.shape[0]))
        else:
            pages.append(nxt)


def _check_page_invariants(
    *,
    page_size: int,
    group_size: int,
    residual_length: int,
) -> None:
    """检查整除约束 P | g、P | R、g | R，参见 metrics.md §8.1。"""
    if page_size <= 0:
        raise ValueError(f"page_size 须为正，得到 {page_size}")
    if group_size % page_size != 0:
        raise ValueError(f"page_size={page_size} 须整除 group_size={group_size}")
    if residual_length % page_size != 0:
        raise ValueError(f"page_size={page_size} 须整除 residual_length={residual_length}")
    if residual_length % group_size != 0:
        raise ValueError(f"residual_length={residual_length} 须能被 group_size={group_size} 整除")


class PagedUniformKVCache:
    """C0–C3 的 paged 后端：每页独立 encode；尾页只含占用 token。"""

    def __init__(
        self,
        codec: KVCodec,
        *,
        num_heads: int,
        head_dim: int,
        page_size: int = DEFAULT_PAGE_SIZE,
        pte_bytes: int = DEFAULT_PTE_BYTES,
        device: torch.device | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size 须为正，得到 {page_size}")
        if pte_bytes <= 0:
            raise ValueError(f"pte_bytes 须为正，得到 {pte_bytes}")
        self.codec = codec
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.pte_bytes = pte_bytes
        self.device = device or torch.device("cpu")
        self._k_pages: list[EncodedKV] = []
        self._v_pages: list[EncodedKV] = []
        # 未满页的 float 暂存；满页才 encode，避免为凑页改统计量
        self._k_tail: torch.Tensor | None = None
        self._v_tail: torch.Tensor | None = None
        self._seq_len: int = 0

    def __len__(self) -> int:
        return self._seq_len

    def page_counts(self) -> dict[str, int]:
        """K/V 两池已分配页数（含尾页）。"""
        n = n_pages_for_tokens(self._seq_len, self.page_size)
        return {"k": n, "v": n}

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """写入 float K/V；满页立即 encode，余数留在尾缓冲。"""
        n_tokens = _validate_kv_append(k_t, v_t, num_heads=self.num_heads, head_dim=self.head_dim)
        k_t = k_t.to(device=self.device)
        v_t = v_t.to(device=self.device)
        self._k_tail = self._commit_full_pages(self._k_pages, self._k_tail, k_t)
        self._v_tail = self._commit_full_pages(self._v_pages, self._v_tail, v_t)
        self._seq_len += n_tokens

    def _commit_full_pages(
        self,
        pages: list[EncodedKV],
        tail: torch.Tensor | None,
        new: torch.Tensor,
    ) -> torch.Tensor | None:
        buf = new if tail is None else torch.cat([tail, new], dim=0)
        while buf.shape[0] >= self.page_size:
            pages.append(self.codec.encode(buf[: self.page_size]))
            buf = buf[self.page_size :]
        if buf.shape[0] == 0:
            return None
        return buf

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """解码全部页（含尾页），返回 float32 ``(seq_len, H, D)``。"""
        empty = (0, self.num_heads, self.head_dim)
        if self._seq_len == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone()
        k = self._decode_side(self._k_pages, self._k_tail)
        v = self._decode_side(self._v_pages, self._v_tail)
        if k.shape[0] != self._seq_len or v.shape[0] != self._seq_len:
            raise RuntimeError(
                f"解码长度与 _seq_len 不一致: k={k.shape[0]}, v={v.shape[0]}, "
                f"seq_len={self._seq_len}"
            )
        return k, v

    def _decode_side(
        self,
        pages: list[EncodedKV],
        tail: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [self.codec.decode(p) for p in pages]
        if tail is not None:
            parts.append(self.codec.decode(self.codec.encode(tail)))
        if not parts:
            raise RuntimeError("非空 cache 却缺少解码片段")
        return torch.cat(parts, dim=0)

    def clear(self) -> None:
        """清空已提交页与尾缓冲。"""
        self._k_pages.clear()
        self._v_pages.clear()
        self._k_tail = None
        self._v_tail = None
        self._seq_len = 0

    def bytes_breakdown(self) -> BytesBreakdown:
        """占用 token 的 payload/scale/zp + ``B_page = n_pages * pte_bytes``。"""
        payload = 0
        scale = 0
        zp = 0
        for pages, tail in ((self._k_pages, self._k_tail), (self._v_pages, self._v_tail)):
            for enc in pages:
                payload += self.codec.bytes_payload(enc)
                s, z = scale_zp_nbytes(enc)
                scale += s
                zp += z
            if tail is not None:
                enc = self.codec.encode(tail)
                payload += self.codec.bytes_payload(enc)
                s, z = scale_zp_nbytes(enc)
                scale += s
                zp += z
        n_pages = sum(self.page_counts().values())
        return BytesBreakdown(payload=payload, scale=scale, zp=zp, page=n_pages * self.pte_bytes)

    def bytes_stored(self) -> tuple[int, int]:
        """返回 ``(payload_bytes, metadata_bytes)``；metadata 含 page。"""
        b = self.bytes_breakdown()
        return b.payload, b.metadata


class PagedKiviKVCache:
    """C4/C5 的 paged 后端：量化历史与 FP16 残差分四池，禁止混页。

    刷窗语义与 ``KiviKVCache`` 相同；Key group（默认 32 token）跨连续 2 页，
    scale/mn 只存一份并挂在 group 第一页。
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        bits: int = 2,
        k_bits: int | None = None,
        v_bits: int | None = None,
        group_size: int = 32,
        residual_length: int = 128,
        page_size: int = DEFAULT_PAGE_SIZE,
        pte_bytes: int = DEFAULT_PTE_BYTES,
        device: torch.device | None = None,
    ) -> None:
        k_bits = bits if k_bits is None else k_bits
        v_bits = bits if v_bits is None else v_bits
        _check_page_invariants(
            page_size=page_size, group_size=group_size, residual_length=residual_length
        )
        if pte_bytes <= 0:
            raise ValueError(f"pte_bytes 须为正，得到 {pte_bytes}")
        if residual_length <= 0:
            raise ValueError(f"residual_length 须为正，得到 {residual_length}")
        if head_dim % group_size != 0:
            raise ValueError(
                f"head_dim={head_dim} 须能被 group_size={group_size} 整除（Value 分组）"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.residual_length = residual_length
        self.page_size = page_size
        self.pte_bytes = pte_bytes
        self.device = device or torch.device("cpu")
        self.k_codec = KiviKeyCodec(bits=k_bits, group_size=group_size)
        self.v_codec = KiviValueCodec(bits=v_bits, group_size=group_size)

        self._k_quant_pages: list[EncodedKV] = []
        self._v_quant_pages: list[EncodedKV] = []
        self._k_residual_pages: list[torch.Tensor] = []
        self._v_residual_pages: list[torch.Tensor] = []
        self._seq_len: int = 0

    def __len__(self) -> int:
        return self._seq_len

    @property
    def k_residual_len(self) -> int:
        """当前 Key 残差窗 token 数。"""
        return _fp16_len(self._k_residual_pages)

    @property
    def v_residual_len(self) -> int:
        """当前 Value 残差窗 token 数。"""
        return _fp16_len(self._v_residual_pages)

    def page_counts(self) -> dict[str, int]:
        """四池已分配页数。"""
        return {
            "k_quant": len(self._k_quant_pages),
            "k_res": len(self._k_residual_pages),
            "v_quant": len(self._v_quant_pages),
            "v_res": len(self._v_residual_pages),
        }

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """追加 K/V，按 KIVI 规则刷窗后写入对应页池。"""
        n_tokens = _validate_kv_append(k_t, v_t, num_heads=self.num_heads, head_dim=self.head_dim)
        k_t = k_t.to(device=self.device, dtype=torch.float16)
        v_t = v_t.to(device=self.device, dtype=torch.float16)
        _append_fp16_pages(self._k_residual_pages, k_t, self.page_size)
        _append_fp16_pages(self._v_residual_pages, v_t, self.page_size)
        self._seq_len += n_tokens
        self._flush_key_residual()
        self._flush_value_residual()

    def _flush_key_residual(self) -> None:
        """Key：刷出 ``residual_length`` 的整数倍完整窗，再按 group 切页。"""
        n = _fp16_len(self._k_residual_pages)
        n_flush = (n // self.residual_length) * self.residual_length
        if n_flush <= 0:
            return
        residual = torch.cat(self._k_residual_pages, dim=0)
        encoded = self.k_codec.encode(residual[:n_flush].float())
        self._k_quant_pages.extend(
            _split_kivi_key_pages(encoded, page_size=self.page_size, group_size=self.group_size)
        )
        rest = residual[n_flush:]
        self._k_residual_pages = _repage_fp16(rest, self.page_size)

    def _flush_value_residual(self) -> None:
        """Value：溢出段立即量化并入历史页；不得为凑页推迟刷写。"""
        n = _fp16_len(self._v_residual_pages)
        if n <= self.residual_length:
            return
        n_overflow = n - self.residual_length
        residual = torch.cat(self._v_residual_pages, dim=0)
        encoded = self.v_codec.encode(residual[:n_overflow].float())
        _extend_encoded_pages(self._v_quant_pages, encoded, self.page_size)
        rest = residual[n_overflow:]
        self._v_residual_pages = _repage_fp16(rest, self.page_size)

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """解码量化页并拼接残差页，返回 float32 ``(seq_len, H, D)``。"""
        empty = (0, self.num_heads, self.head_dim)
        if self._seq_len == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone()

        parts_k: list[torch.Tensor] = []
        k_q = _decode_kivi_key_pages(
            self._k_quant_pages,
            self.k_codec,
            page_size=self.page_size,
            group_size=self.group_size,
        )
        if k_q is not None:
            parts_k.append(k_q)
        if self._k_residual_pages:
            parts_k.append(torch.cat(self._k_residual_pages, dim=0).float())

        parts_v: list[torch.Tensor] = []
        if self._v_quant_pages:
            payload = torch.cat([p.payload for p in self._v_quant_pages], dim=0)
            scale = torch.cat([p.scale for p in self._v_quant_pages], dim=0)
            zp = torch.cat([p.zero_point for p in self._v_quant_pages], dim=0)
            parts_v.append(
                self.v_codec.decode(EncodedKV(payload=payload, scale=scale, zero_point=zp))
            )
        if self._v_residual_pages:
            parts_v.append(torch.cat(self._v_residual_pages, dim=0).float())

        if not parts_k or not parts_v:
            raise RuntimeError("非空 cache 却缺少 K 或 V 片段")
        k = torch.cat(parts_k, dim=0)
        v = torch.cat(parts_v, dim=0)
        if k.shape[0] != self._seq_len or v.shape[0] != self._seq_len:
            raise RuntimeError(
                f"解码长度与 _seq_len 不一致: k={k.shape[0]}, v={v.shape[0]}, "
                f"seq_len={self._seq_len}"
            )
        return k, v

    def clear(self) -> None:
        """清空四池。"""
        self._k_quant_pages.clear()
        self._v_quant_pages.clear()
        self._k_residual_pages.clear()
        self._v_residual_pages.clear()
        self._seq_len = 0

    def bytes_breakdown(self) -> BytesBreakdown:
        """四池占用 token 的载荷 / scale / zp，加上全部页表项。"""
        payload = 0
        scale = 0
        zp = 0
        for enc in self._k_quant_pages:
            payload += self.k_codec.bytes_payload(enc)
            s, z = scale_zp_nbytes(enc)
            scale += s
            zp += z
        for enc in self._v_quant_pages:
            payload += self.v_codec.bytes_payload(enc)
            s, z = scale_zp_nbytes(enc)
            scale += s
            zp += z
        for page in (*self._k_residual_pages, *self._v_residual_pages):
            payload += _fp16_nbytes(page)
        n_pages = sum(self.page_counts().values())
        return BytesBreakdown(payload=payload, scale=scale, zp=zp, page=n_pages * self.pte_bytes)

    def bytes_stored(self) -> tuple[int, int]:
        """返回 ``(payload_bytes, metadata_bytes)``；metadata 含 page。"""
        b = self.bytes_breakdown()
        return b.payload, b.metadata
