"""物理打包后的分页 / 残差 KV：R1 刷窗与页数语义，载荷为 packed 字节。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research.r2_streaming_attention.access_events import (
    EventLog,
    PageAccessConfig,
    expected_page_counts,
    kivi_pool_tokens,
    load_page_access,
)
from research.r2_streaming_attention.physical_pack import (
    PackLayout,
    align_up,
    pack_fp16_bytes,
    pack_int8_bytes,
    pack_signed_int4,
    pack_unsigned_bits,
    tensor_nbytes,
    unpack_fp16_bytes,
    unpack_int8_bytes,
    unpack_signed_int4,
    unpack_unsigned_bits,
)
from research.r2_streaming_attention.r1_bridge import import_r1_codecs

kv_codecs = import_r1_codecs()


def _own(x: torch.Tensor) -> torch.Tensor:
    """独立紧凑副本，避免借用调用方或父存储。"""
    return x.detach().clone(memory_format=torch.contiguous_format)


def _unique_storage_nbytes(parts: list[torch.Tensor]) -> int:
    """同一底层存储只计一次。"""
    storages: dict[tuple[str, int], int] = {}
    for tensor in parts:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


@dataclass
class PackedPage:
    """一个可寻址的 packed 页或 contiguous 已提交块。"""

    payload: torch.Tensor
    scale: torch.Tensor | None
    offset: torch.Tensor | None
    n_tokens: int
    has_group_meta: bool
    pack_pad_values: int


@dataclass(frozen=True)
class Occupancy:
    """实际驻留与按页 DMA 预算；allocated 不是 PyTorch 分配器实测。"""

    payload_packed: int
    residual_fp16: int
    tail_fp16: int
    scale: int
    offset: int
    pte: int
    tags: int
    hbm_payload_allocated: int
    hbm_metadata_allocated: int
    tensor_storage_bytes: int
    pack_pad_values: int

    @property
    def alignment_waste(self) -> int:
        """页/块 DMA 取整多出的字节。"""
        return (
            self.hbm_payload_allocated
            + self.hbm_metadata_allocated
            - self.payload_packed
            - self.scale
            - self.offset
        )

    @property
    def sram_fp16(self) -> int:
        """残差窗与均匀尾页的 FP16 驻留。"""
        return self.residual_fp16 + self.tail_fp16


class PackedPhysicalCache:
    """C0–C3 / C5 的物理打包缓存：paged 含页表，C5 含 residual_length=128。

    已提交低比特历史只保留 packed 字节与 FP16 元数据。``load`` 仅供与 R1 对拍；
    ``scan_read`` 按页记账，不物化完整高精度 KV tile，也不做 Attention。
    """

    def __init__(
        self,
        format_id: str,
        *,
        num_heads: int,
        head_dim: int,
        pack_layout: PackLayout,
        page_access: PageAccessConfig | None = None,
        cache_layout: str = "paged",
        metadata_placement: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        key = format_id.strip().upper()
        if key not in {"C0", "C1", "C2", "C3", "C5"}:
            raise ValueError(f"PackedPhysicalCache 支持 C0/C1/C2/C3/C5，得到 {format_id!r}")
        if cache_layout not in {"contiguous", "paged"}:
            raise ValueError(cache_layout)
        access = page_access or load_page_access()
        placement = metadata_placement or access.default_metadata_placement
        if placement not in {"separate", "colocated"}:
            raise ValueError(f"metadata_placement 须为 separate/colocated，得到 {placement!r}")
        if head_dim % pack_layout.group_size != 0:
            raise ValueError(f"head_dim={head_dim} 须能被 group_size={pack_layout.group_size} 整除")
        if pack_layout.page_tokens != access.page_tokens:
            raise ValueError("pack_layout.page_tokens 须与 page_access 一致")
        if access.page_tokens != pack_layout.group_size // 2:
            raise ValueError("当前实现要求 16 token 页长且 group=32")
        self.format_id = key
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pack_layout = pack_layout
        self.page_access = access
        self.cache_layout = cache_layout
        self.metadata_placement = placement
        self.device = device or torch.device("cpu")
        self.events = EventLog(dma_align_bytes=access.dma_align_bytes)
        self._seq_len = 0
        self._pack_pad_values = 0
        self._k_pages: list[PackedPage] = []
        self._v_pages: list[PackedPage] = []
        self._k_tail: torch.Tensor | None = None
        self._v_tail: torch.Tensor | None = None
        self._k_quant: list[PackedPage] = []
        self._v_quant: list[PackedPage] = []
        self._k_res: list[torch.Tensor] = []
        self._v_res: list[torch.Tensor] = []
        self._codec = None
        if key == "C0":
            self._codec = kv_codecs.FP16Codec()
        elif key == "C1":
            self._codec = kv_codecs.Int8Codec(symmetric=True)
        elif key == "C2":
            self._codec = kv_codecs.Int4Codec(symmetric=True, group_size=pack_layout.group_size)
        elif key == "C3":
            self._codec = kv_codecs.Int4BdrCodec(
                symmetric=True,
                group_size=pack_layout.group_size,
                block_size=32,
                seed=0,
                dim=head_dim,
            )

    def __len__(self) -> int:
        return self._seq_len

    @property
    def k_residual_len(self) -> int:
        """C5 Key 残差 token 数；均匀路径为尾页 token 数。"""
        if self.format_id == "C5":
            return _fp16_len(self._k_res)
        return 0 if self._k_tail is None else int(self._k_tail.shape[0])

    @property
    def v_residual_len(self) -> int:
        """C5 Value 残差 token 数；均匀路径为尾页 token 数。"""
        if self.format_id == "C5":
            return _fp16_len(self._v_res)
        return 0 if self._v_tail is None else int(self._v_tail.shape[0])

    def rotation_sha256(self) -> str | None:
        """C3 旋转矩阵摘要。"""
        import hashlib

        if self.format_id != "C3":
            return None
        matrix = self._codec._rot.matrix.detach().cpu().contiguous()
        return hashlib.sha256(matrix.numpy().tobytes()).hexdigest()

    def page_counts(self) -> dict[str, int]:
        """已分配页数；contiguous 为 0。尾页/残差页计入。"""
        if self.cache_layout == "contiguous":
            return expected_page_counts(
                self.format_id,
                "contiguous",
                self._seq_len,
                page_tokens=self.page_access.page_tokens,
                residual_length=self.page_access.residual_length,
            )
        if self.format_id == "C5":
            return {
                "k_quant": len(self._k_quant),
                "k_res": len(self._k_res),
                "v_quant": len(self._v_quant),
                "v_res": len(self._v_res),
            }
        return {
            "k": len(self._k_pages) + (0 if self._k_tail is None else 1),
            "v": len(self._v_pages) + (0 if self._v_tail is None else 1),
        }

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """写入 float K/V；已提交 packed 页不再重新量化。"""
        if k_t.shape != v_t.shape:
            raise ValueError(f"k_t 与 v_t 形状不一致: {k_t.shape} vs {v_t.shape}")
        if k_t.ndim != 3:
            raise ValueError(f"期望 (n_tokens, num_heads, head_dim)，得到 {tuple(k_t.shape)}")
        n_tokens, n_heads, dim = k_t.shape
        if n_heads != self.num_heads or dim != self.head_dim:
            raise ValueError(
                f"期望 (*, {self.num_heads}, {self.head_dim})，得到 {tuple(k_t.shape)}"
            )
        if n_tokens <= 0:
            raise ValueError("n_tokens 须为正")
        k_t = _own(k_t.to(device=self.device, dtype=torch.float16))
        v_t = _own(v_t.to(device=self.device, dtype=torch.float16))
        self._record(
            action="append_capture",
            field="input",
            side="kv",
            pool="input",
            level="control",
            tokens=n_tokens,
            logical_bytes=tensor_nbytes(k_t) + tensor_nbytes(v_t),
        )
        if self.format_id == "C5":
            self._append_c5(k_t, v_t)
        elif self.cache_layout == "paged":
            self._k_tail = self._commit_full_pages("k", self._k_pages, self._k_tail, k_t)
            self._v_tail = self._commit_full_pages("v", self._v_pages, self._v_tail, v_t)
        else:
            self._commit_chunk("k", self._k_pages, k_t)
            self._commit_chunk("v", self._v_pages, v_t)
        self._seq_len += n_tokens
        self._check_pool_invariants()

    def _append_c5(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        self._append_fp16("k", "k_res", self._k_res, k_t)
        self._append_fp16("v", "v_res", self._v_res, v_t)
        self._flush_key_residual()
        self._flush_value_residual()

    def _append_fp16(
        self, side: str, pool: str, pages: list[torch.Tensor], x: torch.Tensor
    ) -> None:
        page_size = self.page_access.page_tokens
        self._record(
            action="sram_write",
            field="fp16_residual" if pool.endswith("res") else "fp16_tail",
            side=side,
            pool=pool,
            level="sram",
            tokens=int(x.shape[0]),
            logical_bytes=tensor_nbytes(x),
        )
        if self.cache_layout == "contiguous":
            if pages:
                pages[0] = _own(torch.cat([pages[0], x], dim=0))
            else:
                pages.append(x)
            return
        if pages and pages[-1].shape[0] < page_size:
            need = page_size - pages[-1].shape[0]
            take = min(need, x.shape[0])
            pages[-1] = _own(torch.cat([pages[-1], x[:take]], dim=0))
            x = x[take:]
        for start in range(0, x.shape[0], page_size):
            chunk = _own(x[start : start + page_size])
            pages.append(chunk)
            self._record_page_table(side, pool, chunk.shape[0])

    def _flush_key_residual(self) -> None:
        n = _fp16_len(self._k_res)
        residual_length = self.page_access.residual_length
        n_flush = (n // residual_length) * residual_length
        if n_flush <= 0:
            return
        residual = torch.cat(self._k_res, dim=0)
        flushed = residual[:n_flush]
        self._record(
            action="residual_flush",
            field="fp16_residual",
            side="k",
            pool="k_res",
            level="control",
            tokens=n_flush,
            logical_bytes=tensor_nbytes(flushed),
        )
        self._record_fp16_transfer("sram_read", "fp16_residual", "k", "k_res", flushed)
        encoded = kv_codecs.encode_kivi_key(
            flushed.float(), bits=4, group_size=self.pack_layout.group_size
        )
        if self.cache_layout == "paged":
            for page_enc in _split_kivi_key_pages(
                encoded,
                page_size=self.page_access.page_tokens,
                group_size=self.pack_layout.group_size,
            ):
                has_meta = page_enc.scale is not None
                self._alloc_packed_page("k", "k_quant", self._k_quant, page_enc, has_meta=has_meta)
        else:
            self._alloc_packed_page("k", "k_quant", self._k_quant, encoded, has_meta=True)
        rest = residual[n_flush:]
        self._record_residual_compaction("k", "k_res", rest)
        self._k_res = _repage_fp16(rest, self.page_access.page_tokens, self.cache_layout)

    def _flush_value_residual(self) -> None:
        n = _fp16_len(self._v_res)
        residual_length = self.page_access.residual_length
        if n <= residual_length:
            return
        n_overflow = n - residual_length
        residual = torch.cat(self._v_res, dim=0)
        overflow = residual[:n_overflow]
        self._record(
            action="residual_flush",
            field="fp16_residual",
            side="v",
            pool="v_res",
            level="control",
            tokens=n_overflow,
            logical_bytes=tensor_nbytes(overflow),
        )
        self._record_fp16_transfer("sram_read", "fp16_residual", "v", "v_res", overflow)
        encoded = kv_codecs.encode_kivi_value(
            overflow.float(), bits=4, group_size=self.pack_layout.group_size
        )
        if self.cache_layout == "paged":
            self._extend_value_quant(encoded)
        else:
            self._alloc_packed_page("v", "v_quant", self._v_quant, encoded, has_meta=True)
        rest = residual[n_overflow:]
        self._record_residual_compaction("v", "v_res", rest)
        self._v_res = _repage_fp16(rest, self.page_access.page_tokens, self.cache_layout)

    def _record_fp16_transfer(
        self, action: str, field: str, side: str, pool: str, tensor: torch.Tensor
    ) -> None:
        self._record(
            action=action,
            field=field,
            side=side,
            pool=pool,
            level="sram",
            tokens=int(tensor.shape[0]),
            logical_bytes=tensor_nbytes(tensor),
        )

    def _record_residual_compaction(self, side: str, pool: str, rest: torch.Tensor) -> None:
        """前缀刷出后，当前紧凑残差布局需要读写保留部分；不假定免费环形搬移。"""
        if rest.shape[0]:
            for action in ("sram_read", "sram_write"):
                self._record_fp16_transfer(action, "fp16_compaction", side, pool, rest)

    def _commit_full_pages(
        self,
        side: str,
        pages: list[PackedPage],
        tail: torch.Tensor | None,
        new: torch.Tensor,
    ) -> torch.Tensor | None:
        converting_tail = tail is not None
        buf = new if tail is None else _own(torch.cat([tail, new], dim=0))
        self._record(
            action="sram_write",
            field="fp16_tail",
            side=side,
            pool="tail",
            level="sram",
            tokens=int(new.shape[0]),
            logical_bytes=tensor_nbytes(new),
        )
        page_size = self.page_access.page_tokens
        while buf.shape[0] >= page_size:
            chunk = buf[:page_size]
            self._record_fp16_transfer("sram_read", "fp16_tail", side, "tail", chunk)
            encoded = self._codec.encode(chunk)
            if converting_tail:
                page = self._make_page(encoded, has_meta=True)
                pages.append(page)
                self._emit_page_transfer(page, action="hbm_write", side=side, pool="uniform")
                self._record_page_table(side, "uniform", page.n_tokens)
                converting_tail = False
            else:
                self._alloc_packed_page(side, "uniform", pages, encoded, has_meta=True)
            buf = buf[page_size:]
        if buf.shape[0] == 0:
            return None
        owned = _own(buf)
        if not converting_tail:
            self._record_page_table(side, "tail", owned.shape[0])
        return owned

    def _commit_chunk(self, side: str, pages: list[PackedPage], x: torch.Tensor) -> None:
        encoded = self._codec.encode(x)
        self._alloc_packed_page(side, "uniform", pages, encoded, has_meta=True)

    def _extend_value_quant(self, incoming: kv_codecs.EncodedKV) -> None:
        consumed = 0
        if self._v_quant and self._v_quant[-1].n_tokens < self.page_access.page_tokens:
            old = self._v_quant[-1]
            self._emit_page_transfer(old, action="hbm_rmw_read", side="v", pool="v_quant")
            consumed = min(self.page_access.page_tokens - old.n_tokens, incoming.payload.shape[0])
            merged = _cat_encoded(_page_to_grid(self, old), _slice_encoded_t(incoming, 0, consumed))
            page = self._make_page(merged, has_meta=True)
            self._v_quant[-1] = page
            self._emit_page_transfer(page, action="hbm_write", side="v", pool="v_quant")
        # 一个 append 内新生成的片段直接成页，不重复读回刚写出的临时短页。
        rest = _slice_encoded_t(incoming, consumed, incoming.payload.shape[0])
        for page_enc in _split_along_t(rest, self.page_access.page_tokens):
            self._alloc_packed_page("v", "v_quant", self._v_quant, page_enc, has_meta=True)

    def _alloc_packed_page(
        self,
        side: str,
        pool: str,
        pages: list[PackedPage],
        encoded: kv_codecs.EncodedKV,
        *,
        has_meta: bool,
    ) -> PackedPage:
        page = self._make_page(encoded, has_meta=has_meta)
        pages.append(page)
        if self.cache_layout == "paged":
            self._record_page_table(side, pool, page.n_tokens)
        self._emit_page_transfer(page, action="hbm_write", side=side, pool=pool)
        return page

    def _make_page(self, encoded: kv_codecs.EncodedKV, *, has_meta: bool) -> PackedPage:
        packed, pad = self._pack_payload(encoded.payload)
        self._pack_pad_values += pad
        scale = None
        offset = None
        if has_meta:
            if encoded.scale is not None:
                scale = _own(encoded.scale.contiguous())
            if encoded.zero_point is not None:
                offset = _own(encoded.zero_point.contiguous())
        return PackedPage(
            payload=_own(packed),
            scale=scale,
            offset=offset,
            n_tokens=int(encoded.payload.shape[0]),
            has_group_meta=has_meta and encoded.scale is not None,
            pack_pad_values=pad,
        )

    def _pack_payload(self, payload: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self.format_id == "C0":
            return pack_fp16_bytes(payload), 0
        if self.format_id == "C1":
            return pack_int8_bytes(payload), 0
        if self.format_id == "C5":
            return pack_unsigned_bits(payload, 4)
        return pack_signed_int4(
            payload, qmin=self.pack_layout.int4_qmin, qmax=self.pack_layout.int4_qmax
        )

    def _unpack_payload(self, packed: torch.Tensor, n_tokens: int) -> torch.Tensor:
        numel = n_tokens * self.num_heads * self.head_dim
        shape = (n_tokens, self.num_heads, self.head_dim)
        if self.format_id == "C0":
            return unpack_fp16_bytes(packed, numel).reshape(shape).to(torch.float16)
        if self.format_id == "C1":
            return unpack_int8_bytes(packed, numel).reshape(shape)
        if self.format_id == "C5":
            return unpack_unsigned_bits(packed, 4, numel).to(torch.uint8).reshape(shape)
        grid = unpack_signed_int4(
            packed,
            numel,
            qmin=self.pack_layout.int4_qmin,
            qmax=self.pack_layout.int4_qmax,
        )
        return grid.reshape(shape)

    def _record_page_table(self, side: str, pool: str, tokens: int) -> None:
        self._record(
            action="pte_write",
            field="pte",
            side=side,
            pool=pool,
            level=self.page_access.pte_level,
            tokens=tokens,
            logical_bytes=self.page_access.pte_bytes,
        )
        self._record(
            action="tag_write",
            field="tag",
            side=side,
            pool=pool,
            level=self.page_access.tag_level,
            tokens=tokens,
            logical_bytes=self.page_access.tag_bytes,
        )

    def _emit_page_transfer(self, page: PackedPage, *, action: str, side: str, pool: str) -> None:
        payload, meta = _page_parts(page)
        if self.metadata_placement == "colocated":
            self._record(
                action=action,
                field="colocated",
                side=side,
                pool=pool,
                level="hbm",
                tokens=page.n_tokens,
                logical_bytes=payload + meta,
            )
            return
        self._record(
            action=action,
            field="payload",
            side=side,
            pool=pool,
            level="hbm",
            tokens=page.n_tokens,
            logical_bytes=payload,
        )
        if meta:
            self._record(
                action=action,
                field="metadata",
                side=side,
                pool=pool,
                level="hbm",
                tokens=page.n_tokens,
                logical_bytes=meta,
            )

    def _record(
        self,
        *,
        action: str,
        field: str,
        side: str,
        pool: str,
        level: str,
        tokens: int,
        logical_bytes: int,
    ) -> None:
        self.events.record(
            action=action,
            field=field,
            side=side,
            pool=pool,
            level=level,
            tokens=tokens,
            logical_bytes=logical_bytes,
            seq_len=self._seq_len,
        )

    def snapshot_complete_payloads(self) -> dict[str, list[torch.Tensor]]:
        """已满 packed 页的载荷副本；未满 Value 量化尾页排除在外。"""
        out: dict[str, list[torch.Tensor]] = {}
        page_size = self.page_access.page_tokens

        def _take(name: str, pages: list[PackedPage], *, allow_partial: bool) -> None:
            chosen = []
            for index, page in enumerate(pages):
                last = index == len(pages) - 1
                if (
                    self.cache_layout == "paged"
                    and last
                    and page.n_tokens < page_size
                    and not allow_partial
                ):
                    continue
                chosen.append(page.payload.clone())
            out[name] = chosen

        if self.format_id == "C5":
            _take("k_quant", self._k_quant, allow_partial=True)
            _take("v_quant", self._v_quant, allow_partial=False)
        else:
            _take("k", self._k_pages, allow_partial=True)
            _take("v", self._v_pages, allow_partial=True)
        return out

    def scan_read(self) -> None:
        """按页/残差读取当前 KV 并记账；不解包为 FP32，不是 Attention。"""
        if self._seq_len == 0:
            return
        if self.format_id == "C5":
            self._scan_packed("k", "k_quant", self._k_quant)
            self._scan_fp16("k", "k_res", self._k_res, field="fp16_residual")
            self._scan_packed("v", "v_quant", self._v_quant)
            self._scan_fp16("v", "v_res", self._v_res, field="fp16_residual")
            return
        self._scan_packed("k", "uniform", self._k_pages)
        self._scan_fp16("k", "tail", _as_list(self._k_tail), field="fp16_tail")
        self._scan_packed("v", "uniform", self._v_pages)
        self._scan_fp16("v", "tail", _as_list(self._v_tail), field="fp16_tail")

    def _scan_packed(self, side: str, pool: str, pages: list[PackedPage]) -> None:
        for page in pages:
            if self.cache_layout == "paged":
                self._record(
                    action="pte_lookup",
                    field="pte",
                    side=side,
                    pool=pool,
                    level=self.page_access.pte_level,
                    tokens=page.n_tokens,
                    logical_bytes=self.page_access.pte_bytes,
                )
                self._record(
                    action="tag_read",
                    field="tag",
                    side=side,
                    pool=pool,
                    level=self.page_access.tag_level,
                    tokens=page.n_tokens,
                    logical_bytes=self.page_access.tag_bytes,
                )
            self._emit_page_transfer(page, action="hbm_read", side=side, pool=pool)

    def _scan_fp16(self, side: str, pool: str, tensors: list[torch.Tensor], *, field: str) -> None:
        for tensor in tensors:
            tokens = int(tensor.shape[0])
            if self.cache_layout == "paged":
                self._record(
                    action="pte_lookup",
                    field="pte",
                    side=side,
                    pool=pool,
                    level=self.page_access.pte_level,
                    tokens=tokens,
                    logical_bytes=self.page_access.pte_bytes,
                )
                self._record(
                    action="tag_read",
                    field="tag",
                    side=side,
                    pool=pool,
                    level=self.page_access.tag_level,
                    tokens=tokens,
                    logical_bytes=self.page_access.tag_bytes,
                )
            self._record(
                action="sram_read",
                field=field,
                side=side,
                pool=pool,
                level="sram",
                tokens=tokens,
                logical_bytes=tensor_nbytes(tensor),
            )

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """解包并经 R1 decode 得到 float32；不计入访存事件。"""
        empty = (0, self.num_heads, self.head_dim)
        if self._seq_len == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone()
        if self.format_id == "C5":
            k = self._load_c5_key()
            v = self._load_c5_value()
        else:
            k = self._load_uniform("k")
            v = self._load_uniform("v")
        if k.shape[0] != self._seq_len or v.shape[0] != self._seq_len:
            raise RuntimeError(
                f"解码长度与 seq_len 不一致: k={k.shape[0]}, v={v.shape[0]}, seq_len={self._seq_len}"
            )
        return k, v

    def _load_uniform(self, side: str) -> torch.Tensor:
        pages = self._k_pages if side == "k" else self._v_pages
        tail = self._k_tail if side == "k" else self._v_tail
        if self.cache_layout == "paged":
            parts = [self._codec.decode(_page_to_grid(self, page)) for page in pages]
            if tail is not None:
                parts.append(self._codec.decode(self._codec.encode(tail)))
            if not parts:
                raise RuntimeError("非空 cache 缺少均匀解码片段")
            return torch.cat(parts, dim=0)
        if not pages:
            raise RuntimeError("非空 contiguous cache 缺少均匀解码片段")
        grids = [self._unpack_payload(page.payload, page.n_tokens) for page in pages]
        payload = torch.cat(grids, dim=0)
        scale = _cat_meta([page.scale for page in pages])
        offset = _cat_meta([page.offset for page in pages])
        return self._codec.decode(
            kv_codecs.EncodedKV(payload=payload, scale=scale, zero_point=offset)
        )

    def _load_c5_key(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.cache_layout == "contiguous":
            for page in self._k_quant:
                parts.append(
                    kv_codecs.decode_kivi_key(
                        _page_to_grid(self, page), group_size=self.pack_layout.group_size
                    )
                )
        elif self._k_quant:
            parts.append(self._decode_kivi_key_pages(self._k_quant))
        if self._k_res:
            parts.append(torch.cat(self._k_res, dim=0).float())
        if not parts:
            raise RuntimeError("非空 C5 cache 缺少 Key")
        return torch.cat(parts, dim=0)

    def _load_c5_value(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self._v_quant:
            if self.cache_layout == "paged":
                payload = torch.cat(
                    [self._unpack_payload(p.payload, p.n_tokens) for p in self._v_quant], dim=0
                )
                scale = torch.cat([p.scale for p in self._v_quant], dim=0)
                offset = torch.cat([p.offset for p in self._v_quant], dim=0)
                parts.append(
                    kv_codecs.decode_kivi_value(
                        kv_codecs.EncodedKV(payload=payload, scale=scale, zero_point=offset),
                        group_size=self.pack_layout.group_size,
                    )
                )
            else:
                for page in self._v_quant:
                    parts.append(
                        kv_codecs.decode_kivi_value(
                            _page_to_grid(self, page), group_size=self.pack_layout.group_size
                        )
                    )
        if self._v_res:
            parts.append(torch.cat(self._v_res, dim=0).float())
        if not parts:
            raise RuntimeError("非空 C5 cache 缺少 Value")
        return torch.cat(parts, dim=0)

    def _decode_kivi_key_pages(self, pages: list[PackedPage]) -> torch.Tensor:
        pages_per_group = self.pack_layout.group_size // self.page_access.page_tokens
        if len(pages) % pages_per_group != 0:
            raise RuntimeError(f"Key 量化页数 {len(pages)} 不能被 {pages_per_group} 整除")
        parts: list[torch.Tensor] = []
        for start in range(0, len(pages), pages_per_group):
            group = pages[start : start + pages_per_group]
            payload = torch.cat([self._unpack_payload(p.payload, p.n_tokens) for p in group], dim=0)
            meta = group[0]
            if meta.scale is None or meta.offset is None:
                raise RuntimeError("Key group 第一页缺少 scale/min")
            parts.append(
                kv_codecs.decode_kivi_key(
                    kv_codecs.EncodedKV(payload=payload, scale=meta.scale, zero_point=meta.offset),
                    group_size=self.pack_layout.group_size,
                )
            )
        return torch.cat(parts, dim=0)

    def iter_stream_tiles(self, tile_tokens: int, *, domain: str = "original"):
        """按 ``tile_tokens`` 切分当前序列，每次只反量化一个区间。"""
        if tile_tokens <= 0:
            raise ValueError(f"tile_tokens 须为正，得到 {tile_tokens}")
        if domain not in {"original", "rotated"}:
            raise ValueError(f"domain 须为 original/rotated，得到 {domain!r}")
        for start in range(0, self._seq_len, tile_tokens):
            end = min(start + tile_tokens, self._seq_len)
            yield self.decode_token_range(start, end, domain=domain)

    def decode_token_range(
        self,
        start: int,
        end: int,
        *,
        domain: str = "original",
        record_events: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """反量化 ``[start, end)`` 的 K/V 工作缓冲，不解出完整序列。

        ``domain=original`` 与 ``load`` 的数值域一致，但尾页/残差保持已存储的
        FP16，不再二次量化。``domain=rotated`` 仅用于 C3 的 Q/O 路径：已打包
        片段保持旋转域，尾页在读侧旋转进同一域，不改写存储。

        参数
            start, end: 半开区间，单位 token
            domain: ``original`` 或 ``rotated``
            record_events: 是否按重叠页/残差记读事件
        返回
            ``(k, v, scratch_tokens)``；``scratch_tokens`` 为切片前实际反量化
            的 token 数（C5 Key 可能因 group=32 对齐而大于区间长度）
        """
        if start < 0 or end < start or end > self._seq_len:
            raise ValueError(f"区间 [{start}, {end}) 超出 seq_len={self._seq_len}")
        if domain not in {"original", "rotated"}:
            raise ValueError(f"domain 须为 original/rotated，得到 {domain!r}")
        if domain == "rotated" and self.format_id != "C3":
            raise ValueError("rotated 域仅用于 C3")
        n = end - start
        empty = (0, self.num_heads, self.head_dim)
        if n == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone(), 0
        if self.format_id == "C5":
            k, k_scratch, k_reads = self._range_c5_key(start, end)
            v, v_scratch, v_reads = self._range_c5_value(start, end)
        else:
            k, k_scratch, k_reads = self._range_uniform("k", start, end, domain)
            v, v_scratch, v_reads = self._range_uniform("v", start, end, domain)
        if record_events:
            for item in (*k_reads, *v_reads):
                self._record_stream_read(**item)
        return k, v, max(k_scratch, v_scratch)

    def occupancy(self) -> Occupancy:
        """实测张量驻留与按页 DMA 预算。"""
        packed_pages = self._all_packed_pages()
        payload = sum(tensor_nbytes(p.payload) for p in packed_pages)
        scale = sum(tensor_nbytes(p.scale) for p in packed_pages if p.scale is not None)
        offset = sum(tensor_nbytes(p.offset) for p in packed_pages if p.offset is not None)
        residual = sum(tensor_nbytes(t) for t in [*self._k_res, *self._v_res])
        tail = (0 if self._k_tail is None else tensor_nbytes(self._k_tail)) + (
            0 if self._v_tail is None else tensor_nbytes(self._v_tail)
        )
        n_ptes = sum(self.page_counts().values()) if self.cache_layout == "paged" else 0
        hbm_payload = 0
        hbm_meta = 0
        dma = self.page_access.dma_align_bytes  # 按页事务量子，不是校准突发
        for page in packed_pages:
            pay, meta = _page_parts(page)
            if self.metadata_placement == "colocated":
                hbm_payload += align_up(pay + meta, dma)
            else:
                hbm_payload += align_up(pay, dma)
                if meta:
                    hbm_meta += align_up(meta, dma)
        buffers: list[torch.Tensor] = []
        for page in packed_pages:
            buffers.append(page.payload)
            if page.scale is not None:
                buffers.append(page.scale)
            if page.offset is not None:
                buffers.append(page.offset)
        buffers.extend(self._k_res)
        buffers.extend(self._v_res)
        if self._k_tail is not None:
            buffers.append(self._k_tail)
        if self._v_tail is not None:
            buffers.append(self._v_tail)
        return Occupancy(
            payload_packed=payload,
            residual_fp16=residual,
            tail_fp16=tail,
            scale=scale,
            offset=offset,
            pte=n_ptes * self.page_access.pte_bytes,
            tags=n_ptes * self.page_access.tag_bytes,
            hbm_payload_allocated=hbm_payload,
            hbm_metadata_allocated=hbm_meta,
            tensor_storage_bytes=_unique_storage_nbytes(buffers),
            pack_pad_values=self._pack_pad_values,
        )

    def _all_packed_pages(self) -> list[PackedPage]:
        if self.format_id == "C5":
            return [*self._k_quant, *self._v_quant]
        return [*self._k_pages, *self._v_pages]

    def _packed_groups(self) -> list[list[PackedPage]]:
        """K/V 两侧各自的 packed 页，用于对照整段对齐预算。"""
        if self.format_id == "C5":
            return [self._k_quant, self._v_quant]
        return [self._k_pages, self._v_pages]

    def occupancy_dict(self) -> dict[str, int]:
        """占用快照，含按页 DMA 与整段对齐对照。"""
        occ = self.occupancy()
        coalesced_payload = 0
        coalesced_meta = 0
        dma = self.page_access.dma_align_bytes
        for group in self._packed_groups():
            pay = sum(tensor_nbytes(page.payload) for page in group)
            meta = sum(_page_parts(page)[1] for page in group)
            if self.metadata_placement == "colocated":
                coalesced_payload += align_up(pay + meta, dma)
            else:
                coalesced_payload += align_up(pay, dma)
                if meta:
                    coalesced_meta += align_up(meta, dma)
        return {
            "payload_packed": occ.payload_packed,
            "residual_fp16": occ.residual_fp16,
            "tail_fp16": occ.tail_fp16,
            "scale": occ.scale,
            "offset": occ.offset,
            "pte": occ.pte,
            "tags": occ.tags,
            "hbm_payload_allocated": occ.hbm_payload_allocated,
            "hbm_metadata_allocated": occ.hbm_metadata_allocated,
            "coalesced_payload_allocated": coalesced_payload,
            "coalesced_metadata_allocated": coalesced_meta,
            "tensor_storage_bytes": occ.tensor_storage_bytes,
            "pack_pad_values": occ.pack_pad_values,
            "alignment_waste": occ.alignment_waste,
            "sram_fp16": occ.sram_fp16,
            "k_residual_len": self.k_residual_len,
            "v_residual_len": self.v_residual_len,
        }

    def _range_uniform(
        self, side: str, start: int, end: int, domain: str
    ) -> tuple[torch.Tensor, int, list[dict]]:
        pages = self._k_pages if side == "k" else self._v_pages
        tail = self._k_tail if side == "k" else self._v_tail
        pool_packed = "uniform"
        parts: list[torch.Tensor] = []
        reads: list[dict] = []
        scratch = 0
        cursor = 0
        for page in pages:
            page_end = cursor + page.n_tokens
            overlap = _overlap(start, end, cursor, page_end)
            if overlap is not None:
                local_s, local_e = overlap[0] - cursor, overlap[1] - cursor
                encoded = self._encoded_page_slice(page, local_s, local_e)
                parts.append(self._dequant_encoded(encoded, source="packed", domain=domain))
                scratch += local_e - local_s
                reads.append(self._packed_read_desc(side, pool_packed, page, encoded))
            cursor = page_end
        if tail is not None:
            page_end = cursor + int(tail.shape[0])
            overlap = _overlap(start, end, cursor, page_end)
            if overlap is not None:
                local_s, local_e = overlap[0] - cursor, overlap[1] - cursor
                chunk = tail[local_s:local_e].float()
                parts.append(self._dequant_encoded(chunk, source="fp16", domain=domain))
                scratch += local_e - local_s
                reads.append(
                    {
                        "kind": "fp16",
                        "side": side,
                        "pool": "tail",
                        "field": "fp16_tail",
                        "tokens": local_e - local_s,
                        "logical_bytes": tensor_nbytes(chunk.to(torch.float16)),
                        "page": None,
                        "meta_bytes": 0,
                    }
                )
        if not parts:
            raise RuntimeError(f"{side} 区间 [{start}, {end}) 没有覆盖片段")
        return torch.cat(parts, dim=0), scratch, reads

    def _range_c5_key(self, start: int, end: int) -> tuple[torch.Tensor, int, list[dict]]:
        quant_len = sum(page.n_tokens for page in self._k_quant)
        parts: list[torch.Tensor] = []
        reads: list[dict] = []
        scratch = 0
        q_overlap = _overlap(start, end, 0, quant_len)
        if q_overlap is not None:
            decoded, used, page_reads = self._decode_c5_key_quant(q_overlap[0], q_overlap[1])
            parts.append(decoded)
            scratch += used
            reads.extend(page_reads)
        res_start = quant_len
        cursor = res_start
        for tensor in self._k_res:
            page_end = cursor + int(tensor.shape[0])
            overlap = _overlap(start, end, cursor, page_end)
            if overlap is not None:
                local_s, local_e = overlap[0] - cursor, overlap[1] - cursor
                chunk = tensor[local_s:local_e].float()
                parts.append(chunk)
                scratch += local_e - local_s
                reads.append(
                    {
                        "kind": "fp16",
                        "side": "k",
                        "pool": "k_res",
                        "field": "fp16_residual",
                        "tokens": local_e - local_s,
                        "logical_bytes": tensor_nbytes(chunk.to(torch.float16)),
                        "page": None,
                        "meta_bytes": 0,
                    }
                )
            cursor = page_end
        if not parts:
            raise RuntimeError(f"C5 Key 区间 [{start}, {end}) 没有覆盖片段")
        return torch.cat(parts, dim=0), scratch, reads

    def _range_c5_value(self, start: int, end: int) -> tuple[torch.Tensor, int, list[dict]]:
        parts: list[torch.Tensor] = []
        reads: list[dict] = []
        scratch = 0
        cursor = 0
        for page in self._v_quant:
            page_end = cursor + page.n_tokens
            overlap = _overlap(start, end, cursor, page_end)
            if overlap is not None:
                local_s, local_e = overlap[0] - cursor, overlap[1] - cursor
                encoded = self._encoded_page_slice(page, local_s, local_e)
                parts.append(
                    kv_codecs.decode_kivi_value(encoded, group_size=self.pack_layout.group_size)
                )
                scratch += local_e - local_s
                reads.append(self._packed_read_desc("v", "v_quant", page, encoded))
            cursor = page_end
        for tensor in self._v_res:
            page_end = cursor + int(tensor.shape[0])
            overlap = _overlap(start, end, cursor, page_end)
            if overlap is not None:
                local_s, local_e = overlap[0] - cursor, overlap[1] - cursor
                chunk = tensor[local_s:local_e].float()
                parts.append(chunk)
                scratch += local_e - local_s
                reads.append(
                    {
                        "kind": "fp16",
                        "side": "v",
                        "pool": "v_res",
                        "field": "fp16_residual",
                        "tokens": local_e - local_s,
                        "logical_bytes": tensor_nbytes(chunk.to(torch.float16)),
                        "page": None,
                        "meta_bytes": 0,
                    }
                )
            cursor = page_end
        if not parts:
            raise RuntimeError(f"C5 Value 区间 [{start}, {end}) 没有覆盖片段")
        return torch.cat(parts, dim=0), scratch, reads

    def _decode_c5_key_quant(self, start: int, end: int) -> tuple[torch.Tensor, int, list[dict]]:
        """按 group=32 对齐反量化 Key 量化区，再切到 ``[start, end)``。"""
        group = self.pack_layout.group_size
        g0 = start // group
        g1 = (end + group - 1) // group
        pieces: list[torch.Tensor] = []
        reads: list[dict] = []
        scratch = 0
        if self.cache_layout == "paged":
            pages_per_group = group // self.page_access.page_tokens
            for group_idx in range(g0, g1):
                p0 = group_idx * pages_per_group
                group_pages = self._k_quant[p0 : p0 + pages_per_group]
                payload = torch.cat(
                    [self._unpack_payload(p.payload, p.n_tokens) for p in group_pages], dim=0
                )
                meta = group_pages[0]
                pieces.append(
                    kv_codecs.decode_kivi_key(
                        kv_codecs.EncodedKV(
                            payload=payload, scale=meta.scale, zero_point=meta.offset
                        ),
                        group_size=group,
                    )
                )
                scratch += group
                for page in group_pages:
                    grid = _page_to_grid(self, page)
                    reads.append(self._packed_read_desc("k", "k_quant", page, grid))
            decoded = torch.cat(pieces, dim=0)
            local_s = start - g0 * group
            local_e = local_s + (end - start)
            return decoded[local_s:local_e], scratch, reads
        cursor = 0
        for page in self._k_quant:
            page_end = cursor + page.n_tokens
            overlap = _overlap(g0 * group, g1 * group, cursor, page_end)
            if overlap is not None:
                local_s, local_e = overlap[0] - cursor, overlap[1] - cursor
                if local_s % group != 0 or (local_e - local_s) % group != 0:
                    raise RuntimeError("contiguous C5 Key 切片须对齐 group")
                encoded = self._encoded_page_slice(page, local_s, local_e)
                pieces.append(kv_codecs.decode_kivi_key(encoded, group_size=group))
                scratch += local_e - local_s
                reads.append(self._packed_read_desc("k", "k_quant", page, encoded))
            cursor = page_end
        if not pieces:
            raise RuntimeError("C5 Key 量化区缺少覆盖页")
        decoded = torch.cat(pieces, dim=0)
        local_s = start - g0 * group
        local_e = local_s + (end - start)
        return decoded[local_s:local_e], scratch, reads

    def _encoded_page_slice(self, page: PackedPage, local_start: int, local_end: int):
        """解包一页内的 token 切片，只取该区间的 packed 字节与元数据。"""
        if local_start < 0 or local_end > page.n_tokens or local_end <= local_start:
            raise ValueError(f"页切片 [{local_start}, {local_end}) 超出 n_tokens={page.n_tokens}")
        payload = self._unpack_payload_slice(page, local_start, local_end)
        scale = _slice_group_or_token_meta(page.scale, page.n_tokens, local_start, local_end)
        offset = _slice_group_or_token_meta(page.offset, page.n_tokens, local_start, local_end)
        return kv_codecs.EncodedKV(payload=payload, scale=scale, zero_point=offset)

    def _unpack_payload_slice(
        self, page: PackedPage, local_start: int, local_end: int
    ) -> torch.Tensor:
        n = local_end - local_start
        elems = n * self.num_heads * self.head_dim
        per_token = self.num_heads * self.head_dim
        shape = (n, self.num_heads, self.head_dim)
        packed = page.payload
        if self.format_id == "C0":
            b0 = local_start * per_token * 2
            b1 = local_end * per_token * 2
            return unpack_fp16_bytes(packed[b0:b1], elems).reshape(shape).to(torch.float16)
        if self.format_id == "C1":
            b0 = local_start * per_token
            b1 = local_end * per_token
            return unpack_int8_bytes(packed[b0:b1], elems).reshape(shape)
        if per_token % 2 != 0:
            raise RuntimeError("4-bit 按 token 切片要求每 token 元素数为偶数")
        b0 = (local_start * per_token) // 2
        b1 = (local_end * per_token) // 2
        sl = packed[b0:b1]
        if self.format_id == "C5":
            return unpack_unsigned_bits(sl, 4, elems).to(torch.uint8).reshape(shape)
        grid = unpack_signed_int4(
            sl,
            elems,
            qmin=self.pack_layout.int4_qmin,
            qmax=self.pack_layout.int4_qmax,
        )
        return grid.reshape(shape)

    def _dequant_encoded(self, encoded, *, source: str, domain: str) -> torch.Tensor:
        if source == "fp16":
            x = encoded if isinstance(encoded, torch.Tensor) else encoded.payload.float()
            if self.format_id == "C3" and domain == "rotated":
                return self._codec._rot.rotate(x.float(), axis=-1)
            return x.float()
        if self.format_id == "C0":
            return encoded.payload.float()
        if self.format_id == "C3":
            y = kv_codecs._decode_uniform_int(encoded, group_size=self.pack_layout.group_size)
            if domain == "original":
                return self._codec._rot.inverse(y, axis=-1)
            return y
        if self.format_id in {"C1", "C2"}:
            group = None if self.format_id == "C1" else self.pack_layout.group_size
            return kv_codecs._decode_uniform_int(encoded, group_size=group)
        raise RuntimeError(f"不支持的反量化格式 {self.format_id}")

    def _packed_read_desc(
        self, side: str, pool: str, page: PackedPage, encoded: kv_codecs.EncodedKV
    ) -> dict:
        n_tokens = int(encoded.payload.shape[0])
        payload = self._packed_logical_bytes(n_tokens)
        meta = 0
        if encoded.scale is not None:
            meta += tensor_nbytes(encoded.scale)
        if encoded.zero_point is not None:
            meta += tensor_nbytes(encoded.zero_point)
        return {
            "kind": "packed",
            "side": side,
            "pool": pool,
            "field": "payload",
            "tokens": n_tokens,
            "logical_bytes": payload,
            "meta_bytes": meta,
            "page": page,
        }

    def _packed_logical_bytes(self, n_tokens: int) -> int:
        elems = n_tokens * self.num_heads * self.head_dim
        if self.format_id == "C0":
            return elems * 2
        if self.format_id == "C1":
            return elems
        return (elems + 1) // 2

    def _record_stream_read(
        self,
        *,
        kind: str,
        side: str,
        pool: str,
        field: str,
        tokens: int,
        logical_bytes: int,
        page: PackedPage | None,
        meta_bytes: int = 0,
    ) -> None:
        if self.cache_layout == "paged":
            self._record(
                action="pte_lookup",
                field="pte",
                side=side,
                pool=pool,
                level=self.page_access.pte_level,
                tokens=tokens,
                logical_bytes=self.page_access.pte_bytes,
            )
            self._record(
                action="tag_read",
                field="tag",
                side=side,
                pool=pool,
                level=self.page_access.tag_level,
                tokens=tokens,
                logical_bytes=self.page_access.tag_bytes,
            )
        if kind == "packed":
            if self.metadata_placement == "colocated":
                self._record(
                    action="hbm_read",
                    field="colocated",
                    side=side,
                    pool=pool,
                    level="hbm",
                    tokens=tokens,
                    logical_bytes=logical_bytes + meta_bytes,
                )
                return
            self._record(
                action="hbm_read",
                field="payload",
                side=side,
                pool=pool,
                level="hbm",
                tokens=tokens,
                logical_bytes=logical_bytes,
            )
            if meta_bytes:
                self._record(
                    action="hbm_read",
                    field="metadata",
                    side=side,
                    pool=pool,
                    level="hbm",
                    tokens=tokens,
                    logical_bytes=meta_bytes,
                )
            return
        self._record(
            action="sram_read",
            field=field,
            side=side,
            pool=pool,
            level="sram",
            tokens=tokens,
            logical_bytes=logical_bytes,
        )

    def _check_pool_invariants(self) -> None:
        counts = self.page_counts()
        expected = expected_page_counts(
            self.format_id,
            self.cache_layout,
            self._seq_len,
            page_tokens=self.page_access.page_tokens,
            residual_length=self.page_access.residual_length,
        )
        if counts != expected:
            raise RuntimeError(f"页数与公式不一致: 得到 {counts}，期望 {expected}")
        if self.format_id != "C5":
            return
        pools = kivi_pool_tokens(self._seq_len, self.page_access.residual_length)
        if _fp16_len(self._k_res) != pools["k_res"] or _fp16_len(self._v_res) != pools["v_res"]:
            raise RuntimeError(
                f"残差长度与公式不一致: k={_fp16_len(self._k_res)} v={_fp16_len(self._v_res)} "
                f"期望 {pools}"
            )
        k_quant_tokens = sum(p.n_tokens for p in self._k_quant)
        v_quant_tokens = sum(p.n_tokens for p in self._v_quant)
        if k_quant_tokens != pools["k_quant"] or v_quant_tokens != pools["v_quant"]:
            raise RuntimeError(
                f"量化 token 与公式不一致: k={k_quant_tokens} v={v_quant_tokens} 期望 {pools}"
            )
        if k_quant_tokens % self.pack_layout.group_size != 0:
            raise RuntimeError("Key 量化 token 不能被 group_size 整除")
        if self.v_residual_len > self.page_access.residual_length:
            raise RuntimeError("Value 残差超过 residual_length")


def _fp16_len(pages: list[torch.Tensor]) -> int:
    return sum(int(page.shape[0]) for page in pages)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    """两个半开区间的交集；无重叠则返回 None。"""
    start = max(a0, b0)
    end = min(a1, b1)
    if end <= start:
        return None
    return start, end


def _slice_group_or_token_meta(
    meta: torch.Tensor | None, n_tokens: int, local_start: int, local_end: int
) -> torch.Tensor | None:
    """按 token 或按 group 轴切 scale/min。"""
    if meta is None:
        return None
    if meta.shape[0] == n_tokens:
        return meta[local_start:local_end]
    if n_tokens % meta.shape[0] != 0:
        raise RuntimeError(f"元数据第一维 {meta.shape[0]} 与 n_tokens={n_tokens} 无法对齐")
    tokens_per = n_tokens // meta.shape[0]
    if local_start % tokens_per != 0 or local_end % tokens_per != 0:
        raise RuntimeError("group 元数据切片须对齐组边界")
    return meta[local_start // tokens_per : local_end // tokens_per]


def _as_list(tensor: torch.Tensor | None) -> list[torch.Tensor]:
    return [] if tensor is None else [tensor]


def _repage_fp16(x: torch.Tensor, page_size: int, cache_layout: str) -> list[torch.Tensor]:
    if x.shape[0] == 0:
        return []
    if cache_layout == "contiguous":
        return [_own(x)]
    return [_own(x[i : i + page_size]) for i in range(0, x.shape[0], page_size)]


def _page_parts(page: PackedPage) -> tuple[int, int]:
    payload = tensor_nbytes(page.payload)
    meta = 0
    if page.has_group_meta:
        if page.scale is not None:
            meta += tensor_nbytes(page.scale)
        if page.offset is not None:
            meta += tensor_nbytes(page.offset)
    return payload, meta


def _page_to_grid(cache: PackedPhysicalCache, page: PackedPage) -> kv_codecs.EncodedKV:
    payload = cache._unpack_payload(page.payload, page.n_tokens)
    return kv_codecs.EncodedKV(payload=payload, scale=page.scale, zero_point=page.offset)


def _slice_encoded_t(encoded: kv_codecs.EncodedKV, start: int, end: int) -> kv_codecs.EncodedKV:
    scale = None if encoded.scale is None else encoded.scale[start:end]
    zp = None if encoded.zero_point is None else encoded.zero_point[start:end]
    return kv_codecs.EncodedKV(payload=encoded.payload[start:end], scale=scale, zero_point=zp)


def _cat_encoded(left: kv_codecs.EncodedKV, right: kv_codecs.EncodedKV) -> kv_codecs.EncodedKV:
    def _cat_opt(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
        if a is None and b is None:
            return None
        if a is None or b is None:
            raise ValueError("scale/min 不能只在一侧存在")
        return torch.cat([a, b], dim=0)

    return kv_codecs.EncodedKV(
        payload=torch.cat([left.payload, right.payload], dim=0),
        scale=_cat_opt(left.scale, right.scale),
        zero_point=_cat_opt(left.zero_point, right.zero_point),
    )


def _split_along_t(encoded: kv_codecs.EncodedKV, page_size: int) -> list[kv_codecs.EncodedKV]:
    t = encoded.payload.shape[0]
    return [
        _slice_encoded_t(encoded, start, min(start + page_size, t))
        for start in range(0, t, page_size)
    ]


def _split_kivi_key_pages(
    encoded: kv_codecs.EncodedKV,
    *,
    page_size: int,
    group_size: int,
) -> list[kv_codecs.EncodedKV]:
    """Key group 的 scale/min 只挂在第一页，与 R1 ``paged_cache`` 相同。"""
    if encoded.scale is None or encoded.zero_point is None:
        raise ValueError("KIVI Key 切页需要 scale 与 min")
    t = encoded.payload.shape[0]
    if t % group_size != 0:
        raise ValueError(f"Key payload T={t} 不能被 group_size={group_size} 整除")
    if group_size % page_size != 0:
        raise ValueError(f"group_size={group_size} 不能被 page_size={page_size} 整除")
    pages_per_group = group_size // page_size
    n_groups = t // group_size
    pages: list[kv_codecs.EncodedKV] = []
    for group in range(n_groups):
        t0 = group * group_size
        scale_g = encoded.scale[group : group + 1]
        zp_g = encoded.zero_point[group : group + 1]
        for page in range(pages_per_group):
            start = t0 + page * page_size
            sl = encoded.payload[start : start + page_size]
            if page == 0:
                pages.append(kv_codecs.EncodedKV(payload=sl, scale=scale_g, zero_point=zp_g))
            else:
                pages.append(kv_codecs.EncodedKV(payload=sl))
    return pages


def _cat_meta(parts: list[torch.Tensor | None]) -> torch.Tensor | None:
    if all(item is None for item in parts):
        return None
    if any(item is None for item in parts):
        raise ValueError("scale/min 不能只在部分块存在")
    return torch.cat(parts, dim=0)
