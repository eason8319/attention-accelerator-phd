"""物理打包后的连续 KV 追加：沿用 R1 统计量，只改载荷表示。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


@dataclass(frozen=True)
class StorageBytes:
    """缓存张量实际字节与布局预算；allocated 字段不是 PyTorch 分配实测。"""

    payload_packed: int
    payload_allocated: int
    scale: int
    scale_allocated: int
    offset: int
    offset_allocated: int
    open_group_fp16: int
    pack_pad_values: int
    tensor_storage_bytes: int

    @property
    def alignment_waste(self) -> int:
        """DMA 整段取整多出的字节。"""
        return (
            self.payload_allocated
            - self.payload_packed
            + self.scale_allocated
            - self.scale
            + self.offset_allocated
            - self.offset
        )

    @property
    def allocated_total(self) -> int:
        """冻结布局所需字节，含分侧整段垫齐与开口组 FP16。"""
        return (
            self.payload_allocated
            + self.scale_allocated
            + self.offset_allocated
            + self.open_group_fp16
        )


def _cat_opt(parts: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [p for p in parts if p is not None]
    if not present:
        return None
    if len(present) != len(parts):
        raise ValueError("scale/offset 不能只在部分片段存在")
    return torch.cat(present, dim=0)


def _dma_bytes(n: int, layout: PackLayout) -> int:
    return align_up(n, layout.dma_align_bytes)


class PackedContiguousCache:
    """C0/C2/C3 连续物理打包缓存，以及 C5 的 group 对齐打包（无 128 残差窗）。

    已提交片段只保留 packed 字节与 FP16 元数据，不保留完整高精度 KV 副本。
    ``load`` 仅供对拍，把存储解包后交给 R1 decode，不是流式 Attention 路径。
    """

    def __init__(
        self,
        format_id: str,
        *,
        num_heads: int,
        head_dim: int,
        layout: PackLayout,
        device: torch.device | None = None,
    ) -> None:
        key = format_id.strip().upper()
        if key not in {"C0", "C1", "C2", "C3", "C5"}:
            raise ValueError(f"PackedContiguousCache 支持 C0/C1/C2/C3/C5，得到 {format_id!r}")
        if head_dim % layout.group_size != 0:
            raise ValueError(f"head_dim={head_dim} 须能被 group_size={layout.group_size} 整除")
        self.format_id = key
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layout = layout
        self.device = device or torch.device("cpu")
        self._k_packed: list[torch.Tensor] = []
        self._v_packed: list[torch.Tensor] = []
        self._k_scale: list[torch.Tensor] = []
        self._v_scale: list[torch.Tensor] = []
        self._k_offset: list[torch.Tensor] = []
        self._v_offset: list[torch.Tensor] = []
        self._k_open: torch.Tensor | None = None
        self._seq_len = 0
        self._k_logical = 0
        self._v_logical = 0
        self._pack_pad_values = 0
        self._codec = None
        if key == "C0":
            self._codec = kv_codecs.FP16Codec()
        elif key == "C1":
            self._codec = kv_codecs.Int8Codec(symmetric=True)
        elif key == "C2":
            self._codec = kv_codecs.Int4Codec(symmetric=True, group_size=layout.group_size)
        elif key == "C3":
            self._codec = kv_codecs.Int4BdrCodec(
                symmetric=True,
                group_size=layout.group_size,
                block_size=32,
                seed=0,
                dim=head_dim,
            )

    def __len__(self) -> int:
        return self._seq_len

    @property
    def k_open_len(self) -> int:
        """C5 Key 开口组 token 数；其他格式为 0。"""
        return 0 if self._k_open is None else int(self._k_open.shape[0])

    def rotation_sha256(self) -> str | None:
        """C3 旋转矩阵字节摘要；其他格式为 None。"""
        import hashlib

        if self.format_id != "C3":
            return None
        matrix = self._codec._rot.matrix.detach().cpu().contiguous()
        return hashlib.sha256(matrix.numpy().tobytes()).hexdigest()

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """写入 float K/V；已提交 packed 片段不再重新量化。"""
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
        k_t = k_t.to(device=self.device, dtype=torch.float16)
        v_t = v_t.to(device=self.device, dtype=torch.float16)
        if self.format_id == "C5":
            self._append_c5(k_t, v_t)
        else:
            self._append_uniform(k_t, v_t)
        self._seq_len += n_tokens

    def _append_uniform(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        self._commit_uniform("k", k_t)
        self._commit_uniform("v", v_t)

    def _commit_uniform(self, side: str, x: torch.Tensor) -> None:
        encoded = self._codec.encode(x)
        packed, pad = self._pack_r1_payload(encoded.payload)
        packed_list, scale_list, offset_list, logical_attr = self._side_lists(side)
        packed_list.append(packed)
        if encoded.scale is not None:
            scale_list.append(encoded.scale.contiguous())
        if encoded.zero_point is not None:
            offset_list.append(encoded.zero_point.contiguous())
        setattr(self, logical_attr, getattr(self, logical_attr) + int(x.shape[0]))
        self._pack_pad_values += pad

    def _append_c5(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        open_k = k_t if self._k_open is None else torch.cat([self._k_open, k_t], dim=0)
        n_flush = (open_k.shape[0] // self.layout.group_size) * self.layout.group_size
        if n_flush > 0:
            encoded = kv_codecs.encode_kivi_key(
                open_k[:n_flush].float(),
                bits=4,
                group_size=self.layout.group_size,
            )
            packed, pad = self._pack_r1_payload(encoded.payload, unsigned_bits=4)
            self._k_packed.append(packed)
            self._k_scale.append(encoded.scale.contiguous())
            self._k_offset.append(encoded.zero_point.contiguous())
            self._k_logical += n_flush
            self._pack_pad_values += pad
            rest = open_k[n_flush:]
            # 切片和同 dtype 的 to() 不释放父存储；只拥有剩余 token 的独立副本。
            self._k_open = (
                None if rest.shape[0] == 0
                else rest.detach().clone(memory_format=torch.contiguous_format)
            )
        else:
            # 即使尚未提交满组，也不能借用调用方可复用的输入缓冲。
            self._k_open = open_k.detach().clone(memory_format=torch.contiguous_format)
        if self.k_open_len > self.layout.c5_key_open_group_max_tokens:
            raise RuntimeError("C5 Key 开口组超过 31 token，打包边界被破坏")
        encoded_v = kv_codecs.encode_kivi_value(
            v_t.float(),
            bits=4,
            group_size=self.layout.group_size,
        )
        packed_v, pad_v = self._pack_r1_payload(encoded_v.payload, unsigned_bits=4)
        self._v_packed.append(packed_v)
        self._v_scale.append(encoded_v.scale.contiguous())
        self._v_offset.append(encoded_v.zero_point.contiguous())
        self._v_logical += int(v_t.shape[0])
        self._pack_pad_values += pad_v

    def _side_lists(self, side: str) -> tuple[list, list, list, str]:
        if side == "k":
            return self._k_packed, self._k_scale, self._k_offset, "_k_logical"
        if side == "v":
            return self._v_packed, self._v_scale, self._v_offset, "_v_logical"
        raise ValueError(side)

    def _pack_r1_payload(
        self,
        payload: torch.Tensor,
        *,
        unsigned_bits: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        if self.format_id == "C0":
            return pack_fp16_bytes(payload), 0
        if self.format_id == "C1":
            return pack_int8_bytes(payload), 0
        if unsigned_bits is not None:
            packed, pad = pack_unsigned_bits(payload, unsigned_bits)
            return packed, pad
        packed, pad = pack_signed_int4(
            payload,
            qmin=self.layout.int4_qmin,
            qmax=self.layout.int4_qmax,
        )
        return packed, pad

    def snapshot_packed_prefix(self) -> dict[str, torch.Tensor]:
        """已提交 packed 字节的副本，用于检查后续追加不改写历史。"""
        out: dict[str, torch.Tensor] = {}
        if self._k_packed:
            out["k"] = torch.cat(self._k_packed, dim=0).clone()
        if self._v_packed:
            out["v"] = torch.cat(self._v_packed, dim=0).clone()
        if self._k_scale:
            out["k_scale"] = torch.cat(self._k_scale, dim=0).clone()
        if self._v_scale:
            out["v_scale"] = torch.cat(self._v_scale, dim=0).clone()
        if self._k_offset:
            out["k_offset"] = torch.cat(self._k_offset, dim=0).clone()
        if self._v_offset:
            out["v_offset"] = torch.cat(self._v_offset, dim=0).clone()
        return out

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """解包并经 R1 decode 得到 float32 ``(seq_len, H, D)``。"""
        empty = (0, self.num_heads, self.head_dim)
        if self._seq_len == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone()
        k = self._decode_side("k")
        v = self._decode_side("v")
        if k.shape[0] != self._seq_len or v.shape[0] != self._seq_len:
            raise RuntimeError(
                f"解码长度与 seq_len 不一致: k={k.shape[0]}, v={v.shape[0]}, seq_len={self._seq_len}"
            )
        return k, v

    def _decode_side(self, side: str) -> torch.Tensor:
        packed_list, scale_list, offset_list, logical_attr = self._side_lists(side)
        logical = getattr(self, logical_attr)
        parts: list[torch.Tensor] = []
        if packed_list:
            packed = torch.cat(packed_list, dim=0)
            payload = self._unpack_payload(packed, logical)
            scale = _cat_opt(scale_list) if scale_list else None
            offset = _cat_opt(offset_list) if offset_list else None
            encoded = kv_codecs.EncodedKV(payload=payload, scale=scale, zero_point=offset)
            parts.append(self._decode_encoded(encoded, side=side, n_tokens=logical))
        if side == "k" and self._k_open is not None:
            parts.append(self._k_open.float())
        if not parts:
            raise RuntimeError("非空 cache 缺少解码片段")
        return torch.cat(parts, dim=0)

    def _unpack_payload(self, packed: torch.Tensor, n_tokens: int) -> torch.Tensor:
        numel = n_tokens * self.num_heads * self.head_dim
        shape = (n_tokens, self.num_heads, self.head_dim)
        if self.format_id == "C0":
            return unpack_fp16_bytes(packed, numel).reshape(shape).to(torch.float16)
        if self.format_id == "C1":
            return unpack_int8_bytes(packed, numel).reshape(shape)
        if self.format_id == "C5":
            grid = unpack_unsigned_bits(packed, 4, numel)
            return grid.to(torch.uint8).reshape(shape)
        grid = unpack_signed_int4(
            packed,
            numel,
            qmin=self.layout.int4_qmin,
            qmax=self.layout.int4_qmax,
        )
        return grid.reshape(shape)

    def _decode_encoded(
        self, encoded: kv_codecs.EncodedKV, *, side: str, n_tokens: int
    ) -> torch.Tensor:
        if self.format_id == "C5":
            if side == "k":
                return kv_codecs.decode_kivi_key(encoded, group_size=self.layout.group_size)
            return kv_codecs.decode_kivi_value(encoded, group_size=self.layout.group_size)
        return self._codec.decode(encoded)

    def storage_bytes(self) -> StorageBytes:
        """实测张量存储；另按 K/V 各自整段缓冲计算 32 B 布局预算。"""
        payload = _list_nbytes(self._k_packed) + _list_nbytes(self._v_packed)
        scale = _list_nbytes(self._k_scale) + _list_nbytes(self._v_scale)
        offset = _list_nbytes(self._k_offset) + _list_nbytes(self._v_offset)
        open_fp16 = 0 if self._k_open is None else self._k_open.untyped_storage().nbytes()
        buffers = [
            *self._k_packed, *self._v_packed, *self._k_scale, *self._v_scale,
            *self._k_offset, *self._v_offset,
        ]
        if self._k_open is not None:
            buffers.append(self._k_open)
        return StorageBytes(
            payload_packed=payload,
            payload_allocated=sum(
                _dma_bytes(_list_nbytes(parts), self.layout)
                for parts in (self._k_packed, self._v_packed)
            ),
            scale=scale,
            scale_allocated=sum(
                _dma_bytes(_list_nbytes(parts), self.layout)
                for parts in (self._k_scale, self._v_scale)
            ),
            offset=offset,
            offset_allocated=sum(
                _dma_bytes(_list_nbytes(parts), self.layout)
                for parts in (self._k_offset, self._v_offset)
            ),
            open_group_fp16=open_fp16,
            pack_pad_values=self._pack_pad_values,
            tensor_storage_bytes=_unique_storage_nbytes(buffers),
        )

    def committed_k_tokens(self) -> int:
        """已打包的 Key token 数，不含 C5 开口组。"""
        return self._k_logical


def _list_nbytes(parts: list[torch.Tensor]) -> int:
    return sum(tensor_nbytes(p) for p in parts)


def _unique_storage_nbytes(parts: list[torch.Tensor]) -> int:
    """计算缓存缓冲持有的底层存储；同一存储的多个视图只计一次。"""
    storages = {}
    for tensor in parts:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def r1_reference_kv(
    format_id: str,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, object, object]:
    """同一输入上的 R1 encode/decode，作为打包层不得改变的数值参考。

    C5 Key 只对 ``T`` 能被 group_size 整除的前缀做 R1 参考；余数保持 FP16。
    返回 ``(k_ref, v_ref, k_encoded_or_none, v_encoded_or_none)``。
    """
    key = format_id.strip().upper()
    if key == "C5":
        t = k.shape[0]
        n_flush = (t // group_size) * group_size
        parts_k: list[torch.Tensor] = []
        k_enc = None
        if n_flush > 0:
            k_enc = kv_codecs.encode_kivi_key(k[:n_flush].float(), bits=4, group_size=group_size)
            parts_k.append(kv_codecs.decode_kivi_key(k_enc, group_size=group_size))
        if t > n_flush:
            parts_k.append(k[n_flush:].to(torch.float16).float())
        v_enc = kv_codecs.encode_kivi_value(v.float(), bits=4, group_size=group_size)
        v_ref = kv_codecs.decode_kivi_value(v_enc, group_size=group_size)
        k_ref = torch.cat(parts_k, dim=0) if parts_k else k.float()
        return k_ref, v_ref, k_enc, v_enc
    if key == "C0":
        codec = kv_codecs.FP16Codec()
    elif key == "C1":
        codec = kv_codecs.Int8Codec(symmetric=True)
    elif key == "C2":
        codec = kv_codecs.Int4Codec(symmetric=True, group_size=group_size)
    elif key == "C3":
        codec = kv_codecs.Int4BdrCodec(
            symmetric=True,
            group_size=group_size,
            block_size=32,
            seed=0,
            dim=k.shape[-1],
        )
    else:
        raise ValueError(format_id)
    k_enc = codec.encode(k)
    v_enc = codec.encode(v)
    return codec.decode(k_enc), codec.decode(v_enc), k_enc, v_enc


def r1_chunked_reference(
    format_id: str,
    k_chunks: list[torch.Tensor],
    v_chunks: list[torch.Tensor],
    *,
    group_size: int,
):
    """按与 cache 相同的追加切块做 R1 参考。

    C0–C3 对每块独立 encode（C3 共用一个旋转实例）。C5 Key 跨块累积到
    group_size 再编码，与 ``PackedContiguousCache`` 一致。C3 逐 token 旋转
    后 INT4 可能与整段一次 encode 差一档，这是 R1 既有性质，不是打包错误。
    """
    key = format_id.strip().upper()
    if key == "C5":
        k = torch.cat(k_chunks, dim=0)
        v = torch.cat(v_chunks, dim=0)
        k_ref, v_ref, k_enc, v_enc = r1_reference_kv(key, k, v, group_size=group_size)
        k_list = [] if k_enc is None else [k_enc]
        return k_ref, v_ref, k_list, [v_enc]
    if key == "C0":
        codec = kv_codecs.FP16Codec()
    elif key == "C1":
        codec = kv_codecs.Int8Codec(symmetric=True)
    elif key == "C2":
        codec = kv_codecs.Int4Codec(symmetric=True, group_size=group_size)
    elif key == "C3":
        codec = kv_codecs.Int4BdrCodec(
            symmetric=True,
            group_size=group_size,
            block_size=32,
            seed=0,
            dim=k_chunks[0].shape[-1],
        )
    else:
        raise ValueError(format_id)
    k_encs: list = []
    v_encs: list = []
    for k_t, v_t in zip(k_chunks, v_chunks, strict=True):
        k_encs.append(codec.encode(k_t))
        v_encs.append(codec.encode(v_t))
    # 与 cache.load 相同：先拼接已编码网格，再做一次 decode（C3 逆旋转作用在整段上）。
    k_ref = codec.decode(_cat_encoded(k_encs))
    v_ref = codec.decode(_cat_encoded(v_encs))
    return k_ref, v_ref, k_encs, v_encs


def _cat_encoded(parts: list) -> object:
    """沿 token 维拼接 R1 EncodedKV。"""
    payload = torch.cat([p.payload for p in parts], dim=0)

    def _cat_meta(name: str):
        vals = [getattr(p, name) for p in parts]
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            raise ValueError(f"{name} 不能只在部分片段存在")
        return torch.cat(vals, dim=0)

    return kv_codecs.EncodedKV(
        payload=payload,
        scale=_cat_meta("scale"),
        zero_point=_cat_meta("zero_point"),
    )


def r1_in_memory_payload_bytes(encoded_k, encoded_v) -> int:
    """R1 实际张量存储字节（INT4 为 int8 网格，不是名义 nibble）。"""
    total = 0
    for enc in (encoded_k, encoded_v):
        if enc is None:
            continue
        total += tensor_nbytes(enc.payload)
    return total


def r1_nominal_payload_bytes(format_id: str, encoded_k, encoded_v, *, group_size: int) -> int:
    """R1 ``bytes_payload`` 记账口径。"""
    key = format_id.strip().upper()
    total = 0
    if key == "C5":
        k_codec = kv_codecs.KiviKeyCodec(bits=4, group_size=group_size)
        v_codec = kv_codecs.KiviValueCodec(bits=4, group_size=group_size)
        if encoded_k is not None:
            total += k_codec.bytes_payload(encoded_k)
        if encoded_v is not None:
            total += v_codec.bytes_payload(encoded_v)
        return total
    if key == "C0":
        codec = kv_codecs.FP16Codec()
    elif key == "C1":
        codec = kv_codecs.Int8Codec(symmetric=True)
    else:
        # C2/C3 名义载荷均为 0.5 B/元素，不依赖旋转矩阵。
        codec = kv_codecs.Int4Codec(symmetric=True, group_size=group_size)
    if encoded_k is not None:
        total += codec.bytes_payload(encoded_k)
    if encoded_v is not None:
        total += codec.bytes_payload(encoded_v)
    return total
