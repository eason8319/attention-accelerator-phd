"""Contiguous KV Cache：quantize → pack/store → load → dequant。"""

from __future__ import annotations

import torch

from kv_codecs import EncodedKV, KVCodec


class ContiguousKVCache:
    """连续布局的 K/V 缓存；写入即编码，读出时解码。"""

    def __init__(
        self,
        codec: KVCodec,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None = None,
    ) -> None:
        self.codec = codec
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cpu")
        # 内部缓冲：分别累积 K/V 的 EncodedKV 片段（或等价结构）
        self._k_chunks: list[EncodedKV] = []
        self._v_chunks: list[EncodedKV] = []
        self._seq_len: int = 0

    def __len__(self) -> int:
        return self._seq_len

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """写入一步（或一段）float K/V：encode 后追加到缓冲。

        约定 ``k_t``/``v_t`` 形状为 ``(n_tokens, num_heads, head_dim)``，
        ``n_tokens`` 通常为 1（decode）也可 >1（prefill 批量写入）。
        """
        if k_t.shape != v_t.shape:
            raise ValueError(f"k_t 与 v_t 形状不一致: {k_t.shape} vs {v_t.shape}")
        if k_t.ndim != 3:
            raise ValueError(f"期望形状 (n_tokens, num_heads, head_dim)，得到 {k_t.shape}")
        n_tokens, n_heads, d = k_t.shape
        if n_heads != self.num_heads or d != self.head_dim:
            raise ValueError(
                f"期望 (*, {self.num_heads}, {self.head_dim})，得到 {k_t.shape}"
            )
        if n_tokens <= 0:
            raise ValueError("n_tokens 须为正")

        k_t = k_t.to(device=self.device)
        v_t = v_t.to(device=self.device)
        self._k_chunks.append(self.codec.encode(k_t))
        self._v_chunks.append(self.codec.encode(v_t))
        self._seq_len += n_tokens

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """解码全部已缓存 K/V，返回 float 张量，形状 ``(seq_len, num_heads, head_dim)``。"""
        empty = (0, self.num_heads, self.head_dim)
        if self._seq_len == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone()

        k = torch.cat([self.codec.decode(c) for c in self._k_chunks], dim=0)
        v = torch.cat([self.codec.decode(c) for c in self._v_chunks], dim=0)
        if k.shape[0] != self._seq_len or v.shape[0] != self._seq_len:
            raise RuntimeError(
                f"解码长度与 _seq_len 不一致: k={k.shape[0]}, v={v.shape[0]}, "
                f"seq_len={self._seq_len}"
            )
        return k, v

    def clear(self) -> None:
        """清空缓冲。"""
        self._k_chunks.clear()
        self._v_chunks.clear()
        self._seq_len = 0

    def bytes_stored(self) -> tuple[int, int]:
        """返回 ``(payload_bytes, metadata_bytes)`` 合计。"""
        payload = 0
        metadata = 0
        for chunk in (*self._k_chunks, *self._v_chunks):
            payload += self.codec.bytes_payload(chunk)
            metadata += self.codec.bytes_metadata(chunk)
        return payload, metadata
