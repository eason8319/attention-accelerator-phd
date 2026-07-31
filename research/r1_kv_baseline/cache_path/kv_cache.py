"""Contiguous / KIVI 残差窗 KV Cache：quantize → pack/store → load → dequant。"""

from __future__ import annotations

import torch

from kv_codecs import EncodedKV, KVCodec, KiviKeyCodec, KiviValueCodec


def _validate_kv_append(
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> int:
    """校验 append 输入形状，返回 n_tokens。"""
    if k_t.shape != v_t.shape:
        raise ValueError(f"k_t 与 v_t 形状不一致: {k_t.shape} vs {v_t.shape}")
    if k_t.ndim != 3:
        raise ValueError(f"期望形状 (n_tokens, num_heads, head_dim)，得到 {k_t.shape}")
    n_tokens, n_heads, d = k_t.shape
    if n_heads != num_heads or d != head_dim:
        raise ValueError(f"期望 (*, {num_heads}, {head_dim})，得到 {k_t.shape}")
    if n_tokens <= 0:
        raise ValueError("n_tokens 须为正")
    return n_tokens


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
        n_tokens = _validate_kv_append(
            k_t, v_t, num_heads=self.num_heads, head_dim=self.head_dim
        )
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


class KiviKVCache:
    """KIVI 风格 cache：历史低比特 + 近期 FP16 残差窗。

    对齐官方（jy-yuan/KIVI）流式语义：

    - **Key**：残差累积至 ``residual_length`` 的整数倍时，将完整窗刷入
      per-channel 量化（刷完后残差可为空）。
    - **Value**：滑动窗——始终保留最近 ``residual_length`` token 为 FP16，
      溢出的更早 token 做 per-token 量化。

    ``residual_length`` 须能被 ``group_size`` 整除（Key 整窗量化前提）。
    形状约定与 ``ContiguousKVCache`` 相同：``(n_tokens, num_heads, head_dim)``。
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
        device: torch.device | None = None,
    ) -> None:
        k_bits = bits if k_bits is None else k_bits
        v_bits = bits if v_bits is None else v_bits
        if residual_length <= 0:
            raise ValueError(f"residual_length 须为正，得到 {residual_length}")
        if residual_length % group_size != 0:
            raise ValueError(
                f"residual_length={residual_length} 须能被 group_size={group_size} 整除"
            )
        if head_dim % group_size != 0:
            raise ValueError(
                f"head_dim={head_dim} 须能被 group_size={group_size} 整除（Value 分组）"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.residual_length = residual_length
        self.device = device or torch.device("cpu")
        self.k_codec = KiviKeyCodec(bits=k_bits, group_size=group_size)
        self.v_codec = KiviValueCodec(bits=v_bits, group_size=group_size)

        self._k_quant: list[EncodedKV] = []
        self._v_quant: list[EncodedKV] = []
        self._k_residual: torch.Tensor | None = None  # FP16 (T_r, H, D)
        self._v_residual: torch.Tensor | None = None
        self._seq_len: int = 0

    def __len__(self) -> int:
        return self._seq_len

    @property
    def k_residual_len(self) -> int:
        """当前 Key 残差窗 token 数。"""
        return 0 if self._k_residual is None else int(self._k_residual.shape[0])

    @property
    def v_residual_len(self) -> int:
        """当前 Value 残差窗 token 数。"""
        return 0 if self._v_residual is None else int(self._v_residual.shape[0])

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """追加 K/V，并按 KIVI 规则刷残差窗（支持 n=1 decode 与 n>1 prefill）。"""
        n_tokens = _validate_kv_append(
            k_t, v_t, num_heads=self.num_heads, head_dim=self.head_dim
        )
        k_t = k_t.to(device=self.device, dtype=torch.float16)
        v_t = v_t.to(device=self.device, dtype=torch.float16)

        self._k_residual = (
            k_t if self._k_residual is None else torch.cat([self._k_residual, k_t], dim=0)
        )
        self._v_residual = (
            v_t if self._v_residual is None else torch.cat([self._v_residual, v_t], dim=0)
        )
        self._seq_len += n_tokens
        self._flush_key_residual()
        self._flush_value_residual()

    def _flush_key_residual(self) -> None:
        """Key：从残差前端刷出 ``residual_length`` 的整数倍完整窗。"""
        r = self._k_residual
        if r is None:
            return
        n_flush = (r.shape[0] // self.residual_length) * self.residual_length
        if n_flush <= 0:
            return
        self._k_quant.append(self.k_codec.encode(r[:n_flush].float()))
        rest = r[n_flush:]
        self._k_residual = None if rest.shape[0] == 0 else rest

    def _flush_value_residual(self) -> None:
        """Value：溢出部分量化，保留最近 ``residual_length`` 为 FP16。"""
        r = self._v_residual
        if r is None:
            return
        if r.shape[0] <= self.residual_length:
            return
        n_overflow = r.shape[0] - self.residual_length
        self._v_quant.append(self.v_codec.encode(r[:n_overflow].float()))
        self._v_residual = r[n_overflow:]

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """解码量化历史并拼接残差窗，返回 float32 ``(seq_len, H, D)``。"""
        empty = (0, self.num_heads, self.head_dim)
        if self._seq_len == 0:
            z = torch.empty(empty, device=self.device, dtype=torch.float32)
            return z, z.clone()

        parts_k: list[torch.Tensor] = [self.k_codec.decode(c) for c in self._k_quant]
        parts_v: list[torch.Tensor] = [self.v_codec.decode(c) for c in self._v_quant]
        if self._k_residual is not None:
            parts_k.append(self._k_residual.float())
        if self._v_residual is not None:
            parts_v.append(self._v_residual.float())

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
        """清空量化历史与残差窗。"""
        self._k_quant.clear()
        self._v_quant.clear()
        self._k_residual = None
        self._v_residual = None
        self._seq_len = 0

    def bytes_stored(self) -> tuple[int, int]:
        """返回 ``(payload_bytes, metadata_bytes)``。

        残差窗按 FP16 计入 payload；量化段按对应 codec 记账。
        """
        payload = 0
        metadata = 0
        for chunk in self._k_quant:
            payload += self.k_codec.bytes_payload(chunk)
            metadata += self.k_codec.bytes_metadata(chunk)
        for chunk in self._v_quant:
            payload += self.v_codec.bytes_payload(chunk)
            metadata += self.v_codec.bytes_metadata(chunk)
        if self._k_residual is not None:
            payload += int(self._k_residual.numel() * self._k_residual.element_size())
        if self._v_residual is not None:
            payload += int(self._v_residual.numel() * self._v_residual.element_size())
        return payload, metadata
