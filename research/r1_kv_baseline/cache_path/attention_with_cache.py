"""经 ContiguousKVCache 的 decode attention（写 cache → load → attend）。"""

from __future__ import annotations

import torch

from kv_cache import ContiguousKVCache
from kv_codecs import KVCodec, get_codec


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """标准缩放点积 attention。

    约定形状均为 ``(batch, num_heads, seq, head_dim)``；
    decode 时 ``q`` 的 ``seq`` 通常为 1。
    """
    head_dim = q.shape[-1]
    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class AttentionWithCache:
    """持有一个 ContiguousKVCache，提供 prefill 写入与 decode 步进。

    ``codec`` 可为 ``KVCodec`` 实例，或 ``get_codec`` 可识别的格式名
    （如 ``fp16`` / ``int8`` / ``int4`` / ``int4_bdr`` 或 ``C0``–``C3``）。
    字符串路径会传入 ``dim=head_dim``，以便 INT4+BDR 绑定旋转维；
    额外关键字（如 ``seed``）转发给 ``get_codec``。
    """

    def __init__(
        self,
        codec: KVCodec | str,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        **codec_kwargs,
    ) -> None:
        if isinstance(codec, str):
            codec = get_codec(codec, dim=head_dim, **codec_kwargs)
        elif codec_kwargs:
            raise TypeError(
                "传入 KVCodec 实例时不要再给 codec 关键字参数；"
                "请在 get_codec(...) 时设置，或改用格式字符串"
            )
        self.cache = ContiguousKVCache(
            codec, num_heads=num_heads, head_dim=head_dim, device=device
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = self.cache.device

    @property
    def codec(self) -> KVCodec:
        """当前 cache 使用的编解码器。"""
        return self.cache.codec

    def clear(self) -> None:
        """清空内部 KV cache。"""
        self.cache.clear()

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill：清空后写入整段 K/V，并对 ``q`` 做 attention。

        约定 ``q/k/v`` 形状 ``(batch, num_heads, seq, head_dim)``，
        且当前实现默认 ``batch=1``（与协议一致）。
        当前不做 causal mask（与 ``scaled_dot_product_attention`` 一致）。
        """
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                f"q/k/v 形状须一致，得到 q={q.shape}, k={k.shape}, v={v.shape}"
            )
        if q.ndim != 4 or q.shape[0] != 1:
            raise ValueError(f"期望 (1, num_heads, seq, head_dim)，得到 {q.shape}")
        if q.shape[1] != self.num_heads or q.shape[3] != self.head_dim:
            raise ValueError(
                f"期望 (1, {self.num_heads}, seq, {self.head_dim})，得到 {q.shape}"
            )
        if q.shape[2] <= 0:
            raise ValueError("seq 须为正")

        self.clear()
        q = q.to(device=self.device)
        # (1, H, S, D) → cache 布局 (S, H, D)
        k_cache = k.to(device=self.device).squeeze(0).transpose(0, 1)
        v_cache = v.to(device=self.device).squeeze(0).transpose(0, 1)
        self.cache.append(k_cache, v_cache)

        k_all, v_all = self.cache.load()
        k_attn = k_all.transpose(0, 1).unsqueeze(0)
        v_attn = v_all.transpose(0, 1).unsqueeze(0)
        return scaled_dot_product_attention(q, k_attn, v_attn)

    def decode_step(
        self,
        q: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """Decode 一步：将新 ``k_t/v_t`` 写入 cache，用 ``q`` 对全部 cache 做 attention。

        约定 ``q/k_t/v_t`` 形状 ``(batch, num_heads, 1, head_dim)``。
        """
        if q.shape != k_t.shape or q.shape != v_t.shape:
            raise ValueError(
                f"q/k_t/v_t 形状须一致，得到 q={q.shape}, k={k_t.shape}, v={v_t.shape}"
            )
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
            raise ValueError(f"期望 (1, num_heads, 1, head_dim)，得到 {q.shape}")
        if q.shape[1] != self.num_heads or q.shape[3] != self.head_dim:
            raise ValueError(
                f"期望 (1, {self.num_heads}, 1, {self.head_dim})，得到 {q.shape}"
            )

        q = q.to(device=self.device)
        # (1, H, 1, D) → cache 布局 (1, H, D)
        k_cache = k_t.to(device=self.device).squeeze(0).transpose(0, 1)
        v_cache = v_t.to(device=self.device).squeeze(0).transpose(0, 1)
        self.cache.append(k_cache, v_cache)

        k_all, v_all = self.cache.load()
        # (S, H, D) → (1, H, S, D)
        k_attn = k_all.transpose(0, 1).unsqueeze(0)
        v_attn = v_all.transpose(0, 1).unsqueeze(0)
        return scaled_dot_product_attention(q, k_attn, v_attn)

    def bytes_stored(self) -> tuple[int, int]:
        """转发 ``cache.bytes_stored()``。"""
        return self.cache.bytes_stored()
