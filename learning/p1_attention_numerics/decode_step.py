"""带 KV cache 的单 token decode 步 attention。"""

from __future__ import annotations

import torch


def decode_step_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    """单个新 query token 对缓存 K/V 的 attention。

    参数:
        q: (batch, heads, 1, head_dim) — 当前 token 的 query。
        k_cache: (batch, heads, cache_len, head_dim)
        v_cache: (batch, heads, cache_len, head_dim)

    返回:
        (batch, heads, 1, head_dim)
    """
    if q.shape[-2] != 1:
        raise ValueError("decode_step_attention 要求单个 query token（seq=1）")

    head_dim = q.shape[-1]
    scale = head_dim**-0.5

    scores = torch.matmul(q, k_cache.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v_cache)


def prefill_last_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """从完整 prefill attention 中取出最后一行 query 的输出。"""
    from attention_naive import attention

    out = attention(q, k, v, causal=causal)
    return out[:, :, -1:, :]
