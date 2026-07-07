"""Single-token decode-step attention with KV cache."""

from __future__ import annotations

import torch


def decode_step_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    """Attention for one new query token against cached keys/values.

    Args:
        q: (batch, heads, 1, head_dim) — current token query.
        k_cache: (batch, heads, cache_len, head_dim)
        v_cache: (batch, heads, cache_len, head_dim)

    Returns:
        (batch, heads, 1, head_dim)
    """
    if q.shape[-2] != 1:
        raise ValueError("decode_step_attention expects a single query token (seq=1)")

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
    """Extract the last query row from a full prefill attention pass."""
    from attention_naive import attention

    out = attention(q, k, v, causal=causal)
    return out[:, :, -1:, :]
