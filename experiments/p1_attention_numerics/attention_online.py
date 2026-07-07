"""FlashAttention-style online softmax attention (single-pass over KV blocks)."""

from __future__ import annotations

import torch

from attention_naive import _causal_mask
from attention_tiled import _iter_kv_blocks


def attention_online(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """Online softmax attention following FlashAttention Algorithm 1.

    Maintains running row max ``m``, running sum ``l``, and output accumulator ``o``
    while streaming KV blocks. Result is mathematically identical to naive attention
    but independent of ``block_size`` (up to floating-point non-associativity).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, heads, seq_len, head_dim = q.shape
    scale = head_dim**-0.5
    device, dtype = q.device, q.dtype

    m = torch.full((batch, heads, seq_len), float("-inf"), device=device, dtype=dtype)
    l = torch.zeros((batch, heads, seq_len), device=device, dtype=dtype)
    o = torch.zeros((batch, heads, seq_len, head_dim), device=device, dtype=dtype)

    causal_mask = _causal_mask(seq_len, device, dtype) if causal else None

    for kv_start, kv_end in _iter_kv_blocks(seq_len, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        v_blk = v[:, :, kv_start:kv_end, :]

        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]

        m_blk = scores.amax(dim=-1)
        m_new = torch.maximum(m, m_blk)

        # Rescale previous accumulator and incorporate the new block.
        alpha = torch.exp(m - m_new)
        p_blk = torch.exp(scores - m_new.unsqueeze(-1))

        l = alpha * l + p_blk.sum(dim=-1)
        o = alpha.unsqueeze(-1) * o + torch.matmul(p_blk, v_blk)
        m = m_new

    return o / l.unsqueeze(-1)
