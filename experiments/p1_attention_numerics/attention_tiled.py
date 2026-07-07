"""Two-pass tiled attention: global row-max/sum then weighted accumulation."""

from __future__ import annotations

import torch

from attention_naive import _causal_mask


def _iter_kv_blocks(seq_len: int, block_size: int):
    for start in range(0, seq_len, block_size):
        yield start, min(start + block_size, seq_len)


def attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """Two-pass block-wise attention equivalent to the naive implementation.

    Pass 1a: find global row max across all KV blocks.
    Pass 1b: accumulate softmax denominator using the global row max.
    Pass 2: accumulate weighted V using the global normalizer.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, heads, seq_len, head_dim = q.shape
    scale = head_dim**-0.5
    device, dtype = q.device, q.dtype

    row_max = torch.full((batch, heads, seq_len), float("-inf"), device=device, dtype=dtype)
    causal_mask = _causal_mask(seq_len, device, dtype) if causal else None

    # Pass 1a: global row max.
    for kv_start, kv_end in _iter_kv_blocks(seq_len, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]
        row_max = torch.maximum(row_max, scores.amax(dim=-1))

    # Pass 1b: denominator with the fixed global max.
    row_sum = torch.zeros((batch, heads, seq_len), device=device, dtype=dtype)
    for kv_start, kv_end in _iter_kv_blocks(seq_len, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]
        row_sum = row_sum + torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)

    # Pass 2: weighted value accumulation.
    out = torch.zeros((batch, heads, seq_len, head_dim), device=device, dtype=dtype)
    for kv_start, kv_end in _iter_kv_blocks(seq_len, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        v_blk = v[:, :, kv_start:kv_end, :]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]

        weights = torch.exp(scores - row_max.unsqueeze(-1)) / row_sum.unsqueeze(-1)
        out = out + torch.matmul(weights, v_blk)

    return out
