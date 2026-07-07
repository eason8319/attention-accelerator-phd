"""Standard scaled dot-product attention (reference implementation)."""

from __future__ import annotations

import torch


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Upper-triangular mask: positions j > i are masked out."""
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return mask


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Compute softmax(QK^T / sqrt(d)) V.

    Args:
        q, k, v: tensors of shape (batch, heads, seq, head_dim).
        causal: if True, apply causal (lower-triangular) masking.

    Returns:
        Output tensor of shape (batch, heads, seq, head_dim).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (batch, heads, seq, head_dim)")

    head_dim = q.shape[-1]
    scale = head_dim**-0.5

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        scores = scores + _causal_mask(seq_len, scores.device, scores.dtype)

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)
