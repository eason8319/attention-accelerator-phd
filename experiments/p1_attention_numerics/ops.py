"""RoPE and RMSNorm operators used in decoder attention datapaths."""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables for RoPE.

    Returns:
        cos, sin: each of shape (seq_len, head_dim).
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    offset: int = 0,
) -> torch.Tensor:
    """Apply rotary positional embedding.

    Args:
        x: (..., seq, head_dim)
        cos, sin: (max_seq, head_dim) or broadcastable tables
        offset: position offset for decode (KV cache length before current token)
    """
    seq_len = x.shape[-2]
    cos_slice = cos[offset : offset + seq_len]
    sin_slice = sin[offset : offset + seq_len]

    while cos_slice.ndim < x.ndim:
        cos_slice = cos_slice.unsqueeze(0)
        sin_slice = sin_slice.unsqueeze(0)

    return (x * cos_slice) + (_rotate_half(x) * sin_slice)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root-mean-square layer normalization (LLaMA style)."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight
