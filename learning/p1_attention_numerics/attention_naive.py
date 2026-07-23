"""标准缩放点积 attention（参考实现）。"""

from __future__ import annotations

import torch


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """上三角 mask：位置 j > i 的项被屏蔽。"""
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
    """计算 softmax(QK^T / sqrt(d)) V。

    参数:
        q, k, v: 形状为 (batch, heads, seq, head_dim) 的张量。
        causal: 若为 True，应用因果（下三角）mask。

    返回:
        形状为 (batch, heads, seq, head_dim) 的输出张量。
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q、k、v 的形状须为 (batch, heads, seq, head_dim)")

    head_dim = q.shape[-1]
    scale = head_dim**-0.5

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        scores = scores + _causal_mask(seq_len, scores.device, scores.dtype)

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)
