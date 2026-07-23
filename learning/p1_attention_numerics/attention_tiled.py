"""两遍分块 attention：全局行 max/sum，再加权累加。"""

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
    """与 naive 实现等价的按块两遍 attention。

    第一遍 a：在所有 KV 块上求全局行最大值。
    第一遍 b：用全局行 max 累加 softmax 分母。
    第二遍：用全局归一化因子累加加权 V。
    """
    if block_size <= 0:
        raise ValueError("block_size 须为正数")

    batch, heads, seq_len, head_dim = q.shape
    scale = head_dim**-0.5
    device, dtype = q.device, q.dtype

    row_max = torch.full((batch, heads, seq_len), float("-inf"), device=device, dtype=dtype)
    causal_mask = _causal_mask(seq_len, device, dtype) if causal else None

    # 第一遍 a：全局行 max
    for kv_start, kv_end in _iter_kv_blocks(seq_len, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]
        row_max = torch.maximum(row_max, scores.amax(dim=-1))

    # 第一遍 b：在固定全局 max 下求分母
    row_sum = torch.zeros((batch, heads, seq_len), device=device, dtype=dtype)
    for kv_start, kv_end in _iter_kv_blocks(seq_len, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]
        row_sum = row_sum + torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)

    # 第二遍：加权 value 累加
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
