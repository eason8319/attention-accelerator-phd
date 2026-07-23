"""带插桩的 P1 风格 score 行，供 softmax RTL 向量生成。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
P1_DIR = Path(__file__).resolve().parents[2] / "p1_attention_numerics"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(P1_DIR) not in sys.path:
    sys.path.insert(0, str(P1_DIR))


def sample_attention_score_rows(
    *,
    batch: int = 1,
    heads: int = 2,
    seq: int = 64,
    head_dim: int = 32,
    causal: bool = True,
    seed: int = 0,
    max_rows: int = 4,
) -> list[np.ndarray]:
    """返回至多 ``max_rows`` 条完整 attention score 行 ``(S=QK^T/sqrt(d))[b,h,i,:]``。

    与 P1 online softmax 在 exp/rescale 路径之前的输入一致（仅有限项；
    causal ``-inf`` 位置从行中剔除，使 DUT 看到稠密 score 列表）。
    """
    from attention_naive import _causal_mask  # type: ignore

    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, generator=gen)
    k = torch.randn(batch, heads, seq, head_dim, generator=gen)
    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        scores = scores + _causal_mask(seq, q.device, q.dtype)

    rows: list[np.ndarray] = []
    # 优先较长 causal 行（靠后 token 索引），使 RTL 看到多块流
    for b in range(batch):
        for h in range(heads):
            for i in range(seq - 1, -1, -1):
                row = scores[b, h, i].detach().numpy().astype(np.float64)
                finite = np.isfinite(row)
                if not np.any(finite):
                    continue
                rows.append(row[finite])
                if len(rows) >= max_rows:
                    return rows
    return rows
