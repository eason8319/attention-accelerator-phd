# P1 — Attention 数值内核复现

详细步骤与验收标准见 [docs/learning_plan.md](../../docs/learning_plan.md) 的 P1 一节。

## 目标

从零实现标准 attention → 分块 attention → online softmax（FlashAttention 前向），
与 `torch.nn.functional.scaled_dot_product_attention` 数值对拍，
建立后续 RTL 设计（P4）的 golden model。

## 建议文件结构

```
p1_attention_numerics/
├── attention_naive.py      # 标准 attention
├── attention_tiled.py      # 两遍分块 attention
├── attention_online.py     # online softmax 单遍实现
├── ops.py                  # RoPE / RMSNorm
├── decode_step.py          # 单 token decode attention
└── test_numerics.py        # 对拍测试（pytest）
```

## 从这里开始

第一步：写 `attention_naive.py`，实现

```python
def attention(q, k, v, causal: bool = False) -> torch.Tensor:
    # q/k/v: (batch, heads, seq, head_dim)
    ...
```

并用一个 10 行的 pytest 与 `F.scaled_dot_product_attention` 对拍（fp32，max abs error < 1e-5）。

## 验收 checklist

- [x] 标准 / 分块 / online 三种实现对拍通过（fp32 误差 < 1e-5）
- [x] online softmax 结果与块大小无关
- [x] RoPE、RMSNorm 与 transformers 参考实现一致
- [x] decode-step 与 prefill 全量结果一致
- [x] 一页 online softmax rescale 推导笔记

验收记录见 [P1 Attention 数值内核复现验收报告](../../docs/progress/p1_attention_numerics_report.md)。