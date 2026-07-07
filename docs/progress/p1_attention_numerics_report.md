# P1 Attention 数值内核复现验收报告

日期：2026-07-07

## 结论

P1 已完成验收。当前实现覆盖标准 attention、两遍分块 attention、online softmax attention、RoPE、RMSNorm 与单 token decode attention，并已通过 pytest 数值对拍。

验证命令：

```bash
conda run -n p1-attention pytest experiments/p1_attention_numerics/test_numerics.py -q --tb=short
```

验证结果：

```text
22 passed
```

## 验收 Checklist

| learning_plan.md 验收项 | 对应产出 | 状态 |
|---|---|---|
| 三种实现（标准/分块/online）对拍误差 fp32 下 max abs error < 1e-5 | `attention_naive.py`、`attention_tiled.py`、`attention_online.py`、`test_numerics.py` | 通过 |
| online softmax 支持任意块大小且结果与块大小无关 | `test_online_block_size_independence` 覆盖 `block_size=16/128` | 通过 |
| RoPE、RMSNorm 与 transformers 参考实现一致 | `ops.py`，对拍 `LlamaRotaryEmbedding`、`LlamaRMSNorm` | 通过 |
| decode-step attention 与 prefill 全量计算结果一致 | `decode_step.py`，`test_decode_matches_prefill_last_token` | 通过 |
| 写一页 online softmax 的 rescale 推导笔记 | `online_softmax_rescale_notes.md` | 完成 |

## 产出说明

- `attention_naive.py`：标准 scaled dot-product attention，作为 P1 golden reference。
- `attention_tiled.py`：按 KV block 扫描，先求全局 row-max/row-sum，再累加输出，用于理解分块 softmax 的全局归一化需求。
- `attention_online.py`：FlashAttention-style 单遍 online softmax，维护 running max `m`、running sum `l` 和输出累加器 `o`，在块间通过 `alpha = exp(m - m_new)` rescale 旧状态。
- `ops.py`：RoPE 与 RMSNorm，实现 decoder attention datapath 中 attention 前后的基础算子。
- `decode_step.py`：单 query token 对 KV cache 的 decode-step attention，并与 prefill 最后一行对拍。
- `test_numerics.py`：统一数值对拍测试，覆盖 causal/non-causal、不同 block size、fp16 online、RoPE/RMSNorm/decode。
- `online_softmax_rescale_notes.md`：online softmax rescale 推导与硬件设计要点。

## 环境记录

- Conda 环境：`p1-attention`
- PyTorch：`2.12.1+cpu`
- Transformers：`5.13.0`

CPU 版 PyTorch 已满足 P1 数值内核复现与测试需求。

## 后续衔接

P1 产出的 attention golden model 可继续用于：

- P2：低精度量化实验中评估 Q/K/V、softmax 与 KV cache 量化误差。
- P4：RTL exp/softmax/attention 数据流模块的功能对拍。
- P5：tile-level simulator 中校验 attention tile 数据流的数值语义。
