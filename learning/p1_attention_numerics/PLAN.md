# P1 — Attention 数值内核复现（1–2 周）

**目标**：亲手推导并实现 FlashAttention 前向的 online softmax 数据流，建立后续所有硬件设计的 golden model。

## 步骤

1. **标准 attention**：用 PyTorch 实现 `softmax(QK^T / sqrt(d)) V`，支持 causal mask，shape 约定 `(batch, heads, seq, head_dim)`。
2. **分块 attention（两遍法）**：将 K/V 沿序列维分块，先全局求 row-max/row-sum，再第二遍加权求和——理解"为什么需要两遍"。
3. **Online softmax（FlashAttention 前向）**：单遍扫描，维护 running max `m`、running sum `l`、累加器 `O`，每处理一个 KV 块就地 rescale。对照 FlashAttention 论文 Algorithm 1 逐行实现。
4. **数值对拍**：与 `torch.nn.functional.scaled_dot_product_attention` 比较，指标为 max abs error 与 cosine similarity；覆盖 fp32/fp16、多种 seq_len（128 – 8K）、causal 与非 causal。
5. **附加算子**：独立实现 RoPE（cos/sin 旋转）与 RMSNorm，并与 HuggingFace `transformers` 中 LLaMA 实现对拍；明确二者在完整 decoder layer datapath 中的位置。
6. **Decode 模式**：实现单 query token + KV cache 的 decode-step attention，体会 prefill 与 decode 的矩阵形状差异（`n×n` vs `1×n`）。

## 验收标准

- [x] 三种实现（标准/分块/online）对拍误差 fp32 下 max abs error < 1e-5
- [x] online softmax 支持任意块大小且结果与块大小无关
- [x] RoPE、RMSNorm 与 transformers 参考实现一致
- [x] decode-step attention 与 prefill 全量计算结果一致
- [x] 写一页笔记：online softmax 的 rescale 推导（这是主线1硬件设计的核心公式）

## 阅读材料

- FlashAttention (Dao et al., NeurIPS 2022) — 重点 Algorithm 1 与 IO 复杂度分析
- FlashAttention-2 (Dao, 2023) — 工作划分与 rescale 次数优化
- Online normalizer calculation for softmax (Milakov & Gimelshein, 2018) — online softmax 原始出处
- RoFormer (Su et al., 2021) — RoPE 原理
