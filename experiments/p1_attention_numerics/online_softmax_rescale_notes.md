# Online Softmax Rescale 推导笔记

> P1 验收产出。对应 FlashAttention Algorithm 1 中 running max / running sum / output accumulator 的 rescale 逻辑——这是主线 1 硬件设计的核心公式。

## 问题设定

对 attention 的某一行 \(i\)，softmax 权重为

\[
P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}, \quad O_i = \sum_j P_{ij} V_j
\]

其中 \(S_{ij} = (Q_i \cdot K_j) / \sqrt{d}\)。FlashAttention 将 \(K,V\) 沿序列维分块，不能等所有块到齐再算 softmax，因此需要 **online** 更新 row-max 与归一化分母。

## 符号

- 已处理块后的 running max：\(m_i\)
- running sum（未归一化的 exp 和）：\(\ell_i = \sum_{j \in \text{seen}} \exp(S_{ij} - m_i)\)
- running output accumulator：\(O_i\)（当前块贡献的加权和，尚未除以 \(\ell_i\)）

新块分数矩阵为 \(S^{(b)}\)（行 \(i\) 记为 \(S^{(b)}_{i,*}\)）。

## 第一步：更新 row-max

\[
m_i^{\text{new}} = \max(m_i,\; \max_j S^{(b)}_{ij})
\]

## 第二步：rescale 旧状态

旧块在 \(m_i\) 下累积的 exp 和，换到 \(m_i^{\text{new}}\) 时需乘 rescale 因子：

\[
\alpha_i = \exp(m_i - m_i^{\text{new}})
\]

因为 \(\exp(S_{ij} - m_i^{\text{new}}) = \exp(S_{ij} - m_i) \cdot \alpha_i\)。

更新 running sum：

\[
\ell_i^{\text{new}} = \alpha_i \cdot \ell_i + \sum_j \exp(S^{(b)}_{ij} - m_i^{\text{new}})
\]

## 第三步：rescale 输出累加器

旧 accumulator 中的每一项都隐式使用了旧的 \(m_i\)。换基后：

\[
O_i^{\text{new}} = \alpha_i \cdot O_i + P^{(b)} V^{(b)}, \quad P^{(b)}_{ij} = \exp(S^{(b)}_{ij} - m_i^{\text{new}})
\]

## 最终输出

所有块处理完毕后：

\[
\text{Output}_i = O_i / \ell_i
\]

## 正确性直觉

把已见块与新块合并，等价于在更大的 index 集合上做一次 softmax。Rescale 因子 \(\alpha_i\) 保证：**换 max 后，旧块的 exp 权重与新块的 exp 权重在同一基准 \(m_i^{\text{new}}\) 下可直接相加**。

## 与两遍分块法的对比

| 方法 | 扫描次数 | 中间存储 | 硬件含义 |
|------|----------|----------|----------|
| 两遍分块 | 2 遍 KV | 需存全局 row-max / row-sum | 简单但多一次访存 |
| Online（FlashAttention） | 1 遍 KV | 仅 \(m, \ell, O\) 三个 running 状态 | 适合流式 PE 阵列，是 FSA/PLENA 类架构的数据流基础 |

## 实现对应（`attention_online.py`）

```python
m_new = torch.maximum(m, m_blk)
alpha = torch.exp(m - m_new)
p_blk = torch.exp(scores - m_new.unsqueeze(-1))
l = alpha * l + p_blk.sum(dim=-1)
o = alpha.unsqueeze(-1) * o + torch.matmul(p_blk, v_blk)
m = m_new
# 最后: return o / l.unsqueeze(-1)
```

## 硬件设计要点（面向 P4）

1. **每来一个 KV 块**：比较器更新 row-max → 查表/exp 单元算 \(\exp(S - m_{\text{new}})\) → 乘法器应用 \(\alpha\) rescale → MAC 阵列累加 \(PV\)。
2. **Rescale 是块间唯一的数据依赖**：块内可并行，块间需 broadcast \(\alpha\) 到所有 head_dim 通道。
3. **数值稳定性**：始终减去当前 running max，避免 \(\exp(S)\) 溢出；与 fp32 softmax 累加、低精度 \(QK\) 的组合是主线 2/3 的交界点。
