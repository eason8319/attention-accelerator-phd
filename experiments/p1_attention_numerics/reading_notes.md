# P1 阅读笔记

> 对应 [learning_plan.md](../../docs/learning_plan.md) P1 阅读材料。
> 目标：建立实现 golden model 前的概念框架；不必通读原文。
> rescale 公式的完整推导见同目录 [online_softmax_rescale_notes.md](online_softmax_rescale_notes.md)。

## 1. FlashAttention（Dao et al., NeurIPS 2022）

**回答什么问题**：在 SRAM 容量有限时，如何把 attention 做成 **IO-aware** 的——少写 HBM、不落完整 $S=QK^\top$ / $P=\mathrm{softmax}(S)$ 矩阵。

**标准 attention 的 IO 痛点**：

| 中间量 | 形状（单 head） | 问题 |
|--------|-----------------|------|
| $S = QK^\top/\sqrt{d}$ | $N\times N$ | 过大，必须写回 HBM |
| $P = \mathrm{softmax}(S)$ | $N\times N$ | 再读一遍做 $PV$ |
| 结果 | $N\times d$ | 两轮 $O(N^2)$ 片外流量 |

**核心手法**：把 $K,V$（以及 $Q$）沿序列维切成能放进 SRAM 的 **tile/block**，对每个 $Q$ 块与每个 $KV$ 块做局部 GEMM + softmax 统计，最终输出与朴素算法数学等价。

**Algorithm 1 要记住的状态（每行 $i$）**：

| 符号 | 含义 |
|------|------|
| $m_i$ | running row-max |
| $\ell_i$ | running sum of $\exp(S_{ij}-m_i)$ |
| $O_i$ | 未归一化的输出累加器 |

每处理一个新 KV 块：更新 $m$ → 用 $\alpha=\exp(m_{\mathrm{old}}-m_{\mathrm{new}})$ **rescale** 旧的 $\ell$ 与 $O$ → 累加本块贡献 → 全部块结束后 $O/\ell$。

**IO 复杂度直觉**（论文主结论，量级即可）：

- 朴素：片外读写主导项 $\Theta(N^2)$（写/读 $S,P$）。
- FlashAttention：在 SRAM 大小为 $M$ 时，HBM 访问可降到约 $\Theta(N^2 d^2 / M)$ 量级（$d$ 为 head dim）——**分块越大（在 SRAM 允许范围内），越少次扫 KV**。

链接：[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

---

## 2. FlashAttention-2（Dao, 2023）

**相对 FA1 改什么**：

1. **减少非 GEMM 开销**：更少的 rescale / 同步，把工作尽量压回 matmul 吞吐。
2. **并行划分**：在 sequence / head 维上重新切分，提高占用率（occupancy）；与硬件上的「行归约 vs 列并行」选型同源。
3. **前向仍是 online softmax 一族**：状态仍是 $m,\ell,O$；FA2 优化的是 **调度与 rescale 次数**，不是换一套数学。

读 FA2 时重点理解「块间 rescale 是串行依赖、块内可并行」。

链接：[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

---

## 3. Online normalizer（Milakov & Gimelshein, 2018）

**地位**：FlashAttention 所用 online softmax 的 **原始出处**（比 FA 更早、更窄：只谈稳定地 online 算 softmax 归一化因子）。

**一句话算法**：流式看到标量 $x_t$ 时维护

$$
m_t = \max(m_{t-1}, x_t),\quad
d_t = d_{t-1}\,e^{m_{t-1}-m_t} + e^{x_t-m_t}
$$

则 $\mathrm{softmax}$ 分母可用最终的 $d$（再配合分子侧同类 rescale）。FA 把它推广到 **整块向量**，并对输出累加器 $O$ 做同样的 $\alpha$ 缩放。

**数值动机**：始终在当前 max 下算 $\exp(x-m)$，避免 $\exp(x)$ 溢出；换 max 时用乘法修正旧统计量，而不是重扫历史。

链接：[arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

---

## 4. RoFormer / RoPE（Su et al., 2021）与 RMSNorm

**RoPE 在 datapath 中的位置**：在 $Q,K$ 投影之后、$QK^\top$ 之前，对 head 内成对维度做位置相关旋转；$V$ 通常不旋转。

二维子空间直觉（角度 $\theta_m = m\cdot\theta$）：

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix}
=
\begin{pmatrix} \cos\theta_m & -\sin\theta_m \\ \sin\theta_m & \cos\theta_m \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

相对位置体现在 $q_m^\top k_n \propto$ 依赖 $m-n$ 的旋转差。

**RMSNorm**：decoder 层里常见于 attention / FFN 前（Pre-Norm）：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}}\odot\gamma
$$

无 mean-centering（相对 LayerNorm 更省）。属非 GEMM 算子。

链接：[RoFormer arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---

## 5. Prefill vs Decode

| 模式 | Query 形状 | $QK^\top$ | KV 访问 |
|------|------------|-------------|---------|
| Prefill | $N\times d$ | 近似方阵 GEMM | 同批 K/V 被多 query 复用 |
| Decode | $1\times d$（+ cache） | $1\times N$ 瘦 GEMM | 每步仍扫整段 cache |

形状差异本身不改变 softmax 数学，但决定了 **算术强度与 PE 利用率**。

---

## 6. 两道自检题

### Q1：为什么两遍分块「需要两遍」，online 却能一遍？

Softmax 的分母是 **整行** $\sum_j\exp(S_{ij})$。分块后若每块各自归一化再拼，权重错误。

- **两遍**：第一遍扫全部块只收集全局 $m,\ell$（或等价统计）；第二遍用全局量算 $P$ 与 $PV$。
- **Online**：用 running $m,\ell,O$ + rescale，在单遍扫描中保持「与最终全局 softmax 等价」的充分统计量。

### Q2：online softmax 的结果为何与块大小无关？

块大小只改变 **何时** 做 rescale，不改变最终在同一全局 $m$ 基准下累加的 $\sum\exp(S-m)$ 与 $\sum\exp(S-m)\,V$。  
任意能正确处理尾块的 `block_size` 应对拍到同一输出。
