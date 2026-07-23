# P2 阅读笔记

> 对应 [PLAN.md](PLAN.md) 阅读材料。
> 目标：建立 fake-quant / 旋转 / KV cache 实验前的概念框架；不必通读原文。

## 1. QuaRot（Ashkboos et al., 2024）

**回答什么问题**：权重量化（及部分激活）时，少数 **outlier 通道** 拉高动态范围，逼迫 INT4/INT8 网格浪费在极端值上——如何在 **几乎不改网络函数** 的前提下让激活更「量化友好」？

**关键构造**：在合适位置插入正交变换（典型为随机化 Hadamard），使

$$
x \;\mapsto\; Rx,\quad W \;\mapsto\; WR^\top
$$

（具体左右乘位置随层类型而定）。因 $R^\top R=I$，浮点理想算术下层输出不变；量化后误差往往下降，因为能量被摊到更多坐标，峰值 outlier 被抑制。

**直觉分层**：

| 层面 | 说法 |
|------|------|
| 数学 | 旋转（正交）≈ 恒等变换（可吸收进相邻线性层） |
| 数值 | 改变各通道动态范围 → 同一 bit 预算下量化 MSE/相对误差更小 |
| 实验 | 对比「直接 INT4」vs「Hadamard + INT4」的激活误差与下游指标 |

链接：[arXiv:2404.00456](https://arxiv.org/abs/2404.00456)

---

## 2. SAW-INT4 与 BDR（arXiv:2604.19157）

**地位**：面向 **decode / KV cache** 的 INT4，并用 **块对角旋转（BDR）** 抑制 outlier。

**BDR 直觉**：

$$
R = \mathrm{block\_diag}(H,H,\ldots,H)\,D
$$

- $H$：块内 Walsh–Hadamard（块大小如 128，须匹配 head/hidden 可分性）
- $D$：随机 $\pm 1$ 对角（打破符号对称、稳定随机化）
- 相对「整维大 Hadamard」：块对角更易硬件化（局部变换、少全局通信），仍能打散局部 outlier

**与 QuaRot 的关系**：同属「旋转 → 再量化」家族；SAW 更强调 **token-wise / KV 路径** 与 decode kernel 融合，而不仅是权重 PTQ。

链接：[arXiv:2604.19157](https://arxiv.org/abs/2604.19157)

---

## 3. BitDecoding（HPCA 2026）

**回答什么问题**：低比特 KV cache 不只是「存的时候 quant」——decode 要在 **带宽节省** 与 **反量化/计算开销** 之间做系统级权衡（GPU 上协同 CUDA core 与 Tensor Core；格式含 MXFP4 / NVFP4 等）。

**要点**：

1. **动机**：decode 扫 KV → memory-bound；降 KV 位宽直接砍 bytes/token。
2. **精度风险**：Q/K 低比特影响 score；V 低比特影响输出；softmax 累加宜保持较高精度（混合精度 datapath）。
3. **系统视角**：须算清 **反量化算力是否吃掉带宽收益**。

链接：检索 “BitDecoding HPCA 2026”（会议版 / 预印本以书目库为准）。

---

## 4. OCP Microscaling（MX）与 MXFP4

**格式要点**：

| 要素 | MXFP4 典型设定 |
|------|----------------|
| 元素 | 每值 4 bit（含符号的 E2M1 类微浮点） |
| 共享尺度 | 一个 **block**（常 32 元素）共用一个 scale（常为 8-bit 幂次尺度） |
| 与 INT4 per-group 对比 | 都是「块内共享动态范围」；MX 用浮点微格式 + 显式 scale 幂，INT4 用均匀整型网格 + scale/(zp) |

**Fake-quant 口径**：quantize → 落在可表示点 → dequantize 回 fp 计算图（训练/误差分析常用）；与硬件真实 bit-pack 存储可后续再对齐。

**FP8 对照**：E4M3（精度偏多）与 E5M2（动态范围偏大）。

规范入口：[OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

---

## 5. Fake-quant 粒度速查

| 粒度 | Scale 共享范围 | 典型用途 |
|------|----------------|----------|
| Per-tensor | 整个张量一个 scale | 粗基线，outlier 敏感 |
| Per-channel | 沿通道维 | 权重常见 |
| Per-group | 固定 group size（32/64/128） | 激活 / KV 折中 |
| MX block | block=32 + 共享指数 | MXFP4 |

对称量化：零点在 0；非对称：额外 zero-point 覆盖单侧偏移分布。

---

## 6. 两道自检题

### Q1：为什么「旋转在数学上是恒等、在数值上改变量化友好性」？

浮点且无量化时，$y = W(Rx)$ 与吸收旋转后的等价线性层相同。  
一旦对 $Rx$（或权重）做低比特网格投影，误差 $\|Q(Rx)-Rx\|$ 依赖于坐标轴对齐方式：outlier 被打散后，**同一套 scale 网格的裁剪/圆整误差**通常更小，于是端到端 PPL 更好。

### Q2：KV cache INT4 时，为何常保留 softmax 累加为 fp16/fp32？

Score 经 $\exp$ 与归一会放大相对误差；online softmax 的 $m,\ell,O$ 对误差敏感。  
存储/带宽瓶颈在 **KV 字节**，算力瓶颈不在那几次高精度累加——故混合精度常见形态是「Q/K/V 或 cache 低比特 + softmax/O 累加高比特」。
