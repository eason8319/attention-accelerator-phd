# 阶段0：背景、对标基线与代表性工作

本文件对应研究计划阶段0，目的是确立课题的问题边界、对标基线与评估指标体系。

## 1. 问题陈述

大语言模型（LLM）推理已从"计算受限"转向"访存受限"。瓶颈集中在 attention 子系统：

- **长上下文复杂度**：标准 attention 计算量为 \(O(n^2 d)\)，KV cache 容量随生成 token 线性增长。
- **decode 强 memory-bound**：自回归 decode 阶段 batch 小、矩阵"瘦"，受 HBM 带宽限制。
- **FlashAttention 映射难**：分块 + online softmax 数据流难以原生映射到传统 systolic array。
- **非 GEMM 算子打断流水**：softmax / RMSNorm / RoPE 造成 PE 利用率下降。
- **低精度数值风险**：INT4/FP8/MXFP4 降带宽降功耗，但 attention score 与 softmax 累加存在稳定性风险。

## 2. 代表性工作（对标基线）

| 工作 | 出处 | 关键贡献 | 与本课题关系 |
|------|------|----------|--------------|
| FlashAttention / FlashDecoding | 2022-2023 | IO-aware 分块 attention + online softmax | 数据流基线、精度参考 |
| FSA / SystolicAttention | arXiv 2507.11331, 2025 | 在单个 systolic array 上原生跑完整 FlashAttention，row-max/row-sum 就地归约 | 架构基线（主线1） |
| PLENA | arXiv 2509.09505, 2025 | flattened systolic array + 非对称量化 + 原生 FlashAttention + 自定义 ISA/编译器 | 全栈对标（主线1/2/4） |
| BitDecoding | HPCA 2026 | 低比特 KV cache 协同 CUDA/Tensor Core，MXFP4/NVFP4 原生 | 混合精度 KV cache 基线（主线2） |
| SAW-INT4 (BDR) | arXiv 2604.19157, 2026 | 块对角 Hadamard 旋转 + token-wise INT4 KV，融合进 decode kernel | 旋转量化方法（主线2） |

## 3. 本课题的差异化定位

现有工作大多聚焦其中一两点（GPU kernel、单阵列 FlashAttention、KV 量化）。本课题强调
**单加速器内的全栈协同**：FlashAttention-native 阵列 + 混合精度 datapath + 非 GEMM 专用单元 +
编译映射，在统一框架内做 PPA 与精度联合评估。

## 4. 范围界定

- **纳入**：单加速器 attention 推理、prefill/decode 双模式、INT4/FP8/MXFP4、softmax/RMSNorm/RoPE、编译映射。
- **仅作展望**：MoE 稀疏调度、模拟 CIM、多 chiplet、训练加速、完整 serving runtime。

## 5. 核心科学问题

> 在给定片上 SRAM 与 PE 阵列资源约束下，如何设计一种原生支持 FlashAttention 数据流、
> 采用混合精度计算、并融合非 GEMM 算子的 attention 加速架构，并通过编译映射方法，
> 最小化长上下文推理的片外访存、提升能效（tokens/s/W）且保持近无损精度？
