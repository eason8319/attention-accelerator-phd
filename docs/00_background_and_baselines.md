# 背景、对标基线与代表性工作（R0）

> 本文件对齐现行研究计划 [`research_plan.md`](research_plan.md)（R0–R5）。  
> **旧版「四主线全栈」表述已废止**；差异化定位以 decode × 低比特/混合精度 KV 流式通路为准。  
> 文献条目与核实状态见 [`recent_works_comparison.md`](recent_works_comparison.md) 与 [`lit_watch/ledger.yaml`](lit_watch/ledger.yaml)。

## 1. 问题陈述

大语言模型（LLM）推理已从「计算受限」转向「访存受限」。对本课题最关键的矛盾在 **decode 阶段的 KV Cache**：

- **Decode memory-bound**：每步仅一行 $Q$，却需反复读取随 $N$ 增长的 KV；skinny GEMM 使传统阵列利用率崩溃。
- **压缩与展开冲突**：INT4/FP4/MXFP4 等可降流量，若片上先物化完整高精度 KV tile，带宽收益易被抵消。
- **布局与服务约束**：paged KV、规则访存、融合 dequant 决定算法能否落地（见 SAW-INT4、BitDecoding）。
- **精度成为配置变量**：层/时间窗/结构感知混合精度已是 2025–2026 算法前沿，ASIC 代价模型仍滞后。
- **Online softmax 仍必要**：流式通路需维持 $m/\ell$ 与 partial $O$；RMSNorm/RoPE 仅作接口，不作主贡献。

学习管线（`learning/`）的相对证据：decode AI 低于 Roofline 拐点、systolic decode 利用率可降至约 $1\%$、16 MiB 仅容纳短 INT8 KV、INT4+BDR 在 **proxy** 上可恢复部分精度——正式研究必须关闭 proxy。

## 2. 代表性工作（对标基线）

| 工作 | 出处（核实） | 关键贡献 | 与本课题关系 |
|------|--------------|----------|--------------|
| FlashAttention / FlashDecoding | NeurIPS 等 | IO-aware 分块 + online softmax | 数值与流量参考 |
| KIVI | ICML 2024；PMLR 235:32332–32344 | K/V 非对称约 2-bit | 算法精度锚（R1） |
| BitDecoding | HPCA 2026 | TC 友好布局 + fused dequant | GPU 系统锚（R2 原则） |
| SAW-INT4 | arXiv:2604.19157（预印本） | token-wise INT4 + BDR；paged 可部署 | 静态压缩默认可部署点 |
| KVTuner | ICML 2025；PMLR 267:36451–36485 | 层间混合精度离线表 | R3 混合精度入口 |
| Block-GTQ | arXiv:2606.24033（预印本） | RoPE 块感知比特；不物化完整 FP16 KV | R3–R5 精度前沿对标 |
| SystolicAttention (FSA) | arXiv:2507.11331（预印本） | 单阵列内 FA / online softmax | softmax/$PV$ 电路参考，非完整贡献 |
| PLENA | arXiv:2509.09505（预印本） | 扁平阵列 + 非对称量化 + FA ISA | **全栈对照**；本课题不做更大复刻 |
| AccLLM / FlightLLM | 预印本 / FPGA’24 | FPGA 上 W2A8KV4 等 | 可选 FPGA 延伸对照 |
| Salca | arXiv:2604.24820（预印本） | 稀疏长上下文 decode ASIC | 相邻上界，非主路径 |

## 3. 本课题的差异化定位

现有工作在 GPU 上已推进布局感知 fused dequant 与混合精度算法；专用硬件或偏 FA-native（较高内部精度），或偏全栈（如 PLENA），鲜少把下列三者钉在同一可验证刀锋上：

1. **真实 cache-path** 的低比特 / 混合精度 KV；  
2. **不物化完整 FP16 KV** 的流式 ASIC 数据通路；  
3. 将精度配置作为 **映射一等公民**，并报告元数据与控制开销。

最终验证深度：架构模拟 + 关键通路 RTL/PPA（见研究计划）。

## 4. 范围界定

| 纳入主线 | 降级为接口/对照 | 明确排除 |
|----------|-----------------|----------|
| 真实 KV 量化与布局、fused dequant、decode Attention | Prefill 兼容；online softmax（必要） | 分布式训练、chiplet、PIM/CIM 主线 |
| INT4 主格式 + 有限对照；可规则化混合精度 | RMSNorm / RoPE 接口 | 完整编译器 / 自定义 ISA 栈 |
| 专用 decode simulator + 关键 RTL/PPA | Timeloop/SCALE-Sim 对照 | MoE、稀疏 Attention 主实现、流片硬性要求 |

## 5. 核心科学问题

在给定片外带宽、片上 SRAM 与计算预算下，联合设计 KV 的低比特/混合精度表示、存储布局、流式解量化与 decode 映射，在

$$
\Delta\mathrm{Accuracy}\le\epsilon
$$

约束下最小化每 token 代价（优先 energy/token 与 HBM bytes/token，并报告 latency/token）。形式化目标见 [`research_plan.md`](research_plan.md) §二。

## 6. 评估指标入口

统一指标与分层报告约定见 [`metrics.md`](metrics.md)；对照实验矩阵见 [`recent_works_comparison.md`](recent_works_comparison.md) §5。
