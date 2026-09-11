# 面向长上下文解码的精度感知 KV Cache 流式 Attention 架构与映射：长线研究计划

> 本计划为**长线、由浅及深**的学术研究路线，不按日历年切分任务。范围限定为单芯片/单加速器内 Transformer **推理**，以 **decode** 的低比特 / 混合精度 KV Cache 流式 Attention 为核心；prefill 仅作兼容与对照。最终验证深度：**架构模拟 + 关键数据通路 RTL/PPA**（FPGA / 完整 ISA 编译器 / 流片为可选延伸，非主线必需）。

## 〇、修订说明

相对上一版“四年按年排程”计划，本版调整为：

- **取消按年任务表**，改为深度递进的研究阶段（R0 → R5），以验收门槛而非时间驱动；
- **显式对齐 2025–2026 前沿**，以公开方法为对照验证差异，不预先承诺超越已有研究；
- **保留纵向主线**，拒绝回到“FlashAttention 阵列 + 全非 GEMM + 完整编译器”的横向铺开。

R2 增补基础通路、探索实验与阶段末创新评估。三者分别记录：技术验收是否通过、约定实验是否完成、候选创新是否成立；完成探索不保证正向收益、创新成立或论文发表。阶段概述见本文件 R2 章节，实验细则见 [R2 详细计划](../research/r2_streaming_attention/PLAN.md)，执行状态只在[里程碑](progress/milestones.md)维护。

已有基础仍作为 R0：

| 资产 | 路径 | 作用 |
|------|------|------|
| 文献综述与缺口 | `survey/` | 论证碎片化与单芯片共设计空白 |
| 学习管线 P1–P5 | `learning/` | 数值、量化、瓶颈、玩具 RTL、tile 搜索技能与初步证据 |

---

## 一、前沿图景与研究定位（2024–2026）

### 1.1 算法与 GPU 系统侧：已到的前沿

近年工作把 KV Cache 压缩从“能否 INT4/2-bit”推进到**系统可部署**与**混合精度分配**：

| 方向 | 代表工作（非穷尽） | 前沿含义 |
|------|-------------------|----------|
| 非对称 / 低比特 KV | KIVI；BitDecoding（HPCA’26） | 真实 cache-path + fused dequant；布局必须匹配计算单元 |
| 服务约束下的轻量旋转 | SAW-INT4 | 提供 paged INT4、查询侧旋转修正及可选输出侧逆旋转的公开实现；R2 将其作为强基线参照 |
| 极限 2-bit + 自适应保留 | MiniKV；OScaR 等 | 量化与 eviction / 旋转联合，且需与 FlashAttention 类内核兼容 |
| 层/块混合精度 | KVTuner；PM-KVQ；KVmix | **精度本身成为配置变量**（离线搜索或渐进降比特） |
| 查询 / RoPE 感知分配 | MixKVQ；Block-GTQ 等 | 按 RoPE 块能量、query 相关性分配比特；**服务期不物化完整 FP16 KV** |
| Agent / 长 CoT 压力 | UltraQuant；PM-KVQ | 多轮与长推理链下误差累积成为一等公民问题 |

共识趋势：

1. **有效压缩是系统共设计问题**（布局、页表、融合内核、元数据），而非单纯 PTQ 表格；
2. **“不物化完整高精度 KV”** 成为高效实现的默认原则；
3. 固定均匀 INT4 已是强基线，前沿转向 **混合 / 渐进 / 结构感知的比特分配**。

### 1.2 专用硬件侧的比较参照

| 方向 | 代表工作 | 与本课题的关系 |
|------|----------|----------------|
| FlashAttention-native 阵列 | SystolicAttention / FSA、StreamAttention、COSA+、DESA | 多假定较高内部精度，**系统化低比特 KV 证据不足** |
| 全栈单芯片 | PLENA | 扁平阵列 + 非对称量化 + native FA + ISA/编译器；最接近端到端，但并非以 **decode 混合精度 KV 流式通路 + 精度一等公民映射** 为差异化中心 |
| FPGA / 边缘 | AccLLM（W2A8KV4）、VitaLLM 等 | 证明共设计有效，平台与问题设定不同 |
| 稀疏 / PIM / chiplet | Salca、Titanus、AMMA 等 | 相邻上界；**不纳入本课题主实现路径** |

### 1.3 本课题拟验证的研究问题

在算法已进入混合精度、GPU 已进入布局感知 fused dequant 的背景下，本课题围绕以下问题开展单芯片 ASIC 向研究；其与已有工作的差异须由对照和消融证据逐阶段确认：

> 如何在固定带宽 / SRAM / 算力下，使 **混合（乃至结构感知）比特 KV** 以 **规则、可流水** 的布局进入硬件，经 **流式解量化** 直接驱动 decode Attention，并由 **精度—布局—映射联合优化** 在精度约束下逼近流量与能效前沿——且全程 **不依赖完整 FP16 KV 展开**？

相对三类已有工作的比较重点：

- 相对 **BitDecoding / SAW-INT4**：核验具体尺度处理、供给与映射机制在 ASIC 资源约束下的增量收益；迁移到 ASIC、提供 RTL 或增加成本记账本身不证明创新；
- 相对 **KVTuner / MixKVQ / Block-GTQ**：把混合比特从算法配置推进到 **硬件一等公民**（元数据、不规则位宽的规则化编码、控制与 DMA 开销计入能量）；
- 相对 **PLENA**：不做“更大全栈复刻”，而把刀锋放在 **decode × 混合精度 KV 流式通路 × 精度感知映射**。

---

## 二、核心科学问题

给定片外带宽 $B_{\mathrm{mem}}$、片上容量 $S_{\mathrm{SRAM}}$ 与计算预算，在可行配置集 $\mathcal{C}$ 上求解

$$
\min_{c\in\mathcal{C}}\; E_{\mathrm{token}}(c)
\quad\text{s.t.}\quad
\Delta\mathrm{Accuracy}(c)\le\epsilon,\;
L_{\mathrm{token}}(c)\le L_{\max},\;
S_{\mathrm{buf}}(c)\le S_{\mathrm{SRAM}}.
$$

配置 $c$ 至少包含：

- KV 表示：位宽、K/V 非对称、group / microscale、旋转或 RoPE 块策略；
- 布局：packed / paged、比特分组的规则化容器、scale 与载荷共址；
- 数据通路：fused dequant、$QK^\top$、online softmax（含 partial $O$）、$PV$、KV-split / head / batch 并行；
- 映射：tile、缓冲划分、DMA 深度，以及 **混合精度调度本身**。

优先优化 **energy/token** 与 **HBM bytes/token**，并报告 latency/token；所有结论分层：算法精度 / 架构模拟 / RTL–PPA。

---

## 三、由浅及深的技术路线（无年份约束）

```mermaid
flowchart TB
  R0[R0 基础: Survey + Learning]
  R1[R1 真实KV基线与可部署静态压缩]
  R2[R2 流式ASIC通路: 无完整FP16展开]
  R3[R3 混合精度与结构感知比特分配]
  R4[R4 精度-布局-映射联合优化]
  R5[R5 前沿闭环: 校准PPA与长上下文压力测试]

  R0 --> R1 --> R2 --> R3 --> R4 --> R5
  R1 -.->|算法对齐前沿基线| FrontierAlgo[对齐 KIVI / SAW-INT4 / BitDecoding 口径]
  R3 -.->|对齐混合精度前沿| FrontierMix[对齐 KVTuner / PM-KVQ / MixKVQ / Block-GTQ 思想]
  R5 -.->|硬件侧前沿主张| FrontierHW[ASIC原生混合比特流式decode]
```

原则：**下一深度以前一深度的验收为前提**；允许在某一深度纵向挖深后再前进，但不允许跳过 R1–R2 直接做“动态混合精度硬件故事”。

### R0｜已完成：问题地图与技能基线

- Survey：四主题碎片化、decode 带宽墙、单芯片共设计稀缺。
- Learning 相对证据：decode AI $\approx 50.9$ ops/byte（memory-bound）；传统 systolic decode 利用率可降至 $\sim 1\%$；16 MiB 约仅容 $\sim 2\mathrm{K}$ INT8 token 的层内 $K{+}V$；INT4+BDR 在 **proxy** 上可恢复部分 PPL。
- **限制（必须在后续关闭）**：proxy 量化、无完整 $PV$ 的 softmax RTL、tile 模拟器未建模压缩 KV / 元数据 / dequant。

### R1｜真实 Cache-Path 基线：追平“可部署静态压缩”前沿

已完成的研究方法、结果与适用范围见 [R1 总报告](../research/r1_kv_baseline/REPORT.md)，计量约定见 [R1 协议](../research/r1_kv_baseline/protocols/metrics.md)。

**目标**：关闭 proxy，建立与当前算法/系统基线可对话的测量平台。

研究内容：

1. 真实 token-wise KV：quantize → pack → store → load → dequant → attention；
2. 对照谱：FP16；INT8；均匀 INT4；INT4+BDR（SAW-INT4 思想）；**必做** KIVI 风格非对称 2/4-bit（C4/C5）；
3. 布局：contiguous 与 **paged** 双报告；主声称不得只依赖理想连续地址；
4. 指标：任务精度 / PPL、HBM bytes/token、解量化与元数据开销分解；
5. 误差—通信模型：层 / 头 / token 位置敏感性；长上下文与多轮下的误差累积初探。

**对齐前沿的标准**：在约定模型与上下文上，静态 INT4(+BDR) 精度—流量 Pareto 达到可复现的 SOTA 邻域（以公开实现或论文表格为锚），并文档化与 proxy 的差距。

**进入 R2 的门槛**（完成依据见 [R1 总报告](../research/r1_kv_baseline/REPORT.md#6-结论与后续工作)）：

- 真实 cache-path 可复现；
- 至少一条长上下文设定下的 bytes/token–精度曲线（含 paged 列）；
- 专用 decode 模拟器与 Roofline / SCALE-Sim 在约定检查点趋势一致；
- 锁定默认模型列表、硬件包络假设与评测协议。

<a id="r2-overview"></a>

### R2｜流式专用数据通路：追平“无完整 FP16 展开”的系统原则，并落到 ASIC

实验设置、逐步审核与阶段末评估见 [R2 详细计划](../research/r2_streaming_attention/PLAN.md)，执行状态见[里程碑](progress/milestones.md#r2-review)。

**目标**：完成物理低比特缓存、流式 Attention、周期/能耗模型与关键 RTL 的基础闭环；默认路径不物化完整高精度 KV tile，格式规定的 FP16 残差窗单独计量。

研究内容：

1. 物理打包、追加、分页与残差路径，完整计入读写、刷窗及元数据访问；
2. 流式解码、QK、online softmax、PV 与输出归并，完善 C3 的 Q/O 端旋转及缓冲管理；
3. 建立优化 FP16、共享流式解码和 GQA/跨头/KV-split 对照，计入供给、互连、缓冲及归并成本；
4. 双模型、4K–32K、contiguous/paged 成对验证精度、任务与连续生成，64K/128K 仅作架构或轨迹压力测试；
5. 关键 RTL、完整小 Attention 通路与首版综合，按组件和形状校准周期、面积及能耗模型。

**尝试性创新**：

- 尺度处理与 GQA 复用协同，联合研究有界缓冲下的组供给调度，检验相对优化同格式基线的增量收益；
- 缓存页与输出归并范围解耦，检验成本预测选择能否改善最佳固定策略；
- 压缩收益边界模型，预测带宽、SRAM、解量化吞吐与共享比变化下的收益和失效条件。

**对齐前沿与相关工作**：以 BitDecoding、SAW-INT4、Flash-Decoding、QServe 的公开方法与实现建立强基线；对照 PLENA 的架构设计，以及 InnerQ、Multi-Scale Dequant 预印本中的尺度处理机制。具体来源和证据角色见[近年成果对比手册](recent_works_comparison.md)；迁移到 ASIC 或提供 RTL 本身不证明创新。

**进入 R3 的门槛**：

- 物理缓存语义正确，完整流式 Attention 满足预先约定的数值和任务质量要求；
- 在同等精度或明确退化点上，相对优化 FP16 KV 展示 bytes/token 与模拟 energy/token 改善，完整报告成本及不确定性；
- 完成关键模块与完整小通路验证、首版综合和相应校准，源码、配置、结果与证据角色可追溯。

基础验收、探索完成情况与创新成立分别评估；探索允许负结果，不保证收益或发表，也不豁免基础门槛。阶段末按证据将候选处置为论文候选、工程基线、有效否定或证据不足，详见[评估规则](../research/r2_streaming_attention/PLAN.md#r2-assessment)。

### R3｜混合精度与结构感知比特分配：进入当前算法前沿

**目标**：从均匀 INT4 推进到 **与 2025–2026 算法前沿同等级的表示能力**，并始终用硬件代价模型约束“算法好看但不可映射”的方案。

具体起步方向由 [R2 阶段末评估](../research/r2_streaming_attention/PLAN.md#r2-assessment)确定；R2 未成立的假设不自动作为 R3 的已验证前提，后续候选也不因列入计划就自动启动。

优先研究谱系（由易到难，可只深挖其中可规则化的子集）：

1. **层 / 块级混合精度**（KVTuner、PM-KVQ 类）：离线敏感性 → 静态或渐进比特表；
2. **时间维高精度窗口**：recent tokens / attention sink 保高精度，历史 token 降比特（KVmix 类思想）；
3. **结构感知分配**：RoPE 块能量、query 相关 key channel（Block-GTQ / MixKVQ 类）——**仅采纳可规则化、可打包的分配结果**；
4. 明确拒绝作为主线的：高度不规则稀疏索引、强依赖随机 eviction 且无法与规则 DMA 共存的策略（可作对照或 Future Work）。

硬件要点：

- **多速率比特的规则化容器**（如按组对齐到 nibble/byte 通道），避免 PE 侧任意位宽乱序；
- 元数据、重打包、控制流全部计入 traffic / energy；
- 与 R2 通路兼容：混合精度是通路上的模式，而非另起炉灶。

**对齐前沿的标准**：在相同平均比特预算下，精度不低于或接近公开混合精度方法的报告邻域；同时给出 **ASIC 代价下的 Pareto**（这是 GPU 论文通常缺失的一维）。

**进入 R4 的门槛**：

- 至少一种混合 / 结构感知策略在真实 cache-path 上稳定优于均匀 INT4（同预算或同精度下更省流量）；
- 硬件开销模型与算法增益同时报告；若增益被元数据吃掉，则收缩策略粒度，不硬宣称“动态一定更好”。

### R4｜精度—布局—映射联合优化：形成硬件侧独特贡献

**目标**：使精度配置成为映射器的一等决策变量，完成“算法前沿 × 架构约束”的闭环。

研究内容：

1. 联合搜索空间：`{bit plan, group/RoPE-block container, tile, head/batch/KV-split, SRAM 划分, DMA 深度}`；
2. 目标与约束同 §二；输出可解释策略或启发式（**非**通用深度学习编译器 / 完整自定义 ISA 栈）；
3. 服务态因素：上下文长度变化、paged 碎片、可选小 batch；
4. Prefill 仅附录对照；主叙事保持 decode。

**对齐前沿的标准**：相对手工静态 INT4 与相对“仅算法混合精度、忽略硬件开销”的虚高数字，均能展示可重复、可解释的收益。

**进入 R5 的门槛**：联合优化在约定负载上稳定可复现；开销入账；失败则收缩为 R2+R3 的有限层间表 + 手工映射并进入结题准备。

### R5｜前沿闭环：长压力测试、跨层校准与主张固化

**目标**：把主张推到可发表的硬件证据强度，并在最苛刻工作负载上考验。

研究内容：

1. 用 RTL 综合结果校准模拟器中 dequant / softmax / MAC / SRAM 代价；
2. 长上下文（32K–128K）、长 CoT / 多轮 agent 类压力（对齐 UltraQuant、PM-KVQ 问题设定的可复现子集）；
3. 与 GPU 系统基线（FlashDecoding、BitDecoding 类）做 **同模型同上下文的相对比较**（平台不同，报告口径透明）；
4. 与 PLENA 等 ASIC 叙事对齐比较维度：利用率、bytes/token、energy/token、精度，而非无条件倍数；
5. 根据实际证据确定可支持的贡献：**代价模型与测量、混合比特流式 decode 通路、精度感知联合映射**为候选成果方向，不要求三者必然全部成立；未关闭风险写入 Limitation。

**可选延伸（非主线）**：FPGA 原型；更大规模阵列；公开 PDK 之外的先进工艺库；极保守的 ISA 子集仅服务本通路配置下发。

---

## 四、评估矩阵

### 4.1 工作负载

- 模型：0.5B–8B 开源 decoder 为主；更大模型用离线 KV trace / 分层抽样；
- 上下文：≥4K / 32K，条件允许 128K；**decode 主结果，prefill 附录**；
- Batch：默认 1，可选小 batch；
- 布局：contiguous + paged；
- 数值对照：FP16；INT8；均匀 INT4±BDR；混合精度方案；可选 2-bit / MXFP4 作为扩展对照。

R2 的双模型、任务集合、数值阈值与测量要求统一见 [R2 实验范围与质量要求](../research/r2_streaming_attention/PLAN.md#r2-evaluation)。

### 4.2 指标分层

| 层 | 指标 |
|----|------|
| 算法 | 精度 / PPL / 长上下文或推理任务；与 FP 参考误差 |
| 架构 | bytes/token、latency/token、带宽与 PE 利用率、SRAM、energy/token（模型） |
| RTL | 功能匹配、频率、面积、功耗、关键通路延迟 |

禁止跨层偷换单位（如把分析模型 joule 写成硅片实测）。

### 4.3 工具

- 算法：PyTorch、Hugging Face、Triton / FlashAttention 参考、公开 KV 量化实现、lm-eval 子集；
- 架构：自研 decode simulator（主）、SCALE-Sim、Timeloop/Accelergy、Roofline、按需 DRAM 模型；
- RTL：SystemVerilog、Verilator、黄金模型；综合用实验室工具链或 Yosys/OpenROAD + 可用库。

---

## 五、设施（能力导向，非排期）

**最低**：≥128 GB 内存工作站、≥24 GB GPU、大容量 NVMe、Linux、可用综合路径。  
**理想**：更大显存 GPU、256 GB+ 内存、商业综合与功耗签核 + 正规工艺库。  
**非必需**：高端 HBM FPGA、完整 MLIR 编译器、集群、流片。

---

## 六、风险、收缩与停止规则

| 风险 | 收缩 |
|------|------|
| 低比特 / 混合精度在长 CoT 崩塌 | 提高敏感层与 recent token 精度；退回 INT8+布局优化 |
| 解量化与元数据吃掉收益 | 检验融合、容器及映射是否有独立增量；若仍无净收益，保留否定结果，不通过更换叙事认定创新 |
| 结构感知分配无法规则化 | 仅保留层间表 + 时间窗；复杂 RoPE/query 方案降为算法附录 |
| 与已有机制重合 | 按 R2 对照来源复核具体机制、尺度分组、供给/归并、写侧与元数据、资源预算和 RTL 证据；区分工程补齐与研究贡献，不仅按平台不同判断新颖性 |
| 模拟器与 RTL 趋势矛盾 | 冻结功能优先校准；未校准不宣称绝对 energy |
| 范围再次膨胀 | 稀疏主线、MoE、PIM、完整编译器一律 Future Work |
| 候选未出现正收益或资源不足 | 按预定对照完成分析，或记录经用户审核的停止理由；将有效否定与证据不足分开，不持续调参直到出现正结果 |
| 基础通路尚未达到验收门槛 | 单独记录未通过原因并审核收缩方案；探索允许失败不构成基础技术豁免 |

---

## 七、预期成果形态（论文链，不绑会议）

1. **测量与模型**：真实低比特 KV 的误差—元数据—traffic；decode 瓶颈再定位（关闭 proxy）。
2. **架构**：规则布局下的流式 fused-dequant decode Attention（无完整 FP16 展开）+ 关键 RTL。
3. **前沿系统**：混合 / 结构感知比特 × 精度感知映射，在 ASIC 代价模型下相对均匀 INT4 与 GPU 口径的 Pareto；学位论文整合。

上述为可能的成果形态，不预先承诺论文数量、创新成立或发表。研究目标是在 decode、压缩 KV 与硬件映射上形成可验证的差异；能否局部超越公开前沿，由强基线、完整成本、消融和独立配置证据决定。R2 允许形成工程改进、有效否定结果或证据不足的判断，具体处置遵循阶段末评估。

---

## 八、执行约定

1. 本文件维护长线方向和阶段门槛（R0–R5）；按年排程的旧表述作废。R1 的已完成研究见[总报告](../research/r1_kv_baseline/REPORT.md)，R2 实验细则见[详细计划](../research/r2_streaming_attention/PLAN.md)。
2. `survey/`、`learning/` 归档为 R0；后续实验按 R1–R5 深度组织目录，文件与证据管理遵循 [AGENTS.md](../AGENTS.md)。
3. 执行状态、技术验收与创新评估处置仅在[里程碑](progress/milestones.md)维护；正式分析写入所属实验唯一 REPORT.md，不复制结果表或结论。
4. 文档修订不代表实验开始，不提前创建实验报告或结果目录；已开展实验的入口与证据角色加入[实验索引](experiments.md)，维护历史在 [CHANGELOG](progress/CHANGELOG.md)简记。
