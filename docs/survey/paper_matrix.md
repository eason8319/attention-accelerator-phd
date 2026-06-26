# 文献对标矩阵（Paper Matrix）

> 阶段 0 文献收集产出。每条文献登记：出处、核心贡献、评估指标、开源情况、主线映射。指标口径与 [metrics.md](../metrics.md) 对齐。

## 登记字段说明

- **出处**：会议/期刊/arXiv 编号与年份
- **问题切入点**：该工作主要缓解哪类瓶颈（计算 / 访存 / 精度 / 利用率）
- **数据流/精度**：FlashAttention 变体、online softmax、精度组合
- **硬件结构**：阵列类型、非 GEMM 单元、SRAM 层次
- **评估层次**：算法 / 架构仿真 / RTL 综合
- **报告指标**：latency、throughput、PE utilization、HBM traffic、energy/token、精度
- **开源**：代码/RTL/仿真器是否公开
- **主线映射**：L1–L4 或对照 A/B

---

## 主线1：FlashAttention 原生数据流与阵列架构

### FlashAttention / FlashDecoding（dao2022flashattention, dao2023flashattention2, shah2024flashattention3, dao2023flashdecoding）
- **出处**：NeurIPS 2022 / arXiv 2023–2024
- **问题切入点**：IO-aware 分块 attention，消除 \(N \times N\) score 片外物化
- **数据流/精度**：online softmax（row-max/row-sum）；v3 支持 FP8；FlashDecoding 面向 decode 长 KV
- **硬件结构**：GPU kernel（Tensor Core + shared memory tiling）
- **评估层次**：算法 + GPU 实测
- **报告指标**：HBM 读写量、端到端 latency、与标准 attention 数值等价
- **开源**：是（Dao-AILab/flash-attention）
- **主线映射**：L1

### FSA / SystolicAttention（lin2025systolicattention）
- **出处**：arXiv 2507.11331, 2025
- **问题切入点**：传统 systolic array 需外部 vector unit 做 softmax，利用率低
- **数据流/精度**：完整 FlashAttention 在单阵列内执行；FP16 激活 + FP32 累加；exp 线性插值复用 MAC
- **硬件结构**：128×128 systolic + 向上数据路径 + comparator 阵列做 row-max/row-sum；无需 vector unit
- **评估层次**：RTL 综合（16nm, 1.5GHz）+ 架构对比
- **报告指标**：attention FLOPs/s 利用率（相对 TPUv5e 4.83×、Neuron-v2 1.77×）；面积开销 ~12%
- **开源**：否（截至检索时）
- **主线映射**：L1, L3（exp 近似）

### PLENA（plena2025）
- **出处**：arXiv 2509.09505, 2025
- **问题切入点**：长上下文 agentic 推理 memory wall；prefill/decode 利用率失衡
- **数据流/精度**：原生 FlashAttention + 非对称量化（Pathway 2/3）；flattened systolic array
- **硬件结构**：flattened array + head-level 分解 + 混合精度 MAC；完整 ISA/编译器/仿真器/RTL
- **评估层次**：架构仿真 + RTL + 与 A100/TPU v6e 对比
- **报告指标**：throughput（2.23× A100, 4.70× TPU v6e）；energy efficiency 4.04× A100
- **开源**：计划开源（论文声明）
- **主线映射**：L1, L2, L4

### FlatAttention（flatattention2026）
- **出处**：arXiv 2604.02110, 2026
- **问题切入点**：tile-based / wafer-scale 加速器上 attention HBM 流量过高
- **数据流/精度**：fabric collective 优化数据流；支持多种 attention 变体；FP8 decode
- **硬件结构**：32×32 tile + on-chip network fabric collectives
- **评估层次**：架构仿真（与 GH200 对比）
- **报告指标**：92.3% 利用率；4.1× vs FA-3；HBM traffic 16× 降低
- **开源**：否
- **主线映射**：L1, L4

### StreamAttention（streamattention2025）
- **出处**：OpenReview, 2025
- **问题切入点**：单阵列 FlashAttention 利用率仅 ~40%（对比 FSA 的改进方向）
- **数据流/精度**：四阶段 systolic 流式数据流；PWL exp；无专用 softmax 单元
- **硬件结构**：单 systolic array 连续流式 operand
- **评估层次**：架构仿真
- **报告指标**：高利用率（相对 SystolicAttention ~40% 的改进）；能效
- **开源**：否
- **主线映射**：L1

### COSA（wang2023cosa）
- **出处**：MICRO 2023
- **问题切入点**：MHA 层内 QK^T、softmax、AV 需多阵列 + 专用 softmax 单元
- **数据流/精度**：双 systolic array 协同 + 中间 softmax 单元
- **硬件结构**：两个 cooperating systolic array + dedicated softmax
- **评估层次**：架构仿真
- **报告指标**：~95% MHA 层利用率
- **开源**：否
- **主线映射**：L1, L3

### DESA（desa2025）
- **出处**：IEEE TC, 2025
- **问题切入点**：Transformer 中非 GEMM 算子导致 systolic array 低利用率
- **数据流/精度**：score-stationary 变体 + 即时处理单元
- **硬件结构**：systolic array + 可重构 vector unit + IPU
- **评估层次**：架构仿真
- **报告指标**：相对 GPU baseline 加速；PE 利用率提升
- **开源**：否
- **主线映射**：L1, L3

### SpAtten（wang2021spatten）
- **出处**：HPCA 2021
- **问题切入点**：attention 冗余 token/head
- **数据流/精度**：cascade pruning + 稀疏 attention
- **硬件结构**：专用稀疏 attention 架构
- **评估层次**：架构仿真
- **报告指标**：加速比、能耗
- **开源**：否
- **主线映射**：L1, 对照A

---

## 主线2：混合精度 attention datapath 与 KV cache 量化

### SAW-INT4 / BDR（sawint42026）
- **出处**：arXiv 2604.19157, 2026
- **问题切入点**：INT4 KV cache 在真实 serving 约束下精度崩溃
- **数据流/精度**：token-wise INT4 + 块对角 Hadamard 旋转（BDR）；prefill FA3 + decode triton
- **硬件结构**：GPU fused kernel（非 ASIC）；paged KV layout 兼容
- **评估层次**：算法 + 端到端 serving benchmark
- **报告指标**：task accuracy（GSM8K/GPQA/MATH500）；throughput 与 plain INT4 持平
- **开源**：是（togethercomputer/saw-int4）
- **主线映射**：L2

### BitDecoding（bitdecoding2026）
- **出处**：HPCA 2026
- **问题切入点**：decode 阶段 KV cache 带宽瓶颈
- **数据流/精度**：低比特 KV（MXFP4/NVFP4）+ CUDA/Tensor Core 协同
- **硬件结构**：GPU kernel co-design
- **评估层次**：GPU 实测
- **报告指标**：decode throughput、带宽节省、精度
- **开源**：待确认
- **主线映射**：L2

### BATQuant（batquant2026）
- **出处**：arXiv 2603.16590, 2026
- **问题切入点**：全局旋转（QuaRot/SpinQuant）在 MXFP4 上失效
- **数据流/精度**：块仿射变换（BAT），对齐 MXFP 量化粒度（如 32 元素块）
- **硬件结构**：算法层（待硬件映射）
- **评估层次**：算法 + lm-eval
- **报告指标**：W4A4/W4A8KV8 精度；相对 RTN/QuaRot 提升
- **开源**：待确认
- **主线映射**：L2

### KIVI（liu2024kivi）
- **出处**：ICML 2024
- **问题切入点**：KV cache 内存占用
- **数据流/精度**：非对称 2-bit KV；K 按 channel、V 按 token 量化
- **硬件结构**：算法 + GPU kernel
- **评估层次**：算法
- **报告指标**：内存压缩比、下游 task accuracy
- **开源**：是
- **主线映射**：L2

### QuaRot（ashkboos2024quarot）
- **出处**：arXiv 2024
- **问题切入点**：INT4 量化 outlier
- **数据流/精度**：全局 Hadamard 旋转 + INT4 权重/激活/KV
- **硬件结构**：算法（融合旋转进 GEMM）
- **评估层次**：算法
- **报告指标**：WikiText PPL、zero-shot accuracy
- **开源**：是
- **主线映射**：L2

### SpinQuant（liu2024spinquant）
- **出处**：arXiv 2024
- **问题切入点**：旋转矩阵学习
- **数据流/精度**：可学习旋转 + INT4/INT8
- **硬件结构**：算法
- **评估层次**：算法
- **报告指标**：量化精度
- **开源**：是
- **主线映射**：L2

### UltraQuant（ultraquant2026）
- **出处**：arXiv 2606.20474, 2026
- **问题切入点**：agent 多轮对话 KV cache 压力
- **数据流/精度**：FP4 KV + FP8 query + UE8M0 group scale；TurboQuant 风格旋转
- **硬件结构**：AMD CDNA4 MFMA kernel
- **评估层次**：GPU serving 实测
- **报告指标**：TTFT 3.47×（late rounds）；throughput 1.63×
- **开源**：待确认
- **主线映射**：L2

### SmoothQuant（xiao2023smoothquant）
- **出处**：ICML 2023
- **问题切入点**：激活 outlier 导致 INT8 量化困难
- **数据流/精度**：W8A8 per-channel 平滑
- **硬件结构**：算法
- **评估层次**：算法
- **报告指标**：INT8 精度接近 FP16
- **开源**：是
- **主线映射**：L2

---

## 主线3：非 GEMM 专用硬件

### MIVE（mive2026）
- **出处**：arXiv 2606.17781, 2026
- **问题切入点**：Softmax/LayerNorm/RMSNorm 各自独立硬件导致面积重复
- **数据流/精度**：统一整数 datapath；exp/reciprocal 用 ROM LUT
- **硬件结构**：Minimalist Integer Vector Engine；可编程统一 datapath
- **评估层次**：ASIC 综合
- **报告指标**：面积效率 vs 独立加速器；latency (cycles)
- **开源**：否
- **主线映射**：L3

### SOLE（sole2025）
- **出处**：arXiv 2510.17189, 2025
- **问题切入点**：Softmax/LayerNorm 中间结果 memory-bound
- **数据流/精度**：E2Softmax（log2 量化 exp 输出 4-bit）+ AILayerNorm（低精度统计）
- **硬件结构**：Log2Exp Unit（仅 shifter+adder）+ 近似 log 除法
- **评估层次**：FPGA/ASIC + 与 GPU 对比
- **报告指标**：3.04×/3.86× energy-efficiency；2.82×/3.32× area-efficiency
- **开源**：否
- **主线映射**：L3

### FSA exp 近似（lin2025systolicattention, koca2023exp）
- **出处**：arXiv 2507.11331 / 2023
- **问题切入点**：FlashAttention 中 exp 需外部单元或高成本浮点
- **数据流/精度**：输入 ≤0 时 split 整数/小数部分 + 线性插值；复用 systolic MAC
- **硬件结构**：阵列内 split unit
- **评估层次**：RTL
- **报告指标**：数值误差；面积开销
- **开源**：否
- **主线映射**：L1, L3

### Hardware-Oriented Softmax/RMSNorm（softmaxrmsnorm2026）
- **出处**：Electronics 2026
- **问题切入点**：非 GEMM 算子 FPGA 实现资源消耗大
- **数据流/精度**：SafeSoftmax +  bipartite LUT exp；RMSNorm LOD-LUT-MUL rsqrt
- **硬件结构**：FPGA pipelined accelerator
- **评估层次**：FPGA 实测
- **报告指标**：latency、DSP/LUT 用量、精度
- **开源**：否
- **主线映射**：L3

### Guaranteed Normalization（guaranteednorm2026）
- **出处**：arXiv 2604.23647, 2026
- **问题切入点**：近似 softmax 仅保序不保绝对值，score-oriented 任务精度差
- **数据流/精度**：radix-2 exp LUT + 定点除法；LayerNorm Newton rsqrt
- **硬件结构**：Softmax 942 μm² / LayerNorm 1199 μm²（ASIC）
- **评估层次**：ASIC 综合
- **报告指标**：面积 11×/14× 小于参考设计；N+1 cycle latency
- **开源**：否
- **主线映射**：L3

### Softermax（sun2022softmax）
- **出处**：HPCA 2022
- **问题切入点**：softmax exp 硬件代价高
- **数据流/精度**：base-2 softmax 替代 e^x
- **硬件结构**：软硬件协同近似
- **评估层次**：FPGA
- **报告指标**：面积 0.25×；精度
- **开源**：否
- **主线映射**：L3

### RoPE 实现（通用实践）
- **出处**：各加速器论文中的实现描述
- **问题切入点**：位置编码需 cos/sin 查表 + 向量旋转
- **数据流/精度**：FP16/FP32 查表；与 Q/K 投影融合
- **硬件结构**：LUT + 复数旋转（多数工作未独立优化）
- **评估层次**：—
- **报告指标**：—
- **开源**：—
- **主线映射**：L3

---

## 主线4：编译/映射与软硬件协同

### PLENA 编译栈（plena2025）
- **出处**：arXiv 2509.09505, 2025
- **问题切入点**：FlashAttention + 混合精度 + 长上下文需自动映射
- **数据流/精度**：自定义 ISA + 编译器 + DSE flow
- **硬件结构**：与 PLENA 硬件绑定
- **评估层次**：transaction-level 仿真
- **报告指标**：端到端 throughput/energy
- **开源**：计划开源
- **主线映射**：L4

### Timeloop（parashar2019timeloop）
- **出处**：ISCA 2019
- **问题切入点**：DNN 加速器 mapping 搜索
- **数据流/精度**：解析性能模型
- **硬件结构**：通用 accelerator 抽象
- **评估层次**：架构级解析
- **报告指标**：latency、energy（via Accelergy）
- **开源**：是
- **主线映射**：L4

### Accelergy（wu2019accelergy）
- **出处**：ICCAD 2019
- **问题切入点**：架构级能量估计
- **数据流/精度**：energy-per-action 表
- **硬件结构**：组件级能量模型
- **评估层次**：架构级
- **报告指标**：energy、area
- **开源**：是
- **主线映射**：L4

### SCALE-Sim v3（scalesimv3_2025）
- **出处**：arXiv 2504.15377, 2025
- **问题切入点**：systolic array cycle-accurate 仿真不足
- **数据流/精度**：多核 + 稀疏 + Ramulator DRAM + Accelergy
- **硬件结构**：systolic array 仿真器
- **评估层次**：cycle-accurate
- **报告指标**：latency cycles、bandwidth、energy
- **开源**：是
- **主线映射**：L4

### TransInferSim（klhufek2025transinfersim）
- **出处**：IEEE Access 2025
- **问题切入点**：Transformer 推理专用仿真缺失
- **数据流/精度**：Transformer layer 级 cycle-accurate
- **硬件结构**：systolic array + cache hierarchy
- **评估层次**：cycle-accurate + Accelergy
- **报告指标**：latency、energy、area；可导出 execution plan
- **开源**：是（ehw-fit/TransInferSim）
- **主线映射**：L4

### TileLang（tilelang2025）
- **出处**：arXiv 2504.17577, 2025
- **问题切入点**：AI kernel 开发需显式 tile 级控制
- **数据流/精度**：Python DSL；显式 memory hierarchy buffer
- **硬件结构**：通用 GPU/NPU
- **评估层次**：kernel 级 benchmark
- **报告指标**：kernel throughput
- **开源**：待确认
- **主线映射**：L4

### Eyeriss（chen2016eyeriss）
- **出处**：ISSCC 2016
- **问题切入点**：DNN dataflow mapping
- **数据流/精度**：row-stationary
- **硬件结构**：systolic array + NoC
- **评估层次**：芯片实测
- **报告指标**：energy efficiency
- **开源**：否
- **主线映射**：L4

---

## 对照A：稀疏/动态 attention

### Salca（salca2026）
- **出处**：arXiv 2604.24820, 2026
- **问题切入点**：长上下文 decode 带宽 + 计算压力
- **数据流/精度**：dual-compression dynamic sparse attention；ultra-low-precision + feature sparsity
- **硬件结构**：fully pipelined parallel ASIC
- **评估层次**：架构 + 性能模型
- **报告指标**：3.82× speedup、74.19× energy vs A100
- **开源**：否
- **主线映射**：对照A

### Sanger（liu2021sanger）
- **出处**：MICRO 2021
- **问题切入点**：稀疏 attention 负载均衡
- **数据流/精度**：score-stationary + 量化预测
- **硬件结构**：可重构 systolic array
- **评估层次**：架构仿真
- **报告指标**：2.39× vs A3
- **开源**：否
- **主线映射**：对照A

---

## 对照B：访存中心架构（展望）

### AMMA（amma2026）
- **出处**：arXiv 2604.26103, 2026
- **问题切入点**：1M context attention serving；GPU memory bandwidth 不足
- **数据流/精度**：HBM-PNM cubes；two-level hybrid parallelism
- **硬件结构**：multi-chiplet memory-centric
- **评估层次**：架构仿真
- **报告指标**：15.5× latency、6.9× energy vs H100
- **开源**：否
- **主线映射**：对照B

### LoL-PIM（lolpim2024）
- **出处**：arXiv 2412.20166, 2024
- **问题切入点**：长上下文 KV cache 超出单 PIM 容量
- **数据流/精度**：multi-node PIM + pipeline parallelism
- **硬件结构**：DRAM-PIM + DPA controller
- **评估层次**：仿真 + MLIR 编译器
- **报告指标**：8.54×/16.0× speedup
- **开源**：部分
- **主线映射**：对照B

### NeuPIM（heo2024nuepim）
- **出处**：ASPLOS 2024
- **问题切入点**：NPU + PIM 异构
- **数据流/精度**：batched LLM inferencing
- **硬件结构**：NPU-PIM heterogeneous
- **评估层次**：仿真
- **报告指标**：throughput、energy
- **开源**：否
- **主线映射**：对照B

---

## 统计摘要

| 主线 | 收录论文数 | 含 RTL/ASIC 评估 | 含开源 |
|------|-----------|-----------------|--------|
| L1 FlashAttention 数据流 | 8 | 2 (FSA, DESA) | 1 (FlashAttention) |
| L2 混合精度 KV | 8 | 0 (GPU kernel 为主) | 4 |
| L3 非 GEMM | 7 | 4 | 0 |
| L4 编译/工具 | 7 | — | 4 |
| 对照 A 稀疏 | 2 | 1 | 0 |
| 对照 B 访存中心 | 3 | 0 | 0 |
