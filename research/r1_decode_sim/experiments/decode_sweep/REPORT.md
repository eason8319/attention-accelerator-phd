# R1 Attention 模拟器实验

实验日期：2026-09-09；整理日期：2026-09-09。完成范围：流量与精度输入适配、attention 周期模型、小规模检查、完整网格、D(16384,1024) 压力轨迹与独立趋势对照。证据来源：本地流量与精度原始流量与 WikiText 结果、CPU cache 计数采集、CPU 周期模型运行，以及本次重新执行的 SCALE-Sim 3.0.0。当前进度以[里程碑](../../../../docs/progress/milestones.md)为准。

## 1. 实验目的

检验包含元数据的 KV 有效比特能否可靠接入独立的 attention 模拟器，并回答三个问题：上下文增长和量化如何改变流量；计算映射、缓冲容量及解码开销如何影响模拟时延；模拟器在相同负载和硬件包络下是否符合独立 Roofline 与 SCALE-Sim 的基本趋势。

本实验交付架构记账与趋势证据。周期和时延是模型输出，适用对象为 attention；跨实验验收见 [R1 总报告](../../../r1_kv_baseline/REPORT.md)。

## 2. 方法与设置

### 负载、输入与完整网格

使用 Llama-3.1-8B-Instruct 几何：32 个查询头、8 个 KV 头、head dimension=128、32 层、batch=1。每层每个缓存 token 的 K+V 共 2048 个元素。输入包含四个窗口 4096、8192、16384、32768，C0–C5 六种格式及 contiguous/paged 两种布局；分页大小为 16 token，PTE 为 8 字节，C0 走 FP16 codec。

权威流量来自 [流量与精度 summary.json](../../../r1_kv_baseline/experiments/kv_pareto/results/summary.json)。payload、scale、zero point、page 均已包含在单步总量中；全模型字节只除一次层数后供逐层模拟。四窗口的 24 个 WikiText PPL 结果只作为既有窗口精度参照，保留其 contiguous 评测范围，本实验未重跑模型精度。

固定硬件为 32×32 PE、1 GHz、16 MiB SRAM、1 TB/s HBM，每 PE 每周期 1 MAC，即峰值 1.024 TMAC/s 或 2.048 TOP/s。扫描 KV tile=128/512/2048，query tile=32、`query_rows` 头串行映射及 `auto` 缓冲策略。decode 的有效查询行数为 1；prefill 使用完整稠密矩形，每个 Q tile 重新读取 KV，不跳过因果屏蔽块。完整网格为 4×6×2×3×2=288 个候选。

模拟器分别计算 QKᵀ、PV、softmax、反量化和 BDR 逆旋转；默认辅助吞吐为 32 score/cycle、256 quantized elements/cycle 和 256 rotation MAC/cycle。KIVI 的 FP16 残差不计反量化工作。SRAM 同时计入打包 KV、解码后 FP16 KV、Q、FP32 score/累加器/状态，以及 BDR 暂存；双缓冲仅复制 KV 缓冲。DMA 与串行计算通路按缓冲释放时间重叠，各层与生成步串行累计。详细假设见[实现说明](../../README.md#模拟器主体)。

### 压力输入与逐步累计

D(16384,1024) 的 12 个格式/布局组合通过现有 cache 实现实际采集。CPU 以零值 K/V 按真实形状初始化，每次读取当前 `bytes_breakdown()` 后再追加下一个 token，共获得 12,288 条名义字节计数。首步 N=16384，末步 N=17407；首步、末步分项及全程总量均与流量与精度输入一致。原始证据为 [pressure_trace.json](results/pressure_capture/pressure_trace.json)，实际环境与源码标识见同目录 [run_config.json](results/pressure_capture/run_config.json)。

三个 KV tile 各模拟全部轨迹，共 36 组、36,864 次逐步仿真。全程平均时延由所有步骤周期求和后除以 1024 得到。任一步不可行，该轨迹的全程总时延和均值均记为不可用；末步结果独立保留。

### 独立参照与验收口径

对照方案在执行外部参照前保存为 [acceptance_plan.json](results/independent_os/acceptance_plan.json)。解析参照只读取流量与精度原始记录，未导入模拟器计算、分块或调度函数；外部工具也不运行时依赖 `learning/`。

Roofline 使用同一计算范围和 Q-tile KV 重读约定。令 Q 为查询 token 数、N 为缓存长度、H=32、d=128、L=32，则 QKᵀ+PV 的总 MAC 为 `2QNHdL`；HBM 字节为 `ceil(Q/32) × KV_bytes + 4QHdL`，后项是 FP16 Q 读取和最终 FP16 O 写回。计算、带宽下界分别为 MAC/1024 和 HBM_bytes/1000 个周期，取二者较大值。压力总量用独立的等差数列公式求 N 的全程和。

“decode 更偏存储”检查为 decode 算术强度低于 prefill、带宽下界与计算下界的比值更大。近线性流量检查预设相邻窗口倍增偏差不超过 10%，允许固定 KIVI 残差造成偏移。降低有效比特检查针对字节方向，不要求完整周期一定下降。

[SCALE-Sim 官方实现](https://github.com/scalesim-project/SCALE-Sim)重新运行 18 个 QK/PV 形状：每头 QK=(Q_tile, KV_tile, 128)，PV=(Q_tile, 128, KV_tile)。每个代表形状的原始周期按精确完整块数、尾块、查询头和层数展开，形成 66 条算子参照。压力检查选 N=16384、16511、16512、17407，覆盖起点、128-token 残差边界及末步；144 个压力检查点中，可行点为 128 个。

SCALE-Sim 使用 OS、32×32 阵列及同一 16 MiB 总 SRAM；其 IFMAP/Filter/OFMAP 缓冲分为 6/6/4 MiB，显式使用 2 字节 FP16 word，固定通道带宽为 167/167/166 word/cycle，总计 1 TB/s。采用公开的 external-memory 接口设置 word 大小，未修改上游算法。比较量为上游 `Total Cycles − Stall Cycles` 与本模拟器 QK/PV 组件周期。上游包含阵列填充、排空等代价，本模型省略这些代价，因此检查组件周期包络及相对趋势，记录绝对差异，不设置绝对时延相等要求。

本次参照运行环境为 WSL CPU、Python 3.13.13、SCALE-Sim 3.0.0、NumPy 2.2.6；依赖声明见 [requirements-independent.txt](requirements-independent.txt)，完整软件版本及上游源码哈希见 [run_config.json](results/independent_os/run_config.json)。

## 3. 实验结果

### 覆盖与有效性

完整网格 [grid.json](results/full_grid/grid.json) 保留 288 个候选，224 个可行、64 个因 SRAM 不足不可行。不可行项全部来自 KV tile=2048：decode 的 C0/C3 共 16 项，prefill 全格式共 48 项。tile=128/512 的网格均可行。

压力 [pressure_summary.json](results/full_grid/pressure_summary.json) 保留 36 组轨迹，其中 32 组全程可行，4 组为 tile=2048 下 C0/C3 的双布局。全部步骤记录在 [pressure_steps.jsonl](results/full_grid/pressure_steps.jsonl)。完整批次的 59 项输入、主体和压力检查通过；小规模参照的 256 个调度案例、2016 个合成网格案例及 12 项入口检查通过记录见 [small_checks/checks.json](results/small_checks/checks.json)。

以下摘取 contiguous、tile=512 的结果；时延均为 32 层 attention 的模拟值，MiB=2²⁰ 字节。

| 格式 | 32K KV 读 / MiB | 32K decode / ms | 压力全程均值 / ms | 压力末步 / ms |
|---|---:|---:|---:|---:|
| C0 FP16 | 4096 | 269.552 | 139.080 | 143.227 |
| C1 INT8 | 2080 | 277.907 | 143.372 | 147.650 |
| C2 INT4 | 1152 | 277.892 | 143.357 | 147.635 |
| C3 INT4+BDR | 1152 | 546.327 | 281.765 | 290.233 |
| C4 KIVI-2 | 774.5 | 277.869 | 143.327 | 147.596 |
| C5 KIVI-4 | 1285.5 | 277.878 | 143.335 | 147.605 |

### 独立对照

本次 5 项解析参照手算测试通过；[comparison_checks.json](results/independent_os/comparison_checks.json) 的 1,414 项对照检查全部通过。它们是网格与检查点上的逐项比较，不代表 1,414 次独立硬件测量。

| 对照项目 | 覆盖 | 结果 |
|---|---:|---|
| Roofline 工作量、流量一致 | 288 个网格点 | 一致 |
| 可行时延不低于 Roofline 下界 | 224 个网格点、32 组压力轨迹 | 全部满足 |
| 流量随 N 近线性 | 36 次窗口倍增 | 字节比 1.936585–2.000000 |
| decode 相对更依赖带宽 | 48 个格式/布局/窗口组合 | prefill 算术强度为 decode 的 27.814–31.879 倍 |
| 低有效比特降低 KV 字节 | 40 次与 C0 的比较 | 全部下降 |
| SCALE-Sim 计算周期包络 | 448 次网格算子、256 次压力算子比较 | 全部满足 |
| SCALE-Sim 利用率与 N 趋势 | 24 次 decode/prefill、18 次窗口倍增 | decode 利用率更低；decode 周期线性增长 |

详细解析数值见 [roofline.json](results/independent_os/roofline.json) 与 [pressures.json](results/independent_os/pressures.json)。上游直接产生的 [COMPUTE_REPORT.csv](results/independent_os/scalesim/COMPUTE_REPORT.csv)、[BANDWIDTH_REPORT.csv](results/independent_os/scalesim/BANDWIDTH_REPORT.csv)、[DETAILED_ACCESS_REPORT.csv](results/independent_os/scalesim/DETAILED_ACCESS_REPORT.csv) 均保留；算子展开关系见 [gemms.json](results/independent_os/gemms.json)。

例如 N=16384、KV tile=512 时，SCALE-Sim 的 QK 阵列利用率为 decode 2.106%、prefill 67.391%，PV 为 2.789% 和 89.237%。网格中上游计算周期为本模型 QK/PV 组件周期的 1.030–1.484 倍，绝对差异明确存在。

## 4. 分析与讨论

流量方向符合有效比特口径：C2 的 32K KV 字节约为 C0 的 28.1%，C4 更低；固定元数据和 FP16 残差使实际比例偏离名义 4-bit/2-bit 比例。KIVI 的固定残差也解释了窗口倍增时略小于 2 的字节比。paged 元数据在原始输入中已计入，本实验保存了其独立布局行。

在本实验锁定的峰值下，Roofline 的 ridge point 只有 2.048 ops/byte；288 个网格点均落在解析计算侧受限区间。以 16K C0 为例，decode 算术强度约 3.999 ops/byte，计算下界 4.194 ms、带宽下界 2.148 ms；prefill 算术强度约 127.008 ops/byte。decode 对带宽的相对敏感性更高，但本包络并不支持“decode 必然由 HBM 主导”的绝对结论。

SCALE-Sim 同样显示单查询行的阵列利用率低。默认 `query_rows` 串行处理各查询头，decode 的大量 PE 行空闲，降低 KV 字节难以缩短主导计算阶段；反量化反而增加串行工作。因此表中低比特格式的模拟时延略高于 C0，而 C3 的逆旋转假设使时延明显增加。这是当前映射与辅助吞吐设定下的结果，不可据此断言量化在真实硬件上没有加速价值。

独立参照与本模型之间的 3.0%–48.4% 算子周期差异支持把本模型视为简化架构模型。SCALE-Sim 的填充、排空及波次代价随 GEMM 形状变化，单个比例不能作为统一校准系数。本次接受趋势一致性，后续校准应按算子、形状和数据通路分别进行。

压力全程均值与末步时延有稳定区别：C0 的平均 139.080 ms 低于末步 143.227 ms，且两者都来自实际逐步计数驱动的调度。用末步代替均值会高估该段生成的累计 attention 成本；只检查末步也不足以排除中途不可行，完整轨迹因此保留每一步状态。

## 5. 局限与有效性

- 结果不含投影、MLP、权重读取、cache append、采样或完整 decoder 流水，不能转写为整模实测 tokens/s；prefill 为派生稠密矩形。
- cache 采集测得的是实现的名义打包字节计数，未测 HBM 时间、实际 PyTorch 存储或物理 DMA 事务。tile 内元数据与 KIVI 工作量采用守恒分摊，页指针、bank conflict 和互连停顿尚未建模。
- GQA 复用与融合 attention 的 KV 流量由流量与精度和独立解析记账核对。SCALE-Sim 的独立 GEMM 存储访问不具备同一融合、codec 或共享 DMA 语义，其原始 DRAM 字节不作为解码模拟器 KV 流量的替代值。
- SCALE-Sim 共用相同阵列和总硬件包络，但使用固定三通道带宽及分区缓冲。对照只覆盖 QK/PV 阵列趋势，不校准反量化、旋转、softmax 或完整 attention 时延。
- 周期模型的全部 1024 个压力步骤已执行；SCALE-Sim 独立对照只取声明的四个检查点，未独立验证每一个中间步骤。压力点也没有长生成精度测量。
- 本实验输入与原始结果只在本地保留，GitHub 的正式报告链接不意味着这些数据已公开。软件版本、源码/结果哈希及来源映射由 [experiment.json](experiment.json) 登记；SRAM 不可行候选保留为容量边界证据，不计入可行候选的时延统计。

## 6. 结论与后续工作

在声明的 attention 范围、硬件包络和趋势标准内，输入守恒、完整网格、压力轨迹及 Roofline/SCALE-Sim 独立对照均完成，解码模拟器趋势验收通过。有效比特确实降低 KV 字节；实际模拟时延还取决于阵列映射、SRAM 容量及辅助计算吞吐。

真实 cache-path、精度—流量与本实验的联合分析见 [R1 总验收报告](../../../r1_kv_baseline/REPORT.md)。绝对时延、辅助单元吞吐和物理访存仍需后续架构/RTL 校准；本报告不代替这些工作。
