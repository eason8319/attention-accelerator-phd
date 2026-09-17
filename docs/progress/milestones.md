# 研究里程碑

对照现行 [`research_plan.md`](../research_plan.md) 的深度阶段 **R0–R5**（不按日历年）。  
R1 已完成，结果与适用范围见 [R1 总报告](../../research/r1_kv_baseline/REPORT.md)。
旧「阶段 0–6 / 四主线」表已废止。

| 深度 | 内容 | 产出 | 状态 |
|------|------|------|------|
| R0 | Survey + Learning 技能与证据基线 | `survey/`、`learning/`、本仓库计划与对比手册 | **已完成**（持续文献监视除外） |
| R1 | 真实 KV cache-path、误差—流量模型、decode simulator | 可复现测量 + 协议锁定 | **已完成**（[报告与适用边界](../../research/r1_kv_baseline/REPORT.md)） |
| R2 | 静态低比特流式通路、关键 RTL 与候选机制探索 | 基础通路 + 首版综合 + 探索证据与创新评估 | 进行中：步骤 1–6 已通过（协议为 `r2-evaluation-v2`），步骤 7 待开始 |
| R3 | 可规则化混合 / 结构感知比特分配 | 精度—硬件代价 Pareto | 未开始 |
| R4 | 精度—布局—映射联合优化 | 映射方法与系统评估 | 未开始 |
| R5 | 模拟器—RTL 校准、长压力测试、学位论文 | 校准实验包 + 论文 | 未开始 |

## R0 细项

- [x] 英文综述稿 `survey/manuscript/`
- [x] 综述内容整理 `survey/survey_overview.md`
- [x] 学习管线 P1–P5 与验收报告 `learning/`
- [x] 研究计划收敛为 R0–R5 `docs/research_plan.md`
- [x] 近年成果对比手册 + lit_watch `docs/recent_works_comparison.md`、`docs/lit_watch/`
- [x] 背景与基线对齐现行计划 `docs/00_background_and_baselines.md`
- [ ] 综述 PDF 本机编译与作者信息定稿（可选）
- [ ] 选题报告 / 开题材料（按学校要求，内容以 research_plan 为准）

## R1→R2 验收入口

完成依据与 R2 起步建议见 [R1 总报告](../../research/r1_kv_baseline/REPORT.md#6-结论与后续工作)，长期阶段目标见 [研究计划](../research_plan.md)。R2 协议现为 `r2-evaluation-v2`；资源与最小综合已审核。开发用 Qwen2.5-0.5B base 已写入本机 `HF_HOME`；集群按同一约定放入该机 `$HOME/hf-cache`。R1 仍使用 0.5B Instruct。Qwen2.5-7B-Instruct 与公开 GPU 内核仍须在步骤 9 前补齐。缓存规则见 [AGENTS.md](../../AGENTS.md#model-weight-cache)。

R1 的协议、编码对照、KIVI 任务评估、分页布局、长窗口精度与流量、敏感性及模拟器实验均已完成约定范围；实验入口统一见 [实验索引](../experiments.md)，不再保留临时阶段清单。

<a id="r2-review"></a>

## R2 逐步审核

截至 2026-09-17，步骤 1–6 已通过。步骤 5 依据为[单头操作数与 C3 混合功能基线报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)：C3 功能基线为原域 QK＋旋转域 PV/O，保留读侧 K 逆旋转，不称为纯 Q/O 优化成功；纯 Q/O 仅在出现明确新机制时再验证。步骤 6 已通过，依据为[公平映射与成本基线报告](../../research/r2_streaming_attention/experiments/fair_mapping_cost/REPORT.md)：优化 FP16 与共享流式解码采用同一资源和开发搜索空间；动作、调度与完整轨迹已核验。周期收益依赖供给假设，额外端口不作为等面积收益，C3 混合路径仍保留 K 逆旋转；结果及适用边界统一见报告。依据见下表所链报告。评测协议仍为 [`r2-evaluation-v2`](../../research/r2_streaming_attention/experiments/configs/evaluation_protocol.json)：质量门 PPL 改为每窗 prefill 建立压缩 KV、仅计分段走流式读路径，全程逐 token 只保留冻结小集；官方 SAW-INT4 FA3 时延仅在 H100/H800 上报，已验证 Ada 卡不作为该官方复现。CUDA、BitDecoding、SAW 不得装入 R1/步骤 2 的 `torch 2.5.1+cu121`，细则见协议 `gpu_kernels` 与 [隔离 GPU 运行时](../../AGENTS.md#isolated-gpu-runtimes)。v1 算量证据仍见[资源与最小算术报告](../../research/r2_streaming_attention/experiments/arithmetic_feasibility/REPORT.md)，1 ns 未通过不回头优化探针乘加。开发模型 `Qwen/Qwen2.5-0.5B` base 已写入本机 `HF_HOME`（与 R1 共用该机缓存，不互换 Instruct/base）；集群须按同一 ID/revision 放入该机 `$HOME/hf-cache`。7B Instruct 权重与 BitDecoding 构建仍待步骤 9。研究内容、对照与审核要求以[R2 详细计划的执行步骤](../../research/r2_streaming_attention/PLAN.md#r2-execution)为准，本表只维护状态与证据；步骤编号不用于实验或结果命名。状态按“待开始 → 进行中 → 待审核 → 已通过”更新，需调整时留在当前步骤；每一步经用户审核通过后才进入下一步，机器检查不能替代审核。探索步骤通过表示约定工作及有效性已审核，不表示候选创新成立。

| 步骤 | 审核对象 | 状态 | 证据入口 |
|---|---|---|---|
| 1 | 协议、假设与比较口径 | 已通过 | [协议解释与对照](../../research/r2_streaming_attention/PLAN.md#r2-protocol-review)、[共享配置](../../research/r2_streaming_attention/experiments/configs/evaluation_protocol.json) |
| 2 | 资源、工具链与最小综合 | 已通过 | [资源与最小算术报告](../../research/r2_streaming_attention/experiments/arithmetic_feasibility/REPORT.md)、[本地证据清单](../../research/r2_streaming_attention/experiments/arithmetic_feasibility/experiment.json) |
| 3 | 物理打包与追加 | 已通过 | [物理打包与追加报告](../../research/r2_streaming_attention/experiments/physical_pack_append/REPORT.md) |
| 4 | 分页、残差与读写计量 | 已通过 | [分页、残差与访存事件报告](../../research/r2_streaming_attention/experiments/paged_residual_access/REPORT.md) |
| 5 | 完整流式功能与旋转基线 | 已通过 | [单头操作数与 C3 混合功能基线](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md) |
| 6 | 公平映射与成本基线 | 已通过 | [公平映射与周期、能耗基线](../../research/r2_streaming_attention/experiments/fair_mapping_cost/REPORT.md) |
| 7 | 尺度处理与组供给探索 | 待开始 | — |
| 8 | 归并、布局与收益边界探索 | 待开始 | — |
| 9 | 双模型正式评测与 GPU 对照 | 待开始 | — |
| 10 | 关键 RTL 与完整小通路 | 待开始 | — |
| 11 | 综合、校准与收益复核 | 待开始 | — |
| 12 | 压力测试与阶段末评估 | 待开始 | — |

## R2 阶段末评估登记

评估尚未开展，规则见[基础验收与创新评估](../../research/r2_streaming_attention/PLAN.md#r2-assessment)。后续仅在此登记状态、处置和正式证据链接；候选机制的有效性、收益与新颖性分析写入所属实验唯一 REPORT.md，不在本表复制结果或结论段落。未运行与未发现收益分别记录。

| 评估对象 | 状态 | 处置 | 证据入口 |
|---|---|---|---|
| 基础通路技术验收 | 待评估 | — | — |
| 探索实验完成情况 | 待评估 | — | — |
| 尺度处理与 GQA 复用协同 | 待评估 | — | — |
| 有界缓冲下的组供给调度 | 待评估 | — | — |
| 缓存页与输出归并范围解耦 | 待评估 | — | — |
| 压缩收益边界 | 待评估 | — | — |
| 后续候选及 R3/R4 去向 | 待评估 | — | — |

## 学习阶段（P1–P5）— 已归档

P1–P5 属 R0 技能建设，保留历史验收记录；P1 当前数值检查有一项未达阈值，详见对应报告。旧「主线1–4」映射仅作历史说明。
详见 [`learning/README.md`](../../learning/README.md)。

| 项目 | 状态 |
|------|------|
| P1 Attention 数值 | 21/22 检查通过，FP16 online 待核验 |
| P2 量化（含 proxy 局限） | 已完成 |
| P3 架构评估 | 已完成 |
| P4 RTL 玩具模块 | 已完成 |
| P5 tile 模拟器 | 已完成 |
