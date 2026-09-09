# 研究里程碑

对照现行 [`research_plan.md`](../research_plan.md) 的深度阶段 **R0–R5**（不按日历年）。  
R1 已完成，结果与适用范围见 [R1 总报告](../../research/r1_kv_baseline/REPORT.md)。
旧「阶段 0–6 / 四主线」表已废止。

| 深度 | 内容 | 产出 | 状态 |
|------|------|------|------|
| R0 | Survey + Learning 技能与证据基线 | `survey/`、`learning/`、本仓库计划与对比手册 | **已完成**（持续文献监视除外） |
| R1 | 真实 KV cache-path、误差—流量模型、decode simulator | 可复现测量 + 协议锁定 | **已完成**（[报告与适用边界](../../research/r1_kv_baseline/REPORT.md)） |
| R2 | 静态 INT4（或 R1 选定主格式）流式通路；无完整 FP16 展开；关键 RTL | 架构主张 + 首版综合 | 未开始 |
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

完成依据与 R2 起步建议见 [R1 总报告](../../research/r1_kv_baseline/REPORT.md#6-结论与后续工作)，长期阶段目标见 [研究计划](../research_plan.md)。R2 尚未开始实施。

R1 的协议、编码对照、KIVI 任务评估、分页布局、长窗口精度与流量、敏感性及模拟器实验均已完成约定范围；实验入口统一见 [实验索引](../experiments.md)，不再保留临时阶段清单。

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
