# 研究里程碑

对照现行 [`research_plan.md`](../research_plan.md) 的深度阶段 **R0–R5**（不按日历年）。  
旧「阶段 0–6 / 四主线」表已废止。

| 深度 | 内容 | 产出 | 状态 |
|------|------|------|------|
| R0 | Survey + Learning 技能与证据基线 | `survey/`、`learning/`、本仓库计划与对比手册 | **已完成**（持续文献监视除外） |
| R1 | 真实 KV cache-path、误差—流量模型、decode simulator 骨架 | 可复现测量 + 协议锁定 | **下一步 / 未开始** |
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

## R1 入口门槛（摘自计划）

进入 R2 前须全部满足：

1. 真实 token-wise KV quantize→store→load→dequant→attention 可复现；与 proxy 差异已文档化  
2. 至少一条长上下文 bytes/token–精度 Pareto  
3. 专用模拟器与独立工具在约定检查点趋势一致  
4. 默认模型列表与硬件包络假设已锁定  

工作目录建议：[`../../research/`](../../research/README.md)。

## 学习阶段（P1–P5）— 已归档

P1–P5 **已全部完成**，属 R0 技能建设；**不再定义课题主线**（旧「主线1–4」映射仅作历史说明）。  
详见 [`learning/README.md`](../../learning/README.md)。

| 项目 | 状态 |
|------|------|
| P1 Attention 数值 | 已完成 |
| P2 量化（含 proxy 局限） | 已完成 |
| P3 架构评估 | 已完成 |
| P4 RTL 玩具模块 | 已完成 |
| P5 tile 模拟器 | 已完成 |
