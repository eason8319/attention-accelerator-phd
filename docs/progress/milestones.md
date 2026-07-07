# 研究里程碑

对照 [research_plan.md](../research_plan.md) 第四节「研究阶段与里程碑」。

| 阶段 | 内容 | 产出 | 状态 |
|------|------|------|------|
| 阶段0 | 文献与基线调研 | survey + 选题报告 | 进行中 |
| 阶段1 | Baseline 建模与瓶颈定位 | 瓶颈分析 + 短文 | 未开始 |
| 阶段2 | FlashAttention-native 架构 | 架构论文 | 未开始 |
| 阶段3 | 混合精度 datapath | 核心论文 | 未开始 |
| 阶段4 | 非 GEMM 单元与协同调度 | 模块级论文 | 未开始 |
| 阶段5 | 编译映射框架 | 软硬件协同论文 | 未开始 |
| 阶段6 | 系统集成与博士论文 | PPA 评估 + 学位论文 | 未开始 |

## 阶段 0 细项

- [x] 文献 BibTeX 与对标矩阵
- [x] 综述写作计划
- [x] 论文下载脚本（本地执行）
- [ ] `00_overview.md` 综述总览
- [ ] `01`–`04` 四条主线分章
- [ ] `05` 对照与展望
- [ ] `06` gap 分析与定位
- [ ] 选题报告定稿

## 学习阶段里程碑（P1–P5）

对照 [learning_plan.md](../learning_plan.md)，与阶段 0 → 阶段 1 过渡期并行推进。

| 项目 | 内容 | 验收点 | 状态 |
|------|------|--------|------|
| P1 | Attention 数值内核复现 | 三种实现对拍通过 + rescale 推导笔记 | 已完成（pytest 22/22 通过，见 [验收报告](p1_attention_numerics_report.md)） |
| P2 | 低精度量化实验 | 旋转降误差复现 + KV INT4 困惑度报告 | 未开始 |
| P3 | 架构评估工具链 | SCALE-Sim/Timeloop 跑通 + `analysis.md` 瓶颈短文 | 未开始 |
| P4 | RTL 关键模块 | exp/softmax/systolic array 与 golden model 对拍通过 | 未开始 |
| P5 | tile-level 模拟器 | Pareto 前沿 + 与 SCALE-Sim 趋势校验 | 未开始 |
