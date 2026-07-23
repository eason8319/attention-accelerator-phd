# Attention 加速器博士课题 — 研究进展仓库

面向**长上下文 decode** 的精度感知 KV Cache 流式 Attention 架构与映射（单芯片推理；架构模拟 + 关键通路 RTL/PPA）。

**现行研究计划（唯一真源）：** [docs/research_plan.md](docs/research_plan.md)（深度 R0–R5）  
**GitHub：** [github.com/eason8319/attention-accelerator-phd](https://github.com/eason8319/attention-accelerator-phd)

本仓库用于：研究计划、文献综述与对比手册、学习管线、进展日志的 Git 同步；**综述 PDF 在本机编译**，不进 Git。

## 目录索引

### 研究计划与对照
| 文档 | 说明 |
|------|------|
| [docs/research_plan.md](docs/research_plan.md) | **现行**长线研究计划（R0–R5） |
| [docs/00_background_and_baselines.md](docs/00_background_and_baselines.md) | 背景、基线与范围（已对齐现行计划） |
| [docs/metrics.md](docs/metrics.md) | 评估指标体系 |
| [docs/recent_works_comparison.md](docs/recent_works_comparison.md) | 近年成果对比手册 |
| [docs/lit_watch/](docs/lit_watch/) | 文献监视（queries / inbox / ledger） |
| [research/](research/) | **正式研究入口**（自 R1 起） |

### 学习项目（P1–P5，已归档为 R0）
| 目录 | 说明 |
|------|------|
| [learning/](learning/) | 技能建设（已完成）；不定义课题主线 |
| [learning/p1_attention_numerics/](learning/p1_attention_numerics/) | Attention / online softmax 数值 |
| [learning/p2_quantization/](learning/p2_quantization/) | 量化与旋转（含 proxy 局限） |
| [learning/p3_arch_eval/](learning/p3_arch_eval/) | SCALE-Sim / Timeloop |
| [learning/p4_rtl/](learning/p4_rtl/) | exp / softmax / 小阵列 RTL |
| [learning/p5_tile_sim/](learning/p5_tile_sim/) | tile-level 模拟器 |
| [learning/manuscript/](learning/manuscript/) | P1–P5 英文综合稿 |

Python 依赖见 [requirements.txt](requirements.txt)。

### 文献与综述（R0）
| 文档 | 说明 |
|------|------|
| [survey/](survey/) | 初期调查综述（L1–L4 地图；定位脚注对齐现行计划） |
| [survey/survey_overview.md](survey/survey_overview.md) | 综述内容整理 |
| [survey/manuscript/](survey/manuscript/) | 英文综述 LaTeX + BibTeX |
| [templates/](templates/) | 文本类产出模板 |

### 研究进展
| 文档 | 说明 |
|------|------|
| [docs/progress/README.md](docs/progress/README.md) | 进展目录说明 |
| [docs/progress/milestones.md](docs/progress/milestones.md) | R0–R5 checklist |
| [docs/progress/CHANGELOG.md](docs/progress/CHANGELOG.md) | 进展日志 |
| [docs/progress/logs/](docs/progress/logs/) | 详细研究日志 |
