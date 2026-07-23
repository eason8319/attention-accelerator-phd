# Attention 加速器博士课题 — 研究进展仓库

面向大语言模型推理的高能效 Attention 加速器及软硬件协同优化。

**GitHub：** [github.com/eason8319/attention-accelerator-phd](https://github.com/eason8319/attention-accelerator-phd)

本仓库用于：**研究计划、文献综述 LaTeX 稿、进展日志** 的 Git 同步；**综述 PDF 在本机编译**，不进 Git。

## 目录索引

### 研究计划
| 文档 | 说明 |
|------|------|
| [docs/research_plan.md](docs/research_plan.md) | 完整研究计划 |
| [docs/00_background_and_baselines.md](docs/00_background_and_baselines.md) | 背景与对标基线 |
| [docs/metrics.md](docs/metrics.md) | 评估指标体系 |
| [learning/](learning/) | **项目式学习方案（P1–P5，已完成）** |

### 学习项目（P1–P5）
| 目录 | 说明 |
|------|------|
| [learning/](learning/) | **统一父目录**（计划 + 代码 + 验收报告） |
| [learning/p1_attention_numerics/](learning/p1_attention_numerics/) | P1：attention / online softmax 数值内核复现 |
| [learning/p2_quantization/](learning/p2_quantization/) | P2：INT4/FP8/MXFP4 量化与旋转实验 |
| [learning/p3_arch_eval/](learning/p3_arch_eval/) | P3：SCALE-Sim / Timeloop 架构评估 |
| [learning/p4_rtl/](learning/p4_rtl/) | P4：exp / softmax / systolic array RTL 模块 |
| [learning/p5_tile_sim/](learning/p5_tile_sim/) | P5：tile-level 性能模拟器 |

Python 依赖见 [requirements.txt](requirements.txt)。

### 文献与综述（阶段 0）
| 文档 | 说明 |
|------|------|
| [survey/](survey/) | **项目初期调查综述**（总览 + 英文稿 + 论文库） |
| [survey/survey_overview.md](survey/survey_overview.md) | 综述内容整理（章节、分类法、结论） |
| [survey/manuscript/](survey/manuscript/) | 英文综述 LaTeX 稿 + BibTeX 文献库（编译 PDF） |
| [templates/](templates/) | **文本类产出模板**（LaTeX IEEE 等） |

### 研究进展（请在此更新）
| 文档 | 说明 |
|------|------|
| [docs/progress/README.md](docs/progress/README.md) | 进展目录说明 |
| [docs/progress/milestones.md](docs/progress/milestones.md) | 阶段里程碑 checklist |
| [docs/progress/CHANGELOG.md](docs/progress/CHANGELOG.md) | 进展日志 |
| [docs/progress/logs/](docs/progress/logs/) | 详细研究日志 |
