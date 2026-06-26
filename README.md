# Attention 加速器博士课题 — 研究进展仓库

> **注意：本仓库 [`eason8319/-`](https://github.com/eason8319/-) 已归档。** 请改用专用仓库 **[attention-accelerator-phd](https://github.com/eason8319/attention-accelerator-phd)**。  
> 迁移方法见 [docs/MIGRATE_TO_PHD_REPO.md](docs/MIGRATE_TO_PHD_REPO.md)。可手动关闭仍打开的 [PR #2](https://github.com/eason8319/-/pull/2)。

面向大语言模型推理的高能效 Attention 加速器及软硬件协同优化。

本仓库用于：**研究计划、文献索引、综述计划、进展日志** 的 Git 同步；**论文 PDF 在本机用脚本下载**，不进 Git。

## 快速开始（两台电脑通用）

### 1. 获取仓库

**Git（推荐）**
```powershell
git clone https://github.com/eason8319/attention-accelerator-phd.git
cd attention-accelerator-phd
```

**无 Git：浏览器 Download ZIP**  
https://github.com/eason8319/attention-accelerator-phd

建议本地路径：`C:\Users\21753\博士课题\attention-accelerator-phd`

### 2. 下载论文 PDF（本机一次）

```powershell
cd docs\survey\papers
python download_papers.py
```

约 45 篇、142 MB，按主线分子文件夹。

### 3. 日常同步

```powershell
# 开工前
git pull

# 有进展后（更新 docs/progress/CHANGELOG.md 等）
git add .
git commit -m "进展：简述本次更新"
git push
```

无 Git 可用 [GitHub Desktop](https://desktop.github.com/)。

---

## 目录索引

### 研究计划
| 文档 | 说明 |
|------|------|
| [docs/research_plan.md](docs/research_plan.md) | 完整研究计划 |
| [docs/00_background_and_baselines.md](docs/00_background_and_baselines.md) | 背景与对标基线 |
| [docs/metrics.md](docs/metrics.md) | 评估指标体系 |

### 文献与综述
| 文档 | 说明 |
|------|------|
| [docs/survey/survey_plan.md](docs/survey/survey_plan.md) | 综述写作计划 |
| [docs/survey/manuscript/](docs/survey/manuscript/) | **英文综述 LaTeX 稿** + BibTeX 文献库 |
| [docs/survey/paper_matrix.md](docs/survey/paper_matrix.md) | 文献对标矩阵 |
| [docs/survey/papers/download_papers.py](docs/survey/papers/download_papers.py) | **论文 PDF 下载脚本** |

### 研究进展（请在此更新）
| 文档 | 说明 |
|------|------|
| [docs/progress/README.md](docs/progress/README.md) | 进展目录说明 |
| [docs/progress/milestones.md](docs/progress/milestones.md) | 阶段里程碑 checklist |
| [docs/progress/CHANGELOG.md](docs/progress/CHANGELOG.md) | 进展日志 |
| [docs/progress/logs/](docs/progress/logs/) | 详细研究日志 |

---

## 云端 Cursor Agent

有进展时可让 Agent 更新 `docs/progress/` 并 push 到 GitHub；PDF 仍在本机用脚本维护。
