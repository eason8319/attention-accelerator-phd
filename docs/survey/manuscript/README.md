# 英文综述 LaTeX 稿

本目录为 **IEEE 综述论文** 的自包含写作环境：LaTeX 模板、正文与 BibTeX 文献库在同一文件夹，便于编译与版本管理。

## 目录结构

```
docs/survey/manuscript/
├── attention_accelerator_survey.tex   # 综述正文
├── references.bib                     # BibTeX 文献库（与 paper_matrix 对齐）
├── IEEEtran.cls                       # IEEE 会议模板（自 letax模板 复制）
└── README.md                          # 本说明
```

配套资料（仍在 `docs/survey/` 上级目录）：

- [survey_plan.md](../survey_plan.md) — 写作计划与分类法
- [paper_matrix.md](../paper_matrix.md) — 文献对标矩阵
- [papers/](../papers/) — 本地 PDF 库

原始 IEEE 空白模板见 [letax模板/](../letax模板/conference_101719.tex)。

## 编译

需安装 [MiKTeX](https://miktex.org/download) 或 TeX Live。在本目录执行：

```powershell
cd docs\survey\manuscript
pdflatex attention_accelerator_survey
bibtex attention_accelerator_survey
pdflatex attention_accelerator_survey
pdflatex attention_accelerator_survey
```

正文使用 `\bibliography{references}`，与 `references.bib` 同目录，无需相对路径。

## 引用约定

- 仅使用 `references.bib` 中已有的 cite key
- 新增文献时同步更新 [paper_matrix.md](../paper_matrix.md)
