# 英文综述 LaTeX 稿

本目录为 **IEEE 综述论文** 的自包含编译环境：正文、BibTeX 文献库与本地 `IEEEtran.cls`。

- 综述内容索引：[survey_overview.md](../survey_overview.md)
- 调查综述目录：[survey/](../)
- **模板权威副本**（全项目共用）：[`templates/latex/ieee/`](../../templates/latex/ieee/)

## 目录结构

```
survey/manuscript/
├── attention_accelerator_survey.tex   # 综述正文
├── references.bib                     # BibTeX 文献库
├── IEEEtran.cls                       # 本地副本（便于本目录编译）
└── README.md
```

## 编译

需安装 [MiKTeX](https://miktex.org/download) 或 TeX Live。在本目录执行：

```powershell
cd survey\manuscript
pdflatex attention_accelerator_survey
bibtex attention_accelerator_survey
pdflatex attention_accelerator_survey
pdflatex attention_accelerator_survey
```

正文使用 `\bibliography{references}`，与 `references.bib` 同目录，无需相对路径。

## 引用约定

- 仅使用 `references.bib` 中已有的 cite key
- 新增文献时同步更新 `references.bib` 并在正文中按 IEEE 格式引用
