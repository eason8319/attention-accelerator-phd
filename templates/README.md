# templates — 文本类产出模板库

全仓库**文本产出**的共用模板与范例（LaTeX、后续可扩展 Markdown 等）。  
正文与文献库仍放在各自工作目录（如 `survey/manuscript/`）；此处只放可复用的类文件、官方说明与骨架。

## 目录

```
templates/
├── README.md                 # 本说明
└── latex/
    └── ieee/                 # IEEE 会议稿模板（原 survey 下 letax 模板）
        ├── IEEEtran.cls      # 文档类（权威副本）
        ├── IEEEtran_HOWTO.pdf
        ├── conference_101719.pdf   # 官方示例稿
        └── fig1.png                # 示例配图
```

## 使用约定

1. **新开 LaTeX 文稿**：从 `templates/latex/ieee/` 复制 `IEEEtran.cls`（及需要的示例）到文稿目录，再写 `.tex` / `.bib`。
2. **现有综述稿**：[`survey/manuscript/`](../survey/manuscript/) 内保留一份 `IEEEtran.cls` 以便本目录自包含编译；若升级类文件，以本库 `ieee/IEEEtran.cls` 为准并同步复制。
3. **编译产物**（论文 PDF、`.aux` 等）仍不进 Git（见根目录 `.gitignore`）；模板目录内的说明 PDF 例外，允许入库。

## 后续可扩展

按需在本目录增加子文件夹，例如：

- `markdown/` — 短文 / 验收报告骨架  
- `latex/journal/` — 期刊类模板  
