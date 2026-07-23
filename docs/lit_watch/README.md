# Literature Watch（文献监视）

本目录支持 [`../recent_works_comparison.md`](../recent_works_comparison.md) 的**增量更新**，流程为：

```text
queries.md 检索 → inbox.md 候选 → 人工核实 → ledger.yaml 入库
    → 更新 recent_works_comparison.md 总览表/卡片 → 写 CHANGELOG
```

## Agent 默认技能

在 Cursor 中更新本目录或对比手册时，**默认先加载并遵循** academic-researcher skill  
（`/root/.cursor/skills/academic-researcher/SKILL.md`）。  
项目规则：`.cursor/rules/lit-watch-academic-researcher.mdc`。

## 文件职责

| 文件 | 作用 |
|------|------|
| [`queries.md`](queries.md) | 固定检索词与数据源 |
| [`inbox.md`](inbox.md) | 未审候选（脚本或手工追加） |
| [`CARD_TEMPLATE.md`](CARD_TEMPLATE.md) | 入库卡片模板（复制后填写） |
| [`ledger.yaml`](ledger.yaml) | 已收录论文的**已核实**元数据台账（防重复） |
| [`CHANGELOG.md`](CHANGELOG.md) | 本监视目录与对比手册的修订记录 |

## 更新步骤（每次）

1. 用 `queries.md` 在 arXiv / ACL Anthology / IEEE Xplore / OpenReview 检索。
2. 新文写入 `inbox.md`（只填链接与一句话理由，不写未核实数字）。
3. 打开原文或正式会刊页，按 `CARD_TEMPLATE.md` 填写；**Venue 以会刊/DOI 为准，不以二手摘要为准**。
4. 将核实后的条目追加到 `ledger.yaml`（`status: verified`）。
5. 同步改 `recent_works_comparison.md` 总览表与分篇卡片。
6. 在 `CHANGELOG.md` 与对比手册顶部「修订记录」各记一行。

## 核实规则

- **正式录用/出版**：必须有 venue +（尽量）DOI 或 anthology/PMLR 链接。
- **预印本**：`venue: preprint`，写清 `arxiv_id` 与 `first_posted` / `last_updated`。
- **题名**：以正式出版题名为准；若与 arXiv 题名不同，两者都记在 ledger。
- **实验结果**：优先正文表格；摘要与正文冲突时标注 `result_source: abstract|table|body`。
- **禁止**未读全文就把加速比写入总览表。
