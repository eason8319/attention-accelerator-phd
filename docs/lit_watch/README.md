# Literature Watch（文献监视）

本目录支持 [`../recent_works_comparison.md`](../recent_works_comparison.md) 的**增量更新**，流程为：

```text
queries.md 检索 → inbox.md 候选 → 人工核实 → ledger.yaml 入库
    → 更新 recent_works_comparison.md 总览表/卡片 → 写 CHANGELOG
```

## Agent 默认技能

在 Cursor 中更新本目录或对比手册时，**默认先加载并遵循** academic-researcher skill  
（[`.cursor/skills/academic-researcher/SKILL.md`](../../.cursor/skills/academic-researcher/SKILL.md)）。
项目规则：`.cursor/rules/lit-watch-academic-researcher.mdc`。

## 文件职责

| 文件 | 作用 |
|------|------|
| [`queries.md`](queries.md) | 固定检索词与数据源 |
| [`inbox.md`](inbox.md) | 未审候选（脚本或手工追加） |
| [`CARD_TEMPLATE.md`](CARD_TEMPLATE.md) | 入库卡片模板（复制后填写） |
| [`ledger.yaml`](ledger.yaml) | 已收录论文的**已核实**元数据台账（防重复） |
| [`AUDIT_2026-09-03.md`](AUDIT_2026-09-03.md) | 全文定量复核：数字、基线、设置、正文位置与限制 |
| [`CHANGELOG.md`](CHANGELOG.md) | 本监视目录与对比手册的修订记录 |

## 更新步骤（每次）

1. 用 `queries.md` 在 arXiv / ACL Anthology / IEEE Xplore / OpenReview 检索。
2. 新文写入 `inbox.md`（只填链接与一句话理由；摘要中的数字可暂存为 `unverified_abstract_claim`，不得直接写成已核实结论）。
3. 打开原文或正式会刊页，按 `CARD_TEMPLATE.md` 填写；**Venue 以会刊/DOI 为准，不以二手摘要为准**。摘要定量数字必须保留，并追到正文表/图；追不到时明确标 `result_source: abstract`。
4. 将核实后的条目追加到 `ledger.yaml`（`status: verified`）。
5. 同步改 `recent_works_comparison.md` 总览表与分篇卡片。
6. 在 `CHANGELOG.md` 与对比手册顶部「修订记录」各记一行。

## 核实规则

- **正式录用/出版**：必须有 venue +（尽量）DOI 或 anthology/PMLR 链接。
- **预印本**：`venue: preprint`，写清 `arxiv_id` 与 `first_posted` / `last_updated`。
- **题名**：以正式出版题名为准；若与 arXiv 题名不同，两者都记在 ledger。
- **实验结果**：摘要数字不能省略；优先用正文表格补齐其模型、平台、基线、context、batch 与统计口径。摘要与正文冲突时同时保留并标注 `result_source: abstract|table|body`。
- **倍数语义**：保留 `up to` / 平均 / P50 / P99；速度、吞吐、延迟、容量和能效不得混为一个“加速比”。
- **系统边界**：记录是否为 kernel、单层、端到端 serving、模拟/RTL/硅片，以及是否包含元数据、dense shadow、临时缓冲和模型权重。
- **禁止**未读全文就把加速比写入总览表。
