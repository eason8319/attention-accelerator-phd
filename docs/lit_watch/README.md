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
| [`CHANGELOG.md`](CHANGELOG.md) | 本监视目录与对比手册的修订记录 |

不再按批次新增 `AUDIT_*.md`。核实后的数字、基线、正文位置和限制写入现有台账与对比手册；旧记录提及的 `AUDIT_2026-09-03.md` 当前缺失，不能作为现存证据引用。

## 更新步骤（每次）

全项目引用前的查重、核验和缺项补录要求见 [AGENTS.md](../../AGENTS.md#literature-registration)；本节只说明执行方法。已提供明确来源时可定向核验，广泛查新时再使用完整检索词；记录两者的范围。

1. 先查手册、ledger 及所属参考文献库，按题名、DOI/arXiv ID/规范 URL 匹配已有记录；缺项用 `queries.md` 或提供的原始来源核验。
2. 新文写入 `inbox.md`（只填链接与一句话理由；摘要中的数字可暂存为 `unverified_abstract_claim`，不得直接写成已核实结论）。
3. 打开原文或正式会刊页，按 `CARD_TEMPLATE.md` 填写；**Venue 以会刊/DOI 为准，不以二手摘要为准**。摘要定量数字必须保留，并追到正文表/图；追不到时明确标 `result_source: abstract`。
4. 核实后更新已有条目或补入 `ledger.yaml`（`status: verified`），记录版本、核验日期、来源位置及来源类型；verified 表示所列范围已核验，不等于已在本项目复现。
5. 同步手册总览与卡片；正式 BibTeX 引用补齐所属既有文献库并复用引用键，在 ledger 登记 bibtex_key。将 inbox 候选标为已入库，检查使用它的计划/报告链接。
6. 检查本次引用的手册/台账/引用键覆盖、重复 ID 与链接；在 `CHANGELOG.md` 和手册修订记录登记范围，区分定向补录与全量检索。

## 核实规则

- **非论文来源**：作者技术说明使用 `source_type: technical_post`、`venue_type: technical_post` 和原始 URL；DOI/arXiv 可为空。作者代码附在对应条目，不能凭仓库说明升级论文发表状态。台账 `papers` 是历史集合名，可容纳明确标注类型的技术来源。

- **正式录用/出版**：必须有 venue +（尽量）DOI 或 anthology/PMLR 链接。
- **预印本**：`venue: preprint`，写清 `arxiv_id` 与 `first_posted` / `last_updated`。
- **题名**：以正式出版题名为准；若与 arXiv 题名不同，两者都记在 ledger。
- **实验结果**：摘要数字不能省略；优先用正文表格补齐其模型、平台、基线、context、batch 与统计口径。摘要与正文冲突时同时保留并标注 `result_source: abstract|table|body`。
- **倍数语义**：保留 `up to` / 平均 / P50 / P99；速度、吞吐、延迟、容量和能效不得混为一个“加速比”。
- **系统边界**：记录是否为 kernel、单层、端到端 serving、模拟/RTL/硅片，以及是否包含元数据、dense shadow、临时缓冲和模型权重。
- **禁止**未读全文就把加速比写入总览表。
