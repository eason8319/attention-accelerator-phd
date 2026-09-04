---
name: academic-researcher
description: >-
  Verifies and incrementally updates literature for this attention-accelerator
  PhD repo (lit_watch, comparison handbook, related work, BibTeX). Use when
  searching papers, checking venue/DOI, filling ledger cards, updating
  recent_works_comparison.md, or adding citations. Never invent bibliographic
  metadata or experimental numbers.
---

# Academic Researcher

本仓库的文献核实与增量更新技能。路径一律相对仓库根目录，**不要写** `/root/.cursor/skills/` 或 `C:\Users\...`。

先读 [`docs/lit_watch/README.md`](../../../docs/lit_watch/README.md)。定量声明随后用 **academic-citation-guard**；综述结构用 **literature-review-writer**。

## Hard rules

1. **禁止编造**：未定位并核实题名、作者、venue、年份、DOI/URL 的论文，不得写入 `ledger.yaml`、对比表或 `.bib`。
2. **状态精确**：区分 `proceedings` / `journal` / `findings` / `workshop` / `preprint`。arXiv comment 写 “accepted” 时，未找到会刊页/DOI 仍标 preprint，并注明 DOI 待补。
3. **数字先核实**：实验结果优先正文表格；摘要与正文冲突须标注来源（`abstract` / `table` / `body`）。禁止把摘要倍数直接升级为总览表结论。
4. **Venue 以会刊/DOI 为准**，不以二手博客、模型记忆或 PDF 页眉猜测为准。
5. **不跨平台比绝对倍数**（GPU vs FPGA vs ASIC 模拟）。对本课题写「可对齐 / 不可比」。
6. 新 cite key 必须先入 `.bib` 再 `\cite{}`。

## When this applies

- 更新 `docs/lit_watch/`（检索、inbox、ledger、卡片、审计）
- 更新 `docs/recent_works_comparison.md`
- 为 survey / research 增补参考文献或 related work
- 用户说「更新文献」「查新论文」「核实 venue/DOI」「文献监视」

## Workflow

```text
queries.md → inbox.md → 打开原文/会刊页核实 → ledger.yaml
  → recent_works_comparison.md 总览表/卡片 → lit_watch/CHANGELOG.md
  → 手册顶部修订记录；必要时改 cutoff
```

1. 用 [`docs/lit_watch/queries.md`](../../../docs/lit_watch/queries.md) 检索 arXiv / ACL Anthology / IEEE Xplore / OpenReview / PMLR。
2. 候选只进 `inbox.md`：链接 + 一句话理由，**不写未核实数字**。
3. 打开原文或正式会刊页，按 [`docs/lit_watch/CARD_TEMPLATE.md`](../../../docs/lit_watch/CARD_TEMPLATE.md) 填卡片。
4. 与 `ledger.yaml` 的 `id` / `arxiv_id` / `doi` 去重；已有条目则更新字段，不另造 id。
5. 核实后追加或修改 `ledger.yaml`（`status: verified`，`verified_on`，`verify_sources`）。
6. 同步对比手册总览表与分篇卡片；Cutoff 不早于本次成功检索日。
7. `docs/lit_watch/CHANGELOG.md` 与手册「修订记录」各记一行。

未完成核实的条目留在 inbox，**不要**写入总览表主行。

## Verification sources (priority)

1. 会刊页、DOI、ACL Anthology、PMLR
2. IEEE Xplore / ACM DL 记录
3. arXiv abs + API（预印本元数据）；题名以正式出版为准，若与 arXiv 不同则 ledger 两栏都记
4. 作者 GitHub CITATION / 项目页（仅作线索，venue 仍须 1–2）

检索时用 WebSearch / WebFetch；不要用训练记忆填 Venue 或加速比。

## Ledger fields

新条目对齐现有 `docs/lit_watch/ledger.yaml`：

- `id`：短横线小写 slug（如 `minima_kv`）
- `bucket`：`algo_gpu` / `hw_asic_fpga` / `survey` / `adjacent`
- `venue_type`：`proceedings` | `journal` | `findings` | `preprint` | `workshop`
- `arxiv_id` 或 `doi` 至少其一；`canonical_url` 指向最权威页
- 定量细节可放 `notes`，并指向 table/section；完整审计另见 `AUDIT_*.md`

## Comparison handbook

填写 [`docs/recent_works_comparison.md`](../../../docs/recent_works_comparison.md) 时：

- 绑定 GPU / 模型 / seq / batch / 基线后再写倍数
- 「最高/平均」必须带来源段落
- 对本课题：对齐 R1 KV 基线、R2 混合格式、硬件主线时写清是否可复现、是否 paged、是否物化 FP16 KV
- 公式用 `$...$` / `$$...$$`，不用 `\(...\)`

## Out of scope (default)

不把下列工作写入主对照表（可标 `adjacent`）：分布式训练、仅 MoE routing、纯 CIM/PIM 且无 attention datapath、仅训练量化而无推理 KV。
