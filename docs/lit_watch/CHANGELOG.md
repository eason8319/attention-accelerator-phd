# lit_watch / 对比手册修订记录

按时间倒序。

---

## 2026-07-23（Agent 默认技能）

- 新增项目规则 `.cursor/rules/lit-watch-academic-researcher.mdc`：更新文献 / `lit_watch` / 对比手册时默认先遵循 academic-researcher skill。
- 本 README 增加「Agent 默认技能」说明。

---

## 2026-07-23

- 新建 `docs/lit_watch/`：`README`、`queries`、`inbox`、`CARD_TEMPLATE`、`ledger.yaml`。
- 对对比手册中核心条目做元数据核实（arXiv API + PMLR/ACL/DOI），并写入 `ledger.yaml`。
- **主要更正**：
  - MiniKV：正式题名以 ACL Anthology 为准（与 arXiv 题名不同）；Venue = ACL 2025 Findings；DOI `10.18653/v1/2025.findings-acl.952`。
  - KV 服务综述 `2607.08057`：非“仅预印本”，为 **ACL 2026 Findings**（DOI `10.18653/v1/2026.findings-acl.1916`）。
  - Don’t Waste Bits：arXiv comment 标明 **Accepted by CVPR 2026**（proceedings DOI 待补）。
  - Titanus：**GLSVLSI 2025**（非仅 arXiv）。
  - SystolicAttention：作者为 Jiawei Lin 等；截止日仍为 **preprint**。
  - PLENA / SAW-INT4 / InnerQ / Block-GTQ / UltraQuant / AccLLM / Salca / FlatAttention：截止日标为 preprint 或 “submitted”，避免误标已发表。
  - KIVI：PMLR 235:32332–32344；作者含 **Hongye Jin**。
  - KVTuner：PMLR 267:36451–36485；作者 Xing Li 等。
- 重组 [`../recent_works_comparison.md`](../recent_works_comparison.md)：增加修订记录、核实状态列、指向 lit_watch 的维护入口。
