# lit_watch / 对比手册修订记录

按时间倒序。

---

## 2026-09-03（全文定量与发表状态复核）

- 对 `ledger.yaml` 全部 21 篇记录完成全文级定量复核；每个结果补齐模型/负载、平台、基线、正文表图位置与适用边界，详见 `AUDIT_2026-09-03.md`。
- 当前状态汇总为 10 篇正式会刊/期刊或 workshop、11 篇预印本/在投；未发现重复 ID、DOI 或 BibTeX key。
- 状态升级：PLENA → ISCA 2026；AccLLM → IEEE TVLSI 2026；Don't Waste Bits → CVPR 2026 Workshops（LoViF）；Titanus 补齐 ACM DOI 与页码。
- 发现 SystolicAttention 摘要与 §6.1 对 1.77×/4.83× 的 TPUv5e/Neuron-v2 对应关系互相颠倒，标记为引用前待作者勘误。
- 维护规则改为：摘要定量数字必须记录，同时追溯正文表/图；保留 `up to` 限定，并报告负向结果、元数据、dense shadow、转换峰值与端到端口径。

---

## 2026-09-03（academic-researcher 迁入仓库）

- 项目 skill 落点改为 `.cursor/skills/academic-researcher/SKILL.md`（随 git，本机与 Cloud Agent 共用）。
- 规则与本 README 去掉 `/root/.cursor/skills/...` 绝对路径。

---

## 2026-09-03（2026 年 8 月增量检索）

- 将检索截止从 2026-07-23 更新至 2026-09-03；窗口内最新收录为 PuzzleKV（arXiv:2608.23843，首发 2026-08-24）。
- 新增并核实 4 篇预印本：SPECTRA（2608.07915）、AATC（2608.14191）、Minima-KV（2608.23834）、PuzzleKV（2608.23843）。
- 将 Minima-KV 标为 R2–R3 的最近直接对照：mixed-format paged attention、分格式 partial state、global online-softmax merge、无 cache-sized dense shadow。
- 本轮只依据 arXiv 原始条目核实元数据与摘要；未把摘要中的定量结果直接升级为对比表结论，正式引用前须复核正文表格。
- 扩充 `queries.md`：加入 transform coding、rate-distortion、mixed-format paged attention、page-wise compression 等查询。

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
