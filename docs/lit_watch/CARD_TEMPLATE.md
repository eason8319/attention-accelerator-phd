# 论文卡片模板（入库用）

复制到 `recent_works_comparison.md` 分篇区，或先贴在笔记中。填完后把元数据写入 `ledger.yaml`。

```markdown
### SHORT_NAME（一句话角色）

- **来源类型**：论文 / 作者技术说明（代码作为附属实现来源）
- **题名（正式）**：
- **题名（arXiv，若不同）**：
- **作者**：
- **Venue / 状态**：会议或期刊全名 + 年份；或 `preprint (arXiv)`
- **标识**：arXiv: ; DOI: ; Anthology/PMLR:
- **代码**：（若有，作者仓库与具体文档/版本；实际实验固定提交）
- **引用键 / 版本**：（复用所属既有 .bib；标明实际使用版本）
- **平台**：GPU型号 / FPGA / ASIC工艺与评估层级（模拟/RTL/硅片）
- **方法要点**：（≤3 条）
- **实验设置**：模型；上下文；batch；任务集
- **摘要定量主张**：（逐项记录；保留 up to / 平均 / P50/P99）
- **正文复核结果**：（写清指标、相对谁、绝对值；注明 table/figure/section）
- **口径边界**：kernel/单层/端到端；模型；context；batch；硬件；模拟/RTL/硅片；是否含 metadata / dense shadow / 临时缓冲 / 权重
- **冲突或缺口**：（摘要与正文、版本间、表图间不一致；无则写“无”）
- **结论**：
- **对本课题**：可对齐点 / 不可比点
- **核实**：YYYY-MM-DD；核验来源（会刊页 / DOI / arXiv abs）
```

## 最小必填（写入 ledger 前）

- [ ] 正式题名
- [ ] 第一作者或通讯可定位
- [ ] venue、明确 preprint 或作者技术说明类型
- [ ] 论文记录 DOI/arXiv ID 或正式会刊 URL；作者技术说明记录规范原始 URL，不伪造 DOI/arXiv
- [ ] `bucket`：`algo_gpu` / `hw_asic_fpga` / `survey` / `adjacent`
- [ ] `status: verified` 且 `verified_on`、核验范围已填
- [ ] 摘要中的所有定量数字均已记录，并标注 `result_source`
- [ ] 每个倍数均可追溯到基线、平台、模型、context/batch 与正文位置
