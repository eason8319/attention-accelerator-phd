# 实验结果与报告索引

本文件只维护实验入口与证据角色。当前进度见 [研究里程碑](progress/milestones.md)；实验数值、结论与有效性限制以各实验唯一的 REPORT.md 为准。

## 学习实验

- [P1 数值内核](../learning/p1_attention_numerics/REPORT.md)：数值对拍与基础算子；测试源码、历史验收记录。
- [P2 低精度量化](../learning/p2_quantization/REPORT.md)：outputs/ 保存 PPL 文本、图及历史误差摘录；后续导出机器数据。
- [P3 架构工具链](../learning/p3_arch_eval/REPORT.md)：outputs/ 保存 Roofline、SCALE-Sim、Timeloop 的 CSV 与图。
- [P4 RTL](../learning/p4_rtl/REPORT.md)：模块设计笔记与功能对拍；完整性限制见报告。
- [P5 Tile 模拟器](../learning/p5_tile_sim/REPORT.md)：outputs/ 保存逐点搜索数据、图与交叉检查记录。

## 研究实验

- [Codec 对照](../research/r1_kv_baseline/experiments/codec_compare/REPORT.md)：results/raw_metrics.csv、summary_mean_std.csv 和 run_config.json；合成配对对照。
- [KIVI 整模评估](../research/r1_kv_baseline/experiments/kivi_eval/REPORT.md)：results/table3/ 与 results/longbench/ 是合并结果入口；table3_kivi2/、table3_kivi4/ 是重跑来源，不重复计为独立实验。
- [Paged 布局](../research/r1_kv_baseline/experiments/paged_layout/REPORT.md)：results/raw_rows.json 与 summary.json，布局数值和字节对照。
- [KV 流量](../research/r1_kv_baseline/experiments/kv_pareto/REPORT.md)：results/summary.json，几何与名义字节记账。
- [WikiText PPL](../research/r1_kv_baseline/experiments/wikitext_ppl/REPORT.md)：results/ppl/L4096/、L8192/、L16384/、L32768/，以各目录 ppl_summary.json 为入口。results/ppl/ 根汇总为历史 L4096 副本，不重复统计。
- [M6 敏感性](../research/r1_kv_baseline/experiments/kv_sensitivity/REPORT.md)：results/synth/ 与 synth_b/ 为合成实验；layer/ 与 layer_8b/ 为整模实验；layer_160m/ 仅为通路检查，不混入主结果。

## 证据管理

原始结果保持不变，正式报告按核验后的结果撰写。研究结果中的重复 Markdown 数据表已移除，直接引用 JSON/CSV；学习阶段仍无等价原始数据的历史摘录予以保留。快照、冲突原件和机器核验清单保存在本地 .server-sync/，不为每次操作新增说明文档。

本地保存范围不等于 GitHub 发布范围，仍遵守 .gitignore。同步与文档约束统一见 [AGENTS.md](../AGENTS.md)。
