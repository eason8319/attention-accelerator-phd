# 实验结果与报告索引

本文件只维护实验入口与证据角色。每个实验使用 `experiment.json` 登记源码入口、依赖、`results/` 原始结果和唯一 `REPORT.md`；报告采用统一六节结构。当前进度见 [研究里程碑](progress/milestones.md)，数值与有效性限制以各报告为准。

## 学习实验

- [P1 数值内核](../learning/p1_attention_numerics/REPORT.md)：test_numerics.py；results/ 保存实际测试日志与 JUnit 记录。
- [P2 低精度量化](../learning/p2_quantization/REPORT.md)：kv_cache_ppl.py、error_analysis.py；results/ 保存 PPL 文本、图及历史误差摘录。
- [P3 架构工具链](../learning/p3_arch_eval/REPORT.md)：roofline.py、scale-sim/、timeloop/、collect_results.py；results/ 保存工具 CSV 与图。
- [P4 RTL](../learning/p4_rtl/REPORT.md)：Makefile、RTL、tb/ 与 scripts/；results/rtl/ 保存输入、golden、DUT 输出与对拍日志。
- [P5 Tile 模拟器](../learning/p5_tile_sim/REPORT.md)：run_p5.py、validate_vs_scalesim.py；results/ 保存逐点搜索数据、图与交叉检查记录。

## 研究实验

- [R1 总验收](../research/r1_kv_baseline/REPORT.md)：汇总下列实验的跨实验对照与 R2 验收依据；四窗口 Pareto 图及来源元数据位于 WikiText 实验 `results/ppl_traffic_pareto/`，各实验原始结果继续由各自清单登记。
- [Codec 对照](../research/r1_kv_baseline/experiments/codec_compare/REPORT.md)：results/raw_metrics.csv、summary_mean_std.csv 和 run_config.json；合成配对对照。
- [KIVI 整模评估](../research/r1_kv_baseline/experiments/kivi_eval/REPORT.md)：results/table3/ 与 results/longbench/ 是合并结果入口；分批参数与来源已完整并入 results/table3/run_config.json，不重复保留旧分批汇总。
- [Paged 布局](../research/r1_kv_baseline/experiments/paged_layout/REPORT.md)：results/raw_rows.json 与 summary.json，布局数值和字节对照。
- [KV 流量](../research/r1_kv_baseline/experiments/kv_pareto/REPORT.md)：results/summary.json，几何与名义字节记账。
- [WikiText PPL](../research/r1_kv_baseline/experiments/wikitext_ppl/REPORT.md)：run_wikitext_ppl.py、run_smoke.py；results/ppl/L4096/、L8192/、L16384/、L32768/ 下的 ppl_summary.json 为结果入口。与敏感性共用 r1_kv_baseline 主库；plot_pareto.py 读取既有 PPL 与 KV 流量生成本地关联图，不运行模型或生成报告正文。
- [敏感性实验](../research/r1_kv_baseline/experiments/kv_sensitivity/REPORT.md)：run_synth_grid.py、run_layer_ablation.py、run_smoke.py；模型适配、流量和指标依赖统一在 r1_kv_baseline 主库；当前有效结果为 results/metrics_f64/，必要验证证据保存在 results/archive/validation.tar.gz。
- [Attention 模拟器](../research/r1_decode_sim/experiments/decode_sweep/REPORT.md)：入口说明见 [r1_decode_sim/README.md](../research/r1_decode_sim/README.md)。实验内 `results/input_adapter/`、`core/`、`small_checks/` 保存前置检查；`pressure_capture/` 保存实际 cache 计数，`full_grid/` 保存完整网格和逐步压力仿真，`independent_os/` 保存 Roofline 参照、上游 SCALE-Sim CSV 与对照状态。缓存实现复用 `research/r1_kv_baseline/cache_path/`；依赖声明和独立工具需要的兼容依赖位于实验内。

## 证据管理


有效原始结果保持不变，正式报告按核验后的结果撰写。完整副本及已被汇总逐字段覆盖的分片不重复保留；只有仍未被有效且可复现结果取代的历史记录和必要验证证据才保存在 results/archive/；experiment.json 登记保留内容及淘汰依据。学习阶段仍无等价原始数据的历史摘录、RTL 的独立 golden/DUT 输出予以保留。快照、冲突原件和机器核验清单保存在本地 .server-sync/，不为每次操作新增说明文档。

本地保存范围不等于 GitHub 发布范围，仍遵守 .gitignore。同步与文档约束统一见 [AGENTS.md](../AGENTS.md)。
