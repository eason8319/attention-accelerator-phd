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

- [R2 资源与最小算术](../research/r2_streaming_attention/experiments/arithmetic_feasibility/REPORT.md)：`run_resource_probe.py`、`run_model_runtime.py`、`run_token_count.py`、`run_arithmetic_synthesis.py`、`run_budget_analysis.py`；`results/` 保存资源快照、原生 FP16 短测、完整语料计数、算术对拍/单元时序和运行预算。PPL 算量是 v1 全程逐 token 前缀情景，不是现行 `r2-evaluation-v2` 质量门成本。公开内核版本与依赖在 `runtime/` 登记；完成范围和限制以报告为准，步骤状态见[里程碑](progress/milestones.md#r2-review)。
- [R2 物理打包与追加](../research/r2_streaming_attention/experiments/physical_pack_append/REPORT.md)：`run_pack_consistency.py`；共享实现位于 `research/r2_streaming_attention/` 的 `physical_pack.py` / `packed_kv.py`；`results/pack_storage_verified/` 保存完整一致性格子、缓存存储、分侧布局预算及输入所有权检查；必要负例见清单登记的 `results/archive/`。布局见 [`pack_layout.json`](../research/r2_streaming_attention/experiments/configs/pack_layout.json)。不覆盖分页、残差刷窗或质量评测。
- [R2 分页、残差与访存事件](../research/r2_streaming_attention/experiments/paged_residual_access/REPORT.md)：`run_page_access.py`；共享实现位于 `research/r2_streaming_attention/` 的 `paged_kv.py` / `access_events.py`；`results/page_access_accounted/` 保存双布局格子、刷窗轨迹、元数据放置、输入所有权及独立计量/故障注入检查；历史缺陷证据与所用源码保存在本实验 `results/archive/`，路径和哈希见实验清单。分页参数见 [`page_access.json`](../research/r2_streaming_attention/experiments/configs/page_access.json)。不覆盖流式 Attention、周期、能耗或质量评测。
- [R2 单头操作数流式 Attention 与 C3 混合功能基线](../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)：主入口 `run_stream_attention.py`；冒烟入口 `smoke/run_operand_acceptance.py`、`smoke/run_c3_qo_association.py`（纯 Q/O 延期诊断）、`smoke/run_c3_score_alignment.py`（混合输出验收）。现行 DUT 证据在 `results/hybrid_baseline_stream/`、`results/hybrid_baseline_contracts/`；C3 混合 305 项在 `results/c3_acceptance/`；纯 Q/O 负结果在 `results/c3_qo_fp32_association/`。历史保护网格保留在 `results/guarded_operand_stream/` 与 `results/operand_acceptance/`。参数见 [`stream_attention.json`](../research/r2_streaming_attention/experiments/configs/stream_attention.json)。混合路径保留 K 逆旋转，不是纯 Q/O 优化结论，以唯一报告为准。
- [R2 公平映射与周期、能耗基线](../research/r2_streaming_attention/experiments/fair_mapping_cost/REPORT.md)：主入口 `run_mapping_cost.py`；冒烟 `smoke/run_mapping_invariants.py`、`smoke/run_cost_contracts.py`。现行证据在 `results/mapping_cost_accounted/`。参数见 [`mapping_cost.json`](../research/r2_streaming_attention/experiments/configs/mapping_cost.json)。周期与动态能量是声明模型，不是综合频率或硅片实测；C3 混合路径不是纯 Q/O。独立组合未打开。

## 证据管理

有效原始结果保持不变，正式报告按核验后的结果撰写。完整副本及已被汇总逐字段覆盖的分片不重复保留；只有仍未被有效且可复现结果取代的历史记录和必要验证证据才保存在 results/archive/；experiment.json 登记保留内容及淘汰依据。学习阶段仍无等价原始数据的历史摘录、RTL 的独立 golden/DUT 输出予以保留。快照、冲突原件和机器核验清单保存在本地 .server-sync/，不为每次操作新增说明文档。

本地保存范围不等于 GitHub 发布范围，仍遵守 .gitignore。同步与文档约束统一见 [AGENTS.md](../AGENTS.md)。
