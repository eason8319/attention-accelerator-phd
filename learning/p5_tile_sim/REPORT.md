# 实验报告：P5 Tile 级模拟器

**实验日期**：2026-07-22；**整理日期**：2026-09-08。
**状态**：历史实验完成；本次未重跑全部搜索。
**证据来源**：results/search_results.csv（包含全部负载搜索点）、results/cross_check_vs_scalesim_data.md；测试状态来自原验收记录。

## 1. 实验目的

检验粗粒度 tile 代价模型能否描述小 tile 的 DMA 暴露、大 tile 的 SRAM/双缓冲约束，并比较 prefill/decode 趋势是否与 P3 SCALE-Sim 一致。

## 2. 方法与设置

假设 32×32@1 GHz、16 MiB SRAM、1 TB/s，以 LLaMA-7B attention 几何建模。外层 Q tile、内层 KV tile；decode 固定单行 Q；仅在双份工作集可驻留 SRAM 时允许双缓冲。搜索同时最小化周期与 DRAM 字节。

对比 P3 WS 的 QK_T+PV 聚合结果，仅验相对趋势。入口为 run_p5.py、search.py、validate_vs_scalesim.py。历史环境为 Python 3.11、numpy 1.26.4、matplotlib 3.11.0、pytest 8.3.5。

## 3. 实验结果

原记录为 25 项测试通过、6/6 趋势检查通过，搜索结果和 Pareto 图已保存。历史交叉对照为：

| 指标 | SCALE-Sim | P5 |
|---|---:|---:|
| prefill/decode 利用率比 | 约 69.5 | 约 28.6 |
| prefill 流量 32K/4K | 约 64 | 约 60.7 |
| decode 流量 32K/4K | 8 | 8 |

比率来自历史摘录；原始搜索点见 results/search_results.csv。

## 4. 分析与讨论

两模型支持相同方向，但利用率比差异明显，趋势通过不能代替绝对校准。小 tile 的传输成本、大 tile 的重用收益与 SRAM 上限共同约束搜索空间；双缓冲合法性是解释拐点的重要条件。

## 5. 局限与有效性

模型无阵列 skew、bank conflict 和指令依赖停顿；softmax 仍为吞吐常数，未接入 P4 延迟。SCALE-Sim DRAM words、P5 DRAM bytes 和 DMA 周期比例不可混算。历史通过数本次未重新执行确认。

## 6. 结论与后续工作

当前模型适合趋势探索和 tile 筛选。后续以 RTL 延迟和更细存储约束校准，再决定是否用于定量预测；脚本只保存数据与检查状态，正式分析由本报告承担。
