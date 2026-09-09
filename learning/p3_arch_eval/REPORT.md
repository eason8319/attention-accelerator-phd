# 实验报告：P3 架构评估工具链

**实验日期**：2026-07-15；**整理日期**：2026-09-08。
**状态**：历史实验完成；本次未重跑仿真。
**证据**：outputs/ 下的 roofline_table.csv、scalesim_results.csv、timeloop_energy.csv、timeloop_area.csv、cross_joined.csv；历史摘录 cross_validation_data.md。

## 1. 实验目的

比较 attention prefill 与 decode 的算术强度、阵列利用率、流量与能量，检验三种工具的相对趋势是否一致，并辨明不同模型之间可比较的范围。

## 2. 方法与设置

使用 LLaMA-7B 规模的单层 attention 几何，覆盖 4K/32K/128K。SCALE-Sim 采用 32×32 WS/OS 阵列，将 GEMM 分成不超过 256 的 tile 后重复；Timeloop/Accelergy 在对应 tile 上估计能量和面积；Roofline 使用独立解析峰值。

历史环境为 Python 3.11、SCALE-Sim 3.0.0、numpy 1.26.4，以及 Timeloop/Accelergy Docker。入口为 roofline.py、scale-sim/run_scalesim.py、timeloop/run_timeloop.py；collect_results.py 合并数据与绘图。

## 3. 实验结果

已有 SCALE-Sim CSV 为 48 行，Timeloop 能量 CSV 为 24 行。历史代表性 WS QK_T 利用率为 prefill 约 73.15%、decode 约 1.05%；算术强度约为 248 和 50.9 ops/byte。

历史 decode 动态能量汇总为 SRAM 约 89%、DRAM 约 11%，限于所用 PAT。逐项证据见上述 CSV，详细讨论见 [analysis.md](analysis.md)。

## 4. 分析与讨论

瘦矩阵映射使 decode 利用率明显下降，与算术强度降低方向一致。流量比例、能量比例和周期比例含义不同；SCALE-Sim 的片外访问占比不能代替 Timeloop 的片外能量占比。

## 5. 局限与有效性

固定 tile 重复忽略跨 tile 复用和重叠；PAT 未按目标工艺校准。解析 128 TOPS 与 32×32@1 GHz 约 2 TOPS 不在同一尺度，不能直接对齐绝对吞吐。历史 Docker 标签可变，复现仍需锁定镜像摘要。

## 6. 结论与后续工作

工具链可比较趋势、识别假设差异，但不足以支持未校准的芯片绝对能效。P5 继续用相对比例对照；正式能耗主张前须校准存储模型。
