# 实验报告：P4 RTL 关键模块

**实验日期**：2026-07-22；**整理日期**：2026-09-08。
**状态**：历史功能验收完成；本次未重跑 Verilator。
**证据**：原验收记录、notes/ 下的 exp_unit.md、softmax_unit.md、systolic_array.md 和 scripts/ 对拍程序。

## 1. 实验目的

验证 exp 近似、online softmax 归约和 INT8 systolic GEMM 的定点语义与流水接口，为完整 attention RTL 建立可对拍模块。

## 2. 方法与设置

exp 采用 16 段 PWL，Q6.10 输入、UQ0.24 输出、3 级流水，目标域相对误差小于 $10^{-3}$。softmax 检查 running max 和归约和；阵列为 4×4 WS INT8→INT32，与 numpy GEMM 对拍。

历史环境为 Verilator 5.020、Python 3.11、numpy 1.26.4、torch 2.5.1+cpu。从项目根目录进入 learning/p4_rtl，运行 `make check` 和 `make sim-all`。

## 3. 实验结果

| 模块 | 历史验收结果 |
|---|---|
| exp | 8186 个样本，LSB mismatch 为 0；最大相对误差约 $3.3\times10^{-4}$ |
| online softmax | 同一分块下与定点 golden 比特一致；跨块大小归约和相对偏差小于 1% |
| systolic GEMM | 32 个输出中 mismatch 为 0 |

这些数值来自历史记录，本次没有重新测得以上通过数。

## 4. 分析与讨论

比特一致性支持 RTL 与对应定点模型的一致性。跨块大小的差异来自定点舍入与近似 exp，不能把同一分块对拍通过扩大为任意分块比特等价。完整 attention 的输出累加器合并尚未包含。

## 5. 局限与有效性

未存档波形截图与 Verilator 行覆盖率；无完整 attention 输出累加通路，也无综合时序、面积或功耗。4×4 阵列是学习级实现，不能作为完整 FSA 微架构复现。

## 6. 结论与后续工作

模块具备历史功能证据。后续补波形和覆盖率、连接底边输出累加器，再验证完整 attention；映射讨论见 [notes/fsa_mapping.md](notes/fsa_mapping.md)。
