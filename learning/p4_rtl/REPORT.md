# 实验报告：P4 RTL 关键模块

**实验日期**：2026-07-22；**整理日期**：2026-09-09。
**状态**：历史 RTL 原始输出已于 2026-09-09 复核，三个模块对拍通过；未重新生成 DUT 输出。
**证据来源**：`results/rtl/vec_exp/`、`vec_softmax/`、`vec_sa/` 的输入、golden 和 DUT 文本；[check.log](results/rtl/check.log)、[run_config.json](results/rtl/run_config.json)。

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

对保留的原始输出重新执行 `scripts/compare_*.py`：exp 的 8186 项和 GEMM 的 32 项均无不一致，exp 最大相对误差为 $3.324\times10^{-4}$；softmax 的 running max 与归约和分别为 2574、150765425，均与定点参考一致。该复核验证已有输出，不代表一次新的 RTL 仿真。

## 4. 分析与讨论

比特一致性支持 RTL 与对应定点模型的一致性。跨块大小的差异来自定点舍入与近似 exp，不能把同一分块对拍通过扩大为任意分块比特等价。完整 attention 的输出累加器合并尚未包含。

## 5. 局限与有效性

未存档波形截图与 Verilator 行覆盖率；无完整 attention 输出累加通路，也无综合时序、面积或功耗。4×4 阵列是学习级实现，不能作为完整 FSA 微架构复现。

## 6. 结论与后续工作

模块具备历史功能证据。后续补波形和覆盖率、连接底边输出累加器，再验证完整 attention；映射讨论见 [notes/fsa_mapping.md](notes/fsa_mapping.md)。
