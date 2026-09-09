# 实验报告：P1 Attention 数值内核

**实验日期**：2026-09-09；**整理日期**：2026-09-09。
**状态**：22 项数值检查已执行，21 项通过，1 项未达到阈值。
**证据来源**：[pytest.log](results/20260909_completeness/pytest.log)、[junit.xml](results/20260909_completeness/junit.xml)；源码与结果清单见 [experiment.json](experiment.json)。

## 1. 实验目的

检验标准、分块与 online softmax attention 是否保持相同数值语义，建立可供后续量化及 RTL 对拍使用的浮点参考。另检查 RoPE、RMSNorm 和单 token decode，避免基础算子误差传递到后续实验。

## 2. 方法与设置

以标准 scaled dot-product attention 为参考，对比两遍分块与单遍 online 实现。覆盖 causal/non-causal、不同块大小、FP16 online，以及 RoPE/RMSNorm 与 Transformers 的对照。FP32 最大绝对误差阈值为 $10^{-5}$。

本次在服务器 CPU 环境执行既有 `test_numerics.py`，使用 PyTorch 2.5.1+cu121 的 CPU 路径。在本实验目录执行 `python -m pytest test_numerics.py -q -p no:cacheprovider --junitxml=results/20260909_completeness/junit.xml`；原始日志与逐项检查状态完整保存。

## 3. 实验结果

本次结果为 **21 passed、1 failed**。FP32 attention 对拍、块大小对照、RoPE、RMSNorm 及 decode 检查通过。`test_online_fp16` 的最大绝对误差为 **0.001953125**，未满足严格小于 **0.001** 的阈值。逐项用例见 [test_numerics.py](test_numerics.py)，失败输入的参数和断言见原始日志。

## 4. 分析与讨论

分块计算的关键是维持全局归一化。online 实现通过 running max、running sum 和输出累加器，在块间重新缩放旧状态；块大小对照支持该实现的归约语义。推导见 [online_softmax_rescale_notes.md](online_softmax_rescale_notes.md)。

## 5. 局限与有效性

这是 CPU 数值验收，不包含 GPU 性能、整模任务精度或硬件综合。有限形状通过不能证明所有输入均等价。FP16 用例未通过，不能声称全部精度路径均达标；历史环境与当前环境不同，历史通过数不替代当前结果。

## 6. 结论与后续工作

已检查的 FP32 路径可作为学习阶段的浮点参考；FP16 online 路径需进一步分析误差及参考实现差异，达到既定阈值后才能纳入同等验收范围。
