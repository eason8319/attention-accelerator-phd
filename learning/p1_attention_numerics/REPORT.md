# 实验报告：P1 Attention 数值内核

**实验日期**：2026-07-07；**整理日期**：2026-09-08。
**状态**：历史验收完成；本次未重跑。
**证据**：test_numerics.py、原验收记录；22 项通过为历史记录，本次未发现独立原始 pytest 日志。

## 1. 实验目的

检验标准、分块与 online softmax attention 是否保持相同数值语义，建立可供后续量化及 RTL 对拍使用的浮点参考。另检查 RoPE、RMSNorm 和单 token decode，避免基础算子误差传递到后续实验。

## 2. 方法与设置

以标准 scaled dot-product attention 为参考，对比两遍分块与单遍 online 实现。覆盖 causal/non-causal、不同块大小、FP16 online，以及 RoPE/RMSNorm 与 Transformers 的对照。FP32 最大绝对误差阈值为 $10^{-5}$。

历史环境为 p1-attention、PyTorch 2.12.1+cpu、Transformers 5.13.0。在本实验目录执行 `pytest test_numerics.py -q --tb=short`。

## 3. 实验结果

原验收记录为 **22 passed**：三种 attention 在阈值内一致；block_size=16/128 的 online 结果一致；RoPE/RMSNorm 与参考实现对齐；decode 与 prefill 最后一行对齐。逐项用例见 [test_numerics.py](test_numerics.py)。

## 4. 分析与讨论

分块计算的关键是维持全局归一化。online 实现通过 running max、running sum 和输出累加器，在块间重新缩放旧状态；块大小对照支持该实现的归约语义。推导见 [online_softmax_rescale_notes.md](online_softmax_rescale_notes.md)。

## 5. 局限与有效性

这是 CPU 数值验收，不包含 GPU 性能、整模任务精度或硬件综合。有限形状通过不能证明所有输入均等价。历史通过数未在本次重新确认。

## 6. 结论与后续工作

已有实现可作为学习阶段的浮点 golden model。后续应明确参考路径，在改动基础算子后重新运行对拍，不能沿用旧通过数。
