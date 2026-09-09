# 实验报告：P2 低精度量化与旋转

**实验日期**：2026-07-08 至 2026-07-14；**整理日期**：2026-09-08。
**状态**：历史实验完成；本次核对已有数据，未重跑模型。
**证据**：outputs/kv_cache_ppl.txt、outputs/error_analysis_data.md；测试通过数来自原验收记录。

## 1. 实验目的

检验旋转能否降低 INT4 量化误差，以及单层误差改善是否伴随端到端困惑度改善；同时建立低精度 fake-quant 和旋转工具，为真实 KV cache 实验提供参考。

## 2. 方法与设置

模型为 Qwen/Qwen2.5-0.5B-Instruct，使用自然 outlier，不人工放大。比较直接 INT4、Hadamard+INT4 与 block-Hadamard BDR+INT4。BDR 为块内 Hadamard 乘随机符号矩阵。

K 和 attention 输出以相对 L2 衡量；PPL 使用 k_proj/v_proj 输出 fake-quant 代理，并非真实 cache 读写量化。历史环境为 Python 3.11、PyTorch 2.12.1、Transformers 5.13.0；入口为 error_analysis.py、kv_cache_ppl.py、test_fakequant.py。

## 3. 实验结果

| 方法 | K 相对 L2 | attention 输出相对 L2 | PPL |
|---|---:|---:|---:|
| FP16 对照 | — | — | 1.6840 |
| 直接 INT4 | 0.1312 | 0.1207 | 3.2349 |
| Hadamard+INT4 | 0.0838 | 0.0836 | 2.0175 |
| BDR+INT4 | 0.0763 | 0.1053 | 1.9347 |

原记录为 13 项单元测试通过。本次核对了 PPL 文本和误差摘录，未将历史测试状态当作本次执行结果。

## 4. 分析与讨论

两种旋转均降低 K 误差与 PPL。BDR 的 K 误差和 PPL 更低，但 attention 输出误差高于 Hadamard，说明不同误差指标不能互相替代。PPL 仍高于 FP16，结论应限定为缓解退化。

## 5. 局限与有效性

投影输出代理不等同于真实 cache-path；模型与语料覆盖有限。混合精度图是示意扫参，不能作为独立系统级结论。结果文本未充分记录数据集版本和每次运行提交。历史自动摘录仅作数值证据，固定结论不作为本报告依据。

## 6. 结论与后续工作

本模型上的结果支持继续研究旋转辅助 INT4。后续采用真实 cache 路径，并保存数据集、种子和代码版本；正式研究结果以 R1 独立实验为准。
