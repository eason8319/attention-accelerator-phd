# P2 — 低精度量化实验

详细步骤与验收标准见 [docs/learning_plan.md](../../docs/learning_plan.md) 的 P2 一节。依赖 P1 的 attention 实现。

## 目标

实现 INT4/FP8/MXFP4 的 fake-quant 工具库，复现 Hadamard/块对角旋转（BDR）抑制 outlier 的效果，
在小模型上完成 KV cache INT4 量化的困惑度评估。

## 建议文件结构

```
p2_quantization/
├── fakequant.py            # INT4/INT8/FP8/MXFP4 quantize-dequantize
├── rotation.py             # 随机 Hadamard / 块对角旋转
├── error_analysis.py       # 激活分布与量化误差统计（自动出图）
├── kv_cache_ppl.py         # 小模型 KV cache 量化困惑度评估
└── test_fakequant.py       # 与 torch.float8_* 及手算样例对拍
```

## 从这里开始

第一步：在 `fakequant.py` 实现对称 per-tensor INT8 的 quantize-dequantize，
用手算的 4 元素向量写第一个单元测试，然后逐步扩展到 per-group INT4 与 FP8。

## 验收 checklist

- [ ] fake-quant 库对拍测试通过
- [ ] 复现旋转降低 INT4 量化误差的现象（数据 + 直方图）
- [ ] KV cache INT4 + 旋转的困惑度退化 < 直接 INT4
- [ ] 自动生成的误差分析报告
