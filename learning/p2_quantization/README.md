# P2 — 低精度量化实验

详细步骤与验收标准见 [PLAN.md](PLAN.md)。依赖 P1 的 attention 实现。验收报告：[REPORT.md](REPORT.md)。

## 环境

```bash
conda create -n p2-quantization python=3.11 -y
conda activate p2-quantization
pip install -r ../../requirements.txt scipy tokenizers
```

## 目标

实现 INT4/FP8/MXFP4 的 fake-quant 工具库，复现 Hadamard/块对角旋转（BDR）抑制 outlier 的效果，
在小模型上完成 KV cache INT4 量化的困惑度评估。

## 文件结构

```
p2_quantization/
├── reading_notes.md        # 阅读材料精读笔记
├── fakequant.py            # INT4/INT8/FP8/MXFP4 quantize-dequantize
├── rotation.py             # 随机 Hadamard / 块对角 Hadamard（BDR, QuaRot/SAW 风格）
├── error_analysis.py       # 激活分布与量化误差统计（自动出图）
├── kv_cache_ppl.py         # 小模型 KV cache 量化困惑度评估
├── offline_utils.py        # 离线 tiny 模型与合成语料（无 HF 网络时使用）
├── test_fakequant.py       # 与 torch.float8_* 及手算样例对拍
└── outputs/                # 自动生成的报告与图表
```

## 从这里开始

0. 阅读 [reading_notes.md](reading_notes.md)（QuaRot / SAW-INT4 BDR / BitDecoding / MXFP4）。

## 运行

```bash
conda activate p2-quantization
pytest test_fakequant.py -q
python error_analysis.py
python kv_cache_ppl.py

# 联网时使用真实小模型：
python error_analysis.py --model Qwen/Qwen2.5-0.5B-Instruct
python kv_cache_ppl.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## 验收 checklist

- [x] fake-quant 库对拍测试通过
- [x] 复现旋转降低 INT4 量化误差的现象（数据 + 直方图）
- [x] KV cache INT4 + 旋转的困惑度退化 < 直接 INT4
- [x] 自动生成的误差分析报告
