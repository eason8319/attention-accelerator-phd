# P2 低精度量化实验验收报告

日期：2026-07-14（更新；初版 2026-07-08）

## 结论

P2 已完成验收。实现 INT4/INT8/FP8/MXFP4 fake-quant 库、随机 Hadamard 与 block-Hadamard BDR、误差分析与 KV cache 量化困惑度评估；在 **Qwen/Qwen2.5-0.5B-Instruct** 上复现「旋转降低 INT4 误差 / 困惑度」现象。单元测试 **13/13** 通过。

验证命令：

```bash
# 单元测试
conda run -n p2-quantization pytest experiments/p2_quantization/test_fakequant.py -q --tb=short

# 误差分析（自动出图 + Markdown 报告）
conda run -n p2-quantization python experiments/p2_quantization/error_analysis.py \
  --model Qwen/Qwen2.5-0.5B-Instruct

# KV cache INT4 困惑度对比
conda run -n p2-quantization python experiments/p2_quantization/kv_cache_ppl.py \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

验证结果（Qwen2.5-0.5B-Instruct）：

```text
pytest: 13 passed

K INT4 相对 L2（自然 outlier，无人工放大）:
  direct=0.1312, Hadamard=0.0838 (−36.1%), BDR=0.0763 (−41.8%)
Attention 输出相对 L2:
  direct=0.1207, Hadamard=0.0836, BDR=0.1053  # 均优于直接 INT4

KV PPL:
  fp16=1.6840
  int4=3.2349
  int4_hadamard=2.0175   # < int4 ✓
  int4_bdr=1.9347        # < int4 ✓
```

BDR 实现为 QuaRot/SAW 风格 `R = block_diag(H,…,H) @ D`（块内 Walsh–Hadamard + 全维随机对角符号）。早期 Gaussian-QR 块旋转对 seed 极敏感（默认 seed=0 时 PPL 可崩到 ~20+），已替换。

> 无网络时可用默认 `offline-tiny-llama` 跑通流程；验收数字以 Qwen 结果为准。

## 验收 Checklist

| learning_plan.md 验收项 | 对应产出 | 状态 |
|---|---|---|
| fake-quant 库通过与 torch.float8_* 及手算样例的对拍测试 | `fakequant.py`, `test_fakequant.py` | 通过 (13/13) |
| 复现旋转显著降低 INT4 量化误差 | `error_analysis.py`, `outputs/error_analysis.png` | 通过（K 误差 Hadamard −36% / BDR −42%；输出 L2 同步下降） |
| KV cache INT4 + 旋转困惑度退化 < 直接 INT4 | `kv_cache_ppl.py`, `outputs/kv_cache_ppl.txt` | 通过（Hadamard 与 BDR 均优于直接 INT4） |
| 输出误差分析报告（脚本自动生成图表） | `outputs/error_analysis_report.md` | 完成 |

## 产出说明

- `fakequant.py`：INT4/INT8（per-tensor/channel/group，对称/非对称）、FP8 E4M3/E5M2、MXFP4 block-32 fake-quant。
- `rotation.py`：随机 Hadamard；块对角 Walsh–Hadamard BDR（`R = block_diag(H) @ D`）。
- `error_analysis.py`：导出 Q/K/V、outlier 分布、量化误差、混合精度权衡曲线；真实模型不做人工 outlier 放大。
- `kv_cache_ppl.py`：包装 `k_proj`/`v_proj` 输出做 INT4 fake-quant，近似 KV cache 量化，对比 fp16 / INT4 / 旋转+INT4。
- `offline_utils.py`：离线 tiny Llama 与合成语料。
- `test_fakequant.py`：手算 INT8、FP8 对拍、MXFP4、Hadamard/BDR 正交与块结构测试。

## 已知局限（不影响学习验收）

- KV 量化是对 `k_proj`/`v_proj` 输出的代理，并非 decode 时对 cache 条目做 token-wise 存取量化。
- 混合精度权衡曲线为示意扫参，非独立系统级实验。
- 单层输出 L2 与端到端 PPL 在人工放大 outlier 时可能不完全同向；真实激活下二者一致改善。以 PPL 为端到端主指标。

## 环境记录

- Conda 环境：`p2-quantization`（Python 3.11）
- PyTorch：`2.12.1`
- Transformers：`5.13.0`

创建环境：

```bash
conda create -n p2-quantization python=3.11 -y
conda activate p2-quantization
pip install -r requirements.txt scipy tokenizers
```

## 后续衔接

- 阶段 3 混合精度 datapath 可直接复用 fake-quant 库与旋转模块。
- 与 P1 attention golden model 联调时可量化 Q/K/V 输入，评估 online softmax 累加误差。
- 若需论文级 KV 实验：改为 cache 写入路径量化，并扩展混合精度扫参。
