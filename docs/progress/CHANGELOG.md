# 研究进展日志

按时间倒序记录（最新在上）。

---

## 2026-07-14

- 修复 P2 BDR：由 Gaussian-QR 块旋转改为 QuaRot/SAW 风格 `block_diag(H) @ D`，消除默认 seed 下 PPL 崩坏。
- 在 Qwen2.5-0.5B-Instruct 上更新验收数字：pytest 13/13；PPL fp16=1.68 / INT4=3.23 / Hadamard=2.02 / BDR=1.93；同步 `p2_quantization_report.md`。
- 误差分析对真实模型不再人工放大 outlier；自然激活下 K 误差与 attention 输出 L2 均随旋转下降。

---

## 2026-07-08

- 完成 P2 低精度量化实验验收：fake-quant 库（INT4/INT8/FP8/MXFP4）、Hadamard/BDR 旋转、误差分析与 KV cache 困惑度评估（pytest 11/11）。
- 新建 conda 环境 `p2-quantization`（Python 3.11, PyTorch 2.12.1）。
- 新增 `docs/progress/p2_quantization_report.md` 与 `experiments/p2_quantization/outputs/` 自动报告。

---

## 2026-07-07

- 完成 P1 Attention 数值内核复现验收：标准 / 分块 / online attention、RoPE、RMSNorm、decode-step 测试全部通过（pytest 22/22）。
- 新增 `docs/progress/p1_attention_numerics_report.md`，对照 `learning_plan.md` P1 checklist 记录产出、验证结果与后续衔接。
- 清理 P1 环境配置过程中产生的临时文件与 pytest 缓存。

---

## 2026-06-25

- 新建专用仓库 `attention-accelerator-phd`
- 云端仅保留论文下载脚本；PDF 改由本地 `download_papers.py` 下载
- 新增 `docs/progress/` 进展跟踪目录

---

<!-- 在此上方追加新条目 -->
