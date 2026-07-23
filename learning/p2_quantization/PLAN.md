# P2 — 低精度量化实验（2 周）

**目标**：掌握 INT4/FP8/MXFP4 量化的数值行为，复现旋转抑制 outlier 的效果，为主线 2 混合精度 datapath 打基础。

## 步骤

1. **fake-quant 工具库**：实现 quantize-dequantize 函数：
  - INT4/INT8：per-tensor、per-channel、per-group（group size 32/64/128），对称与非对称（zero-point）
  - FP8：E4M3 与 E5M2 两种格式（可用 `torch.float8_e4m3fn` 对拍）
  - MXFP4：block size 32 的共享指数缩放（对照 OCP Microscaling 规范）
2. **误差分析**：从真实模型（如 Qwen2.5-0.5B）导出若干层的 Q/K/V 激活，统计逐通道分布，定位 outlier 通道；对量化前后的 attention score 与 attention 输出计算相对误差。
3. **旋转抑制 outlier**：实现随机 Hadamard 变换与块对角旋转（BDR，对齐 SAW-INT4），验证旋转后激活分布更接近高斯、量化误差下降；理解"旋转在数学上是恒等变换、在数值上改变量化友好性"。
4. **KV cache 量化端到端**：在小模型上把 KV cache 量化为 INT4（token-wise per-group），跑 WikiText-2 困惑度，对比 fp16 baseline / 直接 INT4 / 旋转+INT4 三组。
5. **混合精度配置实验**：尝试"Q/K INT4 + V FP8 + softmax 累加 fp32"等组合，绘制精度-比特数权衡曲线。

## 验收标准

- [x] fake-quant 库通过与 `torch.float8_*` 及手算样例的对拍测试
- [x] 复现"旋转显著降低 INT4 量化误差"的现象（误差降低量化数据 + 分布直方图）
- [x] 小模型 KV cache INT4 + 旋转的困惑度退化 < 直接 INT4
- [x] 输出一份误差分析报告（脚本自动生成图表）

## 阅读材料

- SAW-INT4 (arXiv 2604.19157) — 块对角旋转 BDR，本课题主线2直接对标
- QuaRot (2024) — Hadamard 旋转量化的代表作
- BitDecoding (HPCA 2026) — 低比特 KV cache 的系统实现
- OCP Microscaling Formats (MX) Specification — MXFP4 格式定义
