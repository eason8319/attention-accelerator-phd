# M2 实验报告：真实 Cache-Path 上 INT4+BDR 相对均匀 INT4 的误差改善

**日期**：2026-07-27  
**阶段**：R1 / M2（C2 均匀 INT4 vs C3 INT4+BDR）  
**性质**：合成张量、真实 contiguous cache-path（非投影 fake-quant；非 PPL）  
**统计**：$n{=}20$ 配对种子；报告 mean±样本标准差及配对 win-rate  
**实验目录**：[`experiments/m2_int4_bdr/`](.)（`REPORT.md` 入库；`results/` 仅本地）

---

## 1. 实验目的与问题

M2 已将 BDR（块对角 Walsh–Hadamard + 随机符号对角）接入真实写/读路径：

$$
\mathrm{encode}:\; x\mapsto \mathrm{INT4}(xR),\qquad
\mathrm{decode}:\; \hat{x}\mapsto \mathrm{dequant}(\cdot)\,R^{\top}
$$

本实验在 **同一 cache-path** 上回答：

1. 相对 float golden，**INT4+BDR 是否相对均匀 INT4 降低误差**？  
2. 改善是否依赖激活分布（各向同性高斯 vs 通道 outlier）？  
3. 流量（bytes）是否与均匀 INT4 同口径（旋转矩阵不计入 metadata）？

验收口径（计划）：方向性结论「BDR 优于直接 INT4」在合理设定下成立；不要求追平历史 P2 proxy 数值。

---

## 2. 方法

### 2.1 被测路径

统一使用 `AttentionWithCache` + `ContiguousKVCache`：

| 格式 | 入口 | 说明 |
|------|------|------|
| FP16 | `"fp16"` | 参考下界 |
| INT4 | `"int4"` | 对称、`group_size=32` |
| INT4+BDR | `"int4_bdr"` / `seed=0` | `block_size=32`，旋转维=`head_dim` |

旋转矩阵 $R\in\mathbb{R}^{d\times d}$ 固定、与序列长度无关；**不计入** `bytes_metadata`（与 SAW-INT4「不把 $R$ 当随 $N$ 增长的 KV 流量」一致）。scale 元数据与 INT4 相同。

### 2.2 Golden 与指标

- Golden：同一 Q/K/V 上未量化 float SDPA。  
- 指标：attention / KV 重建的 rel-$\ell_2$、cosine、max-abs。  
- **配对**：同一 `distribution` + data `seed`（decode 为 `seed*100003+t`）下 INT4 与 BDR 共用同一组张量。  
- 改善定义（正值 = BDR 更优）：

$$
\Delta_{\mathrm{abs}}=\mathrm{rel\text{-}\ell_2}^{(\mathrm{INT4})}-\mathrm{rel\text{-}\ell_2}^{(\mathrm{BDR})},\quad
\Delta_{\mathrm{rel}}=\Delta_{\mathrm{abs}}/\mathrm{rel\text{-}\ell_2}^{(\mathrm{INT4})}
$$

win-rate：$\mathrm{rel\text{-}\ell_2}^{(\mathrm{BDR})}<\mathrm{rel\text{-}\ell_2}^{(\mathrm{INT4})}$ 的种子比例。

### 2.3 两种合成分布

| 名称 | 定义 | 意图 |
|------|------|------|
| `gaussian` | $Q,K,V\sim\mathcal{N}(0,I)$ | 与 M1 同；通道同质，BDR 动机弱 |
| `outlier` | 先高斯，再将末维前 2 个通道 ×20 | 模拟通道异质 / outlier（SAW/QuaRot 动机） |

### 2.4 负载

| 项 | 值 |
|----|-----|
| $B,H,D$ | $1,8,64$ |
| Prefill $S$ | 64 / 256 / 1024 |
| Decode | 256 step；记录 $t\in\{1,16,64,128,256\}$ |
| Seeds | $0..19$（$n{=}20$） |
| 设备 | CPU，`torch 2.13.0+cpu` |

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline
python experiments/m2_int4_bdr/run_m2_int4_bdr.py
```

---

## 3. 结果

### 3.1 流量（$S$ 或 $t{=}256$）

| 格式 | total bytes | 相对 FP16 |
|------|-------------|-----------|
| fp16 | 524288 | 1.00× |
| int4 | 147456 | 0.28× |
| int4_bdr | 147456 | 0.28×（与 INT4 **相同**） |

结论：BDR **不增加** 协议口径下的 KV HBM 记账（相对均匀 INT4）。

### 3.2 KV Round-trip（$S{=}256$）

| 分布 | INT4 rel-$\ell_2$ | BDR rel-$\ell_2$ | 配对 $\Delta_{\mathrm{rel}}$ | win-rate |
|------|-------------------|------------------|------------------------------|----------|
| gaussian | $0.0970\pm 0.0002$ | $0.0970\pm 0.0002$ | $\approx 0\%$ | 50% |
| outlier | $0.150\pm 0.001$ | $0.0657\pm 0.0001$ | **$+56.1\%$** | **100%** |

高斯下旋转几乎不改重建误差；outlier 下 BDR 显著压低量化误差（通道能量被块 Hadamard 摊开）。

### 3.3 Prefill Attention（$S{=}256$）

| 分布 | FP16 | INT4 | INT4+BDR | $\Delta_{\mathrm{rel}}$ | win-rate |
|------|------|------|----------|-------------------------|----------|
| gaussian | $3.04\times 10^{-4}$ | $0.139\pm 0.002$ | $0.139\pm 0.003$ | $+0.3\%\pm 1.1\%$ | 50% |
| outlier | $9.6\times 10^{-4}$ | $0.401\pm 0.024$ | $0.233\pm 0.017$ | **$+41.8\%\pm 3.1\%$** | **100%** |

outlier 下 cosine：INT4 $0.917$ → BDR $0.973$（更接近 1）。

图（本地）：`results/prefill_int4_vs_bdr.png`。

### 3.4 Decode Attention（$t{=}256$）

| 分布 | INT4 | INT4+BDR | $\Delta_{\mathrm{rel}}$ | win-rate |
|------|------|----------|-------------------------|----------|
| gaussian | $0.138\pm 0.009$ | $0.137\pm 0.016$ | $+0.7\%\pm 13\%$ | 70% |
| outlier | $0.394\pm 0.100$ | $0.210\pm 0.063$ | **$+44.9\%\pm 17\%$** | **100%** |

图（本地）：`results/decode_int4_vs_bdr.png`。误差随步数仍呈平台化；outlier 设定下 BDR 曲线整体低于 INT4。

### 3.5 与「BDR 应优于 INT4」验收的关系

| 设定 | 是否观察到稳定改善 |
|------|--------------------|
| 各向同性高斯（M1 同款） | **否**（配对差 ≈ 噪声；win-rate ~50%） |
| 通道 outlier（BDR 动机设定） | **是**（KV 重建 ~56%、prefill ~42%、decode ~45% 相对误差下降；win-rate 100%） |

这与 SAW-INT4 / QuaRot 的物理图像一致：BDR 主要缓解 **通道幅度异质** 导致的量化网格浪费，而非在已近各向同性的高斯上再「无中生有」地降噪。

---

## 4. 结论

1. **在真实 contiguous cache-path 上，INT4+BDR 在通道 outlier 合成设定下相对均匀 INT4 显著降误差**，且 **bytes 与 INT4 相同**——满足 M2「方向性优于 INT4」的验收（在合理动机设定下）。  
2. **各向同性高斯上 BDR 无明显优势**（与 M1 基线同量级）；报告不得把高斯上的持平误写成失败，也不应只报 outlier 而隐瞒高斯。  
3. 改善同时出现在 **KV 重建** 与 **attention 输出**，说明收益来自量化本身，而非偶然的 Softmax 抵消。  
4. **边界**：合成分布 ≠ 真实 LLM KV；未测 PPL/任务；软件路径为 rotate→quant→dequant→inverse 后再 SDPA（非 SAW 的 fused 旋转域 attention）；prefill 无 causal。

**M2 判定（本实验）**：C3 已接入并在 outlier 压力下复现「BDR 优于均匀 INT4」；可进入后续对照谱 / 阶段 B 评测。

---

## 5. 附录

| 路径 | 云端 | 内容 |
|------|------|------|
| `REPORT.md` | 同步 | 本报告 |
| `run_m2_int4_bdr.py` | 不同步 | 复现脚本 |
| `results/` | 不同步 | CSV / 图 / `run_config.json` |

关键表：`summary_mean_std.csv`、`paired_summary.csv`、`paired_int4_vs_bdr.csv`。
