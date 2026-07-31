# 实验报告：真实 Cache-Path 上 C0–C5 编码对照

**日期**：2026-07-28  
**阶段**：R1 / M3 骨架（合并原 M1+M2，并纳入 KIVI 风格 C4/C5）  
**性质**：合成张量、真实 cache-path（非投影 fake-quant；非 PPL）  
**统计**：$n{=}20$ 配对种子；报告 mean ± 样本标准差  
**实验目录**：[`experiments/codec_compare/`](.)（本报告入库；`results/` 仅本地）  
**说明**：合并并取代原 `m1_codec_accuracy` / `m2_int4_bdr`（已删除）。

---

## 1. 实验目的与问题

在同一真实路径

$$
\mathrm{encode}\rightarrow\mathrm{store}\rightarrow\mathrm{load}\rightarrow\mathrm{decode}\rightarrow\mathrm{SDPA}
$$

上，对照协议谱 C0–C5，回答：

1. 各编码相对 float golden 的 attention / KV 重建误差量级？  
2. 精度—流量（`bytes_stored`）如何权衡？  
3. 通道 outlier 下，BDR 与 KIVI 是否相对均匀 INT4 更稳？  
4. KIVI 残差窗（$S\le$ `residual_length`）如何改变有效精度与字节？

---

## 2. 方法

### 2.1 被测格式

| ID | 入口 | Cache | 要点 |
|----|------|-------|------|
| C0 `fp16` | `"fp16"` | Contiguous | 精度下界；无 meta |
| C1 `int8` | `"int8"` | Contiguous | 对称 token-wise INT8 |
| C2 `int4` | `"int4"` | Contiguous | 对称、`group_size=32`；payload 按 0.5 B/元素记账 |
| C3 `int4_bdr` | `"int4_bdr"` | Contiguous | BDR→INT4；旋转矩阵不计入 meta |
| C4 `kivi2` | `"kivi2"` | **KiviKVCache** | K per-channel / V per-token 2-bit；`residual_length=128` FP16 窗 |
| C5 `kivi4` | `"kivi4"` | **KiviKVCache** | 同上 4-bit |

路径统一为 `AttentionWithCache`（[`cache_path/`](../../cache_path/)）。

### 2.2 Golden、指标与分布

- Golden：同一 Q/K/V 上未量化 float SDPA。  
- 指标：rel-$\ell_2$、cosine、max-abs；流量为 payload + metadata。  
- 配对：同一 `distribution` + `seed`（decode 步为 `seed*100003+t`）。  
- **gaussian**：i.i.d. $\mathcal{N}(0,I)$。  
- **outlier**：末维前 2 通道 ×20（BDR / 通道异质动机）。

### 2.3 负载

| 项 | 值 |
|----|-----|
| $B,H,D$ | $1,8,64$ |
| Prefill $S$ | 64 / 256 / 1024 |
| Decode | 256 step；记录 $t\in\{1,16,64,128,256\}$ |
| KIVI | `group_size=32`，`residual_length=128` |
| Seeds | $0..19$ |
| 设备 | CPU |

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline
python experiments/codec_compare/run_codec_compare.py
```

本地产物：`results/raw_metrics.csv`、`summary_mean_std.csv`、`run_config.json`、图（不同步云端）。

---

## 3. 结果

### 3.1 Prefill attention（$S{=}256$）

| 格式 | gaussian rel-$\ell_2$ | outlier rel-$\ell_2$ | total bytes |
|------|----------------------:|---------------------:|------------:|
| C0 FP16 | $3.04\times10^{-4}$ | $9.63\times10^{-4}$ | 524288 |
| C1 INT8 | $8.49\times10^{-3}$ | $6.16\times10^{-2}$ | 270336 |
| C2 INT4 | $1.39\times10^{-1}$ | $4.01\times10^{-1}$ | 147456 |
| C3 INT4+BDR | $1.39\times10^{-1}$ | $2.33\times10^{-1}$ | 147456 |
| C4 KIVI-2 | $5.08\times10^{-1}$ | $6.93\times10^{-1}$ | 204800 |
| C5 KIVI-4 | $9.51\times10^{-2}$ | $1.44\times10^{-1}$ | 253952 |

### 3.2 Decode attention（$t{=}256$）

与上表同序；量级一致（gaussian / outlier）：

| 格式 | gaussian | outlier | bytes |
|------|---------:|--------:|------:|
| C0 | $3.03\times10^{-4}$ | $9.57\times10^{-4}$ | 524288 |
| C1 | $8.40\times10^{-3}$ | $6.33\times10^{-2}$ | 270336 |
| C2 | $1.38\times10^{-1}$ | $3.94\times10^{-1}$ | 147456 |
| C3 | $1.37\times10^{-1}$ | $2.10\times10^{-1}$ | 147456 |
| C4 | $4.90\times10^{-1}$ | $6.71\times10^{-1}$ | 204800 |
| C5 | $9.26\times10^{-2}$ | $1.41\times10^{-1}$ | 253952 |

### 3.3 残差窗效应（$S{=}64 < 128$）

KIVI 尚未刷窗时，K/V 全在 FP16 残差中：C4/C5 的 prefill rel-$\ell_2$ **与 C0 数值相同**（$\sim3\times10^{-4}$ / $\sim7\times10^{-4}$），bytes 亦与 FP16 相同（131072）。说明短序列上 KIVI「看起来无损」来自残差窗，而非 2/4-bit 核本身。

### 3.4 图（本地 `results/`）

- `prefill_rel_l2_all.png`：全体格式 $S{=}256$ 柱状图  
- `decode_error_vs_step.png`：decode 误差曲线  
- `bytes_vs_error_pareto.png`：accuracy–bytes（越左下越好）

---

## 4. 编码优劣（本合成设定下）

| 编码 | 优势 | 劣势 / 适用边界 |
|------|------|----------------|
| **C0 FP16** | 误差地板；实现简单 | 流量最大（本设定下 decode $t{=}256$ 约 0.5 MiB/层形状） |
| **C1 INT8** | 误差仍低（$\sim10^{-2}$）；约半流量 | 压缩比不及 INT4；outlier 下误差升至 $\sim6\times10^{-2}$ |
| **C2 INT4** | **流量最省**（约 0.28× FP16）；实现简单 | gaussian 下 rel-$\ell_2\sim0.14$；**outlier 下崩到 $\sim0.4$** |
| **C3 INT4+BDR** | 流量与 INT4 相同；**outlier 下明显优于 INT4**（prefill 约 $0.40\to0.23$，decode 约 $0.39\to0.21$） | 各向同性高斯下与 INT4 几乎持平；需固定旋转 |
| **C4 KIVI-2** | 对齐算法前沿接口；短于残差窗时等同 FP16 | **2-bit 在本合成负载上误差最大**（$\sim0.5$–$0.7$）；刷窗后 bytes 仍高于 INT4（残差 FP16 + 更重 meta） |
| **C5 KIVI-4** | 刷窗后 **精度优于均匀 INT4**（gaussian $\sim0.095$ vs $0.14$；outlier $\sim0.14$ vs $0.40$），且在 outlier 上优于 BDR | bytes 高于 INT4/BDR（残差窗 + scale/mn）；短序列不省流量 |

**综合权衡（本实验）**：

1. **要极限省流量、通道较均质**：C2 INT4。  
2. **要省流量且有通道 outlier**：优先 C3 BDR（同流量、误差显著下降）。  
3. **要更低误差、可接受更高流量与残差窗语义**：C5 KIVI-4；C4 仅作协议锚点 / 极端压缩对照，本合成设定不推荐作精度主路径。  
4. **精度优先**：C0 / C1。  
5. KIVI 数字**不能**在 $S\le$ residual 时与「已量化」格式并表冒充低比特精度；报告须标注残差比例。

---

## 5. 局限

- 合成张量，非真实 LLM KV / 非 LM-Eval；KIVI 在真实模型上的优势可能被低估或高估。  
- 无 causal mask；无 paged 布局（M4）。  
- INT4 网格未 nibble-pack；KIVI 载荷未 bit-pack——bytes 为协议记账口径。  
- 残差窗超参固定为协议默认；未扫 `residual_length`。

---

## 6. 结论

- 已在同一真实 cache-path 上跑通 **C0–C5** 配对对照，并合并原 M1/M2 口径。  
- **BDR**：outlier 下相对 INT4 稳定增益，流量不变（复现 M2 结论）。  
- **KIVI-4**：刷窗后精度 Pareto 上优于均匀 INT4，代价是更高 bytes 与窗语义；**KIVI-2** 在本合成设定误差过大。  
- 下一步：阶段 B 在 Llama-2-7B / Mistral 上做 Table 3 / LongBench（计划 M3 评测段），勿用本合成表宣称 SOTA。
