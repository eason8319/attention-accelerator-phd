# M1 实验报告：编码方式对真实 Contiguous Cache-Path 精度的影响

**日期**：2026-07-27  
**阶段**：R1 / M1（C0 FP16、C1 INT8、C2 均匀 INT4）  
**性质**：合成张量、算法层中间量对拍（非 PPL / 非端到端任务）  
**实验目录**：[`experiments/m1_codec_accuracy/`](.)（本报告入库；`results/` 仅本地）

---

## 1. 实验目的与问题

M1 已实现真实 token-wise KV 路径：

$$
\mathrm{encode}\rightarrow\mathrm{store}\rightarrow\mathrm{load}\rightarrow\mathrm{decode}\rightarrow\mathrm{SDPA}
$$

本实验回答：

1. 在同一合成负载下，C0 / C1 / C2 相对**未量化 float KV golden**，attention 输出误差差多少？  
2. 误差主要来自 **KV 重建**还是 **attention 非线性放大**？  
3. 随 prefill 长度 / decode 步数增长，误差是否明显累积？  
4. 在协议口径下，精度退化对应多少 **bytes 节省**（payload + metadata）？

本报告指标对齐 [`protocols/metrics.md`](../../protocols/metrics.md) §2.1（cosine、相对 $\ell_2$），流量用 `bytes_stored()` 分项。

---

## 2. 方法

### 2.1 被测对象

| ID | 实现 | 要点 |
|----|------|------|
| C0 `fp16` | `FP16Codec` | 存 FP16，读回 float32；无 scale/zp |
| C1 `int8` | `Int8Codec`（对称） | token-wise 均匀 INT8；scale 为 FP16 |
| C2 `int4` | `Int4Codec`（对称，`group_size=32`） | 末维分组均匀 INT4；网格暂存 int8，`bytes_payload` 按 0.5 B/元素记账 |

路径统一为 `AttentionWithCache` + `ContiguousKVCache`（[`cache_path/`](../../cache_path/)），**不是**对 `k_proj`/`v_proj` 的一次性 fake-quant。

### 2.2 Golden 与指标

- **Golden**：同一随机 Q/K/V 上，直接对 **float32 K/V** 做 `scaled_dot_product_attention`（不经 cache）。  
- **被测**：经 cache encode/load/decode 后的 attention 输出。  
- **指标**（输出或 KV 展平后）：

$$
\mathrm{rel\text{-}\ell_2}
=\frac{\|y-\hat{y}\|_2}{\|y\|_2},\quad
\mathrm{cosine}
=\frac{\langle y,\hat{y}\rangle}{\|y\|_2\|\hat{y}\|_2},\quad
\mathrm{max\text{-}abs}=\|y-\hat{y}\|_\infty
$$

另测 **KV round-trip**：cache `append`+`load` 后的 $\hat{K},\hat{V}$ 相对原始 $K,V$（对 $K\|V$ 拼接度量）。

### 2.3 负载与统计

| 项 | 设定 |
|----|------|
| 形状 | $B{=}1$，$H{=}8$，$D{=}64$ |
| Prefill 长度 $S$ | 64 / 256 / 1024 |
| Decode | 连续 256 step；在 $t\in\{1,16,64,128,256\}$ 记录 |
| 随机种子 | $\{0,1,2,3,4\}$，报告 **mean ± std**（总体标准差） |
| 设备 | CPU，`torch 2.13.0+cpu` |
| Prefill mask | 当前实现 **无 causal**（与模块一致；本实验 golden 同样无 causal） |

脚本：[`run_m1_codec_accuracy.py`](run_m1_codec_accuracy.py)（写出到同目录 `results/`）

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline
python experiments/m1_codec_accuracy/run_m1_codec_accuracy.py
```

本地产物（不同步云端）：`results/raw_metrics.csv`、`results/summary_mean_std.csv`、`results/run_config.json`、图。

---

## 3. 实验结果

### 3.1 KV Round-trip（重建误差）

$S{=}256$，5 seeds 均值：

| 格式 | cosine | rel-$\ell_2$ | max-abs | total bytes |
|------|--------|--------------|---------|-------------|
| fp16 | $\approx 1.000$ | $2.08\times 10^{-4}$ | $1.84\times 10^{-3}$ | 524288 |
| int8 | $\approx 1.000$ | $5.94\times 10^{-3}$ | $1.83\times 10^{-2}$ | 270336 |
| int4 | $0.9953$ | $9.70\times 10^{-2}$ | $3.31\times 10^{-1}$ | 147456 |

相对 FP16 存储：INT8 总字节约 **0.52×**，INT4 约 **0.28×**（含 scale 元数据）。  
$S\in\{64,256,1024\}$ 时，各格式的 rel-$\ell_2$ **几乎不随长度变化**（INT4 稳定在 $\approx 0.097$），说明重建误差由量化网格主导，而非序列长度。

### 3.2 Prefill Attention

$S{=}256$：

| 格式 | cosine | rel-$\ell_2$ | max-abs | total bytes |
|------|--------|--------------|---------|-------------|
| fp16 | $\approx 1.000$ | $3.04\times 10^{-4}$ | $2.95\times 10^{-4}$ | 524288 |
| int8 | $0.999965$ | $8.52\times 10^{-3}$ | $7.45\times 10^{-3}$ | 270336 |
| int4 | $0.9905$ | $1.39\times 10^{-1}$ | $1.08\times 10^{-1}$ | 147456 |

相对 FP16 attention 输出误差量级：

- INT8 ≈ **28×** FP16 的 rel-$\ell_2$  
- INT4 ≈ **460×** FP16，约 **16×** INT8  

图（本地）`results/prefill_rel_l2_bars.png`：三档误差随 $S$ 几乎平坦，与 round-trip 结论一致。

对比 §3.1：同一格式下，**attention 输出 rel-$\ell_2$ 略高于 KV 重建**（INT4：$0.097\to 0.139$），符合 Softmax/加权和对量化噪声的轻度放大，但未出现数量级爆炸。

### 3.3 Decode Attention（逐步）

$t{=}256$：

| 格式 | cosine | rel-$\ell_2$ | max-abs | total bytes |
|------|--------|--------------|---------|-------------|
| fp16 | $\approx 1.000$ | $3.17\times 10^{-4}$ | $1.23\times 10^{-4}$ | 524288 |
| int8 | $0.999964$ | $8.43\times 10^{-3}$ | $2.74\times 10^{-3}$ | 270336 |
| int4 | $0.9906$ | $1.40\times 10^{-1}$ | $4.76\times 10^{-2}$ | 147456 |

与同长度 prefill（$S{=}256$）同量级。图（本地）`results/decode_error_vs_step.png` 显示：

- 前 $\sim$16–64 step 内 rel-$\ell_2$ / $(1-\mathrm{cosine})$ 略升后 **平台化**；  
- **未观察到**随 decode 步数近似线性发散的误差雪崩（在本合成设定、$t\le 256$ 内）；  
- FP16 的 $(1-\mathrm{cosine})$ 已接近浮点噪声（$10^{-8}$–$10^{-7}$）。

INT4 在 $t{=}1$ 时 max-abs 较大（$\approx 0.25$），随后随上下文变长、softmax 平滑，**逐元素峰值略降**，但相对范数误差仍平台在 $\sim 0.13$–$0.14$。

### 3.4 精度—流量对照（$S$ 或 $t{=}256$）

| 格式 | Attn cosine | Attn rel-$\ell_2$ | total / FP16 | 相对 FP16 节省 |
|------|-------------|-------------------|--------------|----------------|
| fp16 | $\approx 1$ | $\sim 3\times 10^{-4}$ | 1.00 | — |
| int8 | $>0.99996$ | $\sim 8.5\times 10^{-3}$ | 0.52 | $\approx 48\%$ |
| int4 | $\approx 0.9906$ | $\sim 1.4\times 10^{-1}$ | 0.28 | $\approx 72\%$ |

INT4 的 metadata（FP16 scale，每 token×head×group×K/V）在 $S{=}256$ 约占总量 $16384/147456\approx 11\%$；更长上下文时元数据占比结构相同（随 token 线性）。

---

## 4. 结论

1. **编码对真实 cache-path 精度有清晰分层**：FP16 接近机器噪声；对称 INT8 在合成 SDPA 上仍保持 cosine $>0.99996$、rel-$\ell_2\sim 10^{-2}$；均匀 INT4（group 32）cosine 降至 $\approx 0.99$，rel-$\ell_2\sim 0.14$，属显著但方向仍大体可辨的退化。  
2. **Attention 误差由 KV 量化主导**，Softmax 路径仅轻度放大；无证据表明本路径把 INT4 误差放大到不可控。  
3. **在 $S,t\le 1024/256$ 的合成设定下，误差不随长度剧烈累积**，而以比特宽度决定的平台误差为主——与「每步独立量化写入」的机制一致（每 token 网格误差近似同分布）。  
4. **流量收益与精度代价同向**：INT8 约半带宽、弱精度损失；INT4 约七成节省、精度损失约一个数量级相对 INT8——**M2（BDR）与后续非均匀/非对称方案的主要动机**。  
5. **本实验边界**（须在后续工作中补齐，不得外推为任务 SOTA）：  
   - 合成高斯 QKV，非真实激活分布；  
   - 无 RoPE / 无真实 Transformer 层堆叠；  
   - Prefill 无 causal；  
   - INT4 未 nibble-pack（记账按协议 0.5 B，物理存储仍为 int8 网格）；  
   - 未测 PPL / LM-Eval（阶段 B）。

**对 M1 的判定**：C0–C2 真实 cache-path 行为符合预期——可区分、可复现、误差—字节关系合理；INT4 精度缺口明确，适合作为 M2 改进基线，而非最终部署点。

---

## 5. 附录

### 5.1 目录约定

| 路径 | 云端 | 内容 |
|------|------|------|
| `REPORT.md` | 同步 | 本报告 |
| `run_m1_codec_accuracy.py` | 不同步（可本地保留） | 复现脚本 |
| `results/` | 不同步 | CSV / 图 / `run_config.json` |

### 5.2 关键数值摘录（decode，$t{=}256$，mean）

见本地 `results/summary_mean_std.csv` 中 `scenario=decode_attn, seq_or_step=256`；正文表格已四舍五入到报告精度。
