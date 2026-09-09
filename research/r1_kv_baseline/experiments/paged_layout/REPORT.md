# 实验报告：contiguous / paged 双布局（阶段 A）

**实验日期**：2026-09-04；**整理日期**：2026-09-09。
**状态**：本报告所列批次已完成；以保留的原始结果为依据。
**证据来源**：results/raw_rows.json、results/summary.json。

**阶段**：R1 / M4 阶段 A  
**性质**：合成张量、真实 cache-path；对照 **布局** 而非 codec 精度（精度谱系见 [`../codec_compare/REPORT.md`](../codec_compare/REPORT.md)）  
**协议**：[`protocols/metrics.md`](../../protocols/metrics.md) §8（v1.1，$P_{\mathrm{size}}=16$，$B_{\mathrm{pte}}=8\,\mathrm{B}$）  
**实验目录**：[`experiments/paged_layout/`](.)（本报告入库；`results/` 与 `run_*.py` 仅本地）

---

## 1. 实验目的

在同一 encode→store→load→SDPA 路径上，为 **两条 cache 后端**（均匀 C0–C3 与 `KiviKVCache` C4/C5）同时给出 contiguous 与 paged 列，回答：

1. paged 相对 contiguous，attention / KV 重建是否对齐到实现容差？  
2. $B_{\mathrm{payload}}$ / $B_{\mathrm{scale}}$ / $B_{\mathrm{zp}}$ 是否按占用 token 与 contiguous 一致？$B_{\mathrm{page}}$ 是否按 §8.5 计页表？  
3. C4/C5 四池（量化历史 vs FP16 残差）页数是否与 §8.4 公式一致？尾页、跨页 decode、残差窗满 $R{=}128$ 是否可跑？

本实验 **不** 报 PPL / 任务分，**不** 做 8B Pareto（M5）。主声称不得只引用 contiguous。

---

## 2. 方法与设置

### 2.1 被测格式与布局

| ID | 入口 | contiguous | paged |
|----|------|------------|-------|
| C0 `fp16` | `"fp16"` | `ContiguousKVCache` | `PagedUniformKVCache` |
| C1 `int8` | `"int8"` | 同上 | 同上 |
| C2 `int4` | `"int4"` | 同上 | 同上 |
| C3 `int4_bdr` | `"int4_bdr"` | 同上 | 同上 |
| C4 `kivi2` | `"kivi2"` | `KiviKVCache` | `PagedKiviKVCache` |
| C5 `kivi4` | `"kivi4"` | 同上 | 同上 |

入口统一 `AttentionWithCache(..., layout=...)`。KIVI：`group_size=32`，`residual_length=128`。

### 2.2 指标与负载

- 对照：同一 Q/K/V 上 paged vs contiguous 的 attention 输出与 `load()` 的 K/V（max-abs、rel-$\ell_2$）。prefill 与逐步 decode **分列**汇总，避免 C3 单行舍入主导总表。  
- 门禁：C0–C2 / C4–C5 逐元素；C3 prefill $\le 10^{-5}$（float32 旋转）；C3 decode 的 attention $\le 10^{-5}$，K/V 允许 INT4 差 1 档（$5\times10^{-3}$）。  
- 流量：`bytes_breakdown()` 四项；contiguous 的 $B_{\mathrm{page}}=0$。decode 与 prefill 的四项 / 页数须一致。  
- 页数：paged 后端 `page_counts()`，对照 §8 公式。

| 项 | 值 |
|----|-----|
| $B,H,D$ | $1,8,64$ |
| $N$ | 16 / 17 / 64 / 128 / 129 / 256 |
| 模式 | prefill 整段写入；decode 逐步 $n$ 步（跨页、刷窗） |
| Seeds | $0..4$（配对同一张量） |
| 设备 | CPU |
| 模型 | `offline`（合成）；未跑 0.5B 冒烟 |

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline
python experiments/paged_layout/run_paged_layout.py
```

本地产物：`results/summary.json`、`raw_rows.json`（不同步云端）。共 360 条（6 格式 × 6 长度 × 5 种子 × 2 模式）。

---

## 3. 实验结果

### 3.1 布局对齐（paged − contiguous）

C0–C2 与 C4/C5 在全部 $N$、种子与两种模式下 **逐元素一致**（含 $N{=}17$ 尾页、逐步 decode、KIVI 在 $N{=}128$ 刷窗与 $N{=}129$ 溢出）。C3 按下表 **分列**（各 30 行）。

**Prefill**（整段写入；两侧 append 粒度相同）：

| 格式 | attention | $K$ | $V$ |
|------|----------:|----:|----:|
| C0 FP16 | $0$ | $0$ | $0$ |
| C1 INT8 | $0$ | $0$ | $0$ |
| C2 INT4 | $0$ | $0$ | $0$ |
| C3 INT4+BDR | $7.75\times10^{-7}$ | $1.19\times10^{-6}$ | $9.54\times10^{-7}$ |
| C4 KIVI-2 | $0$ | $0$ | $0$ |
| C5 KIVI-4 | $0$ | $0$ | $0$ |

C3 prefill 的 $\sim10^{-6}$ 是按页 vs 整段 BDR 旋转的 float32 舍入；$N{=}16$ 时两侧都是一次 encode 一页，差为 $0$。

**Decode**（逐步 $n$ 步；contiguous 按 token encode，paged 满 16 token 才 encode）：

| 格式 | attention | $K$ | $V$ |
|------|----------:|----:|----:|
| C0 FP16 | $0$ | $0$ | $0$ |
| C1 INT8 | $0$ | $0$ | $0$ |
| C2 INT4 | $0$ | $0$ | $0$ |
| C3 INT4+BDR | $5.66\times10^{-6}$ | $1.67\times10^{-6}$ | $1.81\times10^{-3}$† |
| C4 KIVI-2 | $0$ | $0$ | $0$ |
| C5 KIVI-4 | $0$ | $0$ | $0$ |

† 仅 **1 / 30** 行：$N{=}64$、seed $1$ 的 $V$。其余 29 行 $V$ 最大 $1.43\times10^{-6}$。这是 INT4 网格差 1 档（§8.3 允许），**不是**切页公式错误，也 **不是** C3 的典型布局误差。attention 仍为 $10^{-6}$ 量级。C2 无旋转，逐步 decode 仍精确为 $0$，说明差来自 BDR×encode 粒度，不是页表。

### 3.2 流量双列（prefill，seed $0$）

占用 token 的 payload / scale / zp **两列完全相同**；$B_{\mathrm{page}}$ 仅 paged 非零。$N{=}256$：

| 格式 | layout | payload | scale | zp | page | total |
|------|--------|--------:|------:|---:|-----:|------:|
| C0 | contiguous | 524288 | 0 | 0 | 0 | 524288 |
| C0 | paged | 524288 | 0 | 0 | 256 | 524544 |
| C1 | contiguous | 262144 | 8192 | 0 | 0 | 270336 |
| C1 | paged | 262144 | 8192 | 0 | 256 | 270592 |
| C2 | contiguous | 131072 | 16384 | 0 | 0 | 147456 |
| C2 | paged | 131072 | 16384 | 0 | 256 | 147712 |
| C3 | contiguous | 131072 | 16384 | 0 | 0 | 147456 |
| C3 | paged | 131072 | 16384 | 0 | 256 | 147712 |
| C4 | contiguous | 180224 | 12288 | 12288 | 0 | 204800 |
| C4 | paged | 180224 | 12288 | 12288 | 256 | 205056 |
| C5 | contiguous | 229376 | 12288 | 12288 | 0 | 253952 |
| C5 | paged | 229376 | 12288 | 12288 | 256 | 254208 |

$N{=}256$ 时 $B_{\mathrm{page}}=256\,\mathrm{B}$（32 页 × 8 B），相对 C0 payload 约 $0.05\%$。尾页例：$N{=}16\to17$，C0 payload $32768\to34816$，paged 的 $B_{\mathrm{page}}$ $16\to32$（各侧多一未满页）。contiguous 总量与 [`codec_compare`](../codec_compare/REPORT.md) §3.1 同形状字节一致。

$N{=}64<R$：C4/C5 全在 FP16 残差，bytes 与 C0 相同（payload 131072；paged 另加 $B_{\mathrm{page}}=64$），与 `codec_compare` §3.3 残差窗效应一致。

**刷窗边界 $N{=}128$**（Key 已整窗量化，Value 仍全在 FP16 残差；seed $0$ prefill）：

| 格式 | layout | payload | scale | zp | page | total |
|------|--------|--------:|------:|---:|-----:|------:|
| C0 | contiguous | 262144 | 0 | 0 | 0 | 262144 |
| C0 | paged | 262144 | 0 | 0 | 128 | 262272 |
| C4 | contiguous | 147456 | 4096 | 4096 | 0 | 155648 |
| C4 | paged | 147456 | 4096 | 4096 | 128 | 155776 |
| C5 | contiguous | 163840 | 4096 | 4096 | 0 | 172032 |
| C5 | paged | 163840 | 4096 | 4096 | 128 | 172160 |

C4 payload 相对 C0 下降来自 Key 量化，不是页数公式变化（两侧仍 16 页，$B_{\mathrm{page}}=128$）。逐步 decode 与 prefill 的四项 / 页数相同。

### 3.3 KIVI 四池页数（`kivi2`，与公式一致）

| $N$ | $K$ quant | $K$ res | $V$ quant | $V$ res | $B_{\mathrm{page}}$ |
|----:|----------:|--------:|----------:|--------:|--------------------:|
| 16 | 0 | 1 | 0 | 1 | 16 |
| 17 | 0 | 2 | 0 | 2 | 32 |
| 64 | 0 | 4 | 0 | 4 | 64 |
| 128 | 8 | 0 | 0 | 8 | 128 |
| 129 | 8 | 1 | 1 | 8 | 144 |
| 256 | 16 | 0 | 8 | 8 | 256 |

$N{=}128$：Key 整窗量化 8 页，Value 仍 8 页 FP16 残差。$N{=}129$：Key 残差 1 token 新开一页；Value 溢出 1 token 进量化尾页，残差保持 8 页。$R$ 与 $P_{\mathrm{size}}$ 对齐，故两侧页数之和等于均匀路径的 $2\lceil N/16\rceil$，没有因分池多出一页。均匀 C0 页数：$N{=}16$ 起为 $1{+}1$ / $2{+}2$ / $4{+}4$ / $8{+}8$ / $9{+}9$ / $16{+}16$，与上表 $B_{\mathrm{page}}$ 相同。

---

## 4. 分析与讨论

| 布局 | 优势 | 劣势 / 适用边界 |
|------|------|----------------|
| **contiguous** | 无页表；C0–C2 / C4–C5 与 paged 数值对齐 | 理想连续地址上界；正式流量表不能只报这一列 |
| **paged** | 规则 16-token 页；四池可记账；$B_{\mathrm{page}}$ 明确 | 页表开销（本设定下相对 payload 很小）；C3 逐步 decode 与按页 encode 可在个别种子上 INT4 差 1 档 |

**综合**：

1. **正式流量表必须双列**；本实验 $B_{\mathrm{page}}$ 已按 §8.5 可复现。  
2. 切页 **不改变** C0–C2 / C4–C5 的量化语义。  
3. C3 **prefill** 与 contiguous 对齐到 $10^{-6}$。逐步 decode 的典型差仍是 $10^{-6}$；1/30 行可见 1 档 INT4（$V{=}1.81\times10^{-3}$），来自 encode 粒度×BDR，attention 仍为 $10^{-6}$。M5 主曲线不要把该差写成精度退化。  
4. $B_{\mathrm{pad}}$（整页 DMA 空洞）按协议默认 **不** 并入主 `bytes/token`。

---

## 5. 局限与有效性

- 合成张量、无因果 mask、非真实 LLM KV；0.5B 冒烟未跑（不阻塞 M4）。  
- 读侧仍全量 `load`，无 page-wise partial attention（§8 明确不要求）。  
- INT4 / KIVI 载荷仍按协议名义比特记账，未 nibble-/bit-pack。  
- 未扫 $P_{\mathrm{size}}$；未上 8B / 4K–32K Pareto（M5）。

---

## 6. 结论与后续工作

- 两条后端的 paged 布局已在阶段 A 与 contiguous **双报**：页数、$B_{\mathrm{page}}$、占用 token 的 payload/scale/zp 均符合 metrics v1.1。  
- C0–C2 / C4–C5 布局对齐到逐元素。C3 prefill 对齐到 float32 旋转容差；逐步 decode 允许 INT4 差 1 档（本实验 1/30 行），不否定切页规则。  
- 后续 M5 的 [KV 流量](../kv_pareto/REPORT.md) 和 [WikiText 精度](../wikitext_ppl/REPORT.md) 已独立记录；本布局结果不替代整模精度或硬件性能测量。
