# 实验报告：Llama-3.1-8B 几何 bytes/token（双布局）

**日期**：2026-09-04  
**阶段**：R1 / M5 WP2  
**性质**：8B **几何**上的真实 cache-path 流量；未加载权重；无 PPL / 任务分  
**协议**：[`protocols/metrics.md`](../../protocols/metrics.md) v1.1；模型阶梯 [`models_context.md`](../../protocols/models_context.md) v1.2  
**记账**：[`bytes_accounting/traffic_model.py`](../../bytes_accounting/traffic_model.py)（C0 走 FP16 codec）  
**实验目录**：[`experiments/kv_pareto/`](.)（本报告入库；`results/` 与 `run_*.py` 仅本地）

---

## 1. 实验目的与问题

在协议主 Pareto 模型的 KV 几何上，为 C0–C5 × contiguous / paged 给出可引用的 **x 轴**（全模单步 bytes/token 与 $b_{\mathrm{eff}}$），并报正式压力点 $D(16384,1024)$。

回答：

1. 4K / 8K / 16K / 32K 上，各格式相对 C0 FP16 的流量比是多少？paged 的 $B_{\mathrm{page}}$ 占多少？  
2. $D(16384,1024)$ 的全程 KV 读 / $L_{\mathrm{out}}$ 与末步 $N{=}17407$ 单步各是多少？  
3. KIVI 残差窗如何把名义 2/4-bit 抬成有效比特？

本实验 **不** 报精度，**不能**单独构成 Pareto。y 轴（PPL 或任务分）属后续整模评测。M3 Table 3 不能替代。

---

## 2. 方法

### 2.1 几何与对照谱

| 项 | 值 |
|----|-----|
| 模型 ID（几何锚） | `meta-llama/Llama-3.1-8B-Instruct` |
| $n_{\mathrm{kv}},\,d,\,L$ | $8,\;128,\;32$（GQA；按实际 KV head 记账） |
| 格式 | C0–C5（`fp16` / `int8` / `int4` / `int4_bdr` / `kivi2` / `kivi4`） |
| 布局 | contiguous / paged；$P_{\mathrm{size}}{=}16$，$B_{\mathrm{pte}}{=}8\,\mathrm{B}$ |
| $N$ | $4096,\;8192,\;16384,\;32768$ |
| 压力点 | $D(16384,1024)$；末步 $N{=}L_{\mathrm{in}}+L_{\mathrm{out}}-1{=}17407$ |
| Batch | $1$ |
| 设备 | CPU；占位零张量（载荷/meta 形状与数值无关） |

C0 只经 `FP16Codec`，不用原生 HF attention。$B_{\mathrm{pad}}$ 不并入主 `bytes/token`。bytes/token 为 **32 层合计** 的 $\mathrm{Bytes}_{\mathrm{step}}$（协议单层 × 层数）；$b_{\mathrm{eff}}$ 与层数无关。

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline
python experiments/kv_pareto/run_kv_pareto.py
```

本地产物：`results/summary.json`（不同步云端）。

---

## 3. 结果

### 3.1 全模单步 bytes/token（contiguous）

单位 $\mathrm{MiB}{=}2^{20}\,\mathrm{B}$。C2 与 C3 字节相同（BDR 矩阵视为片上，不进 HBM）。C0–C3 随 $N$ 严格线性；C4/C5 因残差窗 $R{=}128$ 略慢于线性，$b_{\mathrm{eff}}$ 随 $N$ 下降。

| $N$ | C0 | C1 | C2/C3 | C4 | C5 |
|----:|---:|---:|------:|---:|---:|
| 4096 | $512$ | $260$ | $144$ | $102.5$ | $165.5$ |
| 8192 | $1024$ | $520$ | $288$ | $198.5$ | $325.5$ |
| 16384 | $2048$ | $1040$ | $576$ | $390.5$ | $645.5$ |
| 32768 | $4096$ | $2080$ | $1152$ | $774.5$ | $1285.5$ |

相对 C0（同一 $N$，contiguous）：

| $N$ | C1 | C2/C3 | C4 | C5 |
|----:|---:|------:|---:|---:|
| 4096 | $0.508$ | $0.281$ | $0.200$ | $0.323$ |
| 32768 | $0.508$ | $0.281$ | $0.189$ | $0.314$ |

$b_{\mathrm{eff}}$（contiguous / paged）：C0 $16$ / $16.004$；C1 $8.125$ / $8.129$；C2/C3 $4.5$ / $4.504$；C4 由 $N{=}4\mathrm{K}$ 的 $3.203$ 降到 $32\mathrm{K}$ 的 $3.025$；C5 由 $5.172$ 降到 $5.021$。

### 3.2 $N{=}32768$ 四项分解（双列）

占用 token 的 payload / scale / zp **两列相同**；$B_{\mathrm{page}}$ 仅 paged 非零。本长度 $N\mid 16$，全模 $B_{\mathrm{page}}{=}32N{=}1\,048\,576\,\mathrm{B}$（相对 C0 payload $0.024\%$）。

| 格式 | layout | payload | scale | zp | page | bytes/token |
|------|--------|--------:|------:|---:|-----:|------------:|
| C0 | contiguous | 4294967296 | 0 | 0 | 0 | 4294967296 |
| C0 | paged | 4294967296 | 0 | 0 | 1048576 | 4296015872 |
| C1 | contiguous | 2147483648 | 33554432 | 0 | 0 | 2181038080 |
| C1 | paged | 2147483648 | 33554432 | 0 | 1048576 | 2182086656 |
| C2 | contiguous | 1073741824 | 134217728 | 0 | 0 | 1207959552 |
| C2 | paged | 1073741824 | 134217728 | 0 | 1048576 | 1209008128 |
| C3 | contiguous | 1073741824 | 134217728 | 0 | 0 | 1207959552 |
| C3 | paged | 1073741824 | 134217728 | 0 | 1048576 | 1209008128 |
| C4 | contiguous | 544210944 | 133955584 | 133955584 | 0 | 812122112 |
| C4 | paged | 544210944 | 133955584 | 133955584 | 1048576 | 813170688 |
| C5 | contiguous | 1080033280 | 133955584 | 133955584 | 0 | 1347944448 |
| C5 | paged | 1080033280 | 133955584 | 133955584 | 1048576 | 1348993024 |

4K / 8K / 16K 的四项见本地 `results/summary.json`。凡 $N\mid 16$，paged 相对 contiguous 只多 $32N$ 字节。

### 3.3 压力点 $D(16384,1024)$

全程 KV 读合计 / $L_{\mathrm{out}}$，以及末步 $N{=}17407$。末步 paged：$P_{\mathrm{pool}}{=}\lceil 17407/16\rceil{=}1088$，全模 $B_{\mathrm{page}}{=}557056\,\mathrm{B}$。

| 格式 | layout | 均值 bytes/token | 末步 bytes/token | 末步 $b_{\mathrm{eff}}$ |
|------|--------|-----------------:|-----------------:|------------------------:|
| C0 | contiguous | 2214526976 | 2281570304 | $16$ |
| C0 | paged | 2215067872 | 2282127360 | $16.004$ |
| C1 | contiguous | 1124564480 | 1158609920 | $8.125$ |
| C1 | paged | 1125105376 | 1159166976 | $8.129$ |
| C2 | contiguous | 622835712 | 641691648 | $4.5$ |
| C2 | paged | 623376608 | 642248704 | $4.504$ |
| C3 | contiguous | 622835712 | 641691648 | $4.5$ |
| C3 | paged | 623376608 | 642248704 | $4.504$ |
| C4 | contiguous | 425420800 | 441372672 | $3.095$ |
| C4 | paged | 425961696 | 441929728 | $3.099$ |
| C5 | contiguous | 700667904 | 724480000 | $5.081$ |
| C5 | paged | 701208800 | 725037056 | $5.084$ |

均值相对 C0 contiguous：C1 $0.508$，C2/C3 $0.281$，C4 $0.192$，C5 $0.316$。不得与弱机 $D(1024,128)$-dev 并表。

---

## 4. 流量权衡

| 格式 | 相对 C0（32K / $D$ 均值） | 说明 |
|------|--------------------------|------|
| **C0 FP16** | $1$ | 上界；无 meta |
| **C1 INT8** | $0.508$ | scale 把名义 8-bit 抬到 $b_{\mathrm{eff}}{=}8.125$ |
| **C2/C3 INT4** | $0.281$ | $b_{\mathrm{eff}}{=}4.5$；C3 流量与 C2 相同 |
| **C4 KIVI-2** | $0.189$ / $0.192$ | 本谱最低流量；残差窗 + scale/zp 使 $b_{\mathrm{eff}}{\approx}3$ 而非 $2$ |
| **C5 KIVI-4** | $0.314$ / $0.316$ | **高于**均匀 INT4：残差 FP16 窗与非对称 meta 超过少掉的 4-bit 载荷 |

Paged 在本几何下相对 payload 可忽略（32K 上约 $0.024\%$），正式表仍须双列。

---

## 5. 局限

- 只记账，不跑 8B 前向；数字依赖几何与 codec，不依赖权重。  
- 无 PPL / 任务分，不能画完整 Pareto。  
- INT4 / KIVI 载荷按名义比特，未 nibble-/bit-pack（与 M4 口径一致）。  
- 未扫 $P_{\mathrm{size}}$；未报 $B_{\mathrm{pad}}$；128K 未做。

---

## 6. 结论

- Llama-3.1-8B GQA 几何下，C0–C5 双列流量与 $D(16384,1024)$ 已可作主 Pareto 的 **x 轴**。  
- 32K 上 C4 约 C0 的 $19\%$；均匀 INT4 约 $28\%$；C5 约 $31\%$，比 C2 更费带宽。  
- 下一步：同一模型上补长上下文精度（WikiText-2 PPL 或任务分），与本表画在同一 Pareto。
