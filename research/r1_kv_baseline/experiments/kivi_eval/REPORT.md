# 实验报告：整模路径上 C0 / C4 / C5 任务精度

**实验日期**：2026-09-03 至 2026-09-04；**整理日期**：2026-09-09。
**状态**：本报告所列批次已完成；以保留的原始结果为依据。
**证据来源**：results/table3/table3_summary.json、results/longbench/longbench_summary.json 及逐项原始输出。

**研究内容**：R1 / KIVI 整模任务评估  
**性质**：真实权重、整模 KV cache-path（非投影 fake-quant；非合成张量）  
**对照**：相对本仓库 FP16；未传 `--reference-json`，**不与论文官方表并排**  
**实验目录**：[`experiments/kivi_eval/`](.)（本报告入库；`results/` 与 `run_*.py` 仅本地）  
**协议**：[`protocols/models_context.md`](../../protocols/models_context.md)、[`protocols/metrics.md`](../../protocols/metrics.md)

合成谱系对照见 [`../codec_compare/REPORT.md`](../codec_compare/REPORT.md)，**不**替代本实验。

---

## 1. 实验目的

在协议锚模型上走同一整模路径（HF 权重 → KIVI patch → `KiviKVCache` → generate / lm-eval），对照 C0 / C4 / C5，回答：

1. 真实 LLM 上，KIVI-2 / KIVI-4 相对本仓库 FP16 的任务分差多少？  
2. 合成 cache-path 上 C4 误差很大（见 `codec_compare`），下游任务是否同样崩？  
3. 路径是否可复现：patch 后与原生 HF 在「量化关闭 + 残差窗盖住 prompt」时数值等价？

本实验只报**算法层**任务分。bytes/token、paged 布局属流量与精度 / 分页布局，此处不主张流量或架构结论。

---

## 2. 方法与设置

### 2.1 被测格式

| ID | 入口 | 路径 | 要点 |
|----|------|------|------|
| C0 `fp16` | `"fp16"` | 原生 HF attention | 未量化参考；不经 `KiviKVCache` |
| C4 `kivi2` | `"kivi2"` | `LlamaKiviAttention` / `MistralKiviAttention` + `KiviKVCache` | K per-channel / V per-token 2-bit |
| C5 `kivi4` | `"kivi4"` | 同上 | 4-bit |

超参与协议一致：`group_size=32`，`residual_length=128`。禁止投影层 fake-quant 冒充。`load_*_for_generate` 按 `config.model_type` 分发：`llama` → `patch_llama`，`mistral` → `patch_mistral`（数值核相同；Mistral sliding-window mask 仍由 HF 构造）。KIVI 默认 `attn_implementation=eager`；8K 显式 mask 按 query 分块（1024），避免 24 GB 级卡 OOM。

### 2.2 Golden、指标与负载

- Golden：同一模型、同一评测脚本下的 **C0 FP16**。  
- $\Delta$：本仓库该格式分数 − 本仓库 FP16（百分点）。  
- Table 3 主分：CoQA `em`、TruthfulQA `bleu_max`、GSM8K `exact_match`。  
- LongBench：四子组各 1 个代表任务，官方 prompt / max_gen / scorer。

| 套件 | 模型 | 任务 | 其它 |
|------|------|------|------|
| Table 3 | `NousResearch/Llama-2-7b-hf` | CoQA / TruthfulQA / GSM8K，全集 | $B{=}1$ |
| LongBench | `mistralai/Mistral-7B-Instruct-v0.2` | qasper / qmsum / trec / lcc，全集 | max length $8192$ |

墙钟平台：NVIDIA RTX 6000 Ada Generation（48 GB），`torch=2.5.1+cu121`。分格式作业后合并到 `results/table3/`、`results/longbench/`。

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline

python experiments/kivi_eval/run_table3.py \
    --model-id NousResearch/Llama-2-7b-hf --formats fp16,kivi2,kivi4 \
    --device cuda --stage B

python experiments/kivi_eval/run_longbench.py \
    --model-id mistralai/Mistral-7B-Instruct-v0.2 --formats fp16,kivi2,kivi4 \
    --device cuda --stage B
```

本地产物（不同步云端）：`results/table3/table3_summary.json`、`results/longbench/longbench_summary.json` 及分格式 `lm_eval_*.json` / `*.jsonl`。任务提交在服务器管理，不属于结果同步范围。

### 2.3 正确性检查（不作精度主张）

| 检查 | 设定 | 结果 |
|------|------|------|
| 接口冒烟 | 玩具 Llama；`run_smoke.py` | 4/4 PASS（patch / generate / 刷窗 / clear） |
| 数值等价 | `JackFram/llama-160m`，CPU；残差窗大于 prompt、量化关闭；`run_equiv_smoke.py` | 4/4 PASS（`max\|\Delta\mathrm{logits}\| = 0.0106`） |

等价性检查覆盖 attention 输出布局及因果 mask 语义；任务分数仅引用本报告列出的有效结果。

---

## 3. 实验结果

### 3.1 Table 3（Llama-2-7B，全集）

| 格式 | CoQA | TruthfulQA | GSM8K | $\Delta$CoQA | $\Delta$TruthfulQA | $\Delta$GSM8K | 墙钟 |
|------|-----:|-----------:|------:|------------:|----------:|-------------:|-----:|
| C0 FP16 | 64.57 | 30.00 | 13.34 | — | — | — | 1.3 h |
| C5 KIVI-4 | 65.08 | 29.66 | 13.87 | $+0.52$ | $-0.34$ | $+0.53$ | 7.8 h |
| C4 KIVI-2 | 60.32 | 30.60 | 11.52 | $-4.25$ | $+0.60$ | $-1.82$ | 5.8 h |

KIVI-4 与 FP16 分数接近；由于只有单次评测，不能判断差异是否处于统计噪声范围。KIVI-2 在 CoQA / GSM8K 上小幅掉点，TruthfulQA 未掉。

### 3.2 LongBench（Mistral-7B-Instruct，max length $8192$，全集）

| 格式 | qasper | qmsum | trec | lcc | $\Delta$qasper | $\Delta$qmsum | $\Delta$trec | $\Delta$lcc |
|------|-------:|------:|-----:|----:|--------------:|------------:|-----------:|----------:|
| C0 FP16 | 29.05 | 22.98 | 69.50 | 56.84 | — | — | — | — |
| C5 KIVI-4 | 29.20 | 22.74 | 70.00 | 56.68 | $+0.15$ | $-0.24$ | $+0.50$ | $-0.16$ |
| C4 KIVI-2 | 28.55 | 22.04 | 70.50 | 54.81 | $-0.50$ | $-0.94$ | $+1.00$ | $-2.03$ |

KIVI-4 最大绝对差为 trec $+0.50$ 个百分点，最大负向差为 qmsum $-0.24$ 个百分点。KIVI-2 主要掉在 lcc（$-2.03$）与 qmsum（$-0.94$）；trec 生成与 FP16 同型。kivi 墙钟各约 2.0 h。

### 3.3 与合成对照的关系

[`codec_compare`](../codec_compare/REPORT.md) 在合成高斯 / outlier 上：C4 rel-$\ell_2\sim 0.5$–$0.7$（最差），C5 刷窗后优于均匀 INT4。本实验说明：**下游任务上 C5 ≈ C0；C4 有可见但有限掉点，不是合成误差所暗示的崩盘。** 两表不可直接比绝对值（合成无因果 mask、无真实 KV 分布），只作方向对照。

---

## 4. 分析与讨论

| 格式 | 优势 | 劣势 / 适用边界 |
|------|------|----------------|
| **C0 FP16** | 未量化参考；实现简单 | 流量最大（本实验未测 bytes） |
| **C5 KIVI-4** | 两套件上与 FP16 对齐；真实模型上可作协议主 4-bit 锚点 | 墙钟明显长于 FP16（eager + 量化核，未优化） |
| **C4 KIVI-2** | 接口与 C5 相同；任务分未崩，可作极端压缩锚点 | CoQA $-4.25$、GSM8K $-1.82$、lcc $-2.03$；合成路径误差更大，不能用任务分反推 cache 重建误差小 |

**综合（本实验）**：

1. **要对齐协议、尽量保任务分**：C5。  
2. **要压到 2-bit 并接受数点掉分**：C4；不宜再把合成 rel-$\ell_2$ 写成「真实模型不可用」。  
3. **未量化参考**：C0。
4. 未扫 `residual_length`；短于残差窗时 KIVI 仍可能「看起来无损」（见 `codec_compare` §3.3）。

---

## 5. 局限与有效性

- 未核对 KIVI 论文 Table 3 / LongBench 官方数字；不得把本仓库 FP16 或 $\Delta$ 写成「复现了论文表」。  
- 无 PPL、无 bytes/token、无 paged（分页布局 / 流量与精度）。  
- LongBench 只跑四子组代表任务，不是全套件。  
- 布局仅为 contiguous；`attn_implementation=eager`，墙钟不是吞吐上限。  
- 单次评测、无多种子；TruthfulQA 主分是 `bleu_max`，与部分论文常用的 MC 指标不同。  
- FP16 与 kivi 分日作业；FP16 不经 KIVI 核，软件栈微调未做 A/B，跨日期环境差异仍是潜在混杂因素。

---

## 6. 结论与后续工作

- 已在协议锚模型上跑通 **C0 / C4 / C5** 整模阶段 B：Table 3 与 LongBench 全集均 `ok`。  
- **KIVI-4 ≈ 本仓库 FP16**；**KIVI-2 小幅掉点**，与合成 cache-path 上 C4 误差过大的图景不同。  
- 整模路径（含 Mistral patch、因果 mask、8K 分块）已用等价性冒烟钉住，失效分数已清掉。  
- 后续已有 [Paged 布局](../paged_layout/REPORT.md)、[KV 流量](../kv_pareto/REPORT.md) 和 [WikiText PPL](../wikitext_ppl/REPORT.md) 独立报告。若要与论文并表，仍须核实参考来源，不能把本节 $\Delta$ 改写成官方差距。
