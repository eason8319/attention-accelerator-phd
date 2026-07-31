# KIVI 模型测试与评估实验

**状态**：冒烟已通过（`kivi_repro` 整模路径）；Table 3 / LongBench 未开始  
**阶段**：R1 / M3  
**实验目录**：[`experiments/kivi_eval/`](.)（本报告入库；`results/` 与 `run_*.py` 仅本地）  
**协议**：[`protocols/models_context.md`](../../protocols/models_context.md)、[`protocols/metrics.md`](../../protocols/metrics.md)

---

## 1. 目的

1. **冒烟（阶段 A）**：确认 [`kivi_repro/`](../../kivi_repro/) 整模路径可正常运作（patch → 前向 → `generate` → clear / bytes），**不作精度主张**。  
2. **正式评测（阶段 B）**：在协议锚模型上对齐 KIVI 官方口径，产出与论文数字的对比表。

合成谱系对照见 [`../codec_compare/REPORT.md`](../codec_compare/REPORT.md)，**不**替代本实验。

## 2. 计划内容

### 2.1 冒烟测试（阶段 A）

| 子项 | 设定 | 检查点 | 禁止 |
|------|------|--------|------|
| patch | 默认玩具 Llama（可 `--model-id` 换 HF） | `patch_llama_model` 后全部层为 `LlamaKiviAttention` | 宣称精度 / 与论文并表 |
| 前向 + generate | KIVI-2 / KIVI-4；prefill 越过残差窗 | logits 有限；`generate_ids` 产出新 token；Value 残差夹窗 | 详细 rel-$\ell_2$ / bytes 对比表 |
| clear / bytes | 同上 | `clear_llama_kivi_caches` 清空；`bytes_stored_llama_kivi` 可调用 | 把冒烟数字写成正式结论 |

脚本：[`run_smoke.py`](run_smoke.py)。

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline
python experiments/kivi_eval/run_smoke.py
# 可选：真实 HF Llama
# python experiments/kivi_eval/run_smoke.py --model-id NousResearch/Llama-2-7b-hf --device cuda
```

### 2.2 正式评测（阶段 B，≥24 GB GPU）

| 子项 | 模型 | 任务 / 设定 | 对照 |
|------|------|-------------|------|
| Table 3 | `NousResearch/Llama-2-7b-hf` | LM-Eval：CoQA / TruthfulQA / GSM8K | FP16 vs KIVI-2 / KIVI-4 vs 论文 Table 3 |
| LongBench | `Mistral-7B-Instruct-v0.2` | max length 8192；四子组各 ≥1 代表任务 | 同上 vs KIVI 论文长上下文表 |

脚本：`run_table3.py`、`run_longbench.py`（待实现）。  
共享逻辑：
- LM-Eval：[`../../kivi_repro/lm_eval_tasks.py`](../../kivi_repro/lm_eval_tasks.py)（默认 CoQA / TruthfulQA / GSM8K）
- LongBench：[`../../kivi_repro/long_bench_tasks.py`](../../kivi_repro/long_bench_tasks.py)（四子组 / 打分 / `evaluate_tasks`）
与论文数字的对照放在实验 `REPORT.md` / 运行脚本中，不内嵌于上述模块。

### 2.3 共同约束

超参：`group_size=32`，正式评测 `residual_length=128`（玩具冒烟用短窗 32 以加速）；核心路径须走 `LlamaKiviAttention` + `KiviKVCache`（禁止投影层 fake-quant 冒充）。

## 3. 目录约定

```text
kivi_eval/
├── REPORT.md          # 本文件（云端同步）
├── run_smoke.py       # 本地：阶段 A 冒烟（kivi_repro）
├── run_table3.py      # 本地：Table 3（待实现）
├── run_longbench.py   # 本地：LongBench（待实现）
└── results/           # 本地：冒烟 JSON（gitignore）
```

共享整模逻辑：[`../../kivi_repro/`](../../kivi_repro/)（`llama_kivi_attn` / `patch_llama` / `hf_generate`）。

## 4. 结果

### 4.1 冒烟（2026-07-31）

| 项 | 值 |
|----|-----|
| 模型 | 玩具 Llama（随机权重；`hidden=64`，2 层） |
| 设备 | CPU |
| 路径 | `kivi_repro.patch_llama` + `hf_generate.generate_ids` |
| 格式 | `kivi2` / `kivi4` |
| 汇总 | **4/4 PASS** |

检查项：`patch_llama`、`forward_generate_kivi2`、`forward_generate_kivi4`、`clear_caches` 均通过（logits 有限、残差窗刷入后仍可 generate、cache 可清空）。

本地明细：`results/smoke_run_config.json`。

**局限（冒烟）**：玩具权重、短残差窗；仅验证接口与数值有限性，不作与论文并表的精度结论。

### 4.2 Table 3 / LongBench

（待跑。）
