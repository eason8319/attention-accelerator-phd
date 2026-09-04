# KIVI 模型测试与评估实验

**状态**：阶段 B 正式评测已完成（Table 3 + LongBench：fp16 / kivi2 / kivi4 全集均 `ok`）  
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

脚本：`run_table3.py`、`run_longbench.py`（已实现，本地未跑；见 2.4）。  
共享逻辑：
- LM-Eval：[`../../kivi_repro/lm_eval_tasks.py`](../../kivi_repro/lm_eval_tasks.py)（默认 CoQA / TruthfulQA / GSM8K）
- LongBench：[`../../kivi_repro/long_bench_tasks.py`](../../kivi_repro/long_bench_tasks.py)（四子组 / 打分 / `evaluate_tasks`）
与论文数字的对照放在实验 `REPORT.md` / 运行脚本中，不内嵌于上述模块。

### 2.4 `run_table3.py` / `run_longbench.py` 用法与已知限制

```bash
conda activate r1-kv-baseline
cd research/r1_kv_baseline

# Table 3（阶段 B，需 ≥24GB GPU + 已授权 Llama-2-7B）
python experiments/kivi_eval/run_table3.py \
    --model-id NousResearch/Llama-2-7b-hf --formats fp16,kivi2,kivi4 \
    --device cuda --stage B

# LongBench（阶段 B）
python experiments/kivi_eval/run_longbench.py \
    --model-id mistralai/Mistral-7B-Instruct-v0.2 --formats fp16,kivi2,kivi4 \
    --device cuda --stage B

# 冒烟（阶段 A，CPU/少量样本即可）
python experiments/kivi_eval/run_table3.py --limit 8 --device cpu --stage A
python experiments/kivi_eval/run_longbench.py --limit 3 --device cpu --stage A
```

两脚本均逐格式容错（一种格式失败不阻断其余格式，结果增量落盘到 `results/<name>/`），
支持 `--reference-json` 传入论文/官方数字算差值（脚本本身不内嵌任何论文数字）。

`load_llama_for_generate` 按 `config.model_type` 分发 KIVI patch：`llama` 走
`patch_llama` / `LlamaKiviAttention`，`mistral` 走 `patch_mistral` /
`MistralKiviAttention`（数值路径相同，sliding-window mask 仍由 HF
`MistralModel` 构造）。LongBench 协议锚点 `mistralai/Mistral-7B-Instruct-v0.2`
的 `kivi2`/`kivi4` 可与 `fp16` 一样走完整加载 → patch → 评测路径。

### 2.3 共同约束

超参：`group_size=32`，正式评测 `residual_length=128`（玩具冒烟用短窗 32 以加速）；核心路径须走 `LlamaKiviAttention` / `MistralKiviAttention` + `KiviKVCache`（禁止投影层 fake-quant 冒充）。

## 3. 目录约定

```text
kivi_eval/
├── REPORT.md              # 本文件（云端同步）
├── run_equiv_smoke.py     # 本地：量化关闭时与原生 HF 的等价性冒烟
├── run_table3.py          # 本地：Table 3
├── run_longbench.py       # 本地：LongBench
└── results/               # 本地（gitignore）：阶段 B 合并报告 + 分格式明细
    ├── table3/            # 合并后的 Table 3（fp16 / kivi2 / kivi4）
    ├── table3_kivi2/      # kivi2 作业原始落盘
    ├── table3_kivi4/      # kivi4 作业原始落盘
    └── longbench/         # LongBench（fp16 / kivi2 / kivi4 + 合并报告）
```

共享整模逻辑：[`../../kivi_repro/`](../../kivi_repro/)（`llama_kivi_attn` / `patch_llama` / `patch_mistral` / `hf_generate`）。

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

### 4.2 Table 3 / LongBench（阶段 B，2026-09-04 合并）

超参：`group_size=32`，`residual_length=128`。kivi 数字来自 2026-09-03 修复
attention 输出 `transpose` / 因果 mask / 8K 分块之后的重跑。Δ 相对**本仓库
fp16**，不是论文官方表（未传 `--reference-json`）。

本地合并报告：`results/table3/table3_report.md`、
`results/longbench/longbench_report.md`。

#### Table 3（`NousResearch/Llama-2-7b-hf`，LM-Eval 全集）

| format | CoQA | TruthfulQA | GSM8K | ΔCoQA | ΔGSM8K |
|--------|-----:|-----------:|------:|------:|-------:|
| fp16 | 64.57 | 30.00 | 13.34 | — | — |
| kivi4 | 65.08 | 29.66 | 13.87 | +0.52 | +0.53 |
| kivi2 | 60.32 | 30.60 | 11.52 | −4.25 | −1.82 |

- fp16：2026-08-07，1.3 h。
- kivi4 / kivi2：2026-09-03，7.8 h / 5.8 h。
- **kivi4 ≈ fp16**（差值在噪声内）；**kivi2 小幅掉点**，不再是修复前的 0 分。

#### LongBench（`Mistral-7B-Instruct-v0.2`，max length 8192，全集）

| format | qasper | qmsum | trec | lcc |
|--------|-------:|------:|-----:|----:|
| fp16 | 29.05 | 22.98 | 69.50 | 56.84 |
| kivi4 | 29.20 | 22.74 | 70.00 | 56.68 |
| kivi2 | 28.55 | 22.04 | 70.50 | 54.81 |

相对 fp16：kivi4 最大偏差 qmsum −0.24；kivi2 为 qasper −0.50 / qmsum −0.94 /
trec +1.00 / lcc −2.03。trec 生成已与 fp16 同型，不再出现乱码。

#### 重跑原因（已失效结果已清掉）

首次 kivi 全集分数崩溃（Table 3 CoQA/GSM8K = 0；LongBench trec = 0），根因是
`LlamaKiviAttention` prefill 少 `transpose(1, 2)`，以及 sdpa 路径下
`attention_mask=None` 未补因果 mask。8K 显式 mask 后又触发 24 GB 卡 OOM，
已按 query 分块（1024）修好。坏结果归档、node6 CUDA 失败日志已删除；
`results/` 只保留有效产物与对应 slurm 日志。

### 4.3 管线冒烟（2026-08-03，阶段 A，CPU）

用小型真 Llama 架构模型（`JackFram/llama-160m`，未受限、无需登录）对两个脚本
做了端到端冒烟，确认 fp16 / kivi2 / kivi4 三种格式的加载 → patch → 评测 →
落盘 → 报告生成全链路均可正常运行（分数本身无意义，仅验证机制）：

| 脚本 | 格式 | 结果 |
|------|------|------|
| `run_table3.py`（gsm8k, limit=1，及 limit=2 三任务的 fp16） | fp16 / kivi2 / kivi4 | 3/3 通过 |
| `run_longbench.py`（trec, limit=1, max_length=512） | fp16 / kivi2 / kivi4 | 3/3 通过 |

过程中发现并修复了一个依赖冲突（已入库，非临时绕过）：`datasets>=4.0` 移除了
脚本式数据集加载，而 `THUDM/LongBench`（v1）仍是脚本式数据集；但反过来
lm-eval 的 CoQA/TruthfulQA 等任务背后的数据集又已用 `datasets>=4.0` 新增的
`List` 特征类型重新导出、只能被 `datasets>=4.0` 读取——不存在同时满足两边的
单一 `datasets` 版本。修复方式：改写
[`kivi_repro/long_bench_tasks.py`](../../kivi_repro/long_bench_tasks.py) 的
`load_longbench`，绕开 `datasets.load_dataset` 的脚本机制，直接用
`huggingface_hub.hf_hub_download` 拉取 `data.zip`、本地解压按行读取 `.jsonl`
后用 `Dataset.from_list` 构造，不再依赖任何远端脚本/schema；因此
[`requirements.txt`](../../requirements.txt) 的 `datasets` 依赖**不需要**设
版本上限（当前环境装的是 5.0.1）。

跑这次冒烟时用到的具体机器（网络/资源限制、缓存预热方式等）相关细节，
不在此记录，见本地文件 `CLUSTER_NOTES.md`（不入库）。

### 4.4 阶段 B 前置准备（2026-08-04）

`hf auth login` 已完成，`NousResearch/Llama-2-7b-hf` 与
`mistralai/Mistral-7B-Instruct-v0.2` 权重已预热到本地 HF 缓存。这两步之后
阶段 B 已在 2026-08-07（fp16）与 2026-09-03（kivi 修复后重跑）跑完，见 4.2。

### 4.5 等价性冒烟（2026-09-03，`run_equiv_smoke.py`）

`JackFram/llama-160m`、CPU：把 `residual_length` 设到大于 prompt，量化关闭
时 patched 模型必须与原生 HF 数值等价。`prefill_logits_equiv` /
`greedy_generate_equiv` / `kivi2_mechanism` / `kivi4_mechanism` **4/4 PASS**
（`max|Δlogits|=0.0106`）。该检查能拦住 4.2 里那两类 prefill bug。
