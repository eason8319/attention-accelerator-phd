# R1 — 真实 KV Cache-Path 基线

正式研究入口（自 R1 起）。现行计划：[`docs/research_plan.md`](../../docs/research_plan.md)。

## 环境

```bash
cd /mnt/f/attention-accelerator-phd/research/r1_kv_baseline
conda env create -f environment.yml
conda activate r1-kv-baseline

# 无 GPU 准备机（CPU torch，体积更小）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 有 ≥24GB GPU 的正式评测机：
# pip install -r requirements.txt

export HF_HOME=/mnt/f/hf-cache
mkdir -p "$HF_HOME"
hf auth whoami
```

## 目录

```text
r1_kv_baseline/
├── environment.yml
├── requirements.txt
├── protocols/              # M0 已锁定：models_context.md / metrics.md
├── cache_path/             # 真实 quantize→pack→store→load→dequant（含 rotation / codecs）
├── kivi_repro/             # KIVI 表格复现
├── bytes_accounting/       # bytes/token 与敏感性
├── experiments/            # 实验统一入口：<name>/{run_*.py, REPORT.md, results/}
│                           # results/ 仅本地；云端只同步 REPORT.md
└── （勿再往 outputs/ 写实验记录；无独立 quant/ 包）
```

## 状态

- **M0 完成**：协议见 [`protocols/`](protocols/)。  
- **M1–M2 完成**：C0–C3 contiguous cache-path。  
- **M3（进行中）**：C4/C5 cache-path 已接入；合成对照 [`codec_compare`](experiments/codec_compare/)；[`kivi_eval`](experiments/kivi_eval/) **阶段 A 冒烟已通过**，Table 3 / LongBench 待阶段 B。  
- 下一步：实现 `run_table3.py` / `run_longbench.py` 并在 ≥24 GB GPU 上复现。
