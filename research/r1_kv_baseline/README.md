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

- **M0 完成**：协议 v1.0 见 [`protocols/`](protocols/)。  
- **M1 完成**：contiguous C0–C2 见 [`cache_path/`](cache_path/)。  
- **M2 完成**：INT4+BDR（C3）接入；实验见 [`experiments/m2_int4_bdr/REPORT.md`](experiments/m2_int4_bdr/REPORT.md)。  
- 下一步：**M3** KIVI 风格复现。
