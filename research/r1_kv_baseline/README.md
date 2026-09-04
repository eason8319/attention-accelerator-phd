# R1 — 真实 KV Cache-Path 基线

正式研究入口（自 R1 起）。长线计划：[`docs/research_plan.md`](../../docs/research_plan.md)。  
R1 实施细则：[`PLAN.md`](PLAN.md)。

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

上面步骤基于 WSL2 路径；在无 conda/GPU 于登录节点的集群上改用
[`activate.sh`](activate.sh)（`source research/r1_kv_baseline/activate.sh`）。
某台具体机器/集群特有的环境限制与踩坑记录见本地文件 `CLUSTER_NOTES.md`
（不入库，见 `.gitignore`；若该文件不存在说明当前机器还没写过这类记录）。

## 目录

```text
r1_kv_baseline/
├── PLAN.md                 # R1 实施细则（M0–M8）
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
- **M3 完成**：C4/C5 cache-path + Llama / Mistral KIVI patch；[`kivi_eval`](experiments/kivi_eval/) 阶段 B Table 3 / LongBench（fp16 / kivi2 / kivi4 全集）已跑通，见该目录 `REPORT.md`。  
- **M4 完成**：contiguous / paged 双报告，见 [`paged_layout`](experiments/paged_layout/REPORT.md)（阶段 A，C0–C5）。  
- **M5 WP1 完成**：[`bytes_accounting/traffic_model.py`](bytes_accounting/traffic_model.py) 封装 `bytes_breakdown`。  
- **M5 WP2 完成**：[`kv_pareto`](experiments/kv_pareto/REPORT.md) 8B 几何 C0–C5 双列流量与 $D(16384,1024)$。  
- **M5 WP3 完成**：`kivi_repro` 整模 C0–C5 cache-path；`fp16` 仍为原生 HF。精度点仍缺。  
- 下一步：M5 精度点（8B 长上下文 PPL 或任务分）。
