# P3 架构评估工具链验收报告

日期：2026-07-15

## 结论

P3 已完成验收。建立独立环境 `p3-arch-eval`，实现 Roofline / SCALE-Sim v3 / Timeloop+Accelergy 工具链，完成 attention GEMM 的 cycle·utilization·traffic 与 energy·area 评估，并产出瓶颈分析短文。三方相对结论一致：**decode memory-bound、PE 利用率显著低于 prefill**。

验证命令：

```bash
conda run -n p3-arch-eval python learning/p3_arch_eval/roofline.py
conda run -n p3-arch-eval python learning/p3_arch_eval/scale-sim/run_scalesim.py
conda run -n p3-arch-eval python learning/p3_arch_eval/timeloop/run_timeloop.py
conda run -n p3-arch-eval python learning/p3_arch_eval/collect_results.py
```

关键数字（LLaMA-7B 规模单层 attention，32×32，WS）：

```text
Roofline: decode QK_T/PV AI≈50.9 ops/byte (memory); prefill≈248 (compute)
SCALE-Sim WS QK_T: prefill util≈73.15% vs decode≈1.05% (≈69×)
SCALE-Sim decode DRAM share≈49% of (SRAM+DRAM) words
Timeloop decode energy: SRAM≈89%, DRAM≈11% (PAT; 勿直接当 DRAM 主导结论)
SRAM KV capacity (INT8 K+V, 16 MiB): S_tile≈2048 tokens
```

## 验收 Checklist

| PLAN.md 验收项 | 对应产出 | 状态 |
|---|---|---|
| SCALE-Sim 跑通 attention GEMM 序列，输出 cycle/utilization/traffic csv | `scale-sim/run_scalesim.py`, `outputs/scalesim_results.csv` | 通过（48 行 WS/OS） |
| Timeloop 跑通同一 workload，输出 energy 分解 | `timeloop/run_timeloop.py`, `outputs/timeloop_energy.csv` | 通过（24 行 + area） |
| 复现 decode PE 利用率显著低于 prefill，并有 roofline 解释 | `outputs/cross_validation.md`, `util_*.png`, `roofline_*.png` | 通过（WS ≈69×，OS ≈32×） |
| 产出 `learning/p3_arch_eval/analysis.md` 瓶颈分析短文 | `learning/p3_arch_eval/analysis.md` | 完成 |

## 产出说明

- `roofline.py`：4K/32K/128K × prefill/decode 的 AI 与 bound 表。
- `scale-sim/run_scalesim.py`：32×32 WS/OS；固定 ≤256 tile × 重复；cycle/util/traffic。
- `timeloop/run_timeloop.py`：同一 tile 的 mapper energy + Accelergy area（Docker）。
- `collect_results.py`：三方合并、出图、偏差说明。
- `analysis.md`：回答片外占比、decode util、16 MiB KV tile 容量。
- `outputs/`：CSV、PNG、`cross_validation.md` 等可复现产物。

## 已知局限（不影响学习验收）

- SCALE-Sim tile 重复不建模跨 tile 复用/overlap，绝对值偏保守。
- Timeloop 捆绑 PAT 下大 SRAM 主导动态能量；不能据此宣称 DRAM energy 主导。
- 分析峰 128 TOPS 与 32×32@1 GHz≈2 TOPS 不是同一尺度；交叉比较用相对趋势。
- 官方 exercises 新版 v0.4 `example_designs` 与镜像前端有 schema 漂移；本项目用兼容 legacy schema，并已跑通 ISPASS 2020 exercise 00。

## 环境记录

- Conda：`p3-arch-eval`（Python 3.11，SCALE-Sim 3.0.0，numpy 1.26.4）
- Docker：`timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64`

创建环境：

```bash
conda env create -f learning/p3_arch_eval/environment.yml
conda activate p3-arch-eval
docker pull timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64
```

## 后续衔接

- 阶段 1 短文可直接引用 `analysis.md` 的三条结论与图。
- P5 tile simulator 对照本仓库 SCALE-Sim **趋势**（util / traffic 随 $S$）。
- 若写论文级 energy，需替换/标定 memory 能量模型后再谈 DRAM 占比。
