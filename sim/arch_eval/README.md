# P3 — 架构评估工具链（SCALE-Sim v3 + Timeloop/Accelergy）

详细步骤与验收标准见 [docs/learning_plan.md](../../docs/learning_plan.md) 的 P3 一节。

## 目标

用 SCALE-Sim v3 与 Timeloop/Accelergy 评估 attention 各 GEMM 的 cycle/utilization/traffic 与 energy/area，
结合手推 roofline 验证 decode 阶段 memory-bound 结论，产出瓶颈分析短文（阶段 1 素材）。

## 环境（WSL2）

```bash
conda env create -f environment.yml
conda activate p3-arch-eval
```

当前独立环境为 `p3-arch-eval`（Python 3.11、SCALE-Sim 3.0.0）。
Timeloop/Accelergy 使用官方 `latest-amd64` Docker 镜像。

## 目录结构

```
arch_eval/
├── reading_notes.md    # Week 0：相关材料精读笔记
├── environment.yml     # P3 独立 Conda 环境
├── scale-sim/          # SCALE-Sim 配置（*.cfg）与 GEMM topology csv
├── timeloop/           # Timeloop arch/workload/mapping YAML
├── roofline.py         # roofline 解析模型
├── collect_results.py  # 三方结果汇总出图
├── outputs/            # 自动生成的 CSV / Markdown / 图表
└── analysis.md         # 瓶颈分析短文（最终产出）
```

## Roofline

```bash
conda run -n p3-arch-eval python sim/arch_eval/roofline.py
```

默认参数为 128 TOPS INT8、1 TB/s HBM、16 MiB SRAM，以及
LLaMA-7B 规模单层 attention。脚本评估 prefill/decode ×
4K/32K/128K，输出 `outputs/roofline_table.csv` 和
`outputs/roofline_table.md`。

## SCALE-Sim

```bash
conda run -n p3-arch-eval python sim/arch_eval/scale-sim/run_scalesim.py
```

配置为 32×32 PE、16 MiB 分区 SRAM，分别运行 WS/OS。脚本覆盖
prefill/decode × 4K/32K/128K × QKV 投影、`QK^T`、`PV`、输出投影，
输出：

- `outputs/scalesim_results.csv`：48 条汇总 cycle/utilization/traffic
- `outputs/scalesim_summary.md`：prefill/decode 利用率对照
- `outputs/scalesim_raw/`：SCALE-Sim 原始三类报告

由于 SCALE-Sim 会显式构造 demand matrix，超长上下文采用每维最大
256 的固定 tile 仿真，再按精确 tile 数汇总。结果用于趋势比较；
不建模跨 tile 复用与 overlap，不能直接视为端到端绝对 latency。

## Timeloop + Accelergy

```bash
conda run -n p3-arch-eval python sim/arch_eval/timeloop/run_timeloop.py
```

脚本复用 SCALE-Sim 的 24 条 WS workload 元数据，对 6 种唯一 tile
运行 Timeloop mapper，再按相同 repetition 汇总 energy。Accelergy
另外生成 45 nm 下的 area reference table。输出：

- `outputs/timeloop_energy.csv`：逐 GEMM 的 MAC/register/SRAM/DRAM energy
- `outputs/timeloop_area.csv`：32×32 MAC、register、16 MiB SRAM 的 area
- `outputs/timeloop_summary.md`：逐层 energy 占比和模型适用范围
- `timeloop/generated/`：每种 tile 的 mapping 和原始 stats

官方 2020 ISPASS tutorial exercise 00 已在该镜像中跑通。仓库较新的
v0.4 `example_designs` 与镜像内前端存在 schema 版本漂移，因此本项目
使用镜像 CLI 原生支持的 legacy schema，保证运行可复现。

## 交叉校验

```bash
conda run -n p3-arch-eval python sim/arch_eval/collect_results.py
```

读取 Roofline / SCALE-Sim / Timeloop 三类 CSV，输出：

- `outputs/cross_joined.csv`：逐 GEMM 合并表
- `outputs/util_prefill_vs_decode.png`：WS/OS 利用率对照
- `outputs/traffic_energy_stack.png`：traffic 与 energy 堆叠
- `outputs/roofline_points.png`：AI vs attained TOPS
- `outputs/cross_validation.md`：相对结论与偏差来源说明

验收以相对趋势为准（decode ≪ prefill、memory-bound），不要求三方绝对值一致。

## 验收 checklist

- [x] SCALE-Sim 跑通 attention GEMM 序列（QK^T / PV / 投影，prefill 与 decode 两组形状）
- [x] Timeloop 同一 workload 的 energy 分解
- [x] decode 利用率显著低于 prefill 的复现 + roofline 解释
- [x] `analysis.md` 瓶颈分析短文（见 [验收报告](../../docs/progress/p3_arch_eval_report.md)）
