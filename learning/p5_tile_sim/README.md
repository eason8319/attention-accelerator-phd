# P5 — 简易 tile-level 模拟器

详细步骤与验收标准见 [PLAN.md](PLAN.md)。  
验收报告：[REPORT.md](REPORT.md)。

## 目标

用 Python 实现粗粒度 tile-level 性能模型：建模 SRAM 容量约束、DMA 与计算的 double buffering 重叠，
对 FlashAttention 数据流做 tile 尺寸搜索，输出 latency/traffic Pareto 前沿。
这是主线 4 编译映射框架（阶段 5）的雏形。

## 环境与一键运行

独立 conda 环境 `p5-tile-sim`（与 P1–P4 并列；仅需 numpy / matplotlib / pytest，不依赖 SCALE-Sim 运行时）：

```bash
conda env create -f learning/p5_tile_sim/environment.yml
conda activate p5-tile-sim
cd learning/p5_tile_sim
python run_p5.py          # pytest → 两端劣化 demo → 搜索/Pareto → SCALE-Sim 趋势校验
# 或：pytest -q
```

默认硬件与 P3 对齐：32×32 @ 1 GHz、16 MiB SRAM、1 TB/s HBM；workload 为 LLaMA-7B 规模单层 attention。

## 文件结构

```
tile_sim/
├── environment.yml
├── reading_notes.md           # Week 0：FA IO / Timeloop mapspace / PLENA ISA
├── hw_config.py               # PE / SRAM / DRAM BW
├── workload.py                # prefill|decode + ElementBytes 混合精度钩子
├── simulator.py               # FA Br×Bc 事件链、串行 vs DB
├── search.py                  # 网格搜索 + Pareto CSV/PNG
├── validate_vs_scalesim.py    # 对照 P3 scalesim_results.csv
├── run_p5.py                  # 一键入口
├── test_*.py
└── outputs/                   # Pareto 图、CSV、cross_check_vs_scalesim.md
```

## 模块要点

| 模块 | 要点 |
|------|------|
| `simulator` | $\mathrm{Footprint}$ 合法性；$2\times$ 才 DB；空间 MAC $\min(B_r,R)\times\min(B_c,C)$ |
| `search` | Prefill 幂次 $(B_r,B_c)$；decode $B_r=1$；Pareto 最小化 latency × traffic |
| `validate` | 只验趋势：util 差距、traffic/latency ∝ $S$、decode 更偏存储 |

分步命令：

```bash
python search.py --seq 4096 32768 --modes prefill decode
python validate_vs_scalesim.py
```

## 验收 checklist

- [x] 复现 tile 过小/过大的两端劣化现象
- [x] tile 搜索输出 Pareto 前沿图
- [x] 与 SCALE-Sim 趋势一致性检查报告
- [x] hw config / workload / scheduler 模块化分离
