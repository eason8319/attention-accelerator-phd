# P5 简易 tile-level 模拟器验收报告

日期：2026-07-22

## 结论

P5 已完成验收。在独立环境 `p5-tile-sim` 下实现粗粒度 FlashAttention tile 性能模型（硬件抽象 / workload / 事件调度 / 网格搜索），复现 tile 两端劣化与 latency–traffic Pareto，并与 P3 SCALE-Sim **趋势一致**（6/6 检查通过）。

验证命令：

```bash
conda activate p5-tile-sim
cd learning/p5_tile_sim
python run_p5.py
# 或分步：pytest -q && python search.py && python validate_vs_scalesim.py
```

验证结果（2026-07-22）：

```text
pytest                          25 passed
dual-end demo                   OK（小 tile DMA 暴露；过大超 SRAM / 失 DB）
tile search                     Pareto PNG + search_results.csv
validate_vs_scalesim            6/6 PASS
```

交叉校验摘要（WS，`QK_T`+`PV`）：

```text
util prefill/decode:  SCALE-Sim ≈69.5×；P5 ≈28.6×（同方向）
traffic 32K/4K:       prefill ≈64× / 60.7×；decode 8× / 8×
decode 更偏存储:      SS DRAM share 0.49>0.38；P5 dma_frac 0.031>0.004
```

## 验收 Checklist

| PLAN.md 验收项 | 对应产出 | 状态 |
|---|---|---|
| 复现 tile 过小 / 过大两端劣化 | `run_p5.py` dual-end demo；`test_simulator.py` | 通过 |
| tile 搜索输出 Pareto 前沿图 | `search.py` → `outputs/pareto_*.png`、`search_results.csv` | 通过 |
| 与 SCALE-Sim 趋势一致性报告 | `validate_vs_scalesim.py` → [cross_check_vs_scalesim.md](outputs/cross_check_vs_scalesim.md) | 通过（6/6） |
| 模块化 + 混合精度字节钩子 | `hw_config` / `workload` / `simulator` / `search`；`ElementBytes` | 通过 |

## 产出说明

| 路径 | 角色 |
|------|------|
| `hw_config.py` | 32×32 @ 1 GHz、16 MiB、1 TB/s（对齐 P3） |
| `workload.py` | prefill/decode、LLaMA-7B 形状、`ElementBytes` |
| `simulator.py` | FA $B_r\times B_c$ 事件链、串行 vs DB、空间 MAC |
| `search.py` | 网格搜索 + Pareto CSV/PNG |
| `validate_vs_scalesim.py` | 对照 P3 `scalesim_results.csv` |
| `run_p5.py` | 一键：pytest → demo → search → validate |
| `reading_notes.md` | Week 0：FA IO / Timeloop mapspace / PLENA ISA |
| `outputs/` | Pareto 图、CSV、cross_check 报告 |

### 关键设计选择（摘要）

- **数据流**：外层 $B_r$（Q）、内层 $B_c$（KV）；decode 强制 $B_r=1$。
- **合法性**：$\mathrm{Footprint}\le\mathrm{SRAM}$；仅当 $2\cdot\mathrm{Footprint}\le\mathrm{SRAM}$ 启用 DB。
- **空间 MAC**：有效吞吐 $\min(B_r,R)\times\min(B_c,C)$，使 decode 瘦矩阵 util 下跌并对齐 SCALE-Sim 叙事。
- **Pareto**：同时最小化 `latency_cycles` 与 `dram_traffic_bytes`。
- **校验口径**：只比趋势 / 比率，不要求绝对 cycle 对齐。

## 已知局限（不影响学习验收）

- 非 cycle-accurate：无阵列 skew、bank conflict、指令依赖 stall。
- Softmax 吞吐为可调常数；未接 P4 RTL 延迟标定。
- Prefill 在 1 TB/s 下多数点接近算力下界，Pareto 主要由 traffic（更大 $B_r$ → 更少 KV 重扫）拉开。
- 不生成 PLENA_ISA 指令流（留给阶段 5 / 主线 4）。

## 环境记录

- Conda：`p5-tile-sim`（Python 3.11，numpy 1.26.4，matplotlib 3.11.0，pytest 8.3.5）
- 定义：`environment.yml`
- 对照只读：`learning/p3_arch_eval/outputs/scalesim_results.csv`（无需 SCALE-Sim 运行时）

```bash
conda env create -f learning/p5_tile_sim/environment.yml
conda activate p5-tile-sim
cd learning/p5_tile_sim && python run_p5.py
```

## 后续衔接

- 主线 4 / 阶段 5：将本代价模型嵌入编译映射（多级缓冲、混合精度字节 sweep、ISA 下发）。
- 可用 P4 的 exp/softmax/阵列延迟常数替换粗吞吐假设，提高相对 SCALE-Sim / RTL 的保真度。
- 阶段 1 短文可并列引用 P3 `analysis.md` 与本仓库 `cross_check_vs_scalesim.md` 的趋势表。
