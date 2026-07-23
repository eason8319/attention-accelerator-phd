# P4 — RTL 关键模块

详细步骤与验收标准见 [PLAN.md](PLAN.md)。验收报告：[REPORT.md](REPORT.md)。

## 目标

用 SystemVerilog 实现三个关键模块，全部通过 Verilator 仿真并与定点 / numpy golden 对拍：

| 模块 | 目录 | 说明 |
|------|------|------|
| A. exp 近似单元 | [exp_unit/](exp_unit/) | 范围规约 + 16 段 PWL，3 级流水 |
| B. online softmax 归约单元 | [softmax_unit/](softmax_unit/) | running max/sum + rescale（主线1电路雏形） |
| C. 小规模 systolic array | [systolic_array/](systolic_array/) | 4×4 weight-stationary INT8 MAC + skew/deskew |

testbench 与对拍脚本统一放在 [tb/](tb/)、[scripts/](scripts/)。

## 环境

- **Verilator 5.020** + `make` + `g++`（WSL2）。
- **conda env `p4-rtl`**（Python 3.11 + numpy/pytest/torch-cpu），见 [environment.yml](environment.yml)。
- 冒烟：`make check`；全量对拍：`make sim-all`（或 `sim-exp` / `sim-softmax` / `sim-sa`）。
- 向量格式与 Q 约定：[scripts/README.md](scripts/README.md)；定点工具：[scripts/fixedpoint.py](scripts/fixedpoint.py)。

```bash
conda env create -f environment.yml   # 一次性
conda activate p4-rtl                 # 或 Makefile 内 conda run -n p4-rtl
make check && make sim-all
```

## 验收 checklist

- [x] exp 单元误差达标，流水时序正确（`make sim-exp`；[notes/exp_unit.md](notes/exp_unit.md)）
- [x] softmax 与定点 golden 比特对拍（`make sim-softmax`；[notes/softmax_unit.md](notes/softmax_unit.md)）
- [x] systolic GEMM 与 numpy 比特一致（`make sim-sa`；[notes/systolic_array.md](notes/systolic_array.md)）
- [x] 设计笔记 + FSA 映射（[notes/](notes/)，含 [fsa_mapping.md](notes/fsa_mapping.md)）
- [x] `make sim-all` + [验收报告](REPORT.md)
- [ ] 波形截图 / Verilator 覆盖率（可选抛光；报告中已记局限）

## 文档索引

| 文档 | 内容 |
|------|------|
| [reading_notes.md](reading_notes.md) | FSA / PLENA / Softermax / TPU 阅读笔记 |
| [notes/exp_approx_sweep.md](notes/exp_approx_sweep.md) | exp 扫参 |
| [notes/exp_unit.md](notes/exp_unit.md) | 模块 A |
| [notes/softmax_unit.md](notes/softmax_unit.md) | 模块 B |
| [notes/systolic_array.md](notes/systolic_array.md) | 模块 C |
| [notes/fsa_mapping.md](notes/fsa_mapping.md) | rescale 如何嵌入阵列 |
| [REPORT.md](REPORT.md) | 验收报告 |

## 状态

**P4 学习验收已完成**（2026-07-22）。后续可选：VCD 波形存档、底边 accumulator 原型、P5 tile 模拟器。
