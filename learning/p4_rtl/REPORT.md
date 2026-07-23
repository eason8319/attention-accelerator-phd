# P4 RTL 关键模块验收报告

日期：2026-07-22

## 结论

P4 已完成验收。在独立环境 `p4-rtl` 下用 Verilator 实现并验证三个关键模块（exp 近似、online softmax 归约、4×4 WS INT8 systolic），全部与定点 / numpy golden **比特对拍通过**；并完成 FSA「rescale 嵌入阵列」映射笔记。

验证命令：

```bash
cd rtl && make sim-all
```

验证结果（2026-07-22）：

```text
compare_exp PASSED       # 8186 samples, 0 LSB mismatch; max_rel≈3.3e-4 < 1e-3
compare_softmax PASSED   # m/l bit-exact vs fixed-point golden
compare_sa PASSED        # 0/32 mismatch vs numpy INT32 GEMM
```

## 验收 Checklist

| PLAN.md 验收项 | 对应产出 | 状态 |
|---|---|---|
| exp 单元目标域相对误差 $<10^{-3}$，流水时序正确 | `exp_unit/exp_approx.sv`，`make sim-exp`，[notes/exp_unit.md](notes/exp_unit.md) | 通过（max_rel≈$3.3\times10^{-4}$） |
| softmax 任意分块语义一致，与定点 golden 对拍 | `softmax_unit/online_softmax.sv`，`make sim-softmax`，[notes/softmax_unit.md](notes/softmax_unit.md) | 通过（$m$/$l$ 比特一致；跨块大小 $\ell$ 相对偏差 $<1\%$ 已记录） |
| systolic GEMM 与 numpy 完全一致 | `systolic_array/*.sv`，`make sim-sa`，[notes/systolic_array.md](notes/systolic_array.md) | 通过（$M{=}8$，$0/32$ mismatch） |
| 每模块设计笔记 + 仿真简报 | `notes/{exp_unit,softmax_unit,systolic_array,fsa_mapping}.md` + 本报告 | 设计笔记与功能对拍简报完成；**未**存档波形截图 / Verilator 行覆盖率（见局限） |

## 产出说明

| 路径 | 角色 |
|------|------|
| `exp_unit/exp_approx.sv` | 16 段 PWL，$\mathrm{exp}$；Q6.10→UQ0.24，3 级流水 |
| `softmax_unit/online_softmax.sv` | 单行 online $m/\ell$（无 $O$/PV）；例化 exp |
| `systolic_array/{pe,systolic_array}.sv` | 4×4 WS INT8→INT32，skew/deskew，$\mathrm{LAT}=7$ |
| `scripts/*` | 定点工具、各模块 golden / gen / compare |
| `tb/tb_{exp,softmax,systolic}.sv` | Verilator TB |
| `Makefile` | `check` / `sim-exp` / `sim-softmax` / `sim-sa` / `sim-all` |
| `reading_notes.md` | FSA / PLENA / Softermax / TPU 概念笔记 |
| `notes/fsa_mapping.md` | rescale 落在底部 accumulator 的映射思考 |

### 关键设计选择（摘要）

- **exp**：真实 attention score（减 max 后）扫参；相对误差预算 $10^{-3}$。
- **softmax**：块缓冲 + FSM；$\alpha$ 与 $\sum\exp$ 走同一 `exp_approx`；不做 $O\leftarrow\alpha O+PV$。
- **systolic**：与 NBA 语义对齐的周期模型；TB 喂数/收数重叠以免漏采首拍 `valid_c`。
- **FSA 映射**：块间 $b\cdot(\ell,O)$ 在**底边 accumulator**，不在 PE MAC 内；见 [fsa_mapping.md](notes/fsa_mapping.md)。

## 已知局限（不影响本周学习验收）

- Softmax 无输出累加器 $O$；接 FSA 式完整 FA 需底边 merge + $b$ 端口。
- 定点 $\ell$ 跨不同 `block_size` 非比特恒等（非结合 + 近似 exp），相对偏差 $<1\%$；$m$ 精确。
- Makefile 开了 `--trace`，TB 未 `$dumpfile`；无波形 PNG / 覆盖率报告。
- 阵列为练手级 4×4 INT8，非 FSA 上行路径 / PWL-in-PE 的完整复制。

## 环境记录

- Verilator：**5.020**
- Conda：`p4-rtl`（Python 3.11，numpy 1.26.4，torch 2.5.1+cpu，pytest）
- 定义：`environment.yml`

```bash
conda env create -f learning/p4_rtl/environment.yml   # 或 conda activate p4-rtl
cd rtl && make check && make sim-all
```

## 后续衔接

- 主线 1：底边 accumulator + $b$ 乘加，把模块 B 的 $\alpha$ 接到 $O$；再考虑上行 / PWL 进 PE（FSA）。
- P5 / 主线 4：tile 调度与本仓库 RTL 延迟常数（$\mathrm{LAT}$、exp 3 拍、softmax FSM）可作粗粒度标定。
- 可选抛光：TB 落盘 VCD + 一页波形截图；`--coverage` 简报。
