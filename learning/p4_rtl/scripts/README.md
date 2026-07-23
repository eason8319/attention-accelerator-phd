# P4 对拍脚本与向量 I/O 约定

所有 Python 一律在 conda env **`p4-rtl`** 中运行（Makefile 已通过 `conda run -n p4-rtl python` 调用；环境定义见 [../environment.yml](../environment.yml)）。

## 对拍数据流

```
golden (P1/numpy) --gen_vecs_*.py--> build/vec_*/  --TB $fscanf--> DUT --$fwrite--> dut_*.txt
                                          |                                        |
                                          +---------------- compare_*.py ---------+
```

每个 `make sim-<mod>` 目标固定三步：

1. `gen_vecs_<mod>.py --out build/vec_<mod>`：调 golden 生成输入 + 期望输出。
2. Verilator 编译并运行 `tb_<mod>.sv`，TB 经 plusarg `+vec_dir=<dir>` 找到向量目录，读入输入、把 DUT 输出写回同目录。
3. `compare_<mod>.py --vec build/vec_<mod>`：按各模块误差预算比对，失败以非零码退出。

## 向量文件格式（统一约定）

- **一行一个样本**，内容为定点 raw 值的**十进制补码整数**（负数带 `-` 号）。
  - 选十进制而非十六进制：SV 侧 `$fscanf(fd, "%d", v)` 对负数直接可用，Python 侧 `np.savetxt/loadtxt` 零成本。
- 多维数据（矩阵/分块）**按行优先展平**；形状等元信息写在同目录 `meta.txt`（`key value` 每行一对），由 gen 脚本产出、TB 与 compare 脚本共同读取。
- 文件命名：

| 文件 | 写入者 | 内容 |
|------|--------|------|
| `x.txt`（或 `a.txt`/`w.txt`/`scores.txt` 等） | gen | DUT 输入（定点 raw） |
| `expected.txt`（可多个，如 `expected_m.txt`/`expected_l.txt`） | gen | golden 期望输出（定点 raw） |
| `dut_out.txt`（可多个，同名对应） | TB | DUT 实际输出（定点 raw） |
| `meta.txt` | gen | 形状 / 块大小 / Q 格式等 |

- TB 中读写模板：`$fscanf` 逐行读入、`$fwrite(fd, "%0d\n", out)` 逐行写出；TB 自身只做 I/O 与时序驱动，**数值判定交给 compare 脚本**（需要逐拍硬断言时才在 TB 里 `$fatal`）。

## Q 格式（模块 A 扫参已定，见 [../notes/exp_approx_sweep.md](../notes/exp_approx_sweep.md)）

由 [fixedpoint.py](fixedpoint.py) 提供 `QFormat` 与量化/文件工具：

| 用途 | 格式 | 说明 |
|------|------|------|
| exp 输入 $x = S - m \le 0$ | `Q6.10`（W=16） | 覆盖真实 score 主质量区；比 Q8.8 多 2 bit 小数 |
| exp 输出 $\exp(x)\in(0,1]$ | `UQ0.24`（W=24） | 小值量化台阶够细，满足 rel $<10^{-3}$ |
| exp PWL 段数 | 16 | 端点匹配 $2^{y_f}$，$y_f\in[0,1)$ |
| softmax 累加 $\ell$ | `Q16.16`（W=32） | 保宽防溢出（模块 B） |
| systolic 输入 | INT8 | raw 即 INT8 补码 |
| systolic 累加/输出 | INT32 | 与 numpy 比特一致 |

约定记法：`Q(I,F)` 有符号（符号位含在 I 内）、`UQ(I,F)` 无符号；LSB $=2^{-F}$。

`python fixedpoint.py` / `python exp_approx.py --self-test` 可跑内置自检。

## 本目录文件规划

| 文件 | 状态 | 作用 |
|------|------|------|
| `fixedpoint.py` | 已有 | Q 格式 + 向量文件 I/O + 误差度量 |
| `exp_approx.py` | 已有 | 范围规约 + PWL 扫参；浮点/定点误差分析 |
| `exp_rtl_model.py` | 已有 | 与 `exp_approx.sv` 比特一致的定点 golden |
| `gen_vecs_exp.py` / `compare_exp.py` | 已有 | 模块 A 对拍 |
| `softmax_rtl_model.py` | 已有 | online softmax 比特级 golden（$m,\ell$） |
| `online_softmax_golden.py` | 已有 | 从 P1 风格 attention 采样 score 行 |
| `gen_vecs_softmax.py` / `compare_softmax.py` | 已有 | 模块 B 对拍 |
| `gen_vecs_sa.py` / `compare_sa.py` | 待做 | 模块 C 对拍（numpy INT32 比特一致） |
