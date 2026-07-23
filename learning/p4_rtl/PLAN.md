# P4 — RTL 关键模块练手（3–4 周）

**目标**：用 SystemVerilog 实现三个与主线 1/3 直接相关的关键模块，全部通过 Verilator 仿真并与 P1 的 PyTorch golden model 对拍。

## 环境说明

Verilator 建议在 WSL2 下安装（`apt install verilator`，或源码编译较新版本）。测试向量由 P1 的 Python 脚本生成为文本文件，testbench 读入比对。

## 步骤

1. **模块 A：exp 近似单元**（`exp_unit/`）
  - 先在 Python 中做算法设计：范围规约（`exp(x) = 2^(x·log2e)`，拆整数/小数部分）+ 分段线性或二次多项式近似，扫描定点位宽 vs 误差
  - SystemVerilog 实现流水线版本（2–3 级），输入定点、输出定点
  - 用 P1 生成的真实 attention score 分布做误差评估（而非均匀随机数）
2. **模块 B：online softmax 归约单元**（`softmax_unit/`）
  - 实现 running max 更新、`exp(x−m)` 调用模块 A、running sum 更新与 rescale 乘法路径
  - 处理一个 attention row 的分块流入，输出归一化前的部分和与最终 `m/l`
  - 这是主线 1「in-place online softmax」的电路雏形
3. **模块 C：小规模 systolic array**（`systolic_array/`）
  - 4x4 或 8x8 weight-stationary INT8 MAC 阵列，含输入 skew/deskew 逻辑
  - 跑通小 GEMM 与 golden model 对拍
  - 完成后带着问题读 FSA 论文：FlashAttention 的 rescale 如何嵌入阵列内部？写一页映射思考笔记
4. **对拍流程**：每个模块配 `tb/` 下的 testbench + Python 参考脚本，`make sim` 一键运行比对。

## 验收标准

- [x] exp 单元在目标输入范围内相对误差 < 1e-3（或自定的误差预算），流水线时序正确
- [x] softmax 单元处理任意分块顺序结果一致，与定点 golden 对拍通过（跨块大小 $\ell$ 容差已记录）
- [x] systolic array GEMM 结果与 numpy 参考完全一致
- [x] 每个模块有一页设计笔记；功能对拍简报见 [REPORT.md](REPORT.md)（波形/覆盖率可选抛光）

## 阅读材料

- FSA / SystolicAttention (arXiv 2507.11331) — 主线1架构基线，P4 的目标读物
- PLENA (arXiv 2509.09505) — flattened array 与 ISA 设计
- Softermax (DAC 2021)、I-BERT (ICML 2021) — softmax/exp 硬件近似的经典做法
- TPU v1 论文 (ISCA 2017) — weight-stationary systolic array 原型
