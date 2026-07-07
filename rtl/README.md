# P4 — RTL 关键模块

详细步骤与验收标准见 [docs/learning_plan.md](../../docs/learning_plan.md) 的 P4 一节。依赖 P1 的 golden model 生成测试向量。

## 目标

用 SystemVerilog 实现三个关键模块，全部通过 Verilator 仿真并与 PyTorch golden model 对拍：

| 模块 | 目录 | 说明 |
|------|------|------|
| A. exp 近似单元 | [exp_unit/](exp_unit/) | 范围规约 + 分段线性/多项式近似，2–3 级流水 |
| B. online softmax 归约单元 | [softmax_unit/](softmax_unit/) | running max/sum 更新 + rescale 路径（主线1电路雏形） |
| C. 小规模 systolic array | [systolic_array/](systolic_array/) | 4x4/8x8 weight-stationary INT8 MAC 阵列 |

testbench 与对拍脚本统一放在 [tb/](tb/)。

## 环境

WSL2 下 `apt install verilator`（建议 v5.x）。测试向量由 P1 的 Python 脚本导出为文本文件，testbench 读入比对。

## 从这里开始

第一步（先做算法再写 RTL）：在 Python 中实现 exp 的范围规约 + 分段线性近似，
扫描"分段数 × 定点位宽"对误差的影响，确定误差预算后再动手写 `exp_unit` 的 SystemVerilog。

## 验收 checklist

- [ ] exp 单元误差达标（真实 attention score 分布上评估），流水时序正确
- [ ] softmax 单元任意分块顺序结果一致，与 P1 逐拍对拍通过
- [ ] systolic array GEMM 与 numpy 参考完全一致
- [ ] 每模块一页设计笔记 + 波形/覆盖率简报
