# 评估指标体系与方法学

对应研究计划"五、研究方法与评估工具链"。

## 1. 指标体系

| 类别 | 指标 | 说明 |
|------|------|------|
| 性能 | latency (cycles, ms) | prefill / decode 分别统计 |
| 性能 | throughput (tokens/s) | 端到端 |
| 利用率 | PE utilization | 有效 MAC / 峰值 MAC |
| 访存 | SRAM access (bytes) | 片上读写总量 |
| 访存 | HBM traffic (bytes) | 片外读写总量 |
| 能效 | energy/token (nJ 或 pJ) | 由能量模型给出 |
| 能效 | TOPS/W | 峰值或有效 |
| 面积 | area (mm^2, 估算) | PE + SRAM + 非 GEMM 单元 |
| 精度 | task accuracy / cosine sim / MSE | 量化与近似引入的偏差 |

## 2. 评估层次

本框架用三层互相校准的评估方法（与真实研究工具链对应）：

1. **算法级（精度）**：Python/numpy 复现 attention、量化、非 GEMM 近似，
   与 FP32 golden 对拍，给出 cosine similarity / 相对误差。对应真实工具：PyTorch + lm-eval。
2. **架构级（性能/能量/面积）**：本仓库的解析模型与 cycle-emulated 仿真器
   （`baseline_model.py`, `flash_dataflow.py`, `simulator.py`, `energy.py`）。
   对应真实工具：Timeloop+Accelergy、SCALE-Sim v3、TransInferSim。
3. **RTL 级（PPA）**：`learning/p4_rtl/` 下 SystemVerilog 关键模块 + Verilator 仿真对拍。
   对应真实工具：Synopsys DC / Cadence Genus 综合。

## 3. 能量模型说明

`energy.py` 采用 Accelergy 风格的"每动作能量（energy-per-action）"表：
对 MAC、SRAM 读写、HBM 读写、非 GEMM 操作分别赋单位能量，并按数据精度（位宽）缩放。
绝对数值为工艺无关的相对量级（默认锚定到一个通用 7-16nm 量级），用于
**相对比较**（baseline vs flash、不同精度），而非绝对功耗预测。这与计划中
"绝对数值为相对量级，用于相对比较"的定位一致。

## 4. 实验矩阵

- 序列长度：{1K, 4K, 16K, 32K, 64K, 128K}
- 阶段：prefill / decode
- 精度组合：FP16 baseline / FP8 / INT8 / INT4(+BDR) / MXFP4
- 模型规模：LLaMA-3-8B-like、Qwen3-like（可在 `config.py` 配置）
