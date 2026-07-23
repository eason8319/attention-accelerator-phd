# P3 — 架构评估工具链上手（2–3 周，重点补短板）

**目标**：掌握 SCALE-Sim v3 与 Timeloop/Accelergy，亲手用仿真数据验证"decode 阶段 memory-bound"结论，产出可演化为阶段 1 短文的瓶颈分析。

## 环境说明（Windows）

Timeloop/Accelergy 与 SCALE-Sim 均建议在 **WSL2 (Ubuntu) + Docker** 下运行：

```bash
# WSL2 内
pip install scalesim            # SCALE-Sim
docker pull timeloopaccelergy/timeloop-accelergy-pytorch:latest  # Timeloop+Accelergy 官方镜像
```

## 步骤

1. **roofline 手推**：给定一组假想硬件参数（如 128 TOPS INT8、1 TB/s HBM、16 MB SRAM），计算 LLaMA-7B 规模单层 attention 在 seq=4K/32K/128K、prefill/decode 两种模式下的算术强度与理论 latency 上限，制成表格。
2. **SCALE-Sim v3**：
  - 配置一个 32x32 systolic array（weight/output stationary 各跑一遍）
  - 将 attention 拆成 GEMM 序列：`QK^T`、`PV`、QKV 投影、输出投影，写成 SCALE-Sim 的 GEMM topology csv
  - 对比 prefill（方阵）与 decode（`1×n` 瘦矩阵）的 cycle 数、PE 利用率、SRAM/DRAM traffic
3. **Timeloop + Accelergy**：
  - 跑通官方 exercises，学会 arch/workload/mapping/constraints 四类 YAML
  - 把步骤 2 的同一 workload 描述给 Timeloop，得到 energy/area 分解（MAC vs SRAM vs DRAM 占比）
4. **交叉校验**：SCALE-Sim 的 cycle 数、Timeloop 的 energy、roofline 的理论值三方互验，解释偏差来源。
5. **瓶颈分析短文**：整理成 3–5 页分析（图 + 表），回答：长上下文下片外访存占比多少？decode 利用率掉到多少？多大 SRAM 能容纳多长上下文的 KV tile？——这些数字直接支撑阶段 1。

## 验收标准

- [x] SCALE-Sim 跑通 attention GEMM 序列，输出 cycle/utilization/traffic csv
- [x] Timeloop 跑通同一 workload，输出 energy 分解
- [x] 复现 decode PE 利用率显著低于 prefill 的现象，并有 roofline 解释
- [x] 产出 `analysis.md` 瓶颈分析短文

## 阅读材料

- SCALE-Sim v3 (arXiv 2504.15377) 与 GitHub 文档
- Timeloop (ISPASS 2019) + Accelergy (ICCAD 2019) + 官方 tutorial exercises
- Roofline model (Williams et al., CACM 2009)
- Efficient Processing of DNN (Sze et al.) 第 5–6 章 — dataflow 分类与 energy 模型
