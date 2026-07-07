# P3 — 架构评估工具链（SCALE-Sim v3 + Timeloop/Accelergy）

详细步骤与验收标准见 [docs/learning_plan.md](../../docs/learning_plan.md) 的 P3 一节。

## 目标

用 SCALE-Sim v3 与 Timeloop/Accelergy 评估 attention 各 GEMM 的 cycle/utilization/traffic 与 energy/area，
结合手推 roofline 验证 decode 阶段 memory-bound 结论，产出瓶颈分析短文（阶段 1 素材）。

## 环境（Windows 下走 WSL2）

```bash
# WSL2 Ubuntu 内
pip install scalesim
docker pull timeloopaccelergy/timeloop-accelergy-pytorch:latest
```

## 目录结构

```
arch_eval/
├── scale-sim/          # SCALE-Sim 配置（*.cfg）与 GEMM topology csv
├── timeloop/           # Timeloop arch/workload/mapping YAML
├── roofline.py         # roofline 手推计算脚本（待实现）
├── collect_results.py  # 仿真结果整理出图（待实现）
└── analysis.md         # 瓶颈分析短文（最终产出）
```

## 从这里开始

第一步：跑通 SCALE-Sim 自带示例（`scalesim` 的 GoogleNet 样例），
然后写一个只含 `QK^T`（prefill，seq=4K，head_dim=128）单条 GEMM 的 topology csv 跑通它。

## 验收 checklist

- [ ] SCALE-Sim 跑通 attention GEMM 序列（QK^T / PV / 投影，prefill 与 decode 两组形状）
- [ ] Timeloop 同一 workload 的 energy 分解
- [ ] decode 利用率显著低于 prefill 的复现 + roofline 解释
- [ ] `analysis.md` 瓶颈分析短文
