# P5 — 简易 tile-level 模拟器

详细步骤与验收标准见 [docs/learning_plan.md](../../docs/learning_plan.md) 的 P5 一节。建议在完成 P3 后开始。

## 目标

用 Python 实现粗粒度 tile-level 性能模型：建模 SRAM 容量约束、DMA 与计算的 double buffering 重叠，
对 FlashAttention 数据流做 tile 尺寸搜索，输出 latency/traffic Pareto 前沿。
这是主线 4 编译映射框架（阶段 5）的雏形。

## 建议文件结构

```
tile_sim/
├── hw_config.py        # 硬件抽象：PE 阵列 / SRAM buffers / DRAM 带宽
├── workload.py         # attention workload 描述（prefill/decode、seq、head_dim）
├── simulator.py        # tile 事件模型：DMA load / compute / store 与重叠
├── search.py           # 合法 tile 配置网格搜索 + Pareto 前沿
└── validate_vs_scalesim.py  # 与 P3 结果的趋势交叉校验
```

## 从这里开始

第一步：写 `hw_config.py` 与一个最简 `simulator.py`——
只建模单个 GEMM 在"无重叠串行 DMA+compute"下的总 cycle，再逐步加入 double buffering。

## 验收 checklist

- [ ] 复现 tile 过小/过大的两端劣化现象
- [ ] tile 搜索输出 Pareto 前沿图
- [ ] 与 SCALE-Sim 趋势一致性检查报告
- [ ] hw config / workload / scheduler 模块化分离
