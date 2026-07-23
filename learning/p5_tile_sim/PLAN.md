# P5 — 简易 tile-level 模拟器（2–3 周）

**目标**：用 Python 写一个粗粒度性能模型，建模 tiling / double buffering / SRAM 约束对 attention latency 与 traffic 的影响，为主线 4 编译映射框架打地基。

## 步骤

1. **硬件抽象**：参数化描述——PE 阵列 (rows×cols×MACs/cycle)、SRAM 各 buffer 容量、DRAM 带宽、非 GEMM 单元吞吐。
2. **事件粒度**：以 tile 为单位建模三类事件：DMA load、compute、DMA store；compute 与 DMA 在 double buffering 下可重叠，SRAM 容量决定能否 double buffer。
3. **FlashAttention 数据流建模**：按 P1 的分块结构建模 `QK^T → online softmax → PV` 的 tile 依赖链，统计每种 tile 尺寸下的总 cycle 与 DRAM traffic。
4. **tile 搜索**：网格搜索合法 tile 配置（受 SRAM 容量约束），输出 latency 最优与 traffic 最优的 Pareto 点；观察 prefill/decode 最优 tile 的差异。
5. **交叉校验**：与 P3 的 SCALE-Sim 结果对比趋势（绝对值可有偏差，趋势须一致），写明模型的简化假设。

## 验收标准

- [x] 模拟器可复现"tile 太小 → DMA 无法被 compute 掩盖；tile 太大 → 超 SRAM"的两端劣化
- [x] tile 搜索输出 Pareto 前沿图（latency vs traffic）
- [x] 与 SCALE-Sim 趋势一致性检查报告
- [x] 代码模块化（hw config / workload / scheduler 分离），可扩展混合精度字节数建模

## 阅读材料

- FlashAttention 论文 IO 复杂度分析一节（tile 与 SRAM 的关系）
- Timeloop 论文的 mapspace 概念 — 你的 tile 搜索就是一个极简 mapper
- PLENA 的编译器/ISA 章节 — 主线4的长期方向
