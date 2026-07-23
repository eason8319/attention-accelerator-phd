# P3 阅读笔记（Week 0）

> 对应 [PLAN.md](PLAN.md) 阅读材料。
> 目标：建立跑通工具前的概念框架；不必通读原文。

## 1. Roofline（Williams et al., CACM 2009）

**核心量**：算术/操作强度（operational intensity）

$$
\mathrm{AI} = \frac{\text{FLOPs（或 MACs）}}{\text{DRAM bytes}}
$$

字节按「过 cache 后仍触及 DRAM 的流量」计，不是寄存器级读写。

**双屋顶**：

| 屋顶 | 公式 | 含义 |
|------|------|------|
| Compute roof | $P_{\max}$（峰值算力） | AI 足够高时，性能被算力封顶 |
| Memory roof | $\mathrm{AI} \times B_{\mathrm{mem}}$ | AI 偏低时，性能被带宽封顶 |

可达性能：

$$
P = \min\!\bigl(P_{\max},\; \mathrm{AI}\cdot B_{\mathrm{mem}}\bigr)
$$

理论 latency 可写成 $\max(T_{\mathrm{compute}}, T_{\mathrm{memory}})$，其中
$T_{\mathrm{compute}}=\mathrm{FLOPs}/P_{\max}$，
$T_{\mathrm{memory}}=\mathrm{bytes}/B_{\mathrm{mem}}$。

**Ridge point**：$\mathrm{AI}^* = P_{\max}/B_{\mathrm{mem}}$。AI 低于此点 → memory-bound；高于 → compute-bound。

假想硬件示例：$P_{\max}=128\,\mathrm{TOPS}$（INT8）、$B=1\,\mathrm{TB/s}$ →
$\mathrm{AI}^* = 128/1 = 128$ ops/byte（按 INT8 峰值口径；手推时注意 ops 与 byte 定义一致）。

**对 attention 的直觉**：

- Prefill `QK^T`：近似 $S\times S$ 方阵 GEMM，计算量大，AI 相对高。
- Decode `QK^T`：$1\times S$ 瘦矩阵，每 token 仍要扫整段 K（约 $S\cdot d$ bytes），MACs 仅 $O(S\cdot d)$，AI 随 $S$ 增长仍偏低 → 易落在 memory roof 下。

链接：[CACM 摘要页](https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/) · [技术报告 PDF](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

---

## 2. SCALE-Sim v3（arXiv:2504.15377 + GitHub）

**回答什么问题**：给定 systolic 阵列配置 + 一层（或一层序列）的形状，给出 **cycle-accurate** 的 compute cycles、stall、**PE utilization**、SRAM/DRAM **traffic/bandwidth**（v3 还可接 Ramulator / Accelergy）。

**输入**：

1. **Config (`.cfg`)**  
   - `general`：run name  
   - `architecture_presets`：阵列尺寸（如 32×32）、三类 SRAM 容量、**Dataflow**（`ws` / `os` / `is`）、带宽等  
   - `run_presets`：用户指定带宽 vs 算最优无 stall 带宽  

2. **Topology (CSV)**  
   - 默认：卷积层参数格式  
   - Attention 用 **GEMM M/N/K** 格式；运行时加 `-i gemm`  

**输出报告（摘要）**：

| 文件 | 内容 |
|------|------|
| `COMPUTE_REPORT.csv` | 每层 cycles、stalls、utilization |
| `BANDWIDTH_REPORT.csv` | SRAM/DRAM 平均与峰值带宽 |
| `DETAILED_ACCESS_REPORT.csv` | 操作数级访问量与访问 cycles |

**Dataflow 与 GEMM 映射（论文 Table II 直觉）**：

| Dataflow | 阵列空间维倾向 | 时间维 |
|----------|----------------|--------|
| Weight Stationary (WS) | K×M | N |
| Output Stationary (OS) | M×N | K |
| Input Stationary (IS) | K×N | M |

形状与阵列不对齐时 utilization 下降；decode 的 $M=1$（或极瘦维）会让大量 PE 空转。

**与 v2 的关系**：v3 增加多核、稀疏、Ramulator、layout、Accelergy；评估 attention GEMM 时先用单核 dense + WS/OS 对照即可。

仓库：[scalesim-project/SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) · 论文 [arXiv:2504.15377](https://arxiv.org/abs/2504.15377)

---

## 3. Timeloop + Accelergy（ISPASS/ICCAD 2019 + tutorial）

**分工**：

| 工具 | 角色 |
|------|------|
| **Timeloop** | 解析模型 + **mapper**：在 mapspace 中搜索 tiling / 循环序 / bypass，估计 latency / 访问计数 |
| **Accelergy** | **energy-per-action**：按 arch 组件与动作类型估能量与面积 |

合在一起回答：**某 arch × 某 workload × 某 mapping 的 energy/area（及性能）分解**（MAC vs SRAM vs DRAM 等），不是 SCALE-Sim 那种阵列拍级调度。

**四类 YAML**：

| 类型 | 内容 |
|------|------|
| `architecture` | 存储层次、PE 阵列、带宽/容量约束 |
| `problem` / workload | 张量维度与投影（如 GEMM 的 M/N/K） |
| `mapping` | 循环因子、排列、哪些数据驻留哪一级（可手写或由 mapper 产出） |
| `constraints` | 缩小 mapspace（强制某些 factor / bypass / spatial） |

常用命令形态（tutorial）：

```text
timeloop-model   arch.yaml problem.yaml map.yaml      # 评估给定 mapping
timeloop-mapper  arch.yaml problem.yaml [constraints] # 搜索 mapping
```

**Accelergy 直觉**：DRAM 访问 ≫ SRAM ≫ RF/MAC；长序列 decode 扫 KV 时 DRAM energy 占比容易主导——与 roofline memory-bound 同故事、不同度量。

材料：[accelergy.mit.edu/tutorial.html](https://accelergy.mit.edu/tutorial.html) · [Accelergy-Project/timeloop-accelergy-exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises) · [timeloop.csail.mit.edu](https://timeloop.csail.mit.edu/)

---

## 4. Dataflow 与能量层次（Sze et al., Efficient Processing of DNN，Ch.5–6）

**为何 dataflow 重要**：一次 MAC 涉及多次访存；能量瓶颈往往在 **数据搬运** 而非乘法本身。Dataflow = loop nest 的顺序与 tiling，决定谁在 RF/PE 里「stationary」、谁在阵列上广播/累加。

**分类（与 SCALE-Sim 配置直接对应）**：

| Dataflow | 尽量少动什么 | 典型代价 |
|----------|--------------|----------|
| **WS** | 权重停在 PE RF | 激活广播、psum 在阵列上累加（TPU 风格） |
| **OS** | 部分和本地累加 | 权重/激活需反复送入 |
| **IS** | 输入激活驻留 | 权重单播、psum 空间累加 |
| **RS**（Eyeriss） | 同时照顾三类数据的行级复用 | 更灵活，mapspace 更大 |

**能量层次（粗量级）**：DRAM ≫ 全局缓冲 ≫ 阵列间通信 ≫ PE 寄存器 / MAC。  
「片外访存占比」问的是：有多少能量/流量落在 DRAM 这一层。

节选/教程：[Eyeriss 书摘 PDF](https://eyeriss.mit.edu/2020_efficient_dnn_excerpt.pdf) · [NeurIPS tutorial slides](https://eyeriss.mit.edu/2019_neurips_tutorial.pdf)

---

## 5. 两道自检题

### Q1：为什么 decode 的 `1×n` GEMM 容易 memory-bound？

Decode 一步对单个 query（或极少 token）做 attention：`QK^T` 形状约 $1\times S\times d$ 的 MACs，但 K（及随后的 V）仍按序列长度 $S$ 从片外/大缓冲读入。  
AI $\approx O(1)$（ops/byte 不随 $S$ 改善，甚至在带宽主导下更差），通常远低于 ridge point → $T_{\mathrm{memory}} > T_{\mathrm{compute}}$。  
阵列侧：空间维之一极小，PE utilization 也会掉（SCALE-Sim 会直接打出低利用率）。

Prefill 则是 $S\times S$ 级计算，同一批 K/V 被多次复用，AI 与利用率都更高。

### Q2：SCALE-Sim 与 Timeloop 各自回答什么？

| | SCALE-Sim | Timeloop (+ Accelergy) |
|--|-----------|-------------------------|
| 粒度 | Cycle-accurate 阵列调度 | 解析模型 + mapping 搜索 |
| 主产出 | **cycles / utilization / traffic** | **energy / area**（及 latency 估计） |
| 强项 | 形状与 dataflow 对 PE 利用率、访存 stall 的影响 | 存储层次能量分解、mapspace 探索 |

Roofline 提供理论上下界与「为何 bound」的一句话解释；二者仿真相对它允许有偏差，**相对趋势**（decode 更 memory-bound）须一致。
