# Attention 加速器瓶颈分析（P3）

日期：2026-07-15  
对象：LLaMA-7B 规模单层 attention（`hidden=4096`，`heads=32`，`head_dim=128`）  
硬件假设：128 TOPS INT8、1 TB/s HBM、16 MiB 片上 SRAM；微架构对照为 32×32 systolic array（WS / OS）

本文汇总 Roofline、SCALE-Sim v3 与 Timeloop/Accelergy 的交叉结果，回答三个阶段 1 相关问题：长上下文下片外访存占比、decode 利用率跌幅，以及 16 MiB SRAM 能容纳多长的 KV tile。图与表来自 `outputs/`；完整偏差说明见 `outputs/cross_validation.md`。

## 1. 方法与假设

| 工具 | 回答什么 | 主要简化 |
|---|---|---|
| Roofline (`roofline.py`) | 算术强度 AI、compute/memory bound、理想 latency | A/B 各读一次、C 写一次；完美带宽重叠 |
| SCALE-Sim (`scale-sim/`) | cycle、PE utilization、SRAM/DRAM traffic | 每维 ≤256 的固定 tile × 重复次数；无跨 tile 复用 |
| Timeloop/Accelergy (`timeloop/`) | MAC / register / SRAM / DRAM energy 与 area | 与 SCALE-Sim 相同 tile；45 nm PAT，能量份额模型相关 |

GEMM 拆解：`QKV_proj` → `QK^T` → `PV` → `O_proj`。Prefill 用方阵/大矩阵；decode 用 `1×n` 瘦矩阵（`QK^T`/`PV` 按 head 乘以 multiplicity）。

**验收口径**：三方绝对值不必对齐；看相对结论——decode 是否 memory-bound、利用率是否显著低于 prefill。

## 2. Roofline：decode 落在 memory roof 下

脊点

$$
\mathrm{AI}_{\mathrm{ridge}} = \frac{128\ \mathrm{TOPS}}{1\ \mathrm{TB/s}} = 128\ \mathrm{ops/byte}
$$

代表性 AI（INT8，一次读写假设）：

| mode | seq | gemm | AI (ops/byte) | bound |
|---|---:|---|---:|---|
| prefill | 4K | QK_T / PV | ≈248 | compute |
| decode | 4K | QK_T / PV | ≈50.9 | memory |
| decode | 4K | QKV / O proj | ≈2.0 | memory |

Decode 投影几乎是读权重写结果（AI≈2），远低于脊点；`QK^T`/`PV` 虽因 KV 复用略高，仍低于 128。Prefill 同算子 AI 更高，落在 compute 侧。层汇总亦如此：decode 4K/32K/128K 均为 memory-bound；prefill 均为 compute-bound。

图：`outputs/roofline_points.png`（WS attained TOPS vs AI；另标 32×32@1 GHz 阵列峰 ≈2.05 TOPS，勿与 128 TOPS 系统峰混读）。

## 3. SCALE-Sim：decode 利用率掉到约 1%–2.5%

### 3.1 Prefill vs decode utilization

对 `QK^T`/`PV`（WS/OS，三档序列长度）：

| dataflow | prefill util | decode util | 比值 |
|---|---:|---:|---:|
| WS | ≈73.1% | ≈1.05% | ≈69× |
| OS | ≈67%–81% | ≈2.1%–2.5% | ≈32× |

图：`outputs/util_prefill_vs_decode.png`。

**结论（问题 2）**：在本配置下，decode PE 利用率掉到约 **1%（WS）/ 2%–2.5%（OS）**，相对 prefill 低 **一到两个数量级**。根因是 decode 保留单行 query（`M=1` 级瘦矩阵），阵列空间维难以铺满；OS/WS 改变绝对 util，但不改变 decode ≪ prefill。

因固定 tile，**单 tile 利用率与序列长度无关**；序列变长主要增加 tile 次数与总 traffic/cycles。

### 3.2 长上下文下片外访存占比

以 WS 全层 traffic（SRAM + DRAM words）计 DRAM 占比：

| mode | seq | DRAM / (SRAM+DRAM) |
|---|---:|---:|
| decode | 4K | ≈49.4% |
| decode | 32K | ≈49.4% |
| decode | 128K | ≈49.4% |
| prefill | 4K | ≈37.4% |
| prefill | 128K | ≈38.1% |

图：`outputs/traffic_energy_stack.png`（左）。

**结论（问题 1）**：在本 SCALE-Sim 分层与 tile 假设下，decode 片外（DRAM）访存约占 **总 words 流量的一半（≈49%）**；prefill 约 **37%–38%**。decode 的 DRAM 与 SRAM 流量量级接近，且随 $S$ 近似线性放大——长上下文放大的是 **绝对片外流量**，而不只是占比。注意：这是 simulator traffic，不是端到端 HBM 实测；tile 重复会重复计跨 tile 本可复用的数据。

## 4. Timeloop：能量分解与模型边界

层能量占比（PAT，45 nm）：

| mode | seq | MAC | SRAM | DRAM |
|---|---:|---:|---:|---:|
| decode | 4K–128K | ≈0.03% | ≈89.2% | ≈10.7% |
| prefill | 4K–128K | ≈0.06% | ≈99.3% | ≈0.3%–0.4% |

图：`outputs/traffic_energy_stack.png`（右）。Area：GlobalBuffer（16 MiB）≈100 mm²（CACTI），占主导。

**解读**：当前工具链下 **片上大 SRAM 动态能量主导**，不能据此宣称「DRAM energy 主导」。与 SCALE-Sim/Roofline 一致的稳健结论是：**带宽压力 + decode 低利用率**。绝对能量份额需工艺与 memory 模型标定后再用于论文。

## 5. 16 MiB SRAM 能放多长的 KV tile？

粗算容量（不含索引/双缓冲/权重）：

$$
\mathrm{Bytes}_{\mathrm{KV}}(S) = 2 \cdot H \cdot S \cdot D \cdot b
$$

其中 $H=32$，$D=128$，$b$ 为元素字节数；$S_{\mathrm{tile}}$ 为可驻留的序列维长度。

$$
S_{\mathrm{tile}} = \frac{C_{\mathrm{SRAM}}}{2 \cdot H \cdot D \cdot b}
$$

取 $C_{\mathrm{SRAM}}=16\ \mathrm{MiB}=2^{24}$ B：

| 精度 $b$ | 内容 | $S_{\mathrm{tile}}$（token） |
|---:|---|---:|
| 1 (INT8) | K+V | $2^{24}/(2\cdot32\cdot128)=2048$ |
| 2 (FP16) | K+V | $1024$ |
| 1 (INT8) | 仅 K 或仅 V | $4096$ |
| 1 (INT8) | 单 head 的 K+V | $65536$ |

**结论（问题 3）**：若把 16 MiB **全部**留给一层的 K+V（INT8），约能驻留 **2K token** 上下文；FP16 约 **1K**。相对 32K/128K 长上下文，片上只能容纳 **一个较短的 KV tile**，其余必须走片外或更激进的压缩/量化（衔接 P2）。FlashAttention 风格分块时，$S_{\mathrm{tile}}\sim10^3$ 量级是 16 MiB 量级缓冲的合理工作点；再大则依赖 HBM 与调度重叠。

实际可用 $S_{\mathrm{tile}}$ 更小：还需放 Q/O 激活、权重 tile、双缓冲与 bank 碎片。上表是上界粗算。

## 6. 三方交叉：一致处与偏差

**一致（相对）**

1. Decode `QK^T`/`PV`/投影为 memory-bound 或极低 util；prefill 同算子高 util / 高 AI。
2. 序列变长 → traffic 与 energy 上升；decode 投影形状不随 $S$ 变，attention GEMM 随 $S$ 变。
3. Dataflow 调绝对 util，不消除 decode 缺口。

**偏差（预期）**

| 来源 | 说明 |
|---|---|
| Roofline vs SCALE-Sim | 理想屋顶 vs 阵列映射/stall；128 TOPS 系统峰 ≠ 32×32@1 GHz≈2 TOPS |
| SCALE-Sim tile 重复 | 无跨 tile 复用 → traffic/cycle 偏保守（偏大） |
| Timeloop PAT | 大 SRAM 能量占比高 → 勿直接当「DRAM 能耗结论」 |

详见 `outputs/cross_validation.md`。

## 7. 对阶段 1 / 后续工作的含义

1. **瓶颈定位**：长上下文 inference 的关键矛盾是 decode 侧 **低算术强度 + 低阵列利用率 + KV 流量随 $S$ 增长**；单纯加大 PE 阵列收益有限，除非同时改善数据复用与带宽。
2. **SRAM 规划**：16 MiB 量级更适合 **KV tile + 工作缓冲**，而非整段 128K KV；需与分块 attention（P1/P5）和 KV 量化/旋转（P2）协同。
3. **架构方向**：FlashAttention-native 数据流、片上 KV tile 调度、以及 decode 友好的映射（避免 $M=1$ 饿死阵列）是阶段 1–2 的直接动机。
4. **工具链**：本仓库 `outputs/` 命名稳定，可供 P5 tile simulator 做 **趋势** 对照（util、traffic 随 $S$），而非绝对值对拍。

## 8. 可复现命令

```bash
conda activate p3-arch-eval
python learning/p3_arch_eval/roofline.py
python learning/p3_arch_eval/scale-sim/run_scalesim.py
python learning/p3_arch_eval/timeloop/run_timeloop.py   # 需 Docker 镜像
python learning/p3_arch_eval/collect_results.py
```

## 9. 一句话结论

在 128 TOPS / 1 TB/s / 16 MiB 假设与 32×32 仿真下：**decode 落在 memory roof 下，PE 利用率仅约 1%–2.5%（较 prefill 低数十倍）；片外 traffic 约占 decode 总流量一半且随上下文增长；16 MiB SRAM 大约只能驻留 INT8 K+V 的 ~2K token tile。** 这三条共同支撑「长上下文 decode 是 memory-bound / 带宽与数据编排问题」的阶段 1 叙事。
