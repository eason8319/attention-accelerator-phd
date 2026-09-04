# R1 协议：指标、流量记账与报告分层

> **状态**：已锁定（2026-07-24；2026-09-04 增补 §8 paged 切分并随 `paged_cache.py` 落地）。  
> **版本**：v1.1  
> 模型与上下文：[`models_context.md`](models_context.md)

---

## 1. 报告分层（禁止跨层偷换）

| 层 | 指标 | 允许说法 |
|----|------|----------|
| 算法 | PPL；LM-Eval / LongBench 任务分；相对 FP16 的 $\Delta$；cosine / 相对 $\ell_2$（中间张量） | 「相对 FP16 PPL 退化 …」 |
| 流量 | HBM **bytes/token**（含元数据）；payload / scale / zp / page 分解 | 「相对 FP16 KV 流量 $\downarrow$ …」 |
| 架构模拟 | latency/token（cycle 模型）、带宽与 PE 利用率、energy/token（相对模型） | 「在锁定硬件包络下相对 …」 |
| 墙钟（可选） | GPU tokens/s、峰值显存 | 必须写明 GPU 型号与软件栈 |

禁止：把解析模型能量写成芯片实测；把阶段 A 短序列写成长上下文 SOTA。

---

## 2. 精度指标

### 2.1 开发与功能（阶段 A / 任意阶段单测）

- Attention 输出相对 FP16 golden：cosine similarity、相对 $\ell_2$
- Cache round-trip：quantize→pack→store→load→dequant 后与直通 FP16 的误差上界（实现时在测试中钉数值阈值）

### 2.2 困惑度

- WikiText-2（或协议注明的子集）：报告 FP16 与各 KV 格式 PPL
- 主叙事以 **真实 cache-path** 为准；若保留投影假量化对照，须单独标注，不得合并进主表

### 2.3 任务精度（阶段 B）

| 套件 | 任务 | 用途 |
|------|------|------|
| LM-Eval | CoQA、TruthfulQA、GSM8K | KIVI Table 3 对齐 |
| LongBench | 四子组各 ≥1 代表（如 Qasper；QMSum 或 MultiNews；TREC/TriviaQA/SAMSum 之一；LCC 或 RepoBench-P） | 与 KIVI 长上下文口径对齐 |

报告时附：官方/论文数字、本仓库数字、差值、可能原因。

---

## 3. Bytes/token 记账

### 3.1 定义

对 **单层、单 decode step、batch=1**，读取已有长度 $N$ 的 KV（生成第 $N{+}1$ 个 token 时）的片外流量估计：

$$
\mathrm{Bytes}_{\mathrm{step}}
= B_{\mathrm{payload}} + B_{\mathrm{scale}} + B_{\mathrm{zp}} + B_{\mathrm{page}}
$$

$$
\mathrm{bytes/token}
\;\triangleq\;
\frac{\mathrm{Bytes}_{\mathrm{step}}}{1}
\quad\text{（逐步）；长序列可报平均或对 } D(L_{\mathrm{in}},L_{\mathrm{out}}) \text{ 全程积分后再除以 } L_{\mathrm{out}}
$$

压力点 $D(L_{\mathrm{in}},L_{\mathrm{out}})$ 须同时报告：

1. 全程 KV 读流量合计 / $L_{\mathrm{out}}$  
2. 末步（$N=L_{\mathrm{in}}+L_{\mathrm{out}}-1$）单步 bytes/token  

### 3.2 分项约定

| 分项 | 计入内容 |
|------|----------|
| $B_{\mathrm{payload}}$ | 量化后的 K/V 载荷（INT4 按 0.5 byte/元素等）；KIVI 残差窗按 FP16 计入此项 |
| $B_{\mathrm{scale}}$ | 各 group / channel / token 的 scale（默认 FP16=2 B，除非实现另定并文档化） |
| $B_{\mathrm{zp}}$ | zero-point（若对称量化则为 0） |
| $B_{\mathrm{page}}$ | paged 布局的页表项；contiguous 记 0。默认公式与页几何见 §8.5；禁止把 scale/zp 或载荷塞进本项 |

多头、GQA 按实际 K/V head 数展开。RoPE 旋转矩阵本身若预计算常驻片上可不计入 HBM；**若每次从 HBM 加载则必须计入并说明**。

### 3.3 有效比特

名义 INT4 不等于有效比特。报告中可附：

$$
b_{\mathrm{eff}}
=
\frac{8\cdot (B_{\mathrm{payload}}+B_{\mathrm{scale}}+B_{\mathrm{zp}}+B_{\mathrm{page}})}
{N\cdot n_{\mathrm{elem}}}
$$

其中 $n_{\mathrm{elem}}$ 为该步读取的标量元素个数（K+V）。

---

## 4. 布局与并行报告字段

每次正式实验表格至少含：

| 字段 | 说明 |
|------|------|
| model | HF ID 或 offline |
| format | C0–C5 |
| layout | contiguous / paged；paged 须满足 §8，并注明 $P_{\mathrm{size}}=16$ |
| $N$ 或 $D(\cdot,\cdot)$ | 上下文或压力点 |
| PPL 或任务分 | 算法层 |
| bytes/token | 含分解列或附录表 |
| stage | A 或 B |

---

## 5. 与模拟器交叉核对（M7 预留）

检查点（趋势一致即可，不要求绝对值相等）：

1. decode 比 prefill 更偏存储（利用率更低或 AI 更低）  
2. 流量随 $N$ 近似线性增长  
3. 降低 $b_{\mathrm{eff}}$ 后，模拟 bytes/token 下降方向与解析模型一致  

工具：本仓库 decode simulator（扩展自 P5）、Roofline（P3）、SCALE-Sim（P3）相对趋势。

---

## 6. 误差—流量敏感性（M6 口径）

至少报告：

- **层**：若干层 PPL 或输出误差贡献  
- **头**：抽样 head 的 score/输出误差  
- **位置**：recent window vs 历史 token  

阶段 A 在 0.5B 上做全网格；阶段 B 在 8B 上对 32K 或 $D(16384,1024)$ 做抽样核验。

---

## 7. 修订规则

与 [`models_context.md`](models_context.md) 相同：改公式默认假设或必报字段 → 版本 +0.1 + CHANGELOG。  
改 §8 的 $P_{\mathrm{size}}$、$B_{\mathrm{pte}}$、切页不变量或 $B_{\mathrm{page}}$ 公式，视为改默认假设。实现若无法遵守，须先修订本文件，禁止在代码里默默换口径。

---

## 8. Paged 布局切分规则

本节约束 **存储布局与流量记账**，不改变 C0–C5 的量化语义。M4 不要求 page-wise partial attention：读侧仍可 `load` 全量再做 SDPA。两条 cache 后端都要能切页；只给均匀 codec 包一层 page **不算**完成 M4。

### 8.1 默认参数与不变量

| 符号 | 默认 | 含义 |
|------|------|------|
| $P_{\mathrm{size}}$ | 16 | 一页的 token 数 |
| $g$ | 32 | KIVI Key 沿 token 维的 group；均匀 INT4 的 group 仍在 channel 维 |
| $R$ | 128 | KIVI `residual_length` |
| $B_{\mathrm{pte}}$ | 8 B | 一条页表项（64-bit 块地址）；K/V（及 KIVI 残差池）分计 |

须保持：

$$
P_{\mathrm{size}} \mid g,\qquad P_{\mathrm{size}} \mid R,\qquad g \mid R
$$

当前默认 $16 \mid 32 \mid 128$。禁止为迁就切页而改 $g$ 或 $R$。

### 8.2 页的几何

- 页沿 **token 维** 切；同一层、同一侧（K 或 V）的全部 KV head 打进同一页。
- 一页逻辑形状：$(P_{\mathrm{size}},\; n_{\mathrm{kv\_heads}},\; d)$。GQA 用实际 K/V head 数，不用 query head 数。
- K 与 V **分池**，各有页表。C0–C3 各一侧一池；C4/C5 见 §8.4（量化历史与残差再分池）。
- 量化载荷与该页的 scale / zp **不得跨页共享**。KIVI Key 的 group 跨两页时，scale / zp 按 group 存一份、挂在该 group 上，不算「跨页共享一组统计量」，也不按页复制。

### 8.3 均匀路径（C0–C3）

适用 `ContiguousKVCache` 的 paged 后端。

- 每满 $P_{\mathrm{size}}$ 个 token 将该页独立 `encode`；未满的尾页见 §8.5。
- Channel 维 group（INT4 / INT4+BDR 的 $g=32$）与 BDR 旋转都在末维 $d$ 上，与 token 切页正交；**禁止**沿 token 维另开量化 group。
- 因此：同一 token 的 scale / zp 只属于它所在页。**同一 append 粒度**下（prefill 整段写入，或两侧都按页 encode），contiguous 与 paged 对 C0–C3 应能 round-trip **逐元素一致**（C3 为 float32 旋转容差，单测钉死）。
- 逐步 decode 时 contiguous 每次 1 token encode、paged 满页才 encode。C0–C2 仍应逐元素一致；C3（INT4+BDR）允许 INT4 差 1 档，**不以该差否定切页规则**，也不写入精度退化。
- C3 旋转矩阵仍视为片上常驻，不进页表、不进 $B_{\mathrm{scale}}$（与 §3.2 一致）。

### 8.4 KIVI 路径（C4/C5）

刷窗规则与现有 `KiviKVCache` 相同，paged 只改已落地张量怎么切块，不改何时量化。

记当前长度 $N$，则四段长度为：

$$
\begin{aligned}
N_{K,\mathrm{quant}} &= \lfloor N/R \rfloor \cdot R, &
N_{K,\mathrm{res}} &= N \bmod R, \\
N_{V,\mathrm{quant}} &= \max(N-R,\,0), &
N_{V,\mathrm{res}} &= \min(N,\,R).
\end{aligned}
$$

| 池 | 内容 | 切页 |
|----|------|------|
| $K$ 量化历史 | per-channel、沿 $T$ 以 $g=32$ 分组 | $N_{K,\mathrm{quant}}$ 个 token；group 起点须为 $g$ 的倍数（因而也是 $P_{\mathrm{size}}$ 的倍数）。一 group = **2 页**；Key 一次 flush $R=128$ token = **8 页** |
| $K$ 残差 | 尚未满窗的 FP16 | $N_{K,\mathrm{res}}$；单独成 FP16 页，不与量化历史混页 |
| $V$ 量化历史 | per-token（channel 维再 $g$ 组） | $N_{V,\mathrm{quant}}$；decode 每步溢出 1 token，允许量化尾页不满 |
| $V$ 残差 | 最近 $\min(N,R)$ 的 FP16 | 窗满时恰好 $R/P_{\mathrm{size}}=8$ 页；同样不与量化历史混页 |

硬约束：

1. **同一页不得混装** FP16 残差与低比特历史。
2. 不得把 Key 的一个 $g$-group 拆到非连续页，也不得从页中途开始一个 group。
3. 不得为凑满页而推迟 Value 溢出量化（那会改变残差窗语义）。未满的量化尾页用 §8.5，而不是改刷窗。
4. 残差页的元素按 FP16 计入 $B_{\mathrm{payload}}$，不另造「残差元数据」项；页表项仍计入 $B_{\mathrm{page}}$。

### 8.5 尾页、页表与 $B_{\mathrm{page}}$

对任一池，已分配页数

$$
P_{\mathrm{pool}} = \lceil N_{\mathrm{pool}} / P_{\mathrm{size}} \rceil
$$

（空池为 0）。尾页只编码真实 token，**不**为对齐而插入 dummy token（数值路径与 contiguous 可比）。

**主口径**（正式双列表的 paged 列）：

$$
B_{\mathrm{page}} = B_{\mathrm{pte}} \sum_{\mathrm{pool}} P_{\mathrm{pool}}
$$

C0–C3 对 K、V 两池求和；C4/C5 对上表四池求和。$B_{\mathrm{payload}}$ / $B_{\mathrm{scale}}$ / $B_{\mathrm{zp}}$ 按**已占用 token** 计，尾页空洞不计入载荷。

**可选附录列** $B_{\mathrm{pad}}$：若实现按整页 DMA 读取，尾页未占用槽位的载荷字节。默认 **不** 并入主 `bytes/token`，以免 M5 主曲线在实现 DMA 策略前被碎片放大。需要并入时须先改本协议版本。

`bytes_stored` 须能拆出 payload / scale / zp / page 四项；contiguous 的 page 恒为 0。M4 报告给出分解即可，完整 Pareto 仍属 M5。

### 8.6 与 contiguous 的关系

| 项 | 预期 |
|----|------|
| C0–C3 精度 | paged 与 contiguous 应对齐到实现容差 |
| C4/C5 精度 | 刷窗不变则应对齐到实现容差 |
| 流量 | paged 的 $B_{\mathrm{page}}>0$；占用 token 的 payload/scale/zp 与 contiguous 同量级 |
| 主声称 | 自 M4 起正式流量表默认双列；禁止只报连续地址上界 |

### 8.7 实现回写

`cache_path/paged_cache.py` 已按本节落地（`PagedUniformKVCache` / `PagedKiviKVCache`）。切页不变量与 $B_{\mathrm{pte}}$ 未改。若日后改 $P_{\mathrm{size}}$、$B_{\mathrm{pte}}$ 或四池定义，先修订本节再改代码。

---

## 修订记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-24 | 锁定指标分层、bytes/token 公式、必报字段 |
| v1.1-draft | 2026-09-04 | 新增 §8 paged 切分（$P_{\mathrm{size}}=16$、均匀 / KIVI 两后端、四池、$B_{\mathrm{pte}}=8\,\mathrm{B}$）；§3.2 / §4 交叉引用 |
| v1.1 | 2026-09-04 | `paged_cache.py` 落地，口径未改；去掉 draft 并锁定 |
| v1.1 | 2026-09-04 | §8.3 补充：逐元素一致绑定同一 append 粒度；C3 逐步 decode 允许 INT4 差 1 档 |
