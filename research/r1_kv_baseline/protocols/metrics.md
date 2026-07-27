# R1 协议：指标、流量记账与报告分层

> **状态**：已锁定（2026-07-24）。  
> **版本**：v1.0  
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
| $B_{\mathrm{payload}}$ | 量化后的 K/V 载荷（INT4 按 0.5 byte/元素等） |
| $B_{\mathrm{scale}}$ | 各 group / channel / token 的 scale（默认 FP16=2 B，除非实现另定并文档化） |
| $B_{\mathrm{zp}}$ | zero-point（若对称量化则为 0） |
| $B_{\mathrm{page}}$ | paged 布局下的页表/块索引等元数据；contiguous 记 0 |

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
| layout | contiguous / paged |
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
