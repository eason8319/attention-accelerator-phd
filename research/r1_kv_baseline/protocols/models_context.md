# R1 协议：模型、上下文、Decode 压力点与阶段分工

> **状态**：已锁定（2026-07-24；2026-07-27 修订路径表述）。后续实验与报告必须引用本文件；若修订，须改版本号并在 CHANGELOG 记一笔。  
> **版本**：v1.1  
> 配套指标口径：[`metrics.md`](metrics.md)

---

## 1. 执行阶段（资源约束）

| 阶段 | 条件 | 允许做什么 | 禁止 |
|------|------|------------|------|
| **A：弱机开发** | 无 GPU 或显存 \<24 GB | 协议、代码骨架、合成张量单测、`cache_path/` 编解码、0.5B 短序列冒烟、解析 bytes/token | 下载/宣称 7B·8B 主结果；把短序列数字写成 SOTA |
| **B：正式评测** | ≥24 GB GPU；主机内存建议 ≥32 GB（理想 ≥64 GB，**不强制 128 GB**） | KIVI 复现、主 Pareto、decode 压力点、LongBench 子组 | 无 |

当前仓库默认处于 **阶段 A**。进入阶段 B 前再下载 7B/8B 权重。

---

## 2. 模型阶梯

| 角色 | 模型 ID | 阶段 | 用途 |
|------|---------|------|------|
| Dev / 冒烟 | `Qwen/Qwen2.5-0.5B-Instruct` | A+B | 单元测试、cache-path 功能正确性、误差敏感性快扫 |
| Offline 烟测 | `offline-tiny-llama`（按需在本目录新建自包含 `offline_utils.py`，不依赖 `learning/`） | A | 无网络时接口冒烟，**不作精度主张** |
| KIVI 表格锚 | `NousResearch/Llama-2-7b-hf`（或等价 ungated Llama-2-7B） | B | 对齐 KIVI Table 3（CoQA / TruthfulQA / GSM8K） |
| LongBench 锚 | `mistralai/Mistral-7B-Instruct-v0.2` | B | LongBench 子组，max length **8192**（对齐 KIVI） |
| 主 Pareto | `meta-llama/Llama-3.1-8B-Instruct` | B | bytes/token–精度主曲线、16K/1K 压力点 |
| Stretch（选做） | Llama-2-13B；Falcon-7B；128K 上下文 | B 且资源允许 | 不阻塞 R1→R2 |

**Batch**：默认 $B=1$。小 batch 对照仅附录，不改变主叙事。

---

## 3. 上下文阶梯

### 3.1 阶段 A（弱机）

| 长度 | 目的 |
|------|------|
| 512 / 1024 / 2048 | cache-path 功能与相对误差冒烟 |

不得将阶段 A 曲线标为「长上下文正式结果」。

### 3.2 阶段 B（正式，必做）

| 长度 | 目的 |
|------|------|
| 4K | 与 KIVI LongBench 常用上限对齐（非 Mistral） |
| 8K | Mistral LongBench 上限；Pareto 第二点 |
| 16K | 衔接 decode 压力点输入侧 |
| 32K | 主长上下文 Pareto 终点（必做） |

| 长度 | 目的 |
|------|------|
| 128K | Stretch；单点验证即可，资源不足则记 Future Work |

---

## 4. Decode 压力点

固定配置，专门报告 **bytes/token** 与 dequant/metadata 开销分解（对齐 SAW-INT4 叙事口径）：

| 字段 | 正式值（阶段 B） | 弱机代理（阶段 A，非正式） |
|------|------------------|----------------------------|
| 输入 tokens | 16384 | 1024 |
| 输出（生成）tokens | 1024 | 128 |
| Batch | 1 | 1 |
| 记名 | `D(16384,1024)` | `D(1024,128)-dev` |

阶段 A 的压力点仅验证记账代码正确性，**不得**与论文/GPU 数字并表冒充。

---

## 5. 对照谱（格式 × 布局）

### 5.1 KV 格式（必做）

| ID | 配置 |
|----|------|
| C0 | FP16 KV |
| C1 | 均匀 token-wise INT8 |
| C2 | 均匀 token-wise INT4 |
| C3 | INT4 + BDR（SAW 思想） |
| C4 | KIVI 风格非对称 2-bit（`group_size=32`，`residual_length=128`） |
| C5 | KIVI 风格非对称 4-bit（同上超参） |

阶段 A 至少跑通 C0–C3 的 cache-path 单元路径；C4/C5 以接口与合成张量为先，完整精度表在阶段 B。

### 5.2 布局（必做双报）

| 布局 | 要求 |
|------|------|
| contiguous | 理想连续地址上界 |
| paged | 规则 page/block（默认建议 16 tokens/page，实现时可在 metrics 中确认） |

主声称不得只依赖 contiguous。

### 5.3 KIVI 复现专用超参（阶段 B）

与公开 KIVI 实现对齐：

- `group_size = 32`
- `residual_length = 128`
- K：per-channel 分组；V：per-token
- 评测：LM-Eval CoQA / TruthfulQA / GSM8K；LongBench 四子组各至少一代表任务

---

## 6. 硬件包络（架构记账假设）

以下用于 bytes→时间/能量**相对模型**与 decode simulator，**不是**声称本机已有该 ASIC。与 P3/P5 对齐，保持跨阶段可比：

| 参数 | 默认值 |
|------|--------|
| PE 阵列 | $32\times 32$ @ 1 GHz |
| 片上 SRAM | 16 MiB（逻辑划分可参考 P5：Q/KV/O/stats） |
| 片外带宽 | 1 TB/s HBM 峰值 |
| Workload 形状参考 | LLaMA-7B 量级单层（$H{=}32$，$d{=}128$）作流量趋势；真实精度实验以第 2 节模型为准 |

正式 GPU 墙钟时间可另报，但须标注平台，**禁止**与上述包络下的模拟 joule/cycle 混称为硅片实测。

---

## 7. 修订规则

1. 改模型列表、上下文必做点或压力点正式值 → 版本号 +0.1，更新本页日期与 CHANGELOG。  
2. 仅补充 stretch / 失败收缩说明 → 可 +0.0.x，但须在 REPORT 写明。  
3. 阶段 A→B 切换不改协议版本，但在实验日志注明「进入阶段 B」。
