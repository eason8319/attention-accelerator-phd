# 低比特 KV / Decode Attention 加速：近年成果对比手册

本手册服务 [`research_plan.md`](research_plan.md)。  
**可更新基础设施**见 [`lit_watch/`](lit_watch/)（检索词、inbox、卡片模板、已核实台账）。  
**Agent 更新本手册时**：默认先遵循 [academic-researcher skill](../.cursor/skills/academic-researcher/SKILL.md)（规则见 `.cursor/rules/lit-watch-academic-researcher.mdc`）。

## 修订记录

| 日期 | 变更 |
|------|------|
| 2026-09-03 | 将 academic-researcher 迁入仓库 `.cursor/skills/`，去掉云端 `/root/...` 绝对路径。 |
| 2026-09-03 | 对台账全部 21 篇论文完成全文定量复核；新增逐篇审计报告；修正 PLENA→ISCA’26、AccLLM→IEEE TVLSI’26、Don’t Waste Bits→CVPRW’26，并补齐 Titanus DOI。 |
| 2026-09-03 | 增量收录 SPECTRA、AATC、Minima-KV、PuzzleKV；Cutoff 更新至 2026-09-03；Minima-KV 升为 R2–R3 最近直接对照。 |
| 2026-07-23 | 约定文献更新默认使用 academic-researcher skill（`.cursor/rules/lit-watch-academic-researcher.mdc`）。 |
| 2026-07-23 | 建立 `lit_watch/`；按 arXiv API + PMLR/ACL/DOI **核实**核心条目的题名/Venue/时间；修正 MiniKV 正式题名、综述 ACL’26 Findings、Don’t Waste Bits→CVPR’26（accepted）、Titanus→GLSVLSI’25 等；总览表增加「状态」列。 |
| 2026-07-23 | 初版对比手册（后续以本表为准）。 |

逐篇数字、基线、平台与限制见 [`lit_watch/AUDIT_2026-09-03.md`](lit_watch/AUDIT_2026-09-03.md)；元数据变更见 [`lit_watch/CHANGELOG.md`](lit_watch/CHANGELOG.md)；机器可读台账见 [`lit_watch/ledger.yaml`](lit_watch/ledger.yaml)。

## 检索截止

- **Cutoff 日期**：2026-09-03
- **窗口内最新收录**：Wang et al., *PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression*，[arXiv:2608.23843](https://arxiv.org/abs/2608.23843)（预印本；首发 2026-08-24）
- **使用约定**：摘要中的数字全部记录，并由正文表/图补齐口径；**不可跨平台直接比绝对倍数**。`状态` 列：`会议/期刊/Workshop` = 已核实正式 venue；`预印本` = 仅 arXiv（或仅有 submitted 声明）。

---

## 1. 怎么用 / 怎么更新

### 1.1 对比字段

| 字段 | 填写要点 |
|------|----------|
| 平台类别 | GPU 服务 / GPU kernel / FPGA / ASIC 模拟 / ASIC RTL |
| 问题入口 | 仅算法精度 / 布局+内核 / 完整服务 / 硬件 datapath |
| KV 格式 | 均匀 INT4/2-bit、非对称、混合精度、MXFP4 等 |
| 是否物化 FP16 KV | 是 / 否 / 未报告 |
| 是否 paged | 是 / 否 / 未报告 |
| 主指标 | bytes 或压缩比、latency/token 或 TPS、精度、energy |
| 对本课题 | 可对齐点与不可比点 |

### 1.2 默认对照锚点

1. **算法精度**：KIVI、SAW-INT4（+BDR）、KVTuner / Block-GTQ；新近理论参照 AATC / SPECTRA
2. **GPU 系统**：BitDecoding；混合格式最近对照 Minima-KV；（可选）UltraQuant
3. **硬件**：SystolicAttention、PLENA、AccLLM；稀疏上界 Salca（非主路径）

### 1.3 更新流程（摘要）

完整步骤见 [`lit_watch/README.md`](lit_watch/README.md)。卡片模板：[`lit_watch/CARD_TEMPLATE.md`](lit_watch/CARD_TEMPLATE.md)。

```text
queries → inbox → 核实 venue/DOI → ledger.yaml → 改本手册表/卡片 → 写修订记录
```

---

## 2. 总览表（按平台）

### 2.1 算法 / GPU 系统（KV 压缩与 decode）

| 工作 | Venue / 时间 | 状态 | 来源 | 平台 | 核心做法 | 报告结果（全文复核） | 结论要点 |
|------|--------------|------|------|------|----------|------------------|----------|
| KIVI | ICML 2024；PMLR 235:32332–32344 | 会议 | [PMLR](https://proceedings.mlr.press/v235/liu24bz.html)；[arXiv:2402.02750](https://arxiv.org/abs/2402.02750) | GPU | K per-channel、V per-token ≈2-bit；近期高精度窗 | §4.2.4/Fig. 5：A100 80GB、Llama-2-7B、ShareGPT 合成负载下峰值内存约 $\downarrow 2.6\times$（含权重）；batch 最高 $4\times$、吞吐 $2.35$–$3.47\times$ | 最大 batch 系统结果，非单 kernel 固定倍数 |
| BitDecoding | HPCA 2026 | 会议 | DOI [10.1109/HPCA68181.2026.11408481](https://doi.org/10.1109/HPCA68181.2026.11408481)；[arXiv:2503.18773](https://arxiv.org/abs/2503.18773) | Ampere–Blackwell GPU | TC 友好布局 + warp dequant + CUDA/TC 流水；MXFP4 | §VI：相对 FP16 FlashDecoding-v2，Blackwell/Hopper/Ada 最高 $8.6/8.0/7.5\times$；相对 QServe 最高 $4.3\times$；A100、Llama-3.1-8B@128K 单请求端到端约 $3\times$ | “最高/平均”须绑定 GPU 与 shape |
| SAW-INT4 | arXiv 2026-04-21 | 预印本 | [arXiv:2604.19157](https://arxiv.org/abs/2604.19157) | H100；paged | token-wise INT4 + BDR；融合 rotate–quant | Table 3–4/Appendix D：Qwen3-8B BDR-128 均分 69.97 vs BF16 70.84；融合旋转 kernel 与 plain INT4 相差约 $0.6\%$；长上下文 system TPS 对 BF16 为 $+8.4\%$–$41.4\%$ | “近零开销”仅指融合实现与给定服务设置 |
| MiniKV | ACL 2025 Findings；pp. 18506–18523 | 会议（Findings） | DOI [10.18653/v1/2025.findings-acl.952](https://doi.org/10.18653/v1/2025.findings-acl.952)；[Anthology](https://aclanthology.org/2025.findings-acl.952/)；[arXiv:2411.18077](https://arxiv.org/abs/2411.18077) | GPU + Triton | 2-bit + 自适应保留；与 FlashAttention 兼容内核 | Table 1/3–4：摘要称 $>80\%$ KV 压缩；Llama2-7B-chat 平均 34.65 vs FP16 35.19；选择性 kernel 工作区 0.25 vs 1.25GB，但 prefill kernel 0.622 vs 0.118ms | 区分完整 cache 压缩与 kernel 工作区 |
| KVTuner | ICML 2025；PMLR 267:36451–36485 | 会议 | [PMLR](https://proceedings.mlr.press/v267/li25dd.html)；[arXiv:2502.04420](https://arxiv.org/abs/2502.04420) | GPU | 层间离线混合精度搜索 | Table 8：Llama-3.1-8B 3.25-bit 对 KIVI-KV8 提升 $16.79\%$–$21.25\%$；最大值对应 BS=64、input=128（4652 vs 3836 token/s） | 最大值不是全上下文统一收益 |
| InnerQ | arXiv（首发 2026-02-26） | 预印本 | [arXiv:2602.23200](https://arxiv.org/abs/2602.23200) | Jetson Xavier NX 微基准 | 内维分组；recent+sink 高精度 | Table 3–4：有效位宽 3.0–3.5 bit/number；单层 fused dequant-GEMV 平均约 $2.7\times$ vs FP16，32K Hybrid 为 3180µs vs FP16 9516µs、KIVI 4331µs | 非端到端 token latency；Hybrid 假定零点 99% 稀疏 |
| Block-GTQ | arXiv 2026-06-23（题名 *RoPE-Aware Bit Allocation…*） | 预印本 | [arXiv:2606.24033](https://arxiv.org/abs/2606.24033) | H800；packed | RoPE 块感知 K 比特；不物化完整 FP16 KV | Table 12：Qwen2.5-3B、128K 时 K3V3 为 $3.24\times$ KV 压缩、70.96→52.95ms（$1.34\times$）、峰值 56.31→19.85GB；≤64K 反而慢于 FP16 FA2 | 结构感知 + packed 路径；速度有 crossover |
| SPECTRA | arXiv 2026-08-08 | 预印本 | [arXiv:2608.07915](https://arxiv.org/abs/2608.07915) | 算法 / GPU 存储 | 谱变换去相关后把比特集中到高信息通道 | Table 1/Fig. 5–6：Llama LongBench 3.56× 为 53.56 vs FP16 53.24；约 8× 内距 FP16 约 1.5 分；H200 容量验证误差 <1%，但无优化压缩 attention kernel | 证明质量/容量，尚未证明 wall-clock 加速 |
| AATC | arXiv 2026-08-14（题名 *KV Cache Compression Through the Lens of Transform Coding*） | 预印本 | [arXiv:2608.14191](https://arxiv.org/abs/2608.14191) | 算法 | attention-aware distortion 分解 + reverse water-filling 比特分配 | Table II：5.82× 时 18 个评测单元均在 FP16 的 $2\sigma$ 内；Llama KV 1.07GB→184MB；Qwen 32K RULER 0.715 vs FP16 0.720 | 无专用 CUDA；R3–R4 理论参照 |
| Minima-KV | arXiv 2026-08-24 | 预印本 | [arXiv:2608.23834](https://arxiv.org/abs/2608.23834) | Blackwell GPU；paged | recent/anchor FP8 + 历史 packed TQ3；分格式 partial attention + global online-softmax merge；无 dense shadow | Table 1/3/§6：18.3KiB/token（4.58 bit/scalar，$3.50\times$ vs BF16）；direct canary 为 2 个 59,008-token 请求，active-KV $3.625\times$、吞吐比 0.9821、无 dense shadow | **R2–R3 最近直接对照**；canary 仅一对运行且 dense dtype 未说明 |
| PuzzleKV | arXiv 2026-08-24 | 预印本 | [arXiv:2608.23843](https://arxiv.org/abs/2608.23843) | GH200；batch 1 prototype | completed page 独立低秩分解；直接在 dense / factorized pages 上计算 | Table 4–5：稳态 KV 为 raw 的 58.76%；16K TPOT 仅 $+0.18\%$，但 32K TTFT $+20.9\%$；Llama 16/32K RULER 保留 FP16 的 96.2%/96.4% | 区分稳态压缩与 prefill/转换峰值 |
| UltraQuant | arXiv 2026-06-18 | 预印本 | [arXiv:2606.20474](https://arxiv.org/abs/2606.20474) | AMD MI355X；TP=2 | FP4 KV + FP8 Q | Table 1：相对 FP8 KV，晚期轮次 P50 TTFT $3.47\times$、全轮次 $2.3\times$、output throughput $1.63\times$；warm rounds 仅 $0.86\times$（FP8 更快） | 收益主要来自 cache residency |
| Don’t Waste Bits! | **CVPR 2026 Workshops（LoViF）**；pp. 4957–4966 | Workshop | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/html/Boroujeni_Dont_Waste_Bits_Adaptive_KV-Cache_Quantization_for_Lightweight_On-Device_LLMs_CVPRW_2026_paper.html)；[arXiv:2604.04722](https://arxiv.org/abs/2604.04722) | 端侧小模型 | 动态 $\{2,4,8,\mathrm{FP16}\}$ | SmolLM-360M/HellaSwag：相对静态 KV 量化，ms/token $-17.75\%$、准确率 $+7.60$ points，距 FP16 0.30 points | 已正式发表，但不是 CVPR main；动态控制开销需计入 |
| KV 服务综述 | **ACL 2026 Findings**；pp. 38450–38476 | 会议（Findings） | DOI [10.18653/v1/2026.findings-acl.1916](https://doi.org/10.18653/v1/2026.findings-acl.1916)；[arXiv:2607.08057](https://arxiv.org/abs/2607.08057) | 文献综合 | 系统感知 KV 优化分类 | 统一粒度/平均比特等比较轴 | Related work 元框架 |

### 2.2 专用硬件 / FPGA / ASIC

| 工作 | Venue / 时间 | 状态 | 来源 | 平台 | 核心做法 | 报告结果（全文复核） | 结论要点 |
|------|--------------|------|------|------|----------|------------------|----------|
| SystolicAttention (FSA) | arXiv 首发 2025-07-15 | 预印本 | [arXiv:2507.11331](https://arxiv.org/abs/2507.11331) | $128\times128$；16 nm RTL | 单阵列融合 FlashAttention | Fig. 15/Table 4：利用率倍数为 $1.77\times/4.83\times$，附加面积占总面积 12.07%；但摘要与 §6.1 对 TPU/Neuron 的对应顺序冲突 | 数字映射待作者勘误；不可无条件引用 |
| PLENA | **ISCA 2026** | 会议 | DOI [10.1109/ISCA66397.2026.00023](https://doi.org/10.1109/ISCA66397.2026.00023)；[arXiv:2509.09505](https://arxiv.org/abs/2509.09505) | 架构模拟 + RTL/ISA 栈 | 扁平阵列 + 非对称量化 + native FA | Table VIII：同 multiplier/HBM 设定下最高 TPS 为 A100 的 $2.23\times$、TPUv6e 的 $4.70\times$；相对 A100 最高 $4.04\times$ Token/J | 已由预印本正式发表；旧摘要数字已过时 |
| FlatAttention | arXiv 2026-04-02；**submitted to IEEE TC** | 预印本（在投） | [arXiv:2604.02110](https://arxiv.org/abs/2604.02110) | Tile 架构模拟/RTL 校准 | tiling + fabric collectives | Fig. 9/13：32×32 tile、S=4096 为 92.3% utilization；同模拟 tile 对 FA3 最高 $4.1\times$、HBM traffic $\downarrow16\times$；64-chip 模型对 FlashMLA 最高系统吞吐 $2.1\times$ | 非实测硅片；短序列利用率下降 |
| AccLLM | **IEEE TVLSI 34(4), 2026；pp. 1217–1227** | 期刊 | DOI [10.1109/TVLSI.2026.3658524](https://doi.org/10.1109/TVLSI.2026.3658524)；[arXiv:2505.03745](https://arxiv.org/abs/2505.03745) | Alveo U280 | 剪枝 + Λ-attention + W2A8KV4 | Table VII：164 token/s、33W、4.96 token/J；相对同 U280 FlightLLM 为 $2.98\times$ throughput、$4.07\times$ energy efficiency | 已由预印本升级为期刊 |
| FlightLLM | **FPGA 2024** | 会议 | DOI [10.1145/3626202.3637562](https://doi.org/10.1145/3626202.3637562)；[arXiv:2401.03868](https://arxiv.org/abs/2401.03868) | U280 / VHK158 | 稀疏 DSP、片上 decode | Fig. 13–15：batch=1 时 U280 相对 V100S 能效最高约 $6.0\times$、成本效率约 $1.8\times$；VHK158 对 A100 吞吐约 $1.2\times$ | 区分 naive/optimized GPU 基线 |
| Salca | arXiv 2026-04-27 | 预印本 | [arXiv:2604.24820](https://arxiv.org/abs/2604.24820) | 28nm RTL 综合（稀疏 decode） | 动态稀疏 + 近似 Top-$K$ | §5.2/Table 5–6：6.4mm²、0.933W；相对 A100 $3.82\times$ speed、$74.19\times$ energy efficiency | 非流片、跨平台；相邻上界 |
| Titanus | **GLSVLSI 2025；pp. 71–77** | 会议 | DOI [10.1145/3716368.3735145](https://doi.org/10.1145/3716368.3735145)；[arXiv:2505.17787](https://arxiv.org/abs/2505.17787) | Chiplet + CIM | 在线 prune+quant | Fig. 14：相对 A100 为 $159.9\times$ energy / $49.6\times$ throughput；相对 FlightLLM 为 $34.8\times/29.2\times$ | 数量级依赖 CIM/跨平台设定，仅作相邻参照 |

### 2.3 本仓库学习结果（内部基线，非论文 SOTA）

来源：`learning/`（约 128 TOPS / 1 TB/s / 16 MiB；LLaMA-7B 量级层）。

| 项目 | 平台 | 结果 | 用途 |
|------|------|------|------|
| Roofline | 解析 | Decode $QK^\top/PV$ AI $\approx 50.9$；Prefill $\approx 248$ | decode memory-bound |
| SCALE-Sim | $32\times32$ | Prefill util ≈ $73\%$ → Decode ≈ $1\%$ | skinny GEMM 失效 |
| 容量 | 解析 | 16 MiB ≈ $2\mathrm{K}$ INT8 token 层内 $K{+}V$ | 需 tiling+压缩 |
| P2 INT4+BDR | Qwen2.5-0.5B **proxy** | Key rel-$\ell_2$ 0.131→0.076；PPL 3.23→1.93（fp16≈1.68） | 动机；不可替代真实 cache-path |

---

## 3. 分篇卡片（已核实元数据）

### 3.1 KIVI

- **正式题名**：KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache  
- **作者**：Zirui Liu, Jiayi Yuan, Hongye Jin, et al.  
- **Venue**：ICML 2024；PMLR 235:32332–32344  
- **链接**：[PMLR](https://proceedings.mlr.press/v235/liu24bz.html)；[arXiv:2402.02750](https://arxiv.org/abs/2402.02750)；[代码](https://github.com/jy-yuan/KIVI)  
- **平台 / 方法 / 结果 / 结论**：见总览表；非对称 2-bit KV 经典锚  
- **对本课题**：R1 必须真实 cache-path，禁止 proxy 冒充  
- **核实**：2026-07-23；PMLR 页  

### 3.2 BitDecoding

- **Venue**：HPCA 2026；DOI [10.1109/HPCA68181.2026.11408481](https://doi.org/10.1109/HPCA68181.2026.11408481)  
- **arXiv**：2503.18773（首发 2025-03-24）  
- **代码**：[OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding)  
- **对本课题**：ASIC 对齐「packed 布局 + fused dequant」  
- **核实**：2026-07-23；IEEE DOI + GitHub citation  

### 3.3 SAW-INT4

- **状态**：预印本（2026-04-21）；尚无 DOI  
- **链接**：[arXiv:2604.19157](https://arxiv.org/abs/2604.19157)  
- **对本课题**：静态 INT4+BDR 默认可部署点  
- **核实**：2026-07-23；arXiv API（无 journal_ref）  

### 3.4 MiniKV

- **正式题名**（ACL）：*…via Compression and System Co-Design for Efficient Long Context Inference*  
- **arXiv 题名不同**：*…via 2-Bit Layer-Discriminative KV Cache*（引用正式题名）  
- **Venue**：ACL 2025 Findings，pp. 18506–18523；DOI [10.18653/v1/2025.findings-acl.952](https://doi.org/10.18653/v1/2025.findings-acl.952)  
- **核实**：2026-07-23；ACL Anthology  

### 3.5 KVTuner

- **作者**：Xing Li, Zeyu Xing, Yiming Li, et al.  
- **Venue**：ICML 2025；PMLR 267:36451–36485  
- **链接**：[PMLR](https://proceedings.mlr.press/v267/li25dd.html)；[代码](https://github.com/cmd2001/KVTuner)  
- **核实**：2026-07-23；PMLR 页  

### 3.6 Block-GTQ（方法名；论文题为 RoPE-Aware Bit Allocation…）

- **状态**：预印本 2026-06-23  
- **链接**：[arXiv:2606.24033](https://arxiv.org/abs/2606.24033)；[代码声明](https://github.com/JIA-Lab-research/blockgtq)  
- **对本课题**：R3–R5 精度前沿对标；多速率比特需规则化打包  
- **核实**：2026-07-23；arXiv API  

### 3.7 InnerQ / UltraQuant / Don’t Waste Bits!

- **InnerQ**：预印本 [2602.23200](https://arxiv.org/abs/2602.23200)；正文 Table 3–4 已复核，注意其延迟是 Jetson Xavier NX 上的单层 fused dequant-GEMV 微基准
- **UltraQuant**：预印本 [2606.20474](https://arxiv.org/abs/2606.20474)  
- **Don’t Waste Bits!**：CVPR 2026 Workshops（LoViF），pp. 4957–4966；[CVF 正式页](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/html/Boroujeni_Dont_Waste_Bits_Adaptive_KV-Cache_Quantization_for_Lightweight_On-Device_LLMs_CVPRW_2026_paper.html)
- **核实**：2026-09-03；最新全文与正式会刊页

### 3.8 KV 服务综述

- **Venue**：ACL 2026 Findings；DOI [10.18653/v1/2026.findings-acl.1916](https://doi.org/10.18653/v1/2026.findings-acl.1916)  
- **arXiv**：2607.08057（2026-07-09）  
- **核实**：2026-09-03；ACL Anthology 正式页

### 3.9 SystolicAttention / PLENA / FlatAttention

- **SystolicAttention**：预印本 [2507.11331](https://arxiv.org/abs/2507.11331)；作者 Jiawei Lin et al.（勿写 “Lin, Yu”）  
- **PLENA**：系统名；论文题 *Combating the Memory Walls…*；ISCA 2026，DOI [10.1109/ISCA66397.2026.00023](https://doi.org/10.1109/ISCA66397.2026.00023)
- **FlatAttention**：预印本；comment 标明 submitted to IEEE TC；[2604.02110](https://arxiv.org/abs/2604.02110)  
- **核实**：2026-09-03；arXiv 全文 + ISCA 官方/机构存档

### 3.10 AccLLM / FlightLLM / Salca / Titanus

- **FlightLLM**：FPGA’24；DOI [10.1145/3626202.3637562](https://doi.org/10.1145/3626202.3637562)  
- **AccLLM**：IEEE TVLSI 34(4), 2026，pp. 1217–1227；DOI [10.1109/TVLSI.2026.3658524](https://doi.org/10.1109/TVLSI.2026.3658524)
- **Salca**：预印本 [2604.24820](https://arxiv.org/abs/2604.24820)；相邻稀疏上界  
- **Titanus**：GLSVLSI 2025，pp. 71–77；DOI [10.1145/3716368.3735145](https://doi.org/10.1145/3716368.3735145)；相邻 CIM/chiplet
- **核实**：2026-09-03；正式 DOI/会刊信息 + 最新全文

### 3.11 2026 年 8 月新增：SPECTRA / AATC / Minima-KV / PuzzleKV

- **SPECTRA**：预印本 [2608.07915](https://arxiv.org/abs/2608.07915)；Jiamu Zhang, Liang Wu, Kelly Wan, Hanjie Chen, Liangjie Hong。通过谱变换去相关，再将比特集中到高信息通道。**学习重点**：变换域为何出现可分配结构；在线变换、元数据和规则打包是否抵消压缩收益。对齐 R3。
- **AATC**：预印本 [2608.14191](https://arxiv.org/abs/2608.14191)；Hannah Laus, Claudio Mayrink Verdun, Hao Wang, Flavio du Pin Calmon, Felix Krahmer。建立 attention-aware distortion 分解并用 reverse water-filling 分配比特。**学习重点**：如何把 KV 张量误差改写为 attention 输出误差，以及如何加入硬件代价项。对齐 R3–R4。
- **Minima-KV**：预印本 [2608.23834](https://arxiv.org/abs/2608.23834)；Sergii Kozyrev, Davyd Maiboroda。FP8 recent/anchor pages 与 packed TQ3 历史页共存，分格式计算 partial attention state，再用全局 online-softmax 合并；摘要明确声明无 cache-sized dense shadow。**学习优先级最高**：它直接收窄 R2–R3 的新颖性空间，需逐项对照 paged layout、异构格式、partial $O$、格式转换与吞吐。
- **PuzzleKV**：预印本 [2608.23843](https://arxiv.org/abs/2608.23843)；Zizhong Wang, Jieying Wang, Zhao Zhang, Jiajia Li。以 completed page 为独立低秩单元，并直接在 dense / factorized pages 上完成 attention。**学习重点**：page 粒度的增量压缩、直接计算和与量化组合；作为相邻路线，不替代低比特主线。
- **核实边界**：4 篇截至 2026-09-03 均为预印本；定量结果已逐表复核，具体基线、平台和限制见 [`lit_watch/AUDIT_2026-09-03.md`](lit_watch/AUDIT_2026-09-03.md)。

---

## 4. 跨工作对比维度

| 维度 | 算法常见 | GPU 系统常见 | ASIC/FPGA 常见 | 本课题应报告 |
|------|----------|--------------|---------------|--------------|
| 精度 | PPL / 任务 | + 服务负载 | 有时较弱 | 真实 cache-path |
| 流量 | 名义比特 | HBM bytes、峰显存 | 常缺 | bytes/token（含元数据） |
| 速度 | 少 | TPS / TTFT | 相对 GPU 倍 | latency/token + 利用率 |
| 能量 | 少 | 有时 | TOPS/W、Token/J | 模拟 + RTL PPA |
| 布局 | 常忽略 | paged / TC | dataflow | packed+paged |
| FP16 物化 | 常隐式 | 开始强调避免 | 少 | 默认不物化完整 FP16 KV |

---

## 5. 建议实验对照矩阵

| ID | 配置 | 目的 |
|----|------|------|
| C0 | FP16 KV + FlashDecoding 类 | 上界 |
| C1 | 朴素 token-wise INT4 | 退化对照 |
| C2 | INT4 + BDR（SAW 思想） | 可部署静态锚 |
| C3 | KIVI 风格非对称 | 经典算法锚 |
| C4 | 层间混合精度（KVTuner 思想） | 混合精度 |
| C5 | 可规则化 RoPE 块混合比特（Block-GTQ 思想） | 精度前沿 |
| H0 | 先解压 FP16 tile 再算 | 负对照 |
| H1 | 本课题流式 fused-dequant | 主主张 |
| G0 | BitDecoding 公开数字（声明平台差） | GPU 参照 |

---

## 6. 维护约定

1. 新文先入 [`lit_watch/inbox.md`](lit_watch/inbox.md)，核实后再写本手册与 [`lit_watch/ledger.yaml`](lit_watch/ledger.yaml)。  
2. Venue **以会刊/DOI/Anthology/PMLR 为准**；仅有 arXiv comment「accepted」时，状态写「accepted，DOI 待补」。  
3. 摘要与正文数字冲突时，卡片注明 `result_source`。  
4. 每次更新：改本页「修订记录」+ [`lit_watch/CHANGELOG.md`](lit_watch/CHANGELOG.md)，并刷新 Cutoff。  
5. 交叉引用：`survey/`、`learning/`、`research_plan.md`。
