# 低比特 KV / Decode Attention 加速：近年成果对比手册

本手册服务 [`research_plan.md`](research_plan.md)。  
**可更新基础设施**见 [`lit_watch/`](lit_watch/)（检索词、inbox、卡片模板、已核实台账）。  
**Agent 更新本手册时**：默认先遵循 academic-researcher skill（见 `.cursor/rules/lit-watch-academic-researcher.mdc`）。

## 修订记录

| 日期 | 变更 |
|------|------|
| 2026-07-23 | 约定文献更新默认使用 academic-researcher skill（`.cursor/rules/lit-watch-academic-researcher.mdc`）。 |
| 2026-07-23 | 建立 `lit_watch/`；按 arXiv API + PMLR/ACL/DOI **核实**核心条目的题名/Venue/时间；修正 MiniKV 正式题名、综述 ACL’26 Findings、Don’t Waste Bits→CVPR’26（accepted）、Titanus→GLSVLSI’25 等；总览表增加「状态」列。 |
| 2026-07-23 | 初版对比手册（后续以本表为准）。 |

详细核验说明见 [`lit_watch/CHANGELOG.md`](lit_watch/CHANGELOG.md)；机器可读台账见 [`lit_watch/ledger.yaml`](lit_watch/ledger.yaml)。

## 检索截止

- **Cutoff 日期**：2026-07-23  
- **窗口内最新收录**：Jiang et al., *Towards Efficient Large Language Model Serving…*，**ACL 2026 Findings**，[arXiv:2607.08057](https://arxiv.org/abs/2607.08057)，DOI [10.18653/v1/2026.findings-acl.1916](https://doi.org/10.18653/v1/2026.findings-acl.1916)（arXiv 首发 2026-07-09）  
- **使用约定**：数字摘自公开摘要/正文，**不可跨平台直接比绝对倍数**；`状态` 列：`会议/期刊` = 已核实正式 venue；`预印本` = 仅 arXiv（或仅有 submitted/accepted 声明）

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

1. **算法精度**：KIVI、SAW-INT4（+BDR）、KVTuner / Block-GTQ  
2. **GPU 系统**：BitDecoding；（可选）UltraQuant  
3. **硬件**：SystolicAttention、PLENA、AccLLM；稀疏上界 Salca（非主路径）

### 1.3 更新流程（摘要）

完整步骤见 [`lit_watch/README.md`](lit_watch/README.md)。卡片模板：[`lit_watch/CARD_TEMPLATE.md`](lit_watch/CARD_TEMPLATE.md)。

```text
queries → inbox → 核实 venue/DOI → ledger.yaml → 改本手册表/卡片 → 写修订记录
```

---

## 2. 总览表（按平台）

### 2.1 算法 / GPU 系统（KV 压缩与 decode）

| 工作 | Venue / 时间 | 状态 | 来源 | 平台 | 核心做法 | 报告结果（摘要） | 结论要点 |
|------|--------------|------|------|------|----------|------------------|----------|
| KIVI | ICML 2024；PMLR 235:32332–32344 | 会议 | [PMLR](https://proceedings.mlr.press/v235/liu24bz.html)；[arXiv:2402.02750](https://arxiv.org/abs/2402.02750) | GPU | K per-channel、V per-token ≈2-bit；近期高精度窗 | 峰值内存约 $\downarrow 2.6\times$（含权重）；batch 可 $\uparrow 4\times$；吞吐约 $2.35$–$3.47\times$ | 非对称粒度基础范式 |
| BitDecoding | HPCA 2026 | 会议 | DOI [10.1109/HPCA68181.2026.11408481](https://doi.org/10.1109/HPCA68181.2026.11408481)；[arXiv:2503.18773](https://arxiv.org/abs/2503.18773) | Ampere–Blackwell GPU | TC 友好布局 + warp dequant + CUDA/TC 流水；MXFP4 | 相对 FP16 FlashDecoding-v2 平均约 $7.5\times$（MXFP4 最高约 $8.6\times$）；相对 QServe 最高约 $4.3\times$；8B@128K 单 batch decode 延迟约 $\downarrow 3\times$ | 布局+融合决定真实加速 |
| SAW-INT4 | arXiv 2026-04-21 | 预印本 | [arXiv:2604.19157](https://arxiv.org/abs/2604.19157) | $2\times$H100；paged | token-wise INT4 + BDR；融合 rotate–quant | 相对朴素 INT4 近无损；与 plain INT4 吞吐接近；复杂 VQ/Hessian 在服务约束下增益有限 | 可部署性优先于复杂度 |
| MiniKV | ACL 2025 Findings；pp. 18506–18523 | 会议（Findings） | DOI [10.18653/v1/2025.findings-acl.952](https://doi.org/10.18653/v1/2025.findings-acl.952)；[Anthology](https://aclanthology.org/2025.findings-acl.952/)；[arXiv:2411.18077](https://arxiv.org/abs/2411.18077) | GPU + Triton | 2-bit + 自适应保留；与 FlashAttention 兼容内核 | 报告 $>80\%$ KV 压缩并保持长上下文精度；改进延迟/吞吐/显存 | 极限比特需算法–内核共设计 |
| KVTuner | ICML 2025；PMLR 267:36451–36485 | 会议 | [PMLR](https://proceedings.mlr.press/v267/li25dd.html)；[arXiv:2502.04420](https://arxiv.org/abs/2502.04420) | GPU | 层间离线混合精度搜索 | Llama-3.1-8B ≈3.25-bit、Qwen2.5-7B ≈4.0-bit 近无损（数学推理）；相对 KIVI-KV8 最大吞吐约 $\uparrow 21.25\%$ | 混合精度应硬件友好可固化 |
| InnerQ | arXiv（首发 2026-02-26） | 预印本 | [arXiv:2602.23200](https://arxiv.org/abs/2602.23200) | GPU | 内维分组；recent+sink 高精度 | 摘要称相对先前 KV 量化 / 半精度 VMM 有加速（版本间数字有出入，引用时查表） | group 轴对齐 datapath |
| Block-GTQ | arXiv 2026-06-23（题名 *RoPE-Aware Bit Allocation…*） | 预印本 | [arXiv:2606.24033](https://arxiv.org/abs/2606.24033) | H800 等；packed | RoPE 块感知 K 比特；不物化完整 FP16 KV | NIAH / LongBench 大幅回升；K3V3 约 $3.24\times$ 压缩；128K 上可快于 fp16 FA2（论文报告） | 结构感知 + packed 路径 |
| UltraQuant | arXiv 2026-06-18 | 预印本 | [arXiv:2606.20474](https://arxiv.org/abs/2606.20474) | AMD CDNA4；vLLM 语境 | FP4 KV + FP8 Q | 后期轮次 P50 TTFT 约 $\downarrow 3.47\times$；吞吐约 $\uparrow 1.63\times$（相对 FP8 KV） | Agent/多轮压力测试 |
| Don’t Waste Bits! | **CVPR 2026（accepted）**；arXiv 2026-04-06 | 会议（accepted，DOI 待补） | [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) | 端侧小模型 | 动态 $\{2,4,8,\mathrm{FP16}\}$ | SmolLM 上优于静态量化（准确率–延迟） | 动态比特需计入控制开销 |
| KV 服务综述 | **ACL 2026 Findings**；pp. 38450–38476 | 会议（Findings） | DOI [10.18653/v1/2026.findings-acl.1916](https://doi.org/10.18653/v1/2026.findings-acl.1916)；[arXiv:2607.08057](https://arxiv.org/abs/2607.08057) | 文献综合 | 系统感知 KV 优化分类 | 统一粒度/平均比特等比较轴 | Related work 元框架 |

### 2.2 专用硬件 / FPGA / ASIC

| 工作 | Venue / 时间 | 状态 | 来源 | 平台 | 核心做法 | 报告结果（摘要） | 结论要点 |
|------|--------------|------|------|------|----------|------------------|----------|
| SystolicAttention (FSA) | arXiv 首发 2025-07-15 | 预印本 | [arXiv:2507.11331](https://arxiv.org/abs/2507.11331) | $128\times128$；16 nm RTL | 单阵列融合 FlashAttention | 相对 Neuron-v2 / TPUv5e 利用率约 $1.77\times$ / $4.83\times$；面积约 $+12\%$ | FA-native 可行；非系统化低比特 KV |
| PLENA | arXiv 首发 2025-09-11（题名 *Combating the Memory Walls…*） | 预印本 | [arXiv:2509.09505](https://arxiv.org/abs/2509.09505) | 架构模拟 + RTL/ISA 栈 | 扁平阵列 + 非对称量化 + native FA | 同资源设定下相对 A100 吞吐最高约 $2.23\times$、TPU v6e 约 $4.70\times$；能效相对 A100 最高约 $4.04\times$ | 全栈对照，非本课题刀锋 |
| FlatAttention | arXiv 2026-04-02；**submitted to IEEE TC** | 预印本（在投） | [arXiv:2604.02110](https://arxiv.org/abs/2604.02110) | Tile 架构模拟 | tiling + fabric collectives | 利用率最高约 $92\%$；相对 FA3 最高约 $4.1\times$；HBM traffic 最高约 $\downarrow 16\times$（论文报告） | 互连可主导代价 |
| AccLLM | arXiv 首发 2025-04-07 | 预印本 | [arXiv:2505.03745](https://arxiv.org/abs/2505.03745) | Alveo U280 | 剪枝 + Λ-attention + W2A8KV4 | 相对 FlightLLM：能效约 $4.07\times$，吞吐约 $2.98\times$ | FPGA 上 KV4 共设计有效 |
| FlightLLM | **FPGA 2024** | 会议 | DOI [10.1145/3626202.3637562](https://doi.org/10.1145/3626202.3637562)；[arXiv:2401.03868](https://arxiv.org/abs/2401.03868) | U280 / VHK158 | 稀疏 DSP、片上 decode | U280 相对 V100S 能效约 $6.0\times$；VHK158 相对 A100 吞吐约 $1.2\times$ | AccLLM 对照锚 |
| Salca | arXiv 2026-04-27 | 预印本 | [arXiv:2604.24820](https://arxiv.org/abs/2604.24820) | ASIC（稀疏 decode） | 动态稀疏 + 近似 Top-$K$ | 相对 A100 约 $3.82\times$ / 能效约 $74.19\times$ | 相邻上界，非主路径 |
| Titanus | **GLSVLSI 2025** | 会议 | [arXiv:2505.17787](https://arxiv.org/abs/2505.17787)（comment: Accepted to GLSVLSI 2025） | Chiplet + CIM | 在线 prune+quant | 相对 A100/FlightLLM 报告大幅增益（设定依赖 CIM） | 相邻；超出单芯片数字主线 |

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

- **InnerQ**：预印本 [2602.23200](https://arxiv.org/abs/2602.23200)；摘要加速数字版本间有出入 → 引用查表  
- **UltraQuant**：预印本 [2606.20474](https://arxiv.org/abs/2606.20474)  
- **Don’t Waste Bits!**：CVPR 2026 **accepted**（arXiv comment）；proceedings DOI **待补**；[2604.04722](https://arxiv.org/abs/2604.04722)  
- **核实**：2026-07-23；arXiv API  

### 3.8 KV 服务综述（Cutoff 最新）

- **Venue**：ACL 2026 Findings；DOI [10.18653/v1/2026.findings-acl.1916](https://doi.org/10.18653/v1/2026.findings-acl.1916)  
- **arXiv**：2607.08057（2026-07-09）  
- **核实**：2026-07-23；arXiv `journal_ref` + `doi` 字段  

### 3.9 SystolicAttention / PLENA / FlatAttention

- **SystolicAttention**：预印本 [2507.11331](https://arxiv.org/abs/2507.11331)；作者 Jiawei Lin et al.（勿写 “Lin, Yu”）  
- **PLENA**：系统名；论文题 *Combating the Memory Walls…*；预印本 [2509.09505](https://arxiv.org/abs/2509.09505)  
- **FlatAttention**：预印本；comment 标明 submitted to IEEE TC；[2604.02110](https://arxiv.org/abs/2604.02110)  
- **核实**：2026-07-23；arXiv API  

### 3.10 AccLLM / FlightLLM / Salca / Titanus

- **FlightLLM**：FPGA’24；DOI [10.1145/3626202.3637562](https://doi.org/10.1145/3626202.3637562)  
- **AccLLM**：预印本 [2505.03745](https://arxiv.org/abs/2505.03745)（首发日期 2025-04-07）  
- **Salca**：预印本 [2604.24820](https://arxiv.org/abs/2604.24820)；相邻稀疏上界  
- **Titanus**：GLSVLSI 2025（arXiv comment）；[2505.17787](https://arxiv.org/abs/2505.17787)；相邻 CIM/chiplet  
- **核实**：2026-07-23；arXiv API + FlightLLM DOI  

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
