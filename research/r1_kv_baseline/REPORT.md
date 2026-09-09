# R1 总验收报告：真实 KV Cache-Path 与精度—流量基线

**实验日期**：2026-07-28 至 2026-09-09，包含期间独立批次的历史结果；**整理及验收日期**：2026-09-09（北京时间）。

**完成范围**：R1 已完成协议、真实 cache-path、任务与 PPL 精度、双布局名义流量、敏感性抽样及 attention 模拟器独立趋势对照。本次根据保留的原始结果进行汇总，重新计算关键指标并生成关联图，未重跑 GPU 模型评测。

**证据来源**：本地各实验的原始 JSON/CSV、运行参数和 `experiment.json`。本文只维护跨实验的总验收与 R2 起步依据，各实验的完整结果和方法仍由其唯一报告维护；当前进度见[研究里程碑](../../docs/progress/milestones.md)。

## 1. 实验目的

判断现有证据是否足以从算法与流量基线进入 R2 流式硬件通路研究。需要回答：真实缓存读写与编码误差是否可核验；低比特格式在同一长上下文模型上的精度与名义流量如何权衡；分页、残差窗和模拟器映射是否改变结论；进入 R2 所需的四项门槛是否全部满足。

研究范围遵循[长期研究计划](../../docs/research_plan.md)与 R1 协议，四项验收结果见 §6.1。本报告接受在已声明负载和硬件包络下可复现的基线与趋势，不将其扩大为官方 KIVI 表格复现、SOTA 优越性、物理 HBM 测量或芯片性能证明。

## 2. 方法与设置

### 2.1 协议、负载与证据角色

默认模型、上下文和压力点采用[模型协议 v1.2](protocols/models_context.md)，数值归约采用[指标协议 v1.2](protocols/metrics.md)；分页与 KV 流量实验的历史记录遵循其中未改变的 v1.1 字节与分页约定。本文没有更换模型、页大小、残差窗或硬件包络。

| 证据角色 | 已完成负载 | 正式结果入口 |
|---|---|---|
| 真实路径与编码误差 | C0–C5；高斯/outlier；20 个配对种子；prefill 与 256 步 decode | [Codec 对照](experiments/codec_compare/REPORT.md) |
| KIVI 整模任务 | Llama-2-7B 的 CoQA/TruthfulQA/GSM8K；Mistral-7B 的四个 LongBench 代表任务，8K 上限 | [KIVI 整模评估](experiments/kivi_eval/REPORT.md) |
| 布局语义 | C0–C5，6 个长度、5 个种子、prefill/decode，共 360 行 | [Paged 布局](experiments/paged_layout/REPORT.md) |
| 主 Pareto 横轴 | Llama-3.1-8B 几何，4K/8K/16K/32K × 六格式 × 双布局；D(16384,1024) | [KV 流量](experiments/kv_pareto/REPORT.md) |
| 主 Pareto 纵轴 | Llama-3.1-8B-Instruct，完整 WikiText-2 test，四窗口六格式 | [WikiText PPL](experiments/wikitext_ppl/REPORT.md) |
| 敏感性 | Qwen 0.5B 层/头/位置网格与短 decode；8B 32K 几何及四层抽样 | [敏感性实验](experiments/kv_sensitivity/REPORT.md) |
| 架构趋势 | 同一 8B 几何的完整网格、逐步压力轨迹与独立 Roofline/SCALE-Sim | [Attention 模拟器](../r1_decode_sim/experiments/decode_sweep/REPORT.md) |

主 Pareto 的 batch=1，模型为 `meta-llama/Llama-3.1-8B-Instruct`，32 个查询头、8 个 KV 头、head dimension=128、32 层。C0–C5 分别为 FP16、INT8、INT4、INT4+BDR、KIVI-2、KIVI-4；KIVI 的 group size=32、FP16 residual length=128。paged 的 page size=16 token、PTE=8 B。

### 2.2 对照、单位与组合规则

KIVI 任务评估的 FP16 是原生 Hugging Face attention；长窗口精度与流量实验的 C0 是真实路径上的 FP16 codec。两者分别作为各自实验的基准，不能交换。历史 4K PPL 的 `cid="fp16"` 行实际入口为 `kv_format="c0"`，依据入口和 `c0_is_codec=true` 归入 C0，原始记录保持不变。

PPL 为 `exp(nll_sum/n_tokens)`，各窗口每格式计分 288936 个 token，stride=512；窗口间分别使用自己的 C0 计算增量。横轴为相同模型几何、相同窗口长度的一次 decode KV 读流量，含 payload、scale、zero point 和 page，单位 MiB=2²⁰ B，数值为全模型 32 层合计。它是与窗口精度关联的名义成本指标，不是该次 PPL 前向的实测流量。

按模型、窗口、格式关联 24 条 PPL 和 48 条流量记录；每个格式同时列出 contiguous 与 paged 横坐标。精度只在 contiguous 测量，paged 点复用这一精度参照，并以分页布局的合成布局对照解释其适用边界；没有把它记为另外 24 次 paged 精度实验。

### 2.3 运行与复现范围

GPU 精度使用既有实际机器结果：KIVI 任务评估使用 RTX 6000 Ada 48 GB，长窗口 PPL 使用 RTX 4090 24 GB，记录的 PyTorch 为 2.5.1+cu121；完整软件环境以各批次元数据为准。合成张量、几何记账和模拟器使用 CPU。重新执行整模评测仍需相应权重、语料和 GPU，保存源码不代表本机已经满足这些条件。

复现入口、依赖和结果 SHA-256 在各实验 `experiment.json` 中登记。关联图的绘图入口为 [plot_pareto.py](experiments/wikitext_ppl/plot_pareto.py)，读取现有五个精度与流量汇总，使用 CPU 和 Matplotlib；本次为 Python 3.13.13、Matplotlib 3.11.0。它只输出图和来源元数据，并拒绝覆盖非空结果目录：

```bash
python research/r1_kv_baseline/experiments/wikitext_ppl/plot_pareto.py \
  --output research/r1_kv_baseline/experiments/wikitext_ppl/results/ppl_traffic_pareto_reproduction
```

以上命令从项目根目录执行，输出目录应尚不存在或为空。解码模拟器的独立 SCALE-Sim 版本为 3.0.0，依赖和运行命令见其[实验入口说明](../r1_decode_sim/README.md)与[依赖声明](../r1_decode_sim/experiments/decode_sweep/requirements-independent.txt)。

## 3. 实验结果

### 3.1 真实缓存路径与编码误差

R1 在 RoPE 后把 K 和相应 V 写入 cache：写侧保存编码载荷及量化参数，读侧从 cache 解码后参与 attention，decode 时保留并更新历史状态。KIVI 路径还保留残差窗及刷窗语义；双布局分别记录 payload、scale、zero point 和 page 字节。入口见[整模 attention](kivi_repro/llama_kivi_attn.py)和[缓存实现](cache_path/kv_cache.py)。这些结果验证缓存数值语义，INT4/KIVI 的物理 bit-pack 仍不在完成范围内。

合成 outlier 配对结果中，S=256 时 C2/C3 的 attention 相对 L2 约为 0.401/0.233；256 步 decode 末点约为 0.394/0.210，支持 BDR 在该分布下改善误差。原始证据为 [raw_metrics.csv](experiments/codec_compare/results/raw_metrics.csv)和[汇总](experiments/codec_compare/results/summary_mean_std.csv)。C4 同负载误差约 0.5–0.7，不能由下游任务分反推其重建误差很小。

### 3.2 KIVI 相对本仓库 FP16 的任务差值

下表增量单位为百分点，正数表示任务分上升。来源为 [Table 3 汇总](experiments/kivi_eval/results/table3/table3_summary.json)和 [LongBench 汇总](experiments/kivi_eval/results/longbench/longbench_summary.json)，差值由各套件自己的 FP16 分数计算。

| 模型/套件 | 任务及主指标 | C4 KIVI-2 Δ | C5 KIVI-4 Δ |
|---|---|---:|---:|
| Llama-2-7B | CoQA EM | −4.2500 | +0.5166 |
| Llama-2-7B | TruthfulQA `bleu_max` | +0.5998 | −0.3404 |
| Llama-2-7B | GSM8K exact match | −1.8195 | +0.5307 |
| Mistral-7B / LongBench | qasper | −0.50 | +0.15 |
| Mistral-7B / LongBench | qmsum | −0.94 | −0.24 |
| Mistral-7B / LongBench | trec | +1.00 | +0.50 |
| Mistral-7B / LongBench | lcc | −2.03 | −0.16 |

C5 在已测任务上接近本仓库 FP16；C4 的下降随任务变化。LongBench 覆盖四个代表任务的全集，不是完整 LongBench 套件。`reference_json` 为空，且 FP16 与低比特分批运行，因此既不声称复现官方表，也不把小幅正增量解释为统计显著改善。

### 3.3 四窗口精度—流量 Pareto

图中纵轴为同窗口 PPL 增量，横轴为全模型每步名义 KV 读流量，越靠左下对应更少流量和更小 PPL 增量。蓝色实点使用 contiguous 流量，橙色空心环使用 paged 流量；两者共用实测 contiguous PPL。图的横轴范围随窗口改变，页表差值小于标记宽度，没有人为放大。

![Llama-3.1-8B 四窗口双布局流量与 contiguous PPL 参照](experiments/wikitext_ppl/results/ppl_traffic_pareto/ppl_traffic_pareto.png)

图及[运行元数据](experiments/wikitext_ppl/results/ppl_traffic_pareto/run_config.json)为本地派生产物。横轴来源为 [KV 流量汇总](experiments/kv_pareto/results/summary.json)，纵轴来源为 [4K](experiments/wikitext_ppl/results/ppl/L4096/ppl_summary.json)、[8K](experiments/wikitext_ppl/results/ppl/L8192/ppl_summary.json)、[16K](experiments/wikitext_ppl/results/ppl/L16384/ppl_summary.json)、[32K](experiments/wikitext_ppl/results/ppl/L32768/ppl_summary.json) 原始汇总。各组 PPL 和差值均由未舍入数值核算。

下表给出必做长上下文终点 32K 的精确查阅值；最后两列均属于 contiguous 精度证据。

| 格式 | contiguous MiB/步 | paged MiB/步 | contiguous PPL | ΔPPL vs 同窗口 C0 |
|---|---:|---:|---:|---:|
| C0 FP16 | 4096.0 | 4097.0 | 6.316474 | 0.000000 |
| C1 INT8 | 2080.0 | 2081.0 | 6.317325 | +0.000852 |
| C2 INT4 | 1152.0 | 1153.0 | 6.476872 | +0.160398 |
| C3 INT4+BDR | 1152.0 | 1153.0 | 6.433739 | +0.117265 |
| C4 KIVI-2 | 774.5 | 775.5 | 7.756936 | +1.440463 |
| C5 KIVI-4 | 1285.5 | 1286.5 | 6.344187 | +0.027713 |

四窗口中 C5 的 ΔPPL 为 0.026188–0.028587；C3 始终低于同名义流量的 C2，差距为 0.043133–0.072752 PPL。32K 上 C2/C3、C4、C5 的 contiguous 流量分别为 C0 的 28.125%、18.909%、31.384%。这些排序是当前模型和语料上的观察，不包含速度、面积或能耗维度。

### 3.4 分页与压力点

分页布局的合成布局对照中，C0–C2/C4–C5 在测试覆盖内逐元素一致；C3 prefill 差异约 10⁻⁶，逐步 decode 有 1/30 个 C3 用例的 V 差异为 1.81×10⁻³，attention 最大差异仍为 5.66×10⁻⁶，处于既定容差。详见[布局原始行](experiments/paged_layout/results/raw_rows.json)。该结果支持已测合成形状的布局语义，不足以证明任意真实 8B paged 生成质量完全相同。

在 8B 几何、页长整除 N 时，paged 比 contiguous 增加 `32N` B 的全模型页表读取：4K/8K/16K/32K 分别为 0.125/0.250/0.500/1.000 MiB。32K 上该项仅占 C0 payload 的约 0.0244%；这不等于物理分页延迟可以忽略，页指针等待和 DMA 空洞未被计入。

正式 D(16384,1024) 首步 N=16384、末步 N=17407。下表为全模型名义 KV 读流量，单位 MiB/步；均值为全程总量除以 1024，不能用末步替代。

| 格式 | contiguous 全程均值 | paged 全程均值 | contiguous 末步 | paged 末步 |
|---|---:|---:|---:|---:|
| C0 | 2111.937500 | 2112.453339 | 2175.875000 | 2176.406250 |
| C1 | 1072.468262 | 1072.984100 | 1104.936523 | 1105.467773 |
| C2/C3 | 593.982422 | 594.498260 | 611.964844 | 612.496094 |
| C4 | 405.712891 | 406.228729 | 420.925781 | 421.457031 |
| C5 | 668.208984 | 668.724823 | 690.917969 | 691.449219 |

解码模拟器使用 CPU 零值 K/V 调用实际 cache 计数接口，取得 12 组共 12288 条逐步记录，首末步分项和全程总量与 KV 流量实验对齐；原始证据为 [pressure_trace.json](../r1_decode_sim/experiments/decode_sweep/results/pressure_capture/pressure_trace.json)。这是实际采集的名义字节轨迹，没有执行 8B 的 1024-token 自由生成或测量物理 HBM 时间。

### 3.5 模拟器与独立趋势对照

解码模拟器在 32×32 PE、1 GHz、16 MiB SRAM、1 TB/s HBM 下，扫描四窗口、六格式、双布局、三个 KV tile 和 decode/派生 dense prefill，共 288 个候选。224 个可行；64 个因 SRAM 不足不可行，均来自 tile=2048。压力仿真覆盖 36 组、36864 个步骤，32 组全程可行，另外四组为 tile=2048 的 C0/C3 双布局。tile=128/512 的网格和压力配置均可行。

完整范围见 [grid.json](../r1_decode_sim/experiments/decode_sweep/results/full_grid/grid.json)和 [pressure_summary.json](../r1_decode_sim/experiments/decode_sweep/results/full_grid/pressure_summary.json)。59 项内部检查通过；独立验收采用另行运行的 Roofline 与 SCALE-Sim，不能把内部检查代替独立证据。

| 独立检查 | 覆盖与结果 |
|---|---|
| 工作量与流量 | 288 个点与独立解析值一致；224 个可行点和 32 组可行压力轨迹均不低于 Roofline 下界 |
| 上下文与有效比特 | 36 次窗口倍增的字节比为 1.936585–2；40 次低比特/C0 比较均降低 KV 字节 |
| decode/prefill | 48 个组合中 decode 算术强度更低；SCALE-Sim 显示 decode 阵列利用率更低 |
| SCALE-Sim 算子 | 18 个上游形状形成 66 条展开参照；448 次网格算子和 256 次压力算子周期包络检查通过 |
| 压力检查点 | N=16384、16511、16512、17407，覆盖起点、残差边界和末点；不代表上游工具逐一验证了全部 1024 步 |

独立批次[检查记录](../r1_decode_sim/experiments/decode_sweep/results/independent_os/comparison_checks.json)共 1414 项，全部通过；上游 [COMPUTE_REPORT.csv](../r1_decode_sim/experiments/decode_sweep/results/independent_os/scalesim/COMPUTE_REPORT.csv) 等原始输出保留。与本模型相比，上游 QK/PV 组件周期为其 1.030–1.484 倍，支持趋势一致性，不支持绝对值相等。

## 4. 分析与讨论

### 4.1 格式的精度优势不等于硬件优势

只考虑 PPL 与名义流量时，C3 在四窗口均优于同流量的 C2；C5 则以较高流量换取更小 PPL 增量。C4 在 32K 最省流量，但 PPL 增加 1.440463。协议未预先规定可接受的 PPL 退化阈值，因此不能仅凭这些点宣布某一格式普遍最优或不可用。

硬件成本改变了这组权衡。解码模拟器的 contiguous、tile=512、32K 设置下，C0/C2/C3/C5 的 attention 模拟时延分别约为 269.552/277.892/546.327/277.878 ms。C3 的逆旋转计入独立的串行计算周期，名义流量相同不代表代价相同；具体模型与完整数值见 [解码模拟器报告](../r1_decode_sim/experiments/decode_sweep/REPORT.md)。

该硬件包络的 Roofline ridge point 为 2.048 ops/B，16K C0 decode 的算术强度约 3.999 ops/B；默认逐头 `query_rows` 映射又使大量 PE 行空闲。因此“decode 相对 prefill 更依赖带宽”在本实验成立，而“decode 必然由 HBM 主导”不成立。R2 需要同时改善映射利用率和反量化/计算衔接，不能把压缩比例直接当作加速比。

### 4.2 敏感性结果的可迁移范围

敏感性实验的 Qwen 0.5B 全层 C3 短窗 PPL 增量达 167.945010，8B 相同角色的增量为 0.105358；Qwen 只量化第 0 层的现象也没有在 8B 抽样中等幅出现。结合长窗口实验中 C3 优于 C2，可以判断旋转效果依赖模型、分布与路径，不能从单一合成分布外推。

敏感性已保存 218 次 NLL/token 可重算记录，但其中 66 次为 160M 通路检查，不能计为额外的主模型结果。8B 只有四层抽样与单种子合成；层、头、位置的诊断结果用于形成后续假设，不能证明完整长生成的误差累积规律。来源与完整边界见 [敏感性报告](experiments/kv_sensitivity/REPORT.md)。

## 5. 局限与有效性

1. **物理实现**：INT4/KIVI 载荷未做物理 bit-pack，读取时仍物化反量化 K/V；HF cache 接口还保留浮点状态。本实验验证缓存数值路径，不证明实际 PyTorch 存储按名义比特缩小，也未达到 R2 的流式、无完整高精度展开目标。
2. **精度覆盖**：主模型 PPL 只测 contiguous；分页布局合成容差不能替代真实 paged PPL。完整 WikiText test 也不等于长对话或 1024-token 自由生成质量。结果为单次评测，没有置信区间。
3. **对照公平性**：KIVI 评估只报告相对本仓库原生 FP16 的差值，FP16 和低比特批次存在环境差异；未核实官方表，不能声称复现论文或追平 SOTA。
4. **历史溯源**：长窗口 PPL 保留 NLL 总量与实际计分 token 数，支持 PPL 重算；它不具备敏感性实验后来增强的全部逐窗口、token 指纹和源码字段。历史信息缺失不反向补造，本次汇总不能提升其原始溯源粒度。
5. **模拟范围**：只建模 attention，不含投影、MLP、权重读取、cache append、采样、完整 decoder 或能耗。辅助单元吞吐、物理分页、bank conflict、互连和绝对时延未校准；独立工具只覆盖声明的算子与四个压力检查点。
6. **容量边界与外部内核限制**：SRAM 不可行候选是容量证据，正式独立结论采用 `independent_os/`。P1 尚有一项 FP16 online 测试未达历史阈值，不作为 R1 模拟器验收依据，后续若复用该内核须独立关闭。
7. **未做的扩展**：128K、13B/Falcon 扩展、完整 8B 长生成质量与 RTL 校准未完成。它们不属于 R1 已声明的完成范围；本报告不将其视为完成，也不推断资源不足是每项未运行的已证实原因。
8. **本地与公开可得性**：图、原始数据、实验脚本和部分运行依赖仅在本地保存；GitHub 上的报告链接不意味着这些文件已公开。公开复现包须另行选择文件并按项目发布范围处理。

本次验收核对了七个实验清单中的 305 条源码/依赖/结果哈希记录，均匹配；这是含重复依赖引用的记录数。24 条长窗口与 218 条敏感性 PPL 独立重算一致，KV 流量分解、压力均值和末步关系一致。哈希只能证明与登记内容一致，不能单独证明算法或硬件正确；有效性还依赖上述实验检查、独立参照和适用范围。

## 6. 结论与后续工作

### 6.1 四项验收门槛

已完成实验对四项门槛的支持如下：

- [x] **真实 cache-path 可复现**：源码具有历史状态、append/load 和量化残差语义；已有合成、整模和压力证据。数值路径见 §3.1，物理打包及高精度展开的边界见 §5。
- [x] **至少一条长上下文 bytes/token–精度 Pareto，含 paged 列**：§3.3 覆盖 4K–32K，给出双布局流量和明确标注的 contiguous 精度参照；不声称独立测过 paged PPL。
- [x] **专用模拟器与 Roofline/SCALE-Sim 在约定检查点趋势一致**：§3.5 的独立批次 1414 项检查通过，接受范围限于已声明的 attention 趋势和压力检查点。
- [x] **默认模型、硬件包络与评测协议已锁定**：§2 按现行协议关联相同模型、窗口与格式，未替换 C0 入口、重复计层数或重复加入 paged 元数据。

**验收结论：R1 已完成约定研究范围，可进入 R2。** 这一结论不扩大 §5 所列覆盖范围，也不表示 R2 已经开始实施。

### 6.2 R2 的起步依据

首版静态数据通路以 **C2 均匀 INT4** 为起点，保留 C0 作数值正确性对照、C5 作低退化精度/流量参照，C3 作为旋转代价消融、C4 作为极低比特对照。选择 C2 是为了先建立不含旋转和 KIVI 残差调度的基础流式通路；这是工程起步选择，最终格式仍需结合精度阈值、吞吐、面积和能耗确定。

R2 首先落实物理打包及元数据读取、tile 内流式反量化与 MAC 衔接、GQA 复用和更适合单查询的映射，并对 SRAM 预算保留不可行点。tile=128/512 可作为已有可行参照，不能把 2048 设为未经容量检查的默认值。按算子和形状校准反量化、旋转、softmax 及 QK/PV，分别报告名义字节减少与实际周期变化。

进入格式或映射决策前，应先确定可接受的精度退化标准；若 R2 主张真实 paged 生成质量、压缩存储或吞吐改善，须补上对应实测证据。更长期的完整长生成、扩展模型与模拟器—RTL 系统校准按[研究计划](../../docs/research_plan.md)推进。
