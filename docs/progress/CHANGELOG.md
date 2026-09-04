# 研究进展日志

按时间倒序记录（最新在上）。

---

## 2026-09-04（M5 WP3：整模 C0–C3 cache-path）

- `LlamaCachePathAttention` 把 C0–C5 接到 HF Llama / Mistral `generate`；C4/C5 仍是 `LlamaKiviAttention` 子类。默认 `layout=contiguous`。
- `fp16` 保持原生 HF（M3 C0 精度上界）；M5 C0 用 `c0` / `fp16_codec`。`int8` / `int4` / `int4_bdr` 走均匀 codec。
- 未跑 8B 精度；M5 仍缺 y 轴。

## 2026-09-04（M5 WP2：8B 几何 bytes/token）

- [`experiments/kv_pareto/REPORT.md`](../../research/r1_kv_baseline/experiments/kv_pareto/REPORT.md)：Llama-3.1-8B GQA 几何，4K–32K × C0–C5 × 双布局；C0 走 FP16 codec。$D(16384,1024)$ 已报全程均值与末步 $N{=}17407$。
- 32K 相对 C0：C4 $\approx 19\%$，均匀 INT4 $\approx 28\%$，C5 $\approx 31\%$（残差窗使 C5 比 C2 更费带宽）。未加载权重、无 PPL。
- 下一步：同一模型补精度 y 轴。

## 2026-09-04（M5 WP1：`traffic_model.py`）

- 新增 [`bytes_accounting/traffic_model.py`](../../research/r1_kv_baseline/bytes_accounting/traffic_model.py)：封装 `cache_path.bytes_breakdown`，输出四项分解、bytes/token、$b_{\mathrm{eff}}$、以及 $D(L_{\mathrm{in}},L_{\mathrm{out}})$ 全程 KV 读 / $L_{\mathrm{out}}$ 与末步。C0 拒绝原生 HF 别名。

## 2026-09-04（M4：paged_layout 分列 C3 门禁）

- [`paged_layout/REPORT.md`](../../research/r1_kv_baseline/experiments/paged_layout/REPORT.md)：布局对齐拆 prefill / decode；C3 decode 的 $V{=}1.81\times10^{-3}$ 标明为 1/30 行（$N{=}64$ seed 1），不当典型值。补 $N{=}128$ C0/C4/C5 刷窗字节。
- 脚本门禁与报告一致：C3 prefill $10^{-5}$，decode K/V 允许 1 档（$5\times10^{-3}$）。`test_paged_cache.py` 增加 C3 逐步 decode 单测；metrics §8.3 写明「逐元素一致」绑定同一 append 粒度。

## 2026-09-04（M4：paged_layout 阶段 A 双报告）

- [`experiments/paged_layout/REPORT.md`](../../research/r1_kv_baseline/experiments/paged_layout/REPORT.md)：C0–C5 contiguous / paged 合成对照。payload/scale/zp 两列一致；$B_{\mathrm{page}}$ 与四池页数符合 metrics v1.1。C0–C2 / C4–C5 逐元素对齐；C3 逐步 decode 见 INT4 舍入。
- M4 勾选完成。下一步 M5。

## 2026-09-04（M4：`paged_cache.py` 落地）

- 新增 [`cache_path/paged_cache.py`](../../research/r1_kv_baseline/cache_path/paged_cache.py)：`PagedUniformKVCache`（C0–C3）与 `PagedKiviKVCache`（C4/C5 四池）。`AttentionWithCache(layout="paged")`；`bytes_breakdown` 拆 payload/scale/zp/page。
- [`protocols/metrics.md`](../../research/r1_kv_baseline/protocols/metrics.md) §8 口径未改，**v1.1 锁定**。M4 实验双报告仍待 `experiments/paged_layout/`。

## 2026-09-04（M4：paged 切分规则写入 metrics 草稿）

- [`protocols/metrics.md`](../../research/r1_kv_baseline/protocols/metrics.md) → **v1.1-draft**：新增 §8（$P_{\mathrm{size}}=16$；C0–C3 按页 encode；C4/C5 量化历史与 FP16 残差分池；Key group=2 页；$B_{\mathrm{pte}}=8\,\mathrm{B}$）。实现 `paged_cache.py` 后若口径未改则去掉 draft。
- [`protocols/models_context.md`](../../research/r1_kv_baseline/protocols/models_context.md) §5.2 改为指向 metrics §8（版本仍为 v1.2）。

## 2026-09-04（R1 实施计划入库并按现状修订）

- 将 Cursor 初稿写入 [`research/r1_kv_baseline/PLAN.md`](../../research/r1_kv_baseline/PLAN.md)；相对初稿的主要修订：官方 KIVI 表不阻塞、实验按语义目录、C4/C5 必做、M4 覆盖两条 cache 后端、M5 的 C0 须走 FP16 codec 记账。
- 核对进度：M0–M3 完成，下一步 M4。同步 `milestones.md`、`research/README.md`、协议 `models_context.md` → v1.2（阶段 B 已用于 M3）。

## 2026-09-04（kivi_eval：按 codec_compare 口径重写 REPORT）

- `experiments/kivi_eval/REPORT.md` 改为目的 / 方法 / 结果 / 权衡 / 局限 / 结论；Δ 只相对本仓库 FP16，不与论文官方表并排。

## 2026-09-04（kivi_eval：阶段 B Table 3 / LongBench）

- 修复 `LlamaKiviAttention` prefill：attention 输出补 `transpose(1, 2)`、`attention_mask=None` 时补因果 mask；8K eager 按 query 分块以免 24 GB OOM。
- `build_*_kivi` 默认 `attn_implementation=eager`。阶段 B 全集重跑：kivi4 ≈ fp16，kivi2 小幅掉点。报告见 [`kivi_eval/REPORT.md`](../../research/r1_kv_baseline/experiments/kivi_eval/REPORT.md)。

## 2026-09-03（academic-researcher 项目 skill）

- 将 academic-researcher 迁入 [`.cursor/skills/academic-researcher/`](../../.cursor/skills/academic-researcher/)；规则改为相对仓库路径，不再依赖云端 `/root/.cursor/skills/`。

## 2026-09-01（kivi_repro：Mistral KIVI patch）

- 新增 `kivi_repro/patch_mistral.py` 与 `MistralKiviAttention`（数值路径复用 `LlamaKiviAttention` + `KiviKVCache`；sliding window 仍由 HF mask 负责）。
- `load_llama_for_generate` 按 `config.model_type` 分发 `llama` / `mistral`；LongBench 协议锚点可跑 kivi2/kivi4。

## 2026-07-31（评测模块去论文数字、通用化）

- `lm_eval_tasks.py`：删除 `PAPER_TABLE3` / `compare_to_paper`；改为 `evaluate_lm_eval` + 可选 `score_delta(reference=...)`。
- `long_bench_tasks.py`：`KIVI_DEFAULT_DATASETS` → `EXTENDED_DATASETS`；弱化论文/KIVI 专用表述。

## 2026-07-31（lm_eval_tasks：Table 3 / LM-Eval）

- 新增 `kivi_repro/lm_eval_tasks.py`：CoQA / TruthfulQA / GSM8K；`KiviHFLM` 包装本仓库 cache-path。

## 2026-07-31（long_bench_tasks：LongBench 四子组）

- 新增 `kivi_repro/long_bench_tasks.py`：子组映射、官方 prompt/max_gen、预测与 scorer（对齐 KIVI）；默认代表 qasper / qmsum / trec / lcc。
- `requirements.txt` 增加 `rouge` / `fuzzywuzzy`（LongBench 打分）。

## 2026-07-31（kivi_eval 冒烟改为整模路径）

- `experiments/kivi_eval/run_smoke.py` 改为测 `kivi_repro`（patch / 前向 / generate / clear）；默认玩具 Llama，仅 PASS/FAIL。
- 更新 `experiments/kivi_eval/REPORT.md` §4.1：4/4 PASS。

## 2026-07-31（hf_generate：generate 封装）

- 新增 `kivi_repro/hf_generate.py`：`load_llama_for_generate` / `generate_ids` / `generate_text`；生成前清空 Kivi cache。
- 玩具模型冒烟：KIVI patch 后贪心续写有限，`GenerateInfo` 含 bytes。

## 2026-07-31（patch_llama：整模替换入口）

- 新增 `kivi_repro/patch_llama.py`：`patch_llama_model` / `build_llama_kivi`；默认超参对齐协议。
- 玩具模型冒烟：幂等 patch、prefill 残差窗与 bytes 汇总正常。

## 2026-07-31（LlamaKiviAttention 整模接入）

- 新增 `kivi_repro/llama_kivi_attn.py`：`LlamaKiviAttention` 经本仓库 `KiviKVCache` 写/读；提供 `from_llama_attention` / `clear_llama_kivi_caches`。
- 玩具 Llama 冒烟：prefill+decode 有限，残差窗长度符合协议。

## 2026-07-28（KIVI 阶段 A 冒烟通过）

- `experiments/kivi_eval/run_smoke.py`：Qwen2.5-0.5B 真实 `past_key_values` → C0/C4/C5 cache-path；512/1024 prefill + 136-step 残差窗 decode；**10/10 PASS**。
- 更新 `experiments/kivi_eval/REPORT.md` §4.1。

## 2026-07-28（新建 KIVI 模型评测实验目录）

- 新增 `experiments/kivi_eval/`：阶段 A 冒烟（0.5B 短序列）+ 阶段 B Table 3 / LongBench；`REPORT.md` 标明未开始。
- 更新 `research/r1_kv_baseline/README.md`、`milestones.md` M3 链接。

## 2026-07-28（删除 m1/m2 实验目录）

- 删除 `experiments/m1_codec_accuracy/`、`experiments/m2_int4_bdr/`；C0–C5 对照统一由 `experiments/codec_compare/` 承担。
- 更新 `milestones.md` M1/M2 链接至 `codec_compare/REPORT.md`。

## 2026-07-28（C0–C5 编码统一对照）

- 新增 `experiments/codec_compare/`：合并原 M1/M2 口径，加入 KIVI 风格 C4/C5；同一真实 cache-path 上配对比较精度与 bytes。
- 结论要点：outlier 下 BDR 仍优于均匀 INT4；KIVI-4 刷窗后精度优于 INT4 但流量更高；KIVI-2 合成设定误差过大；短于残差窗时 KIVI≡FP16。
- 报告：`experiments/codec_compare/REPORT.md`。
- `cache_path/`：KIVI 核 + `KiviKVCache` + `AttentionWithCache` 已支持 C4/C5（M3 B1–B4）。

## 2026-07-27（R1 M2 INT4+BDR 实验）

- （历史）曾用 `experiments/m2_int4_bdr/`；现已并入 `codec_compare` 并删除原目录。

## 2026-07-27（删除 quant/；research 自包含）

- 删除 `research/r1_kv_baseline/quant/`；`BlockDiagonalRotation` 等迁入 `cache_path/rotation.py`，由 `kv_codecs.Int4BdrCodec` 本地导入。
- 约定：正式研究不运行时依赖 `learning/`；需用的逻辑抄入/重写到 `research/`。计划 M1/M2/M7 等「复用 learning」条目已改写。
- `protocols/models_context.md` → v1.1（去掉 `quant/` 路径；offline 按需自建）。

## 2026-07-27（R1 M1 实验归档约定）

- （历史）实验曾落在 `experiments/m1_codec_accuracy/`；现已并入 `codec_compare` 并删除原目录。
- `.gitignore`：`research/**/experiments/**/results/` 与 `research/**/outputs/`；云端仅同步实验 `REPORT.md`。

## 2026-07-27（R1 M1 编码精度实验）

- （历史）曾新增 M1 C0/C1/C2 对照；结论已并入 `experiments/codec_compare/REPORT.md`。

## 2026-07-27（R1 M1 contiguous cache-path）

- 实现 `cache_path/kv_codecs.py`（C0–C2 encode/decode/bytes/`get_codec`）、`kv_cache.py`、`attention_with_cache.py`（prefill/decode_step）。
- milestones：M1 勾选完成；下一步 M2（INT4+BDR）。
- 删除 `cache_path/M1_NOTES.md`、`cache_path/README.md`、`cache_path/test_cache_path.py`；约定非用户要求不主动新增说明文档。

## 2026-07-24（R1 M0 协议锁定）

- 新增 [`research/r1_kv_baseline/protocols/models_context.md`](../../research/r1_kv_baseline/protocols/models_context.md)：模型阶梯、上下文阶梯、decode 压力点 $D(16384,1024)$、阶段 A（弱机）/B（≥24 GB GPU）、硬件包络（32×32 @ 1 GHz / 16 MiB / 1 TB/s）。
- 新增 [`research/r1_kv_baseline/protocols/metrics.md`](../../research/r1_kv_baseline/protocols/metrics.md)：分层指标、bytes/token 分项公式、对照谱 C0–C5、双布局必报。
- 约定：当前为阶段 A，不下 7B/8B；主机内存不强制 128 GB。
- milestones：R1 → 进行中，M0 勾选完成。

## 2026-07-24（R1 quant 独立化）

- 将 `research/r1_kv_baseline/p2_legacy/` 重命名为 `quant/`，去掉 P2 / 副本表述。
- 精简 `quant/`：仅保留 `fakequant.py`、`rotation.py`、`offline_utils.py`；删除测试、报告、PLAN、误差分析脚本与 outputs。
- 更新 `research/r1_kv_baseline/README.md`、`research/README.md`、`docs/progress/milestones.md`。

## 2026-07-23（R1 准备工作）

- 创建 `research/r1_kv_baseline/`（protocols / cache_path / kivi_repro / bytes_accounting / experiments / outputs）与 `research/r1_decode_sim/`。
- 在 `research/r1_kv_baseline/` 下建立量化工具库（现为 `quant/`）；`learning/` 保持归档只读。
- 新增 conda 环境 `r1-kv-baseline`（Python 3.11）及 `requirements.txt`（torch / transformers / lm-eval 等）。
- 更新 `research/README.md`：R1 状态为准备中。

## 2026-07-23（文档与定位对齐）

- 以 `docs/research_plan.md`（R0–R5）为唯一真源，重写 `docs/00_background_and_baselines.md`、`docs/progress/milestones.md`、根 `README.md`、`docs/progress/README.md`。
- 新增 `research/README.md` 作为 R1 正式研究入口；`learning/` 标明已归档且不再定义主线。
- 修正 `survey/manuscript/references.bib`：KIVI 作者 Hongye Jin；SystolicAttention 作者 Jiawei Lin 等。
- 综述 gaps/conclusion 增加 companion-plan（decode-centric）定位，避免推向「更大 PLENA 全栈」。
- 文献监视与对比手册、academic-researcher 规则见既有 `docs/lit_watch/` 与 `.cursor/rules/lit-watch-academic-researcher.mdc`。

---

## 2026-07-22（P1–P5 英文综合稿）

- 新增 [`learning/manuscript/`](../../learning/manuscript/)：IEEE 会议体英文短文 `attention_learning_pipeline.tex`，串联 P1–P5 结果与图表。
- 文献经 arXiv/Crossref 核验（纠正 Timeloop=ISPASS、Softermax=DAC’21）；`references.bib` 为精简可核验子集。

---

## 2026-07-22（目录重组）

- 将 P1–P5 统一迁入 [`learning/`](../../learning/)：每个项目自含 `PLAN.md`、`REPORT.md` 与代码；原 `docs/learning_plan.md` 已删除（计划完成，以 `learning/` 为准）。

---

## 2026-07-22

- 完成 P5 简易 tile-level 模拟器验收：`run_p5.py` 下一键跑通 pytest、两端劣化 demo、Pareto 搜索与 SCALE-Sim 趋势校验（6/6 PASS）。
- 产出 `sim/tile_sim/`（hw/workload/simulator/search/validate）与 `outputs/cross_check_vs_scalesim.md`；独立环境 `p5-tile-sim`。
- 新增 `docs/progress/p5_tile_sim_report.md`。
- 完成 P4 RTL 关键模块验收：`make sim-all` 下 exp / online softmax / 4×4 WS INT8 systolic 全部与 golden 比特对拍通过。
- 产出 `rtl/` 三模块 RTL + TB/脚本、设计笔记与 `notes/fsa_mapping.md`（rescale 落在底部 accumulator）。
- 新增 `docs/progress/p4_rtl_report.md`；独立环境 `p4-rtl`（Verilator 5.020）。

---

## 2026-07-15

- 完成 P3 架构评估工具链：Roofline + SCALE-Sim v3（WS/OS）+ Timeloop/Accelergy Docker，交叉出图与偏差说明。
- 产出 `sim/arch_eval/analysis.md`：decode util≈1%–2.5%、片外 traffic≈49%、16 MiB 约容纳 INT8 KV ~2K token。
- 新增 `docs/progress/p3_arch_eval_report.md`；独立环境 `p3-arch-eval`。

---

## 2026-07-14

- 修复 P2 BDR：由 Gaussian-QR 块旋转改为 QuaRot/SAW 风格 `block_diag(H) @ D`，消除默认 seed 下 PPL 崩坏。
- 在 Qwen2.5-0.5B-Instruct 上更新验收数字：pytest 13/13；PPL fp16=1.68 / INT4=3.23 / Hadamard=2.02 / BDR=1.93；同步 `p2_quantization_report.md`。
- 误差分析对真实模型不再人工放大 outlier；自然激活下 K 误差与 attention 输出 L2 均随旋转下降。

---

## 2026-07-08

- 完成 P2 低精度量化实验验收：fake-quant 库（INT4/INT8/FP8/MXFP4）、Hadamard/BDR 旋转、误差分析与 KV cache 困惑度评估（pytest 11/11）。
- 新建 conda 环境 `p2-quantization`（Python 3.11, PyTorch 2.12.1）。
- 新增 `docs/progress/p2_quantization_report.md` 与 `experiments/p2_quantization/outputs/` 自动报告。

---

## 2026-07-07

- 完成 P1 Attention 数值内核复现验收：标准 / 分块 / online attention、RoPE、RMSNorm、decode-step 测试全部通过（pytest 22/22）。
- 新增 `docs/progress/p1_attention_numerics_report.md`，对照 `learning_plan.md` P1 checklist 记录产出、验证结果与后续衔接。
- 清理 P1 环境配置过程中产生的临时文件与 pytest 缓存。

---

## 2026-06-25

- 新建专用仓库 `attention-accelerator-phd`
- 云端仅保留论文下载脚本；PDF 改由本地 `download_papers.py` 下载
- 新增 `docs/progress/` 进展跟踪目录

---

<!-- 在此上方追加新条目 -->
