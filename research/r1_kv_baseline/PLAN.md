# R1 实施计划：真实 Cache-Path 基线

对应长线计划 [`docs/research_plan.md`](../../docs/research_plan.md) 的 R1 深度。  
本文件是 R1 **实施细则**（M0–M8）；进度勾选以 [`docs/progress/milestones.md`](../../docs/progress/milestones.md) 为准。

源起 Cursor 计划「R1 真实KV基线计划」（2026-07）；下文已按仓库现状修订，不再与那份 YAML 草稿逐字同步。

## 当前状态（2026-09-04）

| 项 | 状态 | 落点 |
|----|------|------|
| M0 协议锁定 | **完成** | [`protocols/`](protocols/)（模型阶梯 v1.2；指标 v1.0） |
| M1 contiguous C0–C2 | **完成** | [`cache_path/`](cache_path/)；对照 [`experiments/codec_compare/REPORT.md`](experiments/codec_compare/REPORT.md) |
| M2 INT4+BDR（C3） | **完成** | `Int4BdrCodec`；同上 |
| M3 KIVI C4/C5 + 阶段 B 任务表 | **完成** | [`kivi_repro/`](kivi_repro/) + [`experiments/kivi_eval/REPORT.md`](experiments/kivi_eval/REPORT.md) |
| M4 paged 双报告 | 未开始 | 缺 `paged_cache.py`；Kivi 路径亦无 page |
| M5 bytes/token Pareto + $D(16384,1024)$ | 未开始 | `bytes_accounting/` 仅占位；kivi_eval 未报流量 |
| M6 误差—流量敏感性 | 未开始 | 同上 |
| M7 decode simulator 挂钩 | 未开始 | [`research/r1_decode_sim/`](../r1_decode_sim/) 仅 stub |
| M8 R1→R2 验收 | 未开始 | 无总 `REPORT.md`（按约定验收时再写） |

下一步：**M4**。M3 已关闭 proxy 任务精度锚；进入 R2 仍缺 paged、Pareto、模拟器交叉核对。

## 相对初稿的修订（必读）

1. **官方 KIVI 表不再作为阻塞项。** 初稿要求「必须」与论文 Table 3 / LongBench 并排。现口径：主声称是相对**本仓库 FP16** 的 $\Delta$；与论文并表仅当已核实的 `--reference-json` 存在。不得把本地 $\Delta$ 写成「已复现官方表」。
2. **实验按语义目录，不按 milestone 名。** 已删除 `m1_codec_accuracy/`、`m2_int4_bdr/`；C0–C5 合成对照统一 `codec_compare/`，整模评测统一 `kivi_eval/`。`kivi_repro/` 是库（patch / 任务），跑数脚本在 `experiments/`。
3. **KIVI 是 R1 必做，不是「可选」。** 长线计划原文写「可选 KIVI」；实施上 C4/C5 已是对照谱必报。13B / Falcon / 128K 仍为 stretch。
4. **M4 必须覆盖两条 cache 后端。** 现有 `ContiguousKVCache` 与 `KiviKVCache` 分离；paged 不能只包一层均匀 codec。
5. **M5 的 C0 流量必须走 cache-path FP16 codec。** `kivi_eval` 里 C0 是原生 HF attention（精度上界，合理）；流量 Pareto 若用这条路径会少算 meta，不可比。
6. **阶段 A/B 已混合。** 协议曾写「默认阶段 A、先不下载 7B」。M3 阶段 B 已跑 Llama-2-7B 与 Mistral-7B。之后：M4 可在 A 开发；M5 主结果必须 B。
7. **不运行时依赖 `learning/`。** 初稿已改写；`r1_decode_sim` 可抄 P5 思路，禁止 `import`/`sys.path` 指向 `learning/`。

## 文档约定

- 默认不新增子目录 `README.md`、`*_NOTES.md`；用户点名再写。
- 日常进度只改 `docs/progress/milestones.md`、`CHANGELOG.md` 与已有协议。
- 实验：`research/**/experiments/<name>/` = `run_*.py` + 本地 `results/` + `REPORT.md`。云端只同步 `REPORT.md`。
- 无独立 `quant/` 包；量化与旋转在 `cache_path/`。
- M8 顶层总报告仅验收时一份；禁止每个 milestone 再写收尾 NOTES。

## 一、协议（M0，已锁定）

权威文本：[`protocols/models_context.md`](protocols/models_context.md)、[`protocols/metrics.md`](protocols/metrics.md)。此处只摘执行要点。

### 模型阶梯

| 角色 | 模型 | 阶段 |
|------|------|------|
| Dev / 快扫 | `Qwen/Qwen2.5-0.5B-Instruct` | A+B |
| KIVI Table 3 锚 | `NousResearch/Llama-2-7b-hf` | B（**已跑**） |
| LongBench 锚 | `mistralai/Mistral-7B-Instruct-v0.2`，max length 8192 | B（**已跑**） |
| 主 Pareto | `meta-llama/Llama-3.1-8B-Instruct` | B（M5） |
| Stretch | Llama-2-13B；Falcon-7B；128K | 不阻塞 R1→R2 |

Batch 默认 $B=1$。

### 上下文与压力点

- KIVI 复现：`group_size=32`，`residual_length=128`；LM-Eval CoQA / TruthfulQA / GSM8K；LongBench 四子组各 $\ge 1$ 代表任务。
- 主 Pareto：4K / 8K / 16K / 32K 必做；128K stretch。
- Decode 压力点：正式 $D(16384,1024)$；弱机代理 $D(1024,128)$ 不得与论文并表。

### 对照谱

- 格式：C0 FP16；C1 INT8；C2 均匀 INT4；C3 INT4+BDR；C4 KIVI-2；C5 KIVI-4。
- 布局：contiguous 与 paged **双报**；主声称不得只依赖连续地址。
- 硬件包络（模拟，非本机 ASIC）：$32\times 32$ PE @ 1 GHz、16 MiB SRAM、1 TB/s HBM。

## 二、现行目录（相对初稿已改）

```text
research/
  r1_kv_baseline/
    protocols/                 # models_context.md / metrics.md
    cache_path/
      kv_codecs.py             # C0–C3 + KIVI K/V 核
      rotation.py              # BDR / Hadamard
      kv_cache.py              # ContiguousKVCache + KiviKVCache
      attention_with_cache.py  # 合成/单元 attention
      paged_cache.py           # M4 待建；须同时服务均匀 codec 与 Kivi
    kivi_repro/                # 库，不是实验目录
      llama_kivi_attn.py
      patch_llama.py / patch_mistral.py
      hf_generate.py
      lm_eval_tasks.py / long_bench_tasks.py
    bytes_accounting/          # M5–M6 待建
    experiments/
      codec_compare/           # C0–C5 合成对照（已完成）
      kivi_eval/               # 整模 Table 3 / LongBench（M3 已完成）
      <pareto 等>/             # M5 起按语义命名
  r1_decode_sim/               # M7；自包含，禁止 import learning/
```

初稿中的 `reproduce_table3.py` 已落实为 `experiments/kivi_eval/run_table3.py` 与 `run_longbench.py`（脚本仅本地，不同步云端）。

## 三、阶段说明

### M1｜contiguous cache-path（C0–C2）— 完成

每个 decode step：新 $k_t,v_t$ 经 quantize→pack→写入 contiguous buffer；读侧 load→dequant→attention。**不是**对 `k_proj`/`v_proj` 一次性 fake-quant。

### M2｜INT4+BDR — 完成

`BlockDiagonalRotation` + `Int4BdrCodec`：写 rotate→quant，读 dequant→inverse-rotate。`codec_compare` 在 outlier 设定下确认 BDR 优于直接 INT4（方向性即可）。

### M3｜KIVI + 阶段 B 任务表 — 完成

- 核：K per-channel 分组、V per-token、残差窗 128；2-bit / 4-bit。
- 整模：`LlamaKiviAttention` / `MistralKiviAttention` 走本仓库 `KiviKVCache`，禁止投影层 fake-quant 冒充。
- 已跑：Llama-2-7B Table 3 全集；Mistral-7B LongBench 四代表任务（qasper / qmsum / trec / lcc），max length 8192。
- 结论要点（相对本仓库 FP16）：KIVI-4 同量级；KIVI-2 小幅掉点。合成路径上 C4 误差很大，**不能**用任务分反推 cache 重建误差小。
- 未做（不阻塞 M3）：与论文官方表并排；PPL；bytes/token；paged。

### M4｜Paged 布局（下一步）

- 新增 page/block 存储（建议 16 token/page）；量化载荷与 scale/zp 对齐 page 边界。
- **两条后端都要有 paged**：均匀 C0–C3 的 contiguous 路径，以及 `KiviKVCache`（残差窗与已量化历史如何切 page 须在实现时写进 `metrics.md` 修订）。
- 自 M4 起，正式流量表默认双列 contiguous / paged。禁止只报连续地址上界。
- 阶段 A 可用合成张量 + 0.5B 冒烟；不必重跑 M3 全集。

### M5｜Bytes/token + 主 Pareto

- `bytes_accounting/traffic_model.py`：按 `metrics.md` 分解 $B_{\mathrm{payload}}+B_{\mathrm{scale}}+B_{\mathrm{zp}}+B_{\mathrm{page}}$。优先封装 `cache_path` 已有 `bytes_*`，不要另起一套口径。
- 主曲线：`Llama-3.1-8B-Instruct`，4K–32K，C0–C5，**双布局**。C0 必须经 FP16 codec 记账。
- 必报压力点 $D(16384,1024)$：全程 KV 读 / $L_{\mathrm{out}}$，以及末步单步 bytes/token。
- 精度侧：至少一种长上下文设定下的任务分或 PPL，与流量画在同一 Pareto 上。M3 的 Table 3 不能替代这条曲线。
- WikiText-2 PPL 若做，放本实验 `REPORT.md`，不另开 milestone。

### M6｜误差—流量敏感性

- Dev 模型（0.5B）上层 / 头 / token 位置网格。
- 在 8B 的 32K 或 $D(16384,1024)$ 上抽样核验。
- 只记录误差随生成步累积的初趋势；完整长压力留给 R5。

### M7｜Decode simulator

- 在 `research/r1_decode_sim/` 自包含实现；用 M5 有效比特（含元数据）驱动 `ElementBytes` 或等价接口。
- 与 Roofline / SCALE-Sim **趋势**交叉核对（decode 更偏存储；流量随 $N$ 近似线性；降低 $b_{\mathrm{eff}}$ 后面 bytes 下降方向一致）。不要求绝对值相等。
- 可对照 `learning/p5_tile_sim` 重写/抄入，运行时不依赖 `learning/`。

### M8｜验收

必要时一份 `research/r1_kv_baseline/REPORT.md`：proxy vs 真实路径、KIVI $\Delta$、Pareto、contiguous vs paged、压力点、模拟器交叉。  
对照下一节门槛逐条勾选后，才进入 R2。

## 四、进入 R2 的门槛

须全部满足：

1. 真实 token-wise KV quantize→store→load→dequant→attention 可复现；与投影假量化的差异已写进验收报告。
2. 至少一条长上下文 bytes/token–精度 Pareto（含 paged 列）。
3. 专用模拟器与 Roofline / SCALE-Sim 在约定检查点趋势一致。
4. 默认模型列表、硬件包络与评测协议已锁定（M0 已满足；M5 不得改协议默默换模型）。

## 五、风险与收缩

写入验收报告，不在实现里默默降级。

- 13B / 多模型 stretch 显存不足：只保留 Llama-2-7B 任务表 + 8B Pareto，并记录原因。
- 128K 不可行：只报 4K–32K；128K 标 Future Work，不阻塞验收。
- 若日后补官方 KIVI 数字且系统性偏差：如实记幅度与可能原因（评测子集、tokenizer、`bleu_max` vs MC 等），不得暗示已追平。
- KIVI-2 合成误差大、任务分未崩：两套数字并存，禁止用下游分否定 cache 重建误差。
