# R1 Decode Simulator

本目录为 R1 独立解码模拟器。当前进度以 [研究里程碑](../../docs/progress/milestones.md) 为准，输入口径见 [指标协议](../r1_kv_baseline/protocols/metrics.md#5-模拟器输入与交叉核对)，研究结果见 [R1 总报告](../r1_kv_baseline/REPORT.md)。

KV 流量输入位于 `../r1_kv_baseline/experiments/kv_pareto/results/summary.json`；完整实验源码、结果仅保存在本地与服务器，单独克隆 GitHub 不包含这些输入。

实现可参考 P5 的设计，运行时不得导入 `learning/`。模拟器实现留在本目录，实验、冒烟测试和结果放在 `experiments/<name>/`。CPU 即可开展流量接口与趋势检查；只有涉及真实模型重测时才需另行申请 GPU。

## 输入接口

[`inputs.py`](inputs.py) 使用 Python ≥3.11 标准库，无第三方依赖、GPU 或模型权重要求。默认按代码所在位置定位项目根目录；指定 `project_root` 时，输入路径相对于该根目录解析。必需的本地输入文件缺失或校验失败时抛出 `InputValidationError`，不跳过失败行。默认参数及输入范围以 [指标协议](../r1_kv_baseline/protocols/metrics.md#5-模拟器输入与交叉核对) 为准。

从项目根目录导入：

```python
from research.r1_decode_sim import load_inputs

inputs = load_inputs()
step = inputs.step(32768, "C2", "paged")
whole_model_bytes = step.bytes_per_token
one_layer_bytes = step.per_layer.total
effective_kv_element_bytes = step.kv_bytes_per_element
pressure = inputs.pressure(16384, 1024, "C2", "paged")
generation_mean = pressure.mean_bytes_per_token
last_step_bytes = pressure.last_step.bytes_per_token
pair = inputs.step_with_quality(32768, "C2", "paged")
quality_scope = pair.quality_scope  # contiguous_reference_only
```

`per_layer` 保留 payload / scale / zp / page 分解。`kv_bytes_per_element` 是当前上下文下 **K+V 合并、包含元数据的平均有效字节**，乘 `step.n * step.geometry.n_elem` 可还原单层总量；它不是分别测得的 K/V 位宽，也不是反量化后 SRAM 占用或任意 tile 的精确打包大小。原始聚合输入不支持恢复逐步压力轨迹或 K/V 各侧流量，不以均值或末步伪造这些信息。

精度对象保留原始 `layout`、窗口、计分覆盖范围及 NLL/PPL。`step_with_quality` 按模型、窗口和格式关联；同布局标记为 `same_layout_window_ppl`，paged 标记为 `contiguous_reference_only`。窗口 PPL 不是该 decode step 的精度测量，压力点不自动关联 PPL。四个 `L*/ppl_summary.json` 是唯一入口，根目录历史 4K 副本不重复读取。

每条输入保留来源路径、JSON 行位置及文件 SHA-256；哈希标识本次读取的文件，不补造历史执行源码或环境信息。适配器只核对输入结构、协议和数值一致性，不重跑原实验或估算硬件时间。

## CPU 输入检查入口

在项目根目录运行：

```bash
python -B research/r1_decode_sim/experiments/decode_sweep/run_input_check.py
```

也可从其他工作目录用脚本的绝对路径启动。入口运行保留的 `smoke/` 检查，将实际检查状态、参数、环境及输入/源码哈希保存到本实验 `results/input_adapter/`，并用非零退出码表示失败；`--output-dir` 可指定本实验 `results/` 内尚不存在的项目相对目录。实验登记见 [`experiment.json`](experiments/decode_sweep/experiment.json)（仅本地保留）；完整结果见[正式报告](experiments/decode_sweep/REPORT.md)，输入检查不代替独立趋势验收。

## 模拟器主体

[`config.py`](config.py) 定义硬件、工作负载和映射参数，[`simulator.py`](simulator.py) 实现 tile 级周期模型。`latency_cycles` 是所选映射下 **32 层 attention 的串行合计**，不包含投影、MLP、权重读取、cache append 或采样，不能解释成完整模型的生成延迟或实测 tokens/s。

```python
from research.r1_decode_sim import load_inputs
from research.r1_decode_sim.config import Hardware, Mapping
from research.r1_decode_sim.simulator import simulate, simulate_pressure

inputs = load_inputs()
result = simulate(inputs.step(32768, "C2", "paged"),
                  Mapping(kv_tile=512, buffering="auto"), Hardware(), trace=True)
if result.feasible:
    attention_cycles = result.latency_cycles
    kv_read_bytes = result.kv_read.total
    hbm_bytes = result.hbm_bytes  # KV + Q 读取 + O 写回
pressure = simulate_pressure(inputs.pressure(16384, 1024, "C2", "paged"))
```

计算、存储与调度约定：

- QKᵀ 与 PV 分别按实际矩阵形状和 PE 行列计算 tile 尾块占用。查询头默认 32，KV 头来自输入；计算量按查询头展开，KV 载荷在 GQA 查询头之间复用，不再次乘查询头数。
- 默认 `query_rows` 映射沿查询行使用 PE，查询头串行。可选 `head_folded` 假设阵列支持不同头的独立操作数通路并折叠到行方向；这是另一个架构假设，不能当作默认阵列已具备的能力。
- decode 将输入的单层四项流量按 token 区间做整数分摊，合计严格守恒；它不是真实 DMA 页序列。派生 prefill 对每个 Q tile 重新读取该 KV 总量，使用完整矩形、不跳过因果屏蔽块，也不代表测得了 prefill 流量。
- SRAM 包括 FP16 Q、FP32 输出累加器/softmax 状态/score tile、打包 KV、FP16 解码 KV。BDR 另预留 FP32 旋转暂存及常驻矩阵。双缓冲只重复 packed/decoded KV 两块空间；不以有效比特代替反量化后占用。
- 一个 DMA 通路与一个串行计算通路调度；反量化、旋转、QKᵀ、softmax、PV 计入计算阶段。双缓冲在前一占用者计算结束后才能复用；Q load 和最终 O store 位于扫描两端，不跨 Q tile 或层重叠。`auto` 在双缓冲空间不足时选择单缓冲；显式 `double` 空间不足则返回 `feasible=false`、延迟 `null`。
- `component_cycles` 是各阶段独占周期之和，等于串行总周期；实际调度周期包含重叠，不能再加一遍 DMA。`first_query_tile_trace` 仅记录第一层第一个 Q tile 的事件，不冒充全生成轨迹。

基础硬件包络对齐 [锁定协议](../r1_kv_baseline/protocols/models_context.md#6-硬件包络架构记账假设)。模型另外明确假设每 PE 每周期 1 次解码后 MAC、softmax 每周期 32 个 score、反量化每周期 256 个量化元素、旋转每周期 256 次 MAC、DMA 启动开销默认 0 周期。BDR 按 32×32 块逆旋转计 MAC；KIVI 的 FP16 残差不收取反量化开销。这些吞吐都未校准，softmax 是 score 吞吐近似，未单列 running-O 重缩放和最终归一化开销。实际 PyTorch 未打包载荷、阵列 skew、bank conflict、互连与页指针停顿不在当前周期模型中。低比特流量下降不保证默认映射下的延迟下降。

仅接收聚合输入的 `simulate_pressure` 保留全程总量、平均流量和末步仿真，并根据总 MAC 工作及峰值带宽给出下界；该接口的 `mean_attention_latency_cycles=null`。完整生成模拟使用下述逐步轨迹接口。结果不提供能量估计，独立趋势验证状态由实验入口单独登记。

## CPU 主体运行入口

```bash
python -B research/r1_decode_sim/experiments/decode_sweep/run_simulator.py
```

默认扫描 decode 与派生 dense prefill、KV tile 128/512/2048，以及输入的全部格式/布局/窗口；压力点只运行 decode。使用 `--mode decode` 可只跑 decode，`--buffering single|double|auto` 和 `--head-mapping query_rows|head_folded` 选择调度。`--hardware-json` 接受项目内 JSON 对象覆盖硬件字段，完整实际参数保存在结果中；改变硬件的探索结果须与锁定包络结果区分。

入口先运行输入与模拟器冒烟检查，再保存 `results/core/simulation.json`、`checks.json`、`run_config.json` 和必要日志。原始输入不改写，已有结果目录不覆盖；不可行候选保留状态，不计入可行延迟比较。CPU 主体检查不替代相同负载与硬件参数下的 Roofline/SCALE-Sim 独立验收。该入口只登记主体批次状态，独立对照使用下述单独入口；当前实验结果见[正式报告](experiments/decode_sweep/REPORT.md)。

硬件 JSON 必须是无重复字段的对象，未知字段、非有限参数、缺失或损坏文件均返回参数错误。已有输出目录同样返回参数错误；某个请求模式没有任何可行候选时，入口保留扫描结果并返回失败，不把它标为成功。JSON 在创建结果文件前完成序列化校验，非法数值不会留下半写入的 JSON。

## CPU 小规模检查入口

```bash
python -B research/r1_decode_sim/experiments/decode_sweep/run_small_checks.py
```

入口运行已有输入/主体检查，再显式加载 `smoke/small_cases.py`：用独立的引擎状态机核对单双缓冲事件；用逐 Q 块、逐 PE 波次展开的参照核对主体中的重复块聚合。合成上下文覆盖 1、15、16、17、127、128、129，包含页边界和 KIVI 残差边界；同时覆盖六格式、双布局、decode/dense prefill、单双缓冲及两种头映射。

合成字节由测试夹具解析构造并明确标记，不是真实 cache-path 测量。参照与主体共享所声明的架构假设，检查的是实现一致性，不能替代 Roofline/SCALE-Sim 独立趋势验收。新检查不会递归触发主体入口的测试发现。

检查还从另一个工作目录调用主体入口，覆盖成功执行、已有目录保全、错误硬件文件、错误 tile 参数、输出范围和全部候选不可行等情况。每轮保存独立的 `results/small_checks/`：`small_cases.json` 保留逐例实际值/参照及入口回执，`checks.json` 登记状态，`run_config.json` 保存参数与源码/输入哈希；入口用非零退出码表示检查失败。失败证据按项目规则保留，只有被完整且可复现的结果覆盖后才登记淘汰。

## 完整网格与压力轨迹

[`traces.py`](traces.py) 读取单独采集的逐步 cache 字节计数：D(16384,1024) 的每组必须有连续的 1024 步，首步对应流量与精度的 16K 输入，末步及全程字节总和必须与流量与精度压力汇总精确一致。格式、布局、几何、分项字节、源文件哈希与 JSON 行位置均保留；缺组、重复、乱序或不完整记录会被拒绝。

```python
from research.r1_decode_sim.traces import load_pressure_traces, simulate_pressure_trace

traces = load_pressure_traces(
    "research/r1_decode_sim/experiments/decode_sweep/results/pressure_capture/pressure_trace.json",
    inputs,
)
generation = simulate_pressure_trace(traces[0], Mapping(kv_tile=512))
if generation["feasible"]:
    total_cycles = generation["total_attention_latency_cycles"]
    mean_cycles = generation["mean_attention_latency_cycles"]
```

每步重新计算 tile 尾块、KIVI 残差工作量、SRAM 占用与重叠调度，再按生成步顺序累加。均值来自全程周期之和；只要任一步不可行，全程总时延与均值即为 `null`，仍保留各步状态和可行末步的结果。范围仍是 attention-only，不包含完整 decoder 的其他工作，也不提供长生成精度结论。

固定硬件包络下运行完整网格：

```bash
python -B research/r1_decode_sim/experiments/decode_sweep/run_full_grid.py --pressure-trace research/r1_decode_sim/experiments/decode_sweep/results/pressure_capture/pressure_trace.json
```

入口扫描四窗口、六格式、双布局、decode/dense prefill 与 KV tile 128/512/2048；压力点对三个 tile 分别模拟完整轨迹。固定 query tile=32、`query_rows`、`auto` 缓冲和默认硬件，参数随批次保存。`--output-dir` 只接受本实验 `results/` 内尚不存在的新目录。

默认 `results/full_grid/grid.json` 保存全部候选；`pressure_steps.jsonl` 按 `trajectory_id` 保存逐步状态与周期；`pressure_summary.json` 保存全程汇总、末步和来源；`checks.json`、`run_config.json`、`smoke.log` 保存检查、参数、环境及哈希。重复运行须通过 `--output-dir` 指定有明确用途的新目录；运行时间保存在元数据中。不可行候选完整保留；基准 tile=512 的网格及压力点必须全部可行才能通过。原始输入、采集源码和参数均核对哈希；历史源码通过实验清单中的归档与成员哈希直接核验，无需解包为运行副本。入口仍使用标准库，不导入 PyTorch 或 `learning/`，也不把守恒检查标为独立工具验收。

重新采集压力输入时使用本实验的 `capture_pressure.py`，依赖声明见 `requirements-trace.txt`（Python ≥3.11、PyTorch、SciPy）。建议分配 4 个 CPU 核和 4 GB 内存，无需 GPU 或模型权重：

```bash
python -B research/r1_decode_sim/experiments/decode_sweep/capture_pressure.py --project-root . --expected-summary research/r1_kv_baseline/experiments/kv_pareto/results/summary.json --output-dir research/r1_decode_sim/experiments/decode_sweep/results/pressure_capture_new --threads 4
```

采集默认调用共享的 [`r1_kv_baseline/cache_path/`](../r1_kv_baseline/cache_path/)，以零值 K/V 按真实形状预填充、逐 token append，并在每次 append 前读取当前 `bytes_breakdown()`。这些是 cache 实现的名义打包字节计数，不是实测 HBM 时间、物理 DMA 轨迹或整模 forward。采集批次保存实际软件版本和共享源码/输入哈希；历史执行版本保存在实验清单登记的源码归档中。实验结果和历史源码归档仅本地保留，GitHub 克隆不包含这些实验输入。当前完成范围统一见[里程碑](../../docs/progress/milestones.md)，有效结果与局限见[正式报告](experiments/decode_sweep/REPORT.md)。

## 独立对照入口

`experiments/decode_sweep/run_independent.py` 对已经保存的完整网格进行对照：独立解析参照直接读取 KV 流量，SCALE-Sim 重新运行声明的 QK/PV 形状，再按精确尾块、头数和层数展开。入口不导入本模拟器的计算函数，也不导入 `learning/`。数值模块源码、输入和完整网格结果均按运行记录核对哈希。注释、格式及经核验的等价写法须通过 `experiment.json` 登记的 `results/validation/source_style_identity.json` 明确匹配旧、新源码哈希；未登记的源码变更仍拒绝使用旧网格。

使用满足 `experiments/decode_sweep/requirements-independent.txt` 的 Python 环境，从项目根目录运行：

```bash
python -B research/r1_decode_sim/experiments/decode_sweep/run_independent.py --output-dir research/r1_decode_sim/experiments/decode_sweep/results/independent_os_recheck
```

本次在 WSL/CPython 3.13 的既有 SCALE-Sim 环境运行，NumPy 兼容版本放在本实验 `runtime/independent_numpy/`，使用额外参数 `--numpy-runtime research/r1_decode_sim/experiments/decode_sweep/runtime/independent_numpy`。该已安装副本适用于对应 Linux/Python 环境；其他环境按依赖声明安装。依赖只用于外部参照，模拟器主体继续使用标准库。建议 4 个 CPU 核和 4 GB 内存，无需 GPU。

入口在调用外部工具前保存 `acceptance_plan.json`，随后保存解析数值、精确 GEMM 展开、上游三个 CSV、逐项对照、软件版本与源码哈希。输出目录必须不存在；重复运行指定含义明确的新目录，失败状态和日志继续保留。`ok=true` 只表示声明的对照标准通过，不把 SRAM 不可行候选或未校准项目变成有效时延。

SCALE-Sim 显式设置 FP16 word 和总计 1 TB/s 的三通道带宽，比较量限定为 QK/PV 阵列周期。其内存分区、通道和独立 GEMM 访问不同于融合 attention，不能直接将上游 DRAM 计数或整体周期替换为本模型的 KV 流量与时延。Roofline 的“decode 更偏存储”按相对算术强度和带宽/计算下界比值解释。验收结果、压力抽样范围及局限统一在[正式报告](experiments/decode_sweep/REPORT.md)维护。
