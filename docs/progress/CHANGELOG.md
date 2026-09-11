# 研究进展日志

按时间倒序记录（最新在上）。

---

## 2026-09-11（相关文献补录与引用登记）

- 补齐[近年成果对比手册](../recent_works_comparison.md)中 R2 的相关来源，统一核验信息、台账及引用键；具体变更见[文献修订记录](../lit_watch/CHANGELOG.md)。
- 在 [AGENTS.md](../../AGENTS.md#literature-registration)约定所有项目相关引用先核对手册，缺项在同次任务核验并补录；同步既有文献流程与 R2 引用入口。本次未运行实验。

## 2026-09-11（R2 计划文档分层）

- 按 R1 的结构精简[长线研究计划的 R2 章节](../research_plan.md#r2-overview)，保留目标、研究内容、候选创新、相关工作与阶段门槛；详细实验、质量要求、12 步审核和阶段末创新评估集中到 [R2 详细计划](../../research/r2_streaming_attention/PLAN.md)。
- 修正研究计划和里程碑中的相关链接，保留既定基础门槛与探索规则。执行状态仍为实验未开始；本次仅整理计划文档，未修改代码、运行实验或创建实验报告及结果目录。

## 2026-09-11（R2 探索实验与阶段末评估计划）

- 扩充 [研究计划](../research_plan.md#r2-overview)中的 R2 基础通路、候选机制和新实验，固定双模型增强范围、逐步审核顺序及各项假设的对照、指标和否定条件；相关研究采用原始论文或作者实现链接。
- 分开规定基础技术验收、探索完成情况与创新成立判断，保留原有基础门槛，新增阶段末四类处置；修订风险与成果预期，不承诺探索收益、创新或发表。
- [里程碑](milestones.md#r2-review)增加 12 步审核和阶段末评估登记，实验仍未开始。本次仅修改三份现有文档，未修改代码或 R1 证据，未运行 GPU/RTL 实验，也未创建实验报告或结果目录。

## 2026-09-09（共享缓存实现与实验结果整理）

- `decode_sweep` 的压力采集统一使用 `research/r1_kv_baseline/cache_path/`，移除 `runtime/pressure_capture/` 中重复的源码、输入和依赖声明；原采集日志移至 `results/pressure_capture/`。六份历史执行源码已逐字节核对既有归档，完整网格入口直接验证归档及成员哈希。
- 共享实现完整重采集 12 组、12,288 步，轨迹与权威文件逐字节一致。移除运行副本后，68 项检查及 288 点网格、36 条完整压力轨迹通过；网格与 36,864 步数据逐字节一致，压力汇总仅来源路径不同。
- 删除已被完整成功批次覆盖的 `independent`、`independent_reference`、`small_checks_failed` 三批失败记录，共 35 个文件。核对相同独立验收计划、硬件与参数，以及 58 项测试、2,272 个配置和 12 项入口用例的覆盖；64 个 SRAM 不可行候选和 4 条不可行压力轨迹继续作为有效边界证据保留。
- 原始有效结果保留原字节；重采集、复验产生的重复数据经核对后去重，执行配置、检查及删除依据保留。路径、旧哈希、归档成员和替代依据登记在实验清单、`results/validation/shared_cache_consolidation.json` 及 `.server-sync/decode-consolidation/`。

## 2026-09-09（源码格式与中文注释）

- 扫描 97 个项目维护的源码及 Makefile，按现有规范统一 UTF-8、LF、缩进与 Python 导入格式；将 88 处说明性注释或文档字符串改为中文。专业术语、命令示例、工具标记和机器输出字段保留原义，冻结运行副本及原始结果不改写。
- 87 个 Python 文件通过 Ruff 检查与格式检查；源码语法树或 RTL 有效记号对照确认计算逻辑不变，导入排序、UTC 别名、可选类型和显式 `zip(strict=False)` 按等价写法核验。
- 模拟器 36 项输入测试、58 项实现测试、2,272 个配置检查和 12 项入口检查通过；P2 的 13 项、P5 的 26 项、R1 指标的 6 项测试通过，P4 两项数值自检与四个 RTL 顶层语法检查通过。初次环境准备检查的失败记录与正式通过批次分别保留。
- 各实验清单保留整理前源码哈希及归档，当前源码哈希同步更新；模拟器独立入口使用新的等价源码证据继续严格核对历史网格。机器记录位于 `.server-sync/source-style/`，实验检查数据留在各自 `results/` 内。

## 2026-09-09（R1 完成归档与用途命名）

- R1 已完成约定范围，删除其实施 PLAN，完成依据保留在唯一总报告，进度只在 `milestones.md` 维护；必要计量和模拟器输入约定保留于既有协议。移除总报告中的 P2 对比。
- 文档、当前源码说明和图注取消临时阶段编号；5 个目录改为用途名称，27 个文件逐一核验哈希不变。原始日志、执行元数据和归档源码保留原字节，旧路径及哈希映射登记在实验清单与 `.server-sync/r1-semantic-names/`。
- 36 项输入测试、58 项模拟器测试、2,272 个配置检查和 12 项入口检查通过；五个核心源码文件核验数值逻辑不变，历史独立参照的 1,414 项比较重新计算一致。报告图用相同五个输入重建，仅更新标签，旧派生图按替代证据淘汰。

## 2026-09-09（R1 总验收）

- 根据本地原始结果人工撰写唯一 `research/r1_kv_baseline/REPORT.md`，整合 proxy 与真实路径差异、KIVI 本地差值、双布局 Pareto、压力点及独立模拟器趋势；四项 R1→R2 门槛按约定范围通过，R2 尚未开始实施。
- 在 WikiText 实验中保存四窗口关联图、绘图入口与来源元数据；图使用 24 条 contiguous PPL 和 48 条双布局名义流量，不增加 paged 精度或 GPU 重跑声称。原始结果保持不变，机器核对清单位于 `.server-sync/r1-acceptance/`。
- 更新进度、导航与实验索引，已有实验报告通过相对链接指向总验收，清单同步登记报告哈希与新派生图。物理打包、绝对时延、真实 paged/长生成质量和 RTL 校准继续按其未验证范围列明。

## 2026-09-09（全部实验结果命名核查）

- 将解码模拟器及两处前置检查、运行副本和 WikiText 检查日志改为用途名称；9 处路径调整涉及 66 个文件，原始字节和哈希不变。敏感性验证归档内 14 个成员也移除日期名称，成员内容保持不变。
- 通过带哈希核验的路径映射保留原执行记录的可追溯性；重新执行解码模拟器完整网格，59 项检查通过，288 个候选、36 条压力轨迹和 36,864 步的数值与现有结果一致，另有 8 项路径映射检查通过。验证证据保存在 `decode_sweep/results/validation/path_reproduction.json`。
- 删除已被有效结果完整替代的旧完整网格批次 6 个文件及本次复现的重复产物；独有失败记录继续保留。所有实验结果文件、目录和结果归档成员统一禁止日期时间命名，原始元数据中的时间及历史执行路径保留。

## 2026-09-09（解码模拟器独立趋势对照与报告）

- 增加不导入模拟器计算函数的 Roofline 参照与 SCALE-Sim 3.0.0 运行入口；按锁定硬件、相同 GEMM 形状和精确尾块重新执行独立对照。`results/independent_os/` 保存上游 CSV、参数、版本、源码哈希及逐项比较；5 项手算测试与 1,414 项对照检查通过。
- 在实验 `runtime/independent_numpy/` 使用兼容依赖，未修改共享 NumPy 或上游算法；早期入口溯源中止、依赖不兼容记录当时分别保留，随后依据上方“共享缓存实现与实验结果整理”的完整覆盖验证淘汰。正式结果只采用通过的批次。
- 人工撰写 `decode_sweep/REPORT.md`，说明相对带宽敏感性、阵列利用率、绝对周期差异和四个压力检查点的范围；解码模拟器按趋势标准完成，R1 总验收仍待办。

## 2026-09-09（解码模拟器完整网格与压力点）

- CPU 任务 19310 用固定 cache 源码采集 D(16384,1024) 的 12 组逐步名义字节计数，首末步及总流量与流量与精度对齐；源码、依赖声明、结果和必要日志完整保存在本实验内，回收哈希及原有结果保全记录位于 `.server-sync/pressure_capture_20260909T083130Z/`。
- 增加完整轨迹适配与全程 attention 周期累计；`results/full_grid/` 保存 288 个网格候选、36 组压力轨迹和 36,864 次逐步仿真。59 项检查通过，不可行候选保留；本批次不代替 Roofline/SCALE-Sim 独立趋势验收。

## 2026-09-09（结果目录命名与旧结果淘汰）

- 非解码模拟器实验目录按用途命名：P1 使用 results/numerics_check/，敏感性使用 results/metrics_f64/；三个敏感性入口的默认输出改为 synth/、layer/、smoke/，继续拒绝覆盖非空目录。运行时间留在元数据中，原始结果字节保持不变。
- 删除实验归档中已被覆盖的 24 份旧记录：敏感性 19 份、KIVI 4 份、P3 Roofline 摘录 1 份。依据包括既有 613 行同条件重测对照、218 条 NLL/PPL 重算、固定与主库输出等价、KIVI 全字段覆盖及 P3 24 行实际复现；本次未重新执行完整模型评测。
- 保留 17 份仍有用途的验证或历史记录，其中 SCALE-Sim/Timeloop 未做完整独立重跑，不能仅凭清单断言可替代。清单登记新路径、旧哈希、有效替代结果及删除依据，不再为本次淘汰内容新建备份。
- 本次排除解码模拟器及其前置检查目录，不修改其源码、实验结果和清单；后续新建目录统一遵守不使用日期时间命名的约束。

## 2026-09-09（实验结果再次去重）

- 核对全部学习与研究实验结果，移除 25 份完整副本或汇总已完整覆盖的分片；41 份独有历史记录与机器对照按原字节压缩到各实验 results/archive/，逐成员哈希验证通过。敏感性 results/ 从 44 个文件收敛为 12 个，五份当前有效结果保持原路径和内容。
- WikiText 已完成汇总支持续跑；可靠写入并逐字段核对后才清理检查点。服务器 CPU 任务 19309 的 9 项检查通过，覆盖 24 条历史记录、部分完成、失败保留和原子写入中断。P3 不再自动生成 Markdown 摘录，P5 只保存一份完整搜索 CSV；P3 24 行与历史一致，P5 7 项检查通过，222 行修改前后数据一致。
- experiment.json 登记归档成员、去重去向及旧路径哈希，后续同步不得重复回收已验证保全的同内容文件。解码模拟器的五个输入文件与敏感性五份当前结果哈希不变；保留独立 golden/DUT 证据。临时检查通过后删除，正式报告只更新证据入口。

## 2026-09-09（解码模拟器小规模检查）

- 增加引擎状态机参照、逐 Q 块/PE 波次展开与真实入口检查。首轮 `results/small_20260909T081633_609978Z/` 保留发现的参数/输出保护缺口；修复格式和布局一致性检查、重复硬件字段拒绝、错误输出处理，以及全部候选不可行时的失败状态。
- `research/r1_decode_sim/experiments/decode_sweep/results/small_20260909T081831_176723Z/` 保存通过记录：58 项测试、256 个调度案例、2016 个合成案例及 12 项入口检查。数值模型假设未变；小规模检查不构成独立架构趋势验收，不新增过程报告。

## 2026-09-09（解码模拟器模拟器主体）

- 实现独立的 attention tile 周期模型，包含 GQA 流量复用、QKᵀ/PV、反量化/旋转预算、物理 SRAM 占用和受缓冲释放约束的 DMA 重叠；保留全部建模假设和未校准范围。
- CPU 主体批次 `research/r1_decode_sim/experiments/decode_sweep/results/core_20260909T080353_057385Z/` 保存 288 个候选及 36 组压力末步/下界结果；53 项检查通过。当前完成范围仅为模拟器主体，独立工具验收与正式报告继续待办。

## 2026-09-09（流量与精度/敏感性运行依赖去重）

- 将 Qwen 接入、层格式切换、混合层流量与敏感性功能统一到 r1_kv_baseline 主库；敏感性的真实运行脚本放回实验目录，WikiText 入口直接引用主库，保留新输出目录和非空目录保护。
- 误差计算统一调用 tensor_metrics 的 float64 实现，不再重复运行旧精度归约和保存过程对照；历史原始结果保持不变。
- CPU 任务 19304 完成旧副本与主库对照：42 个小模型案例、28 个流量案例、60 行合成结果完全一致；6 组指标检查、5 个入口和 3 个非空目录保护检查通过。范围不包括重新测量完整模型。
- 固定源码经过哈希核验后归档，删除本地与服务器运行副本；实验清单分别登记当前共享依赖和历史源码归档。临时检查源码归档后移除，不增加过程报告。

## 2026-09-09（解码模拟器输入适配）

- 在 `research/r1_decode_sim/inputs.py` 接入 KV 流量与 WikiText PPL，提供全模型/单层/平均有效元素字节接口，分离压力均值与末步，并保留 contiguous 精度的证据范围及输入哈希。
- `experiments/decode_sweep/` 保存 CPU 输入检查入口、冒烟测试与机器结果；36 项检查通过，五个原始输入文件保持不变。检查结果及环境/源码标识见本实验 `results/input_adapter_20260909T074633_937518Z/`。仅输入适配完成，不构成模拟器趋势验收，不新增独立核验报告。

## 2026-09-09（R1 进度核对与解码模拟器启动准备）

- 协议至敏感性实验 按已声明范围完成，修正进度表中残留的敏感性待复核状态；解码模拟器实现及其独立趋势验收、R1 总验收继续待办。
- 明确解码模拟器的输入、层数、有效比特、paged 元数据与压力点口径，保留 contiguous PPL 和名义流量的证据边界；模拟器不得运行时依赖 learning/。
- 为流量库补上轻量包入口，消除服务器旧入口与本地主版本的导入冲突。48 个流量与精度单步点、12 个压力点、24 个 PPL 记录校验通过；CPU 任务 19300 的 24 个小规模接口检查通过，P5 的 6 个历史趋势检查通过。这些属于前置检查，不替代解码模拟器验收。
- 本地源码、实验结果和报告按清单同步服务器，差异原件先归档；GitHub 仍按约定发布通用源码、学习源码、规范和正式报告。机器回执留在 `.server-sync/decode-readiness/`，不新增过程报告。

## 2026-09-09（实验源码、结果与报告完整性）

- 核查 5 个学习实验和 6 个研究实验；补齐 WikiText/敏感性源码及必要固定依赖，入口使用项目相对路径。源码与服务器专用任务文件分开管理，本地保留完整实验内容，GitHub 发布范围保持独立。
- 每个实验统一保有源码、results/、REPORT.md 和 experiment.json，清单登记入口、依赖、原始结果哈希与有效性状态；报告统一元信息和六节正文，不包含修复过程。
- 学习实验 outputs/ 及 P4 原始向量/DUT 输出归入 results/，46 个搬移文件哈希不变；同步修正源码输出位置和文档引用。服务器专用提交文件从本地实验目录移入历史归档。
- P1 保存实际 CPU 检查结果：21/22 通过，FP16 online 用例未达原阈值；P4 既有原始输出的三个模块对拍通过，未重新生成 RTL 输出。相应报告与里程碑按实测范围更新。

## 2026-09-09（敏感性高精度指标修复与重测）

- 误差归约独立使用 float64，零范数标为未定义，非有限值和明显越界报错；模型及量化路径保持原实验条件。协议升级 v1.2，通用指标和测试在 bytes_accounting 中维护。
- 服务器任务 19295/19296/19297 完成合成、层消融与接口检查；新结果放在敏感性 `results/metrics_f64_20260909/`，16 个服务器原始结果文件哈希不变。
- 613 行新旧对照中，旧归约值被精确复现，38 个余弦越界在新实现中消失；相对 L2 最大相对修正约 0.505%。218 次 PPL 补齐 NLL/token、窗口和实际源码/参数信息，独立重算通过，PPL 与流量不变。
- CPU/CUDA 指标测试、PPL 解析与计分覆盖检查、敏感性七项自检通过。正式报告依据结果人工修订，脚本不生成报告。服务器敏感性入口使用已验证的独立运行副本并拒绝非空输出目录，实际源码哈希随结果保存。
- 正式报告移除修复与部署过程，只保留当前实验内容。服务器运行副本归入 `research/r1_kv_baseline/experiments/kv_sensitivity/runtime/metrics_f64/`，保留测试归入该实验 `smoke/`；通过后的一次性测试及任务辅助脚本已清理。后续实验与测试位置约束统一写入 AGENTS.md。

## 2026-09-09（项目清理与文档入口收敛）

- 在 AGENTS.md 明确 GitHub 发布范围：保留源码/文档与正式报告，沿用研究实验目录仅报告和共享配置入库的规则；原始结果、临时演示文稿素材和同步快照仅本地保留。补充结果目录、模型缓存及本机环境凭据的忽略规则。
- 删除临时依赖、可重建缓存、工具下载包及空失败快照；保留原始结果、源码、手稿、模型缓存和虚拟环境。
- 删除 8 份可由 JSON 重建的研究数据摘录，合并重复进度说明与独立核验说明；既有独有学习证据保留。
- 进度统一在 milestones.md，报告入口在 docs/experiments.md；各级 README 仅导航。根 AGENTS.md 明确禁止按任务新增同步、清理、核验或交接 Markdown。
- 删除清单及结果保全核验保存在本地 `.server-sync/20260909-cleanup/`，不新增清理报告。

## 2026-09-09（流量与精度/敏感性结果回收与报告核验）

- 新增 15 个结果文件，73 个快照文件哈希核验通过；只同步实验结果，32K 增量日志按规则另存。
- 流量与精度 4K–32K 共 24 条 PPL 重算一致；敏感性 Dev/8B 合成及层消融结果齐全，报告已按六节结构整理。
- 敏感性的 38 个合成余弦值越界，已排除出正式结论；既定批次完成，指标问题仍待复核。
- 流量与精度的 24 条 PPL 已重算；敏感性聚合、差值及流量分解一致，但缺少 NLL 明细，不能独立重算 PPL。有效性限制已写入对应正式报告；逐项机器记录见本地 `.server-sync/20260909T052445Z/audit.json`。

## 2026-09-08（结果回收范围与报告整理）

- 本地为权威版本；服务器只回收实验结果、指标、图、必要日志和随结果保存的参数元数据，不回收运行脚本、任务提交或服务器环境文件。
- 十份学习/研究正式报告统一为六节结构，明确实验目的、证据与完成范围。脚本输出改为数据，禁止自动撰写结论；历史自动 Markdown 归为数据摘录并保留原件。
- WikiText 4K、8K、16K 各六种格式结果已核验并写入报告；32K 快照尚不完整，待完成后撰写分析。本次整理未重跑 GPU 实验。
- 实验与证据入口见 [实验索引](../experiments.md)。

## 2026-09-04（流量与精度 ：整模 C0–C3 cache-path）

- `LlamaCachePathAttention` 把 C0–C5 接到 HF Llama / Mistral `generate`；C4/C5 仍是 `LlamaKiviAttention` 子类。默认 `layout=contiguous`。
- `fp16` 保持原生 HF（KIVI 评估 C0 精度上界）；流量与精度 C0 用 `c0` / `fp16_codec`。`int8` / `int4` / `int4_bdr` 走均匀 codec。
- 未跑 8B 精度；流量与精度仍缺 y 轴。

## 2026-09-04（流量与精度 ：8B 几何 bytes/token）

- [`experiments/kv_pareto/REPORT.md`](../../research/r1_kv_baseline/experiments/kv_pareto/REPORT.md)：Llama-3.1-8B GQA 几何，4K–32K × C0–C5 × 双布局；C0 走 FP16 codec。$D(16384,1024)$ 已报全程均值与末步 $N{=}17407$。
- 32K 相对 C0：C4 $\approx 19\%$，均匀 INT4 $\approx 28\%$，C5 $\approx 31\%$（残差窗使 C5 比 C2 更费带宽）。未加载权重、无 PPL。
- 下一步：同一模型补精度 y 轴。

## 2026-09-04（traffic/PPL ：`traffic_model.py`）

- 新增 [`bytes_accounting/traffic_model.py`](../../research/r1_kv_baseline/bytes_accounting/traffic_model.py)：封装 `cache_path.bytes_breakdown`，输出四项分解、bytes/token、$b_{\mathrm{eff}}$、以及 $D(L_{\mathrm{in}},L_{\mathrm{out}})$ 全程 KV 读 / $L_{\mathrm{out}}$ 与末步。C0 拒绝原生 HF 别名。

## 2026-09-04（分页布局：paged_layout 分列 C3 门禁）

- [`paged_layout/REPORT.md`](../../research/r1_kv_baseline/experiments/paged_layout/REPORT.md)：布局对齐拆 prefill / decode；C3 decode 的 $V{=}1.81\times10^{-3}$ 标明为 1/30 行（$N{=}64$ seed 1），不当典型值。补 $N{=}128$ C0/C4/C5 刷窗字节。
- 脚本门禁与报告一致：C3 prefill $10^{-5}$，decode K/V 允许 1 档（$5\times10^{-3}$）。`test_paged_cache.py` 增加 C3 逐步 decode 单测；metrics §8.3 写明「逐元素一致」绑定同一 append 粒度。

## 2026-09-04（分页布局：paged_layout 阶段 A 双报告）

- [`experiments/paged_layout/REPORT.md`](../../research/r1_kv_baseline/experiments/paged_layout/REPORT.md)：C0–C5 contiguous / paged 合成对照。payload/scale/zp 两列一致；$B_{\mathrm{page}}$ 与四池页数符合 metrics v1.1。C0–C2 / C4–C5 逐元素对齐；C3 逐步 decode 见 INT4 舍入。
- 分页布局勾选完成。下一步流量与精度。

## 2026-09-04（分页布局：`paged_cache.py` 落地）

- 新增 [`cache_path/paged_cache.py`](../../research/r1_kv_baseline/cache_path/paged_cache.py)：`PagedUniformKVCache`（C0–C3）与 `PagedKiviKVCache`（C4/C5 四池）。`AttentionWithCache(layout="paged")`；`bytes_breakdown` 拆 payload/scale/zp/page。
- [`protocols/metrics.md`](../../research/r1_kv_baseline/protocols/metrics.md) §8 口径未改，**v1.1 锁定**。分页布局实验双报告仍待 `experiments/paged_layout/`。

## 2026-09-04（分页布局：paged 切分规则写入 metrics 草稿）

- [`protocols/metrics.md`](../../research/r1_kv_baseline/protocols/metrics.md) → **v1.1-draft**：新增 §8（$P_{\mathrm{size}}=16$；C0–C3 按页 encode；C4/C5 量化历史与 FP16 残差分池；Key group=2 页；$B_{\mathrm{pte}}=8\,\mathrm{B}$）。实现 `paged_cache.py` 后若口径未改则去掉 draft。
- [`protocols/models_context.md`](../../research/r1_kv_baseline/protocols/models_context.md) §5.2 改为指向 metrics §8（版本仍为 v1.2）。

## 2026-09-04（R1 实施计划入库并按现状修订）

- 当时确定 R1 实施范围；当前成果见 [R1 报告](../../research/r1_kv_baseline/REPORT.md)。主要约定：官方 KIVI 表不阻塞、实验按语义目录、C4/C5 必做、分页布局覆盖两条 cache 后端、流量与精度的 C0 须走 FP16 codec 记账。
- 核对进度：协议至KIVI 评估完成，下一步分页布局。同步 `milestones.md`、`research/README.md`、协议 `models_context.md` → v1.2（阶段 B 已用于KIVI 评估）。

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
- 更新 `research/r1_kv_baseline/README.md`、`milestones.md` KIVI 评估链接。

## 2026-07-28（删除基础缓存/旋转编码实验目录）

- 删除原先分立的基础编码与旋转编码实验目录；C0–C5 对照统一由 `experiments/codec_compare/` 承担。
- 更新 `milestones.md` 基础缓存/旋转编码链接至 `codec_compare/REPORT.md`。

## 2026-07-28（C0–C5 编码统一对照）

- 新增 `experiments/codec_compare/`：合并原基础缓存/旋转编码口径，加入 KIVI 风格 C4/C5；同一真实 cache-path 上配对比较精度与 bytes。
- 结论要点：outlier 下 BDR 仍优于均匀 INT4；KIVI-4 刷窗后精度优于 INT4 但流量更高；KIVI-2 合成设定误差过大；短于残差窗时 KIVI≡FP16。
- 报告：`experiments/codec_compare/REPORT.md`。
- `cache_path/`：KIVI 核 + `KiviKVCache` + `AttentionWithCache` 已支持 C4/C5（KIVI 评估 B1–B4）。

## 2026-07-27（R1 旋转编码 INT4+BDR 实验）

- （历史）曾使用独立的旋转编码实验目录；现已并入 `codec_compare` 并删除原目录。

## 2026-07-27（删除 quant/；research 自包含）

- 删除 `research/r1_kv_baseline/quant/`；`BlockDiagonalRotation` 等迁入 `cache_path/rotation.py`，由 `kv_codecs.Int4BdrCodec` 本地导入。
- 约定：正式研究不运行时依赖 `learning/`；需用的逻辑抄入/重写到 `research/`。计划基础缓存/旋转编码/解码模拟器等「复用 learning」条目已改写。
- `protocols/models_context.md` → v1.1（去掉 `quant/` 路径；offline 按需自建）。

## 2026-07-27（R1 基础缓存实验归档约定）

- （历史）曾使用独立的基础编码实验目录；现已并入 `codec_compare` 并删除原目录。
- `.gitignore`：`research/**/experiments/**/results/` 与 `research/**/outputs/`；云端仅同步实验 `REPORT.md`。

## 2026-07-27（R1 基础缓存编码精度实验）

- （历史）曾新增基础缓存 C0/C1/C2 对照；结论已并入 `experiments/codec_compare/REPORT.md`。

## 2026-07-27（R1 base cache contiguous cache-path）

- 实现 `cache_path/kv_codecs.py`（C0–C2 encode/decode/bytes/`get_codec`）、`kv_cache.py`、`attention_with_cache.py`（prefill/decode_step）。
- milestones：基础缓存勾选完成；下一步旋转编码（INT4+BDR）。
- 删除 cache_path 下的旧过程说明、README 与临时测试；约定非用户要求不主动新增说明文档。

## 2026-07-24（R1 协议 协议锁定）

- 新增 [`research/r1_kv_baseline/protocols/models_context.md`](../../research/r1_kv_baseline/protocols/models_context.md)：模型阶梯、上下文阶梯、decode 压力点 $D(16384,1024)$、阶段 A（弱机）/B（≥24 GB GPU）、硬件包络（32×32 @ 1 GHz / 16 MiB / 1 TB/s）。
- 新增 [`research/r1_kv_baseline/protocols/metrics.md`](../../research/r1_kv_baseline/protocols/metrics.md)：分层指标、bytes/token 分项公式、对照谱 C0–C5、双布局必报。
- 约定：当前为阶段 A，不下 7B/8B；主机内存不强制 128 GB。
- milestones：R1 → 进行中，协议 勾选完成。

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
- 新增 `docs/progress/p2_quantization_report.md` 与 `experiments/p2_quantization/results/` 自动报告。

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
