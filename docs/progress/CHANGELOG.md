# 研究进展日志

按时间倒序记录（最新在上）。

---

## 2026-09-17（R2 步骤 6 审核通过）

- 用户确认步骤 6 审核通过；依据为[公平映射与周期、能耗基线报告](../../research/r2_streaming_attention/experiments/fair_mapping_cost/REPORT.md)，审核范围及适用边界以报告为准。[里程碑](milestones.md#r2-review)登记为已通过，步骤 7 待开始。
- 按 R1 既有 GitHub 范围整理本次版本：各实验子目录仅发布唯一 `REPORT.md`，共享配置、通用源码与研究文档正常入库；实验运行入口、原始结果及本地运行依赖保留在本地。

## 2026-09-17（R2 公平映射与成本基线核验）

- 完成 GQA 计算守恒、资源受限 split/跨头调度、独立访存事件对拍、写侧与工作缓冲成本闭合；2048-token 预约单列容量检查，复现入口保留淘汰记录并登记实际结果目录。
- 最终源码全网格复跑：660/660 检查通过，4,248 条结果通过守恒核验，4,096 个压力步骤完整保留；容量 62/64 可行，超限点保留。权威结果仍为 `results/mapping_cost_accounted/`，旧批次按哈希和替代依据登记后替换。
- 阅读机器结果后撰写[唯一正式报告](../../research/r2_streaming_attention/experiments/fair_mapping_cost/REPORT.md)，记录供给敏感性及资源比较边界。下方旧批次数字仅属历史记录；当前结果以报告为准。[步骤 6](milestones.md#r2-review)维持待审核，未进入步骤 7，未提交或推送。

## 2026-09-16（R2 步骤 6 提交审核）

- 冻结 SRAM bank/端口/延迟、DMA 队列和每动作能量来源，建立优化 FP16、共享流式解码及 GQA/跨头/KV-split 对照。C3 混合路径计入读侧 K 逆旋转、写侧旋转、输出逆旋转和短块补零，不称为纯 Q/O。
- WSL CPU 冒烟 157/157、占用 144/144；缺省/对照/单因素/优化/压力格均可放。权威结果 `results/mapping_cost_accounted/`。C2 相对优化前 C0 减少 HBM 字节与动态能量、缺省解量化下更慢；C3 相对 C2 周期约 $2.13\times$、动态能量约 $3.06\times$。依据机器结果撰写[唯一正式报告](../../research/r2_streaming_attention/experiments/fair_mapping_cost/REPORT.md)，[步骤 6](milestones.md#r2-review)提交待审核，未标已通过，未进入步骤 7，未提交或推送。

## 2026-09-16（R2 步骤 5 审核通过）

- 用户确认第五步审核通过；依据为[单头操作数与 C3 混合功能基线报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)：混合 DUT 主网格 233/233、契约 106/106、C3 输出 305/305，保留读侧 K 逆旋转，不称为纯 Q/O 优化成功。[里程碑](milestones.md#r2-review)将步骤 5 登记为已通过，步骤 6 待开始。未启动成本比较，未提交或推送。

## 2026-09-16（C3 混合功能基线提交审核）

- 保持 FP32、独立原域参考、原误差门槛和旋转定义；将 C3 验收对象调整为原域 QK＋旋转域 PV/O。流式实现配置升为 `r2-stream-attention-v3`，关闭整遍数值保护，默认 DUT 保留读侧 K 逆旋转，不得称为纯 Q/O 优化成功。
- WSL CPU 混合主网格 233/233、冒烟 8/8、操作数契约 106/106、边界 121/121 通过；既有 305/305 混合输出验收继续作为 C3 功能证据。纯 Q/O 仍为 47/53 与结合律 0/6，只作延期诊断。依据机器结果改写[唯一正式报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)，[步骤 5](milestones.md#r2-review)提交待审核，未标已通过，未进入步骤 6，未提交或推送。

## 2026-09-16（C3 结论范围与审核方案）

- 将纯 Q/O 结论限定为当前实现和已测试变体，撤回“无法补齐”的一般性断言；修正 softmax 权重变化和逐元素联合容差的解释。
- 在[现有报告 §6](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md#6-结论与后续工作)提出混合路径作为 C3 功能基线候选的具体审核方案。未启动新实验、未改共享配置及计算实现，步骤 5 保持待审核。

## 2026-09-15（C3 纯 Q/O 原域开发验证）

- 在固定 FP32 门槛与独立原域参考下，本轮已测试的查询侧分块/补偿求值尚未全部通过原 6 个纯 Q/O 用例；$T{=}129/257$ 仍未过 allclose。本轮对照中，逆旋转 K 的补充混合路径通过，定义与纯 Q/O 不同。
- 负结果见 `results/c3_qo_fp32_association/`。已修订唯一报告的范围、原因与审核建议；步骤 5 维持待审核，未改门槛、未进入步骤 6、未提交或推送。

## 2026-09-15（流式实验测试与报告收敛）

- 将流式实验 `smoke/` 从 16 个源码文件收敛为 2 个验收入口和 2 个共享辅助文件；边界检查并入通用验收，扩展种子与短尾检查并入 C3 验收。通用 106 项、边界 121 项、C3 输出 305 项、内存 2 项与短尾 10 项完成复跑及原字段一致性核对。
- 报告证据集中为三个结果批次，完整覆盖的重复副本按校验依据去重，独有开发记录与源码按原字节归档。映射、哈希及范围见[实验清单](../../research/r2_streaming_attention/experiments/streaming_attention/experiment.json)。
- 按用户要求人工重写[唯一报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)，聚焦当前方法、有效结果、分析与局限。未改计算实现、FP32 门槛或阶段审核状态。

## 2026-09-15（C3 短矩阵一致性）

- 将混合候选剩余误差定位到短 Key 逆旋转和短 QK 的 FP32 求值形状；二者不足 4 行/列时补零至 4，只让有效 token 进入后续运算。固定参考、原阈值及纯 Q/O 默认路径不变，填充运算和存储另行计数。
- 现行候选通过原登记 53/53、扩展离群值 48/48、新增种子 160/160，边界与归并 44/44、分配检查 2/2、短尾契约 10/10。默认契约回归 106/106、边界回归 121/121。现行方法的结果与局限见[正式报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)。
- 原纯 Q/O 仍为 47/53，全部历史失败及对应源版本保留。修订唯一报告、清单及索引；步骤 5 保持待审核，未提交、推送或进入步骤 6。

## 2026-09-15（C3 QK 误差定位与修正候选）

- 保持 FP32、固定参考和全部门槛，逐段替换分数与输出，确认原 C3 outlier 失败主要来自旋转域 QK 求值重排。查询/输出旋转及点积组合的原始记录由[实验清单](../../research/r2_streaming_attention/experiments/streaming_attention/experiment.json)定位。
- 新增显式原域 QK、旋转域 PV/O 候选，无整遍 Attention 回退：原登记 C3 53/53，扩展离群值 47/48，边界 28/28、归并 16/16、内存检查 2/2。默认契约回归 106/106，纯 Q/O 的原六项仍失败；扩展集也保留全部未过项。
- 短块形状方案在不同种子间转移失败，保留结果与对应源版本，未采用到现行候选。原默认通路、共享协议和阈值不变；唯一报告、实验清单与索引同步更新，步骤 5 仍待审核且未全部完成。未提交、推送或进入步骤 6。

## 2026-09-15（R2 单头操作数与 FP32 验收补齐）

- 按用户要求保持全部 FP32 要求及原功能阈值。新增单头单侧操作数供给，落实独立段边界，补齐非有限值/近零门与独立原域参考；C3 启用显式原域数值保护并单列额外读取和计算。
- WSL CPU 保护输出网格 233/233、冒烟 8/8、操作数契约 106/106、补充边界 121/121 通过；纯 C3 Q/O 仍有 6 个 outlier 未过，两个相关入口维持非零退出状态。依据原始结果人工修订[唯一正式报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)，[步骤 5](milestones.md#r2-review)保持待审核且未满足全部要求，未进入步骤 6。
- 实验配置更新为 `r2-stream-attention-v2`，父评测协议与 R1 语义未改。保留失败、数值诊断及所用源版本，新增空 C5 独立参考的边界分支并做专项复核；批次原始记录不改写，来源与哈希统一登记于实验清单。未提交或推送。

## 2026-09-15（R2 流式功能完成性评估）

- 核对登记源码、配置与原始结果哈希，完整复跑现有流式检查；三份机器结果与原批次逐字节一致。新增独立契约检查、范围解码故障注入及 FP32 参考数值诊断，源码和结果保存在所属实验内。
- 依据机器证据更新[唯一正式报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)与实验有效性清单；[里程碑](milestones.md#r2-review)仍为步骤 5 待审核，需补齐所列缺口。未修改通路实现或协议，未进入步骤 6；原始批次保留，未提交或推送。

## 2026-09-15（R2 流式 Attention 与优化 C3）

- 完成有界 tile 流式 QK / online softmax / PV、跨分段归并与 C3 Q/O 变换，入口见[流式 Attention 与优化 C3 报告](../../research/r2_streaming_attention/experiments/streaming_attention/REPORT.md)。流式参数冻结为 [`stream_attention.json`](../../research/r2_streaming_attention/experiments/configs/stream_attention.json)；不修改 R1 源码。
- WSL CPU 上冒烟 8/8、登记格子 233/233 通过：工作缓冲不超过 tile，DUT 未调用 `load()`；尾页、非整齐分组、C5 刷窗、全掩码与背压单独通过。未实现周期或能耗，未跑质量门，未使用集群。旧口径 `results/stream_attention_verified/` 已按清单淘汰，哈希与替代结果见实验 `experiment.json`。
- [里程碑](milestones.md#r2-review)将步骤 5 登记为待审核。未提交或推送 GitHub。

## 2026-09-15（R2 步骤 4 审核通过）

- 用户确认第四步审核通过；依据为[分页、残差与访存事件报告](../../research/r2_streaming_attention/experiments/paged_residual_access/REPORT.md)及分层计量、独立验收和故障注入的完整结果。[里程碑](milestones.md#r2-review)将步骤 4 登记为已通过，步骤 5 待开始；审核范围以报告为准。

## 2026-09-15（R2 旧结果副本去重）

- 经用户明确授权，删除分页与残差实验 `page_access_verified/`、`access_accounting_review/`、`access_contract_smoke/` 三个结果目录中的 11 个散装 JSON 及空目录，共 7,098,671 B。前九个文件与历史归档成员逐字节一致，另两个预检文件由完整正式结果覆盖。
- 保留 `results/page_access_accounted/` 与 `results/archive/accounting_incomplete_evidence.zip`，复核归档全部 28 个成员及现行源码、报告和结果哈希。旧路径、旧哈希、归档成员或替代结果及删除依据已登记于[实验清单](../../research/r2_streaming_attention/experiments/paged_residual_access/experiment.json)，先前淘汰记录的引用已改为可核验的归档定位；原始记录内部的历史路径保持原字节。

## 2026-09-15（R2 分层访存计量与独立验收）

- 分页计量配置登记为 `r2-page-access-v2`，补齐满页编码、C5 刷窗读取及紧凑残差重排读写，控制覆盖字节与实际流量分列；尾页转为 HBM 时更新 PTE/标签，一次追加只读改写已有 Value 量化尾页一次。R1 与评测协议保持原版本。
- 完整重跑主网格、刷窗轨迹、非连续混合追加及逐类故障注入；现行通过条件包括独立事件预期与 packed/scale/min 内容核对。依据机器结果人工更新[唯一正式报告](../../research/r2_streaming_attention/experiments/paged_residual_access/REPORT.md)，[步骤 4](milestones.md#r2-review)恢复为实现与机器核验完成、待用户审核。
- 当前结果位于本实验 `results/page_access_accounted/`；旧功能检查的逐字段覆盖证据及源码/结果哈希登记于实验清单。历史缺陷与所用源码保存在经成员哈希核验的 `results/archive/accounting_incomplete_evidence.zip`。自动审批拒绝删除旧结果副本，原目录继续保留并登记为历史证据；未提交或推送 GitHub。

## 2026-09-15（R2 分页与残差计量评估）

- 核对步骤 4 登记文件哈希并完整复跑既有格子，缓存、存储、元数据和刷窗结果一致；新增尾页读取及写事件故障注入检查，发现写侧计量与验收覆盖缺口。证据及有效范围更新至[唯一正式报告](../../research/r2_streaming_attention/experiments/paged_residual_access/REPORT.md#35-写侧计量与验收覆盖)与实验清单，原始结果保留，缓存实现未改动。
- [步骤 4](milestones.md#r2-review)仍待审核，须先补齐报告所列计量项；步骤 3 状态不变。

## 2026-09-15（R2 步骤 3 审核通过）

- 用户确认第三步审核通过；依据为[物理打包与追加报告](../../research/r2_streaming_attention/experiments/physical_pack_append/REPORT.md)及其存储核验后的完整结果。步骤 3 登记为已通过，步骤 4 保持既有待审核状态。
- 核对报告与结果哈希；共享 `r1_bridge.py` 仅增加分页辅助入口，移除新增文本后的哈希与第三步原执行版本一致。实验清单登记当前源码及该核验依据，历史运行配置和原始结果保持原字节。

## 2026-09-15（R2 分页、残差与访存事件）

- 完成物理 packed 分页、C5 `residual_length=128` 四池刷窗，以及读写、刷窗、页表和元数据事件计量。入口见[分页、残差与访存事件报告](../../research/r2_streaming_attention/experiments/paged_residual_access/REPORT.md)。分页参数冻结为 [`page_access.json`](../../research/r2_streaming_attention/experiments/configs/page_access.json)；不修改 R1 源码。
- WSL CPU 上 1674/1674 格子通过：页数与 R1 公式一致；C5 量化载荷加残差 FP16 等于 R1 payload；均匀 paged 尾页按 FP16 单独记账。元数据共址保持编码不变。未实现流式 Attention、周期或能耗，未使用集群。
- [里程碑](milestones.md#r2-review)将步骤 4 登记为待审核；步骤 3 仍待审核。未提交或推送 GitHub。

## 2026-09-15（R2 物理打包存储核验）

- C5 开口组改为独立紧凑存储，避免借用调用方输入或保留已提交前缀的 FP16 底层内存；K/V 的 payload、scale、min 分侧对齐。缓存底层实测与整段对齐预算分开登记。
- 新增输入复用、非连续输入、底层存储与分侧边界检查，完整重跑原网格后按结果修订[正式报告](../../research/r2_streaming_attention/experiments/physical_pack_append/REPORT.md)。保留专项失败观测与原源码快照；旧网格在逐字段覆盖核验后淘汰，旧哈希、替代证据及核验依据登记于实验清单。
- 统一本实验相关文本为 UTF-8/LF，刷新源码、依赖、结果与报告哈希。步骤 3 仍待用户审核，步骤 4 待开始。

## 2026-09-14（R2 步骤 3 物理打包与追加）

- 完成连续布局的物理 nibble/bit 打包与追加，入口见[物理打包与追加报告](../../research/r2_streaming_attention/experiments/physical_pack_append/REPORT.md)。布局参数冻结为 [`pack_layout.json`](../../research/r2_streaming_attention/experiments/configs/pack_layout.json)；不修改 R1 源码，语义文件哈希与协议一致。
- WSL CPU 上 1878/1878 格子通过：packed 载荷等于 R1 名义字节，INT4 相对 R1 int8 暂存缩小一半；C5 开口组与 32 B 整段对齐浪费单独记账。未实现分页、残差窗 128 或流式 Attention。
- [里程碑](milestones.md#r2-review)将步骤 3 登记为待审核。未提交或推送 GitHub。

## 2026-09-14（R2 步骤 2 文档口径对齐）

- 明确 v1 全程逐 token 前缀算量（89.90 GPU 天）只属于[资源与最小算术报告](../../research/r2_streaming_attention/experiments/arithmetic_feasibility/REPORT.md)，现行质量门为 `r2-evaluation-v2`，不按新口径重算该表。实验清单分开登记当时协议哈希与当前协议文件哈希；源码注释/格式刷新后的哈希写入 `source.files`，产出结果时的哈希留在 `comment_refresh_prior_files` 与历史 `run_config.json`。
- 计划表写明开发模型为 `Qwen/Qwen2.5-0.5B` base；实验索引标明该实验预算不是 v2 质量门成本。未改原始结果，未进入步骤 3。

## 2026-09-14（R2 实验源码分层）

- 将结果登记助手从 `arithmetic_feasibility/evidence.py` 上移到 [`research/r2_streaming_attention/evidence.py`](../../research/r2_streaming_attention/evidence.py)，供后续实验复用；实验入口、整数乘加 RTL 与对拍测试台仍留在本实验目录。
- 报告已引用的 Windows/WSL/GPU 清单、原生 Llama 短测、语料计数、预算和 2,176 组 RTL/映射对拍均保留。无未写入报告的一次性测试。历史 `run_config.json` 与原始结果不改写；路径映射记入实验清单 `relocated_files`。

## 2026-09-14（GPU 对照隔离写入本机与集群约定）

- 根规则 [AGENTS.md](../../AGENTS.md#isolated-gpu-runtimes) 新增「隔离 GPU 运行时」：CUDA 扩展、BitDecoding、SAW-INT4 不得装入 `r1-kv-baseline` 或步骤 2 的 `torch 2.5.1+cu121`；隔离环境留在该机，不入库、不跨机复制。
- 协议 `gpu_kernels` 写明 CUDA_HOME/toolkit 对齐、BitDecoding 先最小 shape 冒烟再 20/100/5、量化网格不对齐时标作者格式，以及 SAW 官方 FA3 在 H100/H800 与放弃官方时延之间的二选一。未构建内核，未进入步骤 3 或 9。

## 2026-09-14（模型权重缓存在本机与集群统一约定）

- 根规则 [AGENTS.md](../../AGENTS.md#model-weight-cache) 新增「模型权重缓存」：每台机器只使用该机 `HF_HOME`，身份是仓库 ID 与 revision；集群默认 `$HOME/hf-cache`，不得把本机路径写进作业，也不得把权重复制进仓库或 `.server-sync/`。
- `activate.sh` 在已导出 `HF_HOME` 时不再覆盖；冒烟脚本按环境选择本机或 `~/hf-cache`。协议 JSON 改为环境变量约定，不再把 `/mnt/f/hf-cache` 写成唯一根。

## 2026-09-14（本地模型缓存集中到 HF_HOME）

- 将已核验的 R2 开发模型 `Qwen/Qwen2.5-0.5B` base 写入 R1 既有 `HF_HOME=/mnt/f/hf-cache` 的 Hub 布局；`Qwen/Qwen2.5-0.5B-Instruct` 的 blob 哈希、`refs/main` 与 config 字节未改。R1 仍只使用 Instruct。
- 协议缓存根改为该 `HF_HOME`；清单在缓存目录的 `INVENTORY.json`。未改 R1 模型 ID、实验源码或历史结果。

## 2026-09-14（R2 协议 v2 与依赖收缩）

- 用户授权将共享配置升为 `r2-evaluation-v2`：质量门 PPL 改为每窗 prefill 建立压缩 KV、仅对计分 token 走流式读路径；全程逐 token 改为冻结小集，不进入 5%/3% 分母。官方 SAW-INT4 FA3 时延限于 H100/H800；BitDecoding 与 CUDA 扩展改为步骤 9 的隔离环境，不升级已完成探测用的 PyTorch 12.1 运行时。1 ns 负结果不回头优化。步骤 3 按用户要求未开始。
- 开发模型 `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987` 的忽略缓存快照已核验：`model.safetensors` SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`，配置与分词文件与 Hub git blob 一致；权重不入库。未完成的 7B 分片已删除，不作为可用快照。
- 删除未登记的下载/隔离环境脚本、Hub API 整包摘录、可重建的 `build/` 与 `tmp/r2-gpu-setup`。步骤 2 正式报告与 `dependencies.json` 保持原证据，不改写。

## 2026-09-14（R2 步骤 2 审核通过）

- 用户确认 Llama 原生 FP16 已覆盖 4K–32K；步骤 2 登记为已通过。Qwen 正式/开发权重与 BitDecoding、SAW-INT4 公开内核仍为待补依赖，须在步骤 9 前补齐，不作为本步失败，也不视为已安装或已测。
- 未修改 `r2-evaluation-v1` 的 PPL 口径；1 ns 单元时序未通过仍只作为算术探针的负结果。步骤 3 尚未开始。

## 2026-09-14（失败算术尝试清理）

- 核验后删除 `arithmetic_feasibility` 中三次未完成的工具接口尝试：`results/integer_madd`（Yosys `check -assert` 失败）、`results/integer_madd_mapped_cells`（OpenROAD 未读工艺）、`results/integer_madd_nangate45`（OpenROAD STA 参数不兼容），并清除对应 `build/`。
- 成功批次 `results/integer_madd_cell_timing` 使用相同 seed=42、2000 随机向量及相同 RTL/测试台/单元库哈希；golden、inputs、rtl_dut 与后两次失败尝试的 mapped 网表/对拍输出逐字节一致。正式证据仍为 2,176 组 RTL/映射对拍与单元时序；1 ns 约束未通过作为有效负结果保留。
- 46 个旧文件的路径与 SHA-256 记入实验清单 `retired_files`，不另建归档副本。报告正文未改，仍只引用成功批次。

## 2026-09-14（R2 资源与最小算术审核）

- 用户“继续”通过首轮协议审核；完成[资源与最小算术实验](../../research/r2_streaming_attention/experiments/arithmetic_feasibility/REPORT.md)的资源探测、原生 FP16 Llama 短测、完整语料计数、RTL/映射网表对拍和单元时序分析，人工撰写唯一报告。执行版本、原始结果与来源哈希由实验清单及 `.server-sync/r2-resource-recovery/` 登记；本地原有文件未被服务器覆盖。
- 固定公开内核及综合库来源，补充必要依赖声明和容器入口；公开内核仅完成依赖核对，相关来源更新见[文献记录](../lit_watch/CHANGELOG.md)。时序未达标、未安装依赖和成本情景在正式报告如实说明。
- 初期 Yosys 单元声明、OpenROAD 工艺读取及命令版本检查曾失败，原始配置、源码快照和日志完整保留；自动审批拒绝删除这些目录，本次未清理或覆盖其证据。成功的完整工具运行独立登记，不将前期失败计为通过。
- [里程碑](milestones.md#r2-review)登记步骤 2 待审核。未修改已批准的 PPL 协议，未进入步骤 3，未提交或推送 GitHub。

## 2026-09-14（R2 协议与首步审核）

- 核对 R1 验收边界和 R2 的 12 步依赖，在现有 [R2 计划](../../research/r2_streaming_attention/PLAN.md#r2-protocol-review)补齐版本、采样/模板、统计、公平成本与独立配置规则；新增可共享的协议参数 JSON，固定公开模型/评测版本及 R1 格式源码哈希。RULER 和原版 LongBench 的登记见[文献修订记录](../lit_watch/CHANGELOG.md)。
- 检查共享配置解析、任务生成预算/评分入口、独立组合分区、文档链接及九项来源的台账/引用键覆盖；三份 R1 格式源码及七份选定结果汇总/独立检查文件与登记哈希一致。本次不重跑 R1，也不把哈希匹配当作 R2 功能验收。
- [里程碑](milestones.md#r2-review)将步骤 1 登记为待审核，步骤 2–12 仍待开始；未执行 GPU/RTL/综合实验，未创建实验报告或结果目录，未改动 R1 源码与结果。下一步须等用户审核本轮协议后开展。

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
