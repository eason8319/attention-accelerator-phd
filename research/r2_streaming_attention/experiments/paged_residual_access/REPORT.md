# 分页、残差刷窗与访存事件实验

实验日期：2026-09-15；整理日期：2026-09-15。完成范围：物理 packed 分页、C5 `residual_length=128` 不对称刷窗、尾页/四池、声明布局的分层读写与元数据事件，以及独立计量验收和故障检测。证据来源为本目录 [`results/page_access_accounted/`](results/page_access_accounted/summary.json)，源码、参数、依赖与文件哈希登记于 [experiment.json](experiment.json)。执行审核状态只见[里程碑](../../../../docs/progress/milestones.md#r2-review)。

本实验不实现流式 Attention、周期模型或能耗，也不把 32 B DMA 量子写成已校准的 HBM 突发。

## 1. 实验目的

回答步骤 3 留下的布局缺口：连续 packed 缓冲能否变成可寻址的页与残差池，以及写侧追加、刷窗和读侧按页扫描能否分开记账。具体要检查：

1. paged C0–C3 是否按 16 token 成页编码，尾页只含占用 token，页数是否等于 $\lceil N/16\rceil$ 的两侧之和。
2. C5 是否保持 R1 的 K/V 不对称刷窗：Key 按 128 token 整窗量化，Value 保留最近 128 token 为 FP16；四池是否禁止 FP16 与低比特混页。
3. 物理 packed 载荷加残差 FP16 是否与 R1 占用口径可对照，尾页实际驻留是否与 R1 把尾页当作已量化名义字节区分开。
4. `scan_read` 的 HBM 逻辑字节是否等于当前 packed 载荷与元数据，SRAM 读是否等于残差/尾页 FP16；元数据共址是否在编码不变的前提下只改变事务切分。
5. 追加、满页提交、刷窗及读改写是否完整计入声明布局上的数据移动，漏记、重复记账、错误字节数或错误层级是否会触发验收失败。

## 2. 方法与设置

- **布局**：打包使用 [`pack_layout.json`](../configs/pack_layout.json)（`r2-pack-layout-v1`），分页与计量使用 [`page_access.json`](../configs/page_access.json)（`r2-page-access-v2`）：$P_{\mathrm{size}}=16$，$B_{\mathrm{pte}}=8\,\mathrm{B}$，标签另计 1 B/页，$R=128$，DMA 量子 32 B。评测协议仍为 `r2-evaluation-v2`。PTE/标签的创建、转换、失效和活动页描述符口径见共享配置的 `events.accounting_contract`。
- **驻留假设**：已打包历史视为 HBM；C5 残差与均匀路径未满页尾缓冲视为 SRAM FP16；PTE 与标签在 SRAM。这是本步冻结的分层，不是 bank/端口校准。
- **追加语义**：激活入口转为 FP16。paged C0–C3 满页才 encode 并 pack，余数留在独立 FP16 尾页。C5 先写入残差再刷窗，刷窗规则与未修改的 R1 `KiviKVCache` / `PagedKiviKVCache` 相同；Key group 的 scale/min 只挂在该 group 第一页。调用方可在返回后复用输入。`load` 仅供对拍；`scan_read` 按页记账，不解包为完整高精度 KV tile。
- **对照**：paged 与 C5 走 R1 分页/残差缓存的 `load` 与 `page_counts`。contiguous C0–C3 走与步骤 3 相同的同切块一次 decode。C3 两段追加只与相同切块对照；逐 token 与较大切块的 INT4 差 1 档是 R1 已记录的旋转×量化粒度，不作为打包或切页失败。
- **读写计量**：均匀 paged 输入先写 SRAM 尾缓冲，满页时读取整页送编码器；contiguous 均匀输入直接送编码器。C5 追加写 SRAM，刷出部分读取一次，保留残差按当前紧凑布局再读写各一次。Value 已有量化尾页在一次 append 中最多读改写一次，其余新片段直接成页。`append_capture` 与 `residual_flush` 属控制事件，保留覆盖字节但不加入流量总和。实际流量按 HBM/SRAM 和读/写分列，PTE/标签独立记录；`scan_read` 为全缓存扫描。
- **独立验收**：参考计量器仅用格式、头数、维度、分页参数和追加长度推导预期，不读取 DUT 页、占用或事件来构造预期。逐笔比较事件类别、K/V、池、层级、追加时刻、token 数、逻辑/物理字节及重复次数；主网格、两段追加、元数据专项和全部刷窗轨迹均接入通过条件。元数据放置同时逐元素核对 packed、scale/min 内容。
- **故障检测**：分别屏蔽 11 类事件；另对 HBM 写事件注入重复、逻辑字节错误、物理字节错误、层级错误及 K/V 归属错误。两种元数据放置合计 32 项。元数据专项保留三个应发现缺失的正例和两个无 HBM 写入的阴性对照。故障只在测试进程内注入。
- **元数据放置**：默认 `separate`（载荷与 scale/min 分事务）。专项比较 `colocated`（同一页载荷与该页持有的元数据一次事务），编码必须不变。
- **负载**：主网格为合成高斯与末维前两通道 $\times 20$ 的 outlier，种子 0/1；采用协议 Llama KV（8 头、128 维）与 Qwen KV（4 头、128 维），长度为 `boundary_lengths` 的正值，追加为整段与逐 token，C1 作为双布局边界格。独立追加检查使用种子 3、非连续输入、8 头×128 维及 3 头×64 维，追加长度分别为 `[15,1,1,110,1,1,15,17,127,129]` 和 `[129,32,1,95,256]`，总长 417/513；覆盖五格式、双布局和双元数据放置，共 80 格。3 头几何及元数据专项的 1 头×32 维均为合成探针。
- **环境**：WSL2 CPU，`r1-kv-baseline` 的 Python 3.11.15、PyTorch 2.13.0+cpu、NumPy 2.4.6、SciPy 1.17.1，PyTorch CPU 线程数为 1。不使用 GPU、模型权重，不向该环境安装 CUDA 扩展。集群未使用。版本见 [run_config.json](results/page_access_accounted/run_config.json)。

可复现入口（工作目录：`research/r2_streaming_attention/experiments/paged_residual_access`）：

```bash
conda activate r1-kv-baseline
python run_page_access.py --output results/page_access_recheck --threads 1
```

输出目录须为空。源码完整不代表已具备流式 Attention 或已校准的物理 DRAM 模型。

## 3. 实验结果

### 3.1 一致性总表

缓存格子 **1674/1674**、冒烟 **77/77**、存储专项 **5/5**、元数据放置 **5/5** 通过。缓存格子含 1,664 个主网格组合、八个空缓存和两个 C1 边界格。全部非空格子的追加、两段追加及扫描事件均与独立预期一致。来源：[summary.json](results/page_access_accounted/summary.json)、[完整格子](results/page_access_accounted/cases.json)、[存储专项](results/page_access_accounted/storage_contract.json)、[元数据放置](results/page_access_accounted/metadata_placement.json)。

| 对象 | 格子数 | 通过 |
|---|---:|---:|
| C0 | 418 | 418 |
| C1（边界） | 2 | 2 |
| C2 | 418 | 418 |
| C3 | 418 | 418 |
| C5 | 418 | 418 |
| 空缓存 $T{=}0$（已含于各格式，不另加总） | 8 | 8 |

C3 在 `head_dim=128` 上只有一个旋转矩阵哈希 `09717404…c84dbf`，与步骤 3 的 Llama/Qwen 主几何一致。协议三份语义源码及 `paged_cache.py` 与登记哈希一致。

已提交的满 packed 页在继续追加后逐字节不变。C5 残差长度与四池页数与 R1 及公式一致。`scan_read` 的 HBM 逻辑字节等于当前 packed 载荷与元数据，SRAM 读等于残差或尾页 FP16；paged 的 PTE 查找次数等于当前页数。

### 3.2 尾页驻留与 R1 占用对照

Llama KV、$T{=}257$、整段写入、高斯、种子 0。单位字节。R1 的 $B_{\mathrm{page}}$ 与 DUT 的 PTE 占用均为 $17{\times}2{\times}8=272$。

| 格式 | 布局 | packed 载荷 | 尾页/残差 FP16 | DUT PTE | R1 payload | R1 page |
|---|---|---:|---:|---:|---:|---:|
| C0 | paged | 1,048,576 | 4,096（尾页） | 272 | 1,052,672 | 272 |
| C2 | paged | 262,144 | 4,096（尾页） | 272 | 263,168 | 272 |
| C2 | contiguous | 263,168 | 0 | 0 | — | 0 |
| C5 | paged | 197,120 | 264,192（残差） | 272 | 461,312 | 272 |
| C5 | contiguous | 197,120 | 264,192（残差） | 0 | 461,312 | 0 |

C0 的 packed 加尾页 FP16 等于 R1 payload：尾页两侧 $1{\times}8{\times}128{\times}2{\times}2=4096$。C2 paged 只把 256 个满页 token 打成 INT4（$2{\times}256{\times}8{\times}128{\times}1/2=262144$）；R1 把末 token 也按 0.5 B/元素计入，故 payload 多 1,024 B，而 DUT 尾页实际是 4,096 B FP16。contiguous C2 立即编码全部 257 token，packed 等于 R1 名义 263,168 B，无 PTE。

C5 在 $T{=}257$：Key 量化 256 token、残差 1 token；Value 量化 129 token、残差 128 token。packed $=(256+129){\times}8{\times}128{\times}1/2=197120$，残差 FP16 $=(1+128){\times}8{\times}128{\times}2=264192$，二者之和等于 R1 payload 461,312。双布局的量化载荷与残差字节相同，差别只在页表。

$T{=}17$ 的 C2 paged：满页 packed 16,384 B、尾页 FP16 4,096 B、PTE 32 B；R1 payload 17,408 B 仍把尾页按 INT4 记账。写侧 HBM 逻辑 18,432 B 只含已提交页的载荷与 scale，不含尾页。

### 3.3 C5 刷窗轨迹与页数

Llama KV、逐 token、高斯、种子 0。页数来自 [flush_trace.json](results/page_access_accounted/flush_trace.json) 的 paged 列，与 R1 §8.4 公式一致。

| $N$ | $K$ quant | $K$ res | $V$ quant | $V$ res | PTE 字节 | 本步是否刷窗 |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 0 | 1 | 0 | 1 | 16 | 否 |
| 17 | 0 | 2 | 0 | 2 | 32 | 否 |
| 127 | 0 | 8 | 0 | 8 | 128 | 否 |
| 128 | 8 | 0 | 0 | 8 | 128 | 是（Key 整窗） |
| 129 | 8 | 1 | 1 | 8 | 144 | 是（Value 溢出 1） |
| 256 | 16 | 0 | 8 | 8 | 256 | 是 |
| 257 | 16 | 1 | 9 | 8 | 272 | 是 |

$N{<}128$ 时两侧都在 FP16 残差，HBM packed 写为 0。$N{=}128$ 时 Key 整窗变为 8 个量化页，Value 仍为 8 个 FP16 页；该步 HBM 写/读逻辑均为 81,920 B（Key 载荷 65,536 B 加 scale/min 各 8,192 B）。$N{=}129$ 起 Value 溢出立即量化，不得为凑满页推迟；第 256 步对未满的 Value 量化尾页做一次读改写（separate 放置记 2 条 `hbm_rmw_read`，对应载荷与元数据），第 257 步则新开量化页、无 RMW。

contiguous 在相同 $N$ 上残差 token 数与 packed 字节与 paged 相同，页数为 0；刷窗仍把量化历史写入 HBM 块，只是没有 PTE。

下表为一次追加自身的流量，不含随后执行的扫描，控制覆盖字节不加入实际读写。单位 B。双布局各 257 个追加及扫描时刻，共 514 个时刻、1,028 项事件核对通过。

| $N$ | HBM 写 | HBM 读改写读取 | SRAM 数据读 | SRAM 数据写 | PTE 写 | 标签写 |
|---:|---:|---:|---:|---:|---:|---:|
| 127 | 0 | 0 | 0 | 4,096 | 0 | 0 |
| 128 | 81,920 | 0 | 262,144 | 4,096 | 64 | 8 |
| 129 | 640 | 0 | 264,192 | 266,240 | 24 | 3 |
| 256 | 92,160 | 9,600 | 526,336 | 266,240 | 72 | 9 |
| 257 | 640 | 0 | 264,192 | 266,240 | 24 | 3 |

### 3.4 元数据共址与事务切分

同一输入、同一编码下，`separate` 与 `colocated` 解码一致，packed 载荷字节一致。Llama/Qwen 主几何的满页载荷与 scale 已是 32 B 倍数，两种放置的 HBM 物理字节相同，差别在事务条数。

| 格子 | separate 写事件 | colocated 写事件 | 写物理字节 |
|---|---:|---:|---:|
| C2 Llama $T{=}17$ | 4 | 2 | 18,432 |
| C2 探针 $T{=}16$ | 4 | 2 | 576 |
| C5 Llama $T{=}129$ | 14 | 9 | 82,560 |
| C2/C5 探针未满残差窗 | 0 | 0 | 0 |

C2 探针 $T{=}16$：两侧载荷 $16{\times}1{\times}32{\times}1/2=256$ B、scale $16{\times}1{\times}2{\times}2=64$ B，合计 576 B，对齐浪费为 0。$T{=}1$ 的均匀路径仍全部在 SRAM 尾页，没有 HBM packed 事务。共址把每页的两次事务合成一次，不在本几何上减少取整后的物理字节。

### 3.5 写侧计量与验收覆盖

来源：[access_contract.json](results/page_access_accounted/access_contract.json)，参数、环境与源码哈希见同目录 [run_config.json](results/page_access_accounted/run_config.json)。正式入口自动执行这些检查；单独复现可运行 `python smoke/access_accounting_review.py --output results/access_accounting_recheck`，沿用上文 CPU 环境及非空目录保护。

1. **满页读取 3/3 通过**：C0/C2/C3、Llama KV 几何从 15 token 追加到 16 token，既有尾页为 61,440 B，加上新 token 的 4,096 B 后，实际记录整页 SRAM 读取 **65,536 B**。该值与“先写尾缓冲、再整页读取”的声明一致。
2. **独立追加 80/80 通过**：非连续输入和不规则追加跨过多个页、组和残差窗；解码与相同切块的 R1 参照一致，每次追加及扫描事件与独立预期一致。头数为 3 的探针亦通过。
3. **计量故障 32/32 被发现**：11 类事件的遗漏均触发失败，重复、错误字节数、错误层级和错误 K/V 归属也均触发失败；两种元数据放置都受检。
4. **元数据故障对照 5/5 符合预期**：屏蔽 HBM 写事件后，三个实际具有 packed 页的格子均失败；两个全部驻留 SRAM 的格子继续通过。后者没有应被记录的 HBM 写入，不应误报故障。

## 4. 分析与讨论

步骤 3 证明名义 INT4 字节在连续缓冲上可寻址。本步把同一 pack 原语用到页槽：满页才提交 HBM，未满 token 作为独立 FP16 页保留，因此尾页的**实际驻留**和 R1 报表里的**名义量化字节**必须分列。C0 尾页恰好同为 FP16，两口径重合；C2/C3 尾页在 DUT 中比 R1 名义 INT4 更大，因为它还没有被量化。不能用 R1 `bytes_breakdown` 的 payload 直接代替物理尾页占用。

C5 的开口组（最多 31 token）和残差窗（$R=128$）不是同一对象。残差窗按 R1 语义在 Key 上可被刷空，Value 则始终保有最近 $\min(N,R)$ token 的 FP16。$N{=}128$ 的页数之和仍是 $2\lceil N/16\rceil$，没有因为分池多出一页；$N{=}129$ 才因 Key 新残差页和 Value 量化尾页使 PTE 从 128 B 升到 144 B。这与 R1 分页布局的四池公式一致，但载荷已经是 packed nibble，而不是 int8 网格。

`scan_read` 把一次 decode 步的读侧建成“当前全部页 + 残差”的事件和，而不是 tile 级流式供给。因此 HBM 读逻辑等于当前历史占用，均值/尾延迟轨迹里的逐步读字节随 $N$ 上升，不能解释成 Attention 阵列已经按页消费。写侧则只在满页、刷窗或尾页 RMW 时出现 HBM 事务，所以刷窗附近会有写尖峰，而 $N{<}128$ 的 C5 逐步追加可以完全没有 HBM packed 写。

元数据共址在已对齐的主几何上不减少物理字节，只减少事务条数。若后续把 DMA 突发定得比 32 B 更大，或把很短的 scale/min 单独发送，共址才可能改变取整浪费。本步把该差异记下来，供步骤 8 的布局探索使用，不把它写成周期或能耗收益。

未满 Value 量化尾页的读改写是“溢出立即量化、又不为凑页插入 dummy token”的直接代价。第 256 步的写路径包含 92,160 B HBM 写和 9,600 B HBM 读改写读取；二者之和为 101,760 B，但不能把这个和全称为写字节。一次 append 内的新片段直接成页，避免将尚可合并的新片段先写成短页再读回。

紧凑残差保留布局也有明确成本。第 129 步 Value 刷出一个 token，需要读取其 2,048 B；保留的 128 token 为 262,144 B，重排再读写各一次，因此 SRAM 数据读为 264,192 B，数据写为新 K/V 的 4,096 B 加保留残差的 262,144 B。HBM 写量很小并不代表整个写路径的数据移动也小。环形残差缓冲可能改变这部分成本，但当前实验没有实现或验证该机制。

功能、存储和计量分别受到检查。独立预期不依赖 DUT 的物理页对象，故障注入证明非空事件可以被有效监测；控制覆盖字节单列后，刷窗动作与其 SRAM 读取不重复累加。该证据支持声明布局上的分层字节计量，尚不支持从字节数直接推出周期或能耗。

## 5. 局限与有效性

- 完成范围限于共享配置定义的数据移动模型。Python `clone/cat` 和分配器开销不等于目标硬件流量；量化器内部多遍统计、bank/端口、活动描述符实现及页槽物理地址尚未校准。PTE 字节与事件为声明模型，不是已实现的硬件页表。
- 范围为物理页表、残差刷窗与事件计数；没有 QK、online softmax、PV、跨页归并或背压。
- 32 B 是布局事务量子，不是测量得到的 HBM 突发、SRAM bank 或端口宽度。`hbm_*_allocated` 与事件物理字节都按该量子取整，不得称为芯片实测流量。
- 残差在 SRAM、历史在 HBM 是本步冻结的驻留假设。没有容量模型证明残差窗一定装得下，也没有把 PTE 放到片外。
- `scan_read` 是全缓存读，不是流式 tile 供给，不能当作步骤 5 的功能证据。
- C3 只保证与**相同追加切块**的参照一致；contiguous 逐 token 与两段 encode 的差不进入失败门。
- 未接入模型权重或质量门；合成张量不能代替 WikiText / 任务评测。
- 缓存张量实测不含 Python 对象、分配器保留内存、C3 旋转矩阵和 `load` 临时工作区。
- 对拍在 WSL CPU、torch 2.13.0+cpu 上完成。打包是整数位操作，不依赖 GPU；集群未跑本实验，也不证明 GPU 路径已就绪。
- 实验入口与结果在本地保存；GitHub 发布范围遵循根规则，未入库数据不视为可从 GitHub 直接获取。

## 6. 结论与后续工作

物理 packed 分页、C5 四池刷窗、尾页驻留、输入所有权，以及声明布局下的分层读写、PTE/标签和元数据事务均已有通过的机器证据。主网格、非连续输入与混合追加、全程刷窗轨迹及故障检测均满足各自验收条件。当前结果支持步骤 4 在本报告范围内提交用户审核。

经用户审核后，下一步为完整流式 Attention 与优化 C3，完成 QK、online softmax 归并和 PV。后续供给与成本模型应继续使用分层读写事件，显式承担残差重排和 Value 尾页读改写成本，并核验缓冲容量、端口及背压。本报告的 `scan_read` 不构成完整流式功能证据。
