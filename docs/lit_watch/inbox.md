# Inbox（未审候选）

> 只放**尚未核实入库**的条目。核实后移入 [`ledger.yaml`](ledger.yaml)，并从本文件删除或标 `→ ledger`。

## 使用方式

复制下面模板追加到「待审列表」。不要在此处填写未核对的加速比。

```markdown
### YYYY-MM-DD | 短标题
- link: https://arxiv.org/abs/XXXX.XXXXX
- why: 一句话为何可能相关
- suggested_bucket: algo_gpu | hw_asic_fpga | survey | adjacent
- status: pending
```

## 待审列表

### 2026-09-15 | SemKV
- link: https://arxiv.org/abs/2608.28911
- why: 混合精度 KV 与质量悬崖设定，可能收窄通道/token 混合精度对照。
- suggested_bucket: algo_gpu
- status: pending

### 2026-09-15 | RotateKV
- link: https://arxiv.org/abs/2501.16383
- why: MixKVQ 等旋转 2-bit 基线族；IJCAI 2025 会刊页待核验入库。
- suggested_bucket: algo_gpu
- status: pending

### 2026-09-15 | SKVQ
- link: https://arxiv.org/abs/2405.06219
- why: 滑动窗口 KV 量化，静态低比特与时间窗机制近邻。
- suggested_bucket: algo_gpu
- status: pending

### 2026-09-15 | KVQuant
- link: https://arxiv.org/abs/2401.18079
- why: 亚 4-bit KV 与长上下文服务内核，静态低比特近邻。
- suggested_bucket: algo_gpu
- status: pending

### 2026-09-15 | VitaLLM（两个 arXiv 标识）
- link: https://arxiv.org/abs/2604.27396
- why: 三值权重 + 稀疏 KV 取数加速器，相邻硬件；另见 https://arxiv.org/abs/2605.00320，须先核验是否同一工作的版本。
- suggested_bucket: adjacent
- status: pending

### 2026-09-15 | VEDA
- link: https://arxiv.org/abs/2507.00797
- why: 投票驱逐 KV + 数据流加速器，稀疏 decode 相邻对照。
- suggested_bucket: adjacent
- status: pending

### 2026-09-15 | SA-ANT
- link: https://doi.org/10.23919/date69613.2026.11539270
- why: DATE 2026 符号非对称自适应数值类型，与 M-ANT 同族，须核验是否覆盖 KV 还是仅权重。
- suggested_bucket: hw_asic_fpga
- status: pending

## 最近一次检索记录

本次 2026-09-15 第二轮按冻结范围做定向补录（decode × 混合/低比特 KV 流式 Attention，单芯片推理对照），不是全领域重新检索。核验了 Kitty、M-ANT、StreamAttention、COSA+、DESA；下列计划或检索中碰到的近邻未完成核实，留待审。

### 2026-09-15 | Kitty

- link: https://proceedings.mlsys.org/paper_files/paper/2026/hash/e4d8d1b5120be349d3fff8878650cf45-Abstract-Conference.html
- why: 通道混合精度 + paged 反量化，收窄混合格式/通道 paged 对照。
- suggested_bucket: algo_gpu
- status: verified → ledger；MLSys 2026 Oral；bibtex_key `kitty2026`

### 2026-09-15 | M-ANT

- link: https://doi.org/10.1109/HPCA61900.2025.00086
- why: 脉动阵列上的组级实时 KV 量化，硬件对照而非 KV-only 主行。
- suggested_bucket: hw_asic_fpga
- status: verified → ledger；HPCA 2025；bibtex_key `mant2025`

### 2026-09-15 | StreamAttention

- link: https://icml.cc/virtual/2026/75239
- why: 计划点名的 FA-native plumbing；须标清 workshop 而非主会。
- suggested_bucket: hw_asic_fpga
- status: verified → ledger；ICML 2026 AdaptFM workshop；bibtex_key `streamattention2025`

### 2026-09-15 | COSA+

- link: https://doi.org/10.1109/TCAD.2024.3434447
- why: 计划点名的合作脉动阵列注意力加速器。
- suggested_bucket: hw_asic_fpga
- status: verified → ledger；摘要级定量；bibtex_key `wang2023cosa`

### 2026-09-15 | DESA

- link: https://doi.org/10.1109/TC.2025.3549621
- why: 计划点名的脉动注意力加速器；摘要声明 encoder-only。
- suggested_bucket: hw_asic_fpga
- status: verified → ledger；摘要级定量；bibtex_key `desa2025`

### 2026-09-15 | KVmix

- link: https://ojs.aaai.org/index.php/AAAI/article/view/40422
- why: 计划中的层间混合精度 + 时间维高精度窗代表。
- suggested_bucket: algo_gpu
- status: verified → ledger；DOI 10.1609/aaai.v40i37.40422；bibtex_key `kvmix2026`

### 2026-09-15 | PM-KVQ

- link: https://openreview.net/forum?id=Vem6FQvRvq
- why: 计划中的渐进/块级混合精度与长 CoT 误差累积设定。
- suggested_bucket: algo_gpu
- status: verified → ledger；ICLR 2026 Poster，无 Crossref DOI；bibtex_key `pmkvq2026`

### 2026-09-15 | MixKVQ

- link: https://aclanthology.org/2026.acl-long.326/
- why: 计划中的 query 感知通道混合比特代表。
- suggested_bucket: algo_gpu
- status: verified → ledger；DOI 10.18653/v1/2026.acl-long.326；bibtex_key `mixkvq2026`

### 2026-09-15 | OSCAR（谱协方差；不是 OScaR）

- link: https://arxiv.org/abs/2605.17757
- why: 与计划 2-bit + 残差窗同类，且为 paged SGLang INT2 内核；必须与 OScaR 分列。
- suggested_bucket: algo_gpu
- status: verified → ledger；预印本；bibtex_key `zhou2026oscar`

### 2026-09-15 | OScaR（Occam；不是 OSCAR）

- link: https://arxiv.org/abs/2605.19660
- why: 计划点名的极限 2-bit + 旋转方法族。
- suggested_bucket: algo_gpu
- status: verified → ledger；预印本；bibtex_key `su2026oscar`

### 2026-09-11 | Flash-Decoding

- link: https://princeton-nlp.github.io/flash-decoding/
- why: R2 的 KV-split 与 log-sum-exp 归并基线；按作者技术说明核验。
- suggested_bucket: algo_gpu
- status: verified → ledger；已补手册与引用键，证据范围见卡片

### 2026-09-11 | QServe

- link: https://proceedings.mlsys.org/paper_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html
- why: R2 的 KV4 计算瓶颈、量化参数预取和公平 GPU 对照。
- suggested_bucket: algo_gpu
- status: verified → ledger；已补手册与引用键，证据范围见卡片

### 2026-09-11 | Multi-Scale Dequant

- link: https://arxiv.org/abs/2605.13915
- why: R2 的尺度折叠、激活分解和压缩收益边界近邻方法。
- suggested_bucket: algo_gpu
- status: verified → ledger；已补手册与引用键，证据范围见卡片

### 2026-09-03 | SPECTRA
- link: https://arxiv.org/abs/2608.07915
- why: 谱变换后按信息集中度分配通道比特，直接关联 R3 的结构感知混合精度。
- suggested_bucket: algo_gpu
- status: verified → ledger；full-text quantitative audit complete

### 2026-09-03 | Attention-Aware Transform Coding (AATC)
- link: https://arxiv.org/abs/2608.14191
- why: 用 attention-aware distortion 与 reverse water-filling 形式化比特分配，可作为 R3–R4 的理论参照。
- suggested_bucket: algo_gpu
- status: verified → ledger；full-text quantitative audit complete

### 2026-09-03 | Minima-KV
- link: https://arxiv.org/abs/2608.23834
- why: 混合格式 paged KV、分格式 partial attention、online-softmax 合并且无 dense shadow，与 R2–R3 高度重合。
- suggested_bucket: algo_gpu
- status: verified → ledger；full-text quantitative audit complete

### 2026-09-03 | PuzzleKV
- link: https://arxiv.org/abs/2608.23843
- why: 以逻辑页为低秩压缩与直接计算单元，为 R2 的 page/container 粒度提供相邻路线对照。
- suggested_bucket: adjacent
- status: verified → ledger；full-text quantitative audit complete
