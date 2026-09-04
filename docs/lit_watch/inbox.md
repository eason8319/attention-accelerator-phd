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

（当前为空。下次检索从这里追加。）

## 最近一次检索记录

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
