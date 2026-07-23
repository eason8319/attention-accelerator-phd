# 固定检索词（Literature Watch）

> 更新对比手册前，至少跑一遍下列查询。记录日期写入 `inbox.md`。

## 数据源

1. [arXiv API / abs](https://arxiv.org/)：`cs.AR`, `cs.LG`, `cs.CL`
2. [ACL Anthology](https://aclanthology.org/)
3. [IEEE Xplore](https://ieeexplore.ieee.org/)（HPCA / MICRO / ISCA / DAC / FPGA / TC / TCAD）
4. [OpenReview](https://openreview.net/)（ICML / ICLR / NeurIPS）
5. [PMLR](https://proceedings.mlr.press/)

## 核心查询（英文）

```text
"KV cache" quantization
"KV cache" mixed-precision OR "mixed precision" quantization
low-bit OR INT4 OR INT2 OR MXFP4 "KV" decode
BitDecoding OR FlashDecoding "KV"
"attention accelerator" OR systolic FlashAttention
"paged" "KV" quantization serving
RoPE-aware OR "bit allocation" "KV cache"
```

## 硬件向补充查询

```text
FlashAttention systolic OR "online softmax" accelerator
flattened systolic LLM OR PLENA accelerator
FPGA LLM KV cache OR FlightLLM OR AccLLM
long-context attention ASIC decode
```

## 排除（默认不进主对照，可进相邻）

```text
distributed training, MoE routing-only, pure CIM/PIM without attention datapath,
training-time quantization only (no inference KV)
```

## arXiv 日期提示

arXiv id `YYmm.xxxxx`：`2607.*` ≈ 2026-07。对比手册 Cutoff 应不早于最近一次成功检索日。
