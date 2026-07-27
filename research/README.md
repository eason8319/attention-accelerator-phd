# 正式研究目录（自 R1 起）

> 现行计划：[`docs/research_plan.md`](../docs/research_plan.md)。  
> R0（`survey/` + `learning/`）已完成技能与文献地图；**本目录承接正式研究**。

## 当前状态

| 深度 | 状态 | 说明 |
|------|------|------|
| R0 | 完成 | 见 `survey/`、`learning/`、`docs/lit_watch/` |
| R1 | **进行中** | M0–M2 完成；下一步 M3 KIVI |
| R2–R5 | 未开始 | 见研究计划验收门槛 |

## 布局

```text
research/
  README.md                 # 本文件
  r1_kv_baseline/           # 真实 cache-path、评测协议、Pareto（见该目录 README）
  r1_decode_sim/            # 专用 decode 模拟器骨架（M7）
```

环境与依赖：[`r1_kv_baseline/README.md`](r1_kv_baseline/README.md)。进度见 [`docs/progress/milestones.md`](../docs/progress/milestones.md)。

## R1 最低交付（摘录）

1. Token-wise KV：quantize → pack/store → load → dequant → attention（对照 FP16 / INT8 / INT4±BDR）  
2. contiguous **与** paged 双报告  
3. bytes/token–精度 Pareto（至少一种长上下文设定）  
4. 与 Roofline / SCALE-Sim 的相对趋势交叉检查协议  
5. 验收时在总报告中写明相对投影假量化路径的差异（正式结果以真实 cache-path 为准；日常不另写说明文档）  

对照锚与文献台账：[`docs/recent_works_comparison.md`](../docs/recent_works_comparison.md)、[`docs/lit_watch/`](../docs/lit_watch/)。
