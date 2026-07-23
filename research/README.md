# 正式研究目录（自 R1 起）

> 现行计划：[`docs/research_plan.md`](../docs/research_plan.md)。  
> R0（`survey/` + `learning/`）已完成技能与文献地图；**本目录承接正式研究**。

## 当前状态

| 深度 | 状态 | 说明 |
|------|------|------|
| R0 | 完成 | 见 `survey/`、`learning/`、`docs/lit_watch/` |
| R1 | **下一步** | 真实 KV cache-path + 误差—流量模型 + decode simulator 骨架 |
| R2–R5 | 未开始 | 见研究计划验收门槛 |

## 建议布局（实施时创建）

```text
research/
  README.md                 # 本文件
  r1_kv_baseline/           # 真实 cache-path、评测协议、Pareto
  r1_decode_sim/            # 专用 decode 模拟器骨架
  protocols/                # 锁定的模型列表、硬件包络、检查点误差阈值
```

在创建子目录前，先在 [`docs/progress/milestones.md`](../docs/progress/milestones.md) 将 R1 标为进行中，并更新 CHANGELOG。

## R1 最低交付（摘录）

1. Token-wise KV：quantize → pack/store → load → dequant → attention（对照 FP16 / INT8 / INT4±BDR）  
2. contiguous **与** paged 双报告  
3. bytes/token–精度 Pareto（至少一种长上下文设定）  
4. 与 Roofline / SCALE-Sim 的相对趋势交叉检查协议  
5. 文档化相对 `learning/p2` proxy 的差异  

对照锚与文献台账：[`docs/recent_works_comparison.md`](../docs/recent_works_comparison.md)、[`docs/lit_watch/`](../docs/lit_watch/)。
