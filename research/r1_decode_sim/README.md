# R1 Decode Simulator

本目录承接 M7 的独立模拟器实现。当前进度以 [研究里程碑](../../docs/progress/milestones.md) 为准，输入口径与验收要求见 [R1 实施计划](../r1_kv_baseline/PLAN.md)。

M5 流量输入位于 `../r1_kv_baseline/experiments/kv_pareto/results/summary.json`；完整实验源码、结果仅保存在本地与服务器，单独克隆 GitHub 不包含这些输入。

实现可参考 P5 的设计，运行时不得导入 `learning/`。模拟器实现留在本目录，实验、冒烟测试和结果放在 `experiments/<name>/`。CPU 即可开展流量接口与趋势检查；只有涉及真实模型重测时才需另行申请 GPU。
