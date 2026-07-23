"""exp 近似：范围规约 + 分段线性（PWL），供 P4 模块 A 定误差预算与 Q 格式。

算法（对齐 Softermax / FSA）：

$$
\\exp(x) = 2^{x \\log_2 e} = 2^{y_i + y_f} = 2^{y_i}\\cdot 2^{y_f},
\\quad y = x\\log_2 e,\\; y_f \\in [0,1)
$$

对 $2^{y_f}$ 在 $[0,1)$ 上做均匀 PWL。attention 减 max 后 $x\\le 0$，故
$\\exp(x)\\in(0,1]$。

用法（conda env ``p4-rtl``）::

    cd learning/p4_rtl
    conda run -n p4-rtl python scripts/exp_approx.py          # 扫参 + 写笔记
    conda run -n p4-rtl python scripts/exp_approx.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fixedpoint import (  # noqa: E402
    QFormat,
    max_abs_err,
    quantize,
    rel_err,
)

LOG2_E = float(np.log2(np.e))
P1_DIR = Path(__file__).resolve().parents[2] / "p1_attention_numerics"
NOTES_DIR = Path(__file__).resolve().parents[1] / "notes"

# 学习计划验收：目标输入范围内相对误差 < 1e-3
REL_BUDGET = 1e-3
# 评估域：真实 score 主质量区（见扫参笔记）；更负尾部 exp≈0，不考核相对误差
TARGET_X_MIN = -8.0
REF_FLOOR = 1e-4

# 扫参选定后固化（rtl 与 gen_vecs 共用）
CHOSEN_N_SEG = 16
CHOSEN_FMT_IN = QFormat(6, 10)  # Q6.10，W=16，覆盖 [-32,32)
CHOSEN_FMT_OUT = QFormat(0, 24, signed=False)  # UQ0.24，小 exp 量化台阶够细


# --------------------------------------------------------------------------- PWL


@dataclass(frozen=True)
class PWLTable:
    """均匀分段线性表：在 [0, 1) 上逼近 2^f。"""

    n_seg: int
    slopes: np.ndarray
    intercepts: np.ndarray

    @classmethod
    def for_exp2_frac(cls, n_seg: int) -> PWLTable:
        if n_seg < 1:
            raise ValueError("n_seg must be >= 1")
        edges = np.linspace(0.0, 1.0, n_seg + 1)
        vals = np.exp2(edges)
        slopes = (vals[1:] - vals[:-1]) / (edges[1:] - edges[:-1])
        intercepts = vals[:-1]
        return cls(n_seg=n_seg, slopes=slopes, intercepts=intercepts)

    def eval(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        f_c = np.clip(f, 0.0, np.nextafter(1.0, 0.0))
        idx = np.minimum((f_c * self.n_seg).astype(np.int64), self.n_seg - 1)
        left = idx.astype(np.float64) / self.n_seg
        return self.slopes[idx] * (f_c - left) + self.intercepts[idx]


def exp_approx_float(x: np.ndarray | float, table: PWLTable) -> np.ndarray:
    """浮点路径：范围规约 + PWL（无 I/O 量化）。"""
    x = np.asarray(x, dtype=np.float64)
    y = x * LOG2_E
    y_i = np.floor(y).astype(np.int64)
    y_f = y - y_i.astype(np.float64)
    return np.ldexp(table.eval(y_f), y_i)


def exp_approx_fixed(
    x: np.ndarray | float,
    table: PWLTable,
    fmt_in: QFormat,
    fmt_out: QFormat,
) -> np.ndarray:
    """RTL 友好路径：输入量化 → 规约+PWL → 输出量化。

    中间仍用 float64 做规约/PWL，模拟「定点 I/O + 内部足够精度」的误差下界。
    """
    x_q = quantize(x, fmt_in)
    y_approx = exp_approx_float(x_q, table)
    return quantize(y_approx, fmt_out)


def chosen_table() -> PWLTable:
    return PWLTable.for_exp2_frac(CHOSEN_N_SEG)


def exp_approx_chosen(x: np.ndarray | float) -> np.ndarray:
    """使用扫参选定配置的定点路径（模块 A golden）。"""
    return exp_approx_fixed(x, chosen_table(), CHOSEN_FMT_IN, CHOSEN_FMT_OUT)


# --------------------------------------------------------------------------- 真实 score 采样


def sample_attention_exp_inputs(
    *,
    batch: int = 1,
    heads: int = 4,
    seq: int = 512,
    head_dim: int = 64,
    block_size: int = 64,
    causal: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """从 P1 online softmax 采样 ``scores - m_new`` 与 ``m - m_new``（均 ≤ 0）。"""
    if str(P1_DIR) not in sys.path:
        sys.path.insert(0, str(P1_DIR))
    from attention_naive import _causal_mask  # type: ignore
    from attention_tiled import _iter_kv_blocks  # type: ignore

    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, generator=gen)
    k = torch.randn(batch, heads, seq, head_dim, generator=gen)

    scale = head_dim**-0.5
    m = torch.full((batch, heads, seq), float("-inf"))
    causal_mask = _causal_mask(seq, q.device, q.dtype) if causal else None
    chunks: list[torch.Tensor] = []

    for kv_start, kv_end in _iter_kv_blocks(seq, block_size):
        k_blk = k[:, :, kv_start:kv_end, :]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale
        if causal:
            scores = scores + causal_mask[:, kv_start:kv_end]

        finite = torch.isfinite(scores)
        m_blk = scores.masked_fill(~finite, float("-inf")).amax(dim=-1)
        m_new = torch.maximum(m, m_blk)

        s_centered = (scores - m_new.unsqueeze(-1))[finite]
        chunks.append(s_centered.detach().flatten())

        alpha_x = (m - m_new).flatten()
        alpha_ok = torch.isfinite(alpha_x) & (alpha_x > float("-inf")) & (alpha_x <= 0)
        if alpha_ok.any():
            chunks.append(alpha_x[alpha_ok].detach())

        m = m_new

    x = torch.cat(chunks).numpy().astype(np.float64)
    return np.minimum(x, 0.0)


def score_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "n": float(x.size),
        "min": float(x.min()),
        "p1": float(np.percentile(x, 1)),
        "p5": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "frac_gt_neg8": float(np.mean(x > -8.0)),
        "frac_gt_neg16": float(np.mean(x > -16.0)),
        "frac_gt_neg32": float(np.mean(x > -32.0)),
    }


# --------------------------------------------------------------------------- 扫参


@dataclass(frozen=True)
class SweepConfig:
    n_seg: int
    fmt_in: QFormat
    fmt_out: QFormat


@dataclass
class SweepResult:
    cfg: SweepConfig
    max_rel: float
    mean_rel: float
    p99_rel: float
    max_abs: float
    n_eval: int
    pass_budget: bool
    float_max_rel: float  # 同段数纯浮点 PWL，用于拆分算法/量化误差


def evaluate(
    x: np.ndarray,
    cfg: SweepConfig,
    *,
    ref_floor: float = REF_FLOOR,
) -> SweepResult:
    table = PWLTable.for_exp2_frac(cfg.n_seg)
    ref = np.exp(x)
    dut = exp_approx_fixed(x, table, cfg.fmt_in, cfg.fmt_out)
    float_dut = exp_approx_float(x, table)

    mask = ref >= ref_floor
    if not np.any(mask):
        mask = np.ones_like(ref, dtype=bool)

    re = rel_err(ref[mask], dut[mask])
    re_f = rel_err(ref[mask], float_dut[mask])
    max_rel = float(re.max())
    return SweepResult(
        cfg=cfg,
        max_rel=max_rel,
        mean_rel=float(re.mean()),
        p99_rel=float(np.percentile(re, 99)),
        max_abs=max_abs_err(ref, dut),
        n_eval=int(mask.sum()),
        pass_budget=max_rel < REL_BUDGET,
        float_max_rel=float(re_f.max()),
    )


def default_sweep_grid() -> list[SweepConfig]:
    # Q4.12 范围仅 [-8,8)，真实 score min≈-10 会饱和，不纳入候选
    fmt_ins = [
        QFormat(8, 8),
        QFormat(6, 10),
        QFormat(8, 10),
        QFormat(8, 12),
    ]
    fmt_outs = [
        QFormat(2, 14, signed=False),
        QFormat(1, 15, signed=False),
        QFormat(0, 16, signed=False),
        QFormat(0, 20, signed=False),
        QFormat(0, 24, signed=False),  # 优先 24-bit 对齐；略过 22-bit 非对齐宽度
        QFormat(1, 23, signed=False),
    ]
    segs = [4, 8, 16, 32]
    return [
        SweepConfig(n_seg=n, fmt_in=fin, fmt_out=fout)
        for n in segs
        for fin in fmt_ins
        for fout in fmt_outs
    ]


def pick_config(
    results: list[SweepResult],
    *,
    x_min_all: float | None = None,
) -> SweepResult:
    """优先过预算；输入范围须覆盖全样本 min；同档选总位宽小、段数少者。"""

    pool = results
    if x_min_all is not None:
        covered = [r for r in results if r.cfg.fmt_in.min_val <= x_min_all]
        if covered:
            pool = covered

    def key(r: SweepResult) -> tuple:
        return (
            0 if r.pass_budget else 1,
            r.cfg.fmt_in.width + r.cfg.fmt_out.width,
            r.cfg.n_seg,
            r.max_rel,
        )

    return min(pool, key=key)


# --------------------------------------------------------------------------- 报告


def write_notes(
    path: Path,
    stats: dict[str, float],
    results: list[SweepResult],
    chosen: SweepResult,
    x_all: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # 对照：经典 16/16 总线配置
    baseline = next(
        (
            r
            for r in results
            if r.cfg.n_seg == 8 and str(r.cfg.fmt_in) == "Q8.8" and str(r.cfg.fmt_out) == "UQ0.16"
        ),
        None,
    )

    lines: list[str] = []
    lines.append("# exp 近似扫参笔记（模块 A / Python）\n")
    lines.append(
        "> 由 `scripts/exp_approx.py` 生成。对应学习计划：范围规约 + PWL，"
        "用真实 attention score 定误差预算与 Q 格式。\n"
    )
    lines.append("## 算法\n")
    lines.append(
        "$$\n"
        "\\exp(x)=2^{x\\log_2 e}=2^{y_i+y_f}=2^{y_i}\\cdot\\mathrm{PWL}(2^{y_f}),"
        "\\quad y_f\\in[0,1)\n"
        "$$\n"
    )
    lines.append(
        f"PWL 在段端点精确匹配 $2^{{y_f}}$。"
        f"误差预算：**相对误差 $< {REL_BUDGET:g}$**，"
        f"统计域为 $x\\in[{TARGET_X_MIN:g},0]$ 且 $\\exp(x)\\ge {REF_FLOOR:g}$。\n"
    )
    lines.append("## 真实输入分布\n")
    lines.append(
        "从 P1 online softmax 数据流采样 `scores - m_new`（$P$ 路径）与 "
        "`m - m_new`（$\\alpha$ 路径）；batch=1, heads=4, seq=512, $d$=64, "
        f"block=64, causal。共 ${stats['n']:.0f}$ 个有限样本。\n"
    )
    lines.append("| 统计 | 值 |")
    lines.append("|------|-----|")
    for k in ("min", "p1", "p5", "median", "p95", "p99", "max"):
        lines.append(f"| {k} | ${stats[k]:.4f}$ |")
    lines.append(f"| 比例 $x>-8$ | ${stats['frac_gt_neg8']:.4f}$ |")
    lines.append(f"| 比例 $x>-16$ | ${stats['frac_gt_neg16']:.4f}$ |")
    lines.append(f"| 比例 $x>-32$ | ${stats['frac_gt_neg32']:.4f}$ |")
    lines.append("")
    lines.append(
        f"主质量几乎全在 $[{TARGET_X_MIN:g},0]$（本采样 min=${stats['min']:.2f}$）。"
        "更负尾部对 softmax 权重可忽略，故不纳入相对误差验收。\n"
    )
    lines.append("## 误差拆分（浮点 PWL vs 定点 I/O）\n")
    lines.append(
        "纯浮点 PWL（无量化）在 8 段时 max rel 已 $<10^{-3}$；"
        "W=16 输出（`UQ0.16`）在小 $\\exp(x)$ 处量化台阶主导 max rel（约 $2\\%$）。"
        "因此要满足 $<10^{-3}$，需 **更细的输出小数位**，输入亦不宜粗于 ~10 bit 小数。\n"
    )
    lines.append("| 段数 | 浮点 PWL max rel |")
    lines.append("|------|------------------|")
    seen_seg: set[int] = set()
    for r in sorted(results, key=lambda t: t.cfg.n_seg):
        if r.cfg.n_seg in seen_seg:
            continue
        seen_seg.add(r.cfg.n_seg)
        lines.append(f"| {r.cfg.n_seg} | ${r.float_max_rel:.3e}$ |")
    lines.append("")
    lines.append("## 扫参摘录（每档段数中总位宽最小且过预算；否则该档最优）\n")
    lines.append("| 段数 | 输入 | 输出 | $\\sum$W | max rel | p99 rel | 过预算 |")
    lines.append("|------|------|------|--------|---------|---------|--------|")
    for n in sorted({r.cfg.n_seg for r in results}):
        pool = [r for r in results if r.cfg.n_seg == n]
        passed = [r for r in pool if r.pass_budget]
        best = (
            pick_config(passed, x_min_all=stats["min"])
            if passed
            else pick_config(pool, x_min_all=stats["min"])
        )
        mark = "Y" if best.pass_budget else "N"
        tw = best.cfg.fmt_in.width + best.cfg.fmt_out.width
        lines.append(
            f"| {best.cfg.n_seg} | `{best.cfg.fmt_in}` | `{best.cfg.fmt_out}` | "
            f"{tw} | {best.max_rel:.3e} | {best.p99_rel:.3e} | {mark} |"
        )
    if baseline is not None:
        lines.append("")
        lines.append(
            f"对照（面积友好、未过预算）：8 段 `Q8.8`/`UQ0.16` → "
            f"max rel=${baseline.max_rel:.3e}$，p99=${baseline.p99_rel:.3e}$。\n"
        )
    lines.append("## 选定配置\n")
    c = chosen.cfg
    tw = c.fmt_in.width + c.fmt_out.width
    lines.append(f"- **段数**：{c.n_seg}")
    lines.append(
        f"- **输入**：`{c.fmt_in}`（width={c.fmt_in.width}，范围 $[{c.fmt_in.min_val:g},{c.fmt_in.max_val:g}]$）"
    )
    lines.append(
        f"- **输出**：`{c.fmt_out}`（width={c.fmt_out.width}，LSB=$2^{{-{c.fmt_out.frac_bits}}}$）"
    )
    lines.append(f"- **总 I/O 位宽**：{tw}")
    lines.append(
        f"- **误差**（目标域）：max rel=${chosen.max_rel:.3e}$，"
        f"p99 rel=${chosen.p99_rel:.3e}$，mean rel=${chosen.mean_rel:.3e}$，"
        f"max abs=${chosen.max_abs:.3e}$，n={chosen.n_eval}"
    )
    lines.append(f"- **预算**：{'通过' if chosen.pass_budget else '未通过'} （$<{REL_BUDGET:g}$）")
    lines.append("")
    lines.append("### 选型理由\n")
    lines.append(
        "1. 在过预算配置中优先 **总 I/O 位宽更小**，其次更少 PWL 段。\n"
        "2. `Q6.10` 仍为 16-bit 输入总线，范围 $[-32,32)$ 覆盖真实 min "
        f"（${stats['min']:.2f}$）；比 `Q8.8` 多 2 bit 小数，压低输入量化误差。\n"
        "3. `UQ0.24` 使小 $\\exp(x)$ 的量化相对误差落入预算；"
        "比 `UQ0.16` 多 8 bit 小数是满足 $<10^{-3}$ 的关键。\n"
        "4. 16 段 PWL：浮点算法误差已远低于量化误差，再增到 32 段收益有限。\n"
    )
    lines.append("## 对后续 RTL 的约束\n")
    lines.append(
        f"- 流水线与 `exp_approx_fixed` / `exp_approx_chosen` 一致："
        f"量化为 `{c.fmt_in}` → $\\times\\log_2 e$ → 拆 $y_i/y_f$ → "
        f"{c.n_seg} 段 PWL → 按 $y_i$ 算术移位 → 量化 `{c.fmt_out}`。\n"
        f"- 对拍阈值：目标域相对误差 $< {REL_BUDGET:g}$（$|\\mathrm{{ref}}|\\ge {REF_FLOOR:g}$）。\n"
        f"- $x<{TARGET_X_MIN:g}$ 允许饱和到输出最小值，不单独考核相对误差。\n"
        "- 常量 $\\log_2 e$ 可用定点乘法；PWL 斜率/截距可固化为 ROM/"
        "组合逻辑表。\n"
    )
    # 全量域上的健全性检查（不作为验收）
    all_r = evaluate(x_all, chosen.cfg, ref_floor=REF_FLOOR)
    lines.append(
        f"- 全样本（含极负尾）同配置 max rel=${all_r.max_rel:.3e}$（仅供参考，非验收）。\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sweep(*, write: bool = True) -> SweepResult:
    x_all = sample_attention_exp_inputs()
    stats = score_stats(x_all)
    x = x_all[x_all >= TARGET_X_MIN]

    print("=== attention exp-input stats (all) ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.6g}")
    print(f"=== eval domain: x in [{TARGET_X_MIN:g}, 0], n={x.size} ===")

    results = [evaluate(x, cfg) for cfg in default_sweep_grid()]
    chosen = pick_config(results, x_min_all=float(x_all.min()))

    print("\n=== configs that pass budget ===")
    passed = [r for r in results if r.pass_budget]
    print(f"  {len(passed)} / {len(results)}")
    for r in sorted(passed, key=lambda t: (t.cfg.fmt_in.width + t.cfg.fmt_out.width, t.cfg.n_seg))[
        :12
    ]:
        tw = r.cfg.fmt_in.width + r.cfg.fmt_out.width
        print(
            f"  seg={r.cfg.n_seg:2d} {r.cfg.fmt_in!s:>8}/{r.cfg.fmt_out!s:<8} "
            f"ΣW={tw:2d} max_rel={r.max_rel:.3e} p99={r.p99_rel:.3e}"
        )

    print("\n=== chosen ===")
    print(
        f"  seg={chosen.cfg.n_seg} in={chosen.cfg.fmt_in} out={chosen.cfg.fmt_out} "
        f"max_rel={chosen.max_rel:.3e} pass={chosen.pass_budget}"
    )
    # 与模块常量一致性检查
    assert chosen.cfg.n_seg == CHOSEN_N_SEG, (chosen.cfg.n_seg, CHOSEN_N_SEG)
    assert chosen.cfg.fmt_in == CHOSEN_FMT_IN, (chosen.cfg.fmt_in, CHOSEN_FMT_IN)
    assert chosen.cfg.fmt_out == CHOSEN_FMT_OUT, (chosen.cfg.fmt_out, CHOSEN_FMT_OUT)

    if write:
        out = NOTES_DIR / "exp_approx_sweep.md"
        write_notes(out, stats, results, chosen, x_all)
        print(f"\nnotes written: {out}")

    return chosen


# --------------------------------------------------------------------------- 自检


def _self_test() -> None:
    table = PWLTable.for_exp2_frac(8)
    assert abs(float(exp_approx_float(0.0, table)) - 1.0) < 1e-12
    xs = np.linspace(-8.0, 0.0, 1001)
    err = rel_err(np.exp(xs), exp_approx_float(xs, table))
    assert float(err.max()) < 5e-3, err.max()

    dut = exp_approx_chosen(xs)
    assert dut.shape == xs.shape
    # 选定配置在网格上应过预算
    r = evaluate(xs, SweepConfig(CHOSEN_N_SEG, CHOSEN_FMT_IN, CHOSEN_FMT_OUT))
    assert r.pass_budget, (r.max_rel, r.p99_rel)

    x = sample_attention_exp_inputs(seq=128, heads=2, block_size=32)
    assert x.size > 0 and float(x.max()) <= 1e-6
    print("exp_approx self-test PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    chosen = run_sweep(write=not args.no_write)
    if not chosen.pass_budget:
        raise SystemExit(
            f"no config met rel budget {REL_BUDGET}; best max_rel={chosen.max_rel:.3e}"
        )


if __name__ == "__main__":
    main()
