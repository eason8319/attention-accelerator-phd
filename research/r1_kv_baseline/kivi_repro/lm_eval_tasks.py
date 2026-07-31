"""LM-Eval 生成任务封装（CoQA / TruthfulQA / GSM8K 等）。

将已加载的 HF 因果 LM（可经本仓库 KIVI patch）包装为 lm-eval ``HFLM``，
在每次 generate 前清空本仓库 Kivi cache。默认任务集可按需替换。

用法：

  from kivi_repro.hf_generate import load_llama_for_generate
  from kivi_repro.lm_eval_tasks import evaluate_lm_eval

  model, tok = load_llama_for_generate(..., kv_format="kivi2")
  bundle = evaluate_lm_eval(model, tok, kv_format="kivi2", limit=None)
  print(bundle.primary_scores)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch.nn as nn

from .hf_generate import kv_format_to_bits, load_llama_for_generate
from .llama_kivi_attn import clear_llama_kivi_caches
from .patch_llama import is_llama_kivi_patched

__all__ = [
    "DEFAULT_TASKS",
    "TASK_ALIASES",
    "PRIMARY_METRICS",
    "DEFAULT_MODEL_ID",
    "TaskScore",
    "LmEvalBundle",
    "resolve_tasks",
    "normalize_kv_format",
    "wrap_for_lm_eval",
    "run_lm_eval",
    "extract_primary_scores",
    "score_delta",
    "evaluate_lm_eval",
    "evaluate_lm_eval_formats",
]

# ---------------------------------------------------------------------------
# 默认任务与主指标（可被调用方覆盖）
# ---------------------------------------------------------------------------

DEFAULT_TASKS: tuple[str, ...] = ("coqa", "truthfulqa_gen", "gsm8k")

TASK_ALIASES: dict[str, str] = {
    "coqa": "coqa",
    "truthfulqa": "truthfulqa_gen",
    "truthfulqa_gen": "truthfulqa_gen",
    "truthful_qa": "truthfulqa_gen",
    "gsm8k": "gsm8k",
    "gsm": "gsm8k",
}

# 抽主分时的默认 metric 名（lm-eval 结果键前缀）；未知任务须由调用方传入
PRIMARY_METRICS: dict[str, str] = {
    "coqa": "em",
    "truthfulqa_gen": "bleu_max",
    "gsm8k": "exact_match",
}

DEFAULT_MODEL_ID = "NousResearch/Llama-2-7b-hf"

_FORMAT_ALIASES: dict[str, str] = {
    "fp16": "fp16",
    "16bit": "fp16",
    "c0": "fp16",
    "baseline": "fp16",
    "kivi2": "kivi2",
    "kivi_2": "kivi2",
    "c4": "kivi2",
    "kivi4": "kivi4",
    "kivi_4": "kivi4",
    "c5": "kivi4",
}


def resolve_tasks(tasks: str | Sequence[str] | None = "default") -> list[str]:
    """解析任务列表。

    参数
        tasks: ``None`` / ``"default"`` → ``DEFAULT_TASKS``；
          或逗号分隔字符串 / 序列（支持别名如 ``truthfulqa``）。
    """
    if tasks is None or tasks in ("default", "table3"):
        # ``table3`` 保留为 ``default`` 别名，避免旧脚本立刻断裂
        return list(DEFAULT_TASKS)
    if isinstance(tasks, str):
        parts = [p.strip() for p in tasks.split(",") if p.strip()]
    else:
        parts = [str(t).strip() for t in tasks]
    out: list[str] = []
    for p in parts:
        key = p.lower().replace("-", "_")
        if key not in TASK_ALIASES:
            # 允许直接传入未登记别名的 lm-eval 任务名
            name = p
        else:
            name = TASK_ALIASES[key]
        if name not in out:
            out.append(name)
    return out


def normalize_kv_format(kv_format: str) -> str:
    """``fp16`` / ``kivi2`` / ``kivi4`` 规范化。"""
    key = kv_format.strip().lower().replace("-", "_").replace("+", "_")
    if key not in _FORMAT_ALIASES:
        raise ValueError(f"未知 kv_format={kv_format!r}")
    return _FORMAT_ALIASES[key]


# ---------------------------------------------------------------------------
# lm-eval 包装（生成前清空本仓库 Kivi cache）
# ---------------------------------------------------------------------------


def _require_lm_eval():
    try:
        import lm_eval  # noqa: F401
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "需要 lm-eval：pip install 'lm-eval>=0.4.5' accelerate"
        ) from e
    return HFLM


def wrap_for_lm_eval(
    model: nn.Module,
    tokenizer: Any,
    *,
    batch_size: int = 1,
    max_batch_size: int = 1,
    device: str | None = None,
) -> Any:
    """将已加载（可已 KIVI patch）的 HF 模型包装为 lm-eval ``HFLM``。

    每次 ``_model_generate`` 前清空 ``LlamaKiviAttention`` cache，避免跨样本残留。
    """
    HFLM = _require_lm_eval()

    class KiviHFLM(HFLM):  # type: ignore[misc, valid-type]
        """在官方 HFLM 上挂钩本仓库 Kivi cache 清理。"""

        def _model_generate(
            self,
            context,
            max_length: int,
            stop: list[str],
            **generation_kwargs: Any,
        ):
            if is_llama_kivi_patched(self.model):
                clear_llama_kivi_caches(self.model)
            return super()._model_generate(
                context, max_length, stop, **generation_kwargs
            )

    if device is None:
        device = str(next(model.parameters()).device)

    return KiviHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        backend="causal",
    )


# ---------------------------------------------------------------------------
# 跑评测与抽分
# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    """单任务主分 + 原始 metrics 快照。"""

    task: str
    primary_metric: str
    score: float  # 百分制
    raw_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LmEvalBundle:
    """一次 lm-eval 评测结果。"""

    model_id: str
    kv_format: str
    tasks: list[str]
    primary_scores: dict[str, float]
    task_scores: list[TaskScore]
    limit: int | None
    lm_eval_results: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.lm_eval_results is not None:
            d["lm_eval_results"] = {
                k: self.lm_eval_results[k]
                for k in ("results", "n-shot", "config")
                if k in self.lm_eval_results
            }
        return d


def _to_percent(value: float) -> float:
    """lm-eval 常给 0–1；统一成百分制便于报告。"""
    v = float(value)
    if abs(v) <= 1.0 + 1e-9:
        return round(100.0 * v, 4)
    return round(v, 4)


def _pick_metric(metrics: dict[str, Any], primary: str) -> float | None:
    """从 ``results[task]`` 字典中选取主指标。

    键形如 ``em,none`` / ``exact_match,strict-match``；取以 ``primary`` 开头且
    不含 ``stderr`` 的第一项。
    """
    for key, val in metrics.items():
        if key == primary or key.startswith(primary + ","):
            if "stderr" in key:
                continue
            if isinstance(val, (int, float)):
                return float(val)
    for key, val in metrics.items():
        head = key.split(",")[0]
        if head == primary and "stderr" not in key and isinstance(val, (int, float)):
            return float(val)
    return None


def extract_primary_scores(
    lm_results: dict[str, Any],
    tasks: Sequence[str] | None = None,
    *,
    primary_metrics: dict[str, str] | None = None,
) -> dict[str, TaskScore]:
    """从 ``simple_evaluate`` 返回结构抽取百分制主分。"""
    task_list = list(tasks) if tasks is not None else list(DEFAULT_TASKS)
    metric_map = {**PRIMARY_METRICS, **(primary_metrics or {})}
    raw = lm_results.get("results", lm_results)
    out: dict[str, TaskScore] = {}
    for task in task_list:
        if task not in raw:
            raise KeyError(f"lm-eval 结果中缺少任务 {task!r}；现有 {list(raw)}")
        metrics = dict(raw[task])
        if task not in metric_map:
            raise KeyError(
                f"任务 {task} 未配置 primary metric；"
                f"请传入 primary_metrics={{'{task}': '...'}}"
            )
        primary = metric_map[task]
        val = _pick_metric(metrics, primary)
        if val is None:
            raise KeyError(
                f"任务 {task} 找不到主指标 {primary!r}；键={list(metrics)}"
            )
        out[task] = TaskScore(
            task=task,
            primary_metric=primary,
            score=_to_percent(val),
            raw_metrics=metrics,
        )
    return out


def score_delta(
    scores: dict[str, float] | dict[str, TaskScore],
    reference: dict[str, float],
) -> dict[str, float]:
    """``scores - reference``（百分制）；仅对两侧都有的任务求差。

    ``reference`` 由调用方提供（例如外部基准表），本模块不内嵌任何论文数字。
    """
    flat: dict[str, float] = {}
    for k, v in scores.items():
        flat[k] = v.score if isinstance(v, TaskScore) else float(v)
    return {
        task: round(flat[task] - float(ref), 4)
        for task, ref in reference.items()
        if task in flat
    }


def run_lm_eval(
    model: nn.Module,
    tokenizer: Any,
    *,
    tasks: str | Sequence[str] | None = "default",
    limit: int | None = None,
    batch_size: int = 1,
    log_samples: bool = False,
    verbosity: str = "INFO",
    **simple_evaluate_kwargs: Any,
) -> dict[str, Any]:
    """对已加载模型调用 ``lm_eval.simple_evaluate``。

    参数
        limit: 每任务样本上限（冒烟用）；``None`` 为全集。
        batch_size: 默认 1（本仓库 KIVI cache-path 当前仅 batch=1）。
    """
    import lm_eval

    task_list = resolve_tasks(tasks)
    lm = wrap_for_lm_eval(model, tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        log_samples=log_samples,
        verbosity=verbosity,
        **simple_evaluate_kwargs,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate 返回 None（多进程从进程？）")
    return results


def evaluate_lm_eval(
    model: nn.Module | None = None,
    tokenizer: Any = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    kv_format: str = "kivi2",
    tasks: str | Sequence[str] | None = "default",
    limit: int | None = None,
    batch_size: int = 1,
    out_dir: Path | str | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    device: str | None = None,
    log_samples: bool = False,
    primary_metrics: dict[str, str] | None = None,
    **load_kwargs: Any,
) -> LmEvalBundle:
    """端到端：可选加载模型 → lm-eval → 抽取主分。

    若 ``model`` / ``tokenizer`` 已给出则不再下载；否则按 ``kv_format`` 调用
    ``load_llama_for_generate``。
    """
    fmt = normalize_kv_format(kv_format)
    task_list = resolve_tasks(tasks)

    if model is None or tokenizer is None:
        model, tokenizer = load_llama_for_generate(
            model_id,
            kv_format=fmt,
            group_size=group_size,
            residual_length=residual_length,
            device=device,
            **load_kwargs,
        )
        mid = model_id
    else:
        mid = model_id or str(
            getattr(getattr(model, "config", None), "_name_or_path", "") or "unknown"
        )

    bits = kv_format_to_bits(fmt)
    patched = is_llama_kivi_patched(model)
    if bits is not None and not patched:
        raise RuntimeError(
            f"kv_format={fmt} 需要 KIVI patch，但模型未 patch；"
            "请用 load_llama_for_generate(..., kv_format=...) 或 patch_llama_model"
        )

    lm_results = run_lm_eval(
        model,
        tokenizer,
        tasks=task_list,
        limit=limit,
        batch_size=batch_size,
        log_samples=log_samples,
    )
    scored = extract_primary_scores(
        lm_results, task_list, primary_metrics=primary_metrics
    )
    primary = {t: s.score for t, s in scored.items()}

    bundle = LmEvalBundle(
        model_id=mid,
        kv_format=fmt,
        tasks=task_list,
        primary_scores=primary,
        task_scores=list(scored.values()),
        limit=limit,
        lm_eval_results=lm_results,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"lm_eval_{fmt}.json").write_text(
            json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return bundle


def evaluate_lm_eval_formats(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    formats: Sequence[str] = ("fp16", "kivi2", "kivi4"),
    tasks: str | Sequence[str] | None = "default",
    limit: int | None = None,
    batch_size: int = 1,
    out_dir: Path | str | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    device: str | None = None,
    primary_metrics: dict[str, str] | None = None,
    **load_kwargs: Any,
) -> dict[str, LmEvalBundle]:
    """依次评测多种 KV 格式（每种重新加载，避免权重状态串扰）。"""
    bundles: dict[str, LmEvalBundle] = {}
    for fmt in formats:
        key = normalize_kv_format(fmt)
        bundles[key] = evaluate_lm_eval(
            model_id=model_id,
            kv_format=key,
            tasks=tasks,
            limit=limit,
            batch_size=batch_size,
            out_dir=out_dir,
            group_size=group_size,
            residual_length=residual_length,
            device=device,
            primary_metrics=primary_metrics,
            **load_kwargs,
        )
    if out_dir is not None:
        summary = {
            fmt: {"primary_scores": b.primary_scores, "limit": b.limit}
            for fmt, b in bundles.items()
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "lm_eval_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return bundles


# 旧名兼容（仅别名，无论文逻辑）
TABLE3_TASKS = DEFAULT_TASKS
Table3Bundle = LmEvalBundle
evaluate_table3 = evaluate_lm_eval
evaluate_table3_formats = evaluate_lm_eval_formats
