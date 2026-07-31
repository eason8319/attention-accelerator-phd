"""LongBench 评测辅助：子组任务、prompt、预测与打分。

提供四类长上下文子组的代表任务默认集；生成可接 ``hf_generate.generate_ids``
（若模型已 KIVI patch，会在生成前清空 cache）。

依赖：``datasets``；打分可选 ``rouge`` / ``fuzzywuzzy``
（缺省时 code 相似度回退 ``difflib``，摘要 ROUGE 则提示安装）。

用法：

  from kivi_repro.long_bench_tasks import evaluate_tasks, score_jsonl

  scores = score_jsonl(pred_dir)
  result = evaluate_tasks(model, tokenizer, tasks="default", max_length=8192)
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from .hf_generate import generate_ids
from .llama_kivi_attn import clear_llama_kivi_caches
from .patch_llama import is_llama_kivi_patched

__all__ = [
    "SUBGROUPS",
    "DEFAULT_REPRESENTATIVES",
    "EXTENDED_DATASETS",
    "DATASET2PROMPT",
    "DATASET2MAXLEN",
    "DEFAULT_MAX_LENGTH",
    "TaskResult",
    "EvalBundle",
    "resolve_tasks",
    "load_longbench",
    "format_prompt",
    "truncate_middle",
    "build_chat_prompt",
    "predict_dataset",
    "score_predictions",
    "score_jsonl",
    "evaluate_tasks",
    "write_jsonl",
    "read_jsonl",
]

# ---------------------------------------------------------------------------
# 子组与默认代表
# ---------------------------------------------------------------------------

SUBGROUPS: dict[str, tuple[str, ...]] = {
    "single_doc_qa": ("qasper",),
    "summarization": ("qmsum", "multi_news"),
    "few_shot": ("trec", "triviaqa", "samsum"),
    "code": ("lcc", "repobench-p"),
}

DEFAULT_REPRESENTATIVES: dict[str, str] = {
    "single_doc_qa": "qasper",
    "summarization": "qmsum",
    "few_shot": "trec",
    "code": "lcc",
}

# 四子组内较完整的英文任务集（``tasks="extended"``）
EXTENDED_DATASETS: tuple[str, ...] = (
    "triviaqa",
    "qasper",
    "trec",
    "samsum",
    "lcc",
    "repobench-p",
    "qmsum",
    "multi_news",
)

DEFAULT_MAX_LENGTH = 8192

# few-shot / 代码类通常不套 chat template（沿用 LongBench 常见做法）
_SKIP_CHAT_DATASETS = frozenset(
    {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
)
# 打分前取首行
_FIRST_LINE_DATASETS = frozenset({"trec", "triviaqa", "samsum", "lsht"})

DATASET2PROMPT: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story "
        "asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        'If the question cannot be answered based on the information in the '
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Article: {context}\n\n Answer the question based on the above article "
        "as concisely as you can, using a single phrase or sentence if possible. "
        'If the question cannot be answered based on the information in the '
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or "
        "more sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some "
        "examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are "
        "some examples.\n\n{context}\n\n{input}"
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
}

DATASET2MAXLEN: dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "qmsum": 512,
    "multi_news": 512,
    "gov_report": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lcc": 64,
    "repobench-p": 64,
}


# ---------------------------------------------------------------------------
# 指标（对齐 THUDM/LongBench ``metrics.py``；默认子组仅需英文）
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _token_f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_kwargs: Any) -> float:
    """英文 QA F1（token 级）。"""
    pred = _normalize_answer(prediction).split()
    gold = _normalize_answer(ground_truth).split()
    return _token_f1(pred, gold)


def classification_score(
    prediction: str, ground_truth: str, *, all_classes: list[str] | None = None, **_kw: Any
) -> float:
    """TREC 等分类：在预测中命中的类别集合上均分（对齐 LongBench）。"""
    classes = list(all_classes or [])
    em_match_list = [c for c in classes if c in prediction]
    # 官方会边遍历边 remove；这里用副本避免修改迭代中的列表
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def rouge_score(prediction: str, ground_truth: str, **_kwargs: Any) -> float:
    """ROUGE-L F（需 ``pip install rouge``）。"""
    try:
        from rouge import Rouge  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "LongBench 摘要打分需要 rouge：pip install rouge"
        ) from e
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:  # noqa: BLE001 — 官方遇空串等返回 0
        return 0.0
    return float(scores["rouge-l"]["f"])


def _fuzz_ratio(a: str, b: str) -> float:
    """0–100 字符串相似度；优先 fuzzywuzzy / rapidfuzz。"""
    try:
        from fuzzywuzzy import fuzz  # type: ignore[import-untyped]

        return float(fuzz.ratio(a, b))
    except ImportError:
        pass
    try:
        from rapidfuzz import fuzz as rfuzz  # type: ignore[import-untyped]

        return float(rfuzz.ratio(a, b))
    except ImportError:
        return 100.0 * SequenceMatcher(None, a, b).ratio()


def code_sim_score(prediction: str, ground_truth: str, **_kwargs: Any) -> float:
    """代码续写：取首个非注释/非 fence 行与参考的 fuzzy ratio。"""
    prediction_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction_line = line
            break
    return _fuzz_ratio(prediction_line, ground_truth) / 100.0


DATASET2METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "triviaqa": qa_f1_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "gov_report": rouge_score,
    "samsum": rouge_score,
    "trec": classification_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def subgroup_of(dataset: str) -> str | None:
    """返回任务所属子组名；未知则 ``None``。"""
    for name, tasks in SUBGROUPS.items():
        if dataset in tasks:
            return name
    return None


def resolve_tasks(
    tasks: str | Sequence[str] | None = "default",
) -> list[str]:
    """解析任务列表。

    参数
        tasks:
          - ``None`` / ``"default"``：四子组各一代表
          - ``"extended"`` / ``"kivi"``：``EXTENDED_DATASETS``（``kivi`` 为旧别名）
          - ``"all_subgroups"``：四子组全部任务
          - 子组名（如 ``"code"``）或具体任务名列表
    """
    if tasks is None or tasks == "default":
        return list(DEFAULT_REPRESENTATIVES.values())
    if isinstance(tasks, str):
        key = tasks.strip().lower()
        if key in ("extended", "kivi"):
            return list(EXTENDED_DATASETS)
        if key == "all_subgroups":
            out: list[str] = []
            for group_tasks in SUBGROUPS.values():
                out.extend(group_tasks)
            return out
        if key in SUBGROUPS:
            return list(SUBGROUPS[key])
        if key in DATASET2PROMPT or key in DATASET2METRIC:
            return [key]
        raise ValueError(f"未知 tasks={tasks!r}")
    return [str(t) for t in tasks]


# ---------------------------------------------------------------------------
# 数据与 prompt
# ---------------------------------------------------------------------------


def load_longbench(dataset: str, *, longbench_e: bool = False):
    """加载 ``THUDM/LongBench`` 的 ``test`` split。"""
    from datasets import load_dataset

    name = f"{dataset}_e" if longbench_e else dataset
    return load_dataset("THUDM/LongBench", name, split="test")


def format_prompt(dataset: str, example: dict[str, Any]) -> str:
    """用官方模板填充 ``context`` / ``input`` 等字段。"""
    if dataset not in DATASET2PROMPT:
        raise KeyError(f"无 prompt 模板：{dataset}")
    return DATASET2PROMPT[dataset].format(**example)


def truncate_middle(tokenizer: Any, prompt: str, max_length: int) -> str:
    """超长时截断中间（保留两端指令）。"""
    encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
    token_ids = encoded["input_ids"][0]
    if int(token_ids.shape[0]) <= max_length:
        return prompt
    half = max_length // 2
    return (
        tokenizer.decode(token_ids[:half], skip_special_tokens=True)
        + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    )


def build_chat_prompt(tokenizer: Any, prompt: str, model_name: str) -> str:
    """按模型名套 chat template。

    Llama-3-Instruct / Mistral-Instruct-v0.2：套模板；其余原样返回。
    """
    name = model_name.lower()
    if "llama-3" in name and "instruct" in name:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if "mistral" in name and "instruct" in name and "v0.2" in name:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


# ---------------------------------------------------------------------------
# 预测与打分
# ---------------------------------------------------------------------------


@dataclass
class PredRow:
    """单条预测（可写 jsonl）。"""

    pred: str
    answers: list[str]
    all_classes: list[str] | None
    length: int
    dataset: str = ""


@dataclass
class TaskResult:
    """单任务汇总分。"""

    dataset: str
    subgroup: str | None
    score: float | dict[str, float]
    n_samples: int
    metric: str


@dataclass
class EvalBundle:
    """一次 ``evaluate_tasks`` 的结果包。"""

    max_length: int
    model_name: str
    kv_note: str
    results: list[TaskResult] = field(default_factory=list)
    by_subgroup: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "model_name": self.model_name,
            "kv_note": self.kv_note,
            "results": [asdict(r) for r in self.results],
            "by_subgroup": self.by_subgroup,
        }


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any] | PredRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = asdict(row) if isinstance(row, PredRow) else row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.inference_mode()
def predict_one(
    model: nn.Module,
    tokenizer: Any,
    example: dict[str, Any],
    *,
    dataset: str,
    max_length: int,
    max_new_tokens: int | None = None,
    model_name: str = "",
    device: torch.device | None = None,
) -> PredRow:
    """对单条 LongBench 样本生成预测。"""
    if device is None:
        device = next(model.parameters()).device
    max_gen = max_new_tokens or DATASET2MAXLEN.get(dataset, 64)
    prompt = format_prompt(dataset, example)
    prompt = truncate_middle(tokenizer, prompt, max_length)
    if dataset not in _SKIP_CHAT_DATASETS and model_name:
        prompt = build_chat_prompt(tokenizer, prompt, model_name)

    encoded = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if is_llama_kivi_patched(model):
        clear_llama_kivi_caches(model)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_gen,
        "do_sample": False,
        "num_beams": 1,
    }
    # samsum：抑制无尽换行（常见评测做法）
    if dataset == "samsum":
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        eos = getattr(tokenizer, "eos_token_id", None)
        gen_kwargs["eos_token_id"] = [eos, nl_id] if eos is not None else nl_id
        gen_kwargs["min_length"] = int(input_ids.shape[-1]) + 1

    out_ids, _info = generate_ids(
        model,
        input_ids,
        attention_mask=attention_mask,
        clear_kivi=True,
        **gen_kwargs,
    )
    prompt_len = int(input_ids.shape[-1])
    text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)
    answers = example.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    classes = example.get("all_classes")
    if classes is None:
        classes = []
    return PredRow(
        pred=text,
        answers=list(answers),
        all_classes=list(classes) if classes else None,
        length=int(example.get("length", 0)),
        dataset=dataset,
    )


def predict_dataset(
    model: nn.Module,
    tokenizer: Any,
    dataset: str,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    model_name: str = "",
    longbench_e: bool = False,
    limit: int | None = None,
    show_progress: bool = True,
) -> list[PredRow]:
    """对整个数据集（或前 ``limit`` 条）做预测。"""
    data = load_longbench(dataset, longbench_e=longbench_e)
    if limit is not None:
        data = data.select(range(min(limit, len(data))))

    rows: list[PredRow] = []
    iterator: Iterable[Any] = data
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(data, desc=f"LongBench:{dataset}")
        except ImportError:
            pass

    name = model_name or getattr(getattr(model, "config", None), "_name_or_path", "") or ""
    for ex in iterator:
        rows.append(
            predict_one(
                model,
                tokenizer,
                dict(ex),
                dataset=dataset,
                max_length=max_length,
                model_name=str(name),
            )
        )
    return rows


def _sample_score(
    dataset: str,
    prediction: str,
    ground_truths: Sequence[str],
    all_classes: list[str] | None,
) -> float:
    if dataset not in DATASET2METRIC:
        raise KeyError(f"无打分函数：{dataset}")
    metric_fn = DATASET2METRIC[dataset]
    pred = prediction
    if dataset in _FIRST_LINE_DATASETS:
        pred = prediction.lstrip("\n").split("\n")[0]
    best = 0.0
    for gt in ground_truths:
        best = max(
            best,
            float(metric_fn(pred, gt, all_classes=all_classes or [])),
        )
    return best


def score_predictions(
    dataset: str,
    predictions: Sequence[str],
    answers: Sequence[Sequence[str]],
    *,
    all_classes: Sequence[list[str] | None] | None = None,
    lengths: Sequence[int] | None = None,
    longbench_e: bool = False,
) -> float | dict[str, float]:
    """按 LongBench 口径打分；返回百分制分数。

    ``longbench_e=True`` 时按长度桶 ``0-4k`` / ``4-8k`` / ``8k+`` 分别平均。
    """
    if len(predictions) == 0:
        return {} if longbench_e else 0.0
    if longbench_e:
        if lengths is None:
            raise ValueError("LongBench-E 打分需要 lengths")
        buckets: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
        for i, pred in enumerate(predictions):
            classes = None if all_classes is None else all_classes[i]
            s = _sample_score(dataset, pred, answers[i], classes)
            L = int(lengths[i])
            if L < 4000:
                buckets["0-4k"].append(s)
            elif L < 8000:
                buckets["4-8k"].append(s)
            else:
                buckets["8k+"].append(s)
        return {
            k: round(100.0 * float(np.mean(v)), 2) if v else 0.0
            for k, v in buckets.items()
        }

    total = 0.0
    for i, pred in enumerate(predictions):
        classes = None if all_classes is None else all_classes[i]
        total += _sample_score(dataset, pred, answers[i], classes)
    return round(100.0 * total / len(predictions), 2)


def score_pred_rows(
    dataset: str, rows: Sequence[PredRow | dict[str, Any]], *, longbench_e: bool = False
) -> TaskResult:
    """对 ``PredRow`` / jsonl 行打分并包装为 ``TaskResult``。"""
    preds: list[str] = []
    answers: list[list[str]] = []
    classes: list[list[str] | None] = []
    lengths: list[int] = []
    for row in rows:
        d = asdict(row) if isinstance(row, PredRow) else row
        preds.append(str(d["pred"]))
        ans = d.get("answers") or []
        answers.append(list(ans) if not isinstance(ans, str) else [ans])
        classes.append(d.get("all_classes"))
        lengths.append(int(d.get("length") or 0))
    score = score_predictions(
        dataset,
        preds,
        answers,
        all_classes=classes,
        lengths=lengths if longbench_e else None,
        longbench_e=longbench_e,
    )
    metric_name = DATASET2METRIC[dataset].__name__
    return TaskResult(
        dataset=dataset,
        subgroup=subgroup_of(dataset),
        score=score if isinstance(score, dict) else float(score),
        n_samples=len(preds),
        metric=metric_name,
    )


def score_jsonl(
    pred_dir: Path | str,
    *,
    longbench_e: bool = False,
) -> dict[str, float | dict[str, float]]:
    """扫描目录下 ``*.jsonl``，按文件名（任务名）打分。"""
    pred_dir = Path(pred_dir)
    scores: dict[str, float | dict[str, float]] = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        dataset = path.stem
        rows = read_jsonl(path)
        result = score_pred_rows(dataset, rows, longbench_e=longbench_e)
        scores[dataset] = result.score
    return scores


def evaluate_tasks(
    model: nn.Module,
    tokenizer: Any,
    *,
    tasks: str | Sequence[str] | None = "default",
    max_length: int = DEFAULT_MAX_LENGTH,
    model_name: str = "",
    longbench_e: bool = False,
    limit: int | None = None,
    out_dir: Path | str | None = None,
    kv_note: str = "",
    show_progress: bool = True,
) -> EvalBundle:
    """端到端：加载任务 → 预测 → 打分；可选写出 jsonl。

    参数
        limit: 每任务最多样本数（冒烟用）；``None`` 为全集。
        out_dir: 若给定，写入 ``{dataset}.jsonl`` 与 ``result.json``。
        kv_note: 记录 C0/C4/C5 等 KV 格式标签，便于结果归档。
    """
    task_list = resolve_tasks(tasks)
    name = model_name or str(
        getattr(getattr(model, "config", None), "_name_or_path", "") or "unknown"
    )
    note = kv_note
    if not note:
        note = "kivi" if is_llama_kivi_patched(model) else "fp16"

    bundle = EvalBundle(
        max_length=max_length, model_name=name, kv_note=note
    )
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    for dataset in task_list:
        rows = predict_dataset(
            model,
            tokenizer,
            dataset,
            max_length=max_length,
            model_name=name,
            longbench_e=longbench_e,
            limit=limit,
            show_progress=show_progress,
        )
        if out_path is not None:
            write_jsonl(out_path / f"{dataset}.jsonl", rows)
        result = score_pred_rows(dataset, rows, longbench_e=longbench_e)
        bundle.results.append(result)
        sg = result.subgroup or "other"
        if sg not in bundle.by_subgroup:
            bundle.by_subgroup[sg] = {}
        if isinstance(result.score, dict):
            # LongBench-E：记 0-4k/4-8k/8k+ 均值作子组概览
            vals = [float(v) for v in result.score.values()]
            bundle.by_subgroup[sg][dataset] = (
                round(float(np.mean(vals)), 2) if vals else 0.0
            )
        else:
            bundle.by_subgroup[sg][dataset] = float(result.score)

    if out_path is not None:
        (out_path / "result.json").write_text(
            json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return bundle


# 旧名兼容
KIVI_DEFAULT_DATASETS = EXTENDED_DATASETS
