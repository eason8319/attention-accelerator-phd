"""KIVI 整模复现：Llama 接入与评测辅助。"""

from .hf_generate import (
    GenerateInfo,
    generate_ids,
    generate_text,
    kv_format_to_bits,
    load_llama_for_generate,
)
from .llama_kivi_attn import (
    LlamaKiviAttention,
    bytes_stored_llama_kivi,
    clear_llama_kivi_caches,
)
from .lm_eval_tasks import (
    DEFAULT_TASKS,
    evaluate_lm_eval,
    extract_primary_scores,
    score_delta,
)
from .long_bench_tasks import (
    DEFAULT_REPRESENTATIVES,
    SUBGROUPS,
    evaluate_tasks,
    resolve_tasks,
    score_jsonl,
    score_predictions,
)
from .patch_llama import (
    build_llama_kivi,
    is_llama_kivi_patched,
    patch_llama_model,
)

__all__ = [
    "LlamaKiviAttention",
    "bytes_stored_llama_kivi",
    "clear_llama_kivi_caches",
    "build_llama_kivi",
    "is_llama_kivi_patched",
    "patch_llama_model",
    "GenerateInfo",
    "generate_ids",
    "generate_text",
    "kv_format_to_bits",
    "load_llama_for_generate",
    "SUBGROUPS",
    "DEFAULT_REPRESENTATIVES",
    "resolve_tasks",
    "score_predictions",
    "score_jsonl",
    "evaluate_tasks",
    "DEFAULT_TASKS",
    "evaluate_lm_eval",
    "extract_primary_scores",
    "score_delta",
]
