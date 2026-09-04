"""KIVI 整模复现：Llama / Mistral 接入与评测辅助。"""

from .hf_generate import (
    GenerateInfo,
    generate_ids,
    generate_text,
    kv_format_to_bits,
    load_llama_for_generate,
)
from .llama_kivi_attn import (
    LlamaKiviAttention,
    MistralKiviAttention,
    bytes_stored_kivi,
    bytes_stored_llama_kivi,
    clear_kivi_caches,
    clear_llama_kivi_caches,
    is_kivi_patched,
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
from .patch_mistral import (
    build_mistral_kivi,
    is_mistral_kivi_patched,
    patch_mistral_model,
)

__all__ = [
    "LlamaKiviAttention",
    "MistralKiviAttention",
    "bytes_stored_kivi",
    "bytes_stored_llama_kivi",
    "clear_kivi_caches",
    "clear_llama_kivi_caches",
    "is_kivi_patched",
    "build_llama_kivi",
    "is_llama_kivi_patched",
    "patch_llama_model",
    "build_mistral_kivi",
    "is_mistral_kivi_patched",
    "patch_mistral_model",
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
