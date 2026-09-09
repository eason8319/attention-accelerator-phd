"""HuggingFace ``generate`` 封装：原生 FP16 或本仓库 C0–C5 cache-path。

在每次生成前清空 cache-path KV，避免跨样本残留。
默认 ``batch=1``、贪心解码。

``kv_format``：
  - ``fp16`` / ``hf`` / ``baseline``：原生 HF attention（KIVI 评估 C0 精度上界）
  - ``c0`` / ``fp16_codec``：FP16 **codec** cache-path（traffic/PPL Pareto C0）
  - ``int8`` / ``int4`` / ``int4_bdr``：均匀 C1–C3
  - ``kivi2`` / ``kivi4``：C4/C5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .llama_kivi_attn import (
    bytes_stored_llama_kivi,
    canonical_cache_format,
    clear_llama_kivi_caches,
    is_cache_path_patched,
    is_kivi_format,
)
from .patch_llama import (
    build_llama_cache_path,
    build_llama_kivi,
    is_llama_kivi_patched,
)
from .patch_mistral import build_mistral_cache_path, build_mistral_kivi
from .patch_qwen import build_qwen_cache_path, build_qwen_kivi

__all__ = [
    "GenerateInfo",
    "load_llama_for_generate",
    "generate_ids",
    "generate_text",
    "kv_format_to_bits",
    "resolve_kv_load_format",
]

_NATIVE_ALIASES = frozenset({"fp16", "hf", "baseline", "16bit"})


def _as_token_id_list(value: Any) -> list[int]:
    """``eos_token_id`` 在 Llama-3.1 上是 list；统一成 int 列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def _scalar_token_id(value: Any) -> int | None:
    """取单个 token id。列表时用最后一个（Llama-3.1 的 ``<|eot_id|>``）。"""
    ids = _as_token_id_list(value)
    return ids[-1] if ids else None


def _ensure_scalar_pad(model: nn.Module, tokenizer: Any | None = None) -> int | None:
    """给 ``config`` / ``generation_config`` 写标量 ``pad_token_id``。

    Transformers 会做 ``pad_token_id < 0``；若 pad 回落到 list 型 eos 会 TypeError。
    """
    pad = _scalar_token_id(getattr(getattr(model, "config", None), "pad_token_id", None))
    if pad is None and tokenizer is not None:
        pad = _scalar_token_id(getattr(tokenizer, "pad_token_id", None))
    if pad is None:
        pad = _scalar_token_id(getattr(getattr(model, "config", None), "eos_token_id", None))
    if pad is None:
        return None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.pad_token_id = pad
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.pad_token_id = pad
    return pad


def resolve_kv_load_format(kv_format: str) -> str | None:
    """``None`` = 原生 HF；否则为本仓库 cache-path 规范名。"""
    key = kv_format.strip().lower().replace("-", "_").replace("+", "_")
    if key in _NATIVE_ALIASES:
        return None
    return canonical_cache_format(kv_format)


def kv_format_to_bits(kv_format: str) -> int | None:
    """KIVI 比特；原生 HF 与均匀 C0–C3 返回 ``None``。"""
    loaded = resolve_kv_load_format(kv_format)
    if loaded == "kivi2":
        return 2
    if loaded == "kivi4":
        return 4
    return None


@dataclass(frozen=True)
class GenerateInfo:
    """单次生成的附属信息。"""

    prompt_tokens: int
    new_tokens: int
    total_tokens: int
    kv_format: str
    kivi_patched: bool
    payload_bytes: int
    meta_bytes: int
    finish_reason: str


def _native_hf_causal_lm(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    from_pretrained_kwargs: dict[str, Any],
) -> tuple[nn.Module, Any]:
    """加载未 patch 的 HF 因果 LM（KIVI 评估 C0 精度上界）。"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    model.to(device)
    model.eval()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.kivi_patched = False
        cfg.cache_path_patched = False
        cfg.cache_kv_format = "fp16"
    return model, tokenizer


def load_llama_for_generate(
    model_id: str = "NousResearch/Llama-2-7b-hf",
    *,
    kv_format: str = "kivi2",
    bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    layout: str = "contiguous",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> tuple[nn.Module, Any]:
    """按 ``kv_format`` 选择原生 HF、均匀 cache-path 或 KIVI。

    参数
        model_id: HF ID 或本地路径。
        kv_format: 见模块 docstring；``fp16`` 保持原生 HF（勿与 ``c0`` 混淆）。
        bits: 若给定则走 KIVI，并覆盖 ``kv_format`` 解析出的比特。
        group_size / residual_length: cache-path / KIVI 超参。
        layout: ``contiguous``（默认）或 ``paged``；主 Pareto 精度默认连续。
        device / dtype: 见 ``build_llama_kivi``。

    返回
        ``(model, tokenizer)``，已 ``eval()``。

    架构
        原生 FP16 不做 attention patch，支持任意 HF ``AutoModelForCausalLM``。
        cache-path / KIVI 按 ``config.model_type`` 分发 llama / mistral / qwen2。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    if bits is not None:
        if bits not in (2, 4):
            raise ValueError(f"KIVI bits 须为 2 或 4，得到 {bits}")
        loaded = "kivi2" if bits == 2 else "kivi4"
        resolved_bits = bits
    else:
        loaded = resolve_kv_load_format(kv_format)
        resolved_bits = kv_format_to_bits(kv_format)

    if loaded is None:
        model, tokenizer = _native_hf_causal_lm(
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            from_pretrained_kwargs=from_pretrained_kwargs,
        )
        _ensure_scalar_pad(model, tokenizer)
        return model, tokenizer

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    if model_type not in {"llama", "mistral", "qwen2"}:
        raise TypeError(
            f"cache-path 目前支持 llama / mistral / qwen2，得到 model_type={model_type!r} "
            f"（{type(cfg).__name__}）"
        )

    if resolved_bits is not None or is_kivi_format(loaded):
        kivi_bits = resolved_bits if resolved_bits is not None else (2 if loaded == "kivi2" else 4)
        kivi_builder = {
            "llama": build_llama_kivi,
            "mistral": build_mistral_kivi,
            "qwen2": build_qwen_kivi,
        }[model_type]
        model, tokenizer = kivi_builder(
            model_id,
            bits=kivi_bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **from_pretrained_kwargs,
        )
        _ensure_scalar_pad(model, tokenizer)
        return model, tokenizer

    cache_builder = {
        "llama": build_llama_cache_path,
        "mistral": build_mistral_cache_path,
        "qwen2": build_qwen_cache_path,
    }[model_type]
    model, tokenizer = cache_builder(
        model_id,
        kv_format=loaded,
        layout=layout,
        group_size=group_size,
        residual_length=residual_length,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    _ensure_scalar_pad(model, tokenizer)
    return model, tokenizer


@torch.inference_mode()
def generate_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    clear_kivi: bool = True,
    **generate_kwargs: Any,
) -> tuple[torch.Tensor, GenerateInfo]:
    """在 ``input_ids`` 上调用 ``model.generate``，返回完整序列与 ``GenerateInfo``。

    参数
        input_ids: ``(1, prompt_len)``（当前协议 batch=1）。
        clear_kivi: 生成前是否清空本仓库 cache-path KV（C0–C5 建议 True）。

    返回
        ``(output_ids, info)``；``output_ids`` 含 prompt + 新生成 token。
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"期望 input_ids 形状 (1, L)，得到 {tuple(input_ids.shape)}")

    if clear_kivi and is_cache_path_patched(model):
        clear_llama_kivi_caches(model)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = int(input_ids.shape[1])
    eos_ids = _as_token_id_list(getattr(model.config, "eos_token_id", None))
    pad_id = _ensure_scalar_pad(model)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
        "eos_token_id": eos_ids if eos_ids else None,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    gen_kwargs.update(generate_kwargs)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    total = int(out.shape[1])
    new_tokens = max(0, total - prompt_len)

    payload, meta = (0, 0)
    if is_cache_path_patched(model):
        payload, meta = bytes_stored_llama_kivi(model)

    kv_format = str(getattr(model.config, "cache_kv_format", "") or "")
    if not kv_format:
        kv_format = "fp16"

    finish = "length"
    if eos_ids and new_tokens > 0 and int(out[0, -1]) in eos_ids:
        finish = "eos"

    info = GenerateInfo(
        prompt_tokens=prompt_len,
        new_tokens=new_tokens,
        total_tokens=total,
        kv_format=kv_format,
        kivi_patched=is_llama_kivi_patched(model),
        payload_bytes=payload,
        meta_bytes=meta,
        finish_reason=finish,
    )
    return out, info


@torch.inference_mode()
def generate_text(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    clear_kivi: bool = True,
    skip_special_tokens: bool = True,
    **generate_kwargs: Any,
) -> tuple[str, GenerateInfo]:
    """对文本 prompt 生成续写，返回 ``(new_text, info)``（不含 prompt）。"""
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    out_ids, info = generate_ids(
        model,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        clear_kivi=clear_kivi,
        **generate_kwargs,
    )
    new_ids = out_ids[0, info.prompt_tokens :]
    text = tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)
    return text, info
