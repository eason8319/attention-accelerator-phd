"""将 HuggingFace Llama 的 attention 替换为本仓库 cache-path。

提供：
  - ``patch_llama_model`` / ``build_llama_kivi``：C4/C5 KIVI（KIVI evaluation API）
  - ``patch_llama_cache_path`` / ``build_llama_cache_path``：C0–C5

超参默认对齐协议：``group_size=32``，``residual_length=128``。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

from .llama_kivi_attn import (
    LlamaCachePathAttention,
    LlamaKiviAttention,
    bytes_stored_llama_kivi,
    canonical_cache_format,
    clear_llama_kivi_caches,
    is_cache_path_patched,
    is_kivi_format,
    is_kivi_patched,
)

__all__ = [
    "patch_llama_model",
    "patch_llama_cache_path",
    "build_llama_kivi",
    "build_llama_cache_path",
    "is_llama_kivi_patched",
    "is_llama_cache_path_patched",
    "clear_llama_kivi_caches",
    "bytes_stored_llama_kivi",
]


def _iter_llama_layers(model: nn.Module) -> list[nn.Module]:
    """取出 Llama 解码层列表（兼容 ``model.model.layers``）。"""
    if isinstance(model, LlamaForCausalLM):
        return list(model.model.layers)
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise TypeError(
            f"无法定位 Llama layers；期望 LlamaForCausalLM，得到 {type(model).__name__}"
        )
    return list(layers)


def is_llama_kivi_patched(model: nn.Module) -> bool:
    """是否已将全部 decoder 层的 ``self_attn`` 换成 KIVI attention。"""
    return is_kivi_patched(model)


def is_llama_cache_path_patched(model: nn.Module) -> bool:
    """是否已将全部 decoder 层换成 cache-path attention（C0–C5）。"""
    return is_cache_path_patched(model)


def _stamp_cache_config(
    model: nn.Module,
    *,
    kv_format: str,
    layout: str,
    group_size: int,
    residual_length: int,
    kivi_bits: int | None,
) -> None:
    """把格式写到 ``model.config``，供 generate / 评测读取。"""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    cfg.cache_kv_format = kv_format
    cfg.cache_layout = layout
    cfg.cache_path_patched = True
    cfg.kivi_group_size = group_size
    cfg.kivi_residual_length = residual_length
    if kivi_bits is None:
        cfg.kivi_patched = False
        return
    cfg.kivi_bits = kivi_bits
    cfg.kivi_k_bits = kivi_bits
    cfg.kivi_v_bits = kivi_bits
    cfg.kivi_patched = True


def patch_llama_model(
    model: nn.Module,
    *,
    bits: int = 2,
    k_bits: int | None = None,
    v_bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    layout: str = "contiguous",
    inplace: bool = True,
) -> nn.Module:
    """就地将各层 attention 替换为 ``LlamaKiviAttention``（拷贝权重）。"""
    if not inplace:
        raise ValueError("当前仅支持 inplace=True 就地替换")

    layers = _iter_llama_layers(model)
    n_patched = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, LlamaKiviAttention):
            attn.bits = bits
            attn.k_bits = bits if k_bits is None else k_bits
            attn.v_bits = bits if v_bits is None else v_bits
            attn.kv_format = "kivi2" if bits == 2 else "kivi4"
            attn.group_size = group_size
            attn.residual_length = residual_length
            attn.layout = layout
            attn.kivi_cache = None
            n_patched += 1
            continue
        if isinstance(attn, (LlamaAttention, LlamaCachePathAttention)):
            layer.self_attn = LlamaKiviAttention.from_hf_attention(
                attn,
                bits=bits,
                k_bits=k_bits,
                v_bits=v_bits,
                group_size=group_size,
                residual_length=residual_length,
                layout=layout,
            )
            n_patched += 1
            continue
        raise TypeError(
            f"layer.self_attn 类型为 {type(attn).__name__}，"
            f"期望 LlamaAttention / LlamaCachePathAttention"
        )

    if n_patched == 0:
        raise RuntimeError("未找到可 patch 的 Llama attention 层")

    _stamp_cache_config(
        model,
        kv_format="kivi2" if bits == 2 else "kivi4",
        layout=layout,
        group_size=group_size,
        residual_length=residual_length,
        kivi_bits=bits,
    )
    return model


def patch_llama_cache_path(
    model: nn.Module,
    *,
    kv_format: str,
    layout: str = "contiguous",
    group_size: int = 32,
    residual_length: int = 128,
    seed: int = 0,
    inplace: bool = True,
) -> nn.Module:
    """就地换成 C0–C5 cache-path。KIVI 走 ``LlamaKiviAttention``，其余走父类。"""
    if not inplace:
        raise ValueError("当前仅支持 inplace=True 就地替换")
    name = canonical_cache_format(kv_format)
    if is_kivi_format(name):
        bits = 2 if name == "kivi2" else 4
        return patch_llama_model(
            model,
            bits=bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
        )

    layers = _iter_llama_layers(model)
    n_patched = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, LlamaCachePathAttention) and not isinstance(attn, LlamaKiviAttention):
            attn.kv_format = name
            attn.layout = layout
            attn.group_size = group_size
            attn.residual_length = residual_length
            attn.seed = seed
            attn.cache = None
            n_patched += 1
            continue
        if isinstance(attn, (LlamaAttention, LlamaCachePathAttention)):
            layer.self_attn = LlamaCachePathAttention.from_hf_attention(
                attn,
                kv_format=name,
                layout=layout,
                group_size=group_size,
                residual_length=residual_length,
                seed=seed,
            )
            n_patched += 1
            continue
        raise TypeError(
            f"layer.self_attn 类型为 {type(attn).__name__}，"
            f"期望 LlamaAttention / LlamaCachePathAttention"
        )

    if n_patched == 0:
        raise RuntimeError("未找到可 patch 的 Llama attention 层")

    _stamp_cache_config(
        model,
        kv_format=name,
        layout=layout,
        group_size=group_size,
        residual_length=residual_length,
        kivi_bits=None,
    )
    return model


def _load_llama_base(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    from_pretrained_kwargs: dict[str, Any],
) -> tuple[LlamaForCausalLM, Any]:
    """from_pretrained + tokenizer；强制 eager mask。"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from_pretrained_kwargs.setdefault("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(f"仅支持 LlamaForCausalLM，得到 {type(model).__name__}")
    return model, tokenizer


def build_llama_kivi(
    model_id: str = "NousResearch/Llama-2-7b-hf",
    *,
    bits: int = 2,
    k_bits: int | None = None,
    v_bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    layout: str = "contiguous",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> tuple[LlamaForCausalLM, Any]:
    """加载 Llama，替换为 KIVI attention，返回 ``(model, tokenizer)``。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    model, tokenizer = _load_llama_base(
        model_id,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        from_pretrained_kwargs=from_pretrained_kwargs,
    )
    patch_llama_model(
        model,
        bits=bits,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
        layout=layout,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def build_llama_cache_path(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    *,
    kv_format: str,
    layout: str = "contiguous",
    group_size: int = 32,
    residual_length: int = 128,
    seed: int = 0,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> tuple[LlamaForCausalLM, Any]:
    """加载 Llama，换成 C0–C5 cache-path，返回 ``(model, tokenizer)``。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    model, tokenizer = _load_llama_base(
        model_id,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        from_pretrained_kwargs=from_pretrained_kwargs,
    )
    patch_llama_cache_path(
        model,
        kv_format=kv_format,
        layout=layout,
        group_size=group_size,
        residual_length=residual_length,
        seed=seed,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
