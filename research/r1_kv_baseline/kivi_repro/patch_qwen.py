"""将 HuggingFace Qwen2 / Qwen2.5 的 attention 替换为本仓库 cache-path。

敏感性 Dev 模型 ``Qwen/Qwen2.5-0.5B-Instruct`` 走这里。数值核与 Llama 相同。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM

from .llama_kivi_attn import (
    Qwen2CachePathAttention,
    Qwen2KiviAttention,
    bytes_stored_llama_kivi,
    canonical_cache_format,
    clear_llama_kivi_caches,
    is_cache_path_patched,
    is_kivi_format,
)

__all__ = [
    "patch_qwen_model",
    "patch_qwen_cache_path",
    "build_qwen_kivi",
    "build_qwen_cache_path",
    "is_qwen_kivi_patched",
    "is_qwen_cache_path_patched",
    "clear_llama_kivi_caches",
    "bytes_stored_llama_kivi",
]


def _iter_qwen_layers(model: nn.Module) -> list[nn.Module]:
    """取出 Qwen2 解码层列表（兼容 ``model.model.layers``）。"""
    if isinstance(model, Qwen2ForCausalLM):
        return list(model.model.layers)
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise TypeError(
            f"无法定位 Qwen2 layers；期望 Qwen2ForCausalLM，得到 {type(model).__name__}"
        )
    return list(layers)


def is_qwen_cache_path_patched(model: nn.Module) -> bool:
    """是否已将全部 decoder 层换成 cache-path attention（C0–C5）。"""
    return is_cache_path_patched(model)


def is_qwen_kivi_patched(model: nn.Module) -> bool:
    """是否已将全部 decoder 层换成 ``Qwen2KiviAttention``。"""
    try:
        layers = _iter_qwen_layers(model)
    except TypeError:
        return False
    if not layers:
        return False
    return all(isinstance(layer.self_attn, Qwen2KiviAttention) for layer in layers)


def patch_qwen_model(
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
    """就地将各层 attention 替换为 ``Qwen2KiviAttention``。"""
    if not inplace:
        raise ValueError("当前仅支持 inplace=True 就地替换")

    layers = _iter_qwen_layers(model)
    n_patched = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, Qwen2KiviAttention):
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
        if not isinstance(attn, (Qwen2Attention, Qwen2CachePathAttention)):
            raise TypeError(
                f"layer.self_attn 类型为 {type(attn).__name__}，"
                f"期望 Qwen2Attention 或 cache-path attention"
            )
        layer.self_attn = Qwen2KiviAttention.from_qwen_attention(
            attn,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
        )
        n_patched += 1

    if n_patched == 0:
        raise RuntimeError("未找到可 patch 的 Qwen2 attention 层")

    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.kivi_bits = bits
        cfg.kivi_k_bits = bits if k_bits is None else k_bits
        cfg.kivi_v_bits = bits if v_bits is None else v_bits
        cfg.kivi_group_size = group_size
        cfg.kivi_residual_length = residual_length
        cfg.kivi_patched = True
        cfg.cache_kv_format = "kivi2" if bits == 2 else "kivi4"
        cfg.cache_layout = layout
        cfg.cache_path_patched = True
        cfg.cache_layer_formats = [cfg.cache_kv_format] * n_patched
    return model


def patch_qwen_cache_path(
    model: nn.Module,
    *,
    kv_format: str,
    layout: str = "contiguous",
    group_size: int = 32,
    residual_length: int = 128,
    seed: int = 0,
    inplace: bool = True,
) -> nn.Module:
    """就地换成 C0–C5 cache-path。KIVI 走 ``Qwen2KiviAttention``。"""
    if not inplace:
        raise ValueError("当前仅支持 inplace=True 就地替换")
    name = canonical_cache_format(kv_format)
    if is_kivi_format(name):
        bits = 2 if name == "kivi2" else 4
        return patch_qwen_model(
            model,
            bits=bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
        )

    layers = _iter_qwen_layers(model)
    n_patched = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, Qwen2CachePathAttention) and not isinstance(attn, Qwen2KiviAttention):
            attn.set_kv_format(name)
            attn.layout = layout
            attn.group_size = group_size
            attn.residual_length = residual_length
            attn.seed = seed
            n_patched += 1
            continue
        if isinstance(
            attn,
            (Qwen2Attention, Qwen2CachePathAttention, Qwen2KiviAttention),
        ):
            layer.self_attn = Qwen2CachePathAttention.from_qwen_attention(
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
            f"期望 Qwen2Attention / cache-path attention"
        )

    if n_patched == 0:
        raise RuntimeError("未找到可 patch 的 Qwen2 attention 层")

    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.cache_kv_format = name
        cfg.cache_layout = layout
        cfg.cache_path_patched = True
        cfg.kivi_patched = False
        cfg.kivi_group_size = group_size
        cfg.kivi_residual_length = residual_length
        cfg.cache_layer_formats = [name] * n_patched
    return model


def build_qwen_kivi(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
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
) -> tuple[Qwen2ForCausalLM, Any]:
    """加载 Qwen2 因果 LM，替换为 KIVI attention，返回 ``(model, tokenizer)``。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

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
    if not isinstance(model, Qwen2ForCausalLM):
        raise TypeError(f"build_qwen_kivi 仅支持 Qwen2ForCausalLM，得到 {type(model).__name__}")
    patch_qwen_model(
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


def build_qwen_cache_path(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
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
) -> tuple[Qwen2ForCausalLM, Any]:
    """加载 Qwen2，换成 C0–C5 cache-path，返回 ``(model, tokenizer)``。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

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
    if not isinstance(model, Qwen2ForCausalLM):
        raise TypeError(
            f"build_qwen_cache_path 仅支持 Qwen2ForCausalLM，得到 {type(model).__name__}"
        )
    patch_qwen_cache_path(
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
