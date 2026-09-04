"""将 HuggingFace Mistral 的 attention 替换为本仓库 ``MistralKiviAttention``。

提供：
  - ``patch_mistral_model``：就地替换已加载模型的各层 ``self_attn``
  - ``build_mistral_kivi``：``from_pretrained`` + 替换，一键得到可 ``generate`` 的模型

超参默认对齐协议：``group_size=32``，``residual_length=128``。
数值路径与 Llama 相同（本仓库 ``KiviKVCache``）；sliding window 由 HF
``MistralModel`` 的 mask 负责。

用法：

  from kivi_repro.patch_mistral import build_mistral_kivi, patch_mistral_model

  model, tokenizer = build_mistral_kivi(
      "mistralai/Mistral-7B-Instruct-v0.2", bits=2, device="cuda",
  )
  # 或
  patch_mistral_model(model, bits=4)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
)

from .llama_kivi_attn import (
    MistralKiviAttention,
    bytes_stored_llama_kivi,
    clear_llama_kivi_caches,
)

__all__ = [
    "patch_mistral_model",
    "build_mistral_kivi",
    "is_mistral_kivi_patched",
    "clear_llama_kivi_caches",
    "bytes_stored_llama_kivi",
]


def _iter_mistral_layers(model: nn.Module) -> list[nn.Module]:
    """取出 Mistral 解码层列表（兼容 ``model.model.layers``）。"""
    if isinstance(model, MistralForCausalLM):
        return list(model.model.layers)
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise TypeError(
            f"无法定位 Mistral layers；期望 MistralForCausalLM，得到 {type(model).__name__}"
        )
    return list(layers)


def is_mistral_kivi_patched(model: nn.Module) -> bool:
    """是否已将全部 decoder 层的 ``self_attn`` 换成 ``MistralKiviAttention``。"""
    try:
        layers = _iter_mistral_layers(model)
    except TypeError:
        return False
    if not layers:
        return False
    return all(isinstance(layer.self_attn, MistralKiviAttention) for layer in layers)


def patch_mistral_model(
    model: nn.Module,
    *,
    bits: int = 2,
    k_bits: int | None = None,
    v_bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    inplace: bool = True,
) -> nn.Module:
    """就地将各层 ``MistralAttention`` 替换为 ``MistralKiviAttention``（拷贝权重）。

    参数
        model: ``MistralForCausalLM`` 或含 ``model.layers`` 的等价结构。
        bits: K/V 默认比特（2 或 4）；可被 ``k_bits`` / ``v_bits`` 覆盖。
        group_size / residual_length: 对齐 KIVI / 本仓库协议。
        inplace: 必须为 True（当前只支持就地替换）。

    返回
        同一 ``model`` 引用（已 patch）。
    """
    if not inplace:
        raise ValueError("当前仅支持 inplace=True 就地替换")

    layers = _iter_mistral_layers(model)
    n_patched = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, MistralKiviAttention):
            attn.bits = bits
            attn.k_bits = bits if k_bits is None else k_bits
            attn.v_bits = bits if v_bits is None else v_bits
            attn.group_size = group_size
            attn.residual_length = residual_length
            attn.kivi_cache = None
            n_patched += 1
            continue
        if not isinstance(attn, MistralAttention):
            raise TypeError(
                f"layer.self_attn 类型为 {type(attn).__name__}，"
                f"期望 MistralAttention 或 MistralKiviAttention"
            )
        layer.self_attn = MistralKiviAttention.from_mistral_attention(
            attn,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
        )
        n_patched += 1

    if n_patched == 0:
        raise RuntimeError("未找到可 patch 的 Mistral attention 层")

    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.kivi_bits = bits
        cfg.kivi_k_bits = bits if k_bits is None else k_bits
        cfg.kivi_v_bits = bits if v_bits is None else v_bits
        cfg.kivi_group_size = group_size
        cfg.kivi_residual_length = residual_length
        cfg.kivi_patched = True

    return model


def build_mistral_kivi(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    *,
    bits: int = 2,
    k_bits: int | None = None,
    v_bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> tuple[MistralForCausalLM, Any]:
    """加载 Mistral 因果 LM，替换为 KIVI attention，并返回 ``(model, tokenizer)``。

    参数
        model_id: HF 模型 ID 或本地路径（协议锚：``mistralai/Mistral-7B-Instruct-v0.2``）。
        bits / group_size / residual_length: 见 ``patch_mistral_model``。
        device: 如 ``\"cuda\"`` / ``\"cpu\"``；``None`` 则保持 ``from_pretrained`` 默认。
        dtype: 权重 dtype；``None`` 时 GPU 用 ``float16``，CPU 用 ``float32``。
        trust_remote_code: 传给 tokenizer / model。
        from_pretrained_kwargs: 其余传给 ``AutoModelForCausalLM.from_pretrained``。

    返回
        ``(model, tokenizer)``；model 已 ``eval()`` 且 attention 已 patch。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # KIVI attention 是显式 eager 计算，需要 HF 构造加性因果 mask
    # （对 Mistral 同时保证 sliding-window mask 语义显式传入）；
    # sdpa/flash 实现下 HF 可能给各层传 attention_mask=None
    from_pretrained_kwargs.setdefault("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    if not isinstance(model, MistralForCausalLM):
        raise TypeError(
            f"build_mistral_kivi 仅支持 MistralForCausalLM，得到 {type(model).__name__}"
        )

    patch_mistral_model(
        model,
        bits=bits,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
