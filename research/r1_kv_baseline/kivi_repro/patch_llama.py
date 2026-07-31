"""将 HuggingFace Llama 的 attention 替换为本仓库 ``LlamaKiviAttention``。

提供：
  - ``patch_llama_model``：就地替换已加载模型的各层 ``self_attn``
  - ``build_llama_kivi``：``from_pretrained`` + 替换，一键得到可 ``generate`` 的模型

超参默认对齐协议：``group_size=32``，``residual_length=128``。

用法：

  from kivi_repro.patch_llama import build_llama_kivi, patch_llama_model

  model, tokenizer = build_llama_kivi(
      "NousResearch/Llama-2-7b-hf", bits=2, device="cuda",
  )
  # 或
  patch_llama_model(model, bits=4)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

from .llama_kivi_attn import (
    LlamaKiviAttention,
    bytes_stored_llama_kivi,
    clear_llama_kivi_caches,
)

__all__ = [
    "patch_llama_model",
    "build_llama_kivi",
    "is_llama_kivi_patched",
    "clear_llama_kivi_caches",
    "bytes_stored_llama_kivi",
]


def _iter_llama_layers(model: nn.Module) -> list[nn.Module]:
    """取出 Llama 解码层列表（兼容 ``model.model.layers``）。"""
    if isinstance(model, LlamaForCausalLM):
        return list(model.model.layers)
    # 部分包装：仍尝试常见路径
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise TypeError(
            f"无法定位 Llama layers；期望 LlamaForCausalLM，得到 {type(model).__name__}"
        )
    return list(layers)


def is_llama_kivi_patched(model: nn.Module) -> bool:
    """是否已将全部 decoder 层的 ``self_attn`` 换成 ``LlamaKiviAttention``。"""
    layers = _iter_llama_layers(model)
    if not layers:
        return False
    return all(isinstance(layer.self_attn, LlamaKiviAttention) for layer in layers)


def patch_llama_model(
    model: nn.Module,
    *,
    bits: int = 2,
    k_bits: int | None = None,
    v_bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    inplace: bool = True,
) -> nn.Module:
    """就地将各层 ``LlamaAttention`` 替换为 ``LlamaKiviAttention``（拷贝权重）。

    参数
        model: ``LlamaForCausalLM`` 或含 ``model.layers`` 的等价结构。
        bits: K/V 默认比特（2 或 4）；可被 ``k_bits`` / ``v_bits`` 覆盖。
        group_size / residual_length: 对齐 KIVI / 本仓库协议。
        inplace: 必须为 True（当前只支持就地替换）。

    返回
        同一 ``model`` 引用（已 patch）。

    异常
        若某层不是 ``LlamaAttention`` / 已是 ``LlamaKiviAttention`` 则跳过或报错见下。
    """
    if not inplace:
        raise ValueError("当前仅支持 inplace=True 就地替换")

    layers = _iter_llama_layers(model)
    n_patched = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, LlamaKiviAttention):
            # 已 patch：更新超参并清空 cache，避免旧状态
            attn.bits = bits
            attn.k_bits = bits if k_bits is None else k_bits
            attn.v_bits = bits if v_bits is None else v_bits
            attn.group_size = group_size
            attn.residual_length = residual_length
            attn.kivi_cache = None
            n_patched += 1
            continue
        if not isinstance(attn, LlamaAttention):
            raise TypeError(
                f"layer.self_attn 类型为 {type(attn).__name__}，"
                f"期望 LlamaAttention 或 LlamaKiviAttention"
            )
        layer.self_attn = LlamaKiviAttention.from_llama_attention(
            attn,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
        )
        n_patched += 1

    if n_patched == 0:
        raise RuntimeError("未找到可 patch 的 Llama attention 层")

    # 挂到 config 上便于评测脚本读取
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.kivi_bits = bits
        cfg.kivi_k_bits = bits if k_bits is None else k_bits
        cfg.kivi_v_bits = bits if v_bits is None else v_bits
        cfg.kivi_group_size = group_size
        cfg.kivi_residual_length = residual_length
        cfg.kivi_patched = True

    return model


def build_llama_kivi(
    model_id: str = "NousResearch/Llama-2-7b-hf",
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
) -> tuple[LlamaForCausalLM, Any]:
    """加载 Llama 因果 LM，替换为 KIVI attention，并返回 ``(model, tokenizer)``。

    参数
        model_id: HF 模型 ID 或本地路径（协议锚：``NousResearch/Llama-2-7b-hf``）。
        bits / group_size / residual_length: 见 ``patch_llama_model``。
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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(
            f"build_llama_kivi 仅支持 LlamaForCausalLM，得到 {type(model).__name__}"
        )

    patch_llama_model(
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
