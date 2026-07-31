"""HuggingFace ``generate`` 封装：FP16 基线或本仓库 KIVI 整模路径。

在每次生成前清空 ``LlamaKiviAttention`` 的 Kivi cache，避免跨样本残留。
默认 ``batch=1``、贪心解码，便于 Table 3 / 冒烟对齐。

用法：

  from kivi_repro.hf_generate import load_llama_for_generate, generate_text

  model, tok = load_llama_for_generate(
      "NousResearch/Llama-2-7b-hf", kv_format="kivi2", device="cuda",
  )
  text, info = generate_text(model, tok, "Q: 1+1=?\\nA:", max_new_tokens=32)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .llama_kivi_attn import bytes_stored_llama_kivi, clear_llama_kivi_caches
from .patch_llama import build_llama_kivi, is_llama_kivi_patched

__all__ = [
    "GenerateInfo",
    "load_llama_for_generate",
    "generate_ids",
    "generate_text",
    "kv_format_to_bits",
]


def kv_format_to_bits(kv_format: str) -> int | None:
    """格式别名 → bits；``fp16`` / ``C0`` 返回 ``None``（不 patch）。"""
    key = kv_format.strip().lower().replace("-", "_").replace("+", "_")
    aliases: dict[str, int | None] = {
        "fp16": None,
        "c0": None,
        "hf": None,
        "baseline": None,
        "kivi2": 2,
        "kivi_2": 2,
        "kivi_2bit": 2,
        "c4": 2,
        "kivi4": 4,
        "kivi_4": 4,
        "kivi_4bit": 4,
        "c5": 4,
    }
    if key not in aliases:
        raise ValueError(
            f"未知 kv_format={kv_format!r}；支持 fp16/kivi2/kivi4 或 C0/C4/C5"
        )
    return aliases[key]


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


def load_llama_for_generate(
    model_id: str = "NousResearch/Llama-2-7b-hf",
    *,
    kv_format: str = "kivi2",
    bits: int | None = None,
    group_size: int = 32,
    residual_length: int = 128,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> tuple[LlamaForCausalLM, Any]:
    """加载 Llama，按 ``kv_format`` 选择 FP16 基线或 KIVI patch。

    参数
        model_id: HF ID 或本地路径。
        kv_format: ``fp16`` / ``kivi2`` / ``kivi4``（或 C0/C4/C5）。
        bits: 若给定则覆盖 ``kv_format`` 解析出的比特；``None`` 且 format 为 fp16 则不 patch。
        group_size / residual_length: 仅 KIVI 路径生效。
        device / dtype: 见 ``build_llama_kivi``。

    返回
        ``(model, tokenizer)``，已 ``eval()``。
    """
    resolved_bits = bits if bits is not None else kv_format_to_bits(kv_format)

    if resolved_bits is None:
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
                f"仅支持 LlamaForCausalLM，得到 {type(model).__name__}"
            )
        model.to(device)
        model.eval()
        if getattr(model.config, "kivi_patched", False):
            model.config.kivi_patched = False
        return model, tokenizer

    return build_llama_kivi(
        model_id,
        bits=resolved_bits,
        group_size=group_size,
        residual_length=residual_length,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )


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
        clear_kivi: 生成前是否 ``clear_llama_kivi_caches``（KIVI 路径建议 True）。

    返回
        ``(output_ids, info)``；``output_ids`` 含 prompt + 新生成 token。
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"期望 input_ids 形状 (1, L)，得到 {tuple(input_ids.shape)}")

    if clear_kivi and is_llama_kivi_patched(model):
        clear_llama_kivi_caches(model)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = int(input_ids.shape[1])
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": getattr(model.config, "pad_token_id", None)
        or getattr(model.config, "eos_token_id", None),
        "eos_token_id": getattr(model.config, "eos_token_id", None),
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
    if is_llama_kivi_patched(model):
        payload, meta = bytes_stored_llama_kivi(model)

    kv_format = "fp16"
    if is_llama_kivi_patched(model):
        bits = int(getattr(model.config, "kivi_bits", 2))
        kv_format = f"kivi{bits}"

    # 粗略结束原因
    finish = "length"
    eos_id = getattr(model.config, "eos_token_id", None)
    if eos_id is not None and new_tokens > 0 and int(out[0, -1]) == int(eos_id):
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
