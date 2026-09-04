"""Llama / Mistral Attention：本仓库真实 cache-path（C0–C5）。

将 ``q/k/v_proj → RoPE → cache.append/load → SDPA → o_proj`` 接到
HuggingFace attention 槽位，供 ``generate`` / LM-Eval 使用。

约定：
  - 当前仅 ``batch=1``（与 R1 协议一致）
  - 数值路径使用 cache ``load()`` 的反量化 K/V（非投影 fake-quant）
  - 仍调用 HF ``past_key_values.update`` 以维护 generate 的序列长度簿记
  - ``LlamaKiviAttention`` 是 C4/C5 子类，保持 M3 patch API
  - 整模默认 ``layout=contiguous``；paged 精度由 M4 对齐，主 Pareto 不双跑

用法概要：

  attn = LlamaCachePathAttention.from_hf_attention(old_attn, kv_format="int4")
  layer.self_attn = attn
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache_path"
if str(_CACHE_PATH) not in sys.path:
    sys.path.insert(0, str(_CACHE_PATH))

from kv_cache import ContiguousKVCache, KiviKVCache  # noqa: E402
from kv_codecs import KiviFormat, KVCodec, get_codec  # noqa: E402
from paged_cache import (  # noqa: E402
    DEFAULT_PAGE_SIZE,
    DEFAULT_PTE_BYTES,
    PagedKiviKVCache,
    PagedUniformKVCache,
)

CacheBackend = ContiguousKVCache | KiviKVCache | PagedUniformKVCache | PagedKiviKVCache

_QUERY_CHUNK = 1024

_CACHE_FORMAT_ALIASES: dict[str, str] = {
    "fp16": "fp16",
    "c0": "fp16",
    "fp16_codec": "fp16",
    "int8": "int8",
    "c1": "int8",
    "int4": "int4",
    "c2": "int4",
    "int4_bdr": "int4_bdr",
    "int4bdr": "int4_bdr",
    "bdr": "int4_bdr",
    "c3": "int4_bdr",
    "kivi2": "kivi2",
    "kivi_2": "kivi2",
    "kivi_2bit": "kivi2",
    "c4": "kivi2",
    "kivi4": "kivi4",
    "kivi_4": "kivi4",
    "kivi_4bit": "kivi4",
    "c5": "kivi4",
}

_KIVI_FORMATS = frozenset({"kivi2", "kivi4"})
_NATIVE_FORMATS = frozenset({"hf", "baseline", "native", "16bit"})


def canonical_cache_format(kv_format: str) -> str:
    """规范成本仓库 cache-path 短名（``fp16``…``kivi4``）。

    ``fp16`` / ``c0`` / ``fp16_codec`` 都是 FP16 **codec**（不是原生 HF）。
    """
    key = kv_format.strip().lower().replace("-", "_").replace("+", "_")
    if key in _NATIVE_FORMATS:
        raise ValueError(
            f"{kv_format!r} 是原生 HF 路径，不是 cache-path；请用 fp16_codec / C0 或 int8/int4/…"
        )
    if key not in _CACHE_FORMAT_ALIASES:
        raise ValueError(
            f"未知 kv_format={kv_format!r}；支持 fp16_codec/int8/int4/int4_bdr/kivi2/kivi4 或 C0–C5"
        )
    return _CACHE_FORMAT_ALIASES[key]


def is_kivi_format(kv_format: str) -> bool:
    """是否为 C4/C5。"""
    return canonical_cache_format(kv_format) in _KIVI_FORMATS


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float = 0.0,
    return_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """手动 SDPA；输入形状 ``(B, H, T, D)``。

    与 HF ``eager_attention_forward`` 对齐：``attn_output`` 在返回前
    ``transpose(1, 2)`` 为 ``(B, T, H, D)``。``q_len > _QUERY_CHUNK`` 且不需要
    ``attn_weights`` 时按 query 块分块。
    """
    q_len = int(query.shape[2])

    def _attend(q: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, key.transpose(-2, -1)) * scaling
        if mask is not None:
            scores += mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if dropout > 0.0:
            weights = F.dropout(weights, p=dropout)
        return torch.matmul(weights, value), weights

    if return_weights or q_len <= _QUERY_CHUNK:
        attn_output, attn_weights = _attend(query, attention_mask)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    chunks: list[torch.Tensor] = []
    for start in range(0, q_len, _QUERY_CHUNK):
        end = min(start + _QUERY_CHUNK, q_len)
        mask = None if attention_mask is None else attention_mask[:, :, start:end, :]
        out, _ = _attend(query[:, :, start:end, :], mask)
        chunks.append(out)
    attn_output = torch.cat(chunks, dim=2)
    return attn_output.transpose(1, 2).contiguous(), None


def _causal_mask_fallback(
    q_len: int,
    kv_len: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """batch=1 无 padding 时的加性因果 mask，形状 ``(1, 1, q_len, kv_len)``。"""
    offset = kv_len - q_len
    mask = torch.full((q_len, kv_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=offset + 1)
    return mask[None, None, :, :]


def _make_cache_backend(
    kv_format: str,
    *,
    num_heads: int,
    head_dim: int,
    layout: str,
    device: torch.device,
    group_size: int,
    residual_length: int,
    seed: int,
    k_bits: int | None,
    v_bits: int | None,
    page_size: int,
    pte_bytes: int,
) -> CacheBackend:
    """按格式与布局构造 cache；与 ``AttentionWithCache._make_cache`` 同口径。"""
    resolved = get_codec(
        kv_format,
        dim=head_dim,
        group_size=group_size,
        residual_length=residual_length,
        seed=seed,
        k_bits=k_bits,
        v_bits=v_bits,
    )
    if isinstance(resolved, KiviFormat):
        return resolved.make_cache(
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            layout=layout,
            page_size=page_size,
            pte_bytes=pte_bytes,
        )
    if layout == "paged":
        return PagedUniformKVCache(
            resolved,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            pte_bytes=pte_bytes,
            device=device,
        )
    if layout != "contiguous":
        raise ValueError(f"layout 须为 contiguous / paged，得到 {layout!r}")
    assert isinstance(resolved, KVCodec)
    return ContiguousKVCache(resolved, num_heads=num_heads, head_dim=head_dim, device=device)


class LlamaCachePathAttention(nn.Module):
    """带本仓库 KV cache（C0–C5，contiguous / paged）的 Llama 多头注意力。

    接口对齐 transformers≥4.5x / 5.x 的 ``LlamaAttention.forward``。
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        *,
        kv_format: str,
        layout: str = "contiguous",
        group_size: int = 32,
        residual_length: int = 128,
        seed: int = 0,
        k_bits: int | None = None,
        v_bits: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        pte_bytes: int = DEFAULT_PTE_BYTES,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.kv_format = canonical_cache_format(kv_format)
        self.layout = layout
        self.group_size = group_size
        self.residual_length = residual_length
        self.seed = seed
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.page_size = page_size
        self.pte_bytes = pte_bytes

        if self.head_dim % group_size != 0:
            raise ValueError(f"head_dim={self.head_dim} 须能被 group_size={group_size} 整除")
        if self.kv_format in _KIVI_FORMATS and residual_length % group_size != 0:
            raise ValueError(
                f"residual_length={residual_length} 须能被 group_size={group_size} 整除"
            )
        if layout not in {"contiguous", "paged"}:
            raise ValueError(f"layout 须为 contiguous / paged，得到 {layout!r}")

        attn_bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=attn_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=attn_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=attn_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=attn_bias,
        )

        self.cache: CacheBackend | None = None

    @property
    def kivi_cache(self) -> CacheBackend | None:
        """兼容 M3：KIVI 路径曾用这个名字。"""
        return self.cache

    @kivi_cache.setter
    def kivi_cache(self, value: CacheBackend | None) -> None:
        self.cache = value

    @classmethod
    def from_hf_attention(
        cls,
        attn: nn.Module,
        *,
        kv_format: str,
        layout: str = "contiguous",
        group_size: int = 32,
        residual_length: int = 128,
        seed: int = 0,
        k_bits: int | None = None,
        v_bits: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        pte_bytes: int = DEFAULT_PTE_BYTES,
    ) -> LlamaCachePathAttention:
        """从已有 HF attention 拷贝权重，换成 cache-path 写/读。"""
        new = cls(
            attn.config,
            attn.layer_idx,
            kv_format=kv_format,
            layout=layout,
            group_size=group_size,
            residual_length=residual_length,
            seed=seed,
            k_bits=k_bits,
            v_bits=v_bits,
            page_size=page_size,
            pte_bytes=pte_bytes,
        )
        new.q_proj = attn.q_proj
        new.k_proj = attn.k_proj
        new.v_proj = attn.v_proj
        new.o_proj = attn.o_proj
        return new

    def _ensure_cache(self, device: torch.device) -> CacheBackend:
        if self.cache is None or self.cache.device != device:
            self.cache = _make_cache_backend(
                self.kv_format,
                num_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                layout=self.layout,
                device=device,
                group_size=self.group_size,
                residual_length=self.residual_length,
                seed=self.seed,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                page_size=self.page_size,
                pte_bytes=self.pte_bytes,
            )
        return self.cache

    def reset_cache(self) -> None:
        """清空本层 cache（新样本开始时调用）。"""
        if self.cache is not None:
            self.cache.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向：投影 + RoPE + cache 写/读 + attention。"""
        if position_embeddings is None:
            raise ValueError("LlamaCachePathAttention 需要 position_embeddings=(cos, sin)")

        bsz, q_len, _ = hidden_states.shape
        if bsz != 1:
            raise ValueError(
                f"LlamaCachePathAttention 当前仅支持 batch=1（协议默认），得到 batch={bsz}"
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        cache = self._ensure_cache(hidden_states.device)

        hf_len = 0
        if past_key_values is not None:
            hf_len = int(past_key_values.get_seq_length(self.layer_idx))
        if hf_len == 0 and len(cache) > 0:
            cache.clear()

        k_new = key_states.squeeze(0).transpose(0, 1).contiguous()
        v_new = value_states.squeeze(0).transpose(0, 1).contiguous()
        cache.append(k_new, v_new)

        k_all, v_all = cache.load()
        k_all = (
            k_all.transpose(0, 1)
            .unsqueeze(0)
            .to(dtype=query_states.dtype, device=query_states.device)
        )
        v_all = (
            v_all.transpose(0, 1)
            .unsqueeze(0)
            .to(dtype=query_states.dtype, device=query_states.device)
        )
        k_all = repeat_kv(k_all, self.num_key_value_groups)
        v_all = repeat_kv(v_all, self.num_key_value_groups)

        if past_key_values is not None:
            past_key_values.update(key_states, value_states, self.layer_idx)

        if attention_mask is None and q_len > 1:
            attention_mask = _causal_mask_fallback(
                q_len,
                int(k_all.shape[-2]),
                dtype=query_states.dtype,
                device=query_states.device,
            )

        output_attentions = bool(kwargs.get("output_attentions", False))
        dropout = 0.0 if not self.training else self.attention_dropout
        attn_output, attn_weights = _eager_attention(
            query_states,
            k_all,
            v_all,
            attention_mask,
            scaling=self.scaling,
            dropout=dropout,
            return_weights=output_attentions,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class LlamaKiviAttention(LlamaCachePathAttention):
    """C4/C5：``KiviKVCache`` 路径，保持 M3 ``bits=`` API。"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        *,
        bits: int = 2,
        k_bits: int | None = None,
        v_bits: int | None = None,
        group_size: int = 32,
        residual_length: int = 128,
        layout: str = "contiguous",
        seed: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
        pte_bytes: int = DEFAULT_PTE_BYTES,
    ) -> None:
        if bits not in (2, 4):
            raise ValueError(f"KIVI bits 须为 2 或 4，得到 {bits}")
        kv_format = "kivi2" if bits == 2 else "kivi4"
        super().__init__(
            config,
            layer_idx,
            kv_format=kv_format,
            layout=layout,
            group_size=group_size,
            residual_length=residual_length,
            seed=seed,
            k_bits=k_bits,
            v_bits=v_bits,
            page_size=page_size,
            pte_bytes=pte_bytes,
        )
        self.bits = bits
        self.k_bits = bits if k_bits is None else k_bits
        self.v_bits = bits if v_bits is None else v_bits

    @classmethod
    def from_hf_attention(
        cls,
        attn: nn.Module,
        *,
        bits: int = 2,
        k_bits: int | None = None,
        v_bits: int | None = None,
        group_size: int = 32,
        residual_length: int = 128,
        layout: str = "contiguous",
        seed: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
        pte_bytes: int = DEFAULT_PTE_BYTES,
        kv_format: str | None = None,
    ) -> LlamaKiviAttention:
        """从已有 HF attention 拷贝权重，换成 KIVI 写/读路径。"""
        if kv_format is not None:
            name = canonical_cache_format(kv_format)
            if name not in _KIVI_FORMATS:
                raise ValueError(f"LlamaKiviAttention 只接受 kivi2/kivi4，得到 {kv_format!r}")
            bits = 2 if name == "kivi2" else 4
        new = cls(
            attn.config,
            attn.layer_idx,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
            seed=seed,
            page_size=page_size,
            pte_bytes=pte_bytes,
        )
        new.q_proj = attn.q_proj
        new.k_proj = attn.k_proj
        new.v_proj = attn.v_proj
        new.o_proj = attn.o_proj
        return new

    @classmethod
    def from_llama_attention(
        cls,
        attn: LlamaAttention,
        *,
        bits: int = 2,
        k_bits: int | None = None,
        v_bits: int | None = None,
        group_size: int = 32,
        residual_length: int = 128,
        layout: str = "contiguous",
    ) -> LlamaKiviAttention:
        """从已有 ``LlamaAttention`` 拷贝权重，换成 KIVI 写/读路径。"""
        return cls.from_hf_attention(
            attn,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
        )


class MistralCachePathAttention(LlamaCachePathAttention):
    """Mistral 槽位的 C0–C3（及可选 C4/C5）cache-path。"""

    @classmethod
    def from_mistral_attention(
        cls,
        attn: nn.Module,
        *,
        kv_format: str,
        layout: str = "contiguous",
        group_size: int = 32,
        residual_length: int = 128,
        seed: int = 0,
        k_bits: int | None = None,
        v_bits: int | None = None,
    ) -> MistralCachePathAttention:
        """从已有 ``MistralAttention`` 拷贝权重。"""
        return cls.from_hf_attention(
            attn,
            kv_format=kv_format,
            layout=layout,
            group_size=group_size,
            residual_length=residual_length,
            seed=seed,
            k_bits=k_bits,
            v_bits=v_bits,
        )


class MistralKiviAttention(LlamaKiviAttention):
    """Mistral 槽位的 KIVI attention。

    数值路径与 ``LlamaKiviAttention`` 相同。Mistral 的 sliding-window causal mask
    由 ``MistralModel`` 传入，本模块只把它加到 scores 上。
    """

    @classmethod
    def from_mistral_attention(
        cls,
        attn: nn.Module,
        *,
        bits: int = 2,
        k_bits: int | None = None,
        v_bits: int | None = None,
        group_size: int = 32,
        residual_length: int = 128,
        layout: str = "contiguous",
    ) -> MistralKiviAttention:
        """从已有 ``MistralAttention`` 拷贝权重，换成 KIVI 写/读路径。"""
        return cls.from_hf_attention(
            attn,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            layout=layout,
        )


def _iter_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """取出 ``model.model.layers``（Llama / Mistral 因果 LM 的常见结构）。"""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise TypeError(
            f"无法定位 decoder layers；期望含 model.layers 的因果 LM，得到 {type(model).__name__}"
        )
    return list(layers)


def is_cache_path_patched(model: nn.Module) -> bool:
    """全部 decoder 层是否已换成 ``LlamaCachePathAttention``（含 KIVI 子类）。"""
    try:
        layers = _iter_decoder_layers(model)
    except TypeError:
        return False
    if not layers:
        return False
    return all(isinstance(layer.self_attn, LlamaCachePathAttention) for layer in layers)


def is_kivi_patched(model: nn.Module) -> bool:
    """全部 decoder 层的 ``self_attn`` 是否已换成 KIVI attention。"""
    try:
        layers = _iter_decoder_layers(model)
    except TypeError:
        return False
    if not layers:
        return False
    return all(isinstance(layer.self_attn, LlamaKiviAttention) for layer in layers)


def clear_llama_kivi_caches(model: nn.Module) -> None:
    """清空全部 cache-path attention 的 KV cache。"""
    for mod in model.modules():
        if isinstance(mod, LlamaCachePathAttention):
            mod.reset_cache()


def bytes_stored_llama_kivi(model: nn.Module) -> tuple[int, int]:
    """汇总全部 cache-path attention 的 ``(payload, metadata)`` bytes。"""
    payload = 0
    meta = 0
    for mod in model.modules():
        if isinstance(mod, LlamaCachePathAttention) and mod.cache is not None:
            p, m = mod.cache.bytes_stored()
            payload += p
            meta += m
    return payload, meta


clear_kivi_caches = clear_llama_kivi_caches
bytes_stored_kivi = bytes_stored_llama_kivi
clear_cache_path_caches = clear_llama_kivi_caches
bytes_stored_cache_path = bytes_stored_llama_kivi
