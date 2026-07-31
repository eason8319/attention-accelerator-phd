"""Llama Attention：真实 KIVI cache-path 接入（本仓库 KiviKVCache）。

将 ``q/k/v_proj → RoPE → KiviKVCache.append/load → SDPA → o_proj`` 接到
HuggingFace Llama 的 attention 槽位，供 ``generate`` / LM-Eval 使用。

约定：
  - 当前仅 ``batch=1``（与 R1 协议一致）
  - Attention 数值路径使用 ``KiviKVCache.load()`` 的反量化 K/V（非投影 fake-quant）
  - 仍调用 HF ``past_key_values.update`` 以维护 generate 的序列长度簿记
    （短任务可接受与 Kivi 并存；长上下文省显存优化留后续）

用法概要：

  attn = LlamaKiviAttention.from_llama_attention(old_attn, bits=2)
  layer.self_attn = attn
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

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

# cache_path 与 kivi_repro 并列，按路径注入
_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache_path"
if str(_CACHE_PATH) not in sys.path:
    sys.path.insert(0, str(_CACHE_PATH))

from kv_cache import KiviKVCache  # noqa: E402


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """手动 SDPA；形状均为 ``(B, H, T, D)``。"""
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


class LlamaKiviAttention(nn.Module):
    """带本仓库 ``KiviKVCache`` 的 Llama 多头注意力。

    接口对齐 transformers≥4.5x / 5.x 的 ``LlamaAttention.forward``
    （``position_embeddings=(cos,sin)`` + ``past_key_values: Cache``）。
    """

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
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.bits = bits
        self.k_bits = bits if k_bits is None else k_bits
        self.v_bits = bits if v_bits is None else v_bits
        self.group_size = group_size
        self.residual_length = residual_length

        if self.head_dim % group_size != 0:
            raise ValueError(
                f"head_dim={self.head_dim} 须能被 group_size={group_size} 整除"
            )
        if residual_length % group_size != 0:
            raise ValueError(
                f"residual_length={residual_length} 须能被 group_size={group_size} 整除"
            )

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # 延迟到首次 forward（绑定 device）
        self.kivi_cache: KiviKVCache | None = None

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
    ) -> LlamaKiviAttention:
        """从已有 ``LlamaAttention`` 拷贝权重，换成 KIVI 写/读路径。"""
        new = cls(
            attn.config,
            attn.layer_idx,
            bits=bits,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
        )
        new.q_proj = attn.q_proj
        new.k_proj = attn.k_proj
        new.v_proj = attn.v_proj
        new.o_proj = attn.o_proj
        return new

    def _ensure_cache(self, device: torch.device) -> KiviKVCache:
        if self.kivi_cache is None:
            self.kivi_cache = KiviKVCache(
                num_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                bits=self.bits,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                group_size=self.group_size,
                residual_length=self.residual_length,
                device=device,
            )
        elif self.kivi_cache.device != device:
            # 权重迁移后重建空 cache（不跨 device 搬量化缓冲）
            self.kivi_cache = KiviKVCache(
                num_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                bits=self.bits,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                group_size=self.group_size,
                residual_length=self.residual_length,
                device=device,
            )
        return self.kivi_cache

    def reset_cache(self) -> None:
        """清空本层 KIVI cache（新样本开始时调用）。"""
        if self.kivi_cache is not None:
            self.kivi_cache.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向：投影 + RoPE + Kivi 写/读 + attention。

        参数
            hidden_states: ``(batch, seq, hidden)``，当前仅支持 ``batch=1``。
            position_embeddings: ``(cos, sin)``，与 HF Llama 一致。
            attention_mask: 加性 mask，或 ``None``（依赖外部 causal 构造）。
            past_key_values: HF Cache；用于序列长度簿记，attention 不用其 FP16 KV。

        返回
            ``(attn_output, attn_weights)``；``attn_weights`` 在非 output_attentions 时可为 ``None``。
        """
        if position_embeddings is None:
            raise ValueError("LlamaKiviAttention 需要 position_embeddings=(cos, sin)")

        bsz, q_len, _ = hidden_states.shape
        if bsz != 1:
            raise ValueError(
                f"LlamaKiviAttention 当前仅支持 batch=1（协议默认），得到 batch={bsz}"
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        cache = self._ensure_cache(hidden_states.device)

        # 新序列：HF cache 该层长度为 0 → 清空 Kivi，避免跨样本污染
        hf_len = 0
        if past_key_values is not None:
            hf_len = int(past_key_values.get_seq_length(self.layer_idx))
        if hf_len == 0 and len(cache) > 0:
            cache.clear()

        # 写路径：仅追加本步新 token（prefill 可为多 token）
        # HF: (1, H_kv, T, D) → cache: (T, H_kv, D)
        k_new = key_states.squeeze(0).transpose(0, 1).contiguous()
        v_new = value_states.squeeze(0).transpose(0, 1).contiguous()
        cache.append(k_new, v_new)

        # 读路径：反量化历史 + 残差窗
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

        # HF 簿记（generate 依赖 Cache 长度）；不参与下方 attention 数值
        if past_key_values is not None:
            past_key_values.update(key_states, value_states, self.layer_idx)

        dropout = 0.0 if not self.training else self.attention_dropout
        attn_output, attn_weights = _eager_attention(
            query_states,
            k_all,
            v_all,
            attention_mask,
            scaling=self.scaling,
            dropout=dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        output_attentions = bool(kwargs.get("output_attentions", False))
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


def clear_llama_kivi_caches(model: nn.Module) -> None:
    """遍历模型，清空所有 ``LlamaKiviAttention`` 的 Kivi cache。"""
    for mod in model.modules():
        if isinstance(mod, LlamaKiviAttention):
            mod.reset_cache()


def bytes_stored_llama_kivi(model: nn.Module) -> tuple[int, int]:
    """汇总全部 ``LlamaKiviAttention`` 的 ``(payload, metadata)`` bytes。"""
    payload = 0
    meta = 0
    for mod in model.modules():
        if isinstance(mod, LlamaKiviAttention) and mod.kivi_cache is not None:
            p, m = mod.kivi_cache.bytes_stored()
            payload += p
            meta += m
    return payload, meta
