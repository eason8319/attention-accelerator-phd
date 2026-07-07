"""Numerical equivalence tests for P1 attention implementations."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from attention_naive import attention
from attention_online import attention_online
from attention_tiled import attention_tiled
from decode_step import decode_step_attention, prefill_last_token
from ops import apply_rope, build_rope_cache, rms_norm


def _make_qkv(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, generator=gen, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, generator=gen, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, generator=gen, dtype=dtype)
    return q, k, v


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [128, 512])
def test_naive_vs_sdpa(causal: bool, seq: int) -> None:
    q, k, v = _make_qkv(2, 4, seq, 64)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = attention(q, k, v, causal=causal)
    assert _max_abs(out, ref) < 1e-5
    assert _cosine(out, ref) > 1.0 - 1e-6


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [16, 64, 128])
def test_tiled_vs_naive(causal: bool, block_size: int) -> None:
    q, k, v = _make_qkv(1, 2, 256, 32)
    ref = attention(q, k, v, causal=causal)
    out = attention_tiled(q, k, v, block_size=block_size, causal=causal)
    assert _max_abs(out, ref) < 1e-5


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [17, 64, 100])
def test_online_vs_naive(causal: bool, block_size: int) -> None:
    q, k, v = _make_qkv(1, 2, 300, 32)
    ref = attention(q, k, v, causal=causal)
    out = attention_online(q, k, v, block_size=block_size, causal=causal)
    assert _max_abs(out, ref) < 1e-5
    assert _cosine(out, ref) > 1.0 - 1e-6


def test_online_block_size_independence() -> None:
    q, k, v = _make_qkv(1, 2, 512, 64, seed=7)
    out16 = attention_online(q, k, v, block_size=16, causal=True)
    out128 = attention_online(q, k, v, block_size=128, causal=True)
    assert _max_abs(out16, out128) < 1e-5


def test_online_fp16() -> None:
    q, k, v = _make_qkv(1, 2, 256, 64, dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = attention_online(q, k, v, block_size=32, causal=True)
    tol = 1e-3
    assert _max_abs(out, ref) < tol


def test_rope_matches_transformers() -> None:
    pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    batch, heads, seq, head_dim = 2, 4, 16, 64
    x = torch.randn(batch, heads, seq, head_dim)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size=head_dim * heads,
        num_attention_heads=heads,
        head_dim=head_dim,
    )
    ref_module = LlamaRotaryEmbedding(config=config)

    cos_ref, sin_ref = ref_module(x, position_ids)
    q_ref, _ = apply_rotary_pos_emb(x, x, cos_ref, sin_ref)
    ref = q_ref

    cos, sin = build_rope_cache(seq, head_dim, device=x.device, dtype=x.dtype)
    out = apply_rope(x, cos, sin)
    assert _max_abs(out, ref) < 1e-4


def test_rope_manual_formula() -> None:
    """Self-consistency check independent of transformers version."""
    batch, heads, seq, head_dim = 1, 2, 8, 32
    x = torch.randn(batch, heads, seq, head_dim)
    cos, sin = build_rope_cache(seq, head_dim)

    out = apply_rope(x, cos, sin)
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    cos_h, sin_h = cos[..., : head_dim // 2], sin[..., : head_dim // 2]
    while cos_h.ndim < x.ndim:
        cos_h = cos_h.unsqueeze(0)
        sin_h = sin_h.unsqueeze(0)
    expected = torch.cat(
        [x1 * cos_h - x2 * sin_h, x1 * sin_h + x2 * cos_h],
        dim=-1,
    )
    assert _max_abs(out, expected) < 1e-6


def test_rms_norm_matches_transformers() -> None:
    pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    x = torch.randn(2, 8, 64)
    weight = torch.ones(64)
    ref_module = LlamaRMSNorm(64)
    ref_module.weight.data.copy_(weight)
    ref = ref_module(x)
    out = rms_norm(x, weight)
    assert _max_abs(out, ref) < 1e-5


def test_decode_matches_prefill_last_token() -> None:
    q, k, v = _make_qkv(2, 4, 128, 32, seed=3)
    decode_q = q[:, :, -1:, :]
    decode_out = decode_step_attention(decode_q, k, v)
    prefill_out = prefill_last_token(q, k, v, causal=True)
    assert _max_abs(decode_out, prefill_out) < 1e-5
