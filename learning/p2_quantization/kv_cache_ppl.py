"""WikiText-2（或离线合成语料）上 KV cache INT4 量化的困惑度评估。"""

from __future__ import annotations

import argparse
import math
from typing import Literal

import torch
import torch.nn as nn
from fakequant import QuantGranularity, int4_fake_quant
from offline_utils import build_tiny_llama, build_tiny_tokenizer, synthetic_wikitext
from rotation import BlockDiagonalRotation, RandomHadamardRotation
from transformers import AutoModelForCausalLM, AutoTokenizer

Mode = Literal["fp16", "int4", "int4_hadamard", "int4_bdr"]


class _QuantizedLinearWrapper(nn.Module):
    """在线性层输出上包装假量化（模拟量化 KV）。"""

    def __init__(self, linear: nn.Linear, quant_fn):
        super().__init__()
        self.linear = linear
        self.quant_fn = quant_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_fn(self.linear(x))


def _make_quant_fn(mode: Mode, head_dim: int, seed: int):
    if mode == "fp16":
        return lambda t: t

    had = RandomHadamardRotation(head_dim, seed=seed) if mode == "int4_hadamard" else None
    bdr = BlockDiagonalRotation(head_dim, block_size=32, seed=seed) if mode == "int4_bdr" else None
    rot = had or bdr

    def fn(t: torch.Tensor) -> torch.Tensor:
        x = t
        if rot is not None:
            # 投影形状为 (batch, seq, hidden)；reshape 后在各 head 切片上旋转最后一维
            b, s, h = x.shape
            n_heads = h // head_dim
            x = x.view(b, s, n_heads, head_dim)
            x = rot.rotate(x, axis=-1)
            x = x.reshape(b, s, h)
        q = int4_fake_quant(x, granularity=QuantGranularity.PER_GROUP, group_size=32, axis=-1)
        if rot is not None:
            b, s, h = q.shape
            n_heads = h // head_dim
            q = q.view(b, s, n_heads, head_dim)
            q = rot.inverse(q, axis=-1)
            q = q.reshape(b, s, h)
        return q

    return fn


def _patch_kv_projections(
    model: nn.Module, mode: Mode, seed: int = 0
) -> list[tuple[nn.Module, nn.Module]]:
    """用量化包装器包裹 k_proj 与 v_proj；返回用于恢复的句柄。"""
    head_dim = (
        model.config.head_dim
        if getattr(model.config, "head_dim", None)
        else (model.config.hidden_size // model.config.num_attention_heads)
    )
    quant_fn = _make_quant_fn(mode, head_dim, seed)
    handles: list[tuple[nn.Module, nn.Module]] = []

    if mode == "fp16":
        return handles

    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ("k_proj", "v_proj"):
            orig = getattr(attn, name)
            wrapped = _QuantizedLinearWrapper(orig, quant_fn)
            setattr(attn, name, wrapped)
            handles.append((attn, name, orig))
    return handles


def _restore_projections(handles: list) -> None:
    for attn, name, orig in handles:
        setattr(attn, name, orig)


@torch.no_grad()
def eval_ppl(
    model_name: str,
    mode: Mode,
    max_length: int = 512,
    stride: int = 256,
    max_samples: int = 20,
) -> float:
    if model_name.startswith("offline") or model_name == "offline-tiny-llama":
        tokenizer = build_tiny_tokenizer()
        model = build_tiny_llama()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
    model.eval()

    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if t.strip()][:max_samples]
        full_text = "\n\n".join(texts)
    except Exception:
        full_text = synthetic_wikitext(max_chars=max_length * max_samples)

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    total_len = input_ids.shape[1]

    handles = _patch_kv_projections(model, mode)
    nlls = []

    for begin in range(0, total_len - 1, stride):
        end = min(begin + max_length, total_len)
        if end - begin < 2:
            break
        chunk = input_ids[:, begin:end]
        outputs = model(chunk, labels=chunk)
        loss = outputs.loss
        if loss is not None and not math.isnan(loss.item()):
            nlls.append(loss.item())
        if end >= total_len:
            break

    _restore_projections(handles)
    if not nlls:
        return float("nan")
    return math.exp(sum(nlls) / len(nlls))


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache INT4 困惑度评估")
    parser.add_argument("--model", default="offline-tiny-llama")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=15)
    args = parser.parse_args()

    modes: list[Mode] = ["fp16", "int4", "int4_hadamard", "int4_bdr"]
    results = {}
    for mode in modes:
        print(f"Evaluating {mode}...")
        ppl = eval_ppl(
            args.model,
            mode,
            max_length=args.max_length,
            stride=args.stride,
            max_samples=args.max_samples,
        )
        results[mode] = ppl
        print(f"  PPL = {ppl:.2f}")

    print("\n=== Summary ===")
    for mode, ppl in results.items():
        print(f"{mode:16s}: {ppl:.2f}")

    out_path = Path(__file__).resolve().parent / "outputs" / "kv_cache_ppl.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{mode}: {ppl:.4f}" for mode, ppl in results.items()]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    if results["int4_hadamard"] < results["int4"]:
        print("\n✓ Hadamard+INT4 PPL better than direct INT4 (rotation helps)")
    if results["int4_bdr"] < results["int4"]:
        print("✓ BDR+INT4 PPL better than direct INT4 (block-Hadamard helps)")


if __name__ == "__main__":
    from pathlib import Path

    main()
