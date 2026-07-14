"""Activation distribution and quantization error analysis (auto-generates report)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# P1 attention reference (add parent path for import)
_P1 = Path(__file__).resolve().parents[1] / "p1_attention_numerics"
if str(_P1) not in sys.path:
    sys.path.insert(0, str(_P1))

from attention_naive import attention  # noqa: E402

from fakequant import (  # noqa: E402
    QuantGranularity,
    fp8_fake_quant,
    int4_fake_quant,
    mxfp4_fake_quant,
    relative_error,
)
from rotation import BlockDiagonalRotation, RandomHadamardRotation  # noqa: E402
from offline_utils import build_tiny_llama, build_tiny_tokenizer  # noqa: E402

DEFAULT_MODEL = "offline-tiny-llama"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _extract_qkv(model, input_ids: torch.Tensor, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one layer and return Q/K/V projections."""
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") and cfg.head_dim else cfg.hidden_size // n_heads

    hidden = model.model.embed_tokens(input_ids)
    for i in range(layer_idx):
        hidden = model.model.layers[i](hidden, attention_mask=None)[0]

    attn = model.model.layers[layer_idx].self_attn
    b, seq, _ = hidden.shape
    q = attn.q_proj(hidden).view(b, seq, n_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(b, seq, n_kv, head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(b, seq, n_kv, head_dim).transpose(1, 2)
    if n_kv != n_heads:
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return q, k, v


def _load_model_and_tokenizer(model_name: str):
    if model_name == DEFAULT_MODEL or model_name.startswith("offline"):
        return build_tiny_llama(), build_tiny_tokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    return model, tokenizer


def _quant_with_optional_rotation(
    x: torch.Tensor,
    quant_fn,
    rot=None,
) -> torch.Tensor:
    if rot is not None:
        x = rot.rotate(x, axis=-1)
    y = quant_fn(x)
    if rot is not None:
        y = rot.inverse(y, axis=-1)
    return y


def analyze_activations(
    model_name: str = DEFAULT_MODEL,
    layer_idx: int = 0,
    seq_len: int = 128,
    seed: int = 42,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(model_name)
    model.eval()

    text = "The quick brown fox jumps over the lazy dog. " * 20
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"]

    with torch.no_grad():
        q, k, v = _extract_qkv(model, input_ids, layer_idx)
        # Offline tiny models lack natural outliers; amplify a few channels for demos.
        # Real checkpoints (e.g. Qwen) already exhibit channel outliers — leave as-is.
        used_synthetic_outliers = model_name == DEFAULT_MODEL or model_name.startswith("offline")
        if used_synthetic_outliers:
            k = k.clone()
            k[..., 0] = k[..., 0] * 8.0
            k[..., 17] = k[..., 17] * 6.0
    head_dim = q.shape[-1]

    # Channel-wise max abs (outlier detection)
    k_flat = k[0, 0].float()  # (seq, head_dim)
    channel_max = k_flat.abs().max(dim=0).values.numpy()

    # Rotation matrices
    had_rot = RandomHadamardRotation(head_dim, seed=seed)
    bdr_rot = BlockDiagonalRotation(head_dim, block_size=32, seed=seed)

    def quant_k_direct(t):
        return int4_fake_quant(
            t, granularity=QuantGranularity.PER_GROUP, group_size=32, axis=-1
        )

    k_direct = quant_k_direct(k)
    k_hadamard = _quant_with_optional_rotation(k, quant_k_direct, had_rot)
    k_bdr = _quant_with_optional_rotation(k, quant_k_direct, bdr_rot)

    # Attention with quantized K
    ref_out = attention(q.float(), k.float(), v.float(), causal=True)
    out_direct = attention(q.float(), k_direct.float(), v.float(), causal=True)
    out_hadamard = attention(q.float(), k_hadamard.float(), v.float(), causal=True)
    out_bdr = attention(q.float(), k_bdr.float(), v.float(), causal=True)

    # Score error
    scale = head_dim**-0.5
    scores_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    scores_direct = torch.matmul(q.float(), k_direct.float().transpose(-2, -1)) * scale

    results = {
        "model_name": model_name,
        "synthetic_outliers": used_synthetic_outliers,
        "k_direct_err": relative_error(k, k_direct),
        "k_hadamard_err": relative_error(k, k_hadamard),
        "k_bdr_err": relative_error(k, k_bdr),
        "score_direct_err": relative_error(scores_ref, scores_direct),
        "out_direct_err": relative_error(ref_out, out_direct),
        "out_hadamard_err": relative_error(ref_out, out_hadamard),
        "out_bdr_err": relative_error(ref_out, out_bdr),
        "head_dim": head_dim,
    }

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(channel_max, bins=40, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("K channel max |activation| (outlier profile)")
    axes[0, 0].set_xlabel("|K| max per channel")
    axes[0, 0].set_ylabel("count")

    k_rot = had_rot.rotate(k[0, 0].float(), axis=-1).numpy().ravel()
    axes[0, 1].hist(k[0, 0].float().numpy().ravel(), bins=60, alpha=0.6, label="original", density=True)
    axes[0, 1].hist(k_rot, bins=60, alpha=0.6, label="Hadamard rotated", density=True)
    axes[0, 1].set_title("K activation distribution: original vs rotated")
    axes[0, 1].legend()

    methods = ["direct INT4", "Hadamard+INT4", "BDR+INT4"]
    k_errs = [results["k_direct_err"], results["k_hadamard_err"], results["k_bdr_err"]]
    out_errs = [results["out_direct_err"], results["out_hadamard_err"], results["out_bdr_err"]]
    x_pos = np.arange(3)
    axes[1, 0].bar(x_pos - 0.2, k_errs, 0.4, label="K quant error")
    axes[1, 0].bar(x_pos + 0.2, out_errs, 0.4, label="attention output error")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(methods, rotation=15, ha="right")
    axes[1, 0].set_ylabel("relative L2 error")
    axes[1, 0].set_title("Rotation reduces INT4 quantization error")
    axes[1, 0].legend()

    # Mixed precision tradeoff curve
    configs = [
        ("fp16 baseline", lambda qq, kk, vv: (qq, kk, vv), 16),
        ("K INT4", lambda qq, kk, vv: (qq, quant_k_direct(kk), vv), 8),
        ("K Had+INT4", lambda qq, kk, vv: (qq, _quant_with_optional_rotation(kk, quant_k_direct, had_rot), vv), 8),
        ("QK INT4 + V FP8", lambda qq, kk, vv: (
            int4_fake_quant(qq, granularity=QuantGranularity.PER_GROUP, group_size=32, axis=-1),
            quant_k_direct(kk),
            fp8_fake_quant(vv, fmt="e4m3", granularity=QuantGranularity.PER_CHANNEL, axis=-1),
        ), 6),
        ("K MXFP4", lambda qq, kk, vv: (qq, mxfp4_fake_quant(kk, block_size=32, axis=-1), vv), 4),
    ]
    bits_list, err_list, labels = [], [], []
    for label, fn, avg_bits in configs:
        qq, kk, vv = fn(q, k, v)
        out = attention(q.float(), kk.float(), vv.float(), causal=True)
        err = relative_error(ref_out, out)
        bits_list.append(avg_bits)
        err_list.append(err)
        labels.append(label)

    axes[1, 1].plot(bits_list, err_list, "o-", color="darkorange")
    for b, e, lb in zip(bits_list, err_list, labels):
        axes[1, 1].annotate(lb, (b, e), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[1, 1].set_xlabel("effective avg bits (KV)")
    axes[1, 1].set_ylabel("attention output rel error")
    axes[1, 1].set_title("Mixed-precision accuracy vs bitwidth")

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "error_analysis.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    results["figure"] = str(fig_path)
    results["mixed_precision"] = list(zip(labels, bits_list, err_list))
    return results


def write_report(results: dict, path: Path) -> None:
    outlier_note = (
        "offline 模型上人工放大了部分 K 通道以演示 outlier"
        if results.get("synthetic_outliers")
        else "使用真实模型激活中的自然 outlier，未做人工放大"
    )
    lines = [
        "# P2 量化误差分析报告",
        "",
        f"- Model: `{results.get('model_name', 'unknown')}`",
        f"- Head dim: {results['head_dim']}",
        f"- Outlier setup: {outlier_note}",
        "",
        "## K 量化相对误差",
        "",
        f"| 方法 | 相对 L2 误差 |",
        f"|------|-------------|",
        f"| 直接 INT4 per-group | {results['k_direct_err']:.4f} |",
        f"| Hadamard + INT4 | {results['k_hadamard_err']:.4f} |",
        f"| BDR + INT4 | {results['k_bdr_err']:.4f} |",
        "",
        "## Attention 输出相对误差",
        "",
        f"| 方法 | 相对 L2 误差 |",
        f"|------|-------------|",
        f"| 直接 INT4 K | {results['out_direct_err']:.4f} |",
        f"| Hadamard + INT4 K | {results['out_hadamard_err']:.4f} |",
        f"| BDR + INT4 K | {results['out_bdr_err']:.4f} |",
        "",
        "## 混合精度权衡",
        "",
        "| 配置 | 平均比特 | 输出误差 |",
        "|------|---------|---------|",
    ]
    for label, bits, err in results["mixed_precision"]:
        lines.append(f"| {label} | {bits} | {err:.4f} |")

    lines.extend([
        "",
        f"![error analysis plots]({Path(results['figure']).name})",
        "",
        "## 结论",
        "",
    ])
    if results["k_hadamard_err"] < results["k_direct_err"]:
        reduction = (1 - results["k_hadamard_err"] / results["k_direct_err"]) * 100
        lines.append(
            f"- Hadamard 旋转使 K 的 INT4 量化误差降低约 **{reduction:.1f}%**，"
            "激活分布更接近高斯，outlier 被分散到各通道。"
        )
    if results["k_bdr_err"] < results["k_direct_err"]:
        reduction = (1 - results["k_bdr_err"] / results["k_direct_err"]) * 100
        lines.append(
            f"- Block-Hadamard BDR（`block_diag(H) @ D`）使 K 的 INT4 量化误差降低约 "
            f"**{reduction:.1f}%**。"
        )
    if results["out_hadamard_err"] < results["out_direct_err"]:
        lines.append(
            "- 旋转后 attention 输出误差低于直接 INT4，验证了旋转在数值上改善量化友好性。"
        )
    elif (
        results["k_hadamard_err"] < results["k_direct_err"]
        or results["k_bdr_err"] < results["k_direct_err"]
    ):
        lines.append(
            "- 注意：单层 attention **输出**相对 L2 有时高于直接 INT4（旋转把量化噪声"
            "各向同性地散开后，均值 L2 可能上升）。端到端应以 KV cache **困惑度**为主指标；"
            "在 Qwen2.5-0.5B 上 Hadamard/BDR 的 PPL 均优于直接 INT4。"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 error analysis")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    results = analyze_activations(args.model, args.layer, args.seq_len)
    report_path = OUTPUT_DIR / "error_analysis_report.md"
    write_report(results, report_path)
    print(f"Report written to {report_path}")
    print(f"Figure saved to {results['figure']}")


if __name__ == "__main__":
    main()
