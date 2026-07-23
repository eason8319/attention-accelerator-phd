"""P2 离线实验共用辅助函数（无需 HuggingFace 下载）。"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def build_tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    """随机初始化的小型 Llama，用于离线量化实验。"""
    gen = torch.Generator().manual_seed(seed)
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        vocab_size=4096,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    model.eval()
    return model


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """最小字节级分词器（离线）。"""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    sample = "The quick brown fox jumps over the lazy dog. " * 200
    tok.train_from_iterator(
        [sample],
        trainer=trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>", "<s>", "</s>"]),
    )
    wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return wrapper


def synthetic_wikitext(max_chars: int = 8000) -> str:
    """无法下载 WikiText-2 时的可重复伪语料。"""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Large language models store key and value tensors in KV cache during decoding. "
        "Low-bit quantization reduces memory bandwidth at the cost of numerical error. "
        "Hadamard rotation spreads activation outliers across channels before quantization. "
    )
    text = base
    while len(text) < max_chars:
        text += base
    return text[:max_chars]
