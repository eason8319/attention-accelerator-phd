#!/usr/bin/env python3
"""Download and organize survey literature PDFs."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER_AGENT = "Mozilla/5.0 (compatible; literature-downloader/1.0)"
TIMEOUT = 120


PAPERS: list[dict] = [
    # 00 baseline
    {"key": "vaswani2017attention", "dir": "00_baseline", "file": "2017_Vaswani_Attention_Is_All_You_Need.pdf", "urls": ["https://arxiv.org/pdf/1706.03762.pdf"]},
    {"key": "dao2022flashattention", "dir": "00_baseline", "file": "2022_Dao_FlashAttention.pdf", "urls": ["https://arxiv.org/pdf/2205.14135.pdf"]},
    {"key": "dao2023flashattention2", "dir": "00_baseline", "file": "2023_Dao_FlashAttention-2.pdf", "urls": ["https://arxiv.org/pdf/2307.08691.pdf"]},
    {"key": "shah2024flashattention3", "dir": "00_baseline", "file": "2024_Shah_FlashAttention-3.pdf", "urls": ["https://arxiv.org/pdf/2407.08608.pdf"]},
    {"key": "dao2023flashdecoding", "dir": "00_baseline", "file": "2023_Dao_Flash-Decoding.pdf", "urls": ["https://arxiv.org/pdf/2309.17453.pdf"]},
    {"key": "zhang2019root", "dir": "00_baseline", "file": "2019_Zhang_RMSNorm.pdf", "urls": ["https://arxiv.org/pdf/1910.07467.pdf"]},
    {"key": "su2024roformer", "dir": "00_baseline", "file": "2024_Su_RoFormer_RoPE.pdf", "urls": ["https://arxiv.org/pdf/2104.09864.pdf"]},
    # L1
    {"key": "lin2025systolicattention", "dir": "L1_flashattention", "file": "2025_Lin_SystolicAttention_FSA.pdf", "urls": ["https://arxiv.org/pdf/2507.11331.pdf"]},
    {"key": "plena2025", "dir": "L1_flashattention", "file": "2025_PLENA_Long_Context_LLM_Inference.pdf", "urls": ["https://arxiv.org/pdf/2509.09505.pdf"]},
    {"key": "flatattention2026", "dir": "L1_flashattention", "file": "2026_FlatAttention_Tile_Accelerator.pdf", "urls": ["https://arxiv.org/pdf/2604.02110.pdf"]},
    {"key": "streamattention2025", "dir": "L1_flashattention", "file": "2025_StreamAttention_Systolic.pdf", "urls": ["https://openreview.net/pdf/a39a0010a18ace89b15cdc1c381334c93d7a4955.pdf"]},
    {"key": "wang2023cosa", "dir": "L1_flashattention", "file": "2023_Wang_COSA_Systolic_Attention.pdf", "urls": ["https://arxiv.org/pdf/2305.10725.pdf", "https://people.eecs.berkeley.edu/~ysshao/assets/papers/cosa-micro23.pdf"]},
    {"key": "desa2025", "dir": "L1_flashattention", "file": "2025_DESA_Transformer_Systolic.pdf", "urls": ["https://arxiv.org/pdf/2403.16564.pdf"]},
    {"key": "ham2021a3", "dir": "L1_flashattention", "file": "2021_Ham_A3_Attention_Approximation.pdf", "urls": ["https://arxiv.org/pdf/2102.03977.pdf"]},
    {"key": "wang2021spatten", "dir": "L1_flashattention", "file": "2021_Wang_SpAtten_Sparse_Attention.pdf", "urls": ["https://arxiv.org/pdf/2012.09850.pdf"]},
    {"key": "kao2023flat", "dir": "L1_flashattention", "file": "2023_Kao_FLAT_Attention_Dataflow.pdf", "urls": ["https://arxiv.org/pdf/2306.06536.pdf"]},
    # L2
    {"key": "sawint42026", "dir": "L2_mixed_precision", "file": "2026_SAW-INT4_KV_Cache_Quantization.pdf", "urls": ["https://arxiv.org/pdf/2604.19157.pdf"]},
    {"key": "bitdecoding2026", "dir": "L2_mixed_precision", "file": "2026_BitDecoding_Low-Bit_KV_Cache.pdf", "urls": ["https://arxiv.org/pdf/2503.18773.pdf"]},
    {"key": "batquant2026", "dir": "L2_mixed_precision", "file": "2026_BATQuant_MXFP4.pdf", "urls": ["https://arxiv.org/pdf/2603.16590.pdf"]},
    {"key": "liu2024kivi", "dir": "L2_mixed_precision", "file": "2024_Liu_KIVI_2bit_KV_Cache.pdf", "urls": ["https://arxiv.org/pdf/2402.02750.pdf"]},
    {"key": "ashkboos2024quarot", "dir": "L2_mixed_precision", "file": "2024_Ashkboos_QuaRot_INT4_Inference.pdf", "urls": ["https://arxiv.org/pdf/2404.00456.pdf"]},
    {"key": "liu2024spinquant", "dir": "L2_mixed_precision", "file": "2024_Liu_SpinQuant_Learned_Rotations.pdf", "urls": ["https://arxiv.org/pdf/2405.16406.pdf"]},
    {"key": "ultraquant2026", "dir": "L2_mixed_precision", "file": "2026_UltraQuant_4bit_KV_Caching.pdf", "urls": ["https://arxiv.org/pdf/2606.20474.pdf"]},
    {"key": "xiao2023smoothquant", "dir": "L2_mixed_precision", "file": "2023_Xiao_SmoothQuant.pdf", "urls": ["https://arxiv.org/pdf/2211.10438.pdf"]},
    {"key": "frantar2023gptq", "dir": "L2_mixed_precision", "file": "2023_Frantar_GPTQ.pdf", "urls": ["https://arxiv.org/pdf/2210.17323.pdf"]},
    {"key": "micikevicius2022fp8", "dir": "L2_mixed_precision", "file": "2022_Micikevicius_FP8_Formats.pdf", "urls": ["https://arxiv.org/pdf/2209.05433.pdf"]},
    # L3
    {"key": "mive2026", "dir": "L3_non_gemm", "file": "2026_MIVE_Softmax_LayerNorm_RMSNorm.pdf", "urls": ["https://arxiv.org/pdf/2606.17781.pdf"]},
    {"key": "sole2025", "dir": "L3_non_gemm", "file": "2025_SOLE_Softmax_LayerNorm_Codesign.pdf", "urls": ["https://arxiv.org/pdf/2510.17189.pdf"]},
    {"key": "softmaxrmsnorm2026", "dir": "L3_non_gemm", "file": "2026_Hardware_Softmax_RMSNorm_Approx.pdf", "urls": ["https://europepmc.org/articles/PMC12843610?pdf=render", "https://www.mdpi.com/2072-666X/17/1/84/pdf"]},
    {"key": "guaranteednorm2026", "dir": "L3_non_gemm", "file": "2026_Guaranteed_Normalization_Softmax_LayerNorm.pdf", "urls": ["https://arxiv.org/pdf/2604.23647.pdf"]},
    {"key": "koca2023exp", "dir": "L3_non_gemm", "file": "2023_Koca_Exp_Approximation.pdf", "urls": ["https://arxiv.org/pdf/2303.04545.pdf"]},
    {"key": "sun2022softmax", "dir": "L3_non_gemm", "file": "2022_Sun_Softermax.pdf", "urls": ["https://arxiv.org/pdf/2203.05924.pdf"]},
    # L4
    {"key": "parashar2019timeloop", "dir": "L4_compiler", "file": "2019_Parashar_Timeloop.pdf", "urls": ["https://arxiv.org/pdf/1901.04677.pdf", "https://parashar.org/timeloop.pdf"]},
    {"key": "wu2019accelergy", "dir": "L4_compiler", "file": "2019_Wu_Accelergy.pdf", "urls": ["https://arxiv.org/pdf/1910.10509.pdf"]},
    {"key": "scalesimv3_2025", "dir": "L4_compiler", "file": "2025_SCALE-Sim_v3.pdf", "urls": ["https://arxiv.org/pdf/2504.15377.pdf"]},
    {"key": "klhufek2025transinfersim", "dir": "L4_compiler", "file": "2025_Klhufek_TransInferSim.pdf", "urls": ["https://dspace.vut.cz/bitstreams/108c95b8-c42b-4f18-9210-9996037b3ae3/download", "https://arxiv.org/pdf/2509.00000.pdf"]},
    {"key": "tilelang2025", "dir": "L4_compiler", "file": "2025_TileLang_AI_Kernel_DSL.pdf", "urls": ["https://arxiv.org/pdf/2504.17577.pdf"]},
    {"key": "chen2016eyeriss", "dir": "L4_compiler", "file": "2016_Chen_Eyeriss.pdf", "urls": ["https://arxiv.org/pdf/1606.06140.pdf"]},
    {"key": "ma2024tpuv4i", "dir": "L4_compiler", "file": "2024_Google_TPU_v4.pdf", "urls": ["https://arxiv.org/pdf/2304.01433.pdf"]},
    # adjacent A
    {"key": "salca2026", "dir": "adjacent_sparse", "file": "2026_Salca_Sparse_Long_Context_Decoding.pdf", "urls": ["https://arxiv.org/pdf/2604.24820.pdf"]},
    {"key": "liu2021sanger", "dir": "adjacent_sparse", "file": "2021_Liu_Sanger_Sparse_Attention.pdf", "urls": ["https://arxiv.org/pdf/2107.03610.pdf"]},
    {"key": "dao2022flashattention_sparse", "dir": "adjacent_sparse", "file": "2022_Dao_FlashAttention_Block-Sparse.pdf", "urls": ["https://arxiv.org/pdf/2307.02491.pdf"]},
    # adjacent B
    {"key": "amma2026", "dir": "adjacent_memory", "file": "2026_AMMA_Memory_Centric_Attention.pdf", "urls": ["https://arxiv.org/pdf/2604.26103.pdf"]},
    {"key": "lolpim2024", "dir": "adjacent_memory", "file": "2024_LoL-PIM_Long_Context_Decoding.pdf", "urls": ["https://arxiv.org/pdf/2412.20166.pdf"]},
    {"key": "heo2024nuepim", "dir": "adjacent_memory", "file": "2024_Heo_NeuPIM_NPU-PIM.pdf", "urls": ["https://arxiv.org/pdf/2311.01120.pdf"]},
]


def is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def download(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
            if is_pdf(data):
                return data
            # Some hosts redirect HTML landing pages
            if b"application/pdf" in (resp.headers.get("Content-Type") or "").encode():
                return data if is_pdf(data) else None
            return None
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def main() -> None:
    manifest: list[dict] = []
    for item in PAPERS:
        target_dir = ROOT / item["dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / item["file"]
        status = "skipped_exists"
        source = None

        if not target.exists() or target.stat().st_size < 1024:
            ok = False
            for url in item["urls"]:
                data = download(url)
                if data:
                    target.write_bytes(data)
                    source = url
                    status = "downloaded"
                    ok = True
                    break
                time.sleep(0.5)
            if not ok:
                status = "failed"
        else:
            source = "existing"

        size_kb = round(target.stat().st_size / 1024, 1) if target.exists() else 0
        manifest.append(
            {
                "key": item["key"],
                "dir": item["dir"],
                "file": item["file"],
                "path": str(target.relative_to(ROOT)),
                "status": status,
                "source": source,
                "size_kb": size_kb,
            }
        )
        print(f"[{status}] {item['key']} -> {target.name} ({size_kb} KB)")

    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    ok_count = sum(1 for m in manifest if m["status"] in ("downloaded", "skipped_exists") and m["size_kb"] > 1)
    fail_count = sum(1 for m in manifest if m["status"] == "failed" or m["size_kb"] <= 1)
    print(f"\nDone: {ok_count} ok, {fail_count} failed/missing")


if __name__ == "__main__":
    main()
