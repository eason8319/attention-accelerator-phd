"""保存评测参数、输入及实际源码标识，不自动撰写实验报告。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from .tensor_metrics import METRIC_VERSION


def tensor_sha256(tensor: torch.Tensor) -> str:
    """按 dtype、形状和连续原始字节标识输入。"""
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def evaluation_provenance(root: Path, args: Any) -> dict[str, Any]:
    """记录实际执行源码哈希；Git HEAD 不能替代含未提交改动的源码标识。"""
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob("*.py")) if "results" not in p.parts}
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        head = None
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"metric_version": METRIC_VERSION, "metric_dtype": "float64",
            "arguments": vars(args), "argv": sys.argv, "git_head_context": head,
            "source_sha256": sources, "python": platform.python_version(),
            "packages": packages, "torch_threads": torch.get_num_threads(),
            "cuda_version": torch.version.cuda, "job_id": os.getenv("SLURM_JOB_ID"),
            "node": platform.node(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
