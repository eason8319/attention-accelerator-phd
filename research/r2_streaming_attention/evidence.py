"""R2 实验共用的结果登记：文件哈希、run_config 与非空结果目录保护。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PACKAGE_NAMES = (
    "torch",
    "numpy",
    "transformers",
    "datasets",
    "triton",
    "accelerate",
    "huggingface-hub",
    "tokenizers",
    "flash-attn",
    "ninja",
    "rouge",
    "fuzzywuzzy",
    "python-Levenshtein",
    "PyYAML",
)


def sha256(path: Path | str) -> str:
    """返回文件内容的 SHA-256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path | str, value: object) -> None:
    """以 UTF-8 JSON 写入新文件；已存在则失败，避免覆盖有效结果。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def start_output(
    experiment_dir: Path | str,
    relative: str | Path,
    parameters: dict,
    sources: list[Path | str],
) -> Path:
    """在实验 ``results/`` 下创建空批次目录并写入 ``run_config.json``。

    参数
        experiment_dir: 实验根目录，须含 ``experiment.json`` 与 ``results/``
        relative: 相对实验根的输出路径，必须落在 ``results/`` 内
        parameters: 本批实际参数
        sources: 本批实验源码或数据文件；本模块会自动记入登记助手自身
    返回
        已创建的输出目录
    异常
        FileExistsError: 目标目录已存在且非空
        ValueError: 输出路径离开 ``results/``，或 ``experiment.json`` 缺字段
    """
    experiment_dir = Path(experiment_dir).resolve()
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    try:
        experiment_id = manifest["experiment_id"]
        evidence_scope = manifest["evidence_scope"]
    except KeyError as exc:
        raise ValueError("experiment.json 缺少 experiment_id 或 evidence_scope") from exc
    out = (experiment_dir / relative).resolve()
    out.relative_to(experiment_dir / "results")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty result directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    recorded = []
    seen: set[Path] = set()
    for item in [*sources, Path(__file__)]:
        path = Path(item).resolve()
        if path in seen:
            continue
        seen.add(path)
        recorded.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
        )
    save(
        out / "run_config.json",
        {
            "experiment": experiment_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "parameters": parameters,
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "package_versions": versions,
            "source_files": recorded,
            "evidence_scope": evidence_scope,
        },
    )
    return out
