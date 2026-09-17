"""只读接入 R1 ``cache_path``：加入导入路径并核对接力协议登记的源码哈希。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
R1_CACHE = ROOT / "research" / "r1_kv_baseline" / "cache_path"
PROTOCOL = (
    ROOT
    / "research"
    / "r2_streaming_attention"
    / "experiments"
    / "configs"
    / "evaluation_protocol.json"
)
PAGE_ACCESS = (
    ROOT / "research" / "r2_streaming_attention" / "experiments" / "configs" / "page_access.json"
)


def sha256_file(path: Path | str) -> str:
    """返回文件内容的 SHA-256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_r1_semantic_hashes(protocol_path: Path | str | None = None) -> list[dict[str, str]]:
    """核对协议 ``formats.semantic_sources`` 与当前 R1 文件字节；不一致则失败。

    参数
        protocol_path: 协议 JSON；默认本阶段共享配置
    返回
        每条源的相对路径、期望哈希与实测哈希
    """
    proto = json.loads((protocol_path or PROTOCOL).read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for item in proto["formats"]["semantic_sources"]:
        rel = item["path"]
        expected = item["sha256"]
        path = ROOT / rel
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"R1 语义源码哈希与协议不一致: {rel} 期望 {expected}，实际 {actual}")
        rows.append({"path": rel, "sha256": actual, "matched_protocol": "true"})
    return rows


def verify_listed_hashes(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """核对待查文件哈希；不一致则失败。

    参数
        items: 含 ``path`` 与 ``sha256`` 的登记项
    返回
        每条源的相对路径、期望哈希与实测哈希
    """
    rows: list[dict[str, str]] = []
    for item in items:
        rel = item["path"]
        expected = item["sha256"]
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise ValueError(f"源码哈希不一致: {rel} 期望 {expected}，实际 {actual}")
        rows.append({"path": rel, "sha256": actual, "matched": "true"})
    return rows


def verify_r1_page_layout_hashes(
    page_access_path: Path | str | None = None,
) -> list[dict[str, str]]:
    """核对分页配置登记的 R1 页表/残差实现哈希。"""
    raw = json.loads((page_access_path or PAGE_ACCESS).read_text(encoding="utf-8"))
    return verify_listed_hashes(raw["r1_layout_sources"])


def import_r1_codecs() -> ModuleType:
    """导入 R1 ``kv_codecs``；不修改其源码，也不把 R1 缓存对象当作物理打包实现。"""
    cache_dir = str(R1_CACHE)
    if cache_dir not in sys.path:
        sys.path.insert(0, cache_dir)
    import kv_codecs

    return kv_codecs


def import_r1_kv_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """导入 R1 codec、连续缓存与分页缓存；仅作语义参照。"""
    codecs = import_r1_codecs()
    import kv_cache
    import paged_cache

    return codecs, kv_cache, paged_cache
