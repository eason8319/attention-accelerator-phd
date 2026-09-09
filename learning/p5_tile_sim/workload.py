"""P5 tile 级仿真器的 attention 工作负载描述。

形状遵循 P3 中 LLaMA-7B 规模单层 attention：
``hidden=4096``、``heads=32``、``head_dim=128``。Prefill 使用全长 query；
decode 用单 query token 对长度为 ``seq_len`` 的 KV cache。

各张量字节宽（``ElementBytes``）为混合精度扩展点：
默认 INT8（1 byte）；后续 sweep 可设如 Q/K INT4 + V FP8。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

Mode = Literal["prefill", "decode"]

DEFAULT_SEQ_LENS = (4_096, 32_768, 131_072)


@dataclass(frozen=True)
class ElementBytes:
    """Q / K / V / O 的每元素字节数，作为混合精度配置接口。"""

    q: float = 1.0
    k: float = 1.0
    v: float = 1.0
    o: float = 1.0

    def __post_init__(self) -> None:
        for name, val in (
            ("q", self.q),
            ("k", self.k),
            ("v", self.v),
            ("o", self.o),
        ):
            if val <= 0:
                raise ValueError(f"bytes.{name} must be positive, got {val}")


@dataclass(frozen=True)
class Workload:
    """一次 attention 层调用的负载，覆盖单批次的所有头。"""

    mode: Mode
    seq_len: int
    batch: int = 1
    heads: int = 32
    head_dim: int = 128
    hidden: int = 4_096
    bytes: ElementBytes = ElementBytes()

    def __post_init__(self) -> None:
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"mode must be 'prefill' or 'decode', got {self.mode!r}")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.batch <= 0 or self.heads <= 0 or self.head_dim <= 0:
            raise ValueError("batch, heads, and head_dim must be positive")
        if self.hidden != self.heads * self.head_dim:
            raise ValueError(
                "hidden must equal heads * head_dim, got "
                f"{self.hidden} != {self.heads} * {self.head_dim}"
            )

    @property
    def n_q(self) -> int:
        """查询序列长度；单步 decode 时为 1。"""
        return 1 if self.mode == "decode" else self.seq_len

    @property
    def n_kv(self) -> int:
        """KV 缓存中的键值序列长度。"""
        return self.seq_len

    def with_bytes(
        self,
        *,
        q: float | None = None,
        k: float | None = None,
        v: float | None = None,
        o: float | None = None,
    ) -> Workload:
        """返回副本，并更新指定张量的每元素字节数。"""
        b = self.bytes
        return replace(
            self,
            bytes=ElementBytes(
                q=b.q if q is None else q,
                k=b.k if k is None else k,
                v=b.v if v is None else v,
                o=b.o if o is None else o,
            ),
        )

    def tile_footprint_bytes(self, br: int, bc: int) -> float:
        """单头 FA 分块的单缓冲 SRAM 占用，参见阅读笔记 §1.2。

        字节数为 Br×d×bQ + Bc×d×(bK+bV) + Br×d×bO。
        """
        if br <= 0 or bc <= 0:
            raise ValueError("br and bc must be positive")
        d = self.head_dim
        b = self.bytes
        return br * d * b.q + bc * d * (b.k + b.v) + br * d * b.o

    def num_q_tiles(self, br: int) -> int:
        if br <= 0:
            raise ValueError("br must be positive")
        return int(math.ceil(self.n_q / br))

    def num_kv_tiles(self, bc: int) -> int:
        if bc <= 0:
            raise ValueError("bc must be positive")
        return int(math.ceil(self.n_kv / bc))


def llama7b_attention(
    mode: Mode,
    seq_len: int,
    *,
    bytes: ElementBytes | None = None,
) -> Workload:
    """构造与 P3 LLaMA-7B 规模一致的单层 attention 负载。"""
    return Workload(
        mode=mode,
        seq_len=seq_len,
        bytes=ElementBytes() if bytes is None else bytes,
    )
