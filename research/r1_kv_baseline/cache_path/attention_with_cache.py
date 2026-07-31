"""经 ContiguousKVCache / KiviKVCache 的 decode attention（写 cache → load → attend）。"""

from __future__ import annotations

import torch

from kv_cache import ContiguousKVCache, KiviKVCache
from kv_codecs import KVCodec, KiviFormat, get_codec


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """标准缩放点积 attention。

    约定形状均为 ``(batch, num_heads, seq, head_dim)``；
    decode 时 ``q`` 的 ``seq`` 通常为 1。
    """
    head_dim = q.shape[-1]
    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class AttentionWithCache:
    """持有 ContiguousKVCache（C0–C3）或 KiviKVCache（C4/C5），提供 prefill / decode。

    ``codec`` 可为：

    - ``KVCodec`` 实例，或 C0–C3 格式名（``fp16`` / ``int8`` / ``int4`` / ``int4_bdr``）
    - ``KiviFormat`` 实例，或 C4/C5 格式名（``kivi2`` / ``C4`` / ``kivi4`` / ``C5``）

    字符串路径会传入 ``dim=head_dim``（供 INT4+BDR）；其余关键字转发给 ``get_codec``
    （如 ``seed``、``group_size``、``residual_length``）。
    """

    def __init__(
        self,
        codec: KVCodec | KiviFormat | str,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        **codec_kwargs,
    ) -> None:
        self._kivi_format: KiviFormat | None = None
        self.num_heads = num_heads
        self.head_dim = head_dim

        if isinstance(codec, str):
            resolved = get_codec(codec, dim=head_dim, **codec_kwargs)
            if isinstance(resolved, KiviFormat):
                self._kivi_format = resolved
                self.cache: ContiguousKVCache | KiviKVCache = resolved.make_cache(
                    num_heads=num_heads, head_dim=head_dim, device=device
                )
            else:
                self.cache = ContiguousKVCache(
                    resolved, num_heads=num_heads, head_dim=head_dim, device=device
                )
        elif isinstance(codec, KiviFormat):
            if codec_kwargs:
                raise TypeError(
                    "传入 KiviFormat 时不要再给 codec 关键字参数；"
                    "请在 get_codec(...) 时设置，或改用格式字符串"
                )
            self._kivi_format = codec
            self.cache = codec.make_cache(
                num_heads=num_heads, head_dim=head_dim, device=device
            )
        elif isinstance(codec, KVCodec):
            if codec_kwargs:
                raise TypeError(
                    "传入 KVCodec 实例时不要再给 codec 关键字参数；"
                    "请在 get_codec(...) 时设置，或改用格式字符串"
                )
            self.cache = ContiguousKVCache(
                codec, num_heads=num_heads, head_dim=head_dim, device=device
            )
        else:
            raise TypeError(
                f"codec 须为 KVCodec / KiviFormat / str，得到 {type(codec).__name__}"
            )

        self.device = self.cache.device

    @property
    def codec(self) -> KVCodec | KiviFormat:
        """C0–C3 返回 ``KVCodec``；C4/C5 返回构造时的 ``KiviFormat``。"""
        if self._kivi_format is not None:
            return self._kivi_format
        assert isinstance(self.cache, ContiguousKVCache)
        return self.cache.codec

    def clear(self) -> None:
        """清空内部 KV cache。"""
        self.cache.clear()

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill：清空后写入整段 K/V，并对 ``q`` 做 attention。

        约定 ``q/k/v`` 形状 ``(batch, num_heads, seq, head_dim)``，
        且当前实现默认 ``batch=1``（与协议一致）。
        当前不做 causal mask（与 ``scaled_dot_product_attention`` 一致）。
        """
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                f"q/k/v 形状须一致，得到 q={q.shape}, k={k.shape}, v={v.shape}"
            )
        if q.ndim != 4 or q.shape[0] != 1:
            raise ValueError(f"期望 (1, num_heads, seq, head_dim)，得到 {q.shape}")
        if q.shape[1] != self.num_heads or q.shape[3] != self.head_dim:
            raise ValueError(
                f"期望 (1, {self.num_heads}, seq, {self.head_dim})，得到 {q.shape}"
            )
        if q.shape[2] <= 0:
            raise ValueError("seq 须为正")

        self.clear()
        q = q.to(device=self.device)
        # (1, H, S, D) → cache 布局 (S, H, D)
        k_cache = k.to(device=self.device).squeeze(0).transpose(0, 1)
        v_cache = v.to(device=self.device).squeeze(0).transpose(0, 1)
        self.cache.append(k_cache, v_cache)

        k_all, v_all = self.cache.load()
        k_attn = k_all.transpose(0, 1).unsqueeze(0)
        v_attn = v_all.transpose(0, 1).unsqueeze(0)
        return scaled_dot_product_attention(q, k_attn, v_attn)

    def decode_step(
        self,
        q: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """Decode 一步：将新 ``k_t/v_t`` 写入 cache，用 ``q`` 对全部 cache 做 attention。

        约定 ``q/k_t/v_t`` 形状 ``(batch, num_heads, 1, head_dim)``。
        """
        if q.shape != k_t.shape or q.shape != v_t.shape:
            raise ValueError(
                f"q/k_t/v_t 形状须一致，得到 q={q.shape}, k={k_t.shape}, v={v_t.shape}"
            )
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
            raise ValueError(f"期望 (1, num_heads, 1, head_dim)，得到 {q.shape}")
        if q.shape[1] != self.num_heads or q.shape[3] != self.head_dim:
            raise ValueError(
                f"期望 (1, {self.num_heads}, 1, {self.head_dim})，得到 {q.shape}"
            )

        q = q.to(device=self.device)
        # (1, H, 1, D) → cache 布局 (1, H, D)
        k_cache = k_t.to(device=self.device).squeeze(0).transpose(0, 1)
        v_cache = v_t.to(device=self.device).squeeze(0).transpose(0, 1)
        self.cache.append(k_cache, v_cache)

        k_all, v_all = self.cache.load()
        # (S, H, D) → (1, H, S, D)
        k_attn = k_all.transpose(0, 1).unsqueeze(0)
        v_attn = v_all.transpose(0, 1).unsqueeze(0)
        return scaled_dot_product_attention(q, k_attn, v_attn)

    def bytes_stored(self) -> tuple[int, int]:
        """转发 ``cache.bytes_stored()``。"""
        return self.cache.bytes_stored()
