"""流式 Attention：单头操作数供给、online softmax、跨段归并与 C3 Q/O 变换。

默认路径不解出完整高精度 KV tile。C5 残差与均匀尾页的 FP16 按已存储片段进入
工作缓冲，单独计量。本模块不做周期或能耗结论。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

_BYTES = Path(__file__).resolve().parents[1] / "r1_kv_baseline" / "bytes_accounting"
if str(_BYTES) not in sys.path:
    sys.path.insert(0, str(_BYTES))
from tensor_metrics import tensor_error

from research.r2_streaming_attention.paged_kv import PackedPhysicalCache
from research.r2_streaming_attention.streaming_decode import PackedOperandReader


def attention_scale(head_dim: int) -> float:
    """缩放点积的 :math:`1/\\sqrt{d}`。"""
    return float(head_dim) ** -0.5


def qk_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float,
    gqa_ratio: int,
    min_tokens: int = 1,
) -> torch.Tensor:
    """计算 ``(H_q, T)`` 分数；GQA 时 ``H_q = H_kv * gqa_ratio``。"""
    if q.ndim != 2 or k.ndim != 3:
        raise ValueError(f"期望 q=(H_q,D)、k=(T,H_kv,D)，得到 {tuple(q.shape)} / {tuple(k.shape)}")
    n_q, dim = q.shape
    tokens, n_kv, k_dim = k.shape
    if dim != k_dim:
        raise ValueError(f"Q/K head_dim 不一致: {dim} vs {k_dim}")
    if gqa_ratio <= 0 or n_q != n_kv * gqa_ratio:
        raise ValueError(
            f"GQA 要求 H_q={n_kv * gqa_ratio}，得到 H_q={n_q}, H_kv={n_kv}, ratio={gqa_ratio}"
        )
    q_g = q.reshape(n_kv, gqa_ratio, dim)
    if tokens < min_tokens:
        padded = k.new_zeros(min_tokens, n_kv, dim)
        padded[:tokens] = k
        k = padded
    # 只返回有效列；补零列不进入 max、softmax 或 PV。
    return torch.einsum("hgd,thd->hgt", q_g, k).reshape(n_q, -1)[:, :tokens] * scale


def pv_weighted(
    weights: torch.Tensor,
    v: torch.Tensor,
    *,
    gqa_ratio: int,
) -> torch.Tensor:
    """``(H_q, T)`` 权重与 ``(T, H_kv, D)`` 的 V 相乘，返回 ``(H_q, D)``。"""
    n_q, tokens = weights.shape
    t2, n_kv, dim = v.shape
    if t2 != tokens:
        raise ValueError(f"权重 T={tokens} 与 V T={t2} 不一致")
    if n_q != n_kv * gqa_ratio:
        raise ValueError("GQA 头数与 V 不一致")
    p_g = weights.reshape(n_kv, gqa_ratio, tokens)
    out = torch.einsum("hgt,thd->hgd", p_g, v)
    return out.reshape(n_q, dim)


@dataclass
class OnlineState:
    """一行 decode query 的 running max / 归一化量 / 部分输出。"""

    m: torch.Tensor
    l: torch.Tensor
    o: torch.Tensor

    @classmethod
    def empty(cls, n_q: int, head_dim: int, device: torch.device) -> OnlineState:
        """全掩码初值：max 为 -inf，累加器为 0。"""
        return cls(
            m=torch.full((n_q,), float("-inf"), device=device, dtype=torch.float32),
            l=torch.zeros(n_q, device=device, dtype=torch.float32),
            o=torch.zeros(n_q, head_dim, device=device, dtype=torch.float32),
        )

    @property
    def is_empty(self) -> bool:
        """没有任何有效分数被吸收。"""
        return bool((~torch.isfinite(self.m)).all() or (self.l == 0).all())

    def absorb(self, scores: torch.Tensor, v: torch.Tensor, *, gqa_ratio: int) -> None:
        """并入一块 ``(H_q, T)`` 分数；全为 -inf 的块不参与归并。"""
        if not bool(torch.isfinite(v).all()):
            raise RuntimeError("V 包含非有限值")
        weights = self.begin_block(scores)
        self.o += pv_weighted(weights, v, gqa_ratio=gqa_ratio)

    def begin_block(self, scores: torch.Tensor) -> torch.Tensor:
        """更新在线最大值与归一化量，返回供分块 PV 使用的权重。"""
        if bool((torch.isnan(scores) | torch.isposinf(scores)).any()):
            raise RuntimeError("有效分数包含 NaN/+Inf")
        if scores.numel() == 0:
            return torch.zeros_like(scores)
        finite = torch.isfinite(scores)
        if not bool(finite.any()):
            return torch.zeros_like(scores)
        block_max = scores.masked_fill(~finite, float("-inf")).amax(dim=-1)
        m_new = torch.maximum(self.m, block_max)
        alpha = torch.exp(self.m - m_new)
        alpha = torch.where(torch.isfinite(self.m), alpha, torch.zeros_like(alpha))
        shifted = scores - m_new.unsqueeze(-1)
        p_blk = torch.exp(shifted)
        p_blk = torch.where(finite, p_blk, torch.zeros_like(p_blk))
        self.l = alpha * self.l + p_blk.sum(dim=-1)
        self.o = alpha.unsqueeze(-1) * self.o
        self.m = m_new
        return p_blk

    def finalize(self) -> torch.Tensor:
        """最终归一化；全掩码输出定义为 0。"""
        if bool((torch.isnan(self.m) | torch.isposinf(self.m)).any()) or not bool(
            torch.isfinite(self.l).all() and torch.isfinite(self.o).all()
        ):
            raise RuntimeError("在线状态含非有限值")
        out = torch.zeros_like(self.o)
        valid = torch.isfinite(self.m) & (self.l > 0)
        if bool(valid.any()):
            out[valid] = self.o[valid] / self.l[valid].unsqueeze(-1)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("Attention 输出含非有限值")
        return out


def merge_states(states: list[OnlineState]) -> OnlineState:
    """Flash-Decoding 风格按 log-sum-exp 归并独立分段；空段不参与。"""
    if not states:
        raise ValueError("至少需要一个分段状态")
    for item in states:
        item.finalize()
    alive = [item for item in states if not item.is_empty]
    if not alive:
        sample = states[0]
        return OnlineState.empty(int(sample.m.shape[0]), int(sample.o.shape[-1]), sample.m.device)
    acc = OnlineState(
        m=alive[0].m.clone(),
        l=alive[0].l.clone(),
        o=alive[0].o.clone(),
    )
    for extra in alive[1:]:
        m_new = torch.maximum(acc.m, extra.m)
        a_acc = torch.exp(acc.m - m_new)
        a_ext = torch.exp(extra.m - m_new)
        a_acc = torch.where(torch.isfinite(acc.m), a_acc, torch.zeros_like(a_acc))
        a_ext = torch.where(torch.isfinite(extra.m), a_ext, torch.zeros_like(a_ext))
        acc.l = a_acc * acc.l + a_ext * extra.l
        acc.o = a_acc.unsqueeze(-1) * acc.o + a_ext.unsqueeze(-1) * extra.o
        acc.m = m_new
    return acc


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    gqa_ratio: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """同 codec 展开参考：一次 softmax，仅供对拍。"""
    if not bool(torch.isfinite(q).all() and torch.isfinite(k).all() and torch.isfinite(v).all()):
        raise RuntimeError("参考输入含非有限值")
    scale = attention_scale(int(q.shape[-1]))
    scores = qk_scores(q, k, scale=scale, gqa_ratio=gqa_ratio)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("参考掩码前分数含非有限值")
    if valid_mask is not None:
        if valid_mask.shape != (k.shape[0],):
            raise ValueError("valid_mask 须为 (T,)")
        scores = scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
        if not bool(valid_mask.any()):
            return torch.zeros_like(q)
    if not bool(torch.isfinite(scores).any()):
        return torch.zeros_like(q)
    weights = torch.softmax(scores, dim=-1)
    out = pv_weighted(weights, v, gqa_ratio=gqa_ratio)
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("参考 Attention 输出含非有限值")
    return out


@dataclass
class StreamStats:
    """有界工作缓冲与背压计数；不解出完整序列的证据。"""

    tile_tokens: int
    seq_len: int
    max_live_tokens: int = 0
    max_scratch_tokens: int = 0
    n_tiles: int = 0
    n_segments: int = 0
    n_backpressure: int = 0
    load_calls: int = 0
    tail_rotate_tokens: int = 0
    max_operand_bytes: int = 0
    operand_read_calls: int = 0
    high_precision_kv_tile_materialized: bool = False
    c3_guard_checked: bool = False
    c3_qo_guard_passed: bool | None = None
    c3_original_fallback: bool = False
    c3_guard_max_abs: float = 0.0
    c3_guard_max_rel_l2: float | None = None
    c3_guard_extra_read_calls: int = 0
    c3_guard_extra_tiles: int = 0
    qk_domain: str = "original"
    c3_key_inverse_tokens: int = 0
    c3_inverse_padded_token_heads: int = 0
    c3_inverse_padding_max_bytes: int = 0
    qk_padded_token_heads: int = 0
    qk_padding_max_bytes: int = 0

    def observe_tile(self, assembled: int, scratch: int) -> None:
        """记录一块工作缓冲的存活规模。"""
        self.n_tiles += 1
        self.max_live_tokens = max(self.max_live_tokens, assembled)
        self.max_scratch_tokens = max(self.max_scratch_tokens, scratch)

    def bounded(self, scratch_limit: int | None = None) -> bool:
        """工作缓冲不超过 tile；scratch 默认允许对齐到 group=32。"""
        limit = self.tile_tokens if scratch_limit is None else scratch_limit
        if scratch_limit is None:
            limit = max(self.tile_tokens, 32)
        if self.max_live_tokens > self.tile_tokens:
            return False
        if self.max_scratch_tokens > limit:
            return False
        if self.seq_len > self.tile_tokens and self.max_live_tokens >= self.seq_len:
            return False
        if self.load_calls != 0:
            return False
        if self.high_precision_kv_tile_materialized:
            return False
        return True

    def as_dict(self) -> dict[str, object]:
        """可 JSON 化的计数。"""
        return {
            "tile_tokens": self.tile_tokens,
            "seq_len": self.seq_len,
            "max_live_tokens": self.max_live_tokens,
            "max_scratch_tokens": self.max_scratch_tokens,
            "n_tiles": self.n_tiles,
            "n_segments": self.n_segments,
            "n_backpressure": self.n_backpressure,
            "load_calls": self.load_calls,
            "tail_rotate_tokens": self.tail_rotate_tokens,
            "max_operand_bytes": self.max_operand_bytes,
            "operand_read_calls": self.operand_read_calls,
            "high_precision_kv_tile_materialized": self.high_precision_kv_tile_materialized,
            "c3_guard_checked": self.c3_guard_checked,
            "c3_qo_guard_passed": self.c3_qo_guard_passed,
            "c3_original_fallback": self.c3_original_fallback,
            "c3_guard_max_abs": self.c3_guard_max_abs,
            "c3_guard_max_rel_l2": self.c3_guard_max_rel_l2,
            "c3_guard_extra_read_calls": self.c3_guard_extra_read_calls,
            "c3_guard_extra_tiles": self.c3_guard_extra_tiles,
            "qk_domain": self.qk_domain,
            "c3_key_inverse_tokens": self.c3_key_inverse_tokens,
            "c3_inverse_padded_token_heads": self.c3_inverse_padded_token_heads,
            "c3_inverse_padding_max_bytes": self.c3_inverse_padding_max_bytes,
            "qk_padded_token_heads": self.qk_padded_token_heads,
            "qk_padding_max_bytes": self.qk_padding_max_bytes,
            "bounded": self.bounded(),
        }


@dataclass
class BoundedTileBuffer:
    """容量为 ``capacity`` 的 tile 槽；满则记背压后再等待释放。"""

    capacity: int
    live: int = 0
    stalls: int = 0
    events: list[dict[str, int]] = field(default_factory=list)

    def acquire(self, tokens: int) -> None:
        """占用一个工作槽；已满时先记背压。"""
        if self.capacity <= 0:
            raise ValueError("tile 槽容量须为正")
        if self.live >= self.capacity:
            self.stalls += 1
            self.events.append({"tokens": tokens, "live": self.live})
        if self.live >= self.capacity:
            raise RuntimeError("有界缓冲在未释放时不能同时持有下一块 tile")
        self.live += 1

    def release(self) -> None:
        """释放当前 tile 的工作缓冲。"""
        if self.live <= 0:
            raise RuntimeError("没有可释放的 tile")
        self.live -= 1


def prepare_query(q: torch.Tensor, cache: PackedPhysicalCache, domain: str) -> torch.Tensor:
    """QK 坐标域为旋转域时才旋 Q；混合基线的 QK 保持原域。"""
    q = q.float()
    if cache.format_id == "C3" and domain == "rotated":
        return cache._codec._rot.rotate(q, axis=-1)
    return q


def finish_output(o: torch.Tensor, cache: PackedPhysicalCache, domain: str) -> torch.Tensor:
    """C3 旋转输出路径对部分输出做逆旋转；混合基线与纯 Q/O 共用该步。"""
    if cache.format_id == "C3" and domain == "rotated":
        return cache._codec._rot.inverse(o, axis=-1)
    return o


def stream_decode_attention(
    cache: PackedPhysicalCache,
    q: torch.Tensor,
    *,
    gqa_ratio: int = 1,
    tile_tokens: int = 128,
    merge_tokens: int | None = None,
    group_buffers: int = 1,
    valid_mask: torch.Tensor | None = None,
    domain: str = "original",
    record_events: bool = True,
    force_backpressure: bool = False,
    c3_numerical_guard: bool = False,
    c3_qk_domain: str | None = None,
) -> tuple[torch.Tensor, StreamStats]:
    """双遍单头供给。C3 旋转输出默认原域 QK、旋转域 PV/O，保留读侧 K 逆旋转。"""
    if q.ndim != 2 or q.shape != (cache.num_heads * gqa_ratio, cache.head_dim):
        raise ValueError("Q 的形状与缓存/GQA 不一致")
    if tile_tokens <= 0 or group_buffers <= 0 or gqa_ratio <= 0:
        raise ValueError("tile、缓冲槽和 GQA 比例必须为正")
    if merge_tokens is not None and merge_tokens <= 0:
        raise ValueError("merge_tokens 须为正")
    if domain not in {"original", "rotated"} or (domain == "rotated" and cache.format_id != "C3"):
        raise ValueError("无效的缓存坐标域")
    if c3_qk_domain is not None and (
        cache.format_id != "C3"
        or domain != "rotated"
        or c3_qk_domain not in {"original", "rotated"}
    ):
        raise ValueError("独立 QK 坐标域仅用于 C3 旋转输出路径")
    if c3_qk_domain is None and cache.format_id == "C3" and domain == "rotated":
        # 功能基线为混合路径；纯 Q/O 须显式 c3_qk_domain="rotated"。
        qk_domain = "original"
    else:
        qk_domain = domain if c3_qk_domain is None else c3_qk_domain
    if not bool(torch.isfinite(q).all()):
        raise RuntimeError("Q 含非有限值")
    seq_len = len(cache)
    if valid_mask is not None and (
        valid_mask.shape != (seq_len,) or valid_mask.dtype != torch.bool
    ):
        raise ValueError("valid_mask 须为布尔 (seq_len,)")
    stats = StreamStats(tile_tokens=tile_tokens, seq_len=seq_len, qk_domain=qk_domain)
    if not seq_len:
        return torch.zeros_like(q.float()), stats
    q_work = prepare_query(q, cache, qk_domain)
    if not bool(torch.isfinite(q_work).all()):
        raise RuntimeError("变换后的 Q 含非有限值")
    # 混合基线为短 K 逆旋转和短 QK 使用四行/列 FP32 矩阵计算。
    # 纯 Q/O 维持自身求值；本选择不依赖分数、误差或随机种子。
    c3_hybrid = cache.format_id == "C3" and domain == "rotated" and qk_domain == "original"
    min_rows = 4 if c3_hybrid else 1
    reader = PackedOperandReader(cache, c3_inverse_min_rows=min_rows)
    reader.record_events = record_events
    buffer = BoundedTileBuffer(group_buffers)
    scale = attention_scale(cache.head_dim)
    segment_size = seq_len if merge_tokens is None else merge_tokens
    merged = OnlineState.empty(len(q), cache.head_dim, q.device)
    for seg_start in range(0, seq_len, segment_size):
        seg_end = min(seg_start + segment_size, seq_len)
        current = OnlineState.empty(len(q), cache.head_dim, q.device)
        for start in range(seg_start, seg_end, tile_tokens):
            end = min(start + tile_tokens, seg_end)
            if force_backpressure and stats.n_tiles:
                buffer.acquire(1)
                try:
                    for _ in range(group_buffers):
                        buffer.acquire(1)
                except RuntimeError:
                    stats.n_backpressure += 1
                    if record_events:
                        cache.events.record(
                            action="backpressure",
                            field="operand_slot",
                            side="kv",
                            pool="work",
                            level="control",
                            tokens=end - start,
                            logical_bytes=0,
                            seq_len=seq_len,
                        )
                while buffer.live:
                    buffer.release()
            scores = torch.empty(len(q), end - start, dtype=torch.float32, device=q.device)
            for head in range(cache.num_heads):
                h0, h1 = head * gqa_ratio, (head + 1) * gqa_ratio
                for a in range(start, end, reader.operand_tokens):
                    b = min(a + reader.operand_tokens, end)
                    buffer.acquire(b - a)
                    operand = reader.read("k", a, b, head, domain=qk_domain)
                    score = qk_scores(
                        q_work[h0:h1],
                        operand.unsqueeze(1),
                        scale=scale,
                        gqa_ratio=gqa_ratio,
                        min_tokens=min_rows,
                    )
                    if b - a < min_rows:
                        stats.qk_padded_token_heads += min_rows - (b - a)
                        stats.qk_padding_max_bytes = min_rows * cache.head_dim * 4
                    if not bool(torch.isfinite(score).all()):
                        raise RuntimeError("掩码前有效 QK 分数含非有限值")
                    scores[h0:h1, a - start : b - start] = score
                    del operand, score
                    buffer.release()
            if valid_mask is not None:
                scores = scores.masked_fill(~valid_mask[start:end].unsqueeze(0), float("-inf"))
            weights = current.begin_block(scores)
            del scores
            for head in range(cache.num_heads):
                h0, h1 = head * gqa_ratio, (head + 1) * gqa_ratio
                for a in range(start, end, reader.operand_tokens):
                    b = min(a + reader.operand_tokens, end)
                    buffer.acquire(b - a)
                    operand = reader.read("v", a, b, head, domain=domain)
                    current.o[h0:h1] += pv_weighted(
                        weights[h0:h1, a - start : b - start],
                        operand.unsqueeze(1),
                        gqa_ratio=gqa_ratio,
                    )
                    del operand
                    buffer.release()
            del weights
            stats.n_tiles += 1
            if record_events:
                cache.events.record(
                    action="attention_mac",
                    field="qk_pv",
                    side="kv",
                    pool="work",
                    level="control",
                    tokens=end - start,
                    logical_bytes=0,
                    seq_len=seq_len,
                )
        merged = merge_states([merged, current])
        stats.n_segments += 1
        if record_events and merge_tokens is not None:
            cache.events.record(
                action="output_merge",
                field="partial_o",
                side="o",
                pool="work",
                level="control",
                tokens=seg_end - seg_start,
                logical_bytes=0,
                seq_len=seq_len,
            )
    stats.max_live_tokens = reader.max_operand_tokens
    stats.max_scratch_tokens = reader.max_operand_tokens
    stats.max_operand_bytes = reader.max_operand_bytes
    stats.operand_read_calls = reader.read_calls
    stats.c3_inverse_padded_token_heads = reader.inverse_padded_token_heads
    stats.c3_inverse_padding_max_bytes = reader.inverse_padding_max_bytes
    if reader.inverse_padded_token_heads or stats.qk_padded_token_heads:
        stats.max_scratch_tokens = max(stats.max_scratch_tokens, min_rows)
    if cache.format_id == "C3" and domain == "rotated":
        stats.tail_rotate_tokens = cache.v_residual_len
        if qk_domain == "rotated":
            stats.tail_rotate_tokens += cache.k_residual_len
    if cache.format_id == "C3" and qk_domain == "original":
        stats.c3_key_inverse_tokens = seq_len - cache.k_residual_len
    out = finish_output(merged.finalize(), cache, domain)
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("变换后的输出含非有限值")
    if cache.format_id == "C3" and domain == "rotated" and c3_numerical_guard:
        # 保护结果的可用性；纯 Q/O 失败与每次额外原域计算均明确记账。
        original, guard_stats = stream_decode_attention(
            cache,
            q,
            gqa_ratio=gqa_ratio,
            tile_tokens=tile_tokens,
            merge_tokens=merge_tokens,
            group_buffers=group_buffers,
            valid_mask=valid_mask,
            domain="original",
            record_events=record_events,
            c3_numerical_guard=False,
        )
        guard = evaluate_head_errors(
            out,
            original,
            atol=1e-5,
            rtol=1e-4,
            rel_l2_max=0.01,
            near_zero_rms=1e-6,
            near_zero_abs=1e-5,
        )
        stats.c3_guard_checked = True
        stats.c3_qo_guard_passed = bool(guard["passed"])
        stats.c3_guard_max_abs = float(guard["max_abs"])
        stats.c3_guard_max_rel_l2 = guard["max_rel_l2"]
        stats.c3_guard_extra_read_calls = guard_stats.operand_read_calls
        stats.c3_guard_extra_tiles = guard_stats.n_tiles
        stats.operand_read_calls += guard_stats.operand_read_calls
        stats.max_operand_bytes = max(stats.max_operand_bytes, guard_stats.max_operand_bytes)
        if not guard["passed"]:
            out = original
            stats.c3_original_fallback = True
    return out, stats


def expand_storage_reference(
    cache: PackedPhysicalCache,
    *,
    domain: str = "original",
) -> tuple[torch.Tensor, torch.Tensor]:
    """把当前存储一次解成完整 K/V，仅作参考，不得用于 DUT。"""
    k, v, _ = cache.decode_token_range(0, len(cache), domain=domain, record_events=False)
    return k, v


def evaluate_head_errors(
    pred: torch.Tensor,
    gold: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    rel_l2_max: float,
    near_zero_rms: float,
    near_zero_abs: float,
) -> dict[str, object]:
    """按 query head 报告相对 L2，并用协议功能阈值判定。"""
    if pred.shape != gold.shape:
        return {"passed": False, "reason": "shape", "max_rel_l2": None}
    if not bool(torch.isfinite(pred).all() and torch.isfinite(gold).all()):
        return {"passed": False, "reason": "nonfinite", "max_rel_l2": None}
    close = bool(torch.allclose(pred, gold, atol=atol, rtol=rtol, equal_nan=False))
    max_rel: float | None = None
    max_abs = 0.0
    heads: list[dict[str, float | None | bool]] = []
    passed_heads = True
    for index in range(pred.shape[0]):
        err = tensor_error(pred[index], gold[index])
        rms = (
            float(gold[index].double().pow(2).mean().sqrt().item()) if gold[index].numel() else 0.0
        )
        if err.rel_l2 is None:
            head_ok = err.max_abs <= near_zero_abs
            rel = None
        else:
            head_ok = err.rel_l2 <= rel_l2_max
            rel = err.rel_l2
            max_rel = max(max_rel or 0.0, rel)
            if rms <= near_zero_rms:
                head_ok = head_ok and err.max_abs <= near_zero_abs
        passed_heads = passed_heads and head_ok
        max_abs = max(max_abs, err.max_abs)
        heads.append(
            {
                "head": index,
                "rel_l2": rel,
                "max_abs": err.max_abs,
                "passed": head_ok,
            }
        )
    return {
        "passed": close and passed_heads,
        "allclose": close,
        "heads_passed": passed_heads,
        "max_rel_l2": max_rel,
        "max_abs": max_abs,
        "n_heads": pred.shape[0],
        "failed_heads": [row["head"] for row in heads if not row["passed"]],
    }
