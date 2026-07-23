"""逐周期 4x4 WS systolic 模型 + numpy golden（INT8 -> INT32）。"""

from __future__ import annotations

import numpy as np

N = 4
# 与 systolic_array/systolic_array.sv 一致（always_ff 在 NBA 前采样 bottom）
LAT = 2 * (N - 1) + 1


def gemm_int8(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """C = A @ W，INT8 输入、INT32 累加（逐位目标）。"""
    a = np.asarray(a, dtype=np.int8)
    w = np.asarray(w, dtype=np.int8)
    return (a.astype(np.int32) @ w.astype(np.int32)).astype(np.int32)


def systolic_ws_sim(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """模拟 WS 阵列：输入 skew + 输出 deskew（与 RTL 一致）。

    A: [M, N], W: [N, N], C: [M, N]
    PE[r][c] 驻留 W[r][c]；a 左→右流动；psum 上→下流动；全部寄存。
    """
    a = np.asarray(a, dtype=np.int8)
    w = np.asarray(w, dtype=np.int8)
    m_rows = a.shape[0]
    assert a.shape == (m_rows, N) and w.shape == (N, N)

    a_reg = np.zeros((N, N), dtype=np.int8)
    p_reg = np.zeros((N, N), dtype=np.int32)
    skew = [np.zeros(r + 1, dtype=np.int8) for r in range(N)]
    # hist[c][k] = 周期 k 采样的 bottom（NBA 前），类似 RTL hist 移位寄存器
    hist = [np.zeros(0, dtype=np.int32) for _ in range(N)]

    results: list[np.ndarray] = []
    total_cycles = m_rows + LAT

    for t in range(total_cycles):
        if t < m_rows:
            a_logic = a[t]
            valid_a = 1
        else:
            a_logic = np.zeros(N, dtype=np.int8)
            valid_a = 0
        _ = valid_a  # 发射时序与 RTL valid_a 流水线一致

        west = np.zeros(N, dtype=np.int8)
        for r in range(N):
            skew[r] = np.concatenate(([a_logic[r]], skew[r][:-1]))
            west[r] = skew[r][r]

        # RTL always_ff 看到更新前的 bottom（即上一周期 PE 输出）
        old_bottom = p_reg[N - 1, :].copy()

        a_in = np.zeros((N, N), dtype=np.int8)
        p_in = np.zeros((N, N), dtype=np.int32)
        for r in range(N):
            for c in range(N):
                a_in[r, c] = west[r] if c == 0 else a_reg[r, c - 1]
                p_in[r, c] = 0 if r == 0 else p_reg[r - 1, c]

        for r in range(N):
            for c in range(N):
                a_reg[r, c] = a_in[r, c]
                p_reg[r, c] = np.int32(p_in[r, c]) + np.int32(a_in[r, c]) * np.int32(w[r, c])

        # 用当前采样 + 移位前的 hist 输出（NBA 顺序）
        emit_t = t - LAT
        if emit_t >= 0 and emit_t < m_rows:
            crow = np.zeros(N, dtype=np.int32)
            for c in range(N):
                d = N - 1 - c
                if d == 0:
                    crow[c] = old_bottom[c]
                else:
                    # hist 当前含至 t-1 的采样；索引 [d-1] == 周期 t-d
                    crow[c] = int(hist[c][-(d)]) if len(hist[c]) >= d else 0
            results.append(crow)

        for c in range(N):
            hist[c] = np.concatenate([hist[c], old_bottom[c : c + 1]])

    return np.stack(results, axis=0)


def _self_test() -> None:
    rng = np.random.default_rng(0)
    for m in (1, 4, 7, 8):
        a = rng.integers(-128, 127, size=(m, N), dtype=np.int8)
        w = rng.integers(-128, 127, size=(N, N), dtype=np.int8)
        if m == 1:
            a[0, :] = np.array([-128, -1, 0, 127], dtype=np.int8)
            w[:, :] = np.eye(N, dtype=np.int8) * 2
        ref = gemm_int8(a, w)
        got = systolic_ws_sim(a, w)
        assert got.shape == ref.shape, (got.shape, ref.shape)
        assert np.array_equal(got, ref), (
            m,
            int(np.max(np.abs(got.astype(np.int64) - ref.astype(np.int64)))),
            got,
            ref,
        )
    print("systolic_rtl_model self-test PASSED")


if __name__ == "__main__":
    _self_test()
