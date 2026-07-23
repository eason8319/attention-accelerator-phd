"""P4 RTL 共用定点工具：Q 格式量化/反量化、饱和、误差度量、向量文件 I/O。

约定（与 scripts/README.md 一致）：
- Q(I,F) 表示有符号定点：1 位符号含在 I 内，总位宽 W = I + F，LSB = 2^-F。
  例：Q8.8 -> W=16, 范围 [-128, 128 - 2^-8]。
- UQ(I,F) 表示无符号定点：总位宽 W = I + F，范围 [0, 2^I - 2^-F]。
- 向量文件：一行一个样本，有符号数写十进制补码整数（即 raw int），
  便于 SystemVerilog 侧 $fscanf("%d") 直接读入定点寄存器。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QFormat:
    """定点格式描述。int_bits 含符号位（signed=True 时）。"""

    int_bits: int
    frac_bits: int
    signed: bool = True

    @property
    def width(self) -> int:
        return self.int_bits + self.frac_bits

    @property
    def scale(self) -> float:
        return float(1 << self.frac_bits)

    @property
    def min_int(self) -> int:
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max_int(self) -> int:
        return (1 << (self.width - 1)) - 1 if self.signed else (1 << self.width) - 1

    @property
    def min_val(self) -> float:
        return self.min_int / self.scale

    @property
    def max_val(self) -> float:
        return self.max_int / self.scale

    def __str__(self) -> str:
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.int_bits}.{self.frac_bits}"


# P4 默认格式（模块 A 扫参见 notes/exp_approx_sweep.md）
Q8_8 = QFormat(8, 8)  # 对照用；选定输入为 Q6.10
Q6_10 = QFormat(6, 10)  # exp 输入 x = s - m <= 0（选定）
UQ0_24 = QFormat(0, 24, signed=False)  # exp 输出 (0, 1]（选定）
UQ2_14 = QFormat(2, 14, signed=False)  # 早期默认，扫参后不再用于 exp
Q16_16 = QFormat(16, 16)  # softmax 的 l 等宽累加


def float_to_fixed(x: np.ndarray | float, fmt: QFormat) -> np.ndarray:
    """浮点 -> 定点 raw int（round-to-nearest + 饱和）。"""
    arr = np.asarray(x, dtype=np.float64)
    q = np.rint(arr * fmt.scale)
    q = np.clip(q, fmt.min_int, fmt.max_int)
    return q.astype(np.int64)


def fixed_to_float(q: np.ndarray | int, fmt: QFormat) -> np.ndarray:
    """定点 raw int -> 浮点。"""
    return np.asarray(q, dtype=np.int64).astype(np.float64) / fmt.scale


def quantize(x: np.ndarray | float, fmt: QFormat) -> np.ndarray:
    """quantize-dequantize：返回可与 golden 直接比较的浮点值。"""
    return fixed_to_float(float_to_fixed(x, fmt), fmt)


def max_abs_err(ref: np.ndarray, dut: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(ref, dtype=np.float64) - np.asarray(dut, dtype=np.float64)))
    )


def rel_err(ref: np.ndarray, dut: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """逐元素相对误差 |ref-dut| / max(|ref|, eps)。"""
    ref = np.asarray(ref, dtype=np.float64)
    dut = np.asarray(dut, dtype=np.float64)
    return np.abs(ref - dut) / np.maximum(np.abs(ref), eps)


# ---------------------------------------------------------------- 向量文件 I/O


def write_vec(path: str | Path, raw: np.ndarray) -> None:
    """写向量文件：一行一个十进制补码整数（配 SV 侧 $fscanf("%d")）。"""
    raw = np.asarray(raw, dtype=np.int64).reshape(-1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, raw, fmt="%d")


def read_vec(path: str | Path) -> np.ndarray:
    """读向量文件 -> int64 一维数组。空文件返回空数组。"""
    p = Path(path)
    if p.stat().st_size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


# ---------------------------------------------------------------- 自检


def _self_test() -> None:
    fmt = Q8_8
    xs = np.array([0.0, -1.0, 1.5, fmt.max_val, fmt.min_val, 1000.0, -1000.0])
    raw = float_to_fixed(xs, fmt)
    back = fixed_to_float(raw, fmt)
    # 饱和检查
    assert back[5] == fmt.max_val and back[6] == fmt.min_val
    # 量化步长内还原
    assert max_abs_err(xs[:5], back[:5]) <= 0.5 / fmt.scale + 1e-12
    # 文件 round-trip
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "v.txt"
        write_vec(f, raw)
        assert np.array_equal(read_vec(f), raw)
    print(f"fixedpoint self-test PASSED ({fmt}, width={fmt.width})")


if __name__ == "__main__":
    _self_test()
