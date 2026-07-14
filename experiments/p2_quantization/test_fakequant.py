"""Unit tests for fake-quant library."""

from __future__ import annotations

import math

import pytest
import torch

from fakequant import (
    QuantGranularity,
    fp8_fake_quant,
    int4_fake_quant,
    int8_fake_quant,
    mxfp4_fake_quant,
    relative_error,
)


def test_int8_symmetric_per_tensor_hand_calc() -> None:
    # Hand-calculated: values [-1, 0, 0.5, 1], symmetric INT8, scale = 1/127
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    y = int8_fake_quant(x, granularity=QuantGranularity.PER_TENSOR, symmetric=True)
    expected = torch.round(x * 127) / 127
    assert torch.allclose(y, expected, atol=1e-6)


def test_int8_asymmetric_per_tensor() -> None:
    x = torch.tensor([0.0, 0.25, 0.5, 1.0])
    y = int8_fake_quant(x, granularity=QuantGranularity.PER_TENSOR, symmetric=False)
    qmax = 255
    scale = (x.max() - x.min()) / qmax
    zp = torch.round(0 - x.min() / scale)
    expected = (torch.round(x / scale + zp).clamp(0, qmax) - zp) * scale
    assert torch.allclose(y, expected, atol=1e-5)


def test_int4_per_group() -> None:
    for gs in (32, 64):
        x = torch.randn(gs * 2)
        y = int4_fake_quant(
            x,
            granularity=QuantGranularity.PER_GROUP,
            group_size=gs,
            axis=0,
        )
        assert y.shape == (gs * 2,)


@pytest.mark.parametrize("granularity", [QuantGranularity.PER_TENSOR, QuantGranularity.PER_CHANNEL])
def test_int8_granularities(granularity: QuantGranularity) -> None:
    x = torch.randn(4, 8, 16)
    y = int8_fake_quant(x, granularity=granularity, axis=-1)
    assert y.shape == x.shape
    assert relative_error(x, y) < 0.5


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_fp8_vs_torch(fmt: str) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("FP8 not available")
    x = torch.randn(32, 64) * 10
    max_abs = x.abs().max()
    fp8_dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    scale = max_abs / (448.0 if fmt == "e4m3" else 57344.0)
    ref = (x / scale).to(fp8_dtype).float() * scale
    y = fp8_fake_quant(x, fmt=fmt, granularity=QuantGranularity.PER_TENSOR)
    assert torch.allclose(y, ref, atol=1e-2, rtol=1e-2)


def test_mxfp4_block() -> None:
    x = torch.randn(32) * 3
    y = mxfp4_fake_quant(x, block_size=32, axis=0)
    assert y.shape == x.shape
    assert relative_error(x, y) < 1.0


def test_int4_symmetric_levels() -> None:
    x = torch.linspace(-2, 2, 8)
    y = int4_fake_quant(x, granularity=QuantGranularity.PER_TENSOR)
    qmax = 7
    scale = x.abs().max() / qmax
    expected = torch.round(x / scale).clamp(-qmax, qmax) * scale
    assert torch.allclose(y, expected, atol=1e-5)


def test_rotation_preserves_norm() -> None:
    from rotation import RandomHadamardRotation

    dim = 64
    rot = RandomHadamardRotation(dim, seed=42)
    x = torch.randn(2, 4, 32, dim)
    y = rot.rotate(x, axis=-1)
    assert torch.allclose(x.norm(), y.norm(), atol=1e-4)


def test_bdr_orthogonal() -> None:
    from rotation import BlockDiagonalRotation

    rot = BlockDiagonalRotation(64, block_size=32, seed=0)
    m = rot.matrix
    identity = torch.eye(64)
    assert torch.allclose(m @ m.T, identity, atol=1e-5)


def test_bdr_is_block_hadamard() -> None:
    """BDR is block_diag(H,...,H) @ D: block-sparse with flat |entries|."""
    import math

    from rotation import BlockDiagonalRotation

    block_size = 32
    rot = BlockDiagonalRotation(64, block_size=block_size, seed=0)
    m = rot.matrix
    expected = 1.0 / math.sqrt(block_size)
    # Off-block entries are zero; on-block entries have constant magnitude.
    for b in range(2):
        sl = slice(b * block_size, (b + 1) * block_size)
        block = m[sl, sl].abs()
        assert torch.allclose(block, torch.full_like(block, expected), atol=1e-6)
        other = slice((1 - b) * block_size, (2 - b) * block_size)
        assert torch.allclose(m[sl, other], torch.zeros_like(m[sl, other]), atol=1e-7)
    # Random signs are on the input side: columns have uniform ±expected magnitude.
    assert torch.allclose(m.abs().sum(dim=0), torch.full((64,), expected * block_size), atol=1e-5)


def test_bdr_preserves_norm() -> None:
    from rotation import BlockDiagonalRotation

    rot = BlockDiagonalRotation(64, block_size=32, seed=42)
    x = torch.randn(2, 4, 8, 64)
    y = rot.rotate(x, axis=-1)
    assert torch.allclose(x.norm(), y.norm(), atol=1e-4)
    z = rot.inverse(y, axis=-1)
    assert torch.allclose(x, z, atol=1e-5)
