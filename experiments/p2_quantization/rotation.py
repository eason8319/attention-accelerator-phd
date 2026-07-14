"""Hadamard and block-diagonal rotation for outlier suppression."""

from __future__ import annotations

import math

import torch
from scipy.linalg import hadamard


def _hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return normalized n×n Hadamard matrix (n must be power of 2)."""
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard size must be power of 2, got {n}")
    h = hadamard(n)
    return torch.tensor(h, device=device, dtype=dtype) / math.sqrt(n)


def random_hadamard_matrix(
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """Random signed Hadamard: D @ H where D is random ±1 diagonal."""
    if dim & (dim - 1) != 0:
        # Pad to next power of 2 for Hadamard, caller handles padding if needed
        next_pow2 = 1 << (dim - 1).bit_length()
        raise ValueError(
            f"dim {dim} must be power of 2 for Hadamard; use pad_to_pow2=True or dim={next_pow2}"
        )
    device = device or torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    signs = torch.randint(0, 2, (dim,), generator=gen, device=device, dtype=dtype) * 2 - 1
    h = _hadamard_matrix(dim, device, dtype)
    return signs.diag() @ h


def apply_rotation(x: torch.Tensor, matrix: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Apply orthogonal matrix along the last feature axis."""
    return torch.matmul(x, matrix.to(device=x.device, dtype=x.dtype))


def pad_to_pow2(x: torch.Tensor, axis: int = -1) -> tuple[torch.Tensor, int]:
    """Zero-pad axis to next power of 2; return padded tensor and original size."""
    axis = axis if axis >= 0 else x.ndim + axis
    dim = x.shape[axis]
    if dim & (dim - 1) == 0:
        return x, dim
    padded = 1 << (dim - 1).bit_length()
    pad_amount = padded - dim
    pad_spec = [0] * (2 * x.ndim)
    pad_spec[2 * (x.ndim - 1 - axis) + 1] = pad_amount
    return torch.nn.functional.pad(x, pad_spec), dim


def unpad(x: torch.Tensor, axis: int, original: int) -> torch.Tensor:
    axis = axis if axis >= 0 else x.ndim + axis
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, original)
    return x[tuple(sl)]


class RandomHadamardRotation:
    """Apply random Hadamard rotation along head_dim (power-of-2 padded)."""

    def __init__(self, dim: int, seed: int = 0, device: torch.device | None = None):
        self.original_dim = dim
        self.padded_dim = 1 << (dim - 1).bit_length() if (dim & (dim - 1)) else dim
        self.matrix = random_hadamard_matrix(
            self.padded_dim, device=device or torch.device("cpu"), seed=seed
        )

    def rotate(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        x_pad, orig = pad_to_pow2(x, axis)
        m = self.matrix.to(device=x.device, dtype=x.dtype)
        out = apply_rotation(x_pad, m, axis)
        return unpad(out, axis, orig)

    def inverse(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """Inverse equals transpose for orthogonal Hadamard rotation."""
        x_pad, orig = pad_to_pow2(x, axis)
        m = self.matrix.T.to(device=x.device, dtype=x.dtype)
        out = apply_rotation(x_pad, m, axis)
        return unpad(out, axis, orig)


class BlockDiagonalRotation:
    """Block-diagonal Walsh–Hadamard rotation (BDR), QuaRot/SAW-INT4 style.

    Constructs ``R = block_diag(H, ..., H) @ D`` where ``H`` is the normalized
    Walsh–Hadamard matrix of size ``block_size`` and ``D`` is a random ±1
    diagonal over the full ``dim``.

    Putting the random signs on the *input* side (right-multiply by ``D``) keeps
    each quant group aligned with a flat Hadamard block and avoids the severe
    seed sensitivity of per-block ``D_b @ H`` or Gaussian-QR orthogonal blocks.
    Within each block, ``|R_ij| = 1/sqrt(block_size)``.
    """

    def __init__(
        self,
        dim: int,
        block_size: int = 32,
        seed: int = 0,
        device: torch.device | None = None,
    ):
        if dim % block_size != 0:
            raise ValueError(f"dim {dim} must be divisible by block_size {block_size}")
        if block_size & (block_size - 1) != 0:
            raise ValueError(f"block_size must be a power of 2 for Hadamard, got {block_size}")
        self.dim = dim
        self.block_size = block_size
        self.n_blocks = dim // block_size
        device = device or torch.device("cpu")

        h = _hadamard_matrix(block_size, device, torch.float32)
        block_h = torch.block_diag(*[h for _ in range(self.n_blocks)])

        gen = torch.Generator(device=device).manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), generator=gen, device=device).float() * 2 - 1
        self.matrix = block_h @ signs.diag()

    def rotate(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        m = self.matrix.to(device=x.device, dtype=x.dtype)
        return apply_rotation(x, m, axis)

    def inverse(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        m = self.matrix.T.to(device=x.device, dtype=x.dtype)
        return apply_rotation(x, m, axis)
