"""Device-native FFTW halfcomplex normal modes used by RPMDrate.

The transform is FFTW R2HC/sqrt(P), with inverse HC2R/sqrt(P). It is
invertible but NOT an orthonormal real transform: paired real/imaginary
components have metric weight 2 (and Gaussian variance 1/2). No FFTW or
RPMDrate runtime is needed; the implementation uses Torch FFTs on the device.
"""
from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor


class RPMDrateNormalModes:
    """RPMDrate's packed real normal modes, with an explicit bead axis.

    For ``z = rfft(q, norm='ortho')`` the packing is
    ``[Re(z0), ..., Re(z_floor(P/2)), Im(z_floor((P-1)/2)), ..., Im(z1)]``.
    The imaginary components have the Fourier *negative-sine* sign. The
    centroid is ``sqrt(P)*mean(q)``. An even-P Nyquist mode is real and unique.

    All operations retain float64 and the selected device. Inputs are not
    mutated. Shape/dtype/device validation does not synchronize CUDA; callers
    retain responsibility for finiteness checks at observation boundaries.
    Default ``bead_axis=-1`` accepts RPMDrate's [3,N,P] layout. TorchDynamics
    passes ``bead_axis=1`` for [trajectories,P,N,3]. No units/masses are changed.
    """

    def __init__(self, nbeads, *, device="cpu"):
        if (isinstance(nbeads, bool) or not isinstance(nbeads, numbers.Integral)
                or nbeads < 1):
            raise ValueError("nbeads must be a positive integer")
        requested = torch.device(device)
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("normal modes support CPU and CUDA devices")
        self._nbeads = int(nbeads)
        self._device = torch.empty(0, device=requested).device
        half = self.nbeads // 2
        self._indices = torch.cat((
            torch.arange(half + 1, device=self.device),
            torch.arange((self.nbeads - 1) // 2, 0, -1, device=self.device),
        ))
        weight = torch.full((self.nbeads,), 2., dtype=torch.float64, device=self.device)
        weight[0] = 1.
        if self.nbeads % 2 == 0:
            weight[half] = 1.
        self._metric = weight

    @property
    def nbeads(self):
        return self._nbeads

    @property
    def device(self):
        return self._device

    @property
    def frequency_indices(self):
        """Nonnegative Fourier frequency index of every packed real slot."""
        return self._indices.clone()

    @property
    def metric(self):
        """Weights such that sum(q**2) == sum(metric * forward(q)**2)."""
        return self._metric.clone()

    def frequencies(self, beta):
        """Ring frequencies in atomic units (hbar=1), in packed order."""
        if (isinstance(beta, bool) or not isinstance(beta, numbers.Real)
                or not math.isfinite(beta) or beta <= 0):
            raise ValueError("beta must be positive and finite")
        k = torch.arange(self.nbeads // 2 + 1, dtype=torch.float64, device=self.device)
        return (2 * self.nbeads / beta * torch.sin(math.pi * k / self.nbeads))[self._indices]

    def _axis(self, value, bead_axis):
        if (not isinstance(value, Tensor) or value.dtype != torch.float64
                or value.device != self.device or value.ndim < 1):
            raise ValueError(f"expected a nonempty float64 tensor on {self.device}")
        if (isinstance(bead_axis, bool) or not isinstance(bead_axis, numbers.Integral)
                or not -value.ndim <= bead_axis < value.ndim):
            raise ValueError("bead_axis must be a valid integer tensor axis")
        axis = int(bead_axis) % value.ndim
        if value.shape[axis] != self.nbeads or min(value.shape) < 1:
            raise ValueError(f"bead axis must have {self.nbeads} entries; empty batches are invalid")
        return axis

    def forward(self, positions, *, bead_axis=-1):
        axis = self._axis(positions, bead_axis)
        z = torch.fft.rfft(positions, dim=axis, norm="ortho")
        tail = z.imag.narrow(axis, 1, (self.nbeads - 1) // 2).flip((axis,))
        return torch.cat((z.real, tail), dim=axis)

    def inverse(self, modes, *, bead_axis=-1):
        axis = self._axis(modes, bead_axis)
        half = self.nbeads // 2 + 1
        real = modes.narrow(axis, 0, half)
        zero = torch.zeros_like(real.narrow(axis, 0, 1))
        tail = modes.narrow(axis, half, (self.nbeads - 1) // 2).flip((axis,))
        imaginary = torch.cat((zero, tail, zero) if self.nbeads % 2 == 0
                              else (zero, tail), dim=axis)
        return torch.fft.irfft(torch.complex(real, imaginary), n=self.nbeads,
                               dim=axis, norm="ortho")


__all__ = ["RPMDrateNormalModes"]
