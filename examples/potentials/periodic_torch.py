"""Cell-aware differentiable toy PES, not a fitted materials model.

The cosine depends on fractional positions; an isotropic cell term gives a
nonzero analytic strain virial. No neighbor graph or wrapping is supplied by
PyRPMD: a trained model must implement its own cell/PBC-aware geometry.
"""
import torch


class PeriodicCosine(torch.nn.Module):
    def forward(self, q, cell, pbc):
        fractional = q @ torch.linalg.inv(cell)
        periodic_energy = .002 * ((1 - torch.cos(2 * torch.pi * fractional)) * pbc[:, None, :]).sum((1, 2))
        volume = torch.linalg.det(cell).abs()
        cell_energy = 1e-7 * (volume - 1000.) ** 2
        return periodic_energy + cell_energy


def build(device="cpu"):
    return PeriodicCosine().to(device=device, dtype=torch.float64)
