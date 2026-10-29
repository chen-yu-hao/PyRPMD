"""Same illustrative Gaussian repulsion, with PyTorch autograd forces."""

import torch


class GaussianRepulsion(torch.nn.Module):
    def __init__(self, height_hartree=0.002, width_bohr=1.5):
        super().__init__()
        self.height = float(height_hartree)
        self.width = float(width_bohr)

    def forward(self, q):
        displacement = q[:, 0] - q[:, 1]
        return self.height * torch.exp(
            -torch.sum(displacement**2, dim=1) / (2 * self.width**2)
        )


def build_model(device="cpu", **params):
    return GaussianRepulsion(**params).to(device=device, dtype=torch.float64)
