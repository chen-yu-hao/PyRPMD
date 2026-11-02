"""Small, reproducible *untrained* MACE used by the batching checks.

Nothing is downloaded.  This is a throughput/reference fixture, not a model
that can predict chemical energies.  Imports remain optional and local.
"""

from __future__ import annotations

import numpy as np


MODEL_DESCRIPTION = {
    "architecture": "ScaleShiftMACE",
    "interactions": 2,
    "hidden_irreps": "8x0e + 8x1o",
    "correlation": 2,
    "radial_mlp": [8, 8],
    "cutoff_angstrom": 2.8,
    "atomic_numbers": [1, 8],
    "seed": 1729,
    "trained": False,
}


def make_random_mace(dtype="float64", device="cpu", seed=1729):
    """Construct a small real model without changing the caller's RNG/dtype."""
    import torch
    # MACE initializes the compatibility handling its pinned e3nn requires.
    from mace import modules
    from e3nn import o3

    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, dtype))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = modules.ScaleShiftMACE(
                r_max=2.8,
                num_bessel=4,
                num_polynomial_cutoff=5,
                max_ell=1,
                interaction_cls=modules.RealAgnosticResidualInteractionBlock,
                interaction_cls_first=modules.RealAgnosticInteractionBlock,
                num_interactions=2,
                num_elements=2,
                hidden_irreps=o3.Irreps("8x0e + 8x1o"),
                MLP_irreps=o3.Irreps("8x0e"),
                atomic_energies=np.array([0.1, -0.2]),
                avg_num_neighbors=6.0,
                atomic_numbers=[1, 8],
                correlation=2,
                gate=torch.nn.functional.silu,
                radial_MLP=[8, 8],
                atomic_inter_scale=1.2,
                atomic_inter_shift=-0.1,
            )
    finally:
        torch.set_default_dtype(original_dtype)
    return model.to(device=device).eval()
