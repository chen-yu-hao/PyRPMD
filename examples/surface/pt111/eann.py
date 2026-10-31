"""Source-pinned, CPU float64 adapter for the public CO/O/Pt(111) EANN.

Li et al., Nature Communications 17, 10859 (2026),
doi:10.1038/s41467-026-77226-4. Model: doi:10.5281/zenodo.21401971,
CC-BY-4.0. Coordinates stay unwrapped; the model constructs periodic images.
"""
from pathlib import Path
import hashlib
import numpy as np
from pyrpmd.potentials.base import Potential, PESResult, coordinates
from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU, AMU_TO_AU

SHA256 = "64a2a91636d11c091585aaf1bc72af6397e9672a0a3bd02602f95ec1772fb6f4"


class EANN(Potential):
    metadata = {"supports_pbc": True, "thread_safety": "context_serial", "execution_device": "cpu",
                "model_doi": "10.5281/zenodo.21401971", "model_sha256": SHA256}

    def __init__(self, model):
        import torch
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError("Missing Pt(111) model; run python examples/surface/pt111/fetch_model.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Pt(111) model SHA256 mismatch; expected the published REANN_PES_DOUBLE.pt")
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cached = None

    def bind_system(self, system):
        get = system.get if isinstance(system, dict) else lambda k: getattr(system, k)
        t = self.torch
        self.cell = t.tensor(np.asarray(get("cell")) * BOHR_ANGSTROM, dtype=t.float64)
        self.pbc = t.tensor(np.asarray(get("pbc"), dtype=np.int32))
        self.species = t.tensor([{"C": 0, "O": 1, "Pt": 2}[s] for s in get("symbols")], dtype=t.int64)
        self.masses = t.tensor(np.asarray(get("masses")) / AMU_TO_AU, dtype=t.float64)
        self.cached = None

    def evaluate(self, q):
        q = coordinates(q)
        if self.cached is not None and np.array_equal(q, self.cached[0]):
            return PESResult(self.cached[1].copy(), self.cached[2].copy())
        energies, forces = [], []
        # The TorchScript forward differentiates energy internally; no no_grad.
        for bead in q:
            e, f = self.model(self.pbc, self.torch.tensor(bead * BOHR_ANGSTROM, dtype=self.torch.float64),
                              self.cell, self.species, self.masses)
            energies.append(float(e) * EV_TO_AU)
            forces.append(f.detach().numpy().copy() * EV_TO_AU * BOHR_ANGSTROM)
        energy, force = np.asarray(energies), np.asarray(forces)
        self.cached = q.copy(), energy.copy(), force.copy()
        return PESResult(energy, force)


def build(model):
    return EANN(model)
