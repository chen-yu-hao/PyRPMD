"""C_S x eta_NA for a reaction path that passes a conical intersection (research).

A 2D model in mass-weighted coordinates: two diabats cross at x = 0 and are
coupled by lam * (y - y_c), so a conical intersection sits at (0, y_c), 2.5
transverse zero-point widths from the ground-state reaction path.  The
subspace pipeline builds its 1D path model on the ground state, places
electronic geometries along the same path, diabatizes the (sign- and
order-scrambled) adiabatic data from a Python source and multiplies C_S by
eta_NA = k_QM(coupled states) / k_QM(ground adiabat).

    python examples/subspace_correction/conical_intersection_path.py --temperature 300

For a molecule, replace the Python source by an excited-state backend, for
example (TOML for ``pyrpmd subspace-correct``):

    [electronic]
    source = "backend"
    object = "pyrpmd.potentials.excited_state:OpenMolcasExcitedStateBackend"
    arguments = { symbols = ["C", "H", "H", "H", "H"], nstates = 2, program_version = "24.10",
                  basis = "ANO-S-VDZP" }
    nstates = 2
    points = 64

Linear-response TDDFT does not give the correct dimensionality of S0/S1
conical intersections; use a multireference method (or spin-flip TDDFT) for
the states that meet.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.subspace_complex.models import abstract_rate_run  # noqa: E402

from pyrpmd.analysis import subspace_pipeline  # noqa: E402

CM = 1.0 / 219474.6313632


@dataclass(frozen=True)
class CrossingCI2D:
    D: float = 0.04             # diabatic asymptotic splitting (Eh)
    width: float = 20.0         # switching width of the diabats along x
    omega: float = 1500.0 * CM  # transverse frequency
    lam: float = 1.3333e-4      # coupling slope; the gap on the path is ~ 2 lam y_c
    y_c: float = 30.0           # position of the conical intersection
    c: float = 40.0             # range of the coupling along x

    @property
    def dimension(self):
        return 2

    def diabatic(self, q):
        x, y = np.asarray(q, dtype=float)[..., 0], np.asarray(q, dtype=float)[..., 1]
        rise = np.tanh(x / self.width)
        harmonic = 0.5 * self.omega**2 * y**2
        out = np.zeros(x.shape + (2, 2))
        out[..., 0, 0] = 0.5 * self.D * (1 + rise) + harmonic
        out[..., 1, 1] = 0.5 * self.D * (1 - rise) + harmonic
        out[..., 0, 1] = out[..., 1, 0] = self.lam * (y - self.y_c) * np.exp(-((x / self.c) ** 2))
        return out

    def energy(self, q):
        return np.linalg.eigvalsh(self.diabatic(q))[..., 0]

    def gradient(self, q, step=1e-5):
        q = np.asarray(q, dtype=float)
        return np.array([(self.energy(q + e) - self.energy(q - e)) / (2 * step) for e in np.eye(2) * step])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--coupling-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: a temporary one)")
    args = parser.parse_args()
    model = CrossingCI2D(lam=1.3333e-4 * args.coupling_scale)
    rng = np.random.default_rng(1)

    def electronic(geometry):
        energies, vectors = np.linalg.eigh(model.diabatic(geometry))
        order = rng.permutation(2)  # an electronic-structure code has arbitrary order and phases
        return {"energies": energies[order], "vectors": (vectors * rng.choice([-1.0, 1.0], 2))[:, order]}

    run = abstract_rate_run(model, args.temperature, 16)
    out = args.out or Path(tempfile.mkdtemp(prefix="ci_path_"))
    spec = {
        "output": {"directory": str(out)},
        "path": {"source": "mep", "step": 4.0},
        "subspace": {"dimension": 1, "environment": "adiabatic"},
        "rpmd": {"nbeads": 16, "windows": 16, "walkers": 16, "burn_in": 50, "sweeps": 100,
                 "trajectories": 128, "seed": 7},
        "electronic": {"source": "python", "object": electronic, "nstates": 2, "points": 64},
    }
    report = subspace_pipeline.run_subspace_pipeline(spec, run=run)
    record = report["nonadiabatic"]["1"]
    print(json.dumps({
        "status": report["status"],
        "applied_factors": report["applied_factors"],
        "nonadiabatic_status": record["status"],
        "minimum_gap_on_path_Eh": record.get("electronic", {}).get("minimum_adiabatic_gap"),
        "branching": record.get("branching"),
        "gates": {g["gate"]: g["passed"] for g in record.get("validity", {}).get("gates", [])},
        "report": str(out / "subspace_pipeline.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
