"""Small reproducible rate study: pyrpmd instanton production.py --output result.json.

The separable quartic plus harmonic oscillator is an analytic validation model,
not a fitted molecular PES. All inputs and Hessians are in atomic units.
"""
import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.instanton import instanton_seed
from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.base import PESResult


class Quartic:
    def __init__(self, height=0.01, bath=0.02):
        self.height, self.bath = height, bath

    def evaluate(self, q):
        x, y = q[:, 0, 0], q[:, 0, 1]
        force = np.zeros_like(q)
        force[:, 0, 0] = -4 * self.height * x * (x * x - 1)
        force[:, 0, 1] = -self.bath * y
        return PESResult(self.height * (x * x - 1)**2 + 0.5 * self.bath * y * y, force)

    def hessian(self, q):
        hessian = np.zeros((len(q), 3, 3), dtype=np.float64)
        hessian[:, 0, 0] = 4 * self.height * (3 * q[:, 0, 0]**2 - 1)
        hessian[:, 1, 1] = self.bath
        return hessian


def build():
    potential = Quartic()
    masses = np.array([1822.0])
    mask = np.array([[True, True, False]])
    ts = np.zeros((1, 3))
    temperature = 0.55 * np.sqrt(4 * potential.height / masses[0]) / (2 * np.pi * KB_AU)
    reactant = HarmonicState([[-1.0, 0.0, 0.0]], masses, 0.0,
                             np.diag([8 * potential.height, potential.bath]), mask)
    return {"potential": potential, "masses": masses,
            "potential_identity": {"model": "quartic-plus-harmonic-v1", "height_hartree": 0.01,
                                   "minima_bohr": [-1.0, 1.0], "bath_hartree_per_bohr2": 0.02},
            "initial_path": instanton_seed(ts, [[1.0, 0.0, 0.0]], 32, 0.95),
            "temperature_kelvin": temperature, "transition_state": ts,
            "active_mask": mask, "reactants": [reactant], "bead_counts": [32, 64, 128, 256],
            "optimizer_options": {"method": "bofill", "gradient_tolerance": 1e-8},
            "convergence_relative_tolerance": 0.02}


if __name__ == "__main__":
    import json
    from pyrpmd.workflows.instanton import run_instanton
    print(json.dumps(run_instanton(**build()), indent=2, allow_nan=False))
