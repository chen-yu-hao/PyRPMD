"""Condon golden-rule study using PyRPMD's existing two-state matrix provider.

Run ``pyrpmd instanton examples/instanton/golden_rule.py --output golden-rule.json``.
The displaced harmonic model checks discretization and the normal-regime
semiclassical rate. It is not an irreversible-rate benchmark for a finite
undamped oscillator bath and is not a production molecular potential.
The reactant harmonic partition uses the same two time steps and interval
counts as each optimized path, including grids with unequal segment steps.
"""
import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.golden_rule_adapters import diabatic_surfaces
from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.nonadiabatic_models import DiscreteSpinBoson


def build():
    mass, curvature, displacement = 1822.0, 0.02, 1.0
    temperature, coupling = 300.0, 1e-5
    omega = np.sqrt(curvature / mass)
    parameters = {"frequencies": [float(omega)], "couplings": [curvature * displacement],
                  "masses": [mass], "bias": 0.0, "tunnelling": coupling}
    matrix_potential = DiscreteSpinBoson(**parameters)
    surfaces = diabatic_surfaces(matrix_potential, coupling_hartree=coupling)
    mask = np.array([[True, False, False]])
    beta = 1.0 / (KB_AU * temperature)
    tau = beta / 2
    # Exact continuous harmonic bridges are seeds only; each grid and tau
    # allocation is independently reoptimized by the numerical workflow.
    t = np.linspace(0.0, tau, 9)
    bridge = np.cosh(omega * (t - tau / 2)) / np.cosh(omega * tau / 2)
    path0 = np.zeros((len(t), 1, 3))
    path0[:, 0, 0] = displacement * (bridge - 1)
    path1 = -path0[::-1].copy()
    reference = -0.5 * curvature * displacement**2
    reactant = HarmonicState([[-displacement, 0.0, 0.0]], [mass], reference,
                             [[curvature]], mask)
    return {"mode": "golden_rule", "potential": surfaces, "masses": [mass],
            "initial_path": (path0, path1), "active_mask": mask,
            "temperature_kelvin": temperature, "energy_reference_hartree": reference,
            "reactants": [reactant], "bead_counts": [(8, 8), (16, 16), (32, 32), (64, 64)],
            "potential_identity": {
                f"surface{state}": {"provider": "DiscreteSpinBoson", "parameters": parameters,
                                     "diabatic_state": state} for state in (0, 1)},
            "golden_rule_options": {"coupling_hartree": coupling, "initial_tau_fraction": 0.5,
                                     "partition_discretization": "matched",
                                     "gradient_tolerance": 1e-8, "energy_tolerance_hartree": 1e-10},
            "convergence_relative_tolerance": 0.01}


if __name__ == "__main__":
    import json
    from pyrpmd.workflows.instanton import run_instanton
    print(json.dumps(run_instanton(**build()), indent=2, allow_nan=False))
