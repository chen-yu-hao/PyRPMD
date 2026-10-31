"""Runnable KC, MV and iso-Ehrenfest model APIs; no chemistry validation claim.

Run from the repository root after installation:
    python examples/nonadiabatic/method_models.py
"""

import json

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.isomorphic import IsomorphicEhrenfestDynamics, IsomorphicEhrenfestState
from pyrpmd.kc_rpmd import KCCalibration, KCRPMDDynamics, KCRPMDSampler, KCRPMDState
from pyrpmd.mv_rpmd import MVRPMDDynamics, MVRPMDSampler, MVRPMDState
from pyrpmd.nonadiabatic import DiabaticResult


class CoupledOscillators:
    """One artificial Cartesian nucleus, two coupled harmonic diabatic states."""

    def evaluate(self, q):
        x = q[:, 0, 0]
        common = 0.2 * np.sum(q*q, axis=(1, 2))
        matrix = np.zeros((len(q), 2, 2))
        gradient = np.zeros((len(q), 2, 2, 1, 3))
        matrix[:, 0, 0], matrix[:, 1, 1] = common + 0.3*x, common - 0.3*x + 0.1
        matrix[:, 0, 1] = matrix[:, 1, 0] = 0.2
        gradient[:, 0, 0] = gradient[:, 1, 1] = 0.4*q
        gradient[:, 0, 0, 0, 0] += 0.3
        gradient[:, 1, 1, 0, 0] -= 0.3
        return DiabaticResult(matrix, gradient)


def run_demo():
    beta, count, dt = 3.0, 3, 0.001
    rng = np.random.default_rng(17)
    nuclear = State(rng.normal(scale=0.1, size=(count, 1, 3)),
                    rng.normal(scale=0.04, size=(count, 1, 3)))
    model = CoupledOscillators()
    mv_state = MVRPMDState(nuclear.q, nuclear.p,
                          mapping_q=rng.normal(size=(count, 2)),
                          mapping_p=rng.normal(size=(count, 2)))
    mv = MVRPMDDynamics(model, [1], beta, count, dt, 2, native=False)
    # For constant K=0.2 and gap slope 0.6, the constrained averages in
    # original KC Eqs. 27/32 are exactly eta=3 and <1/K>=5.
    calibration = KCCalibration(mean_gap_gradient=3.0, mean_inverse_coupling=5.0)
    kc = KCRPMDDynamics(model, [1], beta, count, dt,
                        calibration=calibration, a=0.1, b=8, native=False)
    kc_state = KCRPMDState(nuclear.q, nuclear.p, y=0.1, p_y=0.01)
    iso = IsomorphicEhrenfestDynamics(model, [1], beta, count, dt, native=False)
    iso_state = IsomorphicEhrenfestState(nuclear.q, nuclear.p, amplitudes=[0.8, 0.6j])
    result = {"beta_hartree_inverse": beta, "beads": count,
              "timestep_atomic": dt, "propagation_steps": 20,
              "physical_converged": False, "model_equilibrated": False}
    for name, dynamics, state in (("mv_rpmd", mv, mv_state),
                                  ("kc_rpmd", kc, kc_state),
                                  ("iso_ehrenfest", iso, iso_state)):
        initial = dynamics.energy(state)
        for _ in range(20):
            dynamics.step(state)
        result[name] = {"initial_sum_energy_hartree": initial,
                        "final_sum_energy_hartree": dynamics.energy(state),
                        "centroid_bohr": state.q.mean(axis=0).tolist()}
    result["mv_rpmd"]["signed_populations"] = mv.populations(mv_state).tolist()
    result["mv_rpmd"]["weight_sign"] = mv.potential.evaluate(mv_state).sign
    result["kc_rpmd"]["auxiliary_y"] = kc_state.y
    result["kc_rpmd"]["published_convention_auxiliary_mass"] = calibration.auxiliary_mass(beta)
    result["iso_ehrenfest"]["diabatic_populations"] = iso_state.populations().tolist()
    samplers = (("mv_rpmd", MVRPMDSampler(mv, position_step=0.02, mapping_step=0.03, seed=7), mv_state.copy()),
                ("kc_rpmd", KCRPMDSampler(kc, position_step=0.02, auxiliary_step=0.04, seed=7), kc_state.copy()))
    for name, sampler, state in samplers:
        for _ in range(100):
            sampler.step(state)
        sampler.draw_momenta(state)
        result[name]["sampler_attempts"] = sampler.attempted
        result[name]["sampler_accepted"] = sampler.accepted
    return result


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))

