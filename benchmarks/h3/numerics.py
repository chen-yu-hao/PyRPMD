"""Physical BKMP2 NVE checks independent of the rate sampling estimator.

The same initial ring-polymer state is propagated at three time steps. This
checks integration accuracy, not equilibrium or bead-number convergence.
Energy errors are divided by P*kBT so the test is invariant to the PES zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pes import build
from pyrpmd.constants import AMU_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics, State, native_capabilities


def propagate(potential, initial, masses, beta, dt_fs, duration_fs, native):
    dynamics = Dynamics(
        potential,
        masses,
        beta,
        len(initial.q),
        dt_fs * FS_TO_AU,
        native=native,
    )
    state = initial.copy()
    energy0 = dynamics.energy(state)
    errors = []
    for _ in range(round(duration_fs / dt_fs)):
        dynamics.step(state)
        errors.append(dynamics.energy(state) - energy0)
    maximum = float(np.max(np.abs(errors)))
    return state, {
        "dt_fs": dt_fs,
        "duration_fs": duration_fs,
        "steps": state.step,
        "maximum_energy_error_hartree_per_bead": maximum / len(state.q),
        "maximum_energy_error_kbt": maximum * beta / len(state.q),
        "endpoint_energy_error_hartree_per_bead": float(errors[-1]) / len(state.q),
    }


def check():
    start = time.monotonic()
    capabilities = native_capabilities()
    if not capabilities["available"]:
        raise RuntimeError("Build PyRPMD's native kernel before this benchmark")
    potential = build()
    temperature = 1500.0
    beta = 1.0 / (KB_AU * temperature)
    masses = np.full(3, 1.00782503223 * AMU_TO_AU)
    geometry = np.array([[-1.757, 0.0, 0.0], [0.0, 0.0, 0.0], [1.757, 0.0, 0.0]])
    comparison = []
    refinement = []
    try:
        for beads in (1, 8, 16, 32):
            rng = np.random.default_rng(7391 + beads)
            q = np.repeat(geometry[None], beads, axis=0)
            q += rng.normal(scale=0.015, size=q.shape)
            p = rng.normal(size=q.shape) * np.sqrt(masses[None, :, None] * beads / beta)
            # Remove only total translation, retaining centroid relative flux.
            p -= p.sum(axis=(0, 1), keepdims=True) / (3 * beads)
            state = State(q, p)
            native, diagnostic = propagate(
                potential, state, masses, beta, 0.05, 10.0, "required"
            )
            reference, _ = propagate(potential, state, masses, beta, 0.05, 10.0, False)
            position_error = float(np.max(np.abs(native.q - reference.q)))
            momentum_error = float(np.max(np.abs(native.p - reference.p)))
            np.testing.assert_allclose(native.q, reference.q, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(native.p, reference.p, rtol=1e-8, atol=1e-8)
            comparison.append(
                {
                    "beads": beads,
                    "maximum_position_difference_bohr": position_error,
                    "maximum_momentum_difference_au": momentum_error,
                    **diagnostic,
                }
            )
            if beads == 16:
                coarse, coarse_diag = propagate(
                    potential, state, masses, beta, 0.1, 10.0, "required"
                )
                finest, finest_diag = propagate(
                    potential, state, masses, beta, 0.025, 10.0, "required"
                )
                coarse_error = float(np.linalg.norm(coarse.q - finest.q))
                fine_error = float(np.linalg.norm(native.q - finest.q))
                ratio = fine_error / coarse_error
                # A strict inequality avoids an accidental equal-endpoint pass.
                if not 0 < ratio < 0.5:
                    raise AssertionError(
                        f"Time-step refinement did not reduce position error: {ratio}"
                    )
                if (
                    not diagnostic["maximum_energy_error_kbt"]
                    < coarse_diag["maximum_energy_error_kbt"]
                ):
                    raise AssertionError(
                        "Time-step halving did not improve the maximum NVE energy error"
                    )
                if coarse_diag["maximum_energy_error_kbt"] > 0.1:
                    raise AssertionError(
                        "NVE energy error at dt=0.1 fs exceeds 0.1 kBT per bead"
                    )
                refinement = [coarse_diag, diagnostic, finest_diag]
                refinement_comparison = {
                    "coarse_endpoint_error_bohr": coarse_error,
                    "fine_endpoint_error_bohr": fine_error,
                    "fine_to_coarse_error_ratio": ratio,
                    "reference_dt_fs": 0.025,
                }
    finally:
        if hasattr(potential, "close"):
            potential.close()
    return {
        "system": "H + H2 / BKMP2",
        "status": "passed",
        "temperature_K": temperature,
        "native": capabilities,
        "native_vs_numpy": comparison,
        "time_step_refinement": refinement,
        "refinement_comparison": refinement_comparison,
        "energy_error_measure": "max_t |H_P(t)-H_P(0)| / (P kBT); independent of energy zero",
        "initial_ensemble": "seeded near-saddle numerical stress test; not an equilibrium ensemble",
        "bead_convergence_demonstrated": False,
        "elapsed_seconds": time.monotonic() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
