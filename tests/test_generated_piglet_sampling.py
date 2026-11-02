"""Generated small-P baths against quantum harmonic energies in real trajectories."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")

from pyrpmd.gle_parameters import generate_piglet


_EXAMPLE = Path(__file__).resolve().parents[1] / "examples/path_integral/gle_parameter_validation.py"
_SPEC = importlib.util.spec_from_file_location("generated_gle_validation", _EXAMPLE)
validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validation)


@pytest.fixture(scope="module", params=[2, 4])
def parameters(request):
    return generate_piglet((0.5 / 8, 12 / 8), beta=8., beads=request.param, naux=2)


def test_small_bead_piglet_samples_both_quantum_energies(parameters):
    beta, beads, omega, dt = 8., parameters["beads"], 1., .01
    reference = validation.harmonic_reference(parameters, [omega], dt=dt, burn_time=100.)[0]
    assert reference["maximum_normalized_lyapunov_residual"] < 1e-10
    assert max(abs(value) for value in reference["continuous_relative_error"]) < .01
    assert max(abs(value) for value in reference["timestep_bias_relative_to_continuous"]) < .005
    assert max(abs(value) for value in reference["initialization_relative_bias_at_burn"]) < .005
    options = dict(beta=beta, beads=beads, frequencies=[omega], dt=dt,
                   natoms=16, burn_time=100., production_time=200.)
    piglet = validation.sample_harmonics({"mode": "piglet", "gle": parameters}, seed=20261102 + beads, **options)
    canonical = validation.sample_harmonics({"mode": "pile-l", "gamma_centroid": np.sqrt(.5 / 8 * 12 / 8)},
                                           seed=20262102 + beads, **options)
    exact = reference["exact_quantum_potential_and_kinetic_hartree"]
    for index, name in enumerate(("potential", "kinetic_centroid_virial")):
        observed = piglet["cases"][0][name]
        baseline = canonical["cases"][0][name]
        # Six standard errors protect deterministic statistical regressions;
        # the independent covariance check separately bounds fit and dt bias.
        assert abs(observed["mean_hartree"] - exact) < 6 * observed["sampling_standard_error_hartree"] + .01 * exact
        assert abs(observed["relative_error_to_exact"]) < .05
        assert abs(observed["relative_error_to_exact"]) < abs(baseline["relative_error_to_exact"]) / 3
        finite_p = reference["canonical_finite_bead_hartree"][index]
        assert abs(baseline["mean_hartree"] - finite_p) < 6 * baseline["sampling_standard_error_hartree"] + .01 * finite_p
    assert piglet["python_step_callbacks"] == 0
    assert piglet["physical_converged"] is False


def test_independent_discrete_covariance_resolves_timestep_error(parameters):
    coarse = validation.harmonic_reference(parameters, [.5, 1., 1.5], dt=.01, burn_time=100.)
    fine = validation.harmonic_reference(parameters, [.5, 1., 1.5], dt=.005, burn_time=100.)
    for original, halved in zip(coarse, fine):
        error = np.abs(original["timestep_bias_relative_to_continuous"])
        half_error = np.abs(halved["timestep_bias_relative_to_continuous"])
        assert np.max(error) < .005
        assert np.all(half_error < .3 * error + 1e-9)
