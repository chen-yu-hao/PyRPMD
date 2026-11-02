"""Published PIGLET data must reproduce quantum harmonic, not classical, widths."""

import numpy as np
import pytest

from benchmarks.condensed_phase.piglet_reference import harmonic_reference


def test_published_piglet_fit_and_temperature_scaling_from_continuous_lyapunov():
    original = harmonic_reference(beta=5.0)
    scaled = harmonic_reference(beta=8.0)
    assert original["maximum_absolute_potential_relative_error"] < 0.004
    assert original["maximum_absolute_kinetic_relative_error"] < 0.004
    for first, second in zip(original["cases"], scaled["cases"]):
        assert first["maximum_normalized_lyapunov_residual"] < 1e-10
        assert second["maximum_normalized_lyapunov_residual"] < 1e-10
        assert first["potential_relative_error"] == pytest.approx(second["potential_relative_error"], abs=2e-10)
        assert first["kinetic_relative_error"] == pytest.approx(second["kinetic_relative_error"], abs=2e-10)
        assert first["potential_relative_error"] == pytest.approx(first["kinetic_relative_error"], abs=2e-10)
    # The fitted data must actually add quantum fluctuations at large beta*w.
    quantum = original["cases"][-1]
    assert abs(quantum["canonical_primitive_same_beads_relative_error"]) > 0.5
    assert abs(quantum["potential_relative_error"]) < 0.004


@pytest.mark.parametrize("beta", [0, -1, np.nan, True, 1 + 0j])
def test_harmonic_reference_rejects_unphysical_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        harmonic_reference(beta)
