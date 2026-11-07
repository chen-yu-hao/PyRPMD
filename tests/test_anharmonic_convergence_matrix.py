"""Small convergence matrices for the anharmonic quantum references.

The benchmark helpers use finite boxes, finite grids, truncated oscillator
bases, and finite lead discretisations.  These tests make those numerical
limits visible across temperature and resolution changes; they are not claims
that the finite reference is an exact result for a general physical system.
"""

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "anharmonic_quantum"))

from quantum import (  # noqa: E402
    eckart_scattering,
    fermi_quantum,
    morse_quantum,
    spectrum,
    thermal_rate,
)


@pytest.mark.parametrize("beta", [4.0, 8.0])
def test_morse_beta_grid_and_box_refinement_is_stable(beta):
    """Morse ground and thermal energies are stable under two resolution limits."""
    _, coarse = morse_quantum(beta=beta, n=80, bounds=(-3.0, 10.0))
    _, refined = morse_quantum(beta=beta, n=120, bounds=(-3.5, 12.0))

    coarse_values = np.array([(record["E0"], record["thermal_energy"]) for record in coarse])
    refined_values = np.array([(record["E0"], record["thermal_energy"]) for record in refined])
    # The lower-temperature case is almost machine-precision stable; the
    # warmer case retains a small finite-box/grid tail in the thermal energy.
    np.testing.assert_allclose(coarse_values, refined_values, rtol=0.0, atol=3.0e-7)

    for low, high in zip(coarse, refined):
        assert low["grid_points"] == 80
        assert high["grid_points"] == 120
        assert low["box"] != high["box"]
        assert np.isfinite(low["population_above_dissociation"])
        assert np.isfinite(high["population_above_dissociation"])


@pytest.mark.parametrize("beta", [4.0, 8.0])
def test_fermi_spectrum_is_stable_under_basis_refinement(beta):
    """The finite oscillator reference has a reproducible spectrum at each beta."""
    low_de, low_coeff, low_meta = fermi_quantum(
        beta=beta, dim=2, basis=(10, 14, 8), states=50
    )
    high_de, high_coeff, high_meta = fermi_quantum(
        beta=beta, dim=2, basis=(12, 16, 10), states=50
    )
    omega = np.linspace(0.3, 3.0, 120)
    low = spectrum(low_de, low_coeff, omega, 0.04)
    high = spectrum(high_de, high_coeff, omega, 0.04)

    relative_change = np.linalg.norm(low - high) / np.linalg.norm(high)
    assert relative_change < 5.0e-6
    assert low_meta["basis"] == [10, 14]
    assert high_meta["basis"] == [12, 16]
    assert low_meta["dimension"] == high_meta["dimension"] == 2
    assert abs(low_meta["response_sum_rule"] - 1.0) < 1.0e-8
    assert abs(high_meta["response_sum_rule"] - 1.0) < 1.0e-8


def test_fermi_temperature_matrix_has_ordered_thermal_energy():
    """Changing beta changes thermal weights in the expected direction."""
    energies = []
    for beta in (4.0, 6.0, 8.0):
        _, _, metadata = fermi_quantum(
            beta=beta, dim=2, basis=(12, 16, 10), states=50
        )
        energies.append(metadata["thermal_energy"])
    assert energies[0] > energies[1] > energies[2]
    assert np.all(np.isfinite(energies))


def test_curved_eckart_grid_box_and_temperature_matrix_converges():
    """Curved Eckart transmission and thermal rates improve with grid/box refinement."""
    energy = np.linspace(0.02, 3.0, 48)
    resolutions = [(0.20, 5.0), (0.15, 6.0), (0.10, 7.0)]
    results = []
    for dq, extent in resolutions:
        crp, ground_channel, metadata = eckart_scattering(
            energy, dim=2, nx=8, nz=3, dq=dq, L=extent
        )
        rates = np.array(
            [
                thermal_rate(energy, crp, beta, metadata["relative_lead_energies"])
                for beta in (4.0, 8.0)
            ]
        )
        results.append((ground_channel, rates, metadata))

    coarse, coarse_rates, coarse_meta = results[0]
    medium, medium_rates, medium_meta = results[1]
    refined, refined_rates, refined_meta = results[2]
    coarse_error = np.linalg.norm(coarse - refined) / np.linalg.norm(refined)
    medium_error = np.linalg.norm(medium - refined) / np.linalg.norm(refined)
    assert medium_error < coarse_error
    assert medium_error < 1.0e-2

    for rates in (coarse_rates, medium_rates, refined_rates):
        assert rates[0] > rates[1] > 0.0
        assert np.all(np.isfinite(rates))
    for rates in (coarse_rates, medium_rates):
        np.testing.assert_allclose(rates, refined_rates, rtol=2.0e-2, atol=1.0e-10)

    for metadata in (coarse_meta, medium_meta, refined_meta):
        assert metadata["dimension"] == 2
        assert metadata["q_discretization"] == "nearest-neighbor open-lead finite difference"
        assert metadata["max_unitarity_defect"] < 1.0e-7
