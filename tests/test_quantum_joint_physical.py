"""Physical sanity checks for the online mixed quantum/RPMD reference driver.

These tests are deliberately small and deterministic.  They do not turn the
centroid/Ehrenfest closure into an exact system--bath method; they guard the
numerical properties that a reference implementation must retain while the
larger convergence studies live in benchmark workflows.
"""

import numpy as np
import pytest

from pyrpmd.analysis.quantum_joint import OnlineQuantumRPMD


def _gaussian_state(grid, center=0.5, momentum=0.4, width=1.0):
    psi = np.exp(-0.5 * ((grid - center) / width) ** 2 + 1j * momentum * grid)
    dx = grid[1] - grid[0]
    return psi / np.sqrt(dx * np.sum(np.abs(psi) ** 2))


def _harmonic_runner(dt, *, box=8.0, points=128):
    grid = np.linspace(-box, box, points, endpoint=False)
    return OnlineQuantumRPMD(
        grid,
        quantum_mass=1.0,
        beta=4.0,
        dt=dt,
        positions=np.zeros((1, 1)),
        momenta=np.zeros((1, 1)),
        bath_masses=np.ones(1),
        quantum_potential=lambda x, centroid: 0.5 * x**2,
        quantum_gradient=lambda x, centroid: np.zeros((1, len(x))),
        wavefunction=_gaussian_state(grid),
    )


def test_harmonic_online_propagation_is_second_order_in_time_step():
    coarse = _harmonic_runner(0.01).propagate(20)
    fine = _harmonic_runner(0.005).propagate(40)

    coarse_observables = np.array([
        coarse.quantum_position[-1],
        coarse.quantum_momentum[-1],
        coarse.quantum_energy[-1],
    ])
    fine_observables = np.array([
        fine.quantum_position[-1],
        fine.quantum_momentum[-1],
        fine.quantum_energy[-1],
    ])
    # Both runs reach t=0.2.  The error is already small at this short time,
    # and the finer Strang step must improve the recorded observables.
    assert coarse.time[-1] == pytest.approx(0.2)
    assert fine.time[-1] == pytest.approx(0.2)
    assert np.linalg.norm(coarse_observables - fine_observables) < 2.0e-6
    assert np.max(np.abs(fine.wavefunction_norm - 1.0)) < 2.0e-11


def test_harmonic_wavepacket_converges_when_periodic_box_is_enlarged():
    compact = _harmonic_runner(0.005, box=6.0, points=96).propagate(40)
    enlarged = _harmonic_runner(0.005, box=8.0, points=128).propagate(40)

    compact_values = np.array([
        compact.quantum_position[-1],
        compact.quantum_momentum[-1],
        compact.quantum_energy[-1],
    ])
    enlarged_values = np.array([
        enlarged.quantum_position[-1],
        enlarged.quantum_momentum[-1],
        enlarged.quantum_energy[-1],
    ])
    np.testing.assert_allclose(compact_values, enlarged_values, rtol=0.0, atol=2.0e-8)

    def edge_probability(result, points):
        nedge = max(1, points // 20)
        indices = np.r_[0:nedge, points - nedge:points]
        dx = 2.0 * (6.0 if points == 96 else 8.0) / points
        return dx * np.sum(np.abs(result.wavefunctions[:, indices]) ** 2, axis=1)

    assert np.max(edge_probability(compact, 96)) < 1.0e-8
    assert np.max(edge_probability(enlarged, 128)) < 1.0e-14


def test_analytic_and_finite_difference_ehrenfest_forces_agree():
    grid = np.linspace(-6.0, 6.0, 96, endpoint=False)
    psi = _gaussian_state(grid, center=0.3, momentum=0.0)
    coupling = 0.37
    centroid = 0.42

    def potential(x, q):
        return 0.5 * x**2 + coupling * q[0] * x + 0.2 * q[0] ** 2

    analytic = OnlineQuantumRPMD(
        grid, 1.0, 4.0, 0.002, [[centroid]], [[0.0]], [1.0], potential,
        quantum_gradient=lambda x, q: np.full((1, len(x)), coupling * x + 0.4 * q[0]),
        wavefunction=psi,
    ).propagate(0)
    finite_difference = OnlineQuantumRPMD(
        grid, 1.0, 4.0, 0.002, [[centroid]], [[0.0]], [1.0], potential,
        finite_difference_step=1.0e-6, wavefunction=psi,
    ).propagate(0)
    np.testing.assert_allclose(
        analytic.quantum_force,
        finite_difference.quantum_force,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_online_result_records_scope_and_diagnostic_provenance():
    result = _harmonic_runner(0.01).propagate(2)
    assert result.diagnostics["quantum_coordinate_count"] == 1
    assert result.diagnostics["quantum_propagator"] == "periodic_fourier_strang"
    assert result.diagnostics["physical_converged"] is False
    assert "no generally exact flux-side rate" in result.diagnostics["rate_claim"]
