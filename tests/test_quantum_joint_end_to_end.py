"""End-to-end regression checks for full-dimensional online QE--RPMD.

The fixture contains three ring-polymer beads and two classical bath
coordinates.  It is intentionally small, but it exercises the same coupled
path used by a production RPMD driver: the bath force, ring-polymer springs,
centroid-dependent quantum potential, wavefunction propagation, and the
Ehrenfest force are advanced by one shared clock.
"""

import numpy as np
import pytest

from pyrpmd.analysis.quantum_joint import OnlineQuantumRPMD


def _gaussian(grid, center=0.25, width=0.8, momentum=0.15):
    psi = np.exp(-0.5 * ((grid - center) / width) ** 2 + 1j * momentum * grid)
    dx = grid[1] - grid[0]
    return psi / np.sqrt(dx * np.sum(np.abs(psi) ** 2))


def _runner(*, coupling=True, dt=2.0e-4):
    grid = np.linspace(-10.0, 10.0, 192, endpoint=False)
    positions = np.array(
        [[0.30, -0.15], [-0.05, 0.10], [-0.20, 0.05]], dtype=float
    )
    momenta = np.array(
        [[0.10, -0.04], [0.00, 0.02], [-0.06, 0.03]], dtype=float
    )
    masses = np.array([1.5, 2.0])
    coefficients = np.array([0.31, -0.22]) if coupling else np.zeros(2)
    stiffness = np.array([[0.8, 0.12], [0.12, 0.55]])

    def quantum_potential(x, centroid):
        return 0.5 * 0.7**2 * x**2 + np.dot(coefficients, centroid) * x

    def quantum_gradient(x, centroid):
        return coefficients[:, None] * x[None, :]

    def bath_force(q):
        return -(np.asarray(q) @ stiffness.T)

    def bath_potential(q):
        q = np.asarray(q)
        return 0.5 * float(q @ stiffness @ q)

    return OnlineQuantumRPMD(
        grid,
        quantum_mass=1.0,
        beta=5.0,
        dt=dt,
        positions=positions,
        momenta=momenta,
        bath_masses=masses,
        quantum_potential=quantum_potential,
        quantum_gradient=quantum_gradient,
        bath_force=bath_force,
        bath_potential=bath_potential,
        wavefunction=_gaussian(grid),
    )


def test_full_dimensional_joint_trajectory_updates_beads_and_wavefunction():
    runner = _runner()
    initial_positions = runner.positions.copy()
    initial_wavefunction = runner.wavefunction.copy()
    initial_centroid = initial_positions.mean(axis=0)
    result = runner.propagate(12, sample_stride=3)

    assert result.positions.shape == (5, 3, 2)
    assert result.momenta.shape == (5, 3, 2)
    assert result.wavefunctions.shape == (5, 192)
    assert result.quantum_force.shape == (5, 2)
    assert result.time[-1] == pytest.approx(12 * runner.dt)
    assert not np.array_equal(result.final_positions, initial_positions)
    assert not np.array_equal(result.final_wavefunction, initial_wavefunction)
    assert np.max(np.abs(result.wavefunction_norm - 1.0)) < 5.0e-10
    assert np.all(np.isfinite(result.total_energy))
    assert np.all(np.isfinite(result.final_positions))
    assert np.all(np.isfinite(result.final_momenta))
    assert not np.allclose(result.positions[-1].mean(axis=0), initial_centroid)

    # This is the actual multi-coordinate, multi-bead path rather than a
    # one-bead unit test.  The production scope remains explicit in metadata.
    assert result.diagnostics["environment_beads"] == 3
    assert result.diagnostics["bath_masses"] == [1.5, 2.0]
    assert result.diagnostics["quantum_force_distribution"] == "equal_over_beads"
    assert result.diagnostics["physical_converged"] is False


def test_ehrenfest_force_is_shared_across_beads_in_full_rpmd_step():
    """The centroid quantum force is distributed equally over every bead."""
    dt = 1.0e-7
    coupled = _runner(coupling=True, dt=dt)
    uncoupled = _runner(coupling=False, dt=dt)
    initial_force = coupled.propagate(0).quantum_force[0]
    coupled.propagate(1)
    uncoupled.propagate(1)

    delta_momenta = coupled.momenta - uncoupled.momenta
    expected = dt * initial_force / coupled.nbeads
    # At this small step the wavefunction changes only at O(dt), so the
    # difference isolates the equal beadwise Ehrenfest kick to first order.
    np.testing.assert_allclose(
        delta_momenta,
        np.broadcast_to(expected, delta_momenta.shape),
        rtol=2.0e-5,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(delta_momenta[0], delta_momenta[1], rtol=2.0e-5, atol=2.0e-12)
    np.testing.assert_allclose(delta_momenta[1], delta_momenta[2], rtol=2.0e-5, atol=2.0e-12)


def test_joint_trajectory_is_deterministic_and_resumable():
    first = _runner().propagate(8, sample_stride=2)
    second = _runner().propagate(8, sample_stride=2)
    for name in (
        "time", "positions", "momenta", "wavefunctions", "quantum_force",
        "total_energy", "wavefunction_norm",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    continuation = _runner()
    first_half = continuation.propagate(4)
    second_half = continuation.propagate(4)
    full = _runner().propagate(8)
    np.testing.assert_allclose(second_half.final_positions, full.final_positions, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(second_half.final_momenta, full.final_momenta, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(second_half.final_wavefunction, full.final_wavefunction, rtol=0.0, atol=1.0e-14)
    assert first_half.time[-1] == pytest.approx(4 * continuation.dt)
    assert second_half.time[-1] == pytest.approx(8 * continuation.dt)
