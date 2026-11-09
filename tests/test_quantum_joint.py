import numpy as np
import pytest

from pyrpmd.analysis.quantum_joint import OnlineQuantumRPMD


def gaussian(grid, center=0.0, width=1.0):
    psi = np.exp(-0.5 * ((grid - center) / width) ** 2).astype(complex)
    dx = grid[1] - grid[0]
    return psi / np.sqrt(dx * np.sum(np.abs(psi) ** 2))


def test_online_joint_decoupled_norm_and_bath_state():
    grid = np.linspace(-8.0, 8.0, 128, endpoint=False)
    psi = gaussian(grid, center=0.3, width=0.8)
    omega = 1.0
    runner = OnlineQuantumRPMD(
        grid, 1.0, 4.0, 0.002, [[1.2]], [[0.4]], [2.0],
        lambda x, q: 0.5 * omega**2 * x**2,
        quantum_gradient=lambda x, q: np.zeros((1, len(x))),
        bath_force=lambda q: np.zeros_like(q),
        wavefunction=psi,
    )
    result = runner.propagate(40, sample_stride=10)
    assert result.positions.shape == (5, 1, 1)
    # With no bath or quantum force, the one-bead RPMD coordinate is free and
    # follows the exact drift q(t)=q(0)+p(0)t/m.
    expected_time = result.time
    assert np.allclose(result.positions[:, 0, 0], 1.2 + 0.4 * expected_time / 2.0)
    assert np.allclose(result.momenta[:, 0, 0], 0.4)
    assert np.max(np.abs(result.wavefunction_norm - 1.0)) < 5e-10
    assert result.diagnostics["centroid_coupling"] is True
    assert result.physical_converged is False


def test_online_joint_linear_coupling_force_and_finite_difference():
    grid = np.linspace(-10.0, 10.0, 192, endpoint=False)
    psi = gaussian(grid, center=0.6, width=0.55)
    coupling = 0.35
    harmonic = 0.5 * 0.9**2 * grid**2

    def potential(x, q):
        return harmonic + coupling * q[0] * x

    def bath_force(q):
        return -0.7**2 * q

    runner = OnlineQuantumRPMD(
        grid, 1.0, 5.0, 0.001, [[0.8]], [[0.0]], [1.0], potential,
        bath_force=bath_force, wavefunction=psi,
    )
    expected = -coupling * (grid[1] - grid[0]) * np.sum(np.abs(psi) ** 2 * grid)
    result = runner.propagate(3)
    # The first recorded force is the force of the initial quantum state,
    # before any bath displacement.  The finite-difference gradient path is
    # intentionally exercised by omitting quantum_gradient above.
    assert np.isclose(result.quantum_force[0, 0], expected, rtol=0.0, atol=3e-5)
    assert np.all(np.isfinite(result.total_energy))
    assert result.positions.shape[1:] == (1, 1)


def test_online_joint_ring_polymer_force_distribution_and_resume():
    grid = np.linspace(-7.0, 7.0, 96, endpoint=False)
    psi = gaussian(grid, width=0.9)
    p0 = np.zeros((4, 1))
    q0 = np.array([[0.0], [0.2], [0.0], [-0.2]])
    runner = OnlineQuantumRPMD(
        grid, 1.0, 3.0, 0.001, q0, p0, [1.0],
        lambda x, q: 0.25 * x**2 + 0.15 * q[0] * x,
        quantum_gradient=lambda x, q: np.full((1, len(x)), 0.15 * x),
        wavefunction=psi,
    )
    first = runner.propagate(2)
    second = runner.propagate(2)
    assert first.time[-1] == pytest.approx(0.002)
    assert second.time[0] == pytest.approx(0.002)
    assert second.time[-1] == pytest.approx(0.004)
    assert second.quantum_force.shape == (3, 1)
    assert np.all(np.isfinite(second.positions))


def test_online_joint_input_validation():
    grid = np.linspace(-4.0, 4.0, 32, endpoint=False)
    psi = gaussian(grid)
    with pytest.raises(ValueError, match="wavefunction is required"):
        OnlineQuantumRPMD(
            grid, 1.0, 2.0, 0.01, [[0.0]], [[0.0]], [1.0],
            lambda x, q: np.zeros_like(x),
        )
    with pytest.raises(ValueError, match="unit dx-weighted norm"):
        OnlineQuantumRPMD(
            grid, 1.0, 2.0, 0.01, [[0.0]], [[0.0]], [1.0],
            lambda x, q: np.zeros_like(x), wavefunction=np.ones_like(grid),
        )


def test_online_joint_step_rolls_back_on_callback_failure():
    grid = np.linspace(-4.0, 4.0, 64, endpoint=False)
    psi = gaussian(grid, width=0.8)
    calls = {"count": 0}

    def potential(x, q):
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("intentional callback failure")
        return np.zeros_like(x)

    runner = OnlineQuantumRPMD(
        grid, 1.0, 2.0, 0.01, [[0.5]], [[0.1]], [1.0],
        potential, quantum_gradient=lambda x, q: np.zeros((1, len(x))), wavefunction=psi,
    )
    initial_q = runner.positions.copy()
    initial_p = runner.momenta.copy()
    initial_psi = runner.wavefunction.copy()
    with pytest.raises(RuntimeError, match="intentional callback failure"):
        runner.step()
    assert runner.time == 0.0
    assert np.array_equal(runner.positions, initial_q)
    assert np.array_equal(runner.momenta, initial_p)
    assert np.array_equal(runner.wavefunction, initial_psi)


def test_online_joint_ring_polymer_uses_standard_rpmd_forces_and_conserves_energy():
    """P > 1: unscaled physical force per bead, full centroid force, conserved H_P."""
    grid = np.linspace(-7.0, 7.0, 128, endpoint=False)
    psi = gaussian(grid, center=0.2, width=0.8)
    omega_bath, coupling = 0.8, 0.1
    q0 = np.array([[0.3], [0.5], [0.1], [-0.2]])
    runner = OnlineQuantumRPMD(
        grid, 1.0, 2.0, 0.002, q0, np.zeros_like(q0), [1.5],
        lambda x, q: 0.5 * x**2 + coupling * q[0] * x,
        quantum_gradient=lambda x, q: np.full((1, len(x)), coupling * x),
        bath_force=lambda q: -1.5 * omega_bath**2 * q,
        bath_potential=lambda q: 0.75 * omega_bath**2 * float(q[0] ** 2),
        wavefunction=psi,
    )
    force, qforce = runner._forces(runner.positions, runner.wavefunction)
    spring = runner._spring_force(runner.positions)
    expected = spring - 1.5 * omega_bath**2 * q0 + qforce[None, :]
    assert np.allclose(force, expected)
    result = runner.propagate(2000, sample_stride=200)
    drift = np.max(np.abs(result.total_energy - result.total_energy[0]))
    assert drift < 2e-5 * max(1.0, abs(result.total_energy[0]))
