"""Regression tests for the path-times-transverse-channel Hamiltonian.

These tests deliberately use separable and translated harmonic potentials.  In
that limit the local transverse spectrum is known analytically, while a
translated oscillator exercises the phase matching between neighbouring
sigma slices without requiring a benchmark PES.
"""

import numpy as np
import pytest

from pyrpmd.analysis.subspace_quantum import (
    _GridOperators,
    _dense_rate,
    _dense_rate_matrix,
    _reactant_partition,
    absorbing_potential,
    sinc_dvr_kinetic,
)
from pyrpmd.analysis.subspace_model import model_from_function
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate


try:
    from pyrpmd.analysis.subspace_channels import (
        assemble_channel_hamiltonian,
        build_subspace_channels,
    )
except ImportError:  # pragma: no cover - gives a useful failure below
    assemble_channel_hamiltonian = None
    build_subspace_channels = None


MASS_Q = 1.0
OMEGA = 0.22


def _barrier(s):
    return 0.08 / np.cosh(np.asarray(s) / 0.8) ** 2


def _separable(points):
    points = np.asarray(points)
    s, q = points[..., 0], points[..., 1]
    return _barrier(s) + 0.5 * MASS_Q * OMEGA**2 * q**2


def _translated(points):
    points = np.asarray(points)
    s, q = points[..., 0], points[..., 1]
    centre = 0.35 * np.tanh(s / 1.2)
    return _barrier(s) + 0.5 * MASS_Q * OMEGA**2 * (q - centre) ** 2


def _make(potential, n_channels=4, *, curvature=None):
    sigma = np.linspace(-4.0, 4.0, 33)
    # The fourth oscillator state needs a box several oscillator widths wide;
    # the walls should not contaminate the local spectrum used by the test.
    q = np.linspace(-10.0, 10.0, 161)
    return build_subspace_channels(
        potential,
        sigma,
        q,
        n_channels=n_channels,
        masses=(1.0, MASS_Q),
        curvature=curvature,
        dividing_surface=0.0,
        reactant_reference=-2.0,
    )


@pytest.mark.skipif(build_subspace_channels is None, reason="subspace_channels is not available")
def test_separable_channels_have_known_local_levels_and_no_couplings():
    result = _make(_separable, n_channels=4, curvature=0.0)
    assert result.energies.shape == (33, 4)
    expected_transverse = (np.arange(4) + 0.5) * OMEGA
    assert np.allclose(result.energies - _barrier(result.sigma)[:, None], expected_transverse[None, :], atol=3e-4)

    # A separable oscillator has no derivative or curvature channel coupling.
    assert np.max(np.abs(result.derivative_couplings)) < 2e-5
    assert np.max(np.abs(result.curvature_couplings)) < 2e-5

    h = np.asarray(result.hamiltonian)
    assert h.shape == (33 * 4, 33 * 4)
    assert np.allclose(h, h.conj().T, atol=1e-11)
    # In the one-channel limit this is exactly the existing one-dimensional
    # sinc-DVR Hamiltonian with the transverse ground-state energy added.
    one = _make(_separable, n_channels=1, curvature=0.0)
    expected = sinc_dvr_kinetic(one.sigma, 1.0) + np.diag(_barrier(one.sigma) + 0.5 * OMEGA)
    assert np.allclose(np.asarray(one.hamiltonian), expected, atol=5e-7)
    uncoupled = build_subspace_channels(_separable, one.sigma, one.q, n_channels=1,
                                        masses=(1.0, MASS_Q), curvature=0.05,
                                        include_nonadiabatic=False,
                                        dividing_surface=0.0, reactant_reference=-2.0)
    assert np.allclose(uncoupled.hamiltonian, expected, atol=5e-7)

    # With one channel and zero curvature the dense channel rate is the
    # existing one-dimensional rate for the same effective potential.
    beta = 5.0
    channel_rate = one.exact_rate(beta)
    one_d = model_from_function(
        lambda points: _barrier(np.asarray(points)[..., 0]) + 0.5 * OMEGA,
        [one.sigma], masses=1.0, dividing_surface=0.0, reactant_reference=-2.0,
    )
    reference_rate = exact_quantum_rate(one_d, beta, method="sylvester")
    assert channel_rate.rate == pytest.approx(reference_rate.rate, rel=1e-12)
    assert channel_rate.diagnostics["normal_grid_points"] == 161
    assert channel_rate.diagnostics["truncation_margin"] > 2.0
    # A zero-curvature one-channel model keeps the established constant-mass
    # RPMD path; the metric hook is unnecessary when g^{sigma sigma}=1.
    assert one.channel_model().metric is None


@pytest.mark.skipif(build_subspace_channels is None, reason="subspace_channels is not available")
def test_translated_oscillator_has_continuous_phases_and_antisymmetric_derivative_coupling():
    result = _make(_translated, n_channels=4, curvature=0.0)
    states = np.asarray(result.states)
    # The phase convention is observable through adjacent-sigma overlaps.  All
    # diagonal overlaps must be positive after the sign matching step.
    overlaps = np.einsum("iqn,iqn->in", states[:-1].conj(), states[1:])
    assert np.all(np.real(overlaps) > 0.0)
    assert np.max(np.abs(np.imag(overlaps))) < 1e-12

    coupling = np.asarray(result.derivative_couplings)
    assert coupling.shape[1:] == (4, 4)
    assert np.allclose(coupling + np.swapaxes(coupling, -1, -2), 0.0, atol=3e-3)
    assert np.max(np.abs(np.diagonal(coupling, axis1=-2, axis2=-1))) < 3e-3


@pytest.mark.skipif(build_subspace_channels is None, reason="subspace_channels is not available")
def test_curvature_coupling_is_hermitian_and_assembly_is_deterministic():
    first = _make(_translated, n_channels=3, curvature=0.05)
    second = _make(_translated, n_channels=3, curvature=0.05)
    assert np.allclose(first.hamiltonian, second.hamiltonian, atol=1e-12)
    coupling = np.asarray(first.curvature_couplings)
    assert coupling.shape[1:] == (3, 3)
    assert np.allclose(coupling, np.swapaxes(coupling.conj(), -1, -2), atol=5e-8)
    assembled = assemble_channel_hamiltonian(first)
    assert np.allclose(assembled, first.hamiltonian, atol=1e-12)
    model = first.channel_model()
    points = np.array([[0.0, -0.4], [0.0, 0.7]])
    expected_metric = 1.0 / (1.0 + 0.05 * points[:, 1]) ** 2
    assert np.allclose(model.inverse_metric(points), expected_metric)
    assert model.describe()["position_dependent_metric"]
    # Curved exact matrices are available, but a matching constrained
    # variable-mass RPMD estimator has not been validated.
    from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

    with pytest.raises(ValueError, match="position-dependent metric RPMD"):
        ring_polymer_rate(model, beta=20.0, nbeads=2)
    with pytest.raises(ValueError, match="position-dependent metric"):
        exact_quantum_rate(model, beta=20.0)
    with pytest.raises(ValueError, match="flat reactant reference"):
        first.exact_rate(beta=20.0)


@pytest.mark.skipif(build_subspace_channels is None, reason="subspace_channels is not available")
def test_channel_count_and_grid_validation():
    sigma = np.linspace(-4.0, 4.0, 33)
    q = np.linspace(-10.0, 10.0, 161)
    with pytest.raises(ValueError, match="n_channels"):
        build_subspace_channels(_separable, sigma, q, n_channels=0)
    with pytest.raises(ValueError, match="uniform"):
        build_subspace_channels(_separable, sigma[[0, 1, 2, 4] + list(range(5, 33))], q, n_channels=2)


@pytest.mark.parametrize("dimension", [1, 2])
def test_dense_rate_matrix_refactor_matches_existing_wrapper(dimension):
    sigma = np.linspace(-3.0, 3.0, 17)
    if dimension == 1:
        model = model_from_function(
            lambda points: _barrier(np.asarray(points)[..., 0]), [sigma], masses=1.0,
            dividing_surface=0.0, reactant_reference=-1.5,
        )
    else:
        q = np.linspace(-1.0, 1.0, 8)
        model = model_from_function(
            _separable, [sigma, q], masses=(1.0, 1.0),
            dividing_surface=0.0, reactant_reference=-1.5,
        )
    beta = 5.0
    operators = _GridOperators(model)
    shift = float(operators.potential.min())
    eps = absorbing_potential(sigma, width=0.6, strength=0.2)
    eps_grid = eps if dimension == 1 else np.broadcast_to(eps[:, None], model.shape)
    old_flux, old_partition, _ = _dense_rate(model, operators, eps_grid, beta, shift)
    matrix_flux, matrix_partition, _ = _dense_rate_matrix(
        operators.dense_hamiltonian(), eps_grid.ravel(),
        operators.product.ravel(), beta, shift,
        partition=lambda energies, vectors: _reactant_partition(
            model, beta, shift, energies=energies, vectors=vectors, operators=operators,
        ),
    )
    assert matrix_flux == pytest.approx(old_flux, rel=1e-12, abs=1e-14)
    assert matrix_partition == pytest.approx(old_partition, rel=1e-12, abs=1e-14)
    assert (matrix_flux / matrix_partition) == pytest.approx(
        old_flux / old_partition, rel=1e-12, abs=1e-14,
    )



def test_full_channel_basis_is_unitarily_equivalent_to_same_two_dimensional_dvr():
    sigma = np.linspace(-3.0, 3.0, 9)
    q = np.linspace(-4.0, 4.0, 10)
    channels = build_subspace_channels(_translated, sigma, q, n_channels=len(q),
                                      masses=(2.0, 1.3), reactant_reference=-1.5)
    h_full = (np.kron(sinc_dvr_kinetic(sigma, 2.0), np.eye(len(q)))
              + np.kron(np.eye(len(sigma)), sinc_dvr_kinetic(q, 1.3))
              + np.diag(channels.potential_values.ravel()))
    transform = np.zeros_like(h_full, dtype=complex)
    for i, states in enumerate(channels.states):
        transform[i*len(q):(i+1)*len(q), i*len(q):(i+1)*len(q)] = states
    assert np.allclose(channels.hamiltonian, transform.conj().T @ h_full @ transform,
                       atol=2e-12, rtol=2e-12)
    assert np.allclose(np.linalg.eigvalsh(channels.hamiltonian), np.linalg.eigvalsh(h_full),
                       atol=2e-12, rtol=2e-12)


def test_truncated_channels_retain_out_of_subspace_derivative_contribution():
    sigma = np.linspace(-3.0, 3.0, 17)
    q = np.linspace(-8.0, 8.0, 65)
    channels = build_subspace_channels(_translated, sigma, q, n_channels=1,
                                      reactant_reference=-1.5)
    uncoupled = sinc_dvr_kinetic(sigma, 1.0) + np.diag(channels.energies[:, 0])
    # A varying ground-state basis has a positive derivative norm, even
    # though its retained diagonal derivative coupling is exactly zero.
    assert np.linalg.eigvalsh(channels.hamiltonian)[0] > np.linalg.eigvalsh(uncoupled)[0] + 1e-5


def test_phase_alignment_never_mixes_nondegenerate_eigenstates():
    from pyrpmd.analysis.subspace_channels import _align_phases

    states = np.array([np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])], dtype=complex)
    aligned = _align_phases(states)
    h = np.diag([2.0, 1.0])
    assert np.allclose(h @ aligned[1], aligned[1] * np.array([1.0, 2.0]))


def test_metric_singularity_between_grid_points_is_rejected():
    sigma = np.linspace(-3.0, 3.0, 9)
    q = np.linspace(-10.0, 10.0, 16)
    assert not np.any(np.isclose(1.0 + 0.15 * q, 0.0))
    with pytest.raises(ValueError, match="singular on the Q domain"):
        build_subspace_channels(_separable, sigma, q, n_channels=2, curvature=0.15)


def test_uncoupled_channels_do_not_advertise_matching_rpmd_potential():
    sigma = np.linspace(-3.0, 3.0, 9)
    q = np.linspace(-8.0, 8.0, 33)
    channels = build_subspace_channels(_translated, sigma, q, n_channels=2,
                                      include_nonadiabatic=False)
    with pytest.raises(ValueError, match="no matching"):
        channels.channel_model()
    with pytest.raises(ValueError, match="include_coupling must be boolean"):
        build_subspace_channels(_separable, sigma, q, include_coupling="false")


def test_path_normal_and_metric_use_the_same_signed_curvature():
    from pyrpmd.analysis.subspace_channels import project_path_channels

    angle = np.linspace(-0.4, 0.4, 101)
    path = 5.0 * np.column_stack((np.cos(angle), np.sin(angle)))
    sigma = np.linspace(0.1, 3.9, 9)
    q = np.linspace(-1.0, 1.0, 17)
    channels = project_path_channels(lambda x: np.sum(x*x, axis=-1), None,
                                     path, [1.0, 1.0], sigma, q_axis=q,
                                     n_channels=2, reactant_reference=0.5)
    # Frenet normal points towards the centre: positive Q shortens the arc.
    assert np.allclose(channels.curvature, -0.2, atol=1e-5)
    metric = channels.channel_model().inverse_metric(np.array([[2.0, 0.5]]))
    assert metric[0] == pytest.approx(1.0 / 0.9**2, rel=2e-5)


def test_zero_curvature_channel_model_retains_working_rpmd_rate():
    from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

    channels = build_subspace_channels(_separable, np.linspace(-3, 3, 17),
                                      np.linspace(-8, 8, 65), n_channels=2,
                                      reactant_reference=-1.5)
    model = channels.channel_model()
    assert model.metric is None
    result = ring_polymer_rate(model, beta=5.0, nbeads=1, windows=4, walkers=2,
                               burn_in=2, sweeps=2, leapfrog_steps=1, trajectories=4,
                               md_time=0.2, md_dt=0.02, bootstrap=2)
    assert np.isfinite(result.qtst_rate)
