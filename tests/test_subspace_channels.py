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
                                        masses=(1.0, MASS_Q), curvature=0.15,
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
    first = _make(_translated, n_channels=3, curvature=0.15)
    second = _make(_translated, n_channels=3, curvature=0.15)
    assert np.allclose(first.hamiltonian, second.hamiltonian, atol=1e-12)
    coupling = np.asarray(first.curvature_couplings)
    assert coupling.shape[1:] == (3, 3)
    assert np.allclose(coupling, np.swapaxes(coupling.conj(), -1, -2), atol=5e-8)
    assembled = assemble_channel_hamiltonian(first)
    assert np.allclose(assembled, first.hamiltonian, atol=1e-12)
    model = first.channel_model()
    points = np.array([[0.0, -0.4], [0.0, 0.7]])
    expected_metric = 1.0 / (1.0 + 0.15 * points[:, 1]) ** 2
    assert np.allclose(model.inverse_metric(points), expected_metric)
    assert model.describe()["position_dependent_metric"]
    # The metric Jacobian is evaluated per bead by the RPMD potential.  A
    # direct small-batch call catches accidental reduction over beads before
    # broadcasting the correction back onto the energy array.
    from pyrpmd.analysis.subspace_rpmd import _RingPolymer

    ring = _RingPolymer(model, beta=20.0, nbeads=2)
    energy, gradient = ring.potential(np.zeros((3, 2, 2)))
    assert energy.shape == (3,)
    assert gradient.shape == (3, 2, 2)

    modes = np.zeros((3, 2, 2))
    modes[:, 0, 0] = np.array([-0.8, 0.0, 0.8])
    modes[:, 1, 1] = np.array([0.2, -0.3, 0.1])
    beads = ring.beads(modes)
    coordinate_mass = np.ones_like(beads)
    coordinate_mass[..., 0] = 1.0 / model.inverse_metric(beads)
    difference = np.roll(beads, -1, axis=1) - beads
    edge_mass = 0.5 * (coordinate_mass + np.roll(coordinate_mass, -1, axis=1))
    expected_spring = 0.5 / ring.beta_p**2 * np.sum(edge_mass * difference**2, axis=(1, 2))
    assert np.allclose(ring.spring(modes), expected_spring)
    momenta = ring.sample_momenta(np.random.default_rng(4), 3, False, modes)
    expected_kinetic = 0.5 * np.sum(ring.beads(momenta)**2 / coordinate_mass, axis=(1, 2))
    assert np.allclose(ring.kinetic(momenta, modes), expected_kinetic)
    stepped = ring.step(modes, momenta, np.zeros_like(modes), 1e-3, None, False)
    assert all(np.isfinite(value).all() for value in stepped[:2])


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

