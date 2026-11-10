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
                                        masses=(1.0, MASS_Q), curvature=0.0,
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


def test_frenet_jacobian_matches_cartesian_embedding_and_rejects_interior_pole():
    from pyrpmd.analysis.subspace_channels import _path_jacobian

    radius, normal = 10.0, 1.0
    # Inward Frenet normal: x(s,Q)=(R-Q)*(cos(s/R),sin(s/R)).
    h = 1e-4
    embedding = lambda s: (radius-normal)*np.array([np.cos(s/radius), np.sin(s/radius)])
    tangent = (embedding(h)-embedding(-h))/(2*h)
    expected_inverse = 1/(tangent@tangent)
    jacobian = _path_jacobian(np.array([1/radius]), np.array([-normal, normal]))
    assert 1/jacobian[0, 1]**2 == pytest.approx(expected_inverse, rel=1e-9)
    # Neither endpoint nor any grid node needs to hit the pole exactly.
    with pytest.raises(ValueError, match="singular metric"):
        _path_jacobian(np.array([.15]), np.linspace(-8, 8, 41))
    with pytest.raises(NotImplementedError, match="Cartesian"):
        _make(_translated, n_channels=3, curvature=.02)


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



def _cartesian_model(*, reactant="asymptotic", metric=None):
    sigma, normal = np.linspace(-3, 3, 11), np.linspace(-5, 5, 9)
    return model_from_function(_translated, [sigma, normal], masses=[1.7, 1.0],
                               dividing_surface=0., reactant_reference=-1.5,
                               reactant=reactant, metric=metric)


def _full_channel_embedding(channels):
    ns, nq, nc = channels.states.shape
    embedding = np.zeros((ns*nq, ns*nc), complex)
    for i in range(ns):
        embedding[i*nq:(i+1)*nq, i*nc:(i+1)*nc] = channels.states[i]
    return embedding


@pytest.mark.parametrize("count", [1, 2, 4, 6, 9])
def test_channels_are_galerkin_projection_of_the_same_cartesian_hamiltonian(count):
    from pyrpmd.analysis.subspace_channels import channels_from_model

    model = _cartesian_model()
    channels = channels_from_model(model, n_channels=count)
    embedding = _full_channel_embedding(channels)
    full = _GridOperators(model).dense_hamiltonian()
    oracle = embedding.conj().T @ full @ embedding
    np.testing.assert_allclose(channels.hamiltonian, oracle, atol=2e-13)
    assert channels.channel_model() is model
    if count == len(model.axes[1]):
        np.testing.assert_allclose(np.linalg.eigvalsh(channels.hamiltonian),
                                   np.linalg.eigvalsh(full), atol=2e-13)


def test_one_translated_channel_retains_born_huang_term():
    from pyrpmd.analysis.subspace_channels import channels_from_model

    model = _cartesian_model()
    channels = channels_from_model(model, n_channels=1)
    adiabatic_only = sinc_dvr_kinetic(channels.sigma, model.masses[0]) + np.diag(channels.energies[:, 0])
    assert np.max(np.abs(channels.hamiltonian-adiabatic_only)) > 1e-5
    assert np.linalg.eigvalsh(channels.hamiltonian)[0] > np.linalg.eigvalsh(adiabatic_only)[0]


def test_polar_alignment_does_not_rotate_nondegenerate_eigenstates():
    from pyrpmd.analysis.subspace_channels import _align_phases

    states = np.array([np.eye(2), [[0., 1.], [1., 0.]]], complex)
    aligned = _align_phases(states)
    energies = np.diag([1., 2.])
    for before, after in zip(states, aligned):
        np.testing.assert_allclose(before@energies@before.conj().T,
                                   after@energies@after.conj().T, atol=1e-14)


@pytest.mark.parametrize("reactant", ["asymptotic", "well"])
def test_complete_channels_reproduce_full_cartesian_flux_partition_and_rate(reactant):
    from pyrpmd.analysis.subspace_channels import channels_from_model

    model = _cartesian_model(reactant=reactant)
    channels = channels_from_model(model, n_channels=len(model.axes[1]))
    options = dict(cap_width=1., cap_strength=.2)
    beta = 5.
    actual = channels.exact_rate(beta, **options)
    expected = exact_quantum_rate(model, beta, method="sylvester", **options)
    assert actual.rate == pytest.approx(expected.rate, rel=2e-11)
    # The two implementations choose different harmless Boltzmann shifts.
    # Compare flux and partition after restoring a common zero.
    from scipy.linalg import eigvalsh
    channel_shift = channels.energies.min()
    full_shift = model.values.min()
    factor = np.exp(beta*(channel_shift-full_shift))
    assert actual.flux/factor == pytest.approx(expected.flux, rel=2e-11)
    assert actual.reactant_partition/factor == pytest.approx(expected.reactant_partition, rel=2e-11)
    assert actual.diagnostics["channel_basis_complete"]


def test_uncoupled_diagnostic_cannot_be_paired_with_cartesian_rpmd():
    model = _cartesian_model()
    channels = build_subspace_channels(model.energy, *model.axes, n_channels=2,
                                       masses=model.masses, include_nonadiabatic=False,
                                       dividing_surface=0., reactant_reference=-1.5)
    with pytest.raises(NotImplementedError, match="diagnostic only"):
        channels.exact_rate(5.)
    with pytest.raises(NotImplementedError, match="diagnostic only"):
        channels.channel_model()


def test_constant_metric_has_the_same_quantum_operator_partition_and_rpmd_mass():
    from dataclasses import replace
    from pyrpmd.analysis.subspace_rpmd import _RingPolymer, ring_polymer_rate

    base = _cartesian_model()
    g = .25
    metric = replace(base, masses=np.array([3., 2.]), metric=np.full(base.shape, g))
    transformed = replace(metric, masses=np.array([12., 2.]), metric=None)
    np.testing.assert_allclose(_GridOperators(metric).dense_hamiltonian(),
                               _GridOperators(transformed).dense_hamiltonian(), atol=1e-14)
    assert _RingPolymer(metric, 5., 2).mass.tolist() == [12., 2.]
    actual = exact_quantum_rate(metric, 5., method="sylvester")
    expected = exact_quantum_rate(transformed, 5., method="sylvester")
    assert actual.rate == pytest.approx(expected.rate, rel=1e-13)
    assert actual.reactant_partition == pytest.approx(expected.reactant_partition, rel=1e-13)
    options = dict(windows=4, walkers=2, burn_in=1, sweeps=2, leapfrog_steps=1,
                   trajectories=2, hmc_dt=.01, md_dt=.01, md_time=.1, bootstrap=0, seed=71)
    actual_rpmd = ring_polymer_rate(metric, 5., 2, **options)
    expected_rpmd = ring_polymer_rate(transformed, 5., 2, **options)
    np.testing.assert_array_equal(actual_rpmd.mean_force, expected_rpmd.mean_force)
    assert actual_rpmd.qtst_rate == expected_rpmd.qtst_rate
    assert actual_rpmd.rate == expected_rpmd.rate


@pytest.mark.parametrize("kind", ["callable", "grid"])
def test_variable_metric_cannot_silently_reach_quantum_or_rpmd_or_correction(kind):
    from dataclasses import replace
    from pyrpmd.analysis.subspace_correction import subspace_correction
    from pyrpmd.analysis.subspace_rpmd import _RingPolymer, ring_polymer_rate

    base = _cartesian_model()
    metric = lambda x: 1/(1+.2*np.asarray(x)[...,0])**2
    model = replace(base, metric=metric if kind=="callable" else metric(base.grid_points()))
    for call in [lambda: _GridOperators(model), lambda: exact_quantum_rate(model, 5.),
                 lambda: _RingPolymer(model, 5., 2), lambda: ring_polymer_rate(model, 5., 2),
                 lambda: subspace_correction(model, 5., 2)]:
        with pytest.raises(NotImplementedError, match="[Pp]osition-dependent"):
            call()


def test_constant_metric_propagation_is_reversible_and_volume_preserving():
    from dataclasses import replace
    from pyrpmd.analysis.subspace_rpmd import _RingPolymer

    base = _cartesian_model()
    model = replace(base, metric=np.full(base.shape, .25))
    ring = _RingPolymer(model, 5., 1)
    def step(z, dt):
        modes, momenta = z[:2].reshape(1,1,2), z[2:].reshape(1,1,2)
        energy, gradient = ring.potential(modes)
        out = ring.step(modes, momenta, -ring.modes(gradient), dt,
                        ring.free_propagator(dt), False)
        return np.r_[out[0].ravel(), out[1].ravel()]
    z = np.array([.3,.2,.8,.1]);dt=.1;h=1e-5
    np.testing.assert_allclose(step(step(z,dt),-dt), z, atol=1e-12)
    jac = np.column_stack([(step(z+h*e,dt)-step(z-h*e,dt))/(2*h) for e in np.eye(4)])
    assert np.linalg.det(jac) == pytest.approx(1., abs=1e-9)


def test_cartesian_channel_correction_uses_original_rpmd_model(monkeypatch):
    from pyrpmd.analysis.subspace_channels import channels_from_model
    import pyrpmd.analysis.subspace_rpmd as rpmd

    model = _cartesian_model()
    channels = channels_from_model(model, n_channels=4)
    real_run = rpmd.ring_polymer_rate
    seen=[]
    def run(received, *args, **kwargs):
        seen.append(received)
        return real_run(received, *args, **kwargs)
    monkeypatch.setattr(rpmd, "ring_polymer_rate", run)
    result = channels.correction(5., 2, rpmd_options=dict(windows=4,walkers=2,burn_in=1,
        sweeps=2,leapfrog_steps=1,trajectories=2,hmc_dt=.01,md_dt=.01,md_time=.1,bootstrap=0,seed=71))
    assert seen == [model]
    assert result.factor > 0
    assert result.diagnostics["channel_truncation_converged"] is False
    assert result.to_dict()["physical_converged"] is False


def test_complete_truncated_channels_match_cartesian_slice_rate_and_projected_absorber():
    from pyrpmd.analysis.subspace_channels import channels_from_model

    model = _cartesian_model()
    cutoff = .35
    channels = channels_from_model(model, n_channels=len(model.axes[1]), truncation_energy=cutoff)
    embedding = _full_channel_embedding(channels)[:, channels.active_channels.ravel()]
    retained = np.flatnonzero(model.values.ravel() <= cutoff)
    assert channels.size == len(retained) < np.prod(model.shape)
    subspace = embedding[retained]
    oracle = _GridOperators(model).dense_hamiltonian(retained)
    np.testing.assert_allclose(channels.hamiltonian, subspace.conj().T@oracle@subspace, atol=3e-13)
    def absorber(points):
        points = np.asarray(points)
        return .15 * np.maximum(abs(points[:,0])-1.,0.)**2 * (1+.04*points[:,1]**2)
    options = dict(absorber=absorber, normalization="slice", energy_floor=.1)
    actual = channels.exact_rate(5., **options)
    expected = exact_quantum_rate(model, 5., method="sylvester", truncation_energy=cutoff, **options)
    assert actual.rate == pytest.approx(expected.rate, rel=3e-11)
    shift = np.min(channels.energies[channels.active_channels])-model.values.min()
    factor = np.exp(5.*shift)
    assert actual.flux/factor == pytest.approx(expected.flux, rel=3e-11)
    assert actual.reactant_partition/factor == pytest.approx(expected.reactant_partition, rel=3e-11)
    assert actual.diagnostics["channel_basis_complete"]
    # The public rate entry can build the same truncated basis from its options.
    untruncated = channels_from_model(model, n_channels=len(model.axes[1]))
    rebuilt = untruncated.exact_rate(5., truncation_energy=cutoff, **options)
    assert rebuilt.rate == pytest.approx(actual.rate, rel=1e-13)


def test_channel_correction_accepts_fine_same_potential_and_rejects_changed_hamiltonian(monkeypatch):
    from dataclasses import replace
    from pyrpmd.analysis.subspace_channels import channels_from_model
    import pyrpmd.analysis.subspace_rpmd as rpmd

    model = _cartesian_model()
    fine_axes = [np.linspace(a[0], a[-1], 2*len(a)-1) for a in model.axes]
    fine = model_from_function(_translated, fine_axes, masses=model.masses,
                               dividing_surface=model.dividing_surface,
                               reactant_reference=model.reactant_reference)
    channels = channels_from_model(model, n_channels=4)
    observed=[]
    original = rpmd.ring_polymer_rate
    def run(received, *args, **kwargs):
        observed.append(received)
        return original(received, *args, **kwargs)
    monkeypatch.setattr(rpmd, "ring_polymer_rate", run)
    options=dict(windows=4,walkers=2,burn_in=1,sweeps=2,leapfrog_steps=1,
                  trajectories=2,hmc_dt=.01,md_dt=.01,md_time=.1,bootstrap=0,seed=71)
    result=channels.correction(5.,2,rpmd_model=fine,rpmd_options=options)
    assert result.factor > 0 and observed == [fine]
    wrong = replace(fine, masses=np.array([2.,1.]))
    with pytest.raises(ValueError, match="same masses"):
        channels.correction(5.,2,rpmd_model=wrong,rpmd_options=options)
    wrong = model_from_function(lambda x:_translated(x)+.01, fine_axes,masses=model.masses,
                                dividing_surface=0.,reactant_reference=-1.5)
    with pytest.raises(ValueError, match="potential differs"):
        channels.correction(5.,2,rpmd_model=wrong,rpmd_options=options)


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "99", None, np.inf])
def test_channel_dense_limit_requires_positive_integer(limit):
    from pyrpmd.analysis.subspace_channels import channels_from_model

    channels = channels_from_model(_cartesian_model(), n_channels=2)
    with pytest.raises(ValueError, match="dense_limit must be a positive integer"):
        channels.exact_rate(5., dense_limit=limit)


def test_channel_dense_limit_is_strict_and_applies_after_energy_truncation(monkeypatch):
    import pyrpmd.analysis.subspace_channels as channel_module

    model = _cartesian_model()
    channels = channel_module.channels_from_model(model, n_channels=9)
    real_solver = channel_module._dense_rate_matrix
    invoked = []

    def solve(hamiltonian, *args, **kwargs):
        invoked.append(len(hamiltonian))
        return real_solver(hamiltonian, *args, **kwargs)

    monkeypatch.setattr(channel_module, "_dense_rate_matrix", solve)
    with pytest.raises(ValueError, match="size 99 exceeds dense_limit=98"):
        channels.exact_rate(5., dense_limit=98)
    assert invoked == []
    truncated = channels.exact_rate(5., dense_limit=63, truncation_energy=.35)
    assert truncated.diagnostics["active_channel_dimension"] == 63
    assert invoked == [63]
    with pytest.raises(ValueError, match="size 63 exceeds dense_limit=62"):
        channels.exact_rate(5., dense_limit=62, truncation_energy=.35)
    assert invoked == [63]
