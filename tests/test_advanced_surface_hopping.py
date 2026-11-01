"""Independent analytic, representation and conservation checks of new kernels."""
import copy
import json

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.advanced_surface_hopping import (
    AdvancedRPSHDynamics, ComplexRPSHDynamics, RPSHBADynamics, RPSHDMDynamics,
    RPSHBAState, RPSHDMState, berry_curvature, complex_rescaling_direction,
)
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.surface_hopping import RPSHDynamics


class LinearTwoState:
    """H = (offset+a.x) sigma_z + g sigma_x - b*y sigma_y."""
    def __init__(self, a=(.7,), g=.4, b=0., offset=.2):
        self.a = np.asarray(a)
        self.g, self.b, self.offset = g, b, offset

    def evaluate(self, q):
        x = q[:, :, 0]@self.a+self.offset
        v = np.zeros((len(q), 2, 2), dtype=complex)
        gradient = np.zeros((len(q), 2, 2, q.shape[1], 3), dtype=complex)
        v[:, 0, 0], v[:, 1, 1] = x, -x
        v[:, 0, 1] = self.g+1j*self.b*q[:, 0, 1]
        v[:, 1, 0] = v[:, 0, 1].conj()
        gradient[:, 0, 0, :, 0] = self.a
        gradient[:, 1, 1, :, 0] = -self.a
        gradient[:, 0, 1, 0, 1], gradient[:, 1, 0, 0, 1] = 1j*self.b, -1j*self.b
        return DiabaticResult(v, gradient)


class RotatingSurfaces:
    """Constant eigenvalues +/-1/2, real eigenvectors rotated by theta=x."""
    def evaluate(self, q):
        x = q[:, 0, 0]
        v = np.zeros((len(q), 2, 2))
        gradient = np.zeros((len(q), 2, 2, 1, 3))
        v[:, 0, 0], v[:, 1, 1] = -.5*np.cos(2*x), .5*np.cos(2*x)
        v[:, 0, 1] = v[:, 1, 0] = -.5*np.sin(2*x)
        gradient[:, 0, 0, 0, 0], gradient[:, 1, 1, 0, 0] = np.sin(2*x), -np.sin(2*x)
        gradient[:, 0, 1, 0, 0] = gradient[:, 1, 0, 0, 0] = -np.cos(2*x)
        return DiabaticResult(v, gradient)


class DiagonalQuadratic:
    def evaluate(self, q):
        x = q[:, 0, 0]
        v = np.zeros((len(q), 2, 2))
        gradient = np.zeros((len(q), 2, 2, 1, 3))
        v[:, 0, 0], v[:, 1, 1] = -(1+x*x), 1+x*x
        gradient[:, 0, 0, 0, 0], gradient[:, 1, 1, 0, 0] = -2*x, 2*x
        return DiabaticResult(v, gradient)


def nuclear(xs, momenta=None, natoms=1):
    q = np.zeros((len(xs), natoms, 3))
    q[:, 0, 0] = xs
    p = np.zeros_like(q)
    if momenta is not None:
        p[:, 0, 0] = momenta
    return State(q, p)


def driver(method="dm", potential=None, nbeads=1, dt=.002, **kwargs):
    return AdvancedRPSHDynamics(potential or LinearTwoState(), [2.], 3., nbeads, dt, 2,
                               method=method, native=False, seed=72, **kwargs)


@pytest.mark.parametrize("method", ["ca", "ba", "dm"])
def test_state_roundtrip_and_detached_copy(method):
    dynamics = driver(method, nbeads=3)
    state = dynamics.initialize(nuclear([-.2, .1, .4]), adiabatic_coefficients=[.6, .8j])
    restored = type(state).from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.to_dict() == state.to_dict()
    clone = state.copy()
    clone.coefficients.flat[0] *= -1
    assert not np.shares_memory(state.coefficients, clone.coefficients)
    if method == "ba":
        assert not np.shares_memory(state.basis_vectors, clone.basis_vectors)
    assert np.trace(dynamics.density_matrix(state)) == pytest.approx(1.)
    assert np.sum(dynamics.populations(state)) == pytest.approx(1.)


def test_dm_dephases_by_bead_phase_spread_where_ba_remains_pure():
    c = np.array([1., 1.])/np.sqrt(2)
    states, densities = {}, {}
    for method in ("ca", "ba", "dm"):
        dynamics = driver(method, DiagonalQuadratic(), 2, .6, electronic_substeps=3)
        state = dynamics.initialize(nuclear([0., 1.]), adiabatic_coefficients=c)
        state.coefficients[:], _, frames = dynamics._electronic_step(state, state.q)
        if frames is not None:
            state.basis_vectors[:] = frames
        states[method] = state
        densities[method] = dynamics.density_matrix(state)
    expected = np.mean([np.outer(c*np.exp(-1j*.6*np.array([-a, a])),
                                (c*np.exp(-1j*.6*np.array([-a, a]))).conj())
                        for a in (1., 2.)], axis=0)
    np.testing.assert_allclose(densities["dm"], expected, atol=2e-15)
    assert np.trace(densities["dm"]@densities["dm"]).real == pytest.approx(.5+.5*np.cos(.6)**2)
    assert np.trace(densities["ba"]@densities["ba"]).real == pytest.approx(1.)
    assert np.linalg.norm(densities["ba"]-densities["ca"]) > .1
    np.testing.assert_allclose(np.sum(abs(states["dm"].coefficients)**2, axis=1), 1.)


def test_dm_current_includes_bead_hamiltonian_in_common_centroid_basis():
    dynamics = driver("dm", nbeads=2, dt=1e-6, electronic_substeps=2)
    state = dynamics.initialize(nuclear([-.5, .7]))
    state.coefficients[:] = np.array([[1., 1j], [1., -1j]])/np.sqrt(2.)
    v = dynamics.potential.evaluate(state.q).matrix
    rho_b = np.einsum("bi,bj->bij", state.coefficients, state.coefficients.conj())
    derivative = (-1j*(v@rho_b-rho_b@v)).mean(axis=0)
    _, u = np.linalg.eigh(dynamics.potential.evaluate(state.q.mean(axis=0, keepdims=True)).matrix[0])
    dp = np.diag(u.conj().T@derivative@u).real
    state.active_surface = int(np.argmin(dp))
    active = state.active_surface
    population = dynamics.populations(state)[active]
    z, probability, _ = dynamics._electronic_step(state, state.q)
    assert probability[1-active] > 1e-8
    assert probability[1-active]/dynamics.dt == pytest.approx(-dp[active]/population, rel=2e-6)
    exact = np.stack([expm(-1j*dynamics.dt*vb)@cb for vb, cb in zip(v, state.coefficients)])
    np.testing.assert_allclose(z, exact, atol=1e-14)


def test_dm_current_includes_motion_of_the_centroid_basis():
    dynamics = driver("dm", nbeads=2, dt=1e-7, electronic_substeps=2)
    state = dynamics.initialize(nuclear([.2, .5]))
    state.coefficients[:] = np.array([[.8, .6j], [.6j, .8]])
    velocity = np.zeros_like(state.q)
    velocity[:, 0, 0] = [.4, .6]
    v = dynamics.potential.evaluate(state.q).matrix
    _, u0 = np.linalg.eigh(dynamics.potential.evaluate(state.q.mean(axis=0, keepdims=True)).matrix[0])
    def populations_at(t):
        _, u = np.linalg.eigh(dynamics.potential.evaluate((state.q+t*velocity).mean(axis=0, keepdims=True)).matrix[0])
        overlap = np.diag(u0.conj().T@u)
        u *= (overlap.conj()/abs(overlap))[None]
        z = np.stack([expm(-1j*t*vb)@c for vb, c in zip(v, state.coefficients)])
        return abs(z@u.conj())**2
    difference_step = 1e-5
    derivative = (populations_at(difference_step)-populations_at(-difference_step)).mean(axis=0)/(2*difference_step)
    state.active_surface = int(np.argmin(derivative))
    active = state.active_surface
    _, probability, _ = dynamics._electronic_step(state, state.q+dynamics.dt*velocity)
    assert probability[1-active]/dynamics.dt == pytest.approx(
        -derivative[active]/dynamics.populations(state)[active], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("nbeads", [1, 4])
def test_identical_beads_dm_reduces_to_centroid_electronics(nbeads):
    ca, dm = driver("ca", nbeads=nbeads), driver("dm", nbeads=nbeads)
    ns = nuclear([.1]*nbeads, [.5]*nbeads)
    sc, sd = ca.initialize(ns, adiabatic_coefficients=[.8, .6j]), dm.initialize(ns, adiabatic_coefficients=[.8, .6j])
    for _ in range(5):
        ca.step(sc)
        dm.step(sd)
    np.testing.assert_allclose(sc.q, sd.q, atol=2e-14)
    np.testing.assert_allclose(sc.p, sd.p, atol=2e-14)
    np.testing.assert_allclose(ca.density_matrix(sc), dm.density_matrix(sd), atol=2e-14)
    np.testing.assert_allclose(ca.last_probabilities, dm.last_probabilities, atol=2e-14)
    assert sc.active_surface == sd.active_surface


def test_p1_ba_matches_independent_constant_adiabatic_generator():
    dynamics = driver("ba", RotatingSurfaces(), dt=.06, electronic_substeps=3)
    state = dynamics.initialize(nuclear([.1], [.6]), adiabatic_coefficients=[.8, .6j])
    new_q = state.q.copy()
    new_q[:, 0, 0] += .3*dynamics.dt
    z, _, _ = dynamics._electronic_step(state, new_q)
    # U(theta) columns; d01=-1, d10=1.  The initial sign convention is fixed
    # by the positive largest components and agrees with this analytic U.
    generator = np.array([[-.5, .3j], [-.3j, .5]])
    expected = expm(-1j*dynamics.dt*generator)@state.coefficients
    np.testing.assert_allclose(z, expected, atol=2e-15)


def test_ba_uses_internal_bead_velocities_not_only_centroid_velocity():
    # Nonlinear NAC makes the mean of v_b d_b nonzero despite v_centroid=0.
    dynamics = driver("ba", nbeads=2, dt=.001, electronic_substeps=8)
    state = dynamics.initialize(nuclear([-.4, .6]), adiabatic_coefficients=[.8, .6])
    new_q = state.q.copy()
    new_q[:, 0, 0] += np.array([1., -1.])*dynamics.dt
    z, _, _ = dynamics._electronic_step(state, new_q)
    assert abs(abs(z[0])**2-abs(state.coefficients[0])**2) > 1e-5
    centroid = driver("ca", nbeads=2, dt=.001)
    cstate = centroid.initialize(nuclear([-.4, .6]), adiabatic_coefficients=[.8, .6])
    cz, cp, _ = centroid._electronic_step(cstate, new_q)
    np.testing.assert_allclose(cp, 0., atol=1e-14)
    assert np.linalg.norm(cz-cstate.coefficients) > 0.  # Phase still evolves.


def test_p1_advanced_ca_reproduces_existing_ca_without_extensions():
    potential = LinearTwoState()
    new = driver("ca", potential, rescaling="centroid", geometric_force=False)
    old = RPSHDynamics(potential, [2.], 3., 1, .002, 2, native=False, seed=72, rescaling="ring")
    state = new.initialize(nuclear([.2], [.5]))
    original = state.copy()
    for _ in range(5):
        new.step(state)
        old.step(original)
    np.testing.assert_allclose(state.q, original.q, atol=1e-14)
    np.testing.assert_allclose(state.p, original.p, atol=1e-14)
    np.testing.assert_allclose(state.coefficients, original.coefficients, atol=1e-14)
    np.testing.assert_allclose(new.last_probabilities, old.last_probabilities, atol=1e-14)


def test_dboc_mass_weighted_energy_and_gradient_match_analytic_expression():
    potential = LinearTwoState(a=(.7, -.3), offset=0.)
    masses = np.array([2., 5.])
    dynamics = AdvancedRPSHDynamics(potential, masses, 3., 3, .001, 2, native=False,
                                   dboc=True, dboc_step=2e-5)
    q = nuclear([-.3, .1, .7], natoms=2).q
    q[:, 1, 0] = [.2, -.1, .3]
    data = dynamics.surface_data(q)
    x = q[:, :, 0]@potential.a
    r = np.sqrt(x*x+potential.g**2)
    weighted = np.sum(potential.a**2/masses)
    correction = potential.g**2*weighted/(8*r**4)
    dc_dx = -.5*potential.g**2*weighted*x/r**6
    expected_gradient = np.array([-1., 1.])[None, :, None]*(x/r)[:, None, None]*potential.a[None, None, :]
    expected_gradient += dc_dx[:, None, None]*potential.a[None, None, :]
    np.testing.assert_allclose(data.dboc, np.broadcast_to(correction[:, None], (3, 2)), rtol=2e-14)
    np.testing.assert_allclose(data.gradients[..., 0], expected_gradient, atol=3e-9)
    np.testing.assert_allclose(data.gradients[..., 1:], 0., atol=1e-14)
    np.testing.assert_allclose(data.energies, r[:, None]*[-1., 1.]+correction[:, None], atol=1e-14)


class ThreeStateLinear:
    def evaluate(self, q):
        coupling = np.array([[0., .3, .15], [.3, 0., .5], [.15, .5, 0.]])
        v = np.diag([-1., .4, 2.])[None]+q[:, 0, 0, None, None]*coupling[None]
        g = np.zeros((len(q), 3, 3, 1, 3))
        g[:, :, :, 0, 0] = coupling
        return DiabaticResult(v, g)


def test_three_state_dboc_changes_hop_gaps_and_electronic_phases_consistently():
    dynamics = AdvancedRPSHDynamics(ThreeStateLinear(), [2.], 3., 2, .01, 3,
                                   method="dm", native=False, dboc=True)
    state = dynamics.initialize(nuclear([.1, .3], [10., 9.]),
                                adiabatic_coefficients=np.ones(3)/np.sqrt(3))
    data = dynamics.surface_data(state.q)
    assert abs(np.sum(data.dboc[:, 2]-data.dboc[:, 0])) > 1e-3
    effective = np.einsum("bsi,bi,bti->bst", data.vectors, data.energies, data.vectors.conj())
    expected = np.stack([expm(-1j*dynamics.dt*ham)@c for ham, c in zip(effective, state.coefficients)])
    z, _, _ = dynamics._electronic_step(state, state.q)
    np.testing.assert_allclose(z, expected, atol=3e-14)
    kinetic = np.sum(state.p**2)/4.
    energy = dynamics.energy(state)
    gap = np.sum(data.energies[:, 2]-data.energies[:, 0])
    result = dynamics.attempt_hop(state, 2)
    assert result.accepted
    assert np.sum(state.p**2)/4.-kinetic == pytest.approx(-gap, abs=4e-14)
    assert dynamics.energy(state) == pytest.approx(energy, abs=4e-14)


@pytest.mark.parametrize("method", ["ca", "ba", "dm"])
@pytest.mark.parametrize("rescaling", ["centroid", "bead"])
def test_hops_conserve_ring_energy_and_report_ledger(method, rescaling):
    dynamics = driver(method, nbeads=3, dboc=True, rescaling=rescaling)
    state = dynamics.initialize(nuclear([-.2, .1, .4], [4., 3., 5.]))
    energy = dynamics.energy(state)
    result = dynamics.attempt_hop(state, 1)
    assert result.accepted and not result.frustrated
    assert result.ring_energy_change == pytest.approx(0., abs=1e-13)
    assert dynamics.energy(state) == pytest.approx(energy, abs=1e-13)
    assert state.active_surface == 1
    result = dynamics.attempt_hop(state, 0)
    assert result.accepted
    assert dynamics.energy(state) == pytest.approx(energy, abs=1e-13)


@pytest.mark.parametrize("reverse", ["keep", "reverse"])
def test_frustrated_hop_preserves_energy_and_active_surface(reverse):
    dynamics = driver("dm", nbeads=2, frustrated_hops=reverse)
    state = dynamics.initialize(nuclear([.1, .3], [.05, .07]))
    p, energy = state.p.copy(), dynamics.energy(state)
    result = dynamics.attempt_hop(state, 1)
    assert result.frustrated and not result.accepted and state.active_surface == 0
    assert dynamics.energy(state) == pytest.approx(energy, abs=1e-14)
    if reverse == "keep":
        np.testing.assert_array_equal(state.p, p)
    else:
        assert np.linalg.norm(state.p-p) > .01


def test_complex_direction_is_phase_invariant_and_rejects_ambiguous_ray():
    d = np.array([1.+.3j, -.2+.4j, .1-.1j])
    expected = complex_rescaling_direction(d)
    for phase in np.linspace(-np.pi, np.pi, 19):
        np.testing.assert_allclose(complex_rescaling_direction(d*np.exp(1j*phase)), expected, atol=4e-15)
    # Independent one-dimensional phase sweep verifies that this ray maximizes
    # the real part of the coupling, rather than simply taking Re(d).
    projections = np.outer(np.cos(np.linspace(0., 2*np.pi, 10001)), d.real)
    projections -= np.outer(np.sin(np.linspace(0., 2*np.pi, 10001)), d.imag)
    opt = projections[np.argmax(np.sum(projections**2, axis=1))]
    assert abs(np.dot(expected, opt/np.linalg.norm(opt))) > 1.-1e-7
    with pytest.raises(PotentialError, match="no unique"):
        complex_rescaling_direction([1., 1j, 0.])
    np.testing.assert_array_equal(complex_rescaling_direction([0., 0j, 0.]), [0., 0., 0.])


class PauliCone:
    """H=x sigma_x+y sigma_y+gap sigma_z, avoiding the degeneracy."""
    def __init__(self, gap=.8):
        self.gap = gap

    def evaluate(self, q):
        v = np.zeros((len(q), 2, 2), dtype=complex)
        gradient = np.zeros((len(q), 2, 2, 1, 3), dtype=complex)
        v[:, 0, 0], v[:, 1, 1] = self.gap, -self.gap
        v[:, 0, 1] = q[:, 0, 0]-1j*q[:, 0, 1]
        v[:, 1, 0] = v[:, 0, 1].conj()
        gradient[:, 0, 1, 0, 0] = gradient[:, 1, 0, 0, 0] = 1.
        gradient[:, 0, 1, 0, 1], gradient[:, 1, 0, 0, 1] = -1j, 1j
        return DiabaticResult(v, gradient)


def test_berry_curvature_sign_magnitude_and_kinetic_rotation():
    dynamics = driver("ca", PauliCone(), dt=.001)
    ns = nuclear([0.], [.7])
    ns.p[0, 0, 1] = -.3
    data = dynamics.surface_data(ns.q)
    curvature = berry_curvature(data.couplings[0], 0)
    expected = 1/(2*.8**2)
    np.testing.assert_allclose(curvature, [[0., expected, 0.], [-expected, 0., 0.], [0., 0., 0.]], atol=2e-15)
    velocity = ns.p[0].ravel()/2.
    assert velocity@(curvature@velocity) == pytest.approx(0., abs=1e-16)
    energy = np.sum(ns.p**2)/4.
    dynamics._rotate_momenta(ns.q, ns.p, 0, .7, data)
    theta = 2*np.arctan(expected*.7/(2*2.))
    expected_p = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])@np.array([.7, -.3])
    np.testing.assert_allclose(ns.p[0, 0, :2], expected_p, atol=2e-15)
    assert np.sum(ns.p**2)/4. == pytest.approx(energy, abs=2e-16)


def test_ambiguous_complex_hop_direction_fails_transactionally():
    dynamics = driver("ca", PauliCone())
    state = dynamics.initialize(nuclear([0.], [3.]))
    saved = state.to_dict()
    with pytest.raises(PotentialError, match="no unique"):
        dynamics.attempt_hop(state, 1)
    assert state.to_dict() == saved
    assert dynamics.last_hop is None


class RandomGaugeDynamics(AdvancedRPSHDynamics):
    def __init__(self, *args, **kwargs):
        self.gauge_rng = np.random.default_rng(109)
        super().__init__(*args, **kwargs)

    def _eigensystem(self, matrix):
        e, u = np.linalg.eigh(matrix)
        phases = np.exp(1j*self.gauge_rng.uniform(-np.pi, np.pi, e.shape))
        return e, u*phases[:, None, :]


@pytest.mark.parametrize("method", ["ca", "ba", "dm"])
def test_random_adiabatic_eigenvector_phases_do_not_change_dynamics(method):
    potential = LinearTwoState(b=0. if method == "ba" else .17)
    kwargs = dict(method=method, native=False, seed=4, dboc=True, electronic_substeps=4)
    normal = AdvancedRPSHDynamics(potential, [2.], 3., 3, .001, 2, **kwargs)
    gauged = RandomGaugeDynamics(potential, [2.], 3., 3, .001, 2, **kwargs)
    ns = nuclear([-.2, .1, .4], [3., 4., 5.])
    sn = normal.initialize(ns, adiabatic_coefficients=[.8, .6j])
    sg = gauged.initialize(ns, adiabatic_coefficients=[.8, .6j])
    for _ in range(3):
        normal.step(sn)
        gauged.step(sg)
    np.testing.assert_allclose(normal.density_matrix(sn), gauged.density_matrix(sg), atol=3e-13)
    np.testing.assert_allclose(sn.q, sg.q, atol=1e-13)
    np.testing.assert_allclose(sn.p, sg.p, atol=1e-12)
    np.testing.assert_allclose(normal.last_probabilities, gauged.last_probabilities, atol=1e-13)
    rn, rg = normal.attempt_hop(sn, 1), gauged.attempt_hop(sg, 1)
    assert rn.accepted and rg.accepted
    np.testing.assert_allclose(sn.p, sg.p, atol=1e-12)


class RotatedProvider:
    def __init__(self, potential, rotation):
        self.potential, self.rotation = potential, rotation

    def evaluate(self, q):
        r = self.potential.evaluate(q)
        w = self.rotation
        return DiabaticResult(w.conj().T@r.matrix@w,
                              np.einsum("si,bstac,tj->bijac", w.conj(), r.gradients, w))


@pytest.mark.parametrize("method", ["ca", "dm"])
def test_complex_constant_diabatic_basis_change_is_covariant(method):
    potential = LinearTwoState(b=.17)
    w, _ = np.linalg.qr(np.array([[1+.2j, .3-1j], [.2+.4j, 1-.1j]]))
    original = driver(method, potential, nbeads=3, dboc=True)
    rotated = driver(method, RotatedProvider(potential, w), nbeads=3, dboc=True)
    ns = nuclear([-.2, .1, .4], [3., 4., 5.])
    so, sr = original.initialize(ns), rotated.initialize(ns)
    for _ in range(3):
        original.step(so)
        rotated.step(sr)
    np.testing.assert_allclose(rotated.density_matrix(sr), w.conj().T@original.density_matrix(so)@w, atol=1e-12)
    np.testing.assert_allclose(so.q, sr.q, atol=1e-13)
    np.testing.assert_allclose(so.p, sr.p, atol=2e-12)
    original.attempt_hop(so, 1)
    rotated.attempt_hop(sr, 1)
    np.testing.assert_allclose(so.p, sr.p, atol=2e-12)


def test_dm_independent_bead_wavefunction_phases_leave_density_and_hops_unchanged():
    d1, d2 = driver("dm", nbeads=3), driver("dm", nbeads=3)
    ns = nuclear([-.2, .1, .4], [1., 2., 3.])
    s1 = d1.initialize(ns, adiabatic_coefficients=[.8, .6j])
    s2 = s1.copy()
    s2.coefficients *= np.exp(1j*np.array([.2, -1.4, 2.9]))[:, None]
    d1.step(s1)
    d2.step(s2)
    np.testing.assert_allclose(d1.density_matrix(s1), d2.density_matrix(s2), atol=2e-14)
    np.testing.assert_allclose(d1.last_probabilities, d2.last_probabilities, atol=2e-14)
    np.testing.assert_array_equal(s1.p, s2.p)


@pytest.mark.parametrize("dboc", [False, True])
def test_complex_nuclear_energy_error_decreases_with_timestep(dboc):
    errors = []
    for dt in (.01, .005):
        dynamics = driver("ca", LinearTwoState(b=.17), dt=dt, dboc=dboc,
                          electronic_substeps=2, rescaling="centroid")
        ns = nuclear([.1], [.4])
        ns.q[0, 0, 1], ns.p[0, 0, 1] = .2, .3
        state = dynamics.initialize(ns)
        energy = dynamics.energy(state)
        for _ in range(round(.1/dt)):
            dynamics.step(state)
        errors.append(abs(dynamics.energy(state)-energy))
    assert errors[0] < 2e-6
    assert errors[1] < .3*errors[0]


@pytest.mark.parametrize("method", ["ca", "ba", "dm"])
def test_failure_after_rng_consumption_rolls_back_all_state_and_diagnostics(method):
    dynamics = driver(method, nbeads=2)
    state = dynamics.initialize(nuclear([.1, .3], [.5, .7]))
    saved, rng = state.to_dict(), copy.deepcopy(dynamics.rng.bit_generator.state)
    diagnostics = (dynamics.last_hop, dynamics.last_probabilities)
    def fail(_):
        dynamics.rng.random()
        raise RuntimeError("injected late failure")
    dynamics._apply_decoherence = fail
    with pytest.raises(RuntimeError, match="late failure"):
        dynamics.step(state)
    assert state.to_dict() == saved
    assert dynamics.rng.bit_generator.state == rng
    assert (dynamics.last_hop, dynamics.last_probabilities) == diagnostics


def test_excess_fssh_probability_fails_without_renormalizing_or_mutating():
    dynamics = AdvancedRPSHDynamics(RotatingSurfaces(), [1.], 3., 1, .1, 2,
                                   method="ba", native=False, electronic_substeps=1)
    state = dynamics.initialize(nuclear([0.], [20.]))
    saved, rng = state.to_dict(), copy.deepcopy(dynamics.rng.bit_generator.state)
    with pytest.raises(ValueError, match="probabilities exceed one"):
        dynamics.step(state)
    assert state.to_dict() == saved
    assert dynamics.rng.bit_generator.state == rng


def test_explicit_domain_rejections():
    with pytest.raises(PotentialError, match="real symmetric"):
        driver("ba", LinearTwoState(b=.2)).initialize(nuclear([.1]))
    with pytest.raises(ValueError, match="RPSH-DM requires none"):
        driver("dm", decoherence="edc")
    with pytest.raises(PotentialError, match="degenerate"):
        driver("dm", LinearTwoState(g=0., offset=0.)).initialize(nuclear([0.]))
    for invalid in (0., -1., np.nan, np.inf, True, "0.01"):
        with pytest.raises(ValueError, match="dboc_step"):
            driver(dboc=True, dboc_step=invalid)
    assert isinstance(RPSHBADynamics(LinearTwoState(), [2.], 3., 1, .001, 2, native=False), AdvancedRPSHDynamics)
    assert isinstance(RPSHDMDynamics(LinearTwoState(), [2.], 3., 1, .001, 2, native=False), AdvancedRPSHDynamics)
    assert isinstance(ComplexRPSHDynamics(LinearTwoState(b=.2), [2.], 3., 1, .001, 2, native=False), AdvancedRPSHDynamics)


def test_electronic_state_validation_rejects_invalid_norms_and_frames():
    ns = nuclear([0., .1])
    with pytest.raises(ValueError, match="normalized"):
        RPSHDMState(ns.q, ns.p, coefficients=np.ones((2, 2)))
    with pytest.raises(ValueError, match="unitary"):
        RPSHBAState(ns.q, ns.p, coefficients=[1., 0.], basis_vectors=np.zeros((2, 2, 2)))

