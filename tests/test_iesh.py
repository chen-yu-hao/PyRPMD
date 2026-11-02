"""Independent determinant, force, energy, gauge and restart controls for IESH."""
import copy
import json

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import State
from pyrpmd.iesh import (
    IESHDynamics, IESHState, NewnsAndersonModel,
    determinant_hopping_probabilities, slater_determinant_ratios,
)
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.diabatic import check_diabatic_potential


def nuclear(q=0.1, p=0.3, natoms=1):
    x, momentum = np.zeros((1, natoms, 3)), np.zeros((1, natoms, 3))
    x[0, 0, 0], momentum[0, 0, 0] = q, p
    return State(x, momentum)


def dynamics(**kwargs):
    options = dict(nelectrons=2, native=False, seed=11, electronic_substeps=2)
    options.update(kwargs)
    return IESHDynamics(NewnsAndersonModel(nband=4), [20.], 10., 1, .05, **options)


def random_orbitals(nstates=6, nelectrons=3, seed=17):
    rng = np.random.default_rng(seed)
    c, _ = np.linalg.qr(rng.normal(size=(nstates, nelectrons))
                        + 1j*rng.normal(size=(nstates, nelectrons)))
    return c


@pytest.mark.parametrize("discretization", ["midpoint", "gauss-legendre"])
@pytest.mark.parametrize("nband", [2, 5, 16])
def test_metal_quadrature_hybridization_sum_rule_and_identity(nband, discretization):
    model = NewnsAndersonModel(nband=nband, band_min=-.7, band_max=1.1,
                               chemical_potential=.2, discretization=discretization,
                               hybridization=.04)
    assert np.all(np.diff(model.bath_energies) > 0.)
    assert np.all(model.bath_weights > 0.)
    assert np.sum(model.bath_weights) == pytest.approx(1.8)
    q = np.zeros((1, 1, 3))
    h = model.evaluate(q).matrix[0]
    # Integral of Gamma(epsilon)/(2*pi), independent of grid size.
    assert np.sum(h[0, 1:]**2) == pytest.approx(.04*1.8/(2*np.pi))
    restored = NewnsAndersonModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(restored.evaluate(q).matrix, h[None])
    assert restored.metadata["model_identity"] == model.to_dict()
    with pytest.raises(AttributeError, match="immutable"):
        model.level_slope = 0.
    with pytest.raises(ValueError):
        model.bath_weights.setflags(write=True)


def test_newns_matrix_and_bare_force_match_independent_finite_differences():
    model = NewnsAndersonModel(nband=5, level_slope=-.12, hybridization_slope=.7,
                               bare_force_constant=.23, coordinate=4, natoms=2, center=.6)
    q = np.random.default_rng(7).normal(size=(3, 2, 3))
    check = check_diabatic_potential(model, q, model.nstates)
    assert check["max_absolute_gradient_error"] < 2e-11
    bare = model.bare_potential(q)
    epsilon = 1e-5
    plus, minus = q.copy(), q.copy()
    plus[:, 1, 1] += epsilon
    minus[:, 1, 1] -= epsilon
    fd = -(model.bare_potential(plus).energy-model.bare_potential(minus).energy)/(2*epsilon)
    np.testing.assert_allclose(bare.forces[:, 1, 1], fd, atol=1e-10)


@pytest.mark.parametrize("bad", [
    {"nband": True}, {"nband": 1}, {"band_min": .4, "band_max": .3},
    {"chemical_potential": -.5}, {"hybridization": -.1}, {"hybridization": True},
    {"hybridization_slope": np.inf}, {"bare_force_constant": -1},
    {"coordinate": 3}, {"coordinate": 1.2}, {"discretization": "trapezoid"},
])
def test_invalid_model_parameters_fail(bad):
    with pytest.raises(ValueError):
        NewnsAndersonModel(**bad)


def test_model_identity_rejects_extra_or_missing_fields():
    identity = NewnsAndersonModel().to_dict()
    for bad in ({**identity, "version": True}, {**identity, "extra": 1},
                {**identity, "parameters": {"nband": 10}}):
        with pytest.raises(ValueError, match="identity"):
            NewnsAndersonModel.from_dict(bad)


def test_determinant_ratios_match_independent_explicit_slater_determinants():
    c = random_orbitals()
    occupied = np.array([4, 0, 2])  # Deliberately unsorted; signs matter.
    overlap = c[occupied]
    current = np.linalg.det(overlap)
    ratios = slater_determinant_ratios(c, occupied)
    expected = np.empty_like(ratios)
    for electron in range(len(occupied)):
        for target in range(len(c)):
            replaced = overlap.copy()
            replaced[electron] = c[target]
            expected[target, electron] = np.linalg.det(replaced)/current
    np.testing.assert_allclose(ratios, expected, rtol=2e-14, atol=2e-14)


def test_ratio_solve_survives_exponentially_small_determinant_underflow():
    nelectrons, amplitude = 90, 1e-4
    c = np.vstack((amplitude*np.eye(nelectrons),
                   np.sqrt(1-amplitude**2)*np.eye(nelectrons)))
    occupied = np.arange(nelectrons)
    assert np.linalg.det(c[occupied]) == 0.  # Direct probability division fails.
    ratios = slater_determinant_ratios(c, occupied)
    np.testing.assert_allclose(ratios[:nelectrons], np.eye(nelectrons), atol=1e-15)
    np.testing.assert_allclose(ratios[nelectrons:],
                               np.sqrt(1-amplitude**2)/amplitude*np.eye(nelectrons))


def test_determinant_current_sign_matches_finite_difference_many_electron_population():
    c = random_orbitals()
    occupied = np.array([0, 3, 5])
    rng = np.random.default_rng(4)
    d = rng.normal(size=(6, 6))
    d = .02*(d-d.T)
    ratios = slater_determinant_ratios(c, occupied)
    ratios[occupied] = 0.
    outgoing_rate = 2*np.sum(ratios.T.real*d[occupied])
    epsilon = 1e-5
    plus = abs(np.linalg.det((expm(-epsilon*d) @ c)[occupied]))**2
    minus = abs(np.linalg.det((expm(epsilon*d) @ c)[occupied]))**2
    population = abs(np.linalg.det(c[occupied]))**2
    assert outgoing_rate == pytest.approx(-(plus-minus)/(2*epsilon*population), rel=2e-8, abs=2e-10)


def test_one_electron_reduces_to_fssh_with_correct_probability_sign():
    c = np.array([np.sqrt(.6), np.sqrt(.4)*np.exp(.3j)])[:, None]
    d = np.array([[0., .15], [-.15, 0.]])
    expected = 2*.08*np.real(c[0, 0].conj()*c[1, 0])*.15/.6
    probabilities = determinant_hopping_probabilities(c, [0], d, .08)
    np.testing.assert_allclose(probabilities, [[0., expected]])
    np.testing.assert_array_equal(determinant_hopping_probabilities(c, [1], d, .08), [[0., 0.]])


def test_hopping_invariant_under_orbital_unitary_rotation_and_eigenvector_signs():
    c = random_orbitals()
    occupied = [4, 0, 2]
    rng = np.random.default_rng(61)
    d = rng.normal(size=(6, 6))
    d = .002*(d-d.T)
    expected = determinant_hopping_probabilities(c, occupied, d, .01)
    rotation = random_orbitals(3, 3, seed=2)
    actual = determinant_hopping_probabilities(c @ rotation, occupied, d, .01)
    np.testing.assert_allclose(actual, expected, atol=2e-16)
    signs = np.array([-1., 1., -1., -1., 1., 1.])
    actual = determinant_hopping_probabilities(signs[:, None]*c, occupied,
                                              signs[:, None]*d*signs[None, :], .01)
    np.testing.assert_allclose(actual, expected, atol=2e-16)
    permutation = [2, 0, 1]
    actual = determinant_hopping_probabilities(c, np.asarray(occupied)[permutation], d, .01)
    np.testing.assert_allclose(actual, expected[permutation], atol=2e-16)


def test_pauli_exclusion_and_full_occupation_zero_hops():
    c = random_orbitals(5, 5)
    d = np.ones((5, 5))
    d = np.triu(d, 1)-np.tril(d, -1)
    probabilities = determinant_hopping_probabilities(c, np.arange(5), d, 10.)
    np.testing.assert_array_equal(probabilities, np.zeros((5, 5)))
    probabilities = determinant_hopping_probabilities(c[:, :2], [0, 3], .001*d, .01)
    np.testing.assert_array_equal(probabilities[:, [0, 3]], np.zeros((2, 2)))


def test_singular_determinant_and_excess_probability_are_explicit_failures():
    with pytest.raises(ValueError, match="singular"):
        slater_determinant_ratios(np.eye(3)[:, :2], [1, 2])
    c = np.array([[1.], [1.]])/np.sqrt(2.)
    with pytest.raises(ValueError, match="reduce the nuclear timestep"):
        determinant_hopping_probabilities(c, [0], [[0., 1.], [-1., 0.]], 1.)
    with pytest.raises(ValueError, match="max_hop_probability"):
        determinant_hopping_probabilities(c, [0], [[0., 1.], [-1., 0.]], .1, max_probability=.05)


class ConstantMatrix:
    natoms = 1

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.nstates = len(matrix)

    def evaluate(self, q):
        return DiabaticResult(np.broadcast_to(self.matrix, (len(q), *self.matrix.shape)),
                              np.zeros((len(q), self.nstates, self.nstates, q.shape[1], 3)))


def test_fixed_nuclei_electronic_evolution_matches_independent_matrix_exponential():
    h = np.array([[-.2, .03, .01], [.03, .05, -.02], [.01, -.02, .3]])
    dyn = IESHDynamics(ConstantMatrix(h), [2.], 3., 1, .07,
                       nelectrons=2, native=False, electronic_substeps=3)
    state = IESHState(nuclear(0., 0.).q, nuclear(0., 0.).p, coefficients=random_orbitals(3, 2),
                      occupied=np.array([0, 1]))
    initial = state.coefficients.copy()
    rng = copy.deepcopy(dyn.rng.bit_generator.state)
    for _ in range(17):
        dyn.step(state)
    expected = expm(-1j*h*(17*.07)) @ initial
    np.testing.assert_allclose(state.coefficients, expected, atol=3e-14)
    np.testing.assert_allclose(state.coefficients.conj().T @ state.coefficients, np.eye(2), atol=4e-14)
    assert dyn.rng.bit_generator.state == rng


def test_density_matrix_has_electron_trace_idempotency_and_separate_active_counts():
    dyn = dynamics()
    state = dyn.initialize(nuclear(), adiabatic_coefficients=random_orbitals(5, 2))
    for basis in ("diabatic", "adiabatic"):
        rho = dyn.density_matrix(state, basis=basis)
        np.testing.assert_allclose(rho @ rho, rho, atol=2e-15)
        assert np.trace(rho).real == pytest.approx(2.)
    assert dyn.populations(state).sum() == pytest.approx(2.)
    np.testing.assert_array_equal(dyn.active_populations(state), [1., 1., 0., 0., 0.])
    assert not np.allclose(dyn.populations(state), dyn.active_populations(state))


def test_bare_nuclear_potential_counted_once_not_once_per_electron():
    model = NewnsAndersonModel(nband=4, level_slope=0., bare_force_constant=.3)
    dyn = IESHDynamics(model, [2.], 3., 1, .1, nelectrons=3, native=False)
    state = dyn.initialize(nuclear(.5, 0.))
    eigenvalues = np.linalg.eigvalsh(model.evaluate(state.q).matrix[0])
    assert dyn.energy(state) == pytest.approx(.5*.3*.5**2+eigenvalues[:3].sum())
    dyn.step(state)
    # Exact one-step harmonic velocity Verlet, not a threefold larger force.
    q1 = .5-.5*.1**2*.3*.5/2.
    p1 = -.5*.1*.3*(.5+q1)
    assert state.q[0, 0, 0] == pytest.approx(q1)
    assert state.p[0, 0, 0] == pytest.approx(p1)


def test_uncoupled_metal_has_no_hops_and_no_rng_consumption():
    model = NewnsAndersonModel(nband=4, level_energy=.7, level_slope=.01,
                               hybridization=0., bare_force_constant=0.)
    dyn = IESHDynamics(model, [2.], 3., 1, .1, nelectrons=2, native=False)
    state = dyn.initialize(nuclear(.5, .2))
    initial_rng = copy.deepcopy(dyn.rng.bit_generator.state)
    initial_energy = dyn.energy(state)
    for _ in range(10):
        dyn.step(state)
    np.testing.assert_array_equal(dyn.last_probabilities, np.zeros((2, 5)))
    assert dyn.rng.bit_generator.state == initial_rng
    assert dyn.energy(state) == pytest.approx(initial_energy, abs=1e-15)


class UnequalMassModel:
    nstates, natoms = 3, 2

    def evaluate(self, q):
        x = q[:, 0, 0]+.4*q[:, 1, 1]
        matrix = np.broadcast_to(np.diag([-.3, .1, .8]), (len(q), 3, 3)).copy()
        matrix[:, 0, 1] = matrix[:, 1, 0] = .2+.05*x
        gradients = np.zeros((len(q), 3, 3, 2, 3))
        gradients[:, 0, 1, 0, 0] = gradients[:, 1, 0, 0, 0] = .05
        gradients[:, 0, 1, 1, 1] = gradients[:, 1, 0, 1, 1] = .02
        return DiabaticResult(matrix, gradients)


def test_unequal_mass_hop_conserves_energy_and_perpendicular_mass_weighted_momentum():
    dyn = IESHDynamics(UnequalMassModel(), [2., 11.], 2., 1, .02,
                       nelectrons=1, native=False)
    initial = nuclear(.4, 6., natoms=2)
    initial.p[0, 1] = [2., 4., -1.]
    state = dyn.initialize(initial)
    energy = dyn.energy(state)
    p0 = state.p.copy()
    _, _, _, d = dyn._adiabatic(state.q, couplings=True)
    weighted_direction = d[0, 0, 1]/np.sqrt(dyn.masses[:, None])
    weighted_direction /= np.linalg.norm(weighted_direction)
    u0 = p0[0]/np.sqrt(dyn.masses[:, None])
    result = dyn.attempt_hop(state, 0, 1)
    assert result.accepted and not result.frustrated
    assert abs(result.energy_error) < 1e-14
    assert dyn.energy(state) == pytest.approx(energy, abs=2e-14)
    u1 = state.p[0]/np.sqrt(dyn.masses[:, None])
    perpendicular0 = u0-weighted_direction*np.sum(u0*weighted_direction)
    perpendicular1 = u1-weighted_direction*np.sum(u1*weighted_direction)
    np.testing.assert_allclose(perpendicular0, perpendicular1, atol=1e-15)


def test_frustrated_hop_keeps_momentum_and_occupation_and_pauli_hop_rejected():
    dyn = dynamics()
    state = dyn.initialize(nuclear(.1, 0.))
    before = state.copy()
    result = dyn.attempt_hop(state, 1, 4)
    assert result.frustrated and not result.accepted
    np.testing.assert_array_equal(state.p, before.p)
    np.testing.assert_array_equal(state.occupied, before.occupied)
    with pytest.raises(ValueError, match="Pauli"):
        dyn.attempt_hop(state, 0, 1)


def test_zero_velocity_downhill_hop_has_finite_energy_conserving_root():
    dyn = dynamics()
    state = dyn.initialize(nuclear(.1, 0.), occupied=[0, 4])
    energy = dyn.energy(state)
    result = dyn.attempt_hop(state, 1, 2)
    assert result.accepted
    assert np.isfinite(state.p).all()
    assert dyn.energy(state) == pytest.approx(energy, abs=3e-16)


def test_state_and_rng_checkpoint_resume_is_bitwise_exact():
    dyn = dynamics()
    state = dyn.initialize(nuclear(.2, 3.))
    for _ in range(7):
        dyn.step(state)
    payload = json.loads(json.dumps({"state": state.to_dict(), "rng": dyn.rng.bit_generator.state}))
    resumed = IESHState.from_dict(payload["state"])
    other = dynamics()
    other.rng.bit_generator.state = payload["rng"]
    for _ in range(11):
        dyn.step(state)
        other.step(resumed)
    for key in ("q", "p", "coefficients", "occupied"):
        np.testing.assert_array_equal(getattr(state, key), getattr(resumed, key))
    assert (state.step, state.time) == (resumed.step, resumed.time)
    assert dyn.rng.bit_generator.state == other.rng.bit_generator.state
    np.testing.assert_array_equal(dyn.last_probabilities, other.last_probabilities)


def test_failed_step_after_random_draw_rolls_back_nuclei_orbitals_rng_and_diagnostics(monkeypatch):
    dyn = dynamics()
    state = dyn.initialize(nuclear())
    before, rng = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    old_probabilities = dyn.last_probabilities.copy()

    def guaranteed_hop(*args):
        p = np.zeros((2, 5))
        p[0, 2] = 1.
        return state.coefficients.copy(), p

    def fail(*args, **kwargs):
        raise PotentialError("injected failure after RNG draw")

    monkeypatch.setattr(dyn, "_electronic_step", guaranteed_hop)
    monkeypatch.setattr(dyn, "_hop", fail)
    with pytest.raises(PotentialError, match="after RNG"):
        dyn.step(state)
    for key in ("q", "p", "coefficients", "occupied"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
    assert dyn.rng.bit_generator.state == rng
    assert dyn.last_hop is None
    np.testing.assert_array_equal(dyn.last_probabilities, old_probabilities)


@pytest.mark.parametrize("bad", [{"nelectrons": 6}, {"nelectrons": True},
                                 {"electronic_substeps": 0}, {"max_hop_probability": 1.1},
                                 {"gap_tolerance": 0.}, {"native": 1}, {"native": 0.}])
def test_invalid_dynamics_parameters_fail(bad):
    with pytest.raises(ValueError):
        dynamics(**bad)


def test_rejects_ring_polymer_complex_soc_and_degenerate_states():
    with pytest.raises(ValueError, match="P=1"):
        IESHDynamics(NewnsAndersonModel(), [2.], 2., 2, .1, native=False)
    for h, message in ((np.array([[0., 1j], [-1j, 1.]]), "complex SOC"),
                       (np.eye(2), "degenerate")):
        dyn = IESHDynamics(ConstantMatrix(h), [2.], 2., 1, .1, native=False)
        with pytest.raises(PotentialError, match=message):
            dyn.initialize(nuclear())


def test_invalid_live_state_and_modified_parameters_are_rejected():
    dyn = dynamics()
    state = dyn.initialize(nuclear())
    state.coefficients[0, 0] += .3
    with pytest.raises(ValueError, match="orthonormal"):
        dyn.step(state)
    state = dyn.initialize(nuclear())
    state.occupied[1] = state.occupied[0]
    with pytest.raises(ValueError, match="distinct"):
        dyn.step(state)
    dyn.electronic_substeps = 20
    with pytest.raises(ValueError, match="immutable"):
        dyn.initialize(nuclear())


@pytest.mark.parametrize("occupied", [[True, 2], (0, False), [np.bool_(True), 3]])
def test_heterogeneous_boolean_occupation_list_is_not_silently_integer_cast(occupied):
    dyn = dynamics()
    with pytest.raises(ValueError, match="boolean"):
        dyn.initialize(nuclear(), occupied=occupied)


def test_explicit_native_request_cannot_silently_fall_back(monkeypatch):
    import pyrpmd.dynamics as nuclear_module
    original = nuclear_module.importlib.import_module
    calls = []

    def unavailable(name, *args, **kwargs):
        if name == "pyrpmd._native":
            calls.append(name)
            raise ImportError("injected unavailable extension")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(nuclear_module.importlib, "import_module", unavailable)
    with pytest.raises(ValueError, match="native=True requires"):
        dynamics(native=True)
    with pytest.raises(RuntimeError, match="Native kernel not installed"):
        dynamics(native="required")
    automatic = dynamics(native=None)
    assert automatic.native is False
    attempts = len(calls)
    explicit = dynamics(native=np.bool_(False))
    assert explicit.native is False and len(calls) == attempts


def test_batched_midpoint_evaluations_match_sequential_quadrature():
    class RecordBatches:
        def __init__(self, potential):
            self.potential, self.nstates = potential, potential.nstates
            self.batches = []

        def evaluate(self, q):
            self.batches.append(len(q))
            return self.potential.evaluate(q)

    provider = RecordBatches(NewnsAndersonModel(nband=4, hybridization_slope=.1))
    dyn = IESHDynamics(provider, [20.], 10., 1, .03, nelectrons=2,
                       native=False, electronic_substeps=3)
    state = dyn.initialize(nuclear(.2, .7))
    new_q = state.q+.03*state.p/20.
    provider.batches.clear()
    actual_c, actual_p = dyn._electronic_step(state, new_q)
    assert provider.batches == [3]
    expected_c = state.coefficients.copy()
    expected_p = np.zeros((2, 5))
    for k in range(3):
        q = state.q+(new_q-state.q)*(k+.5)/3
        e, u, _, d = dyn._adiabatic(q, couplings=True)
        c = u[0].T @ expected_c
        phase = np.exp(-.005j*e[0])[:, None]
        mid = c*phase
        velocity_coupling = np.einsum("ac,ijac->ij", (new_q[0]-state.q[0])/.03, d[0])
        expected_p += determinant_hopping_probabilities(mid, state.occupied, velocity_coupling, .01)
        expected_c = u[0] @ (mid*phase)
    np.testing.assert_allclose(actual_c, expected_c, atol=2e-15)
    np.testing.assert_allclose(actual_p, expected_p, atol=2e-15)
