"""Independent forces, centroid hop diagnostics and restarts for RP-IESH."""
import copy
import json

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import State
from pyrpmd.iesh import IESHDynamics, IESHState, NewnsAndersonModel
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.rp_iesh import RPIESHDynamics, RPIESHState


def nuclear(beads=4, natoms=1, *, seed=7):
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=.3, size=(beads, natoms, 3))
    p = rng.normal(scale=.2, size=q.shape)
    p[:, 0, 0] += 3.
    return State(q, p)


def dynamics(beads=4, **kwargs):
    options = dict(nelectrons=2, native=False, seed=11, electronic_substeps=3)
    options.update(kwargs)
    return RPIESHDynamics(NewnsAndersonModel(nband=4), [20.], 10., beads, .025, **options)


def random_orbitals(states=5, electrons=2):
    rng = np.random.default_rng(12)
    return np.linalg.qr(rng.normal(size=(states, electrons))
                        + 1j*rng.normal(size=(states, electrons)))[0]


def assert_state_equal(first, second):
    for key in ("q", "p", "coefficients", "occupied"):
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key), err_msg=key)
    assert (first.step, first.time) == (second.step, second.time)
    if isinstance(first, RPIESHState) and isinstance(second, RPIESHState):
        assert first.hop_energy_change == second.hop_energy_change
        assert first.hop_energy_change_abs == second.hop_energy_change_abs


class AvoidedCrossing:
    nstates, natoms = 2, 1

    def evaluate(self, q):
        x = q[:, 0, 0]
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = .2*x, -.2*x
        matrix[:, 0, 1] = matrix[:, 1, 0] = .02
        gradient = np.zeros((len(q), 2, 2, 1, 3))
        gradient[:, 0, 0, 0, 0] = .2
        gradient[:, 1, 1, 0, 0] = -.2
        return DiabaticResult(matrix, gradient)


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_single_bead_matches_existing_iesh_bitwise_including_a_stochastic_hop(integrator):
    params = dict(nelectrons=1, native=False, seed=11, electronic_substeps=4,
                  integrator=integrator)
    classical = IESHDynamics(AvoidedCrossing(), [2.], 10., 1, .02, **params)
    ring = RPIESHDynamics(AvoidedCrossing(), [2.], 10., 1, .02, **params)
    initial = State([[[-1., 0., 0.]]], [[[3., 0., 0.]]])
    first, second = classical.initialize(initial), ring.initialize(initial)
    accepted = 0
    for _ in range(100):
        classical.step(first)
        ring.step(second)
        assert_state_equal(first, second)
        np.testing.assert_array_equal(classical.last_probabilities, ring.last_probabilities)
        assert classical.rng.bit_generator.state == ring.rng.bit_generator.state
        assert classical.energy(first) == ring.energy(second)
        if classical.last_hop is not None:
            assert ring.last_hop is not None
            for key in ("electron", "source", "target", "accepted", "frustrated", "energy_error"):
                assert getattr(classical.last_hop, key) == getattr(ring.last_hop, key)
            accepted += int(classical.last_hop.accepted)
        else:
            assert ring.last_hop is None
    assert accepted > 0


def test_state_roundtrip_is_detached_and_preserves_hop_accumulators():
    dyn = dynamics()
    state = dyn.initialize(nuclear(), occupied=[0, 4])
    assert dyn.attempt_hop(state, 1, 2).accepted
    restored = RPIESHState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert_state_equal(state, restored)
    copied = state.copy()
    assert_state_equal(state, copied)
    for other in (restored, copied):
        for name in ("q", "p", "coefficients", "occupied"):
            assert not np.shares_memory(getattr(state, name), getattr(other, name))
    assert state.coefficients.shape == (5, 2)
    assert state.occupied.shape == (2,)
    assert state.hop_energy_change_abs >= abs(state.hop_energy_change)
    with pytest.raises(ValueError, match="fields"):
        RPIESHState.from_dict({key: value for key, value in state.to_dict().items()
                              if key != "hop_energy_change"})


@pytest.mark.parametrize("signed,absolute", [(True, 1.), (np.inf, np.inf), (0., -1.), (2., 1.)])
def test_invalid_accumulated_diagnostics_are_rejected(signed, absolute):
    state = dynamics().initialize(nuclear())
    value = state.to_dict()
    value.update(hop_energy_change=signed, hop_energy_change_abs=absolute)
    with pytest.raises(ValueError, match="hop_energy_change"):
        RPIESHState.from_dict(value)
    state.hop_energy_change, state.hop_energy_change_abs = signed, absolute
    with pytest.raises(ValueError, match="hop_energy_change"):
        dynamics().step(state)


def test_old_iesh_still_rejects_ring_states_and_new_driver_requires_its_state():
    ring = nuclear()
    with pytest.raises(ValueError, match="P=1"):
        IESHState(ring.q, ring.p, coefficients=random_orbitals(), occupied=[0, 1])
    with pytest.raises(ValueError, match="P=1"):
        IESHDynamics(NewnsAndersonModel(), [2.], 2., 4, .1, native=False)
    old = IESHDynamics(NewnsAndersonModel(nband=4), [20.], 10., 1, .025,
                       nelectrons=2, native=False).initialize(nuclear(1))
    with pytest.raises(ValueError, match="RPIESHState"):
        dynamics(1).step(old)


@pytest.mark.parametrize("beads", [0, -1, True, np.bool_(True), 1.5])
def test_bead_count_is_a_strict_positive_integer(beads):
    with pytest.raises(ValueError, match="nbeads"):
        dynamics(beads)


def test_ring_forces_match_full_hamiltonian_finite_differences_with_bare_energy_once():
    model = NewnsAndersonModel(nband=4, natoms=2, coordinate=4, level_slope=-.13,
                               hybridization_slope=.2, bare_force_constant=.3)
    dyn = RPIESHDynamics(model, [2., 11.], 5., 4, .01, nelectrons=3, native=False)
    state = dyn.initialize(nuclear(4, 2), occupied=[0, 1, 3])
    energies, _, gradients = dyn._adiabatic(state.q)
    force = (model.bare_potential(state.q).forces-gradients[:, state.occupied].sum(axis=1)
             + dyn.integrator.spring_force(state.q))
    kinetic = np.sum(state.p**2/(2*dyn.masses[None, :, None]))
    delta = state.q-np.roll(state.q, -1, axis=0)
    spring = .5*(4/5.)**2*np.sum(dyn.masses[None, :, None]*delta**2)
    expected_energy = kinetic+spring+model.bare_potential(state.q).energy.sum()+energies[:, state.occupied].sum()
    assert dyn.energy(state) == pytest.approx(expected_energy, abs=2e-14)
    epsilon = 2e-5
    finite_difference = np.zeros_like(force)
    for index in np.ndindex(state.q.shape):
        plus, minus = state.copy(), state.copy()
        plus.q[index] += epsilon
        minus.q[index] -= epsilon
        finite_difference[index] = -(dyn.energy(plus)-dyn.energy(minus))/(2*epsilon)
    np.testing.assert_allclose(force, finite_difference, rtol=2e-8, atol=2e-9)


def test_centroid_nac_matches_independent_eigenvector_overlap_derivative():
    model = NewnsAndersonModel(nband=4, natoms=2, coordinate=4,
                               level_slope=-.13, hybridization_slope=.2)
    dyn = RPIESHDynamics(model, [2., 11.], 5., 4, .01, nelectrons=2, native=False)
    state = dyn.initialize(nuclear(4, 2))
    center = state.q.mean(axis=0, keepdims=True)
    _, u, _, couplings = dyn._adiabatic(center, couplings=True)
    epsilon = 1e-5
    independent = np.zeros_like(couplings)
    for atom, coordinate in np.ndindex(2, 3):
        plus, minus = center.copy(), center.copy()
        plus[0, atom, coordinate] += epsilon
        minus[0, atom, coordinate] -= epsilon
        up = np.linalg.eigh(model.evaluate(plus).matrix)[1][0]
        um = np.linalg.eigh(model.evaluate(minus).matrix)[1][0]
        up *= np.sign(np.sum(u[0]*up, axis=0))
        um *= np.sign(np.sum(u[0]*um, axis=0))
        independent[0, :, :, atom, coordinate] = u[0].T @ ((up-um)/(2*epsilon))
    np.testing.assert_allclose(couplings, independent, rtol=1e-7, atol=1e-9)


def test_electronic_midpoints_are_centroid_geometries_and_form_one_batch():
    class RecordedModel:
        def __init__(self):
            self.model = NewnsAndersonModel(nband=4, hybridization_slope=.2)
            self.nstates = self.model.nstates
            self.batches = []

        def evaluate(self, q):
            self.batches.append(q.copy())
            return self.model.evaluate(q)

    model = RecordedModel()
    dyn = RPIESHDynamics(model, [20.], 10., 5, .03, nelectrons=2,
                         native=False, electronic_substeps=3)
    state = dyn.initialize(nuclear(5), adiabatic_coefficients=random_orbitals())
    new_q = state.q+.03*state.p/20.
    model.batches.clear()
    actual, probability = dyn._electronic_step(state, new_q)
    old_center, new_center = state.q.mean(axis=0, keepdims=True), new_q.mean(axis=0, keepdims=True)
    fraction = (np.arange(3)+.5)/3
    expected_q = old_center+(new_center-old_center)*fraction[:, None, None]
    assert len(model.batches) == 1
    np.testing.assert_array_equal(model.batches[0], expected_q)
    reference = IESHDynamics(model.model, [20.], 10., 1, .03, nelectrons=2,
                             native=False, electronic_substeps=3)
    classical = IESHState(old_center, state.p.mean(axis=0, keepdims=True),
                          coefficients=state.coefficients, occupied=state.occupied)
    expected, expected_probability = reference._electronic_step(classical, new_center)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(probability, expected_probability)


class ConstantMatrix:
    natoms = 1

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.nstates = len(matrix)

    def evaluate(self, q):
        return DiabaticResult(np.broadcast_to(self.matrix, (len(q), *self.matrix.shape)),
                              np.zeros((len(q), self.nstates, self.nstates, 1, 3)))


def test_moving_ring_electronics_match_independent_matrix_exponential():
    h = np.array([[-.2, .03, .01], [.03, .05, -.02], [.01, -.02, .3]])
    dyn = RPIESHDynamics(ConstantMatrix(h), [2.], 3., 8, .07,
                         nelectrons=2, native=False, electronic_substeps=3)
    parent = nuclear(8)
    state = RPIESHState(parent.q, parent.p, coefficients=random_orbitals(3, 2),
                        occupied=[0, 1])
    initial, rng = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    for _ in range(17):
        dyn.step(state)
    expected = expm(-1j*h*(17*.07)) @ initial.coefficients
    np.testing.assert_allclose(state.coefficients, expected, atol=4e-14)
    np.testing.assert_allclose(dyn.density_matrix(state), expected @ expected.conj().T, atol=7e-14)
    assert dyn.rng.bit_generator.state == rng
    assert state.hop_energy_change == state.hop_energy_change_abs == 0.
    assert not np.allclose(state.q, initial.q)


class UnequalMassModel:
    nstates, natoms = 3, 2

    def evaluate(self, q):
        x = q[:, 0, 0]+.4*q[:, 1, 1]
        matrix = np.broadcast_to(np.diag([-.3, .1, .8]), (len(q), 3, 3)).copy()
        matrix[:, 0, 1] = matrix[:, 1, 0] = .2+.3*x
        gradient = np.zeros((len(q), 3, 3, 2, 3))
        gradient[:, 0, 1, 0, 0] = gradient[:, 1, 0, 0, 0] = .3
        gradient[:, 0, 1, 1, 1] = gradient[:, 1, 0, 1, 1] = .12
        return DiabaticResult(matrix, gradient)


def unequal_mass_state():
    dyn = RPIESHDynamics(UnequalMassModel(), [2., 11.], 2., 4, .02,
                         nelectrons=1, native=False)
    parent = nuclear(4, 2)
    parent.q[:, 0, 0] = [-1.2, -.3, .4, 1.1]
    parent.p[:, 0, 0] += 3.
    parent.p[:, 1, 1] += 4.
    return dyn, dyn.initialize(parent)


def test_hop_uniformly_shifts_beads_and_reports_centroid_and_full_ring_changes_separately():
    dyn, state = unequal_mass_state()
    before = state.copy()
    energy = dyn.energy(state)
    center = state.q.mean(axis=0, keepdims=True)
    centroid_e, _, _, d = dyn._adiabatic(center, couplings=True)
    bead_e = np.linalg.eigvalsh(dyn.potential.evaluate(state.q).matrix)
    centroid_gap = centroid_e[0, 1]-centroid_e[0, 0]
    ring_gap = np.sum(bead_e[:, 1]-bead_e[:, 0])
    assert abs(ring_gap-4*centroid_gap) > .02
    normal = d[0, 0, 1]/np.sqrt(dyn.masses[:, None])
    normal /= np.linalg.norm(normal)
    initial_weighted = state.p.mean(axis=0)/np.sqrt(dyn.masses[:, None])
    result = dyn.attempt_hop(state, 0, 1)
    assert result.accepted and not result.frustrated
    assert abs(result.centroid_energy_error) < 2e-14
    assert result.energy_error == result.centroid_energy_error
    assert result.ring_energy_change == pytest.approx(ring_gap-4*centroid_gap, abs=3e-14)
    assert dyn.energy(state)-energy == pytest.approx(result.ring_energy_change, abs=3e-14)
    np.testing.assert_allclose(state.p-state.p.mean(axis=0),
                               before.p-before.p.mean(axis=0), atol=2e-15)
    shift = state.p-before.p
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-15)
    weighted = state.p.mean(axis=0)/np.sqrt(dyn.masses[:, None])
    np.testing.assert_allclose(weighted-normal*np.sum(weighted*normal),
                               initial_weighted-normal*np.sum(initial_weighted*normal), atol=2e-15)
    assert state.hop_energy_change == result.ring_energy_change
    assert state.hop_energy_change_abs == abs(result.ring_energy_change)
    reverse = dyn.attempt_hop(state, 0, 0)
    assert reverse.accepted
    assert state.hop_energy_change == result.ring_energy_change+reverse.ring_energy_change
    assert state.hop_energy_change_abs == abs(result.ring_energy_change)+abs(reverse.ring_energy_change)
    assert state.hop_energy_change_abs > .04


def test_frustrated_hop_keeps_every_bead_and_diagnostics_and_pauli_target_rejected():
    dyn, state = unequal_mass_state()
    state.p[:] = 0.
    before = state.copy()
    result = dyn.attempt_hop(state, 0, 1)
    assert result.frustrated and not result.accepted
    assert result.centroid_energy_error == result.ring_energy_change == 0.
    assert_state_equal(state, before)
    with pytest.raises(ValueError, match="Pauli"):
        dyn.attempt_hop(state, 0, 0)


def test_zero_centroid_velocity_downhill_root_is_finite_and_internal_momenta_preserved():
    dyn, state = unequal_mass_state()
    state.occupied[:] = 1
    state.p[:] = 0.
    state.p[:, 0, 0] = [-.3, -.1, .1, .3]
    internal = state.p.copy()
    result = dyn.attempt_hop(state, 0, 0)
    assert result.accepted and abs(result.centroid_energy_error) < 1e-14
    assert np.isfinite(state.p).all()
    np.testing.assert_allclose(state.p-state.p.mean(axis=0), internal, atol=2e-16)


def test_state_and_rng_restart_remain_bitwise_exact_including_nonzero_hop_diagnostics():
    dyn = dynamics()
    state = dyn.initialize(nuclear(), occupied=[0, 4])
    assert dyn.attempt_hop(state, 1, 2).accepted
    assert state.hop_energy_change_abs > 0.
    for _ in range(7):
        dyn.step(state)
    payload = json.loads(json.dumps({"state": state.to_dict(), "rng": dyn.rng.bit_generator.state}))
    resumed, other = RPIESHState.from_dict(payload["state"]), dynamics()
    other.rng.bit_generator.state = payload["rng"]
    for _ in range(11):
        dyn.step(state)
        other.step(resumed)
        assert_state_equal(state, resumed)
        np.testing.assert_array_equal(dyn.last_probabilities, other.last_probabilities)
        assert dyn.last_hop == other.last_hop
        assert dyn.rng.bit_generator.state == other.rng.bit_generator.state


def test_failure_after_random_draw_does_not_commit_hop_diagnostics_or_any_state(monkeypatch):
    dyn = dynamics()
    state = dyn.initialize(nuclear(), occupied=[0, 4])
    assert dyn.attempt_hop(state, 1, 2).accepted
    before, rng = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    previous_hop, previous_probabilities = dyn.last_hop, dyn.last_probabilities.copy()
    original = dyn._hop

    def guaranteed_hop(*args):
        probability = np.zeros((2, 5))
        probability[1, 1] = 1.
        return state.coefficients.copy(), probability

    def fail_after_hop(*args, **kwargs):
        assert original(*args, **kwargs).accepted
        raise PotentialError("injected failure after accepted hop and RNG draw")

    monkeypatch.setattr(dyn, "_electronic_step", guaranteed_hop)
    monkeypatch.setattr(dyn, "_hop", fail_after_hop)
    with pytest.raises(PotentialError, match="after accepted hop"):
        dyn.step(state)
    assert_state_equal(state, before)
    assert dyn.rng.bit_generator.state == rng
    assert dyn.last_hop == previous_hop
    np.testing.assert_array_equal(dyn.last_probabilities, previous_probabilities)


def test_accumulator_overflow_rejects_hop_transactionally():
    class LargeBeadGap:
        nstates, natoms = 2, 1

        def evaluate(self, q):
            x = q[:, 0, 0]
            matrix = np.zeros((len(q), 2, 2))
            matrix[:, 1, 1] = 1.+1e307*x*x
            matrix[:, 0, 1] = matrix[:, 1, 0] = x
            gradient = np.zeros((len(q), 2, 2, 1, 3))
            gradient[:, 1, 1, 0, 0] = 2e307*x
            gradient[:, 0, 1, 0, 0] = gradient[:, 1, 0, 0, 0] = 1.
            return DiabaticResult(matrix, gradient)

    dyn = RPIESHDynamics(LargeBeadGap(), [2.], 10., 16, .01, native=False)
    q, p = np.zeros((16, 1, 3)), np.zeros((16, 1, 3))
    q[:, 0, 0] = np.tile([-1., 1.], 8)
    p[:, 0, 0] = 6.
    state = dyn.initialize(State(q, p))
    state.hop_energy_change_abs = 1e308
    before = state.copy()
    with pytest.raises(ValueError, match="hop_energy_change_abs"):
        dyn.attempt_hop(state, 0, 1)
    assert_state_equal(state, before)
    assert dyn.last_hop is None


def test_full_occupation_has_no_hops_or_rng_consumption():
    dyn = dynamics(nelectrons=5)
    state = dyn.initialize(nuclear())
    before = copy.deepcopy(dyn.rng.bit_generator.state)
    for _ in range(4):
        dyn.step(state)
    np.testing.assert_array_equal(dyn.last_probabilities, np.zeros((5, 5)))
    assert dyn.rng.bit_generator.state == before
    assert state.hop_energy_change == state.hop_energy_change_abs == 0.
    assert np.trace(dyn.density_matrix(state)).real == pytest.approx(5.)


def test_complex_soc_and_degeneracy_remain_explicitly_unsupported():
    for matrix, message in (([[0., 1j], [-1j, 1.]], "complex SOC"),
                            (np.eye(2), "degenerate")):
        dyn = RPIESHDynamics(ConstantMatrix(matrix), [2.], 2., 4, .01, native=False)
        with pytest.raises(PotentialError, match=message):
            dyn.initialize(nuclear())


def test_native_true_and_required_cannot_silently_fall_back(monkeypatch):
    import pyrpmd.dynamics as module
    original = module.importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "pyrpmd._native":
            raise ImportError("injected unavailable native extension")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    with pytest.raises(ValueError, match="native=True requires"):
        dynamics(native=True)
    with pytest.raises(RuntimeError, match="Native kernel not installed"):
        dynamics(native="required")
    assert dynamics(native=None).native is False


def test_native_and_python_ring_paths_agree_when_extension_available():
    pytest.importorskip("pyrpmd._native")
    python, native = dynamics(native=False), dynamics(native="required")
    first, second = python.initialize(nuclear()), native.initialize(nuclear())
    for _ in range(5):
        python.step(first)
        native.step(second)
    np.testing.assert_allclose(first.q, second.q, atol=2e-15)
    np.testing.assert_allclose(first.p, second.p, atol=2e-15)
    np.testing.assert_allclose(first.coefficients, second.coefficients, atol=2e-14)
    np.testing.assert_array_equal(first.occupied, second.occupied)
