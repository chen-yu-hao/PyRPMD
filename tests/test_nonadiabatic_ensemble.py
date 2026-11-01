"""Ensemble observations checked against exact electronic and free-nuclear motion."""

import copy

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.nonadiabatic_ensemble import (
    EnsemblePropagationError, density_matrix, run_nonadiabatic_ensemble,
)
from pyrpmd.surface_hopping import RPSHDynamics, RPSHState


class ConstantMatrix:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.closed = False

    def evaluate(self, q):
        matrix = np.broadcast_to(self.matrix, (len(q), *self.matrix.shape)).copy()
        return DiabaticResult(matrix, np.zeros((*matrix.shape, *q.shape[1:]), dtype=matrix.dtype))

    def close(self):
        self.closed = True


def nuclear(beads=1, *, step=0, time=0.):
    q = np.broadcast_to([[[.2, -.1, .3]]], (beads, 1, 3)).copy()
    p = np.broadcast_to([[[.4, -.2, .1]]], q.shape).copy()
    return State(q, p, step=step, time=time)


def nrpmd_factory(potential, *, dt=.04, beads=1, mass=2., beta=3.):
    return lambda index, rng: NRPMDDynamics(potential, [mass], beta, beads, dt,
                                           len(potential.matrix), native=False)


def rpsh_factory(potential, *, dt=.04, beads=1):
    return lambda index, rng: RPSHDynamics(potential, [2.], 3., beads, dt,
                                          len(potential.matrix), native=False, seed=rng)


def test_density_matrix_conjugation_signed_populations_and_basis_covariance():
    state = NRPMDState(np.zeros((2, 1, 3)), np.zeros((2, 1, 3)),
                       mapping_q=[[np.sqrt(3), 0.], [1., 2.]],
                       mapping_p=[[0., 1.], [.5, -.3]])
    estimate = density_matrix(state)
    np.testing.assert_allclose(estimate.diagonal().real, state.populations(), atol=1e-15)
    np.testing.assert_allclose(estimate, estimate.conj().T, atol=1e-15)
    u = expm(-.3j*np.array([[.3, .1+.2j], [.1-.2j, -.2]]))
    rotated = state.copy()
    z = (state.mapping_q+1j*state.mapping_p)@u.T
    rotated.mapping_q[:], rotated.mapping_p[:] = z.real, z.imag
    np.testing.assert_allclose(density_matrix(rotated), u@estimate@u.conj().T, atol=2e-15)
    hopping = RPSHState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                        coefficients=[1/np.sqrt(2), 1j/np.sqrt(2)])
    assert density_matrix(hopping)[0, 1] == pytest.approx(-.5j)
    # A valid MMST estimator can have negative populations; do not project it.
    signed = NRPMDState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                         mapping_q=[[2., 0.]], mapping_p=[[0., 0.]])
    np.testing.assert_array_equal(density_matrix(signed).diagonal().real, [1.5, -.5])


def test_nrpmd_constant_complex_matrix_matches_exact_unitary_and_free_nuclei():
    model = ConstantMatrix([[.23, .11+.19j], [.11-.19j, -.07]])
    parents = tuple(NRPMDState.from_nuclear(nuclear(3, step=7, time=2.+i), 2, i, 40+i)
                    for i in range(2))
    originals = tuple(parent.copy() for parent in parents)
    result = run_nonadiabatic_ensemble(parents, nrpmd_factory(model, beads=3),
                                      steps=11, sample_stride=4, independent=True)
    np.testing.assert_array_equal(result.sample_steps, [0, 4, 8, 11])
    np.testing.assert_allclose(result.time, [.0, .16, .32, .44], atol=1e-15)
    np.testing.assert_array_equal(result.start_times, [2., 3.])
    np.testing.assert_array_equal(result.start_steps, [7, 7])
    for trajectory, (before, after, parent) in enumerate(zip(originals, result.final_states, parents)):
        rho = density_matrix(before)
        for frame, time in enumerate(result.time):
            unitary = expm(-1j*time*model.matrix)
            np.testing.assert_allclose(result.diabatic_density[trajectory, frame],
                                       unitary@rho@unitary.conj().T, atol=3e-14)
        np.testing.assert_allclose(after.q, before.q+.44*before.p/2., atol=5e-15)
        np.testing.assert_allclose(after.p, before.p, atol=5e-15)
        assert after.time == pytest.approx(before.time+.44)
        assert after.step == before.step+11
        for name in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_array_equal(getattr(parent, name), getattr(before, name))
            assert not np.shares_memory(getattr(after, name), getattr(parent, name))
    np.testing.assert_allclose(result.energy_drift, 0., atol=2e-14)
    np.testing.assert_allclose(result.norm_drift, 0., atol=2e-13)
    assert result.electronic_norm.shape == (2, 4, 3)
    assert result.active_surface_counts is None
    assert not model.closed


def test_statistics_use_trajectory_axis_and_require_explicit_independence():
    model = ConstantMatrix(np.zeros((2, 2)))
    parents = tuple(NRPMDState.from_nuclear(nuclear(5), 2, i, 82+i) for i in range(2))
    factory = nrpmd_factory(model, beads=5)
    result = run_nonadiabatic_ensemble(parents, factory, steps=3, independent=True)
    np.testing.assert_allclose(result.mean_populations, .5, atol=2e-15)
    # Two Bernoulli-like trajectory contributions have SEM 0.5, regardless of
    # the five beads or four saved frames in each trajectory.
    np.testing.assert_allclose(result.population_sem, .5, atol=2e-15)
    expected_real = result.diabatic_density.real.std(axis=0, ddof=1)/np.sqrt(2)
    expected_imag = result.diabatic_density.imag.std(axis=0, ddof=1)/np.sqrt(2)
    np.testing.assert_allclose(result.density_sem_real, expected_real)
    np.testing.assert_allclose(result.density_sem_imag, expected_imag)
    unasserted = run_nonadiabatic_ensemble(parents, factory, steps=3)
    assert unasserted.population_sem is None
    assert unasserted.density_sem_real is None
    assert unasserted.sem_unavailable_reason == "trajectory_independence_not_asserted"
    single = run_nonadiabatic_ensemble(parents[:1], factory, steps=0, independent=True)
    assert single.population_sem is None
    assert single.density_sem_imag is None
    assert single.sem_unavailable_reason == "fewer_than_two_trajectories"
    np.testing.assert_array_equal(single.sample_steps, [0])
    assert single.diabatic_density.shape == (1, 1, 2, 2)


def test_rpsh_exact_coherence_amplitudes_and_active_counts_are_separate():
    model = ConstantMatrix([[0., .3], [.3, 0.]])
    parents = tuple(RPSHState(nuclear(2).q, nuclear(2).p, coefficients=[1., 0.],
                              active_surface=active) for active in (0, 0, 1))
    result = run_nonadiabatic_ensemble(parents, rpsh_factory(model, beads=2),
                                      steps=9, sample_stride=4, independent=True)
    for frame, time in enumerate(result.time):
        c = expm(-1j*time*model.matrix)@np.array([1., 0.])
        expected = np.outer(c, c.conj())
        np.testing.assert_allclose(result.diabatic_density[:, frame],
                                   np.broadcast_to(expected, (3, 2, 2)), atol=5e-14)
    assert result.mean_density[-1, 0, 1].imag > 0
    np.testing.assert_allclose(result.adiabatic_amplitude_populations, .5, atol=2e-14)
    np.testing.assert_array_equal(result.active_surface_counts, [[2, 1]]*4)
    np.testing.assert_allclose(result.active_surface_fractions, [[2/3, 1/3]]*4)
    np.testing.assert_allclose(result.active_surface_sem, 1/3, atol=1e-15)
    assert not np.allclose(result.mean_populations, result.active_surface_fractions)
    np.testing.assert_allclose(result.energy_drift, 0., atol=2e-14)
    np.testing.assert_allclose(result.electronic_norm, 1., atol=3e-14)
    for after, before in zip(result.final_states, parents):
        np.testing.assert_allclose(after.q, before.q+.36*before.p/2., atol=3e-15)
    assert result.metadata["density_basis"] == "fixed_diabatic"
    assert result.metadata["active_surface_basis"] == "centroid_adiabatic"


def test_focused_nuclear_initialization_and_rng_streams_are_reproducible_and_independent():
    model = ConstantMatrix([[.2, .1], [.1, -.2]])
    parents = [nuclear(), nuclear()]
    first = run_nonadiabatic_ensemble(parents, nrpmd_factory(model), steps=4,
                                     initial_state=0, seed=627)
    second = run_nonadiabatic_ensemble(parents, nrpmd_factory(model), steps=4,
                                      initial_state=0, seed=627)
    np.testing.assert_array_equal(first.diabatic_density, second.diabatic_density)
    assert first.rng_states == second.rng_states
    # Mapping phases differ by trajectory, even though initial populations agree.
    assert not np.allclose(first.diabatic_density[0, 0], first.diabatic_density[1, 0])
    assert first.rng_states[0]["initialization"] != first.rng_states[0]["dynamics"]
    assert first.rng_states[0]["dynamics"] != first.rng_states[1]["dynamics"]


def test_rpsh_factory_receives_independent_rngs_and_can_restore_them():
    model = ConstantMatrix([[.2, .1], [.1, -.2]])
    draws, supplied = [], []

    def factory(index, rng):
        supplied.append(rng)
        draws.append(rng.random(8))
        return RPSHDynamics(model, [2.], 3., 1, .04, 2, native=False, seed=rng)

    result = run_nonadiabatic_ensemble([nuclear(), nuclear()], factory,
                                      steps=4, active_surface=0, seed=281)
    assert not np.array_equal(draws[0], draws[1])
    original_draws = np.array(draws)
    draws.clear()
    repeated = run_nonadiabatic_ensemble([nuclear(), nuclear()], factory,
                                        steps=4, active_surface=0, seed=281)
    np.testing.assert_array_equal(draws, original_draws)
    np.testing.assert_array_equal(result.diabatic_density, repeated.diabatic_density)
    assert result.rng_states == repeated.rng_states
    saved = copy.deepcopy(result.rng_states)
    supplied[0].random(11)
    assert result.rng_states == saved

    def restored_factory(index, rng):
        rng.bit_generator.state = copy.deepcopy(saved[index]["dynamics"])
        return RPSHDynamics(model, [2.], 3., 1, .04, 2, native=False, seed=rng)

    continued = run_nonadiabatic_ensemble(result.final_states, restored_factory, steps=0)
    assert continued.rng_states[0]["dynamics"] == saved[0]["dynamics"]
    np.testing.assert_array_equal(continued.diabatic_density[:, 0], result.diabatic_density[:, -1])


@pytest.mark.parametrize("keyword,value", [("steps", True), ("steps", -1), ("steps", 1.2),
                                           ("sample_stride", 0), ("sample_stride", False),
                                           ("sample_stride", 1.5), ("seed", -1),
                                           ("seed", True), ("independent", 1)])
def test_strict_ensemble_arguments(keyword, value):
    options = {"steps": 1, "initial_state": 0, keyword: value}
    with pytest.raises(ValueError):
        run_nonadiabatic_ensemble([nuclear()], nrpmd_factory(ConstantMatrix(np.eye(2))), **options)


@pytest.mark.parametrize("dt", [0., -.02])
def test_forward_ensemble_requires_positive_dt_even_for_initial_only(dt):
    with pytest.raises(EnsemblePropagationError, match="dt must be positive") as failure:
        run_nonadiabatic_ensemble([nuclear()], nrpmd_factory(ConstantMatrix(np.eye(2)), dt=dt),
                                  steps=0, initial_state=0)
    assert failure.value.trajectory_index == 0
    assert failure.value.relative_step == 0


def test_mixed_parameters_reused_driver_and_wrong_rpsh_rng_fail_explicitly():
    model = ConstantMatrix([[.2, .1], [.1, -.2]])

    def mixed(index, rng):
        return NRPMDDynamics(model, [2.], 3.+index, 1, .04, 2, native=False)

    with pytest.raises(EnsemblePropagationError, match="parameters/method must match") as failure:
        run_nonadiabatic_ensemble([nuclear(), nuclear()], mixed, steps=0, initial_state=0)
    assert failure.value.trajectory_index == 1
    shared = nrpmd_factory(model)(0, None)
    with pytest.raises(EnsemblePropagationError, match="reused a driver"):
        run_nonadiabatic_ensemble([nuclear(), nuclear()], lambda index, rng: shared,
                                  steps=0, initial_state=0)
    with pytest.raises(EnsemblePropagationError, match="supplied rng"):
        run_nonadiabatic_ensemble([nuclear()],
                                  lambda index, rng: RPSHDynamics(model, [2.], 3., 1, .04, 2,
                                                                 seed=9, native=False),
                                  steps=0, active_surface=0)


def test_failed_trajectory_is_not_dropped_and_borrowed_states_and_provider_are_untouched():
    class FailingMatrix(ConstantMatrix):
        def evaluate(self, q):
            if q[:, 0, 0].max() > .201:
                raise RuntimeError("deliberate electronic-structure failure")
            return super().evaluate(q)

    model = FailingMatrix([[.2, .1], [.1, -.2]])
    stationary = nuclear()
    stationary.p[:] = 0
    moving = nuclear()
    parents = [stationary, moving]
    originals = [state.copy() for state in parents]
    with pytest.raises(EnsemblePropagationError, match="deliberate") as failure:
        run_nonadiabatic_ensemble(parents, nrpmd_factory(model), steps=3, initial_state=0)
    error = failure.value
    assert (error.trajectory_index, error.relative_step, error.stage) == (1, 1, "propagation")
    assert isinstance(error.__cause__, RuntimeError)
    for parent, before in zip(parents, originals):
        np.testing.assert_array_equal(parent.q, before.q)
        np.testing.assert_array_equal(parent.p, before.p)
        assert parent.time == before.time and parent.step == before.step
    assert not model.closed


def test_nuclear_parent_state_selection_is_explicit_and_prepared_states_are_not_overwritten():
    model = ConstantMatrix([[.2, .1], [.1, -.2]])
    with pytest.raises(EnsemblePropagationError, match="explicit initial_state"):
        run_nonadiabatic_ensemble([nuclear()], nrpmd_factory(model), steps=0)
    with pytest.raises(EnsemblePropagationError, match="explicit active_surface"):
        run_nonadiabatic_ensemble([nuclear()], rpsh_factory(model), steps=0)
    prepared = NRPMDState.from_nuclear(nuclear(), 2, 0, 4)
    with pytest.raises(EnsemblePropagationError, match="cannot overwrite"):
        run_nonadiabatic_ensemble([prepared], nrpmd_factory(model), steps=0, initial_state=1)
