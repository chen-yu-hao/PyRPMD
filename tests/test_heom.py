"""Independent spin--boson HEOM references, continuation and strict validation."""

from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from pyrpmd.analysis.heom import HEOMState, SpinBosonHEOM


def solver(**kwargs):
    settings = dict(epsilon=.4, delta=1., reorganization_energy=.15, cutoff=1., beta=1.,
                    max_depth=4, matsubara_terms=2, terminator=True)
    settings.update(kwargs)
    return SpinBosonHEOM(**settings)


def spectral_dephasing(times, *, epsilon=.3, lam=.15, gamma=1., beta=1., scale=1.):
    """Exact Gaussian influence functional integrated directly over J(omega).

    This does not use the production Matsubara coefficients or HEOM generator.
    The omitted integral above w=2000 is bounded by 2*lam*gamma/(pi*w**2)
    (coth is one there to double precision for these test temperatures).
    """
    values = []
    for time in times:
        def integrand(omega):
            if omega == 0:
                return 2*lam*time*time/(np.pi*beta*gamma)
            spectral = 2*lam*gamma*omega/(omega*omega+gamma*gamma)
            return (spectral/np.pi * 2*np.sin(omega*time/2)**2
                    / (omega*omega*np.tanh(beta*omega/2)))
        integral, error = quad(integrand, 0, 2000., epsabs=2e-10, epsrel=2e-9, limit=3000)
        assert error < 2e-9
        values.append(.5*np.exp(-4*scale*scale*integral-1j*epsilon*time))
    return np.asarray(values)


@pytest.mark.parametrize("zero", ["reorganization_energy", "coupling_scale"])
def test_isolated_limit_has_analytic_rabi_populations(zero):
    model = solver(**{zero: 0.})
    time = np.linspace(0, 12, 81)
    result = model.propagate([[1., 0.], [0., 0.]], time)
    frequency = np.hypot(model.epsilon, model.delta)
    excited = (model.delta/frequency)**2*np.sin(frequency*time/2)**2
    np.testing.assert_allclose(result.populations[:, 1], excited, atol=4e-14, rtol=0.)
    np.testing.assert_allclose(result.populations[:, 0], 1-excited, atol=4e-14, rtol=0.)
    assert model.ado_count == 1
    assert result.max_trace_error < 4e-14
    assert result.max_hermiticity_error < 4e-14
    assert result.physical_converged is False


def test_dissipative_dephasing_matches_full_spectral_integral_and_converges_in_k():
    time = np.array([0., .5, 1., 2., 4.])
    expected = spectral_dephasing(time)
    errors = []
    for terms in (0, 2, 4):
        model = solver(epsilon=.3, delta=0., max_depth=8, matsubara_terms=terms)
        result = model.propagate(np.ones((2, 2))/2, time)
        errors.append(np.max(abs(result.density[:, 0, 1]-expected)))
        np.testing.assert_allclose(result.populations, .5, atol=5e-14, rtol=0.)
        assert result.min_density_eigenvalue > -1e-13
    assert errors[0] > 10*errors[1]
    assert errors[1] > 2*errors[2]
    assert errors[2] < 1.3e-4
    assert abs(expected[-1]) < .02  # substantial physical dephasing, not lambda=0


def test_hierarchy_depth_converges_to_closed_form_gaussian_influence_function():
    time = np.linspace(0., 4., 21)
    errors = []
    for depth in (2, 4, 8):
        model = solver(epsilon=.3, delta=0., max_depth=depth)
        # Exact solution of the chosen finite-exponential Gaussian bath.  This
        # independently checks upward/downward ADO coefficients and scaling.
        frequencies, coefficients = model.frequencies, model.coefficients
        integral = np.sum(coefficients*(np.expm1(-time[:, None]*frequencies)
                                         +time[:, None]*frequencies)/frequencies**2, axis=1)
        expected = .5*np.exp(-4*(integral.real+model.tail_residue*time)-1j*.3*time)
        actual = model.propagate(np.ones((2, 2))/2, time)
        errors.append(np.max(abs(actual.density[:, 0, 1]-expected)))
    assert errors[0] > 20*errors[1]
    assert errors[1] > 1000*errors[2]
    assert errors[2] < 2.5e-7


@pytest.mark.parametrize("terminator", [False, True])
def test_population_dynamics_agree_with_independent_qutip_heom(terminator):
    qt = pytest.importorskip("qutip", minversion="5.0")
    from qutip.solver.heom import DrudeLorentzBath, HEOMSolver

    model = solver(terminator=terminator, coupling_scale=.7)
    time = np.linspace(0., 6., 31)
    bath = DrudeLorentzBath(qt.Qobj(model.coupling), model.reorganization_energy,
                            model.cutoff, 1/model.beta, model.matsubara_terms)
    hamiltonian = qt.Qobj(model.hamiltonian)
    if terminator:
        residue, boundary = bath.terminator()
        assert residue == pytest.approx(model.tail_residue, abs=1e-15)
        hamiltonian = qt.liouvillian(hamiltonian)+boundary
    reference = HEOMSolver(hamiltonian, bath, max_depth=model.max_depth,
                          options={"atol": 2e-13, "rtol": 2e-12, "method": "bdf",
                                   "nsteps": 20000, "progress_bar": ""})
    rho = np.array([[.8, .2j], [-.2j, .2]])
    expected = reference.run(qt.Qobj(rho), time)
    actual = model.propagate(rho, time)
    np.testing.assert_allclose(actual.density, np.array([x.full() for x in expected.states]),
                               atol=3e-10, rtol=0.)


def test_sparse_action_matches_independent_dense_exponential_for_small_hierarchy():
    model = solver(max_depth=2, matsubara_terms=1)
    initial = model.initialize([[.6, .2j], [-.2j, .4]])
    expected = expm(model.generator.toarray()*.37)@initial.ados.reshape(-1)
    result = model.propagate(initial, [0., .37])
    np.testing.assert_allclose(result.final_state.ados.reshape(-1), expected, atol=2e-15)


def test_coupling_convention_matches_rescaled_spectral_density():
    time = np.linspace(0., 3., 21)
    half_q = solver(coupling_scale=.5).propagate([[1., 0.], [0., 0.]], time)
    quarter_j = solver(reorganization_energy=.15/4).propagate([[1., 0.], [0., 0.]], time)
    np.testing.assert_allclose(half_q.density, quarter_j.density, atol=2e-14)


def test_full_ados_continue_in_new_solver_without_resetting_bath_memory():
    model = solver()
    time = np.linspace(0., 4., 41)
    initial = np.array([[1., 0.], [0., 0.]])
    full = model.propagate(initial, time)
    first = model.propagate(initial, time[:21])
    resumed = solver().propagate(first.final_state, time[20:])
    np.testing.assert_array_equal(resumed.density, full.density[20:])
    np.testing.assert_array_equal(resumed.final_state.ados, full.final_state.ados)
    reset = model.propagate(model.initialize(first.density[-1], time=time[20]), time[20:])
    assert np.max(abs(reset.density-resumed.density)) > .01
    assert np.any(abs(first.final_state.ados[1:]) > .01)


def test_irregular_observation_times_and_single_time_preserve_state():
    model = solver()
    initial = model.initialize(np.eye(2)/2, time=2.5)
    only = model.propagate(initial, [2.5])
    np.testing.assert_array_equal(only.final_state.ados, initial.ados)
    times = [2.5, 2.51, 2.7, 3.2]
    result = model.propagate(initial, times)
    direct = model.propagate(initial, [2.5, 3.2])
    np.testing.assert_allclose(result.final_state.ados, direct.final_state.ados, atol=3e-14)


def test_large_observation_intervals_do_not_consume_random_state():
    model = solver(max_depth=4, matsubara_terms=4)
    before = np.random.get_state()
    result = model.propagate(np.eye(2)/2, [0., 8.])
    after = np.random.get_state()
    for first, second in zip(before, after):
        np.testing.assert_array_equal(first, second)
    repeat = solver(max_depth=4, matsubara_terms=4).propagate(np.eye(2)/2, [0., 8.])
    np.testing.assert_array_equal(result.final_state.ados, repeat.final_state.ados)


@pytest.mark.parametrize("name,value", [
    ("epsilon", .3), ("delta", .9), ("reorganization_energy", .16), ("cutoff", 1.1),
    ("beta", .9), ("max_depth", 3), ("matsubara_terms", 1), ("terminator", False),
    ("coupling_scale", .5),
])
def test_every_scientific_option_changes_identity_and_rejects_incompatible_state(name, value):
    original = solver()
    changed = solver(**{name: value})
    assert original.run_identity != changed.run_identity
    with pytest.raises(ValueError, match="run_identity"):
        changed.propagate(original.initialize(np.eye(2)/2), [0., 1.])


def test_allocation_guard_does_not_change_identity_or_truncate_requested_hierarchy():
    model = solver(max_depth=4, matsubara_terms=2)
    assert model.ado_count == 35
    assert len(model.labels) == 35
    assert max(map(sum, model.labels)) == 4
    assert solver(max_ados=35).run_identity == model.run_identity
    with pytest.raises(ValueError, match="35 ADOs"):
        solver(max_ados=34)


def test_owned_immutable_arrays_and_detached_generator():
    model = solver()
    initial = np.eye(2)/2
    state = model.initialize(initial)
    initial[0, 0] = 100
    assert state.density[0, 0] == .5
    with pytest.raises(AttributeError, match="immutable"):
        model.beta = 2.
    for array in (model.hamiltonian, model.coupling, model.coefficients,
                  model.frequencies, state.ados):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    generator = model.generator
    generator.data[:] = 0
    assert model.generator.nnz > 0
    identity = model.identity
    identity["beta"] = -1
    assert model.identity["beta"] == 1
    result = model.propagate(state, [0., .1])
    for array in (result.time, result.density):
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize("name,value", [
    ("epsilon", True), ("delta", np.nan), ("reorganization_energy", -.1),
    ("cutoff", 0.), ("cutoff", np.inf), ("beta", 0.), ("beta", np.inf),
    ("coupling_scale", 1j), ("coupling_scale", np.nan), ("max_depth", True),
    ("max_depth", -1), ("max_depth", 2.1), ("matsubara_terms", -1),
    ("matsubara_terms", True), ("max_ados", 0), ("terminator", 1),
])
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        solver(**{name: value})


def test_coalescing_poles_and_invalid_tail_closure_fail_explicitly():
    with pytest.raises(ValueError, match="coincides"):
        solver(beta=2*np.pi, matsubara_terms=3)
    with pytest.raises(ValueError, match="omitted Matsubara"):
        solver(beta=10., matsubara_terms=0)
    # A slow retained Matsubara term can have a negative coefficient; that
    # correct low-temperature expansion is not silently clipped or removed.
    low_temperature = solver(beta=10., matsubara_terms=2)
    assert low_temperature.coefficients[1].real < 0
    assert low_temperature.tail_residue > 0


@pytest.mark.parametrize("rho", [
    np.eye(3)/3, [[1, 0], [0, np.nan]], [[.6, .1], [.2, .4]],
    np.eye(2), [[1.2, 0], [0, -.2]], [[True, False], [False, False]],
])
def test_invalid_initial_density_is_rejected(rho):
    with pytest.raises(ValueError, match="density"):
        solver().propagate(rho, [0., 1.])


@pytest.mark.parametrize("times", [[], [True, False], [0., np.inf], [0., 0.],
                                    [0., -.1], [1., 2.], [[0., 1.]], [0., 1j]])
def test_invalid_times_are_rejected(times):
    with pytest.raises(ValueError, match="times"):
        solver().propagate(np.eye(2)/2, times)


def test_malformed_full_state_and_wrong_ado_count_are_rejected():
    model = solver()
    state = model.initialize(np.eye(2)/2)
    with pytest.raises(ValueError, match="ADO"):
        HEOMState(0., np.zeros((1, 3, 3)), state.run_identity)
    with pytest.raises(ValueError, match="run_identity"):
        HEOMState(0., state.ados, "unknown")
    with pytest.raises(ValueError, match="state time"):
        replace(state, time=-1)
    with pytest.raises(ValueError, match="ADO count"):
        model.propagate(replace(state, ados=state.ados[:1]), [0., 1.])
    with pytest.raises(ValueError, match="state density"):
        model.propagate(replace(state, ados=np.zeros_like(state.ados)), [0., 1.])


def test_convergence_example_has_reproducible_experiment_identity_and_honest_diagnostics():
    namespace = runpy.run_path(str(Path(__file__).parents[1]
                                   /"examples/nonadiabatic/heom_spin_boson.py"))
    result = namespace["run"](duration=.2, points=3, depths=(1, 2), matsubara=(0, 1))
    assert result["physical_converged"] is False
    assert len(result["experiment_identity"]) == 64
    assert result["experiment"]["observation_times"] == [0., .1, .2]
    assert len(result["hierarchy_scan"]) == 2
    assert len(result["matsubara_scan"]) == 2
    assert result["analytic_dephasing"]["max_coherence_absolute_error"] < 1.3e-4
    assert result["hierarchy_scan"][-1]["max_population_difference_from_largest"] == 0
