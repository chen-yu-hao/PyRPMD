"""Independent projected-trace, single-surface and thermal-rate checks."""

import copy
import itertools

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm

from pyrpmd.dynamics import Dynamics, State, INTEGRATORS
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.thermal_nrpmd import (
    ThermalEnsemble, ThermalNRPMDDynamics, ThermalNRPMDSampler,
    ThermalRateCorrelation, thermal_flux_side, thermal_population_rate,
    thermal_weight, _split_matrix, _weight,
)


class MatrixModel:
    def __init__(self, matrix, slope=None, spring=0.):
        self.matrix = np.asarray(matrix)
        self.slope = np.zeros_like(self.matrix) if slope is None else np.asarray(slope)
        self.spring = spring

    def evaluate(self, q):
        b, n, _ = q.shape
        s = len(self.matrix)
        matrix = np.broadcast_to(self.matrix, (b, s, s)).copy()
        matrix += q[:, 0, 0, None, None]*self.slope
        matrix += 0.5*self.spring*np.sum(q*q, axis=(1, 2))[:, None, None]*np.eye(s)
        gradients = np.array(np.broadcast_to(self.spring*q[:, None, None]*np.eye(s)[None, :, :, None, None],
                                             (b, s, s, n, 3)), dtype=matrix.dtype, copy=True)
        gradients[:, :, :, 0, 0] += self.slope
        return DiabaticResult(matrix, gradients)


def random_state(beads=2, states=2, seed=12):
    rng = np.random.default_rng(seed)
    return NRPMDState(rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)),
                      mapping_q=rng.normal(size=(beads, states)),
                      mapping_p=rng.normal(size=(beads, states)))


@pytest.mark.parametrize("beads", [1, 2])
def test_gauss_hermite_integration_recovers_exact_electronic_trace_and_projectors(beads):
    # Three-point Hermite integrates the polynomial SEO projectors exactly.
    # The independent answer comes directly from finite-matrix exponentials,
    # including noncommuting matrices at different nuclear beads.
    pot = MatrixModel([[0.3, 0.24], [0.24, -0.4]], [[0.6, 0.1], [0.1, -0.2]], spring=.7)
    beta = 1.8
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = np.linspace(-.4, .7, beads)
    masses = np.array([1.7])
    split = _split_matrix(pot, q, beta, 2)
    nodes, weights = hermgauss(3)
    weights /= np.sqrt(np.pi)
    trace = 0.
    projected = np.zeros(2)
    spring = .5*(beads/beta)**2*np.sum(masses[None, :, None]*(q-np.roll(q, -1, axis=0))**2)
    for index in itertools.product(range(3), repeat=4*beads):
        point = nodes[list(index)].reshape(2, beads, 2)
        x, y = point
        value = _weight(split, q, x, y, masses, beta)
        if not value.sign:
            continue
        integrand = (4**beads*np.prod(weights[list(index)])*value.sign
                     * np.exp(value.log_density+np.sum(point**2)+beta/beads*spring))
        trace += integrand
        projected += integrand*value.electronic_populations
    boltzmann = [expm(-beta/beads*matrix) for matrix in pot.evaluate(q).matrix]
    products = []
    for origin in range(beads):
        product = np.eye(2)
        for offset in range(beads):
            product = product@boltzmann[(origin+offset) % beads]
        products.append(product)
    exact = np.trace(products[0])
    np.testing.assert_allclose(trace, exact, rtol=2e-14, atol=3e-14)
    np.testing.assert_allclose(projected, np.mean([np.diag(x) for x in products], axis=0), atol=3e-14)


def test_weight_includes_signed_paths_and_exact_population_partition():
    pot = MatrixModel([[.1, .3], [.3, -.1]])
    found = False
    for seed in range(100):
        state = random_state(beads=3, seed=seed)
        value = thermal_weight(pot, state, [1.7], 2.)
        np.testing.assert_allclose(value.electronic_populations.sum(), 1., atol=1e-12)
        if value.sign < 0:
            found = True
            break
    assert found, "negative paths must not be clipped or discarded"


def test_weight_common_energy_shift_and_node():
    state = random_state()
    base = thermal_weight(MatrixModel([[.1, .3], [.3, -.1]]), state, [1.7], 2.3)
    shifted = thermal_weight(MatrixModel([[1e4+.1, .3], [.3, 1e4-.1]]), state, [1.7], 2.3)
    assert shifted.sign == base.sign
    assert shifted.log_density-base.log_density == pytest.approx(-2.3e4, abs=2e-10)
    np.testing.assert_allclose(shifted.electronic_populations, base.electronic_populations, atol=2e-11)
    state.mapping_q[:] = 0
    node = thermal_weight(MatrixModel(np.zeros((2, 2))), state, [1.7], 2.3)
    assert node.sign == 0 and node.log_density == -np.inf


def test_one_state_finite_bead_weight_has_exact_harmonic_normal_mode_stiffness():
    # Independent Hessian of -log rho compared with the analytic Gaussian
    # normal-mode distribution. This resolves the beta/P and spring factors.
    beads, beta, mass, spring = 5, 2.3, 1.7, .6
    potential = MatrixModel([[0.]], spring=spring)
    state = random_state(beads=beads, states=1)
    state.q[:] = 0.
    def action(x):
        trial = state.copy()
        trial.q[:, 0, 0] = x
        return -thermal_weight(potential, trial, [mass], beta).log_density
    eps = 1e-3
    zero = np.zeros(beads)
    basis = np.eye(beads)
    hessian = np.empty((beads, beads))
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            hessian[i, j] = (action(eps*(ei+ej))-action(eps*(ei-ej))
                              -action(eps*(-ei+ej))+action(-eps*(ei+ej)))/(4*eps**2)
    frequencies = 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)
    exact = np.sort(beta/beads*(spring+mass*frequencies**2))
    np.testing.assert_allclose(np.linalg.eigvalsh(hessian), exact, atol=3e-9)


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_one_state_thermal_dynamics_reduces_exactly_to_rpmd_for_arbitrary_mapping_norm(scheme):
    state = random_state(beads=3, states=1)
    # Thermal mapping norms are deliberately not focused to a unit population.
    assert not np.isclose(state.populations().sum(), 1.)
    ordinary = State(state.q, state.p)
    potential = MatrixModel([[0.]], spring=.7)
    thermal = ThermalNRPMDDynamics(potential, [1.7], 2.8, 3, .01, 1, native=False, integrator=scheme)
    scalar = Dynamics(HarmonicPotential(.7), [1.7], 2.8, 3, .01, native=False, integrator=scheme)
    for _ in range(20):
        thermal.step(state)
        scalar.step(ordinary)
    np.testing.assert_allclose(state.q, ordinary.q, atol=1e-14)
    np.testing.assert_allclose(state.p, ordinary.p, atol=1e-14)
    assert thermal.energy(state) == pytest.approx(scalar.energy(ordinary), abs=1e-14)


def test_thermal_hamiltonian_force_matches_finite_difference_with_fluctuating_norm():
    state = random_state(beads=3)
    state.p[:] = 0.
    pot = MatrixModel([[.8, .2], [.2, .1]], [[.7, .03], [.03, -.2]], spring=.4)
    dyn = ThermalNRPMDDynamics(pot, [1.7], 2.8, 3, 1e-7, 2, native=False)
    gradient = np.zeros_like(state.q)
    for index in np.ndindex(state.q.shape):
        plus, minus = state.copy(), state.copy()
        plus.q[index] += 2e-6
        minus.q[index] -= 2e-6
        gradient[index] = (dyn.energy(plus)-dyn.energy(minus))/4e-6
    dyn.step(state)
    np.testing.assert_allclose(state.p/dyn.dt, -gradient, atol=2e-7)


def test_thermal_propagation_reversal_and_energy():
    state = random_state(beads=3)
    original = state.copy()
    pot = MatrixModel([[.8, .2], [.2, .1]], [[.7, .03], [.03, -.2]], spring=.4)
    dyn = ThermalNRPMDDynamics(pot, [1.7], 2.8, 3, .004, 2, native=False)
    initial_energy = dyn.energy(state)
    for _ in range(100):
        dyn.step(state)
    assert abs(dyn.energy(state)-initial_energy) < 3e-5
    state.p *= -1
    state.mapping_p *= -1
    for _ in range(100):
        dyn.step(state)
    state.p *= -1
    state.mapping_p *= -1
    for name in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_allclose(getattr(state, name), getattr(original, name), atol=2e-12)


def assert_snapshot_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key]


def test_sampler_checkpoint_continuation_and_detachment():
    potential = MatrixModel([[.1, .2], [.2, -.1]], spring=.7)
    sampler = ThermalNRPMDSampler(potential, [1.7], 2., np.zeros((2, 1, 3)), 2, seed=18)
    for _ in range(5):
        sampler.sweep()
    checkpoint = sampler.snapshot()
    a = sampler.sample(5, stride=2)
    sampler.restore(checkpoint)
    b = sampler.sample(5, stride=2)
    for first, second in zip(a.states, b.states):
        for key in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    np.testing.assert_array_equal(a.signs, b.signs)
    np.testing.assert_array_equal(a.electronic_populations, b.electronic_populations)
    checkpoint["q"][:] = 999
    assert np.max(abs(sampler.q)) < 10
    a.states[0].q[:] = -999
    assert np.max(abs(b.states[0].q)) < 10


@pytest.mark.parametrize("failure", [PotentialError, KeyboardInterrupt])
def test_sampler_mid_sweep_failure_rolls_back_rng_and_counters(failure):
    class Failing(MatrixModel):
        calls = 0
        fail_at = None
        def evaluate(self, q):
            self.calls += 1
            if self.calls == self.fail_at:
                raise failure("injected provider failure")
            return super().evaluate(q)
    potential = Failing([[.1, .2], [.2, -.1]], spring=.7)
    sampler = ThermalNRPMDSampler(potential, [1.7], 2., np.zeros((2, 1, 3)), 2, seed=18)
    before = sampler.snapshot()
    potential.fail_at = potential.calls+2
    with pytest.raises(failure, match="injected"):
        sampler.sweep()
    assert_snapshot_equal(before, sampler.snapshot())
    invalid = copy.deepcopy(before)
    invalid["accepted"]["mapping"] = 100
    with pytest.raises(ValueError, match="accepted"):
        sampler.restore(invalid)
    assert_snapshot_equal(before, sampler.snapshot())


def test_clock_overflow_rolls_back_thermal_propagation():
    potential = MatrixModel([[0.]])
    state = NRPMDState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                       time=np.finfo(float).max, mapping_q=np.zeros((1, 1)),
                       mapping_p=np.zeros((1, 1)))
    before = state.copy()
    dynamics = ThermalNRPMDDynamics(potential, [1.], 1., 1, np.finfo(float).max,
                                    1, native=False)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="clock"):
            dynamics.step(state)
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
    assert (state.step, state.time) == (before.step, before.time)


def test_sampler_one_state_gaussian_nuclei_and_seo_mapping_moments():
    # Independent analytical moments of the actual projected density:
    # q variance=1/(beta*k), p variance=m/beta; x²,y² each have mean 3/2.
    sampler = ThermalNRPMDSampler(MatrixModel([[0.]], spring=1.), [3.], 2.,
                                  np.zeros((1, 1, 3)), 1, seed=189,
                                  nuclear_step=.65, centroid_step=.65)
    ensemble = sampler.sample(2500, burn_in=300, stride=2)
    q = np.array([state.q for state in ensemble.states])
    p = np.array([state.p for state in ensemble.states])
    x = np.array([state.mapping_q for state in ensemble.states])
    y = np.array([state.mapping_p for state in ensemble.states])
    np.testing.assert_allclose(np.mean(q*q, axis=(0, 1, 2)), .5, rtol=.13)
    np.testing.assert_allclose(np.mean(p*p, axis=(0, 1, 2)), 1.5, rtol=.07)
    assert np.mean(x*x) == pytest.approx(1.5, rel=.1)
    assert np.mean(y*y) == pytest.approx(1.5, rel=.1)
    np.testing.assert_array_equal(ensemble.signs, 1.)
    np.testing.assert_array_equal(ensemble.electronic_populations, 1.)


def test_sampler_thermal_electronic_population_is_boltzmann_not_focused():
    potential = MatrixModel([[0., 0.], [0., 1.]], spring=.8)
    sampler = ThermalNRPMDSampler(potential, [1.], 1.3, np.zeros((1, 1, 3)), 2,
                                  seed=681, nuclear_step=.5, centroid_step=.5)
    ensemble = sampler.sample(4000, burn_in=300, stride=1)
    population = np.mean(ensemble.electronic_populations, axis=0)
    exact = np.exp(-1.3*np.array([0., 1.]))
    exact /= exact.sum()
    np.testing.assert_allclose(population, exact, atol=.045)
    assert np.std([s.populations().sum() for s in ensemble.states]) > .1


def make_ensemble(potential, states, beta=2., masses=(1.7,), signs=None):
    weights = [thermal_weight(potential, state, masses, beta) for state in states]
    return ThermalEnsemble(tuple(states), [w.sign for w in weights] if signs is None else signs,
                            [w.electronic_populations for w in weights], masses, beta,
                            potential, np.arange(len(states)), {"test_parents": True})


def test_population_flux_matches_analytic_unitary_derivative_and_saved_parents_unchanged():
    potential = MatrixModel([[.3, .2], [.2, -.1]])
    states = [random_state(beads=1, seed=i) for i in range(4)]
    ensemble = make_ensemble(potential, states)
    dynamics = ThermalNRPMDDynamics(potential, [1.7], 2., 1, .03, 2, native=False)
    result = thermal_population_rate(ensemble, dynamics, 0, 1, steps=7, stride=2)
    np.testing.assert_allclose(result.time, [0., .06, .12, .18, .21])
    for sample, initial in enumerate(ensemble.states):
        z0 = (initial.mapping_q+1j*initial.mapping_p)[0]
        for frame, time in enumerate(result.time):
            z = expm(-1j*time*potential.matrix)@z0
            flux = np.imag(z.conj()*(potential.matrix@z))[1]
            expected = ensemble.signs[sample]*ensemble.electronic_populations[sample, 0]*flux
            assert result.numerator[sample, frame] == pytest.approx(expected, abs=3e-14)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_array_equal(getattr(ensemble.states[sample], key), getattr(states[sample], key))


def test_uncoupled_electronic_population_rate_has_zero_flux():
    potential = MatrixModel([[.3, 0.], [0., -.1]], spring=.7)
    ensemble = make_ensemble(potential, [random_state(beads=1, seed=i) for i in range(3)])
    dynamics = ThermalNRPMDDynamics(potential, [1.7], 2., 1, .03, 2, native=False)
    result = thermal_population_rate(ensemble, dynamics, 0, 1, steps=7)
    np.testing.assert_allclose(result.numerator, 0., atol=2e-16)


def test_nuclear_flux_side_matches_free_drift_and_no_extra_bead_factor():
    # P=1 free streaming is analytical, independent of the integrator.
    potential = MatrixModel([[0.]])
    states = [random_state(beads=1, states=1, seed=i) for i in range(5)]
    ensemble = make_ensemble(potential, states)
    dynamics = ThermalNRPMDDynamics(potential, [1.7], 2., 1, .04, 1, native=False)
    gradient = np.array([[1., 0., 0.]])
    def surface(q):
        return q[0, 0]-.2, gradient
    width = .3
    result = thermal_flux_side(ensemble, dynamics, surface, steps=5, smoothing_width=width)
    for i, state in enumerate(states):
        position = state.q[0, 0, 0]
        velocity = state.p[0, 0, 0]/1.7
        tangent = np.tanh((position-.2)/width)
        flux = .5*(1-tangent*tangent)/width*velocity
        side = .5*(1+np.tanh((position+velocity*result.time-.2)/width))
        np.testing.assert_allclose(result.numerator[i], flux*side, atol=3e-16)
        assert result.reactant_weight[i] == pytest.approx(.5*(1-tangent))


def test_population_flux_total_conservation_and_finite_difference():
    state = random_state()
    potential = MatrixModel([[.3, .2], [.2, -.1]], [[.2, .04], [.04, -.1]])
    dynamics = ThermalNRPMDDynamics(potential, [1.7], 2., 2, 1e-7, 2, native=False)
    flux = dynamics.population_flux(state)
    assert flux.sum() == pytest.approx(0., abs=1e-15)
    before = state.populations()
    dynamics.step(state)
    np.testing.assert_allclose((state.populations()-before)/dynamics.dt, flux, atol=8e-8)


def test_rate_sign_guard_ratio_normalization_and_oscillation_rejection():
    times = np.linspace(0, 10, 101)
    numerator = np.full((40, 101), .12)
    denominator = np.full(40, .3)
    result = ThermalRateCorrelation(times, numerator, denominator, np.ones(40), "test")
    np.testing.assert_allclose(result.rate_curve, .4)
    summary = result.summarize(start=2, stop=10, block_size=2)
    assert summary["candidate_rate"] == pytest.approx(.4)
    assert summary["plateau_detected"]
    assert summary["standard_error"] == pytest.approx(0., abs=1e-15)
    assert not summary["physical_converged"]
    assert not result.summarize(start=2, stop=10)["plateau_detected"]
    oscillatory = numerator*(1+.5*np.sin(times*2*np.pi))
    oscillating = ThermalRateCorrelation(times, oscillatory, denominator, np.ones(40), "test")
    assert not oscillating.summarize(start=2, stop=10, block_size=2)["plateau_detected"]
    cancelled = ThermalRateCorrelation(times, numerator, denominator, np.tile([1., -1.], 20), "test")
    assert cancelled.rate_curve is None
    assert "average_sign_below_threshold" in cancelled.summarize()["unavailable_reasons"]


def test_correlated_ratio_jackknife_preserves_numerator_denominator_covariance():
    times = np.arange(5.)
    denominator = np.linspace(.1, .9, 40)
    numerator = denominator[:, None]*np.ones((40, 5))*.7
    result = ThermalRateCorrelation(times, numerator, denominator, np.ones(40), "test")
    summary = result.summarize(start=0, stop=4, block_size=2)
    assert summary["standard_error"] == pytest.approx(0., abs=5e-15)
    assert summary["candidate_rate"] == pytest.approx(.7)


def test_complex_thermal_matrix_is_rejected_without_silently_discarding_phase():
    potential = MatrixModel([[.1, .2j], [-.2j, -.1]])
    with pytest.raises(PotentialError, match="complex|real"):
        ThermalNRPMDSampler(potential, [1.7], 2., np.zeros((1, 1, 3)), 2)


def test_correlations_reject_mismatched_provider_and_bad_surface():
    potential = MatrixModel([[0.]])
    ensemble = make_ensemble(potential, [random_state(beads=1, states=1)])
    wrong = ThermalNRPMDDynamics(MatrixModel([[0.]]), [1.7], 2., 1, .01, 1, native=False)
    with pytest.raises(ValueError, match="match"):
        thermal_flux_side(ensemble, wrong, lambda q: (0., np.ones_like(q)), steps=2)
    dynamics = ThermalNRPMDDynamics(potential, [1.7], 2., 1, .01, 1, native=False)
    with pytest.raises(ValueError, match="gradient"):
        thermal_flux_side(ensemble, dynamics, lambda q: (0., np.ones(3)), steps=2)


@pytest.mark.parametrize("kwargs", [
    {"nstates": 0}, {"beta": 0}, {"nuclear_step": 0},
    {"centroid_step": np.inf}, {"masses": [0.]},
])
def test_sampler_invalid_parameters(kwargs):
    values = dict(potential=MatrixModel([[0.]]), masses=[1.], beta=2.,
                  q=np.zeros((1, 1, 3)), nstates=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        ThermalNRPMDSampler(**values)
