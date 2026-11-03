"""Independent two-surface action, harmonic SCI and Marcus-limit checks."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from pyrpmd.constants import ATOMIC_TIME_S, KB_AU
from pyrpmd.golden_rule import (GoldenRuleAction, golden_rule_rate,
                               optimize_golden_rule_instanton)
from pyrpmd.instanton import InstantonError
from pyrpmd.potentials.base import PESResult


class QuadraticSurface:
    def __init__(self, curvature, center, offset=0.0):
        self.curvature = np.asarray(curvature, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.offset = offset
        self.calls = self.frames = 0

    def evaluate(self, q):
        self.calls += 1
        self.frames += len(q)
        x = q.reshape(len(q), -1)-self.center
        gradient = x @ self.curvature
        return PESResult(0.5*np.sum(x*gradient, axis=1)+self.offset,
                         -gradient.reshape(q.shape))


class AnalyticQuadratic(QuadraticSurface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian_calls = 0

    def hessian(self, q):
        self.hessian_calls += 1
        return np.broadcast_to(self.curvature, (len(q), *self.curvature.shape))


def harmonic_setup(p0=16, p1=None, *, epsilon=0.0, temperature=300., dimensions=1,
                   analytic=True, offset=0.0):
    p1 = p0 if p1 is None else p1
    curvature = np.diag([0.02, 0.01 if dimensions == 2 else 0., 0.])
    shift = np.array([1., 0.4 if dimensions == 2 else 0., 0.])
    cls = AnalyticQuadratic if analytic else QuadraticSurface
    potentials = cls(curvature, -shift, offset), cls(curvature, shift, offset-epsilon)
    paths = np.zeros((p0+1, 1, 3)), np.zeros((p1+1, 1, 3))
    mask = np.array([[True, dimensions == 2, False]])
    return potentials, [1822.], temperature, paths, mask


def solve_harmonic(p0=16, p1=None, **options):
    optimizer = options.pop("optimizer", {})
    potentials, masses, temperature, paths, mask = harmonic_setup(p0, p1, **options)
    result = optimize_golden_rule_instanton(*potentials, masses, temperature, *paths,
                                           active_mask=mask, **optimizer)
    return result, potentials


def harmonic_reference(beta, curvature, displacement, mass, epsilon):
    """Exact Gaussian imaginary-time trace and its SCI time saddle.

    This is not a fictitious smooth rate for a finite isolated oscillator;
    the time integral is evaluated only by the stated saddle approximation.
    """
    omega = np.sqrt(np.asarray(curvature)/mass)
    reorganization = 2*np.asarray(curvature)*np.asarray(displacement)**2
    denominator = np.sinh(0.5*beta*omega)

    def action(tau):
        return float(-epsilon*tau+np.sum(reorganization/omega*(
            np.cosh(0.5*beta*omega)-np.cosh(omega*(0.5*beta-tau)))/denominator))

    def derivative(tau):
        return float(-epsilon+np.sum(reorganization*np.sinh(omega*(0.5*beta-tau))/denominator))

    tau = brentq(derivative, 0., beta)
    curvature_time = -float(np.sum(reorganization*omega*np.cosh(omega*(0.5*beta-tau))/denominator))
    log_qr = -float(np.log(2*np.sinh(0.5*beta*omega)).sum())
    return {"tau": tau, "action": action(tau), "curvature": curvature_time,
            "log_qr": log_qr, "log_rate_without_coupling": 0.5*math.log(2*math.pi/-curvature_time)-action(tau),
            "action_at": action}


def test_two_segment_action_all_derivatives_match_independent_finite_differences():
    rng = np.random.default_rng(816)
    n0, n1, beta = 3, 5, 4.7
    matrices = [np.array([[.4, .12, 0.], [.12, .8, 0.], [0., 0., 0.]]),
                np.array([[.7, -.08, 0.], [-.08, .3, 0.], [0., 0., 0.]])]
    potentials = (AnalyticQuadratic(matrices[0], [-.5, .2, 0.]),
                  AnalyticQuadratic(matrices[1], [.8, -.3, 0.], -.07))
    mask = np.array([[True, True, False]])
    action = GoldenRuleAction(*potentials, [3.2], 1/(KB_AU*beta), active_mask=mask)
    q = rng.normal(scale=.4, size=(n0+n1, 1, 3))
    q[:, 0, 2] = 0.

    def split(path):
        return path[:n0+1], np.concatenate((path[n0:], path[:1]))

    tau, step = .31*beta, 1e-6
    value = action.evaluate(*split(q), tau)
    hessian = action.hessian(*split(q), tau)
    for index in np.argwhere(np.broadcast_to(mask, q.shape)):
        index = tuple(index)
        plus, minus = q.copy(), q.copy()
        plus[index] += step
        minus[index] -= step
        vp, vm = action.evaluate(*split(plus), tau), action.evaluate(*split(minus), tau)
        assert value["gradient"][index] == pytest.approx((vp["action_hbar"]-vm["action_hbar"])/(2*step), abs=2e-8)
        column = index[0]*2+index[2]
        np.testing.assert_allclose(hessian[:, column],
                                   (vp["gradient"][:, mask]-vm["gradient"][:, mask]).ravel()/(2*step), atol=2e-8)
    plus, minus = action.evaluate(*split(q), tau+step), action.evaluate(*split(q), tau-step)
    assert value["energy_gap_hartree"] == pytest.approx((plus["action_hbar"]-minus["action_hbar"])/(2*step), abs=2e-8)
    assert value["time_hessian_partial_hartree2"] == pytest.approx(
        (plus["energy_gap_hartree"]-minus["energy_gap_hartree"])/(2*step), rel=1e-8)
    np.testing.assert_allclose(value["mixed_time_gradient"],
                               (plus["gradient"]-minus["gradient"])/(2*step), atol=2e-8)


def test_displaced_harmonic_action_trace_and_schur_converge_to_analytic_quantum_saddle():
    epsilon, temperature, coupling = .012, 300., 1e-5
    beta = 1/(KB_AU*temperature)
    reference = harmonic_reference(beta, [.02, .01], [1., .4], 1822., epsilon)
    errors = []
    for p in (8, 16, 32):
        orbit, potentials = solve_harmonic(p, epsilon=epsilon, dimensions=2)
        rate = golden_rule_rate(orbit, coupling_hartree=coupling, log_reactant_partition=reference["log_qr"])
        expected_log = 2*math.log(coupling)+reference["log_rate_without_coupling"]
        errors.append(abs(math.expm1(rate["log_rate_au"]-expected_log)))
        assert orbit.gradient_max <= 1e-8 and abs(orbit.energy_gap_hartree) <= 1e-10
        assert orbit.time_schur_curvature_hartree2 < 0
        assert orbit.optimizer_diagnostics["physical_hessian_evaluations"] == 2
        assert orbit.optimizer_diagnostics["finite_difference_hessian_evaluations"] == 0
        assert [potential.hessian_calls for potential in potentials] == [1, 1]
        assert orbit.optimizer_diagnostics["force_calls"] == sum(p.calls for p in potentials)
        assert orbit.optimizer_diagnostics["force_configurations"] == sum(p.frames for p in potentials)
        assert orbit.optimizer_diagnostics["spring_preconditioner_dimension"] == 2*p
        assert rate["physical_converged"] is False
    assert errors[0] > 3*errors[1] > 9*errors[2]
    assert errors[-1] < .005
    assert orbit.tau1_au == pytest.approx(reference["tau"], rel=.001)
    assert orbit.action_hbar == pytest.approx(reference["action"], abs=.005)
    assert orbit.time_schur_curvature_hartree2 == pytest.approx(reference["curvature"], rel=.002)
    # For equal curvatures, the exact coordinate Gaussian prefactor is Q0.
    assert orbit.log_coordinate_prefactor == pytest.approx(reference["log_qr"], abs=.002)
    assert orbit.to_dict()["removed_phase_modes"] == 0


def test_high_temperature_limit_recovers_classical_marcus_and_collapsed_paths_are_valid():
    temperature, epsilon, coupling = 1e5, .01, 1e-5
    orbit, _ = solve_harmonic(8, epsilon=epsilon, temperature=temperature)
    beta, lam, omega = 1/(KB_AU*temperature), .04, math.sqrt(.02/1822.)
    log_qr = -math.log(2*math.sinh(beta*omega/2))
    rate = golden_rule_rate(orbit, coupling_hartree=coupling, log_reactant_partition=log_qr)
    marcus = coupling**2*math.sqrt(math.pi*beta/lam)*math.exp(-beta*(lam-epsilon)**2/(4*lam))
    assert math.exp(rate["log_rate_au"]) == pytest.approx(marcus, rel=2e-5)
    assert orbit.tau1_au/beta == pytest.approx((lam-epsilon)/(2*lam), abs=2e-6)
    assert rate["rate"] == pytest.approx(marcus/ATOMIC_TIME_S, rel=2e-5)
    assert np.ptp(orbit.path0_bohr[:, 0, 0]) < 1e-5


def test_nonseparable_unequal_curvature_surfaces_match_exact_harmonic_propagators():
    # Independent continuum reference: integrate the PRODUCT of two exact
    # multidimensional harmonic kernels over their two hopping endpoints.
    # Only a 4x4 endpoint matrix appears here, not a discretized path Hessian.
    mass, temperature = 1822., 300.
    beta = 1/(KB_AU*temperature)
    matrices = [np.array([[.018, .005], [.005, .012]]),
                np.array([[.029, -.006], [-.006, .014]])]
    centers = [np.array([-.8, .3]), np.array([.6, -.4])]
    energies = [0., -.003]
    modes = [np.linalg.eigh(matrix/mass) for matrix in matrices]

    def kernel_reference(tau1):
        diagonal = np.zeros((2, 2))
        cross = np.zeros((2, 2))
        linear = np.zeros(2)
        constant = log_normalization = 0.
        for (squared, rotation), center, energy, duration in zip(
                modes, centers, energies, (beta-tau1, tau1)):
            omega = np.sqrt(squared)
            a = (rotation*(omega/np.tanh(omega*duration))) @ rotation.T
            b = (rotation*(omega/np.sinh(omega*duration))) @ rotation.T
            u = center*np.sqrt(mass)
            diagonal += a
            cross -= b
            linear += (a-b) @ u
            constant += float(u @ (a-b) @ u)+duration*energy
            log_normalization += .5*float(np.log(omega/np.sinh(omega*duration)).sum())
        endpoint_hessian = np.block([[diagonal, cross], [cross, diagonal]])
        source = np.tile(linear, 2)
        action = constant-.5*float(source @ np.linalg.solve(endpoint_hessian, source))
        log_prefactor = log_normalization-.5*np.linalg.slogdet(endpoint_hessian)[1]
        return action, log_prefactor

    optimum = minimize_scalar(lambda f: -kernel_reference(beta*f)[0], bounds=(.05, .95),
                               method="bounded", options={"xatol": 1e-12})
    tau = beta*optimum.x
    action_reference, prefactor_reference = kernel_reference(tau)
    step = beta*1e-4
    curvature_reference = (kernel_reference(tau+step)[0]+kernel_reference(tau-step)[0]
                           -2*action_reference)/step**2
    errors = []
    for n0, n1 in ((12, 20), (24, 40)):
        surfaces = []
        for matrix, center, energy in zip(matrices, centers, energies):
            full = np.zeros((3, 3))
            full[:2, :2] = matrix
            surfaces.append(AnalyticQuadratic(full, [*center, 0.], energy))
        orbit = optimize_golden_rule_instanton(*surfaces, [mass], temperature,
                                               np.zeros((n0+1, 1, 3)), np.zeros((n1+1, 1, 3)),
                                               active_mask=np.array([[True, True, False]]))
        errors.append(abs(orbit.action_hbar-action_reference))
    assert errors[0] > 3.5*errors[1]
    assert errors[-1] < .004
    assert orbit.tau1_au == pytest.approx(tau, rel=.002)
    assert orbit.log_coordinate_prefactor == pytest.approx(prefactor_reference, abs=.002)
    assert orbit.time_schur_curvature_hartree2 == pytest.approx(curvature_reference, rel=.003)
    assert max(abs(gap) for gap in orbit.hopping_energy_gaps_hartree) < 1e-4
    assert orbit.hopping_distance_bohr < 1e-5


def test_unequal_segment_grids_energy_origin_and_forward_reverse_detailed_balance():
    epsilon = .01
    forward, potentials = solve_harmonic(8, 16, epsilon=epsilon)
    mask = np.array([[True, False, False]])
    beta = forward.tau0_au+forward.tau1_au
    reverse = optimize_golden_rule_instanton(potentials[1], potentials[0], [1822.], 300.,
                                            forward.path1_bohr, forward.path0_bohr,
                                            initial_tau_fraction=forward.tau0_au/beta, active_mask=mask)
    assert reverse.action_hbar == pytest.approx(forward.action_hbar, abs=1e-10)
    assert reverse.log_coordinate_prefactor == pytest.approx(forward.log_coordinate_prefactor, abs=1e-10)
    assert reverse.time_schur_curvature_hartree2 == pytest.approx(forward.time_schur_curvature_hartree2, rel=1e-8)
    log_qr = -math.log(2*math.sinh(beta*math.sqrt(.02/1822.)/2))
    kf = golden_rule_rate(forward, coupling_hartree=1e-5, log_reactant_partition=log_qr)
    kr = golden_rule_rate(reverse, coupling_hartree=1e-5, log_reactant_partition=log_qr+beta*epsilon)
    assert kf["log_rate_au"]-kr["log_rate_au"] == pytest.approx(beta*epsilon, abs=1e-8)
    shifted, _ = solve_harmonic(8, 16, epsilon=epsilon, offset=.17,
                               optimizer={"energy_reference_hartree": .17})
    assert shifted.action_hbar == pytest.approx(forward.action_hbar, abs=1e-10)
    assert golden_rule_rate(shifted, coupling_hartree=1e-5,
                            log_reactant_partition=log_qr)["log_rate_au"] == pytest.approx(kf["log_rate_au"], abs=1e-7)


def test_optional_hessian_provider_and_fd_give_same_rate_with_honest_costs():
    analytic, _ = solve_harmonic(8, 12, epsilon=.01)
    numerical, potentials = solve_harmonic(8, 12, epsilon=.01, analytic=False)
    assert numerical.action_hbar == pytest.approx(analytic.action_hbar, abs=1e-10)
    assert numerical.log_coordinate_prefactor == pytest.approx(analytic.log_coordinate_prefactor, abs=1e-8)
    assert numerical.time_schur_curvature_hartree2 == pytest.approx(analytic.time_schur_curvature_hartree2, rel=1e-8)
    assert numerical.optimizer_diagnostics["finite_difference_hessian_evaluations"] == 2
    assert numerical.optimizer_diagnostics["hessian_provider_calls"] == 0
    assert numerical.optimizer_diagnostics["force_calls"] == sum(p.calls for p in potentials)


def test_second_order_coupling_sign_degeneracy_and_logarithmic_underflow():
    orbit, _ = solve_harmonic(8)
    first = golden_rule_rate(orbit, coupling_hartree=1e-5, log_reactant_partition=0.)
    doubled = golden_rule_rate(orbit, coupling_hartree=-2e-5, log_reactant_partition=0., reaction_degeneracy=3)
    assert doubled["rate"] == pytest.approx(12*first["rate"])
    underflow = golden_rule_rate(orbit, coupling_hartree=1e-200, log_reactant_partition=0.)
    assert underflow["rate"] == 0 and underflow["range_status"] == "underflow"
    assert math.isfinite(underflow["log_rate_au"])
    with pytest.raises(ValueError, match="nonzero"):
        golden_rule_rate(orbit, coupling_hartree=0., log_reactant_partition=0.)
    with pytest.raises(ValueError, match="real"):
        golden_rule_rate(orbit, coupling_hartree=1e-5j, log_reactant_partition=0.)
    with pytest.raises(InstantonError, match="converged"):
        golden_rule_rate(replace(orbit, converged=False), coupling_hartree=1e-5, log_reactant_partition=0.)


def test_surface_verified_coupling_cannot_be_changed_by_rate_postprocessing():
    potentials, masses, temperature, paths, mask = harmonic_setup(8)
    potentials[0].coupling_hartree = 1e-5
    potentials[1].coupling_hartree = -1e-5
    orbit = optimize_golden_rule_instanton(*potentials, masses, temperature, *paths, active_mask=mask)
    assert orbit.verified_coupling_hartree == 1e-5
    golden_rule_rate(orbit, coupling_hartree=-1e-5, log_reactant_partition=0.)
    with pytest.raises(ValueError, match="differs"):
        golden_rule_rate(orbit, coupling_hartree=2e-5, log_reactant_partition=0.)
    potentials, masses, temperature, paths, mask = harmonic_setup(8)
    potentials[0].coupling_hartree = 1e-5
    potentials[1].coupling_hartree = 2e-5
    with pytest.raises(ValueError, match="inconsistent"):
        optimize_golden_rule_instanton(*potentials, masses, temperature, *paths, active_mask=mask)
    assert potentials[0].calls == potentials[1].calls == 0


@pytest.mark.parametrize("options, match", [
    ({"regime": "inverted"}, "normal regime"),
    ({"rigid_body": True}, "unsupported"),
    ({"rigid_body": "false"}, "rigid_body"),
    ({"initial_tau_fraction": 0.}, "bounds"),
    ({"initial_tau_fraction": True}, "real"),
    ({"tau_fraction_bounds": [0., 1.]}, "bounds"),
    ({"tau_fraction_bounds": [True, False]}, "bounds"),
    ({"tau_fraction_tolerance": 1.}, "smaller"),
    ({"gradient_tolerance": 0.}, "positive"),
    ({"energy_tolerance_hartree": np.nan}, "positive"),
    ({"max_iterations": True}, "integer"),
    ({"max_tau_iterations": 0}, "positive integer"),
])
def test_malformed_controls_fail_before_any_pes_evaluation(options, match):
    potentials, masses, temperature, paths, mask = harmonic_setup(8)
    with pytest.raises(ValueError, match=match):
        optimize_golden_rule_instanton(*potentials, masses, temperature, *paths,
                                       active_mask=mask, **options)
    assert potentials[0].calls == potentials[1].calls == 0


def test_invalid_shared_endpoints_and_exhausted_coordinate_budget_fail_explicitly():
    potentials, masses, temperature, paths, mask = harmonic_setup(8)
    paths[1][-1, 0, 0] = .1
    with pytest.raises(ValueError, match="reverse endpoints"):
        optimize_golden_rule_instanton(*potentials, masses, temperature, *paths, active_mask=mask)
    assert potentials[0].calls == potentials[1].calls == 0
    paths[1][-1, 0, 0] = 0.
    with pytest.raises(InstantonError, match="coordinate minimization"):
        optimize_golden_rule_instanton(*potentials, masses, temperature, *paths,
                                       active_mask=mask, max_iterations=1)


def test_inverted_or_missing_time_saddle_is_not_clipped_into_normal_regime():
    with pytest.raises(InstantonError, match="no stationary time split"):
        solve_harmonic(8, epsilon=.06, optimizer={"tau_fraction_bounds": (.05, .95)})
    zero = AnalyticQuadratic(np.diag([.02, 0., 0.]), [0., 0., 0.])
    path = np.zeros((9, 1, 3))
    with pytest.raises(InstantonError, match="no stationary time split"):
        optimize_golden_rule_instanton(zero, zero, [1822.], 300., path, path,
                                       active_mask=np.array([[True, False, False]]))


@pytest.mark.parametrize("bias", [0., 1.5e-11])
def test_small_absolute_energy_gap_does_not_prove_a_normal_time_root(bias):
    # The second case is inverted (epsilon/lambda=1.5), despite every gap
    # being below the default 1e-10 absolute energy stopping tolerance. Even
    # the symmetric first case's exactly zero initial gap requires a bracket.
    curvature = np.diag([5e-12, 0., 0.])
    surfaces = (AnalyticQuadratic(curvature, [-1., 0., 0.]),
                AnalyticQuadratic(curvature, [1., 0., 0.], -bias))
    path = np.zeros((9, 1, 3))
    if bias:
        with pytest.raises(InstantonError, match="resolved normal-regime signs"):
            optimize_golden_rule_instanton(*surfaces, [1.], 300., path, path,
                                           active_mask=np.array([[True, False, False]]))
    else:
        orbit = optimize_golden_rule_instanton(*surfaces, [1.], 300., path, path,
                                               active_mask=np.array([[True, False, False]]))
        bracket = orbit.optimizer_diagnostics["resolved_time_bracket"]
        assert bracket["lower_energy_gap_hartree"] > 0
        assert bracket["upper_energy_gap_hartree"] < 0
        assert orbit.tau0_au == orbit.tau1_au


def test_free_rigid_or_internal_zero_modes_are_not_removed_as_adiabatic_phase():
    potentials, masses, temperature, paths, _ = harmonic_setup(8)
    with pytest.raises(InstantonError, match="free rigid/extra zero"):
        optimize_golden_rule_instanton(*potentials, masses, temperature, *paths)
