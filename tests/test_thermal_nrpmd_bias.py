"""Static-umbrella factors, analytic centroid statistics and safe continuation."""

import copy

import numpy as np
import pytest

from pyrpmd.bias import UmbrellaBias, MetadynamicsBias
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState
from pyrpmd.potentials.base import PotentialError
from pyrpmd.thermal_nrpmd import (
    ThermalNRPMDDynamics, ThermalNRPMDSampler, ThermalRateCorrelation, thermal_flux_side,
    thermal_population_rate, thermal_weight,
)


class HarmonicMatrix:
    def __init__(self, spring=0.7, states=1):
        self.spring, self.states = spring, states

    def evaluate(self, q):
        identity = np.eye(self.states)
        energy = 0.5 * self.spring * np.sum(q * q, axis=(1, 2))
        matrix = energy[:, None, None] * identity
        if self.states == 2:
            matrix += np.array([[0.1, 0.2], [0.2, -0.1]])
        gradients = self.spring * q[:, None, None] * identity[None, :, :, None, None]
        return DiabaticResult(matrix, gradients)


def state(beads=3, states=2, seed=8):
    rng = np.random.default_rng(seed)
    return NRPMDState(
        rng.normal(size=(beads, 1, 3)), np.zeros((beads, 1, 3)),
        mapping_q=rng.normal(size=(beads, states)),
        mapping_p=rng.normal(size=(beads, states)),
    )


@pytest.mark.parametrize("beads", [1, 2, 7])
def test_umbrella_log_factor_is_physical_beta_not_beta_times_beads(beads):
    beta = 2.8
    parent = state(beads=beads)
    cv = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
    center = np.array([0.3, -0.4])
    stiffness = np.array([[1.4, 0.3], [0.3, 0.8]])
    bias = UmbrellaBias(cv, center, stiffness)
    potential = HarmonicMatrix(states=2)
    raw = thermal_weight(potential, parent, [1.7], beta)
    biased = thermal_weight(potential, parent, [1.7], beta, bias=bias)
    delta = parent.q.mean(axis=0)[0, :2] - center
    expected = -beta * 0.5 * delta @ stiffness @ delta
    assert biased.log_density - raw.log_density == pytest.approx(expected, abs=1e-12)
    assert biased.sign == raw.sign
    assert biased.log_abs_mapping_weight == raw.log_abs_mapping_weight
    np.testing.assert_array_equal(biased.electronic_populations, raw.electronic_populations)


def test_umbrella_retains_negative_signed_weights_and_nodes():
    potential = HarmonicMatrix(states=2)
    bias = UmbrellaBias(Cartesian(0), [0.0], 1.3)
    for seed in range(100):
        parent = state(seed=seed)
        raw = thermal_weight(potential, parent, [1.7], 2.3)
        if raw.sign == -1:
            break
    else:
        pytest.fail("test needs a negative SEO trace contribution")
    biased = thermal_weight(potential, parent, [1.7], 2.3, bias=bias)
    assert biased.sign == -1
    np.testing.assert_array_equal(raw.electronic_populations, biased.electronic_populations)
    parent.mapping_q[:] = 0
    node = thermal_weight(potential, parent, [1.7], 2.3, bias=bias)
    assert node.sign == 0 and node.log_density == -np.inf


@pytest.mark.parametrize("beads", [1, 5])
def test_biased_harmonic_centroid_and_internal_mode_curvatures(beads):
    # Independent Gaussian Hessian: a centroid restraint changes only k=0.
    beta, mass, spring, restraint = 2.3, 1.7, 0.6, 1.4
    parent = state(beads=beads, states=1)
    parent.q[:] = 0
    potential = HarmonicMatrix(spring=spring)
    bias = UmbrellaBias(Cartesian(0), [0.0], restraint)

    def action(point):
        trial = parent.copy()
        trial.q[:, 0, 0] = point
        return -thermal_weight(potential, trial, [mass], beta, bias=bias).log_density

    epsilon = 1e-3
    basis = np.eye(beads)
    hessian = np.empty((beads, beads))
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            hessian[i, j] = (
                action(epsilon * (ei + ej)) - action(epsilon * (ei - ej))
                - action(epsilon * (-ei + ej)) + action(-epsilon * (ei + ej))
            ) / (4 * epsilon**2)
    frequencies = 2 * beads / beta * np.sin(np.pi * np.arange(beads) / beads)
    curvature = beta / beads * (spring + mass * frequencies**2)
    curvature[0] += beta / beads * restraint
    np.testing.assert_allclose(np.linalg.eigvalsh(hessian), np.sort(curvature), atol=4e-9)


def test_sampler_biased_centroid_matches_exact_harmonic_statistics():
    beads, beta, spring, restraint, center = 3, 2.0, 0.7, 1.4, 0.8
    sampler = ThermalNRPMDSampler(
        HarmonicMatrix(spring), [1.7], beta, np.zeros((beads, 1, 3)), 1,
        seed=749, nuclear_step=0.45, centroid_step=0.6,
        bias=UmbrellaBias(Cartesian(0), [center], restraint),
    )
    ensemble = sampler.sample(1500, burn_in=250, stride=2)
    centroid = np.array([parent.q.mean(axis=0)[0, 0] for parent in ensemble.states])
    assert centroid.mean() == pytest.approx(restraint * center / (spring + restraint), abs=0.055)
    assert centroid.var() == pytest.approx(1 / (beta * (spring + restraint)), rel=0.14)
    # Mapping independence proposals must still sample the SEO x² e^-x² law;
    # accidentally omitting the bias in those moves would alter their odds.
    x = np.array([parent.mapping_q for parent in ensemble.states])
    assert np.mean(x * x) == pytest.approx(1.5, rel=0.11)
    assert ensemble.sampling_diagnostics["static_bias"] is True
    assert "static_umbrella" in ensemble.sampling_diagnostics["distribution"]
    np.testing.assert_array_equal(ensemble.signs, 1.0)


def test_bias_is_detached_from_caller_and_public_copy():
    original = UmbrellaBias(Cartesian(0), [0.25], 1.3)
    other = copy.deepcopy(original)
    potential = HarmonicMatrix(states=2)
    first = ThermalNRPMDSampler(potential, [1.7], 2.0, np.zeros((2, 1, 3)), 2,
                               seed=18, bias=original)
    second = ThermalNRPMDSampler(potential, [1.7], 2.0, np.zeros((2, 1, 3)), 2,
                                seed=18, bias=other)
    original.center[:] = 100
    original.stiffness[:] = 200
    original.cv.axis = 1
    detached = first.bias
    detached.center = np.array([-999.0])
    detached.cv.axis = 2
    for _ in range(3):
        first.sweep()
        second.sweep()
    for key in ("q", "x", "y"):
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    assert first.snapshot()["static_bias"] == second.snapshot()["static_bias"]
    with pytest.raises(AttributeError):
        first.bias = other


def test_biased_checkpoint_reproduces_and_rejects_changed_bias_semantics():
    potential = HarmonicMatrix(states=2)
    q = np.zeros((2, 1, 3))
    bias = UmbrellaBias(Cartesian(0), [0.25], 1.3)
    sampler = ThermalNRPMDSampler(potential, [1.7], 2.0, q, 2, seed=18, bias=bias)
    sampler.sweep()
    checkpoint = sampler.snapshot()
    expected = sampler.sample(3, stride=2)
    sampler.restore(checkpoint)
    actual = sampler.sample(3, stride=2)
    for left, right in zip(actual.states, expected.states):
        for key in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
    for changed in (
        None,
        UmbrellaBias(Cartesian(0, "y"), [0.25], 1.3),
        UmbrellaBias(Cartesian(0), [0.3], 1.3),
        UmbrellaBias(Cartesian(0), [0.25], 1.4),
    ):
        mismatch = ThermalNRPMDSampler(potential, [1.7], 2.0, q, 2, seed=18, bias=changed)
        before_q = mismatch.q.copy()
        with pytest.raises(ValueError, match="parameter mismatch"):
            mismatch.restore(checkpoint)
        np.testing.assert_array_equal(mismatch.q, before_q)
    # Old unbiased snapshots remain valid after this optional API extension.
    plain = ThermalNRPMDSampler(potential, [1.7], 2.0, q, 2, seed=18)
    old = plain.snapshot()
    del old["static_bias"]
    plain.restore(old)


def test_failed_biased_sweep_preserves_rng_state_and_counters():
    class Failing(HarmonicMatrix):
        calls = 0
        fail_at = None

        def evaluate(self, q):
            self.calls += 1
            if self.calls == self.fail_at:
                raise PotentialError("injected PES failure")
            return super().evaluate(q)

    potential = Failing(states=2)
    sampler = ThermalNRPMDSampler(potential, [1.7], 2.0, np.zeros((2, 1, 3)), 2,
                                 seed=18, bias=UmbrellaBias(Cartesian(0), [0.3], 1.3))
    before = sampler.snapshot()
    potential.fail_at = potential.calls + 2
    with pytest.raises(PotentialError, match="injected"):
        sampler.sweep()
    after = sampler.snapshot()
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, after[key])
        else:
            assert value == after[key]


def test_adaptive_or_overridden_bias_callbacks_rejected():
    potential = HarmonicMatrix()
    parent = state(states=1)
    dynamic = MetadynamicsBias(Cartesian(0), [0.1], 0.1, 2.0)
    with pytest.raises(ValueError, match="static UmbrellaBias"):
        thermal_weight(potential, parent, [1.0], 2.0, bias=dynamic)
    bias = UmbrellaBias(Cartesian(0), [0.0], 1.0)
    bias.cv.value_and_jacobian = lambda q: None
    with pytest.raises(ValueError, match="built-in CVs"):
        ThermalNRPMDSampler(potential, [1.0], 2.0, parent.q, 1, bias=bias)


def test_unresolvable_bias_energy_rejected_before_sampling():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite scalar energy"):
            ThermalNRPMDSampler(
                HarmonicMatrix(), [1.0], 2.0, np.zeros((1, 1, 3)), 1,
                bias=UmbrellaBias(Cartesian(0), [1e300], 1.0),
            )


def test_biased_parents_cannot_silently_enter_equilibrium_rate_estimators():
    potential = HarmonicMatrix(states=2)
    ensemble = ThermalNRPMDSampler(
        potential, [1.7], 2.0, np.zeros((1, 1, 3)), 2,
        bias=UmbrellaBias(Cartesian(0), [0.3], 1.3),
    ).sample(2)
    dynamics = ThermalNRPMDDynamics(potential, [1.7], 2.0, 1, 0.01, 2, native=False)
    with pytest.raises(ValueError, match="must be reweighted"):
        thermal_population_rate(ensemble, dynamics, 0, 1, steps=2)
    # A loose metadata flag must not bypass the biased-parent guard.
    ensemble.sampling_diagnostics["reweighted"] = True
    with pytest.raises(ValueError, match="must be reweighted"):
        thermal_flux_side(ensemble, dynamics, lambda q: (q[0, 0], np.ones_like(q)), steps=2)


def test_mbar_resampled_parents_cannot_claim_independent_block_uncertainty():
    correlation = ThermalRateCorrelation(
        np.arange(5.0), np.full((80, 5), 0.1), np.ones(80), np.ones(80),
        "test_curve",
        metadata={"sampling": {"parent_resampling": "mbar_multinomial",
                               "uncertainty_requires_source_chain_bootstrap": True}},
    )
    summary = correlation.summarize(start=0.0, stop=4.0, block_size=10)
    assert summary["candidate_rate"] == pytest.approx(0.1)
    assert summary["interval_shape_stable"] is True
    assert summary["standard_error"] is None
    assert summary["plateau_detected"] is False
    assert summary["uncertainty_reason"] == "source_chain_bootstrap_required_after_reweighting"
