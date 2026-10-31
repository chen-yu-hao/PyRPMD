"""Equilibrium orientational kernel: measure, detailed balance and ring scaling."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.brhcl.sampling import orientational_metropolis
from pyrpmd.potentials import PESResult, PotentialError


class LinearOrientationPotential:
    """Synthetic exact oracle V=-a(z_Cl-z_H); forces are analytic."""
    def __init__(self, strength=2., minimum=0.):
        self.strength = strength
        self.metadata = {"minimum_pair_distance_bohr": minimum}
        self.evaluated = []

    def evaluate(self, q):
        self.evaluated.append(q.copy())
        force = np.zeros_like(q)
        force[:, 2, 2] = self.strength
        force[:, 1, 2] = -self.strength
        return PESResult(-self.strength*(q[:, 2, 2]-q[:, 1, 2]), force)


def ring(beads, collapsed=False):
    q = np.repeat(np.array([[3.,0.,0.], [0.,0.,0.], [0.,0.,1.]])[None], beads, axis=0)
    if not collapsed:
        q += np.random.default_rng(402).normal(scale=.05, size=q.shape)
    return q


def distances(q):
    c = q.mean(axis=0)
    return np.linalg.norm(c[[0,2]]-c[1], axis=1)


@pytest.mark.parametrize("beads", [1,8,16])
def test_noncollapsed_ring_springs_both_centroid_cvs_and_action_preserved(beads):
    q = ring(beads)
    original = q.copy()
    potential = LinearOrientationPotential()
    moved, report = orientational_metropolis(potential, q, 1.7, np.random.default_rng(618), trials=128)
    assert 0 < report["accepted"] < report["attempted"]
    assert report["invalid_domain"] == 0
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(moved[:, :2], original[:, :2])
    np.testing.assert_allclose(distances(moved), distances(q), atol=3e-15, rtol=0)
    before = np.sum((np.roll(q, -1, axis=0)-q)**2, axis=-1)
    after = np.sum((np.roll(moved, -1, axis=0)-moved)**2, axis=-1)
    np.testing.assert_allclose(after, before, atol=2e-15, rtol=0)
    assert report["initial_action"] == pytest.approx(1.7/beads * potential.evaluate(original).energy.sum(), abs=1e-14)
    assert report["final_action"] == pytest.approx(1.7/beads * potential.evaluate(moved).energy.sum(), abs=1e-14)


def test_sequential_acceptance_and_detailed_balance_log_ratio():
    q = ring(8)
    _, report = orientational_metropolis(LinearOrientationPotential(), q, 1.3,
                                          np.random.default_rng(62), trials=128, record_trace=True)
    current = report["initial_action"]
    assert 2 < report["accepted"] < 126
    for action, ratio, log_uniform, accepted in zip(report["proposal_actions"], report["log_acceptance_ratios"],
                                                   report["log_uniforms"], report["accepted_flags"]):
        assert ratio == pytest.approx(current-action, abs=1e-14)
        # Uniform-orbit proposals have q(x|y)/q(y|x)=1. This is the exact
        # Metropolis detailed-balance identity log(alpha_xy/alpha_yx)=log(pi_y/pi_x).
        assert min(0., ratio)-min(0., -ratio) == pytest.approx(current-action, abs=1e-14)
        assert accepted == (log_uniform < min(0., ratio))
        if accepted:
            current = action
    assert report["final_action"] == current


def test_batching_changes_no_proposals_acceptance_or_rng_draws():
    q = ring(8)
    a, da = orientational_metropolis(LinearOrientationPotential(), q, 1.1,
                                     np.random.default_rng(362), trials=73, record_trace=True)
    b, db = orientational_metropolis(LinearOrientationPotential(), q, 1.1,
                                     np.random.default_rng(362), trials=73, batch_size=1, record_trace=True)
    np.testing.assert_array_equal(a, b)
    assert da == db


def test_only_declared_finite_distance_core_is_rejected_without_pes_calls():
    q = np.array([[[1.,0.,0.], [0.,0.,0.], [-1.,0.,0.]]])
    potential = LinearOrientationPotential(strength=0., minimum=.3)
    _, report = orientational_metropolis(potential, q, 1., np.random.default_rng(19), trials=512)
    assert 0 < report["invalid_domain"] < 512
    assert report["accepted"] == report["attempted"]-report["invalid_domain"]
    evaluated = np.concatenate(potential.evaluated)
    assert np.min(np.linalg.norm(evaluated[:, 0]-evaluated[:, 2], axis=1)) >= .3
    assert len(evaluated) == 1+512-report["invalid_domain"]
    q[:, 2] = q[:, 0]
    with pytest.raises(PotentialError, match="Initial configuration"):
        orientational_metropolis(potential, q, 1., np.random.default_rng(19), trials=1)


@pytest.mark.parametrize("failure_status", [2,3,7])
def test_undeclared_pes_failures_abort_instead_of_becoming_rejections(failure_status):
    class FailedPotential(LinearOrientationPotential):
        def evaluate(self, q):
            result = super().evaluate(q)
            result.status[-1] = failure_status
            result.energy[-1] = np.nan
            result.forces[-1] = np.nan
            result.messages = tuple("undeclared failure" for _ in q)
            return result
    with pytest.raises(PotentialError, match="PES frames failed"):
        orientational_metropolis(FailedPotential(), ring(8), 1., np.random.default_rng(9), trials=16)


def test_exact_angular_equilibrium_and_beta_over_p_for_collapsed_rings():
    # On a unit sphere, V=-2 cos(theta), beta=1: Haar measure gives
    # E[cos(theta)] = coth(2)-1/2, with no extra hand-added sin(theta).
    reports = []
    for beads in (1,8):
        _, report = orientational_metropolis(LinearOrientationPotential(), ring(beads, collapsed=True), 1.,
                                             np.random.default_rng(20836), trials=24000, record_trace=True)
        reports.append(report)
    np.testing.assert_array_equal(reports[0]["accepted_flags"], reports[1]["accepted_flags"])
    np.testing.assert_array_equal(reports[0]["proposal_actions"], reports[1]["proposal_actions"])
    current, cosines = reports[0]["initial_action"], []
    for action, accepted in zip(reports[0]["proposal_actions"], reports[0]["accepted_flags"]):
        if accepted:
            current = action
        cosines.append(-current/2.)
    expected = 1/np.tanh(2.)-.5
    assert np.mean(cosines[1000:]) == pytest.approx(expected, abs=.015)


def test_no_momentum_rng_calls_and_zero_trials_is_identity():
    class UniformOnly:
        def __init__(self):
            self.generator = np.random.default_rng(618)
            self.shapes = []
        def random(self, shape):
            self.shapes.append(shape)
            return self.generator.random(shape)
        def normal(self, *args, **kwargs):
            raise AssertionError("orientation kernel must not draw Gaussian momenta")
    rng = UniformOnly()
    original = ring(8)
    moved, report = orientational_metropolis(LinearOrientationPotential(), original, 1., rng, trials=0)
    np.testing.assert_array_equal(moved, original)
    assert report["attempted"] == report["accepted"] == report["invalid_domain"] == 0
    assert rng.shapes == [(0,3), 0]


@pytest.mark.parametrize("kwargs", [{"beta": 0.}, {"beta": np.inf}, {"beta": True},
                                    {"trials": -1}, {"trials": True}, {"atom": 1},
                                    {"atom": 3}, {"anchor": -1}, {"batch_size": 0}])
def test_bad_kernel_configuration_is_rejected(kwargs):
    configuration = {"beta": 1., "trials": 1}
    configuration.update(kwargs)
    with pytest.raises(ValueError):
        orientational_metropolis(LinearOrientationPotential(), ring(1), rng=np.random.default_rng(1), **configuration)
