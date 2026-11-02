"""Conditional ring measure, constraints, transactional sampling and restart."""

import copy
import hashlib

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.potentials.base import PESResult, Potential
from pyrpmd.qcmd_constraints import ConstrainedQuasiCentroidSampler, KIND
from pyrpmd.qcmd_geometry import TriatomicCoordinates


class BondPotential(Potential):
    def __init__(self, curvature=.4):
        self.geometry = TriatomicCoordinates([(0, 1, 2)], 3)
        self.curvature = curvature
        self.calls = []

    def evaluate(self, q):
        self.calls.append(len(q))
        s = self.geometry.values(q)
        derivative = np.zeros_like(s)
        derivative[:, :, 0] = self.curvature * (s[:, :, 0] - 2.)
        return PESResult(.5 * self.curvature * (s[:, 0, 0] - 2.)**2,
                         self.geometry.forces(q, derivative))


def initial_ring(beads=4):
    q = np.array([[0., 0., 0.], [2., 0., 0.],
                  [1.8 * np.cos(1.7), 1.8 * np.sin(1.7), 0.]])
    return np.repeat(q[None], beads, axis=0)


def sampler(**overrides):
    kwargs = dict(potential=BondPotential(), masses=[16., 1., 2.], beta=4.,
                  positions=initial_ring(), bond_step=.07, angle_step=.04,
                  rotation_step=.1, center_step=.08, seed=19,
                  potential_identity=hashlib.sha256(b"bond-curvature=.4;reference=2").hexdigest())
    kwargs.update(overrides)
    return ConstrainedQuasiCentroidSampler(**kwargs)


def test_conditional_sampling_keeps_mean_geometry_and_com_but_moves_internal_modes():
    sample = sampler()
    geometry = TriatomicCoordinates([(0, 1, 2)], 3)
    mass = np.array([16., 1., 2.])
    initial_com = np.sum(initial_ring().mean(axis=0) * mass[:, None], axis=0) / mass.sum()
    frames = sample.sample(300, interval=3)
    means = geometry.mean_values(frames)
    np.testing.assert_allclose(means, np.broadcast_to(sample.target, means.shape), atol=4e-15)
    com = np.einsum("a,fpac->fpc", mass, frames) / mass.sum()
    np.testing.assert_allclose(com.mean(axis=1), np.broadcast_to(initial_com, (len(frames), 3)), atol=3e-15)
    assert np.max(np.std(com, axis=1)) > .01
    assert np.max(np.std(frames, axis=1)) > .03
    metadata = sample.metadata()
    assert metadata["all_unconstrained_coordinates_mobile"]
    assert not metadata["physical_dynamics"]
    assert not metadata["physical_converged"]
    assert not metadata["converged"]
    assert sum(metadata["attempted"]) == 300
    assert all(np.array(metadata["accepted"]) > 0)
    # Incremental proposals call the PES for only the changed beads.
    assert sample.potential.calls[0] == 4
    assert set(sample.potential.calls[1:]).issubset({1, 2})


def test_conditional_measure_matches_independent_one_dimensional_quadrature():
    # Freeze orientations, COM modes, r2 and theta explicitly. For two beads,
    # r1=(a+d,a-d). The exact conditional density is
    # (a^2-d^2)^2 exp[-beta/2*(k+4*omega_P^2*mu)*d^2].
    # mu=m_terminal*(1-m_terminal/M) follows from fixing each bead COM.
    beta, a, curvature, mu = 4., 2., .4, 2. / 3.
    omega = 2 / beta
    coefficient = beta / 2 * (curvature + 4 * omega**2 * mu)

    def density(d):
        return (a*a - d*d)**2 * np.exp(-coefficient * d*d)

    reference = quad(lambda d: d*d * density(d), -a, a, epsabs=1e-12)[0] / quad(density, -a, a, epsabs=1e-12)[0]
    sample = sampler(masses=[1., 1., 1.], positions=initial_ring(2),
                     bond_step=[.45, 0.], angle_step=0., rotation_step=0., center_step=0.)
    sample.sample(1000, interval=1000)
    frames = sample.sample(10000, interval=2)
    distances = np.linalg.norm(frames[:, 0, 1] - frames[:, 0, 0], axis=1)
    assert np.mean((distances - a)**2) == pytest.approx(reference, abs=.015)
    assert not sample.metadata()["all_unconstrained_coordinates_mobile"]
    # Omitting r^2 factors changes the reference beyond this error tolerance.
    wrong = quad(lambda d: d*d*np.exp(-coefficient*d*d), -a, a)[0] / quad(lambda d: np.exp(-coefficient*d*d), -a, a)[0]
    assert abs(reference - wrong) > .03


def test_log_weight_includes_volume_jacobian_and_sum_bead_spring_convention():
    q = initial_ring(2)
    q[0, 1, 0] += .3
    q[1, 1, 0] -= .3
    sample = sampler(positions=q)
    s = TriatomicCoordinates([(0, 1, 2)], 3).values(sample.positions)[:, 0]
    direct_energy = .5*.4*np.sum((s[:, 0] - 2.)**2)
    mass = np.array([16., 1., 2.])
    x = sample.positions
    # For P=2 the same pair occurs twice in the periodic spring sum.
    direct_spring = (2/4.)**2*np.sum(mass[:, None]*(x[0] - x[1])**2)
    jacobian = np.prod(s[:, 0]**2*s[:, 1]**2*np.sin(s[:, 2]))
    assert sample.log_weight == pytest.approx(-4/2*(direct_energy + direct_spring)+np.log(jacobian), abs=2e-15)


def test_checkpoint_restart_is_bitwise_identical_and_records_rng(tmp_path):
    uninterrupted = sampler()
    uninterrupted.sample(37)
    path = tmp_path / "conditional.npz"
    uninterrupted.save_checkpoint(path)
    restored = sampler()
    restored.load_checkpoint(path)
    expected = uninterrupted.sample(61)
    actual = restored.sample(61)
    np.testing.assert_array_equal(actual, expected)
    assert restored.metadata() == uninterrupted.metadata()
    assert restored.rng.bit_generator.state == uninterrupted.rng.bit_generator.state
    with load_checkpoint(path, expected_kind=KIND, require_integrity=True) as saved:
        assert int(saved["step"]) == 37
        assert "rng" in saved
        assert saved.envelope["integrity"] == "verified"


def test_checkpoint_handles_array_valued_rng_state(tmp_path):
    source = sampler(seed=np.random.Generator(np.random.MT19937(37)))
    source.sample(3)
    path = tmp_path / "mt19937.npz"
    source.save_checkpoint(path)
    target = sampler(seed=np.random.Generator(np.random.MT19937(37)))
    target.load_checkpoint(path)
    np.testing.assert_array_equal(target.sample(8), source.sample(8))


@pytest.mark.parametrize("override", [
    {"beta": 4.1}, {"bond_step": .071}, {"angle_step": .041},
    {"rotation_step": .11}, {"center_step": .09}, {"masses": [15., 1., 2.]},
    {"potential_identity": "0"*64},
    {"seed": 819},
])
def test_checkpoint_rejects_changed_science(tmp_path, override):
    before = sampler()
    before.sample(4)
    path = tmp_path / "conditional.npz"
    before.save_checkpoint(path)
    changed = sampler(**override)
    with pytest.raises(ValueError, match="scientific identity"):
        changed.load_checkpoint(path)


def test_pes_failure_restores_rng_state_and_counters():
    class Failing(BondPotential):
        fail = False

        def evaluate(self, q):
            if self.fail:
                raise RuntimeError("injected PES failure")
            return super().evaluate(q)

    potential = Failing()
    actual = sampler(potential=potential)
    reference = sampler()
    potential.fail = True
    state = actual.positions
    rng = copy.deepcopy(actual.rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="injected"):
        actual.step()
    np.testing.assert_array_equal(actual.positions, state)
    assert actual.rng.bit_generator.state == rng
    assert actual.step_count == 0
    potential.fail = False
    np.testing.assert_array_equal(actual.sample(20), reference.sample(20))


def test_failed_checkpoint_load_is_transactional(tmp_path):
    actual = sampler()
    actual.sample(7)
    path = tmp_path / "conditional.npz"
    actual.save_checkpoint(path)
    with load_checkpoint(path) as saved:
        arrays = {name: saved[name] for name in saved}
    arrays["internal"][0, 0] += .1
    atomic_savez(path, kind=KIND, **arrays)
    before = actual.positions
    metadata = actual.metadata()
    rng = copy.deepcopy(actual.rng.bit_generator.state)
    with pytest.raises(ValueError, match="constraints"):
        actual.load_checkpoint(path)
    np.testing.assert_array_equal(actual.positions, before)
    assert actual.metadata() == metadata
    assert actual.rng.bit_generator.state == rng


def test_one_bead_sampling_rotates_a_rigid_constrained_molecule():
    actual = sampler(positions=initial_ring(1))
    result = actual.sample(20)
    np.testing.assert_allclose(TriatomicCoordinates([(0, 1, 2)], 3).mean_values(result),
                               np.broadcast_to(actual.target, (20, 1, 3)), atol=1e-15)
    assert actual.metadata()["attempted"] == [0, 0, 20]
    assert not np.array_equal(result[0], result[-1])


@pytest.mark.parametrize("options", [
    {"target": [2.1, 1.8, 1.7]}, {"target": [2., 1.8, np.pi]},
    {"potential_identity": "not a content hash"}, {"pbc": True},
    {"beta": 0}, {"masses": [1., 0., 1.]},
    {"bond_step": -1}, {"angle_step": True},
    {"seed": True}, {"seed": -1},
    {"bond_step": 0, "angle_step": 0, "rotation_step": 0, "center_step": 0},
])
def test_invalid_or_unsupported_sampling_settings_fail(options):
    with pytest.raises(ValueError):
        sampler(**options)


@pytest.mark.parametrize("attribute,value", [
    ("required_bond_constraints", ((0, 1, 2.),)),
    ("pbc", [True, True, True]), ("cell", np.eye(3)),
    ("metadata", {"translation_invariant": False}),
])
def test_conditional_sampler_rejects_hidden_base_requirements(attribute, value):
    potential = BondPotential()
    setattr(potential, attribute, value)
    with pytest.raises(ValueError, match="constraints|nonperiodic|translation-invariant"):
        sampler(potential=potential)
