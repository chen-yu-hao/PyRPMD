"""Minus path laws, synchronous statistics and portable RETIS continuation."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import jsonable
from pyrpmd.potentials import PESResult
from pyrpmd.retis import ReplicaExchangeTIS, estimate_retis_flux
from pyrpmd.transition_interface import PathGenerationError, TISPath


class Quartic:
    def evaluate(self, x):
        energy = np.sum((x[:, :, 0] ** 2 - 1) ** 2, axis=1)
        energy += np.sum(x[:, :, 1:] ** 2, axis=(1, 2)) / 2
        force = -x.copy()
        force[:, :, 0] = -4 * x[:, :, 0] * (x[:, :, 0] ** 2 - 1)
        return PESResult(energy, force)


def sampler(*, seed=5, minus=True, nbeads=1, **options):
    dyn = Dynamics(Quartic(), [1.], 1., nbeads, .06, native=False)
    settings = dict(basin_a=-1., basin_b=1., interfaces=(-1., -.4),
                    max_path_steps=300, identity={"potential": "quartic-v1", "cv": "centroid-x-v1"})
    settings.update(options)
    return ReplicaExchangeTIS(dyn, Cartesian(0), seed=seed, minus=minus, **settings)


def point(sampler, x, p):
    q = np.zeros(sampler.dynamics.integrator.shape)
    momenta = q.copy()
    q[:, 0, 0], momenta[:, 0, 0] = x, p
    return State(q, momenta)


def seeds(sampler):
    positive = sampler.generate_path(point(sampler, -.4, 1.4), 1)
    negative = sampler.generate_minus_path(point(sampler, -1.1, -.7)) if sampler.minus_enabled else None
    return [positive, positive], negative


def rng_json(rng):
    return json.dumps(jsonable(rng), sort_keys=True)


def same_results(actual, expected):
    assert rng_json(actual.rng_state) == rng_json(expected.rng_state)
    assert actual.summary(block_size=10) == expected.summary(block_size=10)
    for field in ("swap_attempts", "swap_accepts", "walker_labels", "minus_walker_labels"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for first, second in zip(actual.interfaces, expected.interfaces):
        for field in ("crossing", "accepted", "path_frames"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        np.testing.assert_array_equal(first.final_path.q, second.final_path.q)
        np.testing.assert_array_equal(first.final_path.p, second.final_path.p)
        assert first.reasons == second.reasons
    if actual.minus is not None:
        np.testing.assert_array_equal(actual.minus.final_path.q, expected.minus.final_path.q)
        np.testing.assert_array_equal(actual.minus.final_path.p, expected.minus.final_path.p)
        np.testing.assert_array_equal(actual.minus.path_frames, expected.minus.path_frames)
        np.testing.assert_array_equal(actual.minus.accepted, expected.minus.accepted)


@pytest.mark.parametrize("nbeads", [1, 4])
def test_minus_membership_replay_ownership_and_bead_momenta(nbeads):
    retis = sampler(nbeads=nbeads)
    state = point(retis, -1.1, -.7)
    before = state.copy()
    path = retis.generate_minus_path(state)
    assert path.values[0] > -1 and path.values[-1] > -1
    assert np.all(path.values[1:-1] <= -1)
    retis.validate_minus_path(path)
    retis.validate_minus_path(path.reversed())
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert not path.q.flags.writeable
    # Both samplers use the same real-mass Maxwell convention and own RNG.
    other = sampler(nbeads=nbeads)
    np.testing.assert_array_equal(retis._minus._momenta(), other._momenta())
    assert retis._minus.rng is retis.rng


def test_minus_seed_validation_and_configuration_guards():
    with pytest.raises(ValueError, match="first interface"):
        sampler(interfaces=(-.8, -.4))
    with pytest.raises(ValueError, match="boolean"):
        sampler(minus=1)
    retis = sampler()
    with pytest.raises(ValueError, match="inside A"):
        retis.generate_minus_path(point(retis, 0., 1.))
    with pytest.raises(ValueError, match="minus=True"):
        sampler(minus=False).generate_minus_path(point(retis, -1.1, -.7))
    path = retis.generate_minus_path(point(retis, -1.1, -.7))
    q = path.q.copy()
    q[1, 0, 0, 1] += .2
    with pytest.raises(ValueError, match="trajectory"):
        retis.validate_minus_path(TISPath(q, path.p, path.values, path.dt))
    retis.max_path_steps += 1
    with pytest.raises(ValueError, match="configuration changed"):
        retis.sample_interfaces([path, path], n_samples=1)


def test_zero_swap_bounds_and_numerical_errors_do_not_become_accepted(monkeypatch):
    retis = sampler()
    positive, negative = seeds(retis)
    def reject(*args):
        raise PathGenerationError("max_path_steps")
    monkeypatch.setattr(retis, "_zero_proposal", reject)
    m, p, decision = retis.exchange_zero(negative, positive[0])
    assert m is negative and p is positive[0]
    assert not decision.accepted and decision.reason == "max_path_steps"
    def broken(*args):
        raise ArithmeticError("PES failed")
    monkeypatch.setattr(retis, "_zero_proposal", broken)
    with pytest.raises(ArithmeticError, match="PES failed"):
        retis.exchange_zero(negative, positive[0])


def test_zero_exchange_metropolis_acceptance_frequency(monkeypatch):
    retis = sampler()
    positive, negative = seeds(retis)
    monkeypatch.setattr(retis, "_zero_proposal", lambda *args: (negative, positive[0]))
    monkeypatch.setattr(retis, "_zero_log_ratio", lambda *args: -np.log(2.))
    accepted = sum(retis.exchange_zero(negative, positive[0])[2].accepted for _ in range(1200))
    assert .5 - .05 < accepted / 1200 < .5 + .05
    monkeypatch.setattr(retis, "_zero_log_ratio", lambda *args: np.log(2.))
    assert all(retis.exchange_zero(negative, positive[0])[2].accepted for _ in range(20))


def test_mid_sweep_failure_restores_path_bank_counters_and_rng(tmp_path, monkeypatch):
    retis = sampler()
    positive, negative = seeds(retis)
    before = retis.sample_interfaces(positive, minus_path=negative, n_samples=5)
    original = retis.shoot
    def fail_last(path, index):
        if index == 1:
            raise ArithmeticError("numerical failure")
        return original(path, index)
    monkeypatch.setattr(retis, "shoot", fail_last)
    with pytest.raises(ArithmeticError, match="numerical failure"):
        retis.sample_interfaces(n_samples=1)
    same_results(retis.result(), before)
    filename = tmp_path / "after-error.npz"
    retis.save_checkpoint(filename)
    restored = sampler()
    restored.load_checkpoint(filename)
    same_results(restored.result(), before)


@pytest.mark.parametrize("minus", [False, True])
@pytest.mark.parametrize("split", [0, 13])
def test_checkpoint_resume_exact_rng_paths_labels_and_cumulative_statistics(tmp_path, minus, split):
    full = sampler(minus=minus)
    positive, negative = seeds(full)
    expected = full.sample_interfaces(positive, minus_path=negative, n_samples=36, burn_in=3)
    partial = sampler(minus=minus)
    partial.initialize(positive, minus_path=negative)
    if split:
        partial.sample_interfaces(n_samples=split, burn_in=3)
    checkpoint = tmp_path / "retis.npz"
    partial.save_checkpoint(checkpoint)
    restored = sampler(seed=91, minus=minus)
    restored.load_checkpoint(checkpoint)
    actual = restored.sample_interfaces(n_samples=36 - split, burn_in=0 if split else 3)
    same_results(actual, expected)
    with pytest.raises(ValueError, match="burn_in"):
        restored.sample_interfaces(n_samples=1, burn_in=1)


def test_mt19937_checkpoint_and_diagnostics_do_not_consume_rng(tmp_path):
    full = sampler(seed=np.random.Generator(np.random.MT19937(31)))
    positive, negative = seeds(full)
    full.sample_interfaces(positive, minus_path=negative, n_samples=9)
    checkpoint = tmp_path / "mt.npz"
    full.save_checkpoint(checkpoint)
    expected = full.sample_interfaces(n_samples=11)
    restored = sampler(seed=np.random.Generator(np.random.MT19937(1)))
    restored.load_checkpoint(checkpoint)
    state = rng_json(restored.rng.bit_generator.state)
    restored.result().summary(block_size=4)
    assert state == rng_json(restored.rng.bit_generator.state)
    same_results(restored.sample_interfaces(n_samples=11), expected)


@pytest.mark.parametrize("change", [
    {"interfaces": (-1., -.3)}, {"max_path_steps": 301}, {"minus": False},
    {"nbeads": 2}, {"identity": {"potential": "different", "cv": "x"}},
])
def test_checkpoint_identity_mismatch_rejected_transactionally(tmp_path, change):
    original = sampler()
    positive, negative = seeds(original)
    original.sample_interfaces(positive, minus_path=negative, n_samples=4)
    checkpoint = tmp_path / "identity.npz"
    original.save_checkpoint(checkpoint)
    new = sampler(**change)
    before = copy.deepcopy(new.rng.bit_generator.state)
    with pytest.raises(CheckpointError, match="identity"):
        new.load_checkpoint(checkpoint)
    assert before == new.rng.bit_generator.state and new._paths is None


@pytest.mark.parametrize("damage", ["labels", "frames", "rng", "values", "inventory", "sweeps"])
def test_malformed_checkpoint_with_valid_envelope_is_rejected(tmp_path, damage):
    retis = sampler()
    positive, negative = seeds(retis)
    retis.sample_interfaces(positive, minus_path=negative, n_samples=4)
    checkpoint = tmp_path / "damage.npz"
    retis.save_checkpoint(checkpoint)
    with load_checkpoint(checkpoint) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload["metadata"]))
    if damage == "labels":
        payload["current_labels"][:] = 0
    elif damage == "frames":
        payload["minus_frames"][-1] += 1
    elif damage == "rng":
        metadata["rng_state"] = {"invalid": True}
    elif damage == "values":
        payload["path_0_values"][1] += .1
    elif damage == "inventory":
        payload["unexpected"] = np.array([2])
    else:
        metadata["sweeps"] += 1
    payload["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(checkpoint, kind="PyRPMD.retis.v1", **payload)
    before = rng_json(retis.rng.bit_generator.state)
    with pytest.raises(CheckpointError):
        retis.load_checkpoint(checkpoint)
    assert before == rng_json(retis.rng.bit_generator.state)


def test_persistent_checkpoint_needs_explicit_opaque_physics_identity(tmp_path):
    retis = sampler(identity=None)
    positive, negative = seeds(retis)
    retis.initialize(positive, minus_path=negative)
    with pytest.raises(ValueError, match="identity"):
        retis.save_checkpoint(tmp_path / "missing.npz")


def test_joint_block_errors_match_independent_nonlinear_block_estimates():
    retis = sampler()
    positive, negative = seeds(retis)
    result = retis.sample_interfaces(positive, minus_path=negative, n_samples=43)
    data = result.summary(block_size=10)
    crossing = np.column_stack([x.crossing for x in result.interfaces])
    times = (result.minus.path_frames + result.interfaces[0].path_frames - 4) * .06
    block_rates = [np.prod(crossing[i:i + 10].mean(axis=0)) / times[i:i + 10].mean()
                   for i in range(0, 40, 10)]
    assert data["rate_standard_error_block_au"] == pytest.approx(np.std(block_rates, ddof=1) / 2)
    assert data["block_diagnostics"]["samples_excluded_from_block_error"] == 3
    assert result.summary(block_size=100)["rate_standard_error_block_au"] is None
    assert data["rate_s_inv"] is None and not data["physical_converged"]


@pytest.mark.parametrize("minus,plus", [([2], [3]), ([3], [1]), ([3.0], [2]),
                                        ([True], [3]), ([], []), ([3], [3, 4])])
def test_flux_rejects_invalid_path_counts(minus, plus):
    with pytest.raises(ValueError):
        estimate_retis_flux(minus, plus, dt=.1, basin_a=-1, basin_b=1, first_interface=-1)


def test_flux_block_errors_remain_finite_for_extreme_finite_units():
    report = estimate_retis_flux([3, 4, 5, 6], [2, 2, 2, 2],
        dt=1e-200, basin_a=-1, basin_b=1, first_interface=-1, block_size=1)
    assert np.isfinite(report["flux_standard_error_block_au"])
    assert report["flux_standard_error_block_au"] > 0
    json.dumps(report, allow_nan=False)
    ordinary = estimate_retis_flux([3, 4, 5, 6], [2, 2, 2, 2],
        dt=1., basin_a=-1, basin_b=1, first_interface=-1, block_size=1)
    assert report["flux_standard_error_block_au"] * 1e-200 == pytest.approx(
        ordinary["flux_standard_error_block_au"])


def test_result_does_not_relabel_existing_data_after_option_change():
    retis = sampler()
    positive, negative = seeds(retis)
    retis.sample_interfaces(positive, minus_path=negative, n_samples=2)
    retis.max_path_steps += 1
    with pytest.raises(ValueError, match="configuration changed"):
        retis.result()
