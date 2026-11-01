"""Batched layout is an adapter to the canonical scalar operator stack."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.batched import BatchedDynamics, BatchedState
from pyrpmd.dynamics import Dynamics
from pyrpmd.engine import ForceSplitting
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.contraction import SplitPotential


def thermostat(mode, beads):
    if mode == "nhc":
        return {"mode": mode, "tau": .4, "chain_length": 3, "substeps": 2}
    if mode in ("gle", "piglet"):
        gle = {"A": [[1., 2.], [-2., 1.]], "C": np.eye(2).tolist()}
        if mode == "piglet":
            gle.update(A=np.repeat(np.asarray(gle["A"])[None], beads, axis=0).tolist(),
                       beta=2., beads=beads,
                       provenance="test identity covariance; not calibrated PIGLET parameters")
        return {"mode": mode, "gle": gle}
    return mode


def state():
    rng = np.random.default_rng(818)
    q, p = rng.normal(scale=.1, size=(2, 4, 2, 3)), rng.normal(size=(2, 4, 2, 3))
    q[:, :, 1], p[:, :, 1] = 0., 0.
    return BatchedState(q, p, [3, 7], [.03, .07])


def parameters(mode, split=False):
    potential = HarmonicPotential(.4)
    kwargs = {}
    if split:
        potential = SplitPotential(HarmonicPotential(.3), HarmonicPotential(.1), 4, 2)
        kwargs["force_splitting"] = ForceSplitting(3)
    return dict(potential=potential, masses=[2., 3.], beta=2., nbeads=4, dt=.01,
                native=False, integrator="normal-mode-bab", thermostat=thermostat(mode, 4),
                fixed_atoms=[1], fixed_positions=np.zeros((2, 3)), **kwargs)


@pytest.mark.parametrize("mode", ["andersen", "pile-l", "nhc", "svr", "gle", "piglet"])
@pytest.mark.parametrize("split", [False, True])
def test_batched_operators_match_independent_scalar_engines(mode, split):
    if mode != "andersen":
        pytest.importorskip("pyrpmd._thermal")
    initial = state()
    actual = initial.copy()
    kwargs = parameters(mode, split)
    batch = BatchedDynamics(**kwargs, seed=44)
    result = batch.propagate_block(actual, 4)
    assert result.success
    assert "composed-batched-pes" in result.backend
    for lane, reference in enumerate(initial.to_states()):
        dynamics = Dynamics(**kwargs, seed=np.random.SeedSequence([44, lane]),
                            loop_backend="python")
        for _ in range(4):
            dynamics.step(reference)
        np.testing.assert_allclose(actual.q[lane], reference.q, atol=2e-14, rtol=0)
        np.testing.assert_allclose(actual.p[lane], reference.p, atol=2e-14, rtol=0)
        assert actual.step[lane] == reference.step
        assert actual.time[lane] == reference.time
        assert result.energies[-1, lane] == pytest.approx(dynamics.energy(reference), abs=2e-12)
    np.testing.assert_array_equal(actual.q[:, :, 1], 0.)
    np.testing.assert_array_equal(actual.p[:, :, 1], 0.)


def test_batched_operator_checkpoint_and_inactive_lane_resume():
    pytest.importorskip("pyrpmd._thermal")
    kwargs = parameters("gle", split=True)
    initial = state()
    full = initial.copy()
    reference = BatchedDynamics(**kwargs, seed=19)
    reference.propagate_block(full, 5)
    partial = initial.copy()
    first = BatchedDynamics(**kwargs, seed=19)
    first.propagate_block(partial, 2)
    saved = json.loads(json.dumps(first.operator_state_dict()))
    resumed = BatchedDynamics(**kwargs, seed=19)
    resumed.load_operator_state_dict(saved)
    resumed.propagate_block(partial, 3, active=np.array([False, True]))
    np.testing.assert_array_equal(partial.q[1], full.q[1])
    np.testing.assert_array_equal(partial.p[1], full.p[1])
    resumed.propagate_block(partial, 3, active=np.array([True, False]))
    np.testing.assert_array_equal(partial.q, full.q)
    np.testing.assert_array_equal(partial.p, full.p)


def test_observation_failure_does_not_commit_a_thermostat_or_rng_step():
    pytest.importorskip("pyrpmd._thermal")
    kwargs = parameters("nhc")
    actual = state()
    batch = BatchedDynamics(**kwargs, seed=29)
    batch.propagate_block(actual, 1)
    previous = actual.copy()
    operators = copy.deepcopy(batch.operator_state_dict())

    def stop(lane, state):
        if lane == 0:
            raise ValueError("invalid observation")
        return False

    result = batch.propagate_block(actual, 1, stop=stop, raise_on_error=False)
    assert result.failed.tolist() == [True, False]
    np.testing.assert_array_equal(actual.q[0], previous.q[0])
    np.testing.assert_array_equal(actual.p[0], previous.p[0])
    assert batch.operator_state_dict()["lanes"]["0"] == operators["lanes"]["0"]
    reference = BatchedDynamics(**kwargs, seed=29)
    reference.load_operator_state_dict(operators)
    reference.propagate_block(previous, 1)
    batch.propagate_block(actual, 1, active=np.array([True, False]))
    np.testing.assert_array_equal(actual.q, previous.q)
    np.testing.assert_array_equal(actual.p, previous.p)


@pytest.mark.parametrize("native", [False, True])
def test_vv_mts_checks_inner_spring_stability_and_keeps_outer_clock(native):
    kwargs = parameters("none", split=True)
    kwargs.update(integrator="velocity-verlet", dt=.75, native=native,
                  force_splitting=ForceSplitting(4))
    # P=4, beta=2 gives max(omega)=4: the outer map has dt*omega=3,
    # while each actual spring step has h*omega=.75 and is stable.
    unsplit = {key: value for key, value in kwargs.items() if key != "force_splitting"}
    with pytest.raises(ValueError, match="stability"):
        Dynamics(**unsplit)
    initial = state()
    actual = initial.copy()
    batch = BatchedDynamics(**kwargs)
    assert batch.dt == .75
    assert batch.integrator.dt == .75 / 4
    result = batch.propagate_block(actual, 3)
    assert result.success
    for lane, reference in enumerate(initial.to_states()):
        dynamics = Dynamics(**kwargs, loop_backend="python")
        for _ in range(3):
            dynamics.step(reference)
        np.testing.assert_allclose(actual.q[lane], reference.q, atol=2e-13, rtol=0)
        np.testing.assert_allclose(actual.p[lane], reference.p, atol=2e-13, rtol=0)
        assert actual.step[lane] == initial.step[lane] + 3
        assert actual.time[lane] == reference.time


def test_composed_array_batches_bound_threads_without_changing_lane_rng(monkeypatch):
    import pyrpmd.trajectory_batch as batching

    initial = state()
    full = initial.copy()
    kwargs = parameters("andersen")
    BatchedDynamics(**kwargs, seed=71).propagate_block(full, 3)
    # A small test cap exercises the same partition path used for >256 lanes.
    monkeypatch.setattr(batching, "MAX_TRAJECTORY_BATCH_SIZE", 1)
    actual = initial.copy()
    result = BatchedDynamics(**kwargs, seed=71).propagate_block(actual, 3)
    assert result.success
    np.testing.assert_array_equal(actual.q, full.q)
    np.testing.assert_array_equal(actual.p, full.p)
