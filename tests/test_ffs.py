"""Direct FFS: exact finite-state references, genealogy and durable restart."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.dynamics import State
from pyrpmd.ffs import ForwardFluxSampler, FFSUnresolvedError, combine_ffs_replicates
from pyrpmd.transition_interface import estimate_effective_flux


def state(x=0.0, *, beads=1, step=0, time=0.0):
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = x
    p = np.arange(3 * beads, dtype=float).reshape(beads, 1, 3) / 7
    return State(q, p, step, time)


def walk(s, rng):
    x = s.q[0, 0, 0]
    s.q[:, 0, 0] = np.clip(x + (1 if rng.random() < 0.5 else -1), -1, 3)


def sampler(seed=7, **kwargs):
    params = dict(
        dt=0.25, basin_a=-0.5, basin_b=2.5, interfaces=(0.0, 1.0),
        dynamics_id="reflecting-symmetric-walk-v1", coordinate_id="centroid-x-v1",
        flux_steps=200, trials_per_interface=30, max_trial_steps=1000, seed=seed,
    )
    params.update(kwargs)
    return ForwardFluxSampler(walk, lambda q: q[:, 0, 0].mean(), **params)


def assert_results_equal(first, second):
    assert first.configuration == second.configuration
    assert first.configuration_fingerprint == second.configuration_fingerprint
    assert first.run_identity == second.run_identity
    assert first.status == second.status
    assert first.rng_state == second.rng_state
    for name in ("flux_values", "flux_events", "flux_a_history", "flux_unknown_history"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    for a, b in zip(first.stages, second.stages):
        for name in ("outcomes", "parent_indices", "root_indices", "steps"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for a, b in zip(first.bank_roots, second.bank_roots):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first.banks, second.banks):
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            np.testing.assert_array_equal(sa.q, sb.q)
            np.testing.assert_array_equal(sa.p, sb.p)
            assert sa.step == sb.step and sa.time == sb.time
    assert first.summary() == second.summary()


def test_flux_convention_matches_independent_trace_analysis():
    ffs = sampler()
    initial = state(-1, beads=4, step=37, time=9.0)
    result = ffs.run(initial)
    reference = estimate_effective_flux(
        result.flux_values, basin_a=-0.5, basin_b=2.5, first_interface=0.0, dt=0.25,
    )
    summary = result.summary(block_size=20)
    for key in ("effective_crossings", "a_history_time_au", "flux_au", "flux_s_inv"):
        assert summary[key] == reference[key]
    assert result.complete and result.status == "complete"
    assert summary["rate_standard_error_au"] is None
    assert summary["flux_diagnostics"]["blocks"] == 10
    assert result.flux_final_state.step == 237
    assert result.flux_final_state.time == 59.0
    assert initial.step == 37 and initial.time == 9.0
    for bank in result.banks:
        for crossing in bank:
            # FFS retains all bead momenta, without Maxwell refresh or reversal.
            np.testing.assert_array_equal(crossing.p, initial.p)
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("operations", [0, 73, 200, 217, 241, 260])
def test_checkpoint_resume_exact_in_flux_and_each_stage(tmp_path, operations):
    expected = sampler(125).run(state(-1, beads=3))
    split = sampler(125)
    split.run(state(-1, beads=3), n_operations=operations)
    path = tmp_path / "ffs.npz"
    split.save_checkpoint(path)
    resumed = sampler(999).load_checkpoint(path)
    actual = resumed.advance()
    assert_results_equal(expected, actual)
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_arrays_and_state_banks_do_not_alias_live_sampler():
    ffs = sampler()
    snapshot = ffs.run(state(-1), n_operations=200)
    with pytest.raises(ValueError):
        snapshot.flux_values[0] = 10
    snapshot.banks[0][0].q[:] = 90
    snapshot.banks[0][0].p[:] = 80
    actual = ffs.advance()
    expected = sampler().run(state(-1))
    assert_results_equal(expected, actual)


def test_jump_across_all_interfaces_is_zero_step_success():
    def jump(s, rng):
        s.q[:, 0, 0] = 4 if s.q[0, 0, 0] < 0 else -1
    ffs = ForwardFluxSampler(
        jump, lambda q: q[0, 0, 0], dt=1, basin_a=-0.5, basin_b=3,
        interfaces=(0, 1, 2), dynamics_id="alternating-jump-v1", coordinate_id="x-v1",
        flux_steps=8, trials_per_interface=10,
    )
    result = ffs.run(state(-1))
    assert result.summary()["flux_au"] == 1
    assert result.summary()["overall_crossing_probability"] == 1
    for stage in result.stages:
        assert stage.outcomes.all() and np.all(stage.steps == 0)


def test_trials_choose_parent_uniformly_and_preserve_genealogy():
    result = sampler(23, trials_per_interface=1000).run(state(-1))
    for i, stage in enumerate(result.stages):
        parents = stage.parent_indices
        np.testing.assert_array_equal(stage.root_indices, result.bank_roots[i][parents])
        np.testing.assert_array_equal(result.bank_roots[i + 1], stage.root_indices[stage.outcomes])
        counts = np.bincount(parents, minlength=len(result.banks[i]))
        expected = len(parents) / len(counts)
        # Multinomial goodness of fit is independent of implementation details.
        statistic = np.sum((counts - expected) ** 2 / expected)
        assert statistic < len(counts) + 6 * np.sqrt(2 * len(counts))


def test_unresolved_trial_is_explicit_and_rolls_back_rng_and_records():
    def freeze_after_crossing(s, rng):
        rng.random()
        s.q[:, 0, 0] = 1
    ffs = ForwardFluxSampler(
        freeze_after_crossing, lambda q: q[0, 0, 0], dt=1, basin_a=-1, basin_b=3,
        interfaces=(0, 2), dynamics_id="frozen-v1", coordinate_id="x-v1",
        flux_steps=1, trials_per_interface=5, max_trial_steps=4,
    )
    before = ffs.run(state(-1), n_operations=1)
    with pytest.raises(FFSUnresolvedError, match="not counted as a failure") as caught:
        ffs.advance(1)
    assert caught.value.steps == 4
    assert caught.value.final_state.q[0, 0, 0] == 1
    after = ffs.result()
    assert_results_equal(before, after)
    assert after.summary()["rate_estimate_au"] is None


def test_numerical_error_restores_one_operation_rng_and_state():
    def broken(s, rng):
        s.q[:] = 123
        rng.random()
        raise ArithmeticError("solver failed")
    ffs = ForwardFluxSampler(
        broken, lambda q: q[0, 0, 0], dt=1, basin_a=-1, basin_b=3,
        interfaces=(0,), dynamics_id="bad-v1", coordinate_id="x-v1",
        flux_steps=1, trials_per_interface=5,
    )
    ffs.initialize(state(-1))
    before = ffs.result()
    with pytest.raises(ArithmeticError, match="solver failed"):
        ffs.advance()
    assert_results_equal(before, ffs.result())


@pytest.mark.parametrize("dt", [1.0, 1e-280])
def test_extinct_runs_are_included_in_replicate_errors_without_selection_bias(dt):
    # One flux step can give a crossing or no crossing. A crossing jumps to B,
    # so each complete run estimates either zero or one. Both must be included.
    def jump(s, rng):
        s.q[:, 0, 0] = 3 if rng.random() < 0.5 else -1
    results = []
    for seed in range(20):
        ffs = ForwardFluxSampler(
            jump, lambda q: q[0, 0, 0], dt=dt, basin_a=-1, basin_b=2,
            interfaces=(0,), dynamics_id="bernoulli-jump-v1", coordinate_id="x-v1",
            flux_steps=1, trials_per_interface=2, seed=seed,
        )
        results.append(ffs.run(state(-1)))
    assert {r.status for r in results} == {"complete", "no_flux_crossings"}
    combined = combine_ffs_replicates(results, block_size=2)
    estimates = np.array([r.summary()["rate_estimate_au"] for r in results])
    assert np.any(estimates == 0) and np.any(estimates == 1 / dt)
    assert combined["rate_estimate_au"] == np.mean(estimates)
    assert combined["rate_standard_error_au"] == pytest.approx(
        (estimates * dt).reshape(-1, 2).mean(axis=1).std(ddof=1) / np.sqrt(10) / dt,
    )
    assert combined["overall_crossing_probability"] is None
    json.dumps(combined, allow_nan=False)


def test_empty_success_bank_terminates_without_fake_downstream_trials():
    def returning(s, rng):
        # The extra coordinate carries complete Markov memory.
        s.q[0, 0, 0] = 1 if s.q[0, 0, 1] == 0 else -1
        s.q[0, 0, 1] += 1
    ffs = ForwardFluxSampler(
        returning, lambda q: q[0, 0, 0], dt=1, basin_a=-1, basin_b=3,
        interfaces=(0, 2), dynamics_id="returning-v1", coordinate_id="x-v1",
        flux_steps=1, trials_per_interface=5,
    )
    result = ffs.run(state(-1))
    assert result.status == "no_successes" and result.complete
    assert len(result.stages[0].outcomes) == 5
    assert len(result.stages[1].outcomes) == 0
    assert result.summary()["conditional_probabilities"] == [0, None]
    assert result.summary()["rate_estimate_au"] == 0
    assert result.summary()["rate_s_inv"] is None


@pytest.mark.parametrize("change", [
    {"dt": 0.3}, {"basin_a": -0.4}, {"basin_b": 2.6}, {"interfaces": (0.0, 1.1)},
    {"dynamics_id": "different"}, {"coordinate_id": "different"},
    {"flux_steps": 201}, {"trials_per_interface": 31}, {"max_trial_steps": 999},
])
def test_checkpoint_rejects_every_scientific_option_change(tmp_path, change):
    path = tmp_path / "identity.npz"
    original = sampler()
    original.run(state(-1), n_operations=25)
    original.save_checkpoint(path)
    other = sampler(**change)
    with pytest.raises(CheckpointError, match="scientific identity"):
        other.load_checkpoint(path)
    assert other._state is None


@pytest.mark.parametrize("corrupt", [
    "event", "history", "ancestry", "bank", "status", "rng", "count", "run_identity",
    "initial_clock", "bank_clock", "flux_clock",
])
def test_semantically_invalid_checkpoint_rejected_transactionally(tmp_path, corrupt):
    original = sampler()
    original.run(state(-1))
    path = tmp_path / "bad.npz"
    original.save_checkpoint(path)
    with load_checkpoint(path) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(data["metadata"].item())
    if corrupt == "event":
        data["flux_events"][0] = not data["flux_events"][0]
    elif corrupt == "history":
        metadata["history"] = 0
    elif corrupt == "ancestry":
        data["stage_0_parent_indices"][0] = 99999
    elif corrupt == "bank":
        data["bank_1_q"][0, :, 0, 0] = -5
    elif corrupt == "status":
        metadata["status"] = "collecting_flux"
    elif corrupt == "rng":
        metadata["rng_state"] = {"bit_generator": "not-a-generator"}
    elif corrupt == "run_identity":
        metadata["run_identity"] = "not-the-run-identity"
    elif corrupt == "initial_clock":
        metadata["initial_time"] += 1
    elif corrupt == "bank_clock":
        data["bank_1_step"][0] += 1
    elif corrupt == "flux_clock":
        metadata["state_time"] += 1
    else:
        data["stage_0_steps"][0] = -1
    data["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(path, kind="PyRPMD.ffs.v1", **data)
    target = sampler(95)
    before = target.run(state(-1), n_operations=7)
    with pytest.raises(CheckpointError):
        target.load_checkpoint(path)
    assert_results_equal(before, target.result())


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.nan])
def test_invalid_trial_and_flux_counts_rejected(value):
    for name in ("flux_steps", "trials_per_interface", "max_trial_steps"):
        with pytest.raises(ValueError):
            sampler(**{name: value})


@pytest.mark.parametrize("interfaces", [[], [0, 0], [1, 0], [-1], [3], [np.nan], [True]])
def test_invalid_interfaces_rejected(interfaces):
    with pytest.raises(ValueError):
        sampler(interfaces=interfaces)


def test_changed_live_configuration_and_invalid_callbacks_rejected():
    ffs = sampler()
    ffs.initialize(state(-1))
    ffs.max_trial_steps = 2000
    with pytest.raises(ValueError, match="configuration changed"):
        ffs.advance()
    bad = sampler()
    bad._coordinate = lambda q: np.array([1, 2])
    with pytest.raises(ValueError):
        bad.initialize(state(-1))
    with pytest.raises(ValueError, match="identity string"):
        sampler(dynamics_id="")


def double_well_chain(dimensions):
    """Finite Metropolis process with an independently solved transition matrix."""
    x = np.linspace(-1.5, 1.5, 7)
    y = np.array([-1.0, 0.0, 1.0]) if dimensions == 2 else np.array([0.0])
    points = np.array([(a, b) for a in x for b in y])
    beta = 2.0
    energy = (points[:, 0] ** 2 - 1) ** 2 + 0.5 * points[:, 1] ** 2
    if dimensions == 2:
        energy += 0.25 * points[:, 0] * points[:, 1]
    transition = np.zeros((len(points), len(points)))
    ny = len(y)
    for i in range(len(points)):
        ix, iy = divmod(i, ny)
        for axis in range(dimensions):
            for sign in (-1, 1):
                jx, jy = (ix + sign, iy) if axis == 0 else (ix, iy + sign)
                if 0 <= jx < len(x) and 0 <= jy < ny:
                    j = jx * ny + jy
                    probability = min(1.0, np.exp(-beta * (energy[j] - energy[i]))) / (2 * dimensions)
                    transition[i, j] += probability
        transition[i, i] = 1 - transition[i].sum()
    stationary = np.exp(-beta * energy)
    stationary /= stationary.sum()
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-15)
    in_a, in_b = points[:, 0] <= -1, points[:, 0] >= 1
    middle = ~(in_a | in_b)
    committor = in_b.astype(float)
    committor[middle] = np.linalg.solve(
        np.eye(middle.sum()) - transition[np.ix_(middle, middle)],
        transition[np.ix_(middle, in_b)].sum(axis=1),
    )
    population_a_history = stationary @ (1 - committor)
    flux = np.sum(stationary[in_a, None] * transition[np.ix_(in_a, ~in_a)]) / population_a_history
    rate = np.sum(stationary[in_a] * (transition[in_a] @ committor)) / population_a_history
    lookup = {tuple(point): i for i, point in enumerate(points)}
    cumulative = np.cumsum(transition, axis=1)
    cumulative[:, -1] = 1
    def step(s, rng):
        index = lookup[tuple(s.q[0, 0, :2])]
        chosen = np.searchsorted(cumulative[index], rng.random(), side="right")
        s.q[0, 0, :2] = points[chosen]
    def initial(seed):
        q = np.zeros((1, 1, 3))
        q[0, 0, :2] = points[np.random.default_rng(seed).choice(len(points), p=stationary)]
        return State(q, np.zeros_like(q))
    return step, initial, flux, rate


@pytest.mark.parametrize("dimensions", [1, 2])
def test_double_well_rate_matches_exact_markov_chain_with_replicate_errors(dimensions):
    step, initial, exact_flux, exact_rate = double_well_chain(dimensions)
    runs = []
    for seed in range(8):
        ffs = ForwardFluxSampler(
            step, lambda q: q[0, 0, 0], dt=1, basin_a=-1, basin_b=1,
            interfaces=(-0.75, -0.25, 0.25), dynamics_id=f"{dimensions}d-double-well-metropolis-v1",
            coordinate_id="x-v1", flux_steps=6000, trials_per_interface=500,
            max_trial_steps=1000, seed=4812 + seed,
        )
        runs.append(ffs.run(initial(1591 + seed)))
    combined = combine_ffs_replicates(runs)
    assert combined["rate_standard_error_au"] is not None
    assert abs(combined["rate_estimate_au"] - exact_rate) < 4 * combined["rate_standard_error_au"]
    assert abs(combined["flux_au"] - exact_flux) < 4 * combined["flux_standard_error_au"]
    assert abs(combined["overall_crossing_probability"] - exact_rate / exact_flux) < (
        4 * combined["crossing_probability_standard_error"]
    )


def test_irreversible_cycle_needs_no_reverse_dynamics():
    # Directed cycle 0->1->2->3->0 is nonreversible. Its A-history residence
    # before hitting B=3 lasts exactly three steps; every first crossing succeeds.
    def cycle(s, rng):
        s.q[0, 0, 0] = (s.q[0, 0, 0] + 1) % 4
    ffs = ForwardFluxSampler(
        cycle, lambda q: q[0, 0, 0], dt=1, basin_a=0, basin_b=3,
        interfaces=(0, 1, 2), dynamics_id="directed-cycle-v1", coordinate_id="x-v1",
        flux_steps=400, trials_per_interface=20,
    )
    result = ffs.run(state(0))
    assert result.summary()["flux_au"] == pytest.approx(1 / 3)
    assert result.summary()["rate_estimate_au"] == pytest.approx(1 / 3)
    assert result.summary()["conditional_probabilities"] == [1, 1, 1]


def test_replicates_reject_duplicate_partial_and_mismatched_runs():
    first = sampler(1).run(state(-1))
    with pytest.raises(ValueError, match="duplicate"):
        combine_ffs_replicates([first, first])
    partial = sampler(2).run(state(-1), n_operations=20)
    with pytest.raises(ValueError, match="complete"):
        combine_ffs_replicates([first, partial])
    different = sampler(2, flux_steps=201).run(state(-1))
    with pytest.raises(ValueError, match="identical"):
        combine_ffs_replicates([first, different])


def test_mt19937_generator_state_roundtrip(tmp_path):
    expected = sampler(np.random.Generator(np.random.MT19937(751))).run(state(-1))
    split = sampler(np.random.Generator(np.random.MT19937(751)))
    split.run(state(-1), n_operations=215)
    path = tmp_path / "mt19937.npz"
    split.save_checkpoint(path)
    resumed = sampler(np.random.Generator(np.random.MT19937(999))).load_checkpoint(path)
    actual = resumed.advance()
    for key in ("flux_values", "flux_events"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert actual.summary() == expected.summary()
    np.testing.assert_array_equal(actual.rng_state["state"]["key"], expected.rng_state["state"]["key"])
    assert actual.rng_state["state"]["pos"] == expected.rng_state["state"]["pos"]


def test_time_flux_overflow_and_unrepresentable_clock_are_explicit():
    with pytest.raises(ValueError, match="horizon overflow"):
        sampler(dt=1e308)
    def jump(s, rng):
        s.q[0, 0, 0] = 4
    tiny = ForwardFluxSampler(
        jump, lambda q: q[0, 0, 0], dt=1e-310, basin_a=-1, basin_b=3,
        interfaces=(0,), dynamics_id="jump-v1", coordinate_id="x-v1",
        flux_steps=1, trials_per_interface=1,
    )
    with pytest.raises(ValueError, match="flux time or value overflow"):
        tiny.run(state(-1)).summary()
    cannot_advance = sampler(dt=1e-30)
    cannot_advance.initialize(state(-1, time=1))
    before = cannot_advance.result()
    with pytest.raises(ValueError, match="does not advance"):
        cannot_advance.advance(1)
    assert_results_equal(before, cannot_advance.result())
