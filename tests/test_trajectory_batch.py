"""Scalar workflow/RNG/checkpoint contracts survive shared PES batches."""

import copy
from pathlib import Path
import threading

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.potentials import HarmonicPotential, PESResult, PotentialError
from pyrpmd.potentials.base import validate_result
from pyrpmd.trajectory_batch import (
    map_trajectory_tasks, make_trajectory_potential, run_trajectory_batch,
    trajectory_batch_size,
)


class CountingHarmonic:
    metadata = {"trajectory_batch_safe": True}

    def __init__(self):
        self.calls = []
        self.potential = HarmonicPotential(.4)

    def evaluate(self, q):
        self.calls.append((threading.get_ident(), q.shape, q[:, 0, 0].copy()))
        return self.potential.evaluate(q)


def request_worker(task):
    provider = make_trajectory_potential(task)
    if task.get("fail_before_call"):
        raise RuntimeError("before first PES call")
    values = []
    for _ in range(task["steps"]):
        result = validate_result(provider.evaluate(task["q"]), task["q"].shape)
        values.append((result.energy.copy(), result.forces.copy()))
    task.get("completed", []).append(task["id"])
    return values


def requests(lengths):
    return [dict(id=i, q=np.full((i + 1, 1, 3), i + .2), steps=steps)
            for i, steps in enumerate(lengths)]


def test_rendezvous_has_deterministic_batches_and_releases_finished_lanes():
    provider = CountingHarmonic()
    tasks = requests([0, 1, 3, 2])
    outputs = run_trajectory_batch(request_worker, tasks, provider)
    assert [shape[0] for _, shape, _ in provider.calls] == [9, 7, 3]
    assert len({thread for thread, _, _ in provider.calls}) == 1
    for task, values in zip(tasks, outputs):
        expected = provider.potential.evaluate(task["q"])
        assert len(values) == task["steps"]
        for energy, forces in values:
            np.testing.assert_array_equal(energy, expected.energy)
            np.testing.assert_array_equal(forces, expected.forces)


def test_failed_lane_before_first_request_does_not_abandon_other_checkpoints():
    provider = CountingHarmonic()
    completed = []
    tasks = requests([1, 2, 3])
    for task in tasks:
        task["completed"] = completed
    tasks[0]["fail_before_call"] = True
    with pytest.raises(RuntimeError, match="before first PES call"):
        run_trajectory_batch(request_worker, tasks, provider)
    assert sorted(completed) == [1, 2]
    assert [shape[0] for _, shape, _ in provider.calls] == [5, 5, 3]


def test_thread_start_failure_releases_already_waiting_worker(monkeypatch):
    import pyrpmd.trajectory_batch as batching
    original = batching.ThreadPoolExecutor.submit
    calls = 0

    def fail_second_submit(pool, function, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("thread startup failed")
        return original(pool, function, *args, **kwargs)

    monkeypatch.setattr(batching.ThreadPoolExecutor, "submit", fail_second_submit)
    with pytest.raises(RuntimeError, match="thread startup failed"):
        run_trajectory_batch(request_worker, requests([3, 3]), CountingHarmonic())


def test_coordinator_interrupt_releases_waiting_workers(monkeypatch):
    import pyrpmd.trajectory_batch as batching

    def interrupt(batch):
        raise KeyboardInterrupt("coordinator interrupted")

    monkeypatch.setattr(batching._Rendezvous, "serve", interrupt)
    with pytest.raises(KeyboardInterrupt, match="coordinator interrupted"):
        run_trajectory_batch(request_worker, requests([3, 3]), CountingHarmonic())


def test_frame_error_is_local_and_remaining_lanes_complete():
    class Broken(CountingHarmonic):
        def evaluate(self, q):
            result = super().evaluate(q)
            failed = q[:, 0, 0] < 1
            result.status[failed] = 6
            result.messages = tuple("outside domain" if value else "" for value in failed)
            result.energy[failed] = np.nan
            result.forces[failed] = np.nan
            return result

    provider = Broken()
    completed = []
    tasks = requests([3, 2, 3])
    for task in tasks:
        task["completed"] = completed
    with pytest.raises(PotentialError, match="outside domain"):
        run_trajectory_batch(request_worker, tasks, provider)
    assert sorted(completed) == [1, 2]
    assert [shape[0] for _, shape, _ in provider.calls] == [6, 5, 3]


def test_malformed_whole_batch_fails_all_waiting_requests_without_retry():
    class Broken(CountingHarmonic):
        def evaluate(self, q):
            super().evaluate(q)
            return PESResult(np.zeros(1), np.zeros_like(q))

    provider = Broken()
    with pytest.raises(PotentialError, match="shape mismatch"):
        run_trajectory_batch(request_worker, requests([3, 3]), provider)
    assert len(provider.calls) == 1


def test_rpc_contraction_remains_local_while_physical_components_batch():
    from pyrpmd.potentials.contraction import SplitPotential

    fast, slow = CountingHarmonic(), CountingHarmonic()
    potential = SplitPotential(fast, slow, nbeads=4, contracted_beads=2)
    tasks = [dict(id=i, q=np.random.default_rng(i).normal(size=(4, 1, 3)), steps=2)
             for i in range(3)]
    expected = [potential.evaluate(task["q"]) for task in tasks]
    fast.calls.clear()
    slow.calls.clear()
    results = run_trajectory_batch(request_worker, tasks, potential)
    assert [shape[0] for _, shape, _ in fast.calls] == [12, 12]
    assert [shape[0] for _, shape, _ in slow.calls] == [6, 6]
    for reference, samples in zip(expected, results):
        for energy, forces in samples:
            np.testing.assert_array_equal(energy, reference.energy)
            np.testing.assert_array_equal(forces, reference.forces)


@pytest.mark.parametrize("change", [
    {"trajectory_batch_size": True}, {"trajectory_batch_size": 0},
    {"trajectory_batch_size": 2.0}, {"trajectory_batch_size": 2, "bead_workers": 2},
    {"trajectory_batch_size": 257},
])
def test_invalid_batch_sizes_and_bead_proxies_are_rejected(change):
    with pytest.raises(ValueError):
        trajectory_batch_size({"parallel": change})


def test_oversized_low_level_batch_fails_before_starting_threads(monkeypatch):
    import pyrpmd.trajectory_batch as batching

    def forbidden(*args, **kwargs):
        raise AssertionError("threads started before batch-size validation")

    monkeypatch.setattr(batching, "ThreadPoolExecutor", forbidden)
    with pytest.raises(ValueError, match="at most 256"):
        run_trajectory_batch(request_worker, [{"steps": 0}] * 257, CountingHarmonic())


def test_mutable_cells_and_forced_native_loop_are_rejected():
    with pytest.raises(ValueError, match="fixed cell"):
        trajectory_batch_size({"parallel": {"trajectory_batch_size": 2},
                               "ensemble": {"type": "npt"}})
    with pytest.raises(ValueError, match="Python PES orchestration"):
        trajectory_batch_size({"parallel": {"trajectory_batch_size": 2},
                               "dynamics": {"loop_backend": "native"}})


def _payload(path):
    with load_checkpoint(path) as saved:
        return {key: np.array(saved[key], copy=True) for key in saved.files
                if not key.startswith("_")}


def _compare_checkpoints(left, right):
    for path in sorted(Path(left).rglob("*.npz")):
        other = Path(right) / path.relative_to(left)
        a, b = _payload(path), _payload(other)
        # Envelope checksums and metadata can describe differing orchestration;
        # all actual numerical arrays and the RNG identity remain identical.
        for key, value in a.items():
            if value.dtype.kind in "fiub" or key == "rng":
                np.testing.assert_array_equal(value, b[key], err_msg=f"{path.name}:{key}")


@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
@pytest.mark.parametrize("workers", [1, 2])
def test_umbrella_batch_matches_scalar_rng_samples_and_completed_resume(tmp_path, thermostat, workers):
    from test_fes_pile_l import config, system
    from pyrpmd.workflows.fes import run_fes

    cfg = config()
    cfg["sampling"]["thermostat"] = thermostat
    cfg["sampling"]["replicas"] = workers
    cfg["dynamics"]["loop_backend"] = "python"
    if thermostat == "andersen":
        cfg["sampling"].pop("gamma_centroid")
    baseline = run_fes(cfg, system(), tmp_path / "scalar", return_data=True)
    cfg["parallel"] = {"trajectory_batch_size": 2, "trajectory_workers": workers}
    result = run_fes(cfg, system(), tmp_path / "batch", return_data=True)
    _compare_checkpoints(tmp_path / "scalar", tmp_path / "batch")
    for a, b in zip(baseline["streams"], result["streams"]):
        for key in ("samples", "energies", "last_q", "last_p"):
            np.testing.assert_array_equal(a[key], b[key])
    again = run_fes(cfg, system(), tmp_path / "batch", return_data=True)
    np.testing.assert_array_equal(again["data"]["free_energy"], result["data"]["free_energy"])


def test_input_exposes_batch_size_and_rejects_incompatible_bead_transport():
    import tomllib
    from pyrpmd.input import InputError, JobSpec

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "examples/fes/harmonic2d.inp").read_text())
    config["parallel"] = {"trajectory_batch_size": 3}
    resolved, _ = JobSpec.from_mapping(config).resolve()
    assert resolved["parallel"]["trajectory_batch_size"] == 3
    config["parallel"]["bead_workers"] = 2
    with pytest.raises(InputError, match="bead_workers=1"):
        JobSpec.from_mapping(config).resolve()


def test_failed_batched_umbrella_resumes_with_scalar_rng_and_samples(tmp_path, monkeypatch):
    from test_fes_pile_l import config, system
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.workflows.fes import run_fes

    cfg = config()
    cfg["dynamics"]["loop_backend"] = "python"
    run_fes(cfg, system(), tmp_path / "reference")
    cfg["parallel"] = {"trajectory_batch_size": 2}
    original = Dynamics.energy
    tripped = False
    lock = threading.Lock()

    def fail_once(dynamics, state, bias=None):
        nonlocal tripped
        with lock:
            if state.step == 14 and not tripped:
                tripped = True
                raise RuntimeError("sampling observation failure")
        return original(dynamics, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "energy", fail_once)
        with pytest.raises(RuntimeError, match="sampling observation failure"):
            run_fes(cfg, system(), tmp_path / "resumed")
    assert tripped
    cfg["parallel"]["trajectory_batch_size"] = 1
    run_fes(cfg, system(), tmp_path / "resumed")
    _compare_checkpoints(tmp_path / "reference", tmp_path / "resumed")


def test_drpmd_batches_saved_launches_without_changing_physical_results(tmp_path):
    from test_drpmd import config, system
    from pyrpmd.workflows.drpmd import run_drpmd

    cfg = config()
    cfg["dynamics"] = {"loop_backend": "python"}
    cfg["sampling"]["parents"] = 2
    cfg["collision"].update(trajectories=5, max_steps=6)
    baseline = run_drpmd(cfg, system(), tmp_path / "scalar")
    cfg["parallel"]["trajectory_batch_size"] = 3
    actual = run_drpmd(cfg, system(), tmp_path / "batch")
    for a, b in zip(baseline["records"], actual["records"]):
        for key in ("trajectory_id", "parent_id", "status", "steps", "launch"):
            assert a[key] == b[key]
    _compare_checkpoints(tmp_path / "scalar", tmp_path / "batch")
    cfg["resume"] = True
    again = run_drpmd(cfg, system(), tmp_path / "batch")
    assert again["records"] == actual["records"]


def test_site_pairs_reuse_scalar_checkpoints_and_signed_flux(tmp_path):
    from test_site_rate import configuration, hydrogen
    from pyrpmd.workflows import site_rate as site

    cfg, system = configuration(), hydrogen()
    cfg["recrossing"]["child_steps"] = 5
    q = np.repeat(system.geometry[None], cfg["beads"], axis=0)
    q[:, 0, 0] = .6
    tasks = [dict(config=cfg, system=system, q=q, chain=0, parent=0,
                  pair=i, output=str(tmp_path / "scalar")) for i in range(3)]
    expected = [site._child_task(task) for task in tasks]
    batched = copy.deepcopy(cfg)
    batched["parallel"] = {"trajectory_batch_size": 2}
    grouped = [dict(task, config=batched, output=str(tmp_path / "batch")) for task in tasks]
    results = map_trajectory_tasks(site._child_task, grouped, batched)
    for a, b in zip(expected, results):
        for key in ("numerator", "denominator", "endpoint_energy_error_kbt"):
            np.testing.assert_array_equal(a[key], b[key])
    _compare_checkpoints(tmp_path / "scalar", tmp_path / "batch")
    # Regroup completed pair checkpoints without starting any force request.
    batched["parallel"]["trajectory_batch_size"] = 3
    again = map_trajectory_tasks(site._child_task, grouped, batched)
    for a, b in zip(results, again):
        np.testing.assert_array_equal(a["numerator"], b["numerator"])


def test_site_rate_batches_real_parents_and_children_via_public_options(tmp_path, monkeypatch):
    from test_site_rate import configuration, hydrogen, sampled_harmonic_pmf
    from pyrpmd.workflows import site_rate as site

    cfg, system = configuration(), hydrogen()
    cfg["recrossing"].update(parent_samples=2, child_steps=8)
    sampled = sampled_harmonic_pmf(cfg, system)
    monkeypatch.setattr(site, "run_fes", lambda *args, **kwargs: copy.deepcopy(sampled))
    expected = site.run_site_rate(cfg, system, tmp_path / "scalar")
    cfg["parallel"] = {"trajectory_batch_size": 2}
    cfg["recrossing"]["child_batch_pairs"] = 3
    actual = site.run_site_rate(cfg, system, tmp_path / "batch")
    assert actual["child_pairs"] == 8
    for key in ("kappa", "k_qtst_s", "rate_estimate_s", "maximum_endpoint_energy_error_kbt"):
        assert actual[key] == expected[key]
    _compare_checkpoints(tmp_path / "scalar", tmp_path / "batch")
