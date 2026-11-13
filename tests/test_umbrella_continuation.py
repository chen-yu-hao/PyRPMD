"""Neighbor coordinate seeding keeps independent replicas and durable identities."""
import copy
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, reanalyze_fes
from pyrpmd.workflows import adaptive_umbrella as adaptive


def _system():
    return System(("H",), np.array([1.0]), np.zeros((1, 3)))


def _config():
    return {
        "task": "fes", "beta": 1.0, "beads": 2, "dt_au": 0.03,
        "native": False, "seed": 91,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 10, "production_steps": 40,
                     "sample_interval": 1, "andersen_interval": 10, "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"centers": [[-0.2], [0.0], [0.2]], "stiffness": 1.0},
        "adaptive_umbrella": {"enabled": True, "continuation": True,
                              "pilot_steps": 60, "pilot_equilibration_steps": 10,
                              "max_rounds": 2, "max_windows": 8,
                              "target_overlap": 1e-6, "min_effective_samples": 1.0},
        "analysis": {"bins": [6], "bootstrap": 0},
        "output": {"checkpoint_steps": 10},
    }


def _same_streams(left, right):
    for first, second in zip(left["streams"], right["streams"]):
        for key in ("samples", "last_q", "last_p", "energies"):
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_nearest_neighbor_seed_and_independent_production_replicas(tmp_path):
    config = _config()
    result = run_fes(config, _system(), tmp_path, return_data=True)
    pilot_dir = tmp_path / "adaptation/round_000"
    pilot = json.loads((pilot_dir / "continuation.json").read_text())["initializations"]
    assert pilot["1"] == {"source": "reference_geometry"}
    assert pilot["0"]["source_window"] == pilot["2"]["source_window"] == 1
    with load_checkpoint(pilot_dir / pilot["0"]["source_checkpoint"]) as source:
        np.testing.assert_array_equal(pilot["0"]["q"], source["q"])
    production = json.loads((tmp_path / "continuation.json").read_text())["initializations"]
    for window in range(3):
        record = production[str(window)]
        assert record["source"] == "frozen_pilot"
        assert record["source_window"] == window
        with load_checkpoint(tmp_path / record["source_checkpoint"]) as source:
            np.testing.assert_array_equal(record["q"], source["q"])
        states = []
        for replica in range(2):
            path = tmp_path / "checkpoints" / f"window_{window:05d}_replica_{replica:04d}.npz"
            with load_checkpoint(path) as saved:
                assert int(saved["elapsed"]) == 50
                assert json.loads(str(saved["initialization"])) == record
                states.append(saved["p"].copy())
        assert not np.array_equal(*states)
    assert result["summary"]["samples"] == 3 * 2 * 40
    assert result["summary"]["adaptive_umbrella"]["pilot_samples_used_in_production"] == 0
    _same_streams(result, reanalyze_fes(config, _system(), tmp_path, return_data=True))


def test_continuation_resume_uses_persisted_neighbors_after_interrupted_pilot(tmp_path, monkeypatch):
    config = _config()
    reference = run_fes(config, _system(), tmp_path / "reference", return_data=True)
    original = adaptive._sample_window

    def interrupt(task):
        if task["window"] == 2:
            raise RuntimeError("interrupted neighbor pilot")
        return original(task)

    with monkeypatch.context() as patch:
        patch.setattr(adaptive, "_sample_window", interrupt)
        with pytest.raises(RuntimeError, match="interrupted neighbor pilot"):
            run_fes(config, _system(), tmp_path / "resumed")
    actual = run_fes(config, _system(), tmp_path / "resumed", return_data=True)
    _same_streams(reference, actual)
    assert json.loads((tmp_path / "reference/continuation.json").read_text()) == json.loads(
        (tmp_path / "resumed/continuation.json").read_text())


def test_continuation_resume_after_frozen_plan_keeps_pilot_seeds(tmp_path, monkeypatch):
    config = _config()
    reference = run_fes(config, _system(), tmp_path / "reference", return_data=True)
    original = adaptive._record_round

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("frozen before production")

    with monkeypatch.context() as patch:
        patch.setattr(adaptive, "_record_round", interrupt)
        with pytest.raises(RuntimeError, match="frozen before production"):
            run_fes(config, _system(), tmp_path / "resumed")
    actual = run_fes(config, _system(), tmp_path / "resumed", return_data=True)
    _same_streams(reference, actual)


def test_continuation_matches_process_execution(tmp_path):
    config = _config()
    reference = run_fes(config, _system(), tmp_path / "serial", return_data=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        actual = run_fes(config, _system(), tmp_path / "process", executor=executor, return_data=True)
    _same_streams(reference, actual)
    assert json.loads((tmp_path / "serial/continuation.json").read_text()) == json.loads(
        (tmp_path / "process/continuation.json").read_text())


def test_changed_initialization_is_rejected_by_checkpoint_identity(tmp_path):
    config = _config()
    run_fes(config, _system(), tmp_path)
    path = tmp_path / "continuation.json"
    manifest = json.loads(path.read_text())
    manifest["initializations"]["0"]["q"][0][0][0] += 0.1
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError), match="checkpoint physics differs"):
        reanalyze_fes(config, _system(), tmp_path)


@pytest.mark.parametrize("field", ["pilot_equilibration_steps", "equilibration_steps"])
def test_continuation_requires_equilibration(field):
    config = _config()
    config["adaptive_umbrella" if field.startswith("pilot") else "sampling"][field] = 0
    with pytest.raises(ValueError, match="positive pilot and production equilibration"):
        adaptive.adaptive_options(config)


def test_default_initialization_keeps_legacy_path(tmp_path):
    config = _config()
    config["adaptive_umbrella"].pop("continuation")
    old = run_fes(config, _system(), tmp_path / "default", return_data=True)
    explicit = copy.deepcopy(config)
    explicit["adaptive_umbrella"]["continuation"] = False
    new = run_fes(explicit, _system(), tmp_path / "disabled", return_data=True)
    _same_streams(old, new)
    assert not list(tmp_path.rglob("continuation.json"))


def test_production_refinement_only_proposes_a_fresh_plan():
    plan = {"centers": [[-1.0], [1.0]], "stiffness": [[1.0]],
            "options": {"min_effective_samples": 10.0}}
    saved = copy.deepcopy(plan)
    production = {"overlap_components": [0, 1], "effective_samples_per_window": [1.0, 20.0]}
    proposal = adaptive._production_refinement(plan, production, (None,))
    assert proposal["scope"] == "next_run_only"
    assert proposal["applied"] is False
    assert proposal["requires_fresh_output"] is True
    assert proposal["suggested_additional_centers"] == [[0.0]]
    assert proposal["windows_needing_more_effective_samples"] == [0]
    assert plan == saved


def test_public_input_accepts_opt_in_continuation():
    from pyrpmd.input import JobSpec
    spec = JobSpec.from_file("examples/fes/adaptive_harmonic.inp")
    spec.data["adaptive_umbrella"]["continuation"] = True
    resolved, _ = spec.resolve()
    assert adaptive.adaptive_options(resolved)["continuation"] is True


def test_production_step_failure_resumes_from_own_checkpoint(tmp_path, monkeypatch):
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.workflows import fes
    config = _config()
    reference = run_fes(config, _system(), tmp_path / "reference", return_data=True)
    original_sample = fes._sample_window
    original_step = Dynamics.step

    def fail_step(self, state, *args, **kwargs):
        result = original_step(self, state, *args, **kwargs)
        if state.step == 17:
            raise RuntimeError("interrupted production step")
        return result

    def sample(task):
        with monkeypatch.context() as patch:
            patch.setattr(Dynamics, "step", fail_step)
            return original_sample(task)

    with monkeypatch.context() as patch:
        patch.setattr(fes, "_sample_window", sample)
        with pytest.raises(RuntimeError, match="interrupted production step"):
            run_fes(config, _system(), tmp_path / "resumed")
    actual = run_fes(config, _system(), tmp_path / "resumed", return_data=True)
    _same_streams(reference, actual)
