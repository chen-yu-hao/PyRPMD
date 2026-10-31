"""Budget extension preserves a physical trajectory and its immutable identity."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics
from pyrpmd.initial_states import PreparedInitialEnsemble
from pyrpmd.runner import extend_run, resume_run, run_resolved
from pyrpmd.system import System
from pyrpmd.workflows import drpmd
from pyrpmd.workflows import extend as extension

from test_state_selected_workflow import config as selected_config, system as selected_system


def configuration(tmp_path, *, selected=True, classifier="none"):
    config = selected_config()
    config["collision"]["max_steps"] = 4
    config["collision"]["trajectories"] = 2
    config["dynamics"]["block_steps"] = 3
    config["outcomes"].update(check_interval=2, persistence_steps=4)
    config["output"].update(trajectory_stride=2, checkpoint_steps=2)
    body = {
        "none": "return None",
        "nonreactive": "return 'nonreactive'",
        "product": "return ('product', 'toy')",
        "failed": "raise RuntimeError('deliberate numerical/classifier failure')",
    }[classifier]
    module = tmp_path / f"classifier_{classifier}.py"
    module.write_text(f"def make(system):\n    def classify(state):\n        {body}\n    return classify\n")
    config["outcomes"]["factory"] = str(module) + ":make"
    if not selected:
        del config["initial_ensemble"]
        config["sampling"] = {"parents": 1, "equilibration_steps": 2,
                              "interval_steps": 1, "andersen_interval": 1}
    return config, System.from_dict(selected_system())


def run(config, system, directory):
    return run_resolved(copy.deepcopy(config), system, output=directory)


def frozen_files(directory):
    directory = Path(directory)
    names = [directory / "manifest.json", *sorted((directory / "input").glob("*")),
             *sorted((directory / "checkpoints").glob("parent-*.npz")),
             *sorted((directory / "states").glob("initial-*.npz"))]
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in names if path.is_file()}


def assert_same_trajectory(first, second, trajectory_id=0):
    file = f"trajectories/trajectory-{trajectory_id:08d}.npz"
    with np.load(Path(first) / file) as a, np.load(Path(second) / file) as b:
        for key in ("final_q", "final_p", "steps", "time_au", "centroid",
                    "energy_per_bead", "energy_parts", "ring_radii"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("selected", [True, False])
def test_extension_matches_one_continuous_trajectory_without_resampling(tmp_path, selected):
    config, system = configuration(tmp_path, selected=selected)
    original = run(config, system, tmp_path / "short")
    immutable = frozen_files(tmp_path / "short")
    long_config = copy.deepcopy(config)
    long_config["collision"]["max_steps"] = 8
    uninterrupted = run(long_config, system, tmp_path / "long")
    with (
        patch.object(PreparedInitialEnsemble, "sample", side_effect=AssertionError("resampled initial state")),
        patch.object(Dynamics, "thermalize", side_effect=AssertionError("thermalized a continuation")),
        patch.object(drpmd, "launch_collision", side_effect=AssertionError("relaunched collision")),
    ):
        result = extend_run(tmp_path / "short", max_time=8 * config["dt_au"])
    assert result["original_max_steps"] == 4
    assert result["effective_max_steps"] == 8
    assert result["extensions"]["requests"][-1]["status"] == "completed"
    assert result["counts"]["unresolved"] == 2
    assert frozen_files(tmp_path / "short") == immutable
    for old, new, reference in zip(original["records"], result["records"], uninterrupted["records"]):
        assert new["steps"] == reference["steps"] == 8
        assert new["physics_hash"] == old["physics_hash"]
        assert new["launch"] == old["launch"]
        assert new["propagation"]["python_steps"] == 8
        assert new["initial_energy_per_bead"] == old["initial_energy_per_bead"]
        assert new["max_energy_drift_per_bead"] == reference["max_energy_drift_per_bead"]
        if selected:
            assert new["initial_state"] == old["initial_state"]
        assert_same_trajectory(tmp_path / "long", tmp_path / "short", new["trajectory_id"])


def test_classifier_persistence_clock_crosses_the_old_time_limit(tmp_path):
    config, system = configuration(tmp_path, classifier="nonreactive")
    initial = run(config, system, tmp_path / "short")
    assert initial["counts"]["unresolved"] == 2
    metadata, _ = drpmd._load_npz(tmp_path / "short/trajectories/trajectory-00000000.checkpoint.npz")
    assert metadata["candidate_since"] == 2
    result = extend_run(tmp_path / "short", max_time=8 * config["dt_au"])
    assert result["counts"]["nonreactive"] == 2
    assert [record["steps"] for record in result["records"]] == [6, 6]
    long_config = copy.deepcopy(config)
    long_config["collision"]["max_steps"] = 8
    run(long_config, system, tmp_path / "long")
    assert_same_trajectory(tmp_path / "long", tmp_path / "short")


@pytest.mark.parametrize("status", ["product", "nonreactive", "failed"])
def test_terminal_outcomes_and_failed_trajectories_are_never_propagated_again(tmp_path, status):
    config, system = configuration(tmp_path, classifier=status)
    config["outcomes"]["persistence_steps"] = 0
    before = run(config, system, tmp_path / "run")
    assert before["counts"][status] == 2
    files = {path.name: path.read_bytes() for path in (tmp_path / "run/trajectories").iterdir()}
    with patch.object(extension, "_extend_parent", side_effect=AssertionError("repropagated terminal outcome")):
        result = extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    assert result["counts"][status] == 2
    assert result["records"] == before["records"]
    assert files == {path.name: path.read_bytes() for path in (tmp_path / "run/trajectories").iterdir()}
    assert result["extensions"]["requests"][-1]["trajectory_ids"] == []


def test_repeating_budget_is_idempotent_and_ordinary_resume_keeps_extended_results(tmp_path):
    config, system = configuration(tmp_path)
    run(config, system, tmp_path / "run")
    extended = extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    ledger_bytes = (tmp_path / "run/results/extensions.json").read_bytes()
    with patch.object(drpmd, "_trajectory", side_effect=AssertionError("propagated twice")):
        repeated = extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    assert repeated["records"] == extended["records"]
    assert len(repeated["extensions"]["requests"]) == 1
    assert (tmp_path / "run/results/extensions.json").read_bytes() == ledger_bytes
    resumed = resume_run(tmp_path / "run")
    assert resumed["records"] == extended["records"]
    assert resumed["effective_max_steps"] == 8
    assert resumed["extensions"]["requests"][-1]["status"] == "completed"


def test_interrupted_extension_recovers_new_checkpoint_via_ordinary_resume(tmp_path):
    config, system = configuration(tmp_path)
    run(config, system, tmp_path / "run")
    immutable = frozen_files(tmp_path / "run")
    save = drpmd._atomic_npz

    def persist_then_interrupt(path, metadata, **arrays):
        save(path, metadata, **arrays)
        if (metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1"
                and metadata["done"] == 6):
            raise KeyboardInterrupt("crash after extension checkpoint, before result commit")

    with patch.object(drpmd, "_atomic_npz", side_effect=persist_then_interrupt), pytest.raises(KeyboardInterrupt):
        extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    ledger = json.loads((tmp_path / "run/results/extensions.json").read_text())
    assert ledger["requests"][-1]["status"] == "running"
    old_result = json.loads((tmp_path / "run/trajectories/trajectory-00000000.json").read_text())
    assert old_result["steps"] == 4
    checkpoint, _ = drpmd._load_npz(tmp_path / "run/trajectories/trajectory-00000000.checkpoint.npz")
    assert checkpoint["done"] == 6
    with pytest.raises(ValueError, match="pending extension"):
        resume_run(tmp_path / "run", adaptive_budget={"max_rounds": 10})
    assert json.loads((tmp_path / "run/results/extensions.json").read_text()) == ledger
    with patch.object(PreparedInitialEnsemble, "sample", side_effect=AssertionError("resampled on recovery")):
        result = resume_run(tmp_path / "run")
    assert result["extensions"]["requests"][-1]["status"] == "completed"
    assert [record["steps"] for record in result["records"]] == [8, 8]
    assert frozen_files(tmp_path / "run") == immutable
    config["collision"]["max_steps"] = 8
    run(config, system, tmp_path / "long")
    for i in range(2):
        assert_same_trajectory(tmp_path / "long", tmp_path / "run", i)


@pytest.mark.parametrize("new_steps", [2, 4])
def test_new_limit_must_increase_original_budget(tmp_path, new_steps):
    config, system = configuration(tmp_path)
    run(config, system, tmp_path / "run")
    with pytest.raises(RuntimeError, match="must exceed"):
        extend_run(tmp_path / "run", max_time=new_steps * config["dt_au"])
    assert not (tmp_path / "run/results/extensions.json").exists()


def test_extension_cannot_shrink_a_previously_authorized_budget(tmp_path):
    config, system = configuration(tmp_path)
    run(config, system, tmp_path / "run")
    extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    with pytest.raises(RuntimeError, match="reduce an already authorized"):
        extend_run(tmp_path / "run", max_time=6 * config["dt_au"])


@pytest.mark.parametrize("what", ["input", "model", "missing_checkpoint", "corrupt_checkpoint", "missing_launch"])
def test_identity_and_all_checkpoints_are_validated_before_any_continuation(tmp_path, what):
    config, system = configuration(tmp_path)
    run(config, system, tmp_path / "run")
    if what == "input":
        file = tmp_path / "run/input/resolved.json"
        data = json.loads(file.read_text())
        data["beta"] *= 2
        file.write_text(json.dumps(data))
    elif what == "model":
        (tmp_path / "classifier_none.py").write_text("# changed declared classifier dependency\n")
    elif what == "missing_checkpoint":
        (tmp_path / "run/trajectories/trajectory-00000001.checkpoint.npz").unlink()
    elif what == "corrupt_checkpoint":
        (tmp_path / "run/trajectories/trajectory-00000001.checkpoint.npz").write_bytes(b"not an npz")
    else:
        (tmp_path / "run/trajectories/trajectory-00000001.json").unlink()
    with patch.object(drpmd, "_trajectory", side_effect=AssertionError("propagated before validation")):
        with pytest.raises(RuntimeError):
            extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    assert not (tmp_path / "run/results/extensions.json").exists()
    record = json.loads((tmp_path / "run/trajectories/trajectory-00000000.json").read_text())
    assert record["steps"] == 4


def test_increasing_the_budget_again_keeps_one_original_physics_identity(tmp_path):
    config, system = configuration(tmp_path)
    original = run(config, system, tmp_path / "run")
    extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    final = extend_run(tmp_path / "run", max_time=12 * config["dt_au"])
    assert [item["max_steps"] for item in final["extensions"]["requests"]] == [8, 12]
    assert [item["status"] for item in final["extensions"]["requests"]] == ["completed", "completed"]
    assert all(record["physics_hash"] == original["records"][i]["physics_hash"]
               for i, record in enumerate(final["records"]))
    config["collision"]["max_steps"] = 12
    run(config, system, tmp_path / "long")
    assert_same_trajectory(tmp_path / "long", tmp_path / "run")


def test_selected_state_extension_uses_recorded_groups_after_worker_change(tmp_path):
    config, system = configuration(tmp_path)
    save = drpmd._atomic_json

    def commit_one_launch_then_interrupt(path, value):
        save(path, value)
        if Path(path).name == "trajectory-00000000.json":
            raise KeyboardInterrupt("one selected-state trajectory committed")

    with patch.object(drpmd, "_atomic_json", side_effect=commit_one_launch_then_interrupt), pytest.raises(KeyboardInterrupt):
        run(config, system, tmp_path / "run")
    resumed = resume_run(tmp_path / "run", parallel={"backend": "process", "trajectory_workers": 2})
    assert [record["parent_id"] for record in resumed["records"]] == [0, 1]
    # Saved input still describes the original single-worker allocation.
    saved = json.loads((tmp_path / "run/input/resolved.json").read_text())
    assert saved["parallel"]["backend"] == "serial"
    result = extend_run(tmp_path / "run", max_time=8 * config["dt_au"])
    assert [record["parent_id"] for record in result["records"]] == [0, 1]
    assert [record["steps"] for record in result["records"]] == [8, 8]
    config["collision"]["max_steps"] = 8
    run(config, system, tmp_path / "long")
    for i in range(2):
        assert_same_trajectory(tmp_path / "long", tmp_path / "run", i)
