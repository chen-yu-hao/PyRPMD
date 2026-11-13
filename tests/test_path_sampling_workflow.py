"""Public path workflows: byte-exact continuation, identity and CLI output."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, load_checkpoint
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import _initialize_output, resume_run, run
from pyrpmd.workflows.path_sampling import run_path_sampling


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "sampling"
_SPEC = importlib.util.spec_from_file_location("t23_workflow_reference", _EXAMPLES / "path_sampling_quartic.py")
_REFERENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REFERENCE)


def _input(method, workers=1):
    data = _REFERENCE.workflow_input(method, samples=22, burn_in=7, block_size=5,
                                     flux_steps=180, trials=8, workers=workers)
    data["output"]["checkpoint_steps"] = 5
    if method == "ffs":
        data["path_sampling"].update(equilibration_steps=0, initial_state=_REFERENCE.phase_point(-1.1, 1.5))
    return data


def _arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _equal_archives(first, second):
    a, b = _arrays(first), _arrays(second)
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype, key
        assert a[key].shape == b[key].shape, key
        # Compare IEEE bits as well as values (including signed zero), not allclose.
        assert a[key].tobytes() == b[key].tobytes(), key


@pytest.mark.parametrize("method,workers,stop", [
    ("retis", 1, 0), ("retis", 1, 3), ("retis", 1, 13),
    ("repptis", 1, 3), ("repptis", 1, 13),
    ("ffs", 1, 53), ("ffs", 1, 185),
    ("infinite-retis", 1, 3), ("infinite-retis", 2, 13),
])
def test_public_resume_preserves_every_observation_and_rng_bit(tmp_path, method, workers, stop):
    raw = _input(method, workers)
    uninterrupted = run(raw, output=tmp_path / "full")
    config, system = JobSpec.from_mapping(raw).resolve()
    destination = tmp_path / "split"
    config["output"]["directory"] = str(destination)
    _initialize_output(config, system, destination)
    partial = run_path_sampling(config, system, destination, max_operations=stop)
    assert not partial["completed"]
    actual = resume_run(destination)
    assert actual["completed"]
    expected = dict(uninterrupted, output=actual["output"])
    assert actual == expected
    _equal_archives(tmp_path / "full" / "checkpoints/path_sampling.npz", destination / "checkpoints/path_sampling.npz")
    _equal_archives(tmp_path / "full" / "samples/path_sampling.npz", destination / "samples/path_sampling.npz")
    # A completed resume is idempotent and consumes no additional RNG values.
    saved = _arrays(destination / "checkpoints/path_sampling.npz")
    assert resume_run(destination) == actual
    after = _arrays(destination / "checkpoints/path_sampling.npz")
    assert all(saved[key].tobytes() == after[key].tobytes() for key in saved)
    assert not list(destination.rglob("*.tmp"))


@pytest.mark.parametrize("method", ["retis", "repptis", "ffs", "infinite-retis"])
def test_checkpoint_frequency_does_not_change_the_scientific_chain(tmp_path, method):
    raw = _input(method)
    first = run(raw, output=tmp_path / "a")
    raw["output"]["checkpoint_steps"] = 11
    second = run(raw, output=tmp_path / "b")
    assert first["physics_fingerprint"] == second["physics_fingerprint"]
    assert dict(first, output=second["output"]) == second
    _equal_archives(tmp_path / "a" / "samples/path_sampling.npz", tmp_path / "b" / "samples/path_sampling.npz")


def test_changed_scientific_budget_is_rejected_without_mutating_saved_run(tmp_path):
    raw = _input("retis")
    run(raw, output=tmp_path / "run")
    manifest = tmp_path / "run" / "manifest.json"
    checkpoint = tmp_path / "run" / "checkpoints/path_sampling.npz"
    before = manifest.read_bytes(), checkpoint.read_bytes()
    raw["path_sampling"]["samples"] += 1
    with pytest.raises((InputError, RuntimeError), match="scientific settings"):
        run(raw, output=tmp_path / "run", resume=True)
    assert before == (manifest.read_bytes(), checkpoint.read_bytes())


def test_infinite_resume_rejects_changed_logical_worker_count_before_output(tmp_path):
    run(_input("infinite-retis"), output=tmp_path / "run")
    checkpoint = tmp_path / "run" / "checkpoints/path_sampling.npz"
    before = checkpoint.read_bytes()
    with pytest.raises((ValueError, RuntimeError), match="logical worker|workers must match"):
        resume_run(tmp_path / "run", parallel={"trajectory_workers": 2})
    assert checkpoint.read_bytes() == before


def test_corrupt_sampler_checkpoint_is_never_overwritten_on_resume(tmp_path):
    run(_input("retis"), output=tmp_path / "run")
    checkpoint = tmp_path / "run" / "checkpoints/path_sampling.npz"
    broken = checkpoint.read_bytes()[:80]
    checkpoint.write_bytes(broken)
    with pytest.raises((CheckpointError, ValueError), match="checkpoint|ZIP|archive"):
        resume_run(tmp_path / "run")
    assert checkpoint.read_bytes() == broken


def test_ffs_without_a_history_stops_without_inventing_a_rate(tmp_path):
    raw = _input("ffs")
    raw["path_sampling"].update(flux_steps=1, initial_state=_REFERENCE.phase_point(0.0, 0.0),
                                 dynamics="hamiltonian")
    raw["path_sampling"].pop("collision_rate")
    result = run(raw, output=tmp_path / "no-history")
    assert result["operations_completed"] == 1
    assert result["rate_estimate_au"] is None
    assert result["completed"]
    assert not result["sampling_support_complete"]
    assert result["status"] == "no_flux_crossings"


def test_public_toml_cli_validate_run_analyze_and_resume(tmp_path):
    source = (_EXAMPLES / "path_sampling_retis.inp").read_text()
    source = source.replace('factory = "path_sampling_quartic.py:make_potential"',
                            'factory = "' + (_EXAMPLES / 'path_sampling_quartic.py').as_posix() + ':make_potential"')
    source = source.replace("samples = 2400", "samples = 8").replace("burn_in = 300", "burn_in = 2")
    input_path = tmp_path / "path.inp"
    input_path.write_text(source)

    def cli(*arguments):
        process = subprocess.run([sys.executable, "-m", "pyrpmd", *map(str, arguments)],
                                 text=True, capture_output=True, check=False, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
        return json.loads(process.stdout)

    assert cli("validate", input_path)["valid"]
    result = cli("run", input_path, "--output", tmp_path / "cli")
    assert result["task"] == "path_sampling" and result["completed"]
    assert cli("resume", tmp_path / "cli")["physics_fingerprint"] == result["physics_fingerprint"]
    assert cli("analyze", tmp_path / "cli")["rate_estimate_au"] == result["rate_estimate_au"]
