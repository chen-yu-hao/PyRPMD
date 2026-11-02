"""Window-local budgets must not enter the uniform extension authorization."""

import hashlib

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.runner import extend_run, run_resolved
from pyrpmd.system import System


def _configuration():
    system = System(
        ("H", "H", "H"), np.full(3, 2.0),
        np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]),
        ((0,), (1, 2)),
    )
    return {
        "task": "rate", "beta": 5.0, "beads": 2, "dt_au": .003,
        "seed": 17, "native": False, "potential": {"provider": "free"},
        "sampling": {"equilibration_steps": 7, "production_steps": 31,
                     "sample_interval": 3, "andersen_interval": 7},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 4.0},
        "umbrella": {"force_constant": "10 hartree", "windows": [
            {"id": f"state-{index}", "center": float(center)}
            for index, center in enumerate(np.linspace(0, 1, 5))
        ]},
        "analysis": {"bins": 12, "bootstrap": 0},
        "parallel": {"backend": "serial"},
        "dynamics": {"block_steps": 11, "loop_backend": "python"},
        "output": {"checkpoint_steps": 13},
        "recrossing": {"parent_replicas": 1, "parent_samples": 2,
                       "equilibration_steps": 5, "parent_interval_steps": 3,
                       "child_pairs": 1, "child_steps": 10, "bootstrap": 0,
                       "dividing_surface": 1.0},
    }, system


def _hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("sampling", [
    {"production": {"steps": 18}},
    {"production": {"steps": 45}},
    {"equilibration": {"steps": 14}},
])
def test_local_budget_extension_fails_before_authorizing_or_changing_run(tmp_path, monkeypatch, sampling):
    config, system = _configuration()
    config["umbrella"]["windows"][2]["sampling"] = sampling
    output = tmp_path / "run"
    original = run_resolved(config, system, output=output)
    assert original["pmf"]["ready"]
    before = _hashes(output)
    monkeypatch.setattr(Dynamics, "__init__", lambda *a, **kw: pytest.fail("extension propagated"))
    with pytest.raises(RuntimeError, match="does not yet support per-window or per-region sampling overrides"):
        extend_run(output, production_steps=74)
    assert _hashes(output) == before
    assert not (output / "results" / "extensions.json").exists()


def test_region_sampling_extension_is_also_rejected_without_writes(tmp_path):
    config, system = _configuration()
    config["umbrella"].pop("windows")
    config["umbrella"]["regions"] = [
        {"start": 0, "stop": 1, "spacing": .25,
         "sampling": {"production": {"steps": 18}}},
    ]
    output = tmp_path / "run"
    run_resolved(config, system, output=output)
    before = _hashes(output)
    with pytest.raises(RuntimeError, match="does not yet support per-window or per-region sampling overrides"):
        extend_run(output, production_steps=74)
    assert _hashes(output) == before


def test_uniform_timing_with_heterogeneous_force_constants_still_extends(tmp_path):
    config, system = _configuration()
    config["umbrella"]["windows"][2]["force_constant"] = "15 hartree"
    output = tmp_path / "run"
    original = run_resolved(config, system, output=output)
    assert original["pmf"]["ready"]
    result = extend_run(output, production_steps=74)
    assert result["effective_production_steps"] == 74
    assert result["pmf"]["ready"]
    assert result["child_trajectories"] == 4
    checkpoints = list((output / "umbrella" / "checkpoints").glob("window_*_replica_*.npz"))
    assert len(checkpoints) == 5
    stiffnesses = []
    for checkpoint in checkpoints:
        with load_checkpoint(checkpoint, require_integrity=True) as saved:
            assert int(saved["elapsed"]) == 7 + 74
            stiffnesses.append(float(saved["stiffness"][0, 0]))
    assert sorted(stiffnesses) == [10., 10., 10., 10., 15.]
