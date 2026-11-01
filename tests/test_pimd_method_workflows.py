"""Input, action preflight and exact interrupted recovery for main-engine PI."""
import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare
from pyrpmd.workflows.path_integral import KIND, resolve_path_integral, run_path_integral


def raw_input(method="suzuki_chin", thermostat="pile-l"):
    action = {"method": method}
    if method == "pigs":
        action.update(projection_time=4.0, trial={"alpha": 0.7})
    if method == "bosonic":
        action["groups"] = [[0, 1]]
    return {
        "task": "pimd", "temperature": 1 / (3.1 * KB_AU),
        "beads": 5 if method == "pigs" else 4, "seed": 913, "native": False,
        "system": {"symbols": ["H", "H"], "masses": [2., 2.],
                   "geometry": [[.1, -.2, .3], [-.1, .2, -.3]]},
        "potential": {"provider": "harmonic", "params": {"k": .2}},
        "dynamics": {"integrator": "normal-mode-bab", "timestep": .01},
        "sampling": {"equilibration_steps": 3, "production_steps": 12,
                     "sample_interval": 2, "thermostat": thermostat, "andersen_interval": 3},
        "path_integral": action, "output": {"checkpoint_steps": 4},
    }


def arrays(path):
    with load_checkpoint(path, expected_kind=KIND, require_integrity=True) as saved:
        return {key: saved[key].copy() for key in saved.files}


@pytest.mark.parametrize("method", ["suzuki_chin", "bosonic", "pigs"])
def test_prepare_checks_the_actual_action_without_writing(tmp_path, method):
    raw = raw_input(method)
    raw["output"]["directory"] = str(tmp_path / "unused")
    result = prepare(raw, numerical=True)
    assert result["ready"]
    assert result["path_integral_preparation"]["method"] == method
    assert result["path_integral_preparation"]["trajectory_storage"] == "memory"
    assert result["pes_check"]["sampling_action_potential"]
    assert len(result["pes_check"]["energy_hartree"]) == raw["beads"]
    beta = raw["beads"] * 4 / (raw["beads"] - 1) if method == "pigs" else 3.1
    assert result["pes_check"]["auxiliary_beta_au"] == pytest.approx(beta)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("method", ["suzuki_chin", "bosonic", "pigs"])
@pytest.mark.parametrize("thermostat", ["pile-l", "andersen"])
def test_interrupted_resume_matches_uninterrupted_bitwise(tmp_path, monkeypatch, method, thermostat):
    config, system = JobSpec.from_mapping(raw_input(method, thermostat)).resolve()
    uninterrupted = run_path_integral(config, system, tmp_path / "whole")
    name = "sample_step" if thermostat == "pile-l" else "step"
    original = getattr(Dynamics, name)

    def fail_after_propagation(self, state, **kwargs):
        result = original(self, state, **kwargs)
        if state.step == 8:
            raise RuntimeError("injected after stochastic propagation")
        return result

    monkeypatch.setattr(Dynamics, name, fail_after_propagation)
    with pytest.raises(RuntimeError, match="injected"):
        run_path_integral(config, system, tmp_path / "resumed")
    checkpoint = tmp_path / "resumed" / "checkpoints" / "path_integral_0000.npz"
    partial = arrays(checkpoint)
    assert partial["elapsed"] == partial["step"] == 7
    np.testing.assert_array_equal(partial["sample_steps"], [5, 7])
    monkeypatch.setattr(Dynamics, name, original)
    resumed = run_path_integral(config, system, tmp_path / "resumed")
    a, b = arrays(uninterrupted["replicas"][0]["checkpoint"]), arrays(checkpoint)
    for key in ("q", "p", "step", "time", "rng", "positions", "potential", "sample_steps"):
        np.testing.assert_array_equal(a[key], b[key])
    assert resumed["samples"] == 6
    assert resumed["equilibrium_only"]
    if method == "pigs":
        assert resumed["temperature_K"] is None
        assert resumed["projection_time_au"] == 4.0
    # Already complete: no propagation or new random draw can affect a result.
    monkeypatch.setattr(Dynamics, name, lambda *a, **k: pytest.fail("unexpected propagation"))
    assert run_path_integral(config, system, tmp_path / "resumed") == resumed


@pytest.mark.parametrize("change", [
    {"active_learning": {}}, {"electronic_friction": {}},
    {"output": {"trajectory_storage": "stream"}}, {"output": {"chunk_frames": 2}},
    {"output": {"trajectory_stride": 3}}, {"sampling": {"parents": 2}},
    {"path_integral": {"method": "suzuki_chin", "alpha": .5}},
    {"path_integral": {"method": "pigs", "trial": {"alpha": 1.0}}},
    {"path_integral": {"method": []}},
])
def test_direct_workflow_rejects_ignored_or_incompatible_controls(change):
    config, system = JobSpec.from_mapping(raw_input()).resolve()
    for key, value in change.items():
        if key == "path_integral":
            config[key] = value
        else:
            config.setdefault(key, {}).update(value)
    with pytest.raises((ValueError, InputError)):
        resolve_path_integral(config, system)


@pytest.mark.parametrize("key,value", [("elapsed", 3.5), ("step", True), ("time", np.nan)])
def test_checkpoint_rejects_invalid_clock(tmp_path, key, value):
    config, system = JobSpec.from_mapping(raw_input()).resolve()
    result = run_path_integral(config, system, tmp_path)
    path = result["replicas"][0]["checkpoint"]
    data = arrays(path)
    # Recompute integrity so the domain validation, not checksum, rejects it.
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    data[key] = np.asarray(value)
    atomic_savez(path, kind=KIND, **data)
    with pytest.raises(ValueError, match="clock|time"):
        run_path_integral(config, system, tmp_path)


def test_bosons_require_identical_elements_and_disjoint_groups():
    raw = raw_input("bosonic")
    raw["system"]["symbols"] = ["H", "He"]
    with pytest.raises(InputError, match="identical elements"):
        JobSpec.from_mapping(raw).resolve()
    raw = raw_input("bosonic")
    raw["path_integral"]["groups"] = [[0, 1], [1]]
    with pytest.raises(InputError, match="disjoint"):
        JobSpec.from_mapping(raw).resolve()


def test_cli_validate_run_resume_and_summary(tmp_path, capsys):
    from pyrpmd.cli import main
    raw = raw_input("pigs")
    path = tmp_path / "pigs.py"
    path.write_text("def build():\n    return " + repr(raw) + "\n")
    output = tmp_path / "run"
    assert main(["validate", str(path), "--check-pes"]) == 0
    capsys.readouterr()
    assert not output.exists()
    assert main(["run", str(path), "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["resume", str(output)]) == 0
    assert main(["analyze", str(output)]) == 0
    capsys.readouterr()
