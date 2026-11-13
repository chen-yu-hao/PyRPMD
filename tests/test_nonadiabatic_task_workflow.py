"""Public task, stochastic restart, process placement and saved-data controls."""
import copy
import json

import numpy as np
import pytest

from pyrpmd import JobSpec, run, prepare, resume_run, analyze_run
from pyrpmd.checkpoint import atomic_savez, load_checkpoint, CheckpointError
from pyrpmd.nonadiabatic import NRPMDDynamics
from pyrpmd.surface_hopping import RPSHDynamics


def job(method="nrpmd", **changes):
    value = {
        "task": "nonadiabatic", "method": method, "beta": 4., "beads": 3, "seed": 72,
        "model": {"name": "tully1", "params": {}},
        "system": {"masses": [2000.], "positions": [[-.1, 0., 0.]]},
        "initial": {"kind": "point", "momenta": [[40., 0., 0.]]},
        "dynamics": {"dt": .2, "native": False, "block_steps": 3},
        "ensemble": {"trajectories": 3, "steps": 11, "sample_stride": 2},
    }
    if method == "rpsh":
        value["dynamics"].update(decoherence="edc", decoherence_constant=.1)
    value.update(changes)
    return value


def arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_public_task_and_offline_analysis(tmp_path, monkeypatch, method):
    data = job(method)
    original = copy.deepcopy(data)
    prepared = prepare(data, numerical=True)
    assert prepared["ready"]
    assert prepared["nonadiabatic_preparation"]["matrix_gradient_check"]["max_absolute_gradient_error"] < 1e-8
    result = run(data, output=tmp_path/"run")
    assert data == original
    assert result["task"] == "nonadiabatic" and result["completed"]
    saved = arrays(tmp_path/"run/results/nonadiabatic.npz")
    np.testing.assert_array_equal(saved["steps"], [0, 2, 4, 6, 8, 10, 11])
    assert saved["diabatic_density"].shape == (3, 7, 2, 2)
    assert saved["population_sem"].shape == (7, 2)
    assert np.iscomplexobj(saved["diabatic_density"])
    if method == "rpsh":
        assert saved["active_surfaces"].dtype.kind == "i"
        assert saved["adiabatic_amplitude_populations"].shape == (3, 7, 2)
    def no_new_dynamics(*args, **kwargs):
        raise AssertionError("analysis/completed resume must not propagate")
    monkeypatch.setattr(NRPMDDynamics if method == "nrpmd" else RPSHDynamics, "step", no_new_dynamics)
    assert analyze_run(tmp_path/"run", recompute=True)["final_populations"] == result["final_populations"]
    assert resume_run(tmp_path/"run")["final_populations"] == result["final_populations"]


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_interruption_resume_retains_full_rng_and_observation_schedule(tmp_path, monkeypatch, method):
    data = job(method)
    run(data, output=tmp_path/"reference")
    cls = NRPMDDynamics if method == "nrpmd" else RPSHDynamics
    original_step = cls.step
    def interrupt(self, state):
        if state.step == 5:
            raise RuntimeError("intentional interrupted propagation")
        return original_step(self, state)
    monkeypatch.setattr(cls, "step", interrupt)
    with pytest.raises(RuntimeError, match="intentional interrupted"):
        run(data, output=tmp_path/"interrupted")
    assert not (tmp_path/"interrupted/results/summary.json").exists()
    checkpoint = arrays(tmp_path/"interrupted/checkpoints/nonadiabatic_000000.npz")
    assert checkpoint["step"] == 3
    np.testing.assert_array_equal(checkpoint["history_steps"], [0, 2])
    monkeypatch.setattr(cls, "step", original_step)
    resumed = resume_run(tmp_path/"interrupted", execution={"block_steps": 4, "loop_backend": "python"})
    assert resumed["completed"]
    for index in range(3):
        relative = f"checkpoints/nonadiabatic_{index:06d}.npz"
        left, right = arrays(tmp_path/"reference"/relative), arrays(tmp_path/"interrupted"/relative)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_serial_and_process_reproduce_identical_trajectory_streams(tmp_path, method):
    data = job(method)
    run(data, output=tmp_path/"serial")
    data["parallel"] = {"trajectory_workers": 2}
    run(data, output=tmp_path/"process")
    for index in range(3):
        relative = f"checkpoints/nonadiabatic_{index:06d}.npz"
        serial, process = arrays(tmp_path/"serial"/relative), arrays(tmp_path/"process"/relative)
        for key in serial:
            np.testing.assert_array_equal(serial[key], process[key], err_msg=key)


def test_single_trajectory_zero_steps_has_no_fabricated_error_bar(tmp_path):
    data = job(ensemble={"trajectories": 1, "steps": 0, "sample_stride": 9})
    result = run(data, output=tmp_path/"zero")
    assert result["frames"] == 1
    assert result["final_population_sem"] is None
    assert result["sem_unavailable_reason"] == "fewer_than_two_trajectories"
    assert "population_sem" not in arrays(tmp_path/"zero/results/nonadiabatic.npz")
    assert resume_run(tmp_path/"zero")["final_populations"] == result["final_populations"]


def test_changed_model_is_not_admitted_on_resume(tmp_path):
    data = job()
    run(data, output=tmp_path/"run")
    data["model"]["params"]["c"] = .006
    with pytest.raises(RuntimeError, match="changed scientific"):
        run(data, output=tmp_path/"run", resume=True)


def test_integrity_and_logical_history_rejected_without_dropping_trajectory(tmp_path):
    run(job(), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000001.npz"
    saved = arrays(path)
    saved["history_steps"][1] = 1
    atomic_savez(path, kind="PyRPMD.nonadiabatic_trajectory.v1", **saved)
    with pytest.raises(CheckpointError, match="schedule"):
        analyze_run(tmp_path/"run", recompute=True)
    with pytest.raises(CheckpointError, match="schedule"):
        resume_run(tmp_path/"run")


def test_harmonic_model_initialization_is_independent_and_fixed_spectators(tmp_path):
    data = job(model={"name": "spin_boson", "params": {
        "frequencies": [.5], "couplings": [.08], "masses": [1.], "tunnelling": .1}},
        system={"masses": [1.]}, initial={"kind": "harmonic_ring", "frequencies": [.5]},
        ensemble={"trajectories": 4, "steps": 0, "sample_stride": 1})
    run(data, output=tmp_path/"harmonic")
    parents = [arrays(tmp_path/"harmonic/checkpoints"/f"nonadiabatic_{i:06d}.npz") for i in range(4)]
    assert not np.array_equal(parents[0]["q"], parents[1]["q"])
    for parent in parents:
        np.testing.assert_array_equal(parent["q"][..., 1:], 0)
        np.testing.assert_array_equal(parent["p"][..., 1:], 0)


def test_new_cli_input_runs_without_hdf5_dependency(tmp_path, capsys):
    from pyrpmd.cli import main
    source = tmp_path/"model.inp"
    source.write_text('''task = "nonadiabatic"
method = "nrpmd"
beta = 4.0
beads = 1
[model]
name = "tully1"
[system]
masses = [2000.0]
positions = [[-1.0, 0.0, 0.0]]
[initial]
kind = "point"
momenta = [[20.0, 0.0, 0.0]]
[dynamics]
dt = "0.1 autime"
native = false
[ensemble]
trajectories = 2
steps = 3
sample_stride = 2
''')
    output = tmp_path/"cli"
    assert main(["validate", str(source), "--check-pes", "--output", str(output)]) == 0
    assert not output.exists()
    capsys.readouterr()
    assert main(["run", str(source), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["task"] == "nonadiabatic"
    assert main(["resume", str(output)]) == 0
