"""BA-RP-MASH workflow, scientific identity and transactional restart checks."""

import copy
import json

import numpy as np
import pytest

from pyrpmd import JobSpec, analyze_run, prepare, resume_run, run
from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.input import InputError
from pyrpmd.io import fingerprint
from pyrpmd.nonadiabatic_registry import make_dynamics
from pyrpmd.propagation import scientific_config
from pyrpmd.rp_mash_ba import RPMASHBADynamics, RPMASHBAState
from pyrpmd.workflows import nonadiabatic_extensions as workflow


def job(method="rp-mash-ba", **changes):
    value = {
        "task": "nonadiabatic", "method": method, "beta": 20., "beads": 3,
        "seed": 103, "model": {"name": "tully1"},
        "system": {"masses": [2000.], "positions": [[-1., 0., 0.]]},
        "initial": {"kind": "point", "momenta": [[8., 0., 0.]], "active_surface": 0},
        "dynamics": {"dt": .1, "native": False, "block_steps": 3},
        "ensemble": {"trajectories": 3, "steps": 7, "sample_stride": 2},
    }
    value.update(changes)
    return value


def arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {name: archive[name] for name in archive.files}


def identity(raw):
    config, system = JobSpec.from_mapping(raw).resolve()
    return fingerprint(scientific_config(config), system)


def test_original_ca_identity_is_unchanged_and_ba_has_its_own_identity():
    # Recorded before this change on main ceee59cd5f702165b56c5e541a503d8810fb9111.
    expected = "a15b26d13219cc07185f63a65e1e17a0a6da598ffc9020a2d123dfdff9efc402"
    assert identity(job("rp-mash")) == expected
    assert identity(job()) != expected


@pytest.mark.parametrize("key,value", [
    ("dt", .2), ("rtol", 1e-8), ("atol", 1e-10), ("max_step", .01),
    ("max_events", 16), ("gap_tolerance", 1e-8),
])
def test_every_ba_integration_control_changes_scientific_identity(key, value):
    raw = job()
    before = identity(raw)
    raw["dynamics"][key] = value
    assert identity(raw) != before


def test_ba_execution_layout_preserves_scientific_identity():
    raw = job()
    before = identity(raw)
    raw["dynamics"].update(block_steps=11, loop_backend="python")
    raw["parallel"] = {"trajectory_workers": 2}
    assert identity(raw) == before


@pytest.mark.parametrize("options,match", [
    ({"native": True}, "requires native=false"),
    ({"integrator": "cayley-bcb"}, "adaptive-dop853"),
    ({"rescaling": "centroid"}, "Unknown"),
    ({"electronic_backend": "torch"}, "Unknown"),
    ({"decoherence": "edc"}, "Unknown"),
    ({"max_events": True}, "integer"),
    ({"rtol": 0}, "positive"),
])
def test_ba_rejects_unused_or_unsupported_dynamics_options(options, match):
    raw = job()
    raw["dynamics"].update(options)
    with pytest.raises(InputError, match=match):
        JobSpec.from_mapping(raw).resolve()


def test_ba_rtol_below_solver_minimum_is_rejected_without_changing_ca():
    minimum = 100.0 * np.finfo(float).eps
    raw = job()
    raw["dynamics"]["rtol"] = np.nextafter(minimum, 0.)
    with pytest.raises(InputError, match="double-precision minimum"):
        JobSpec.from_mapping(raw).resolve()
    raw["method"] = "rp-mash"
    ca, _ = JobSpec.from_mapping(raw).resolve()
    assert ca["dynamics"]["rtol"] == raw["dynamics"]["rtol"]
    raw["method"] = "rp-mash-ba"
    raw["dynamics"]["rtol"] = minimum
    ba, _ = JobSpec.from_mapping(raw).resolve()
    assert ba["dynamics"]["rtol"] == minimum


@pytest.mark.parametrize("model,match", [
    ({"name": "lvc", "params": {
        "h0": np.diag([-.2, 0., .2]), "linear": [np.diag([.1, .2, .3])],
    }}, "exactly two"),
    ({"name": "lvc", "params": {
        "h0": {"real": [[-.2, .1], [.1, .2]], "imag": [[0., .02], [-.02, 0.]]},
        "linear": [[[.03, 0.], [0., -.03]]],
    }}, "real symmetric"),
])
def test_ba_rejects_multistate_and_complex_providers(model, match):
    with pytest.raises(InputError, match=match):
        JobSpec.from_mapping(job(model=model)).resolve()


def test_ba_saved_factory_rejects_an_incompatible_integrator():
    config, system = JobSpec.from_mapping(job()).resolve()
    config["dynamics"]["integrator"] = "velocity-verlet"
    with pytest.raises(ValueError, match="adaptive-dop853"):
        make_dynamics(config, system, np.random.default_rng(3))


def test_ba_toml_cli_validate_run_and_resume(tmp_path, capsys):
    from pyrpmd.cli import main

    source = tmp_path/"bead-mash.inp"
    source.write_text('''task = "nonadiabatic"
method = "rp-mash-ba"
beta = 20.0
beads = 3
seed = 103
[model]
name = "tully1"
[system]
masses = [2000.0]
positions = [[-1.0, 0.0, 0.0]]
[initial]
kind = "point"
active_surface = 0
momenta = [[8.0, 0.0, 0.0]]
[dynamics]
dt = 0.1
[ensemble]
trajectories = 2
steps = 2
sample_stride = 1
''', encoding="utf-8")
    output = tmp_path/"run"
    assert main(["validate", str(source), "--check-pes", "--output", str(output)]) == 0
    assert not output.exists()
    capsys.readouterr()
    assert main(["run", str(source), "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["resume", str(output)]) == 0
    capsys.readouterr()
    metadata = json.loads(str(arrays(output/"results/nonadiabatic.npz")["metadata"].item()))
    assert metadata["method"] == "rp-mash-ba"
    assert metadata["population_basis"] == "common_bead_adiabatic"


def test_ba_prepare_outputs_and_completed_resume_are_consistent(tmp_path, monkeypatch):
    source = job()
    original = copy.deepcopy(source)
    ready = prepare(source, numerical=True)
    preparation = ready["nonadiabatic_preparation"]
    assert ready["ready"]
    assert preparation["electronic_state_class"] == "RPMASHBAState"
    assert preparation["matrix_gradient_check"]["max_absolute_gradient_error"] < 1e-8
    result = run(source, output=tmp_path/"run")
    assert source == original
    assert result["completed"]
    assert result["population_basis"] == "common_bead_adiabatic"
    assert result["active_surface_basis"] == "common_bead_adiabatic"
    assert result["coherence_available"] is False
    saved = arrays(tmp_path/"run/results/nonadiabatic.npz")
    metadata = json.loads(str(saved["metadata"].item()))
    assert metadata["method"] == "rp-mash-ba"
    assert metadata["population_estimator"] == "active_surface_indicator"
    assert metadata["population_basis"] == "common_bead_adiabatic"
    assert metadata["density_basis"] is None and metadata["coherence_available"] is False
    assert metadata["statistics_axis"] == "trajectory"
    assert "diabatic_density" not in saved and "density_mean" not in saved
    assert "adiabatic_amplitude_populations" not in saved
    np.testing.assert_array_equal(saved["steps"], [0, 2, 4, 6, 7])
    assert saved["populations"].shape == (3, 5, 2)
    assert saved["electronic_norm"].shape == (3, 5, 1)
    np.testing.assert_array_equal(saved["population_mean"], saved["active_surface_fractions"])
    np.testing.assert_allclose(saved["population_sem"], saved["populations"].std(axis=0, ddof=1)/np.sqrt(3))
    final = arrays(tmp_path/"run/checkpoints/nonadiabatic_000000.npz")
    assert json.loads(str(final["metadata"].item()))["state_class"] == "RPMASHBAState"
    assert final["spin"].shape == (3,) and final["sampling"].item() == "population"

    def forbidden(*args, **kwargs):
        raise AssertionError("completed analysis/resume must not propagate")

    monkeypatch.setattr(RPMASHBADynamics, "step", forbidden)
    assert analyze_run(tmp_path/"run", recompute=True)["final_populations"] == result["final_populations"]
    assert resume_run(tmp_path/"run")["final_populations"] == result["final_populations"]


def test_ba_interrupted_resume_is_bitwise_identical_for_full_ring_and_all_rngs(tmp_path, monkeypatch):
    source = job(
        system={"masses": [2000.]},
        initial={"kind": "harmonic_ring", "frequencies": [.02], "coordinates": [0],
                 "centers": [-1.], "centroid_momenta": [8.], "active_surface": 0},
    )
    run(source, output=tmp_path/"reference")
    original = RPMASHBADynamics.step

    def interrupt(self, state):
        if state.step == 5:
            raise RuntimeError("deliberate BA interruption")
        return original(self, state)

    monkeypatch.setattr(RPMASHBADynamics, "step", interrupt)
    with pytest.raises(RuntimeError, match="deliberate BA"):
        run(source, output=tmp_path/"interrupted")
    assert not (tmp_path/"interrupted/results/summary.json").exists()
    path = tmp_path/"interrupted/checkpoints/nonadiabatic_000000.npz"
    partial = arrays(path)
    assert partial["step"] == 3
    np.testing.assert_array_equal(partial["history_steps"], [0, 2])
    assert not np.array_equal(partial["q"][0], partial["q"][1])
    assert set(json.loads(str(partial["metadata"].item()))["rngs"]) == {"nuclear", "electronic", "dynamics"}
    monkeypatch.setattr(RPMASHBADynamics, "step", original)
    result = resume_run(tmp_path/"interrupted", execution={"block_steps": 4, "loop_backend": "python"})
    assert result["completed"]
    for index in range(3):
        name = f"checkpoints/nonadiabatic_{index:06d}.npz"
        before, after = arrays(tmp_path/"reference"/name), arrays(tmp_path/"interrupted"/name)
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_ba_process_placement_preserves_individual_trajectories(tmp_path):
    source = job(ensemble={"trajectories": 2, "steps": 3, "sample_stride": 2})
    run(source, output=tmp_path/"serial")
    source["parallel"] = {"trajectory_workers": 2}
    run(source, output=tmp_path/"process")
    for index in range(2):
        name = f"checkpoints/nonadiabatic_{index:06d}.npz"
        serial, process = arrays(tmp_path/"serial"/name), arrays(tmp_path/"process"/name)
        assert serial.keys() == process.keys()
        for key in serial:
            np.testing.assert_array_equal(serial[key], process[key], err_msg=key)


@pytest.mark.parametrize("corruption,match", [
    ("spin_dtype", "dtype"), ("hemisphere", "state"), ("sampling", "sampling"),
    ("state_class", "identity"), ("population", "indicators"),
    ("terminal_energy", "final observation"), ("rng", "random"),
])
def test_ba_resealed_checkpoint_corruption_is_explicitly_rejected(tmp_path, corruption, match):
    run(job(ensemble={"trajectories": 1, "steps": 2, "sample_stride": 1}), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    if corruption == "spin_dtype":
        saved["spin"] = saved["spin"].astype(np.float32)
    elif corruption == "hemisphere":
        saved["spin"][2] *= -1
    elif corruption == "sampling":
        saved["sampling"] = np.asarray("uniform-sphere")
    elif corruption == "population":
        saved["history_population"][0] = saved["history_population"][0, ::-1]
    elif corruption == "terminal_energy":
        saved["history_energy"][-1] += 1.
    else:
        metadata = json.loads(str(saved["metadata"].item()))
        if corruption == "state_class":
            metadata["state_class"] = "RPMASHState"
        else:
            metadata["rngs"]["electronic"] = {"broken": True}
        saved["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(path, kind=workflow._KIND, **saved)
    with pytest.raises(CheckpointError, match=match):
        resume_run(tmp_path/"run")


def test_ba_invalid_rng_does_not_partially_restore_live_streams(tmp_path):
    source = job(ensemble={"trajectories": 1, "steps": 1, "sample_stride": 1})
    config, system = JobSpec.from_mapping(source).resolve()
    run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    metadata = json.loads(str(saved["metadata"].item()))
    metadata["rngs"]["dynamics"] = {"broken": True}
    saved["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(path, kind=workflow._KIND, **saved)
    streams = workflow._streams(config, 0)
    for generator in streams:
        generator.random(11)
    before = [copy.deepcopy(generator.bit_generator.state) for generator in streams]
    dynamics = workflow._driver(config, system, streams[2])
    try:
        with pytest.raises(CheckpointError, match="random"):
            workflow._load(path, config, identity(source), 0, dynamics, streams)
        assert before == [generator.bit_generator.state for generator in streams]
    finally:
        dynamics.potential.close()


def test_ba_checkpoint_load_constructs_the_adiabatic_state_type(tmp_path):
    source = job(ensemble={"trajectories": 1, "steps": 0, "sample_stride": 1})
    result = run(source, output=tmp_path/"zero")
    assert result["frames"] == 1 and result["final_population_sem"] is None
    assert result["sem_unavailable_reason"] == "fewer_than_two_trajectories"
    config, system = JobSpec.from_mapping(source).resolve()
    streams = workflow._streams(config, 0)
    dynamics = workflow._driver(config, system, streams[2])
    try:
        state, history = workflow._load(tmp_path/"zero/checkpoints/nonadiabatic_000000.npz",
                                       config, identity(source), 0, dynamics, streams)
        assert type(state) is RPMASHBAState
        assert state.step == 0 and len(history["steps"]) == 1
    finally:
        dynamics.potential.close()
