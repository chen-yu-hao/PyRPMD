"""Centroid RP-IESH identities, diagnostic output and atomic continuation."""
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
from pyrpmd.rp_iesh import RPIESHDynamics
from pyrpmd.workflows import nonadiabatic_multistate as workflow


def job(**changes):
    value = {
        "task": "nonadiabatic", "method": "rp-iesh", "beta": 40., "beads": 4, "seed": 72,
        "model": {"name": "newns_anderson", "params": {"nband": 4}},
        "system": {"masses": [200.]},
        "initial": {"kind": "harmonic_ring", "frequencies": [.02], "centers": [-.1],
                    "centroid_momenta": [4.], "occupied": [0, 1]},
        "dynamics": {"dt": .1, "native": False, "block_steps": 3},
        "ensemble": {"trajectories": 2, "steps": 7, "sample_stride": 2},
    }
    value.update(changes)
    return value


def arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {name: archive[name] for name in archive.files}


def reseal(path, payload):
    atomic_savez(path, kind=workflow._KIND, **payload)


def identity(raw):
    config, system = JobSpec.from_mapping(raw).resolve()
    return fingerprint(scientific_config(config), system)


def assert_same_checkpoints(first, second, trajectories=2):
    for index in range(trajectories):
        relative = f"checkpoints/nonadiabatic_{index:06d}.npz"
        before, after = arrays(first/relative), arrays(second/relative)
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_public_rp_workflow_retains_electron_count_and_reports_hop_energy(tmp_path, monkeypatch):
    source = job()
    original = copy.deepcopy(source)
    checked = prepare(source, numerical=True)
    info = checked["nonadiabatic_preparation"]
    assert checked["ready"]
    assert info["state_shape"] == [4, 1, 3]
    assert info["electronic_state_class"] == "RPIESHState"
    assert info["matrix_gradient_check"]["max_absolute_gradient_error"] < 1e-8
    result = run(source, output=tmp_path/"run")
    assert source == original
    assert result["completed"]
    saved = arrays(tmp_path/"run/results/nonadiabatic.npz")
    metadata = json.loads(str(saved["metadata"].item()))
    assert metadata["method_variant"] == "gardner-2023-centroid"
    assert metadata["method_status"] == "research"
    assert metadata["energy_conservation"] == "centroid_hop_balance_only"
    assert metadata["energy_estimator"] == "ring_polymer_hamiltonian_sum"
    assert metadata["electronic_geometry"] == "centroid"
    np.testing.assert_array_equal(saved["steps"], [0, 2, 4, 6, 7])
    np.testing.assert_allclose(saved["populations"].sum(axis=-1), 2., atol=1e-10)
    np.testing.assert_allclose(saved["active_population_mean"].sum(axis=-1), 2.)
    np.testing.assert_array_equal(saved["populations"], saved["diabatic_density"].diagonal(axis1=-2, axis2=-1).real)
    np.testing.assert_array_equal(saved["energy_drift_minus_hop_changes"],
                                  saved["energy_drift"]-saved["hop_energy_change"])
    assert np.all(np.diff(saved["hop_energy_change_abs"], axis=1) >= 0.)
    state = arrays(tmp_path/"run/checkpoints/nonadiabatic_000000.npz")
    assert state["q"].shape == (4, 1, 3)
    assert np.ptp(state["q"][:, 0, 0]) > 0.
    assert state["coefficients"].shape == (5, 2)
    assert state["occupied"].shape == (2,)
    for key in ("hop_energy_change", "hop_energy_change_abs"):
        assert state[key].dtype == np.float64 and state[key].shape == ()
        assert state[key] == state["history_"+key][-1]
    def forbidden(*args, **kwargs):
        raise AssertionError("completed analysis/resume must not propagate")
    monkeypatch.setattr(RPIESHDynamics, "step", forbidden)
    assert analyze_run(tmp_path/"run", recompute=True)["final_populations"] == result["final_populations"]
    assert resume_run(tmp_path/"run")["final_populations"] == result["final_populations"]


def test_rp_interrupted_resume_is_bitwise_identical_including_diagnostics_and_rng(tmp_path, monkeypatch):
    # A genuine avoided-crossing hop occurs between observations. Its nonzero
    # energy diagnostic must survive a checkpoint beyond the last saved frame.
    source = job(
        beta=10., seed=11, system={"masses": [2.]},
        model={"name": "lvc", "params": {"h0": [[0., .02], [.02, 0.]],
                                           "linear": [[[.2, 0.], [0., -.2]]]}},
        initial={"kind": "harmonic_ring", "frequencies": [1.], "centers": [-1.],
                 "centroid_momenta": [3.], "occupied": [0]},
        dynamics={"dt": .02, "native": False, "block_steps": 11},
        ensemble={"trajectories": 1, "steps": 60, "sample_stride": 13},
    )
    run(source, output=tmp_path/"reference")
    original = RPIESHDynamics.step
    def interrupt(self, state):
        if state.step == 37:
            raise RuntimeError("deliberate RP-IESH interruption")
        return original(self, state)
    monkeypatch.setattr(RPIESHDynamics, "step", interrupt)
    with pytest.raises(RuntimeError, match="deliberate RP-IESH"):
        run(source, output=tmp_path/"interrupted")
    partial = arrays(tmp_path/"interrupted/checkpoints/nonadiabatic_000000.npz")
    assert partial["step"] == 33
    np.testing.assert_array_equal(partial["history_steps"], [0, 13, 26])
    assert partial["hop_energy_change_abs"] > .01
    assert partial["history_hop_energy_change_abs"][-1] == 0.
    assert set(json.loads(str(partial["metadata"].item()))["rngs"]) == {"nuclear", "electronic", "dynamics"}
    monkeypatch.setattr(RPIESHDynamics, "step", original)
    assert resume_run(tmp_path/"interrupted", execution={"block_steps": 13, "loop_backend": "python"})["completed"]
    assert_same_checkpoints(tmp_path/"reference", tmp_path/"interrupted", trajectories=1)
    before = arrays(tmp_path/"reference/results/nonadiabatic.npz")
    after = arrays(tmp_path/"interrupted/results/nonadiabatic.npz")
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_rp_serial_and_process_execution_preserve_logical_streams(tmp_path):
    source = job(ensemble={"trajectories": 2, "steps": 2})
    run(source, output=tmp_path/"serial")
    source["parallel"] = {"trajectory_workers": 2}
    run(source, output=tmp_path/"process")
    assert_same_checkpoints(tmp_path/"serial", tmp_path/"process")


@pytest.mark.parametrize("corruption", ["bead_shape", "diagnostic_precision", "negative_absolute",
                                       "history_increment", "terminal_diagnostic", "missing_diagnostic",
                                       "legacy_state_class", "occupied"])
def test_resealed_rp_checkpoint_corruption_is_rejected(tmp_path, corruption):
    source = job(ensemble={"trajectories": 1, "steps": 2, "sample_stride": 1})
    run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    payload = arrays(path)
    if corruption == "bead_shape":
        payload["q"] = payload["q"][:1]
    elif corruption == "diagnostic_precision":
        payload["hop_energy_change"] = payload["hop_energy_change"].astype(np.float32)
    elif corruption == "negative_absolute":
        payload["hop_energy_change_abs"] = np.asarray(-1.)
    elif corruption == "history_increment":
        payload["history_hop_energy_change"][1] = 1.
        payload["history_hop_energy_change_abs"][:] = [0., 1., 1.]
        payload["hop_energy_change_abs"] = np.asarray(1.)
    elif corruption == "terminal_diagnostic":
        payload["hop_energy_change"] = np.asarray(.1)
        payload["hop_energy_change_abs"] = np.asarray(.1)
    elif corruption == "missing_diagnostic":
        del payload["hop_energy_change"]
    elif corruption == "occupied":
        payload["occupied"][:] = 0
    else:
        metadata = json.loads(str(payload["metadata"].item()))
        metadata["state_class"] = "IESHState"
        payload["metadata"] = np.asarray(json.dumps(metadata))
    reseal(path, payload)
    with pytest.raises(CheckpointError):
        resume_run(tmp_path/"run")


def test_rp_failed_load_does_not_partially_restore_rng(tmp_path):
    source = job(ensemble={"trajectories": 1, "steps": 1})
    run(source, output=tmp_path/"run")
    config, system = JobSpec.from_mapping(source).resolve()
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    payload = arrays(path)
    metadata = json.loads(str(payload["metadata"].item()))
    metadata["rngs"]["dynamics"] = {"broken": True}
    payload["metadata"] = np.asarray(json.dumps(metadata))
    reseal(path, payload)
    streams = workflow._streams(config, 0)
    for generator in streams:
        generator.random(11)
    before = [copy.deepcopy(generator.bit_generator.state) for generator in streams]
    dynamics = make_dynamics(config, system, streams[2])
    try:
        with pytest.raises(CheckpointError, match="random"):
            workflow._load(path, config, fingerprint(scientific_config(config), system), 0, dynamics, streams)
        assert before == [generator.bit_generator.state for generator in streams]
    finally:
        dynamics.potential.close()


@pytest.mark.parametrize("key,value", [("nelectrons", 1), ("electronic_substeps", 8),
                                       ("max_hop_probability", .2), ("gap_tolerance", 1e-8),
                                       ("dt", .05), ("integrator", "normal-mode-bab")])
def test_rp_scientific_dynamics_options_enter_identity(key, value):
    before = job()
    after = copy.deepcopy(before)
    after["dynamics"][key] = value
    if key == "nelectrons":
        after["initial"]["occupied"] = [0]
    assert identity(before) != identity(after)


def test_rp_beads_initial_conditions_and_method_have_distinct_identities():
    before = job()
    after = copy.deepcopy(before)
    after["beads"] = 8
    assert identity(before) != identity(after)
    after = copy.deepcopy(before)
    after["initial"]["occupied"] = [0, 2]
    assert identity(before) != identity(after)
    after = copy.deepcopy(before)
    after["dynamics"].update(block_steps=9, loop_backend="python")
    after["parallel"] = {"trajectory_workers": 2}
    assert identity(before) == identity(after)
    classical = job(beads=1)
    alternative = copy.deepcopy(classical)
    alternative["method"] = "iesh"
    assert identity(classical) != identity(alternative)


@pytest.mark.parametrize("key,value", [("rescaling", "ring"), ("decoherence", "edc"),
                                       ("frustrated_hops", "reverse"), ("electronic_device", "cuda"),
                                       ("nelectrons", True), ("max_hop_probability", 1.1)])
def test_rp_unsupported_scientific_variants_fail_explicitly(key, value):
    source = job()
    source["dynamics"][key] = value
    with pytest.raises(InputError):
        JobSpec.from_mapping(source).resolve()


def test_rp_rejects_complex_soc_at_input_and_preserves_classical_iesh_boundary():
    source = job(model={"name": "lvc", "params": {
        "h0": {"real": [[-.2, .1], [.1, .2]], "imag": [[0., .02], [-.02, 0.]]},
        "linear": [[[.03, 0.], [0., -.03]]]}})
    source["initial"]["occupied"] = [0]
    source["dynamics"]["nelectrons"] = 1
    with pytest.raises(InputError, match="real symmetric"):
        JobSpec.from_mapping(source).resolve()
    with pytest.raises(InputError, match="beads=1"):
        JobSpec.from_mapping(job(method="iesh")).resolve()


def test_pre_t25_iesh_identity_remains_byte_compatible():
    # Captured before editing main ceee59cd5f702165b56c5e541a503d8810fb9111.
    source = {"task": "nonadiabatic", "method": "iesh", "beta": 20., "beads": 1, "seed": 72,
              "model": {"name": "newns_anderson", "params": {"nband": 4}},
              "system": {"masses": [2000.]}, "dynamics": {"native": False, "dt": .02},
              "ensemble": {"trajectories": 2, "steps": 2}}
    assert identity(source) == "3d57654ec43ae6b3be5bdc4075882a8d3a37d66270406569ea9a42387024f09c"


def test_p1_public_workflow_matches_iesh_without_changing_legacy_payload(tmp_path):
    source = job(beads=1)
    run(source, output=tmp_path/"rp")
    source["method"] = "iesh"
    run(source, output=tmp_path/"classical")
    classical = arrays(tmp_path/"classical/results/nonadiabatic.npz")
    ring = arrays(tmp_path/"rp/results/nonadiabatic.npz")
    assert "hop_energy_change" not in classical
    assert "method_variant" not in json.loads(str(classical["metadata"].item()))
    for key in classical.keys()-{"metadata"}:
        np.testing.assert_array_equal(classical[key], ring[key], err_msg=key)
    for index in range(2):
        name = f"checkpoints/nonadiabatic_{index:06d}.npz"
        old, new = arrays(tmp_path/"classical"/name), arrays(tmp_path/"rp"/name)
        for key in old.keys()-{"metadata"}:
            np.testing.assert_array_equal(old[key], new[key], err_msg=key)
        old_meta, new_meta = (json.loads(str(value["metadata"].item())) for value in (old, new))
        assert old_meta["rngs"] == new_meta["rngs"]
