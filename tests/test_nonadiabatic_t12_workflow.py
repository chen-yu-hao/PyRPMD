"""Public IESH/multistate-MASH contracts, identities and atomic restarts."""
import copy
import json

import numpy as np
import pytest

from pyrpmd import JobSpec, analyze_run, prepare, resume_run, run
from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.iesh import IESHDynamics
from pyrpmd.input import InputError
from pyrpmd.io import fingerprint
from pyrpmd.mash import MultistateMASHDynamics
from pyrpmd.nonadiabatic_registry import make_dynamics, potential_from_identity
from pyrpmd.propagation import scientific_config
from pyrpmd.workflows import nonadiabatic_multistate as workflow


METHODS = ("iesh", "multistate-mash")


def job(method="iesh", **changes):
    model = {"name": "newns_anderson", "params": {"nband": 6}} if method == "iesh" else {
        "name": "lvc", "params": {
            "h0": [[-.05, .02, .01], [.02, 0., .015], [.01, .015, .06]],
            "linear": [np.diag([.02, -.01, .03]).tolist()], "harmonic": [.001]}}
    value = {
        "task": "nonadiabatic", "method": method, "beta": 4., "beads": 1, "seed": 72,
        "model": model, "system": {"masses": [2000.], "positions": [[-.1, 0., 0.]]},
        "initial": {"kind": "point", "momenta": [[40., 0., 0.]]},
        "dynamics": {"dt": .2, "native": False, "block_steps": 3},
        "ensemble": {"trajectories": 3, "steps": 7, "sample_stride": 2},
    }
    value.update(changes)
    return value


def arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {name: archive[name] for name in archive.files}


def driver_class(method):
    return IESHDynamics if method == "iesh" else MultistateMASHDynamics


def rewrite(path, payload):
    atomic_savez(path, kind=workflow._KIND, **payload)


@pytest.mark.parametrize("method", METHODS)
def test_public_workflow_preserves_physical_estimator_and_completed_analysis(tmp_path, monkeypatch, method):
    source = job(method)
    original = copy.deepcopy(source)
    prepared = prepare(source, numerical=True)
    assert prepared["ready"]
    assert prepared["nonadiabatic_preparation"]["matrix_gradient_check"]["max_absolute_gradient_error"] < 1e-8
    result = run(source, output=tmp_path/"run")
    assert source == original
    assert result["completed"]
    saved = arrays(tmp_path/"run/results/nonadiabatic.npz")
    metadata = json.loads(str(saved["metadata"].item()))
    np.testing.assert_array_equal(saved["steps"], [0, 2, 4, 6, 7])
    count = 3 if method == "iesh" else 1
    np.testing.assert_allclose(saved["populations"].sum(axis=-1), count, atol=1e-10)
    np.testing.assert_array_equal(saved["populations"], saved["diabatic_density"].diagonal(axis1=-2, axis2=-1).real)
    np.testing.assert_allclose(saved["population_sem"], saved["populations"].std(axis=0, ddof=1)/np.sqrt(3))
    assert metadata["population_normalization"] == count
    assert metadata["population_basis"] == "fixed_diabatic"
    assert metadata["statistics_axis"] == "trajectory"
    if method == "iesh":
        assert metadata["population_estimator"] == "slater_one_particle_density"
        assert saved["active_electronic_states"].shape == (3, 5, 3)
    else:
        assert metadata["population_estimator"] == "runeson_manolopoulos_2023"
        assert np.any(np.linalg.eigvalsh(saved["diabatic_density"]) < 0)
        assert not np.allclose(saved["adiabatic_population_mean"], saved["active_population_mean"])
    def forbidden(*args, **kwargs):
        raise AssertionError("completed analysis/resume must not propagate")
    monkeypatch.setattr(driver_class(method), "step", forbidden)
    assert analyze_run(tmp_path/"run", recompute=True)["final_populations"] == result["final_populations"]
    assert resume_run(tmp_path/"run")["final_populations"] == result["final_populations"]


@pytest.mark.parametrize("method", METHODS)
def test_interrupted_resume_is_bitwise_identical_including_all_rngs(tmp_path, monkeypatch, method):
    source = job(method)
    run(source, output=tmp_path/"reference")
    cls = driver_class(method)
    original = cls.step
    def interrupt(self, state):
        if state.step == 5:
            raise RuntimeError("deliberate T12 interruption")
        return original(self, state)
    monkeypatch.setattr(cls, "step", interrupt)
    with pytest.raises(RuntimeError, match="deliberate T12"):
        run(source, output=tmp_path/"interrupted")
    assert not (tmp_path/"interrupted/results/summary.json").exists()
    path = tmp_path/"interrupted/checkpoints/nonadiabatic_000000.npz"
    partial = arrays(path)
    assert partial["step"] == 3
    np.testing.assert_array_equal(partial["history_steps"], [0, 2])
    assert set(json.loads(str(partial["metadata"].item()))["rngs"]) == {"nuclear", "electronic", "dynamics"}
    monkeypatch.setattr(cls, "step", original)
    resumed = resume_run(tmp_path/"interrupted", execution={"block_steps": 4, "loop_backend": "python"})
    assert resumed["completed"]
    for index in range(3):
        name = f"checkpoints/nonadiabatic_{index:06d}.npz"
        before, after = arrays(tmp_path/"reference"/name), arrays(tmp_path/"interrupted"/name)
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)


@pytest.mark.parametrize("method", METHODS)
def test_trajectory_process_execution_preserves_logical_random_streams(tmp_path, method):
    source = job(method, ensemble={"trajectories": 2, "steps": 3, "sample_stride": 2})
    run(source, output=tmp_path/"serial")
    source["parallel"] = {"trajectory_workers": 2}
    run(source, output=tmp_path/"process")
    for index in range(2):
        name = f"checkpoints/nonadiabatic_{index:06d}.npz"
        serial, process = arrays(tmp_path/"serial"/name), arrays(tmp_path/"process"/name)
        for key in serial:
            np.testing.assert_array_equal(serial[key], process[key], err_msg=key)


@pytest.mark.parametrize("method", METHODS)
def test_zero_step_single_trajectory_does_not_claim_uncertainty(tmp_path, method):
    result = run(job(method, ensemble={"trajectories": 1, "steps": 0}), output=tmp_path/"zero")
    assert result["frames"] == 1 and result["final_population_sem"] is None
    assert result["sem_unavailable_reason"] == "fewer_than_two_trajectories"
    assert all("sem" not in key for key in arrays(tmp_path/"zero/results/nonadiabatic.npz") if key != "metadata")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("corruption", ["state_precision", "history_precision", "terminal_energy", "bad_rng", "bad_active"])
def test_resealed_invalid_checkpoints_are_rejected(tmp_path, method, corruption):
    source = job(method, ensemble={"trajectories": 1, "steps": 2, "sample_stride": 1})
    run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    payload = arrays(path)
    if corruption == "state_precision":
        payload["coefficients"] = payload["coefficients"].astype(np.complex64)
    elif corruption == "history_precision":
        payload["history_population"] = payload["history_population"].astype(np.float32)
    elif corruption == "terminal_energy":
        payload["history_energy"][-1] += 1.
    elif corruption == "bad_rng":
        metadata = json.loads(str(payload["metadata"].item()))
        metadata["rngs"]["electronic"] = {"broken": True}
        payload["metadata"] = np.asarray(json.dumps(metadata))
    elif method == "iesh":
        payload["occupied"][1] = payload["occupied"][0]
    else:
        payload["active_surface"] = np.asarray(100, dtype=np.int64)
    rewrite(path, payload)
    with pytest.raises(CheckpointError):
        resume_run(tmp_path/"run")


@pytest.mark.parametrize("method", METHODS)
def test_failed_load_does_not_partially_restore_random_streams(tmp_path, method):
    source = job(method, ensemble={"trajectories": 1, "steps": 1})
    config, system = JobSpec.from_mapping(source).resolve()
    run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    payload = arrays(path)
    metadata = json.loads(str(payload["metadata"].item()))
    metadata["rngs"]["dynamics"] = {"broken": True}
    payload["metadata"] = np.asarray(json.dumps(metadata))
    rewrite(path, payload)
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


@pytest.mark.parametrize("method", METHODS)
def test_missing_trajectory_preserves_previous_complete_reduction(tmp_path, method):
    run(job(method), output=tmp_path/"run")
    result = tmp_path/"run/results/nonadiabatic.npz"
    before = result.read_bytes()
    (tmp_path/"run/checkpoints/nonadiabatic_000001.npz").unlink()
    with pytest.raises((FileNotFoundError, CheckpointError)):
        analyze_run(tmp_path/"run", recompute=True)
    assert result.read_bytes() == before


@pytest.mark.parametrize("method", METHODS)
def test_resealed_wrong_initial_preparation_is_rejected(tmp_path, method):
    run(job(method), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    payload = arrays(path)
    payload["history_centroid_p"][0, 0, 0] += .01
    rewrite(path, payload)
    with pytest.raises(CheckpointError, match="initial preparation"):
        resume_run(tmp_path/"run")


@pytest.mark.parametrize("method", METHODS)
def test_valid_but_wrong_preparation_rng_is_rejected(tmp_path, method):
    run(job(method), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    payload = arrays(path)
    metadata = json.loads(str(payload["metadata"].item()))
    metadata["rngs"]["nuclear"] = np.random.default_rng(999).bit_generator.state
    payload["metadata"] = np.asarray(json.dumps(metadata))
    rewrite(path, payload)
    with pytest.raises(CheckpointError, match="random stream differs"):
        resume_run(tmp_path/"run")


@pytest.mark.parametrize("method", METHODS)
def test_new_methods_explicitly_reject_ring_polymer_and_complex_soc(method):
    with pytest.raises(InputError, match="beads=1"):
        JobSpec.from_mapping(job(method, beads=2)).resolve()
    raw = job(method, model={"name": "lvc", "params": {
        "h0": {"real": [[-.2, .1], [.1, .2]], "imag": [[0., .02], [-.02, 0.]]},
        "linear": [[[.03, 0.], [0., -.03]]]}})
    with pytest.raises(InputError, match="real symmetric"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("occupied", [[0, 0, 1], [0, 1], [0, 1, 99], [0., 1., 2.],
                                       [False, 1, 2], "012", [[0], [1, 2]]])
def test_iesh_initial_determinant_occupations_are_strict(occupied):
    raw = job()
    raw["initial"]["occupied"] = occupied
    with pytest.raises(InputError, match="occupied"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method,option,value", [
    ("iesh", "nelectrons", True), ("iesh", "nelectrons", 8),
    ("iesh", "max_hop_probability", 1.1), ("iesh", "electronic_substeps", 0),
    ("iesh", "frustrated_hops", "reverse"), ("iesh", "decoherence", "edc"),
    ("multistate-mash", "native", True), ("multistate-mash", "max_events", True),
    ("multistate-mash", "integrator", "cayley-bcb"), ("multistate-mash", "atol", -1.),
])
def test_inapplicable_or_invalid_new_dynamics_options_fail(method, option, value):
    raw = job(method)
    raw["dynamics"][option] = value
    with pytest.raises(InputError):
        JobSpec.from_mapping(raw).resolve()


def test_metal_model_cannot_drop_its_bare_nuclear_potential_in_other_methods():
    raw = job()
    raw["method"] = "rpsh"
    with pytest.raises(InputError, match="bare nuclear potential"):
        JobSpec.from_mapping(raw).resolve()


def test_direct_registry_guards_bare_potential_and_fixed_event_integrator():
    config, system = JobSpec.from_mapping(job()).resolve()
    config["method"] = "rpsh"
    with pytest.raises(ValueError, match="bare nuclear potential"):
        make_dynamics(config, system, np.random.default_rng())
    config, system = JobSpec.from_mapping(job("multistate-mash")).resolve()
    config["dynamics"]["integrator"] = "cayley-bcb"
    with pytest.raises(ValueError, match="adaptive-dop853"):
        make_dynamics(config, system, np.random.default_rng())


def test_one_state_iesh_full_occupation_limit_and_mash_rejection():
    model = {"name": "lvc", "params": {"h0": [[.1]], "linear": [[[.02]]]}}
    config, _ = JobSpec.from_mapping(job("iesh", model=model)).resolve()
    assert config["dynamics"]["nelectrons"] == 1
    assert config["initial"]["occupied"] == [0]
    with pytest.raises(InputError, match="at least two"):
        JobSpec.from_mapping(job("multistate-mash", model=model)).resolve()


@pytest.mark.parametrize("method,key,value", [
    ("iesh", "nelectrons", 2), ("iesh", "electronic_substeps", 8),
    ("iesh", "max_hop_probability", .2), ("iesh", "gap_tolerance", 1e-8),
    ("multistate-mash", "rtol", 1e-8), ("multistate-mash", "atol", 1e-10),
    ("multistate-mash", "max_step", .01), ("multistate-mash", "max_events", 16),
    ("multistate-mash", "gap_tolerance", 1e-8),
])
def test_every_new_scientific_option_enters_run_identity(method, key, value):
    source = job(method)
    before, system = JobSpec.from_mapping(source).resolve()
    source["dynamics"][key] = value
    after, other = JobSpec.from_mapping(source).resolve()
    assert fingerprint(scientific_config(before), system) != fingerprint(scientific_config(after), other)


def test_metal_model_identity_roundtrip_and_scientific_parameters():
    source = job()
    before, system = JobSpec.from_mapping(source).resolve()
    model = potential_from_identity(before["model"])
    try:
        assert model.to_dict() == before["model"]
    finally:
        model.close()
    for key, value in {"nband": 8, "hybridization": .03, "level_slope": .02,
                       "bare_force_constant": .002, "chemical_potential": .1,
                       "discretization": "midpoint"}.items():
        changed = copy.deepcopy(source)
        changed["model"]["params"][key] = value
        after, other = JobSpec.from_mapping(changed).resolve()
        assert fingerprint(scientific_config(before), system) != fingerprint(scientific_config(after), other)


@pytest.mark.parametrize("method,expected", [
    ("nrpmd", "97f7de66372fad0fffe76cb7a40ebd6d020d354b18f0cdae76a6aab210e5ed30"),
    ("rp-mash", "5c284f4007fa2c94a2dfd3686c043d55e1e976aa3b93d4309cc58b8a374feab2"),
    ("rpsh", "610d0b3f74ed3194720998b703539619ff19799ac795f6d6229461993a0f44bb"),
    ("rpsh-ba", "4139f028737d0880e8ebb33f68472dbfd8096c93790d9f7295ee0a347ffc2708"),
    ("rpsh-ca", "5d9920dc220aa5a63fd21a95f33b6a48f44445c29e4dc5a866311d997fc9f256"),
    ("rpsh-dm", "271b5998fa15655a03734af715b12ec695d2a463f01a1e012f6e3f2b2a848c7c"),
    ("sm-nrpmd", "5d73f916b0950789106e22d34aa72c36ecf85399427bef3aa73d84822afa8723"),
    ("soc-rpsh", "2af539ad057f715e7ac66ede6c58d85fe1579ac720825ea5e7d0058e939debac"),
])
def test_pre_t12_method_fingerprints_remain_byte_compatible(method, expected):
    # Captured from unmodified main 8cc402f before the T12 implementation.
    raw = {"task": "nonadiabatic", "method": method, "beta": 20., "beads": 2,
           "model": {"name": "tully1"}, "system": {"masses": [2000.]},
           "dynamics": {"native": False, "dt": .02}, "ensemble": {"trajectories": 2, "steps": 2}}
    config, system = JobSpec.from_mapping(raw).resolve()
    assert fingerprint(scientific_config(config), system) == expected
