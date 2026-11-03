"""BA-RP-MASH public inputs, estimator semantics and exact checkpoint replay."""
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
from pyrpmd.rp_mash_bead import BARPMASHDynamics, BARPMASHState
from pyrpmd.workflows import nonadiabatic_extensions as workflow


def job(method="rp-mash-ba", **changes):
    value = {
        "task": "nonadiabatic", "method": method, "beta": 4., "beads": 4, "seed": 72,
        "model": {"name": "spin_boson", "params": {
            "frequencies": [.5], "couplings": [.08], "masses": [1.], "tunnelling": .1}},
        "system": {"masses": [1.]},
        "initial": {"kind": "harmonic_ring", "frequencies": [.5]},
        "dynamics": {"dt": .025, "block_steps": 3},
        "ensemble": {"trajectories": 2, "steps": 7, "sample_stride": 2},
    }
    value.update(changes)
    return value


def arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {key: archive[key] for key in archive.files}


def test_public_run_reports_common_bead_populations_without_density(tmp_path, monkeypatch):
    source = job()
    untouched = copy.deepcopy(source)
    ready = prepare(source, numerical=True)
    detail = ready["nonadiabatic_preparation"]
    assert ready["ready"] and detail["electronic_state_class"] == "BARPMASHState"
    assert detail["matrix_gradient_check"]["max_absolute_gradient_error"] < 1e-8
    result = run(source, output=tmp_path/"run")
    assert source == untouched
    assert result["completed"] and result["physical_converged"] is False
    saved = arrays(tmp_path/"run/results/nonadiabatic.npz")
    metadata = json.loads(str(saved["metadata"].item()))
    assert metadata["method"] == "rp-mash-ba"
    assert metadata["population_basis"] == "common_bead_adiabatic"
    assert metadata["active_surface_basis"] == "common_bead_adiabatic"
    assert metadata["population_estimator"] == "active_surface_indicator"
    assert metadata["density_basis"] is None and metadata["coherence_available"] is False
    assert metadata["electronic_initial_sampling"] == "adiabatic_hemisphere_importance_abs_Sz"
    assert "diabatic_density" not in saved and "density_mean" not in saved
    assert "adiabatic_amplitude_populations" not in saved
    np.testing.assert_array_equal(saved["steps"], [0, 2, 4, 6, 7])
    assert saved["populations"].shape == (2, 5, 2)
    np.testing.assert_array_equal(saved["populations"], np.eye(2)[saved["active_surfaces"]])
    np.testing.assert_array_equal(saved["population_mean"], saved["active_surface_fractions"])
    np.testing.assert_allclose(saved["population_sem"], saved["populations"].std(axis=0, ddof=1)/np.sqrt(2))

    def no_propagation(*args, **kwargs):
        raise AssertionError("completed analysis/resume must not propagate")

    monkeypatch.setattr(BARPMASHDynamics, "step", no_propagation)
    assert analyze_run(tmp_path/"run", recompute=True)["final_populations"] == result["final_populations"]
    assert resume_run(tmp_path/"run")["final_populations"] == result["final_populations"]


def test_interrupted_resume_preserves_spin_all_rngs_and_every_observation(tmp_path, monkeypatch):
    source = job()
    run(source, output=tmp_path/"reference")
    step = BARPMASHDynamics.step

    def interrupt(self, state):
        if state.step == 5:
            raise RuntimeError("intentional BA-RP-MASH interruption")
        return step(self, state)

    monkeypatch.setattr(BARPMASHDynamics, "step", interrupt)
    with pytest.raises(RuntimeError, match="intentional"):
        run(source, output=tmp_path/"interrupted")
    assert not (tmp_path/"interrupted/results/summary.json").exists()
    partial = arrays(tmp_path/"interrupted/checkpoints/nonadiabatic_000000.npz")
    assert int(partial["step"]) == 3
    np.testing.assert_array_equal(partial["history_steps"], [0, 2])
    assert partial["spin"].shape == (3,) and partial["sampling"].item() == "population"
    assert np.ptp(partial["q"][:, 0, 0]) > .01  # Actual bead fluctuations, not cloned centroids.
    metadata = json.loads(str(partial["metadata"].item()))
    assert metadata["state_class"] == "BARPMASHState"
    assert set(metadata["rngs"]) == {"nuclear", "electronic", "dynamics"}
    monkeypatch.setattr(BARPMASHDynamics, "step", step)
    resumed = resume_run(tmp_path/"interrupted", execution={"block_steps": 4, "loop_backend": "python"})
    assert resumed["completed"]
    for index in range(2):
        relative = f"checkpoints/nonadiabatic_{index:06d}.npz"
        reference, replay = arrays(tmp_path/"reference"/relative), arrays(tmp_path/"interrupted"/relative)
        assert reference.keys() == replay.keys()
        for key in reference:
            np.testing.assert_array_equal(reference[key], replay[key], err_msg=key)


def test_process_placement_keeps_ba_trajectories_bitwise_identical(tmp_path):
    source = job(ensemble={"trajectories": 2, "steps": 2, "sample_stride": 1})
    run(source, output=tmp_path/"serial")
    source["parallel"] = {"trajectory_workers": 2}
    run(source, output=tmp_path/"process")
    for index in range(2):
        relative = f"checkpoints/nonadiabatic_{index:06d}.npz"
        serial, process = arrays(tmp_path/"serial"/relative), arrays(tmp_path/"process"/relative)
        assert serial.keys() == process.keys()
        for key in serial:
            np.testing.assert_array_equal(serial[key], process[key], err_msg=key)


@pytest.mark.parametrize("origin,target", [("rp-mash", "rp-mash-ba"), ("rp-mash-ba", "rp-mash")])
def test_ca_and_ba_checkpoint_spins_cannot_be_reinterpreted(tmp_path, origin, target):
    source = job(origin, ensemble={"trajectories": 1, "steps": 1, "sample_stride": 1})
    run(source, output=tmp_path/"run")
    source["method"] = target
    config, system = JobSpec.from_mapping(source).resolve()
    streams = workflow._streams(config, 0)
    for generator in streams:
        generator.random(5)
    before = [copy.deepcopy(generator.bit_generator.state) for generator in streams]
    dynamics = workflow._driver(config, system, streams[2])
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    encoded = path.read_bytes()
    try:
        with pytest.raises(CheckpointError, match="identity mismatch"):
            workflow._load(path, config, fingerprint(scientific_config(config), system), 0, dynamics, streams)
        assert before == [generator.bit_generator.state for generator in streams]
        assert path.read_bytes() == encoded
    finally:
        dynamics.potential.close()


@pytest.mark.parametrize("corruption,match", [
    ("state_class", "identity"), ("method", "identity"), ("spin_dtype", "dtype"),
    ("sampling", "sampling"), ("hemisphere", "state"), ("rng", "random"),
    ("indicator", "indicators"), ("extra_frame", "inventory"),
])
def test_resealed_checkpoint_corruption_fails_before_rng_restore(tmp_path, corruption, match):
    source = job(ensemble={"trajectories": 1, "steps": 1, "sample_stride": 1})
    config, system = JobSpec.from_mapping(source).resolve()
    run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    metadata = json.loads(str(saved["metadata"].item()))
    if corruption == "state_class":
        metadata["state_class"] = "RPMASHState"
    elif corruption == "method":
        metadata["method"] = "rp-mash"
    elif corruption == "spin_dtype":
        saved["spin"] = saved["spin"].astype(np.float32)
    elif corruption == "sampling":
        saved["sampling"] = np.asarray("uniform-sphere")
    elif corruption == "hemisphere":
        saved["active_surface"] = 1-saved["active_surface"]
    elif corruption == "rng":
        metadata["rngs"]["electronic"] = {"invalid": True}
    elif corruption == "indicator":
        saved["history_population"][0] = [.5, .5]
    else:
        saved["basis_vectors"] = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    saved["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(path, kind=workflow._KIND, **saved)
    streams = workflow._streams(config, 0)
    for generator in streams:
        generator.random(7)
    before = [copy.deepcopy(generator.bit_generator.state) for generator in streams]
    dynamics = workflow._driver(config, system, streams[2])
    try:
        with pytest.raises(CheckpointError, match=match):
            workflow._load(path, config, fingerprint(scientific_config(config), system), 0, dynamics, streams)
        assert before == [generator.bit_generator.state for generator in streams]
    finally:
        dynamics.potential.close()


@pytest.mark.parametrize("key,value", [
    ("dt", .03), ("rtol", 1e-8), ("atol", 1e-10), ("max_step", .002),
    ("max_events", 16), ("gap_tolerance", 1e-8),
])
def test_ba_scientific_solver_options_enter_identity(key, value):
    source = job()
    original, system = JobSpec.from_mapping(source).resolve()
    source["dynamics"][key] = value
    changed, other = JobSpec.from_mapping(source).resolve()
    assert fingerprint(scientific_config(original), system) != fingerprint(scientific_config(changed), other)


def test_ca_legacy_fingerprint_is_unchanged_and_ba_has_separate_identity():
    # Same historical input as T12's pinned main@8cc402f fingerprint.
    source = {"task": "nonadiabatic", "method": "rp-mash", "beta": 20., "beads": 2,
              "model": {"name": "tully1"}, "system": {"masses": [2000.]},
              "dynamics": {"native": False, "dt": .02}, "ensemble": {"trajectories": 2, "steps": 2}}
    ca, system = JobSpec.from_mapping(source).resolve()
    legacy = "5c284f4007fa2c94a2dfd3686c043d55e1e976aa3b93d4309cc58b8a374feab2"
    assert fingerprint(scientific_config(ca), system) == legacy
    source["method"] = "rp-mash-ba"
    ba, other = JobSpec.from_mapping(source).resolve()
    assert fingerprint(scientific_config(ba), other) != legacy


@pytest.mark.parametrize("options,match", [
    ({"native": True}, "native=false"), ({"native": "required"}, "native=false"),
    ({"integrator": "cayley-bcb"}, "adaptive-dop853"),
    ({"electronic_backend": "torch"}, "Unknown"), ({"decoherence": "edc"}, "Unknown"),
    ({"rescaling": "centroid"}, "Unknown"), ({"max_events": True}, "integer"),
    ({"gap_tolerance": 0.}, "positive"), ({"rtol": 1e-20}, "rtol"),
])
def test_unsupported_or_silently_adjustable_options_are_rejected(options, match):
    source = job()
    source["dynamics"].update(options)
    with pytest.raises(InputError, match=match):
        JobSpec.from_mapping(source).resolve()


def test_ba_rejects_more_than_two_states_and_complex_models():
    source = job(model={"name": "lvc", "params": {
        "h0": np.diag([-.2, 0., .2]), "linear": [np.diag([.1, .2, .3])]}},
        initial={"kind": "point"})
    with pytest.raises(InputError, match="exactly two"):
        JobSpec.from_mapping(source).resolve()
    source["model"]["params"] = {
        "h0": {"real": [[-.2, .1], [.1, .2]], "imag": [[0., .01], [-.01, 0.]]},
        "linear": [[[.1, 0.], [0., -.1]]]}
    with pytest.raises(InputError, match="real symmetric"):
        JobSpec.from_mapping(source).resolve()


def test_registry_keeps_the_supplied_rng_and_rejects_incompatible_integrator():
    config, system = JobSpec.from_mapping(job()).resolve()
    rng = np.random.default_rng(13)
    dynamics = make_dynamics(config, system, rng)
    try:
        assert isinstance(dynamics, BARPMASHDynamics) and dynamics.rng is rng
        streams = workflow._streams(config, 0)
        assert isinstance(workflow._initialize(config, system, dynamics, streams), BARPMASHState)
    finally:
        dynamics.potential.close()
    config["dynamics"]["integrator"] = "cayley-bcb"
    with pytest.raises(ValueError, match="adaptive-dop853"):
        make_dynamics(config, system, rng)


def test_harmonic_ba_checks_sampled_beads_instead_of_unused_reference_gap(tmp_path):
    source = job(model={"name": "lvc", "params": {
        "h0": [[0., 0.], [0., 0.]], "linear": [[[.1, 0.], [0., -.1]]],
        "harmonic": [.25]}}, ensemble={"trajectories": 1, "steps": 0, "sample_stride": 1})
    # The reference center has zero gap. Its randomly sampled beads do not.
    config, system = JobSpec.from_mapping(source).resolve()
    np.testing.assert_array_equal(system.geometry, 0.)
    result = run(source, output=tmp_path/"harmonic")
    assert result["completed"] and result["physical_converged"] is False
    saved = arrays(tmp_path/"harmonic/checkpoints/nonadiabatic_000000.npz")
    assert np.all(.2*np.abs(saved["q"][:, 0, 0]) > config["dynamics"]["gap_tolerance"])
    saved["q"][0, 0, 0] = 0.  # A degenerate actual bead must still fail on restore.
    atomic_savez(tmp_path/"harmonic/checkpoints/nonadiabatic_000000.npz", kind=workflow._KIND, **saved)
    with pytest.raises(CheckpointError, match="state"):
        resume_run(tmp_path/"harmonic")
    source["method"] = "rp-mash"
    with pytest.raises(InputError, match="degenerate"):
        JobSpec.from_mapping(source).resolve()
    source["method"] = "rp-mash-ba"
    source["initial"] = {"kind": "point"}
    with pytest.raises(InputError, match="degenerate"):
        JobSpec.from_mapping(source).resolve()


def test_ba_cli_validate_run_resume_and_zero_step_sem(tmp_path, capsys):
    from pyrpmd.cli import main
    source = tmp_path/"bead_mash.inp"
    source.write_text('''task = "nonadiabatic"
method = "rp-mash-ba"
beta = 4.0
beads = 3
[model]
name = "tully1"
[system]
masses = [2000.0]
[dynamics]
dt = "0.1 autime"
[ensemble]
trajectories = 1
steps = 0
sample_stride = 2
''')
    directory = tmp_path/"cli"
    assert main(["validate", str(source), "--check-pes", "--output", str(directory)]) == 0
    assert not directory.exists()
    capsys.readouterr()
    assert main(["run", str(source), "--output", str(directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["method"] == "rp-mash-ba" and result["frames"] == 1
    assert result["final_population_sem"] is None
    assert result["sem_unavailable_reason"] == "fewer_than_two_trajectories"
    assert "population_sem" not in arrays(directory/"results/nonadiabatic.npz")
    assert main(["resume", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["completed"]
