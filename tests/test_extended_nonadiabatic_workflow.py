"""Public extended methods: physical observables, exact restarts and ownership."""
import copy
import json

import numpy as np
import pytest

from pyrpmd import JobSpec, run, prepare, resume_run, analyze_run
from pyrpmd.advanced_surface_hopping import AdvancedRPSHDynamics
from pyrpmd.checkpoint import atomic_savez, load_checkpoint, CheckpointError
from pyrpmd.io import fingerprint
from pyrpmd.propagation import scientific_config
from pyrpmd.rp_mash import RPMASHDynamics
from pyrpmd.spin_mapping import SMNRPMDDynamics
from pyrpmd.workflows import nonadiabatic_extensions as workflow


METHODS = ("sm-nrpmd", "rp-mash", "rpsh-ca", "rpsh-ba", "rpsh-dm", "soc-rpsh")


def job(method="sm-nrpmd", **changes):
    value = {"task": "nonadiabatic", "method": method, "beta": 4., "beads": 3, "seed": 72,
             "model": {"name": "tully1", "params": {}},
             "system": {"masses": [2000.], "positions": [[-.1, 0., 0.]]},
             "initial": {"kind": "point", "momenta": [[40., 0., 0.]]},
             "dynamics": {"dt": .2, "native": False, "block_steps": 3},
             "ensemble": {"trajectories": 3, "steps": 7, "sample_stride": 2}}
    value.update(changes)
    return value


def arrays(path):
    with load_checkpoint(path, require_integrity=True) as archive:
        return {name: archive[name] for name in archive.files}


def rewrite(path, payload):
    atomic_savez(path, kind=workflow._KIND, **payload)


def driver_class(method):
    return SMNRPMDDynamics if method == "sm-nrpmd" else RPMASHDynamics if method == "rp-mash" else AdvancedRPSHDynamics


@pytest.mark.parametrize("method", METHODS)
def test_public_prepare_run_and_completed_analysis_are_consistent(tmp_path, monkeypatch, method):
    source = job(method)
    original = copy.deepcopy(source)
    ready = prepare(source, numerical=True)
    assert ready["ready"]
    assert ready["nonadiabatic_preparation"]["matrix_gradient_check"]["max_absolute_gradient_error"] < 1e-8
    result = run(source, output=tmp_path/"run")
    assert source == original
    assert result["completed"]
    saved = arrays(tmp_path/"run/results/nonadiabatic.npz")
    metadata = json.loads(str(saved["metadata"].item()))
    np.testing.assert_array_equal(saved["steps"], [0, 2, 4, 6, 7])
    assert saved["populations"].shape == (3, 5, 2)
    np.testing.assert_allclose(saved["population_sem"], saved["populations"].std(axis=0, ddof=1)/np.sqrt(3))
    assert metadata["statistics_axis"] == "trajectory"
    assert metadata["electronic_backend"] == "numpy" and metadata["electronic_device"] == "cpu"
    if method == "rp-mash":
        assert metadata["population_basis"] == "centroid_adiabatic"
        assert metadata["coherence_available"] is False
        assert "diabatic_density" not in saved and "density_mean" not in saved
        assert "adiabatic_amplitude_populations" not in saved
        np.testing.assert_array_equal(saved["population_mean"], saved["active_surface_fractions"])
        np.testing.assert_array_equal(saved["active_surface_counts"], saved["population_mean"]*3)
    else:
        assert metadata["population_basis"] == "fixed_diabatic"
        assert saved["diabatic_density"].shape == (3, 5, 2, 2)
        np.testing.assert_array_equal(saved["populations"], saved["diabatic_density"].diagonal(axis1=-2, axis2=-1).real)
    if method not in ("sm-nrpmd", "rp-mash"):
        assert saved["adiabatic_amplitude_populations"].shape == (3, 5, 2)
        assert saved["active_surface_counts"].dtype == np.int64
    def forbidden(*args, **kwargs):
        raise AssertionError("completed analysis/resume must not propagate")
    monkeypatch.setattr(driver_class(method), "step", forbidden)
    assert analyze_run(tmp_path/"run", recompute=True)["final_populations"] == result["final_populations"]
    assert resume_run(tmp_path/"run")["final_populations"] == result["final_populations"]


@pytest.mark.parametrize("method", METHODS)
def test_interrupted_resume_preserves_electronic_gauge_all_rngs_and_history(tmp_path, monkeypatch, method):
    source = job(method)
    run(source, output=tmp_path/"reference")
    cls = driver_class(method)
    original = cls.step
    def interrupt(self, state):
        if state.step == 5:
            raise RuntimeError("deliberate extension interruption")
        return original(self, state)
    monkeypatch.setattr(cls, "step", interrupt)
    with pytest.raises(RuntimeError, match="deliberate"):
        run(source, output=tmp_path/"interrupted")
    assert not (tmp_path/"interrupted/results/summary.json").exists()
    path = tmp_path/"interrupted/checkpoints/nonadiabatic_000000.npz"
    partial = arrays(path)
    assert partial["step"] == 3
    np.testing.assert_array_equal(partial["history_steps"], [0, 2])
    assert set(json.loads(str(partial["metadata"].item()))["rngs"]) == {"nuclear", "electronic", "dynamics"}
    if method == "rpsh-ba":
        assert partial["basis_vectors"].shape == (3, 2, 2)
    if method == "rpsh-dm":
        assert partial["coefficients"].shape == (3, 2)
    if method == "rp-mash":
        assert partial["sampling"].item() == "population"
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
def test_process_placement_keeps_each_trajectory_identical(tmp_path, method):
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
def test_single_trajectory_zero_step_has_no_sem(tmp_path, method):
    result = run(job(method, ensemble={"trajectories": 1, "steps": 0, "sample_stride": 9}), output=tmp_path/"zero")
    assert result["frames"] == 1 and result["final_population_sem"] is None
    assert result["sem_unavailable_reason"] == "fewer_than_two_trajectories"
    saved = arrays(tmp_path/"zero/results/nonadiabatic.npz")
    assert all("sem" not in key for key in saved if key != "metadata")


@pytest.mark.parametrize("method", ["sm-nrpmd", "rp-mash", "rpsh-ba", "rpsh-dm"])
def test_harmonic_nuclear_samples_have_independent_streams_and_zero_spectators(tmp_path, method):
    source = job(method, model={"name": "spin_boson", "params": {
        "frequencies": [.5], "couplings": [.08], "masses": [1.], "tunnelling": .1}},
        system={"masses": [1.]}, initial={"kind": "harmonic_ring", "frequencies": [.5]},
        ensemble={"trajectories": 3, "steps": 0, "sample_stride": 1})
    run(source, output=tmp_path/"harmonic")
    saved = [arrays(tmp_path/"harmonic/checkpoints"/f"nonadiabatic_{index:06d}.npz") for index in range(3)]
    assert not np.array_equal(saved[0]["q"], saved[1]["q"])
    assert not np.array_equal(saved[0]["p"], saved[1]["p"])
    for item in saved:
        np.testing.assert_array_equal(item["q"][..., 1:], 0)
        np.testing.assert_array_equal(item["p"][..., 1:], 0)


@pytest.mark.parametrize("method", METHODS)
def test_resealed_history_dtype_or_terminal_mismatch_is_rejected(tmp_path, method):
    run(job(method), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000001.npz"
    original = arrays(path)
    modified = copy.deepcopy(original)
    modified["history_population"] = modified["history_population"].astype(np.float32)
    rewrite(path, modified)
    with pytest.raises(CheckpointError, match="dtype"):
        analyze_run(tmp_path/"run", recompute=True)
    modified = copy.deepcopy(original)
    modified["history_energy"][-1] += 1.
    rewrite(path, modified)
    with pytest.raises(CheckpointError, match="final observation"):
        resume_run(tmp_path/"run")


@pytest.mark.parametrize("method", METHODS)
def test_resealed_state_electronic_dtype_is_not_silently_cast(tmp_path, method):
    run(job(method, ensemble={"trajectories": 1, "steps": 1, "sample_stride": 1}), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    key = "spin" if method in ("sm-nrpmd", "rp-mash") else "coefficients"
    saved[key] = saved[key].astype(np.float32 if key == "spin" else np.complex64)
    rewrite(path, saved)
    with pytest.raises(CheckpointError, match="dtype"):
        resume_run(tmp_path/"run")


def test_ba_frame_must_diagonalize_current_q_even_between_saved_frames(tmp_path, monkeypatch):
    source = job("rpsh-ba")
    step = AdvancedRPSHDynamics.step
    def interrupt(self, state):
        if state.step == 4:
            raise RuntimeError("interrupted")
        return step(self, state)
    monkeypatch.setattr(AdvancedRPSHDynamics, "step", interrupt)
    with pytest.raises(RuntimeError):
        run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    assert saved["step"] == 3 and saved["history_steps"][-1] == 2
    angle = .31
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    saved["basis_vectors"] = saved["basis_vectors"] @ rotation
    rewrite(path, saved)
    monkeypatch.setattr(AdvancedRPSHDynamics, "step", step)
    with pytest.raises(CheckpointError, match="state"):
        resume_run(tmp_path/"run")


def test_saved_mash_preparation_cannot_change_to_uniform_sphere(tmp_path):
    run(job("rp-mash"), output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    saved["sampling"] = np.asarray("uniform-sphere")
    rewrite(path, saved)
    with pytest.raises(CheckpointError, match="sampling"):
        analyze_run(tmp_path/"run", recompute=True)


def test_failed_rng_validation_never_partially_restores_any_stream(tmp_path):
    source = job("rpsh-dm", ensemble={"trajectories": 1, "steps": 1, "sample_stride": 1})
    config, system = JobSpec.from_mapping(source).resolve()
    run(source, output=tmp_path/"run")
    path = tmp_path/"run/checkpoints/nonadiabatic_000000.npz"
    saved = arrays(path)
    metadata = json.loads(str(saved["metadata"].item()))
    metadata["rngs"]["electronic"] = {"broken": True}
    saved["metadata"] = np.asarray(json.dumps(metadata))
    rewrite(path, saved)
    streams = workflow._streams(config, 0)
    for generator in streams:
        generator.random(11)
    before = [copy.deepcopy(generator.bit_generator.state) for generator in streams]
    dynamics = workflow._driver(config, system, streams[2])
    try:
        with pytest.raises(CheckpointError, match="random"):
            workflow._load(path, config, fingerprint(scientific_config(config), system), 0, dynamics, streams)
        assert before == [generator.bit_generator.state for generator in streams]
    finally:
        dynamics.potential.close()


def test_missing_trajectory_does_not_publish_reduced_success_ensemble(tmp_path):
    run(job("sm-nrpmd"), output=tmp_path/"run")
    output = tmp_path/"run/results/nonadiabatic.npz"
    before = output.read_bytes()
    (tmp_path/"run/checkpoints/nonadiabatic_000001.npz").unlink()
    with pytest.raises((FileNotFoundError, CheckpointError)):
        analyze_run(tmp_path/"run", recompute=True)
    assert output.read_bytes() == before


def test_soc_density_retains_complex_coherence(tmp_path):
    h0 = {"real": [[.2, .1], [.1, -.2]], "imag": [[0., .07], [-.07, 0.]]}
    linear = {"real": [[[.1, .03], [.03, -.1]]], "imag": [[[0., .02], [-.02, 0.]]]}
    source = job("soc-rpsh", model={"name": "lvc", "params": {"h0": h0, "linear": linear}},
                 system={"masses": [10.], "positions": [[.2, 0., 0.]]},
                 initial={"kind": "point", "momenta": [[1., 0., 0.]]})
    run(source, output=tmp_path/"soc")
    saved = arrays(tmp_path/"soc/results/nonadiabatic.npz")
    rho = saved["diabatic_density"]
    assert np.max(abs(rho.imag)) > .01
    np.testing.assert_allclose(rho, rho.conj().swapaxes(-1, -2), atol=1e-14)
    np.testing.assert_allclose(rho@rho, rho, atol=1e-12)


def test_sm_backend_is_recorded_but_not_physics_identity(tmp_path):
    pytest.importorskip("pyrpmd._electronic")
    source = job("sm-nrpmd", ensemble={"trajectories": 1, "steps": 2, "sample_stride": 1})
    first = run(source, output=tmp_path/"numpy")
    source["dynamics"]["electronic_backend"] = "native"
    second = run(source, output=tmp_path/"native")
    assert first["physics_fingerprint"] == second["physics_fingerprint"]
    assert first["electronic_backend"] == "numpy" and second["electronic_backend"] == "native"
    left, right = arrays(tmp_path/"numpy/results/nonadiabatic.npz"), arrays(tmp_path/"native/results/nonadiabatic.npz")
    np.testing.assert_allclose(left["diabatic_density"], right["diabatic_density"], atol=3e-14)
    assert json.loads(str(right["metadata"].item()))["electronic_backend"] == "native"
