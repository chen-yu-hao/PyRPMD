"""Small phase/restart and metadata tests; never execute the64ps live control."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("fcntl", reason="the registered control uses POSIX cohort leases")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water import paired_timestep as control
from benchmarks.water import native_reference as native
from benchmarks.water.campaign import atomic_npz, sha256

# Reuse the excluded all-eight metadata fixture, not any live/historical data.
from test_water_preparation_coordinator import cohort, update_assessment, write_json


@pytest.fixture
def small_source(tmp_path):
    pytest.importorskip("openmm")
    directory = tmp_path / "original-small"
    config = native.Config(cells_per_axis=2, cutoff_angstrom=2.5, dt_fs=.1,
                           equilibration_steps=8, equilibration_record_stride=2,
                           segment_steps=8, segments=1, record_stride=2, checkpoint_steps=4,
                           sampling_refresh_fs=.4, seed=3101, platform="Reference",
                           trajectory_precision="float64")
    manifest = native.run(directory, config)
    selected = next(row for row in manifest["chunks"] if row["step"] == config.equilibration_steps)
    row = {"directory": str(directory), "manifest_sha256": sha256(directory / "manifest.json"),
           "checkpoint": selected, "config": asdict(config),
           "artifacts": manifest["artifacts"], "box_angstrom": manifest["box_angstrom"]}
    return row


def read_checkpoint(directory, index=-1):
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / manifest["chunks"][index]["file"], allow_pickle=False) as data:
        return manifest, {name: data[name].copy() for name in data.files}


def observed_steps(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    arrays = []
    for row in manifest["chunks"]:
        if row["observations"]:
            with np.load(directory / row["file"], allow_pickle=False) as data:
                arrays.append(data["step"].copy())
    return np.concatenate(arrays)


def test_shared_initial_draw_is_exactly_the_pending_COM_Maxwell_draw_and_never_repeated(small_source, tmp_path, monkeypatch):
    source = Path(small_source["directory"])
    original_hash = sha256(source / "manifest.json")
    with np.load(source / small_source["checkpoint"]["file"], allow_pickle=False) as data:
        checkpoint_rng = json.loads(str(data["rng_json"]))
        expected_q = data["q"].copy()
    calls = []
    sampler = native.sample_momenta
    def counted(*args, **kwargs):
        calls.append(kwargs["remove_com"])
        return sampler(*args, **kwargs)
    monkeypatch.setattr(native, "sample_momenta", counted)
    initial = tmp_path / "initial"
    result = control.prepare_initial(initial, small_source)
    repeated = control.prepare_initial(initial, small_source)
    with np.load(initial / "initial.npz", allow_pickle=False) as data:
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint_rng
        expected_p = sampler(rng, data["masses_au"], 298., remove_com=True)
        np.testing.assert_array_equal(data["p"], expected_p)
        np.testing.assert_array_equal(data["q"], expected_q)
        assert json.loads(str(data["rng_after_json"])) == rng.bit_generator.state
        assert np.max(abs(data["p"].sum(axis=1))) < 1e-12
    assert calls == [True] and repeated == result
    assert sha256(source / "manifest.json") == original_hash


def test_both_time_steps_use_same_canonical_state_and_identical_record_times(small_source, tmp_path):
    initial = tmp_path / "initial"
    control.prepare_initial(initial, small_source)
    branches = (control.Branch("coarse", .1, .8, .1, .4), control.Branch("fine", .05, .8, .1, .4))
    states = []
    for branch in branches:
        directory = tmp_path / branch.label
        manifest = control.run_branch(directory, initial, small_source, branch)
        assert manifest["complete"] and manifest["additional_momentum_draws"] == 0 and manifest["thermostat"] == "none"
        np.testing.assert_allclose(observed_steps(directory)*branch.dt_fs, np.arange(9)*.1, rtol=0, atol=1e-15)
        states.append(read_checkpoint(directory, 0)[1])
    for name in ("q", "p"):
        np.testing.assert_allclose(states[0][name], states[1][name], rtol=0, atol=2e-13)
    # Saving observations in float64 is common to both branches.
    assert states[0]["molecular_velocity_au"].dtype == np.float64


def test_off_grid_checkpoint_restart_preserves_state_and_all_recorded_times(small_source, tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    control.prepare_initial(initial, small_source)
    branch = control.Branch("fine", .05, .8, .1, .4)
    full, split = tmp_path / "full", tmp_path / "split"
    control.run_branch(full, initial, small_source, branch)
    partial = control.run_branch(split, initial, small_source, branch, max_steps=3)
    assert partial["committed_steps"] == 3
    monkeypatch.setattr(native, "sample_momenta", lambda *a, **kw: pytest.fail("an NVE restart drew new momenta"))
    resumed = control.run_branch(split, initial, small_source, branch)
    _, expected = read_checkpoint(full)
    _, actual = read_checkpoint(split)
    np.testing.assert_allclose(actual["q"], expected["q"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual["p"], expected["p"], rtol=0, atol=2e-11)
    np.testing.assert_array_equal(observed_steps(full), observed_steps(split))
    assert resumed["restart_validations"][-1]["raw_positions_and_half_step_velocities_exact"]
    monkeypatch.setattr(native, "create_engine", lambda *a, **kw: pytest.fail("completed control recreated an engine"))
    assert control.run_branch(split, initial, small_source, branch)["complete"]


def test_actual_CPU_restore_uses_independent_cache_bound(tmp_path):
    pytest.importorskip("openmm")
    directory = tmp_path / "cpu-source"
    config = native.Config(cells_per_axis=2, cutoff_angstrom=2.5, dt_fs=.1,
                           equilibration_steps=8, equilibration_record_stride=2,
                           segment_steps=8, segments=1, record_stride=2, checkpoint_steps=4,
                           sampling_refresh_fs=.4, seed=3101, platform="CPU",
                           trajectory_precision="float64")
    manifest = native.run(directory, config)
    selected = next(row for row in manifest["chunks"] if row["step"] == 8)
    row = {"directory": str(directory), "manifest_sha256": sha256(directory / "manifest.json"),
           "checkpoint": selected, "config": asdict(config), "artifacts": manifest["artifacts"],
           "box_angstrom": manifest["box_angstrom"]}
    initial, output = tmp_path / "initial", tmp_path / "control"
    prepared = control.prepare_initial(initial, row)
    branch = control.Branch("fine", .05, .8, .1, .4)
    control.run_branch(output, initial, row, branch, max_steps=3)
    finished = control.run_branch(output, initial, row, branch)
    assert prepared["source_restore_validation"]["raw_positions_and_half_step_velocities_exact"]
    checked = finished["restart_validations"][-1]
    assert checked["raw_positions_and_half_step_velocities_exact"] is True
    assert checked["force_RMS_relative_difference"] <= 1e-6
    assert checked["canonical_p_conversion_residual_max_au"] <= 2e-13


def test_changed_shared_initial_or_branch_configuration_is_rejected_before_engine(small_source, tmp_path, monkeypatch):
    initial, output = tmp_path / "initial", tmp_path / "branch"
    control.prepare_initial(initial, small_source)
    branch = control.Branch("coarse", .1, .8, .1, .4)
    control.run_branch(output, initial, small_source, branch, max_steps=3)
    monkeypatch.setattr(native, "create_engine", lambda *a, **kw: pytest.fail("changed control created an engine"))
    with pytest.raises(ValueError, match="identity changed"):
        control.run_branch(output, initial, small_source, control.Branch("coarse", .05, .8, .1, .4))
    (initial / "initial.npz").write_bytes(b"changed shared physical state")
    with pytest.raises(ValueError, match="shared one-draw initial state"):
        control.run_branch(output, initial, small_source, branch)


def test_paired_interval_uses_eight_chains_and_sample_variance():
    coarse = np.full(8, .2)
    fine = coarse + np.arange(1, 9)*.001
    result = control.paired_statistics(coarse, fine)
    assert result["mean"] == pytest.approx(.0045)
    assert result["standard_error"] == pytest.approx(np.sqrt(6e-6/8))
    assert result["CI95_halfwidth"] == pytest.approx(2.3646242510102993*np.sqrt(6e-6/8))
    assert result["degrees_of_freedom"] == 7 and result["equivalent_within_margin"] is True


def test_wide_interval_containing_zero_is_not_equivalence():
    coarse = np.full(8, .2)
    result = control.paired_statistics(coarse, coarse+np.array([-.1]*4+[.1]*4))
    assert result["CI95"][0] < 0 < result["CI95"][1]
    assert result["equivalent_within_margin"] is False
    assert result["status"] == "equivalence_unresolved"
    assert control.paired_statistics(coarse, coarse+.02)["status"] == "difference_exceeds_margin"


@pytest.mark.parametrize("fine", [np.ones(7), np.full(8, np.nan), np.ones((8, 1))])
def test_incomplete_or_nonfinite_pairs_cannot_produce_statistics(fine):
    with pytest.raises(ValueError, match="all eight"):
        control.paired_statistics(np.ones(8), fine)


@pytest.fixture
def registered_control(cohort, tmp_path, monkeypatch):
    monkeypatch.setattr(control, "source_identity", lambda: {**cohort.runner_identity(), "fixture-control": "c"*64})
    protocol = tmp_path / "control-protocol.json"
    write_json(protocol, {"schema": "pyrpmd.water-eight-chain-paired-timestep-protocol.v1",
                          "fixed_seeds": list(control.SEEDS), "original_protocol_sha256": "b"*64,
                          "physics": {"original_frozen_source_sha256": cohort.native_identity()}})
    monkeypatch.setattr(control, "PROTOCOL_SHA256", sha256(protocol))
    assessment = json.loads(cohort.assessment.read_text())
    assessment["protocol_sha256"] = "b"*64
    write_json(cohort.assessment, assessment)
    update_assessment(cohort)
    output = tmp_path / "dt-control"
    registration = control.register(cohort.original, output, protocol, cohort.assessment, cohort.exit_proof)
    cohort.control_output = output
    cohort.control_registration = registration
    return cohort


def test_control_registration_requires_complete_originals_and_analysis_refuses_partial_pairs(registered_control, monkeypatch):
    identity = control.load_registration(registered_control.control_registration)
    assert list(identity["registration"]["selection"]) == [str(seed) for seed in control.SEEDS]
    assert identity["registration"]["branches"] == [asdict(branch) for branch in control.BRANCHES]
    monkeypatch.setattr(control, "analyze_diffusion", lambda *a, **kw: pytest.fail("partial cohort computed a D"))
    with pytest.raises(ValueError, match="all eight complete pairs"):
        control.analyze(registered_control.control_registration, registered_control.control_output / "analysis")
    assert not (registered_control.control_output / "analysis").exists()


def test_control_failure_retains_data_and_stops_before_any_later_seed(registered_control, monkeypatch):
    calls = []
    def prepare(path, row):
        path.mkdir(parents=True)
        (path / "retained-fixture-state").write_text("fixture")
        calls.append(row["config"]["seed"])
        return {}
    def fail(*args, **kwargs):
        raise RuntimeError("Injected control engine failure")
    monkeypatch.setattr(control, "prepare_initial", prepare)
    monkeypatch.setattr(control, "run_branch", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        control.run_control(registered_control.control_registration, wall_seconds=10)
    assert calls == [1101]
    output = registered_control.control_output
    assert (output / "runs/seed-1101/initial/retained-fixture-state").exists()
    failure = json.loads((output / "FAILURE.json").read_text())
    assert failure["seed"] == 1101 and failure["branch"] == "dt025"
    with pytest.raises(RuntimeError, match="retained control failure"):
        control.run_control(registered_control.control_registration, wall_seconds=10)
    assert calls == [1101]


def test_copying_one_seed_identity_cannot_create_a_second_independent_pair(registered_control, monkeypatch):
    identity = control.load_registration(registered_control.control_registration)
    output = registered_control.control_output
    for seed in control.SEEDS:
        row = identity["registration"]["selection"][str(seed)]
        initial_directory = output / "runs" / f"seed-{seed}" / "initial"
        initial_directory.mkdir(parents=True)
        atomic_npz(initial_directory / "initial.npz", excluded_metadata_fixture=np.asarray(seed))
        initial_sha = sha256(initial_directory / "initial.npz")
        write_json(initial_directory / "initial.json", {
            "source_sha256": control.source_identity(), "seed": 1101 if seed == 1102 else seed,
            "original_manifest_sha256": row["manifest_sha256"],
            "original_checkpoint_sha256": row["checkpoint"]["sha256"], "initial_sha256": initial_sha,
            "pending_COM_Maxwell_draw_count": 1, "both_branches_use_this_same_state": True,
        })
        for branch in control.BRANCHES:
            directory = output / "runs" / f"seed-{seed}" / branch.label
            directory.mkdir()
            artifacts = {}
            for name in ("model.xml", "integrator.xml"):
                (directory / name).write_bytes((Path(row["directory"]) / name).read_bytes())
                artifacts[name] = {"sha256": sha256(directory / name)}
            write_json(directory / "manifest.json", {
                "schema": control.BRANCH_SCHEMA, "complete": True, "committed_steps": branch.steps,
                "branch": asdict(branch), "source_sha256": control.source_identity(),
                "environment": native.environment_identity(),
                "native_config": asdict(control._branch_config(native.Config(**row["config"]), branch)),
                "initial_sha256": initial_sha, "thermostat": "none", "additional_momentum_draws": 0,
                "chunks": [], "artifacts": artifacts,
            })
    monkeypatch.setattr(control, "analyze_diffusion", lambda *a, **kw: pytest.fail("duplicated seed reached transport analysis"))
    with pytest.raises(ValueError, match="exact original seed identity"):
        control.analyze(registered_control.control_registration, output / "analysis")
    assert not (output / "analysis").exists()

