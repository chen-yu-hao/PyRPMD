"""Portable metadata/dispatch fixtures; no OpenMM Context or MD is run."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fcntl", reason="the declared execution uses POSIX advisory leases")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.water import native_preparation_coordinator as schedule
from benchmarks.water.campaign import atomic_npz, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def phase_arrays(step, *, phase=-1):
    return dict(state_step=step, active_segment=phase, box_angstrom=18.642822871774577,
                q=np.zeros((1, 648, 3)), p=np.zeros((1, 648, 3)),
                engine_positions_nm=np.zeros((864, 3)), engine_half_step_velocities_nm_ps=np.zeros((864, 3)),
                engine_forces_kj_mol_nm=np.zeros((864, 3)), rng_json="{}",
                openmm_checkpoint=np.frombuffer(b"Metadata fixture, never loaded into OpenMM", dtype=np.uint8),
                next_refresh_step=step)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    sources = tmp_path / "fixture-sources"
    sources.mkdir()
    native_file = sources / "native.txt"
    native_file.write_text("Fixture source identity; no physical integration code")
    runner_file = sources / "runner.txt"
    runner_file.write_text("Fixture continuation identity; never executed")
    coordinator_file = sources / "coordinator.txt"
    coordinator_file.write_text("Fixture scheduling identity")
    native_identity = lambda: {"fixture-native": sha256(native_file)}
    runner_identity = lambda: {**native_identity(), "fixture-runner": sha256(runner_file)}
    coordinator_identity = lambda: {**runner_identity(), "fixture-coordinator": sha256(coordinator_file)}
    environment = lambda: {"fixture": "metadata only, no engine dependency"}
    monkeypatch.setattr(schedule.native, "source_identity", native_identity)
    monkeypatch.setattr(schedule.native, "environment_identity", environment)
    monkeypatch.setattr(schedule.extension, "source_identity", runner_identity)
    monkeypatch.setattr(schedule, "coordinator_identity", coordinator_identity)
    monkeypatch.setattr(schedule.native, "create_engine", lambda *a, **kw: pytest.fail("orchestration must not create an engine"))
    original = tmp_path / "original"
    control = original / "four-worker-control"
    control.mkdir(parents=True)
    for name in ("coordinator.lock", *(f"seed-{seed}.lock" for seed in schedule.SEEDS)):
        (control / name).write_text("")
    for name in ("scheduler-odd.lock", "scheduler-even.lock"):
        (original / name).write_text("")
    write_json(original / "four-worker-registration.json", {"fixture": "original registered cohort"})
    finished = {"event": "coordinator_finished", "run_id": "fixture-original-coordinator", "failed": False}
    (control / "events.jsonl").write_text(json.dumps({"event": "coordinator_started", "run_id": finished["run_id"]}) + "\n" + json.dumps(finished) + "\n")
    exit_proof = tmp_path / "original-normal-exit.json"
    write_json(exit_proof, {"schema": "pyrpmd.water-original-coordinator-normal-exit.v1",
                            "process_exit_code": 0, "pool_drained": True,
                            "scheduler_source_sha256": schedule.HELPERS_SHA256,
                            "original_campaign_registration_sha256": sha256(original / "four-worker-registration.json"),
                            "journal_sha256": sha256(control / "events.jsonl"), "coordinator_finished_event": finished})
    rows = []
    for seed in schedule.SEEDS:
        directory = original / "runs" / f"classical-{seed}"
        directory.mkdir(parents=True)
        artifacts = {}
        for name in ("model.xml", "integrator.xml"):
            (directory / name).write_text("Common model fixture; never deserialized")
            artifacts[name] = {"sha256": sha256(directory / name)}
        # Real first two runs retain distinct historical migration receipts.
        # Those are retained, but are not common engine XML.
        if seed in (1101, 1102):
            name = "manifest.before-checkpoint-contract-v2.json"
            write_json(directory / name, {"historical_seed": seed})
            artifacts[name] = {"sha256": sha256(directory / name)}
        chunks = []
        for step, phase in ((schedule.ORIGIN, -1), (800000, 24)):
            path = directory / f"chunk-{step}.npz"
            atomic_npz(path, **phase_arrays(step, phase=phase))
            chunks.append({"file": path.name, "sha256": sha256(path), "step": step, "observations": 0})
        config = schedule.native.Config(seed=seed, initial_checkpoint="fixture-initial-state.npz")
        manifest = {"schema": "pyrpmd.water-openmm-external-reference.v1", "external_reference": True,
                    "complete": True, "checkpoint_contract_version": 2, "committed_steps": 800000,
                    "config": asdict(config), "source_sha256": native_identity(), "environment": environment(),
                    "chunks": chunks, "artifacts": artifacts, "box_angstrom": 18.642822871774577,
                    "initial_checkpoint_sha256": "a" * 64}
        write_json(directory / "manifest.json", manifest)
        rows.append({"seed": seed, "complete": True, "manifest_sha256": sha256(directory / "manifest.json")})
    assessment = tmp_path / "original-assessment.json"
    write_json(assessment, {"schema": "pyrpmd.water-native-chain-analysis.v1",
                            "all_predeclared_chains_complete": True, "chains": rows,
                            "diffusion_value_deliberately_unused": "No D is present in this fixture"})
    proposal = tmp_path / "proposal.json"
    write_json(proposal, {"schema": "pyrpmd.water-equilibration-extension-proposal.v2",
                          "fixed_seeds": list(schedule.SEEDS),
                          "fixed_extension": {"additional_preparation_ps": 400., "additional_steps_per_seed": schedule.EXTRA,
                                              "production_automatically_started": False},
                          "stationarity": {"each_seed_fixed_half_mean_absolute_thresholds": schedule.SCALAR_THRESHOLDS},
                          "structural_diagnostics": {"RDF_simultaneous_uncertainty_gate": None}})
    monkeypatch.setattr(schedule, "PROPOSAL_SHA256", sha256(proposal))
    adoption = tmp_path / "adoption.json"
    write_json(adoption, {"schema": schedule.ADOPTION_SCHEMA,
                          "decision": "adopt_fixed_additional_preparation_only", "fixed_seeds": list(schedule.SEEDS),
                          "proposal_sha256": sha256(proposal), "additional_preparation_ps": 400.,
                          "production_authorized": False, "RDF_and_coordination_hard_gates_adopted": False,
                          "original_scalar_thresholds_changed": False, "diffusion_used_to_select_budget_or_chains": False,
                          "reviewed_at_utc": "2026-11-01T00:00:00+00:00",
                          "review_record": "Excluded fixture; not a live scientific adoption",
                          "original_coordinator_exit_proof": {"path": str(exit_proof), "sha256": sha256(exit_proof)},
                          "original_cohort_assessment": {"path": str(assessment), "sha256": sha256(assessment)}})
    return SimpleNamespace(original=original, output=tmp_path / "continuation", proposal=proposal,
                           adoption=adoption, assessment=assessment, source=native_file,
                           native_identity=native_identity, runner_identity=runner_identity,
                           environment=environment, exit_proof=exit_proof)


@pytest.fixture
def registered(cohort):
    cohort.registration = schedule.register(cohort.original, cohort.output, cohort.proposal, cohort.adoption)
    return cohort


def update_assessment(fixture):
    assessment = schedule.read_json(fixture.assessment)
    for row in assessment["chains"]:
        directory = fixture.original / "runs" / f"classical-{row['seed']}"
        row["manifest_sha256"] = sha256(directory / "manifest.json")
    write_json(fixture.assessment, assessment)
    adoption = schedule.read_json(fixture.adoption)
    adoption["original_cohort_assessment"]["sha256"] = sha256(fixture.assessment)
    write_json(fixture.adoption, adoption)


def make_branch(fixture, seed, *, step=schedule.TARGET, prefix_replacement=False):
    directory = fixture.output / "runs" / f"preparation-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    original = schedule.read_json(fixture.original / "runs" / f"classical-{seed}" / "manifest.json")
    config = schedule.read_json(fixture.output / "configs" / f"seed-{seed}.json")
    artifacts = {name: original["artifacts"][name] for name in ("model.xml", "integrator.xml")}
    for name in artifacts:
        shutil.copyfile(fixture.original / "runs" / f"classical-{seed}" / name, directory / name)
    chunks = []
    for index, value in enumerate([schedule.ORIGIN, step]):
        path = directory / f"chunk-{index:06d}.npz"
        arrays = phase_arrays(value)
        if prefix_replacement and index == 0:
            arrays["q"] += 1
        atomic_npz(path, **arrays)
        chunks.append({"file": path.name, "sha256": sha256(path), "step": value, "observations": 0})
    manifest = {"schema": schedule.extension.SCHEMA, "config": config, "base_config": original["config"],
                "source_sha256": fixture.runner_identity(), "environment": fixture.environment(),
                "preparation_only": True, "production_started": False, "source_preparation_step": schedule.ORIGIN,
                "target_preparation_step": schedule.TARGET, "committed_steps": step, "complete": step == schedule.TARGET,
                "refresh_count": (step - schedule.ORIGIN + 399) // 400, "next_refresh_step": step,
                "chunks": chunks, "artifacts": artifacts}
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def seed_from_command(command):
    return int(Path(command[command.index("--output") + 1]).name.split("-")[-1])


def test_register_captures_all_eight_exact_checkpoints_without_creating_engine(cohort):
    before = {seed: sha256(cohort.original / "runs" / f"classical-{seed}" / "manifest.json") for seed in schedule.SEEDS}
    path = schedule.register(cohort.original, cohort.output, cohort.proposal, cohort.adoption)
    identity = schedule.load_registration(path)
    assert sorted(identity["configs"]) == list(schedule.SEEDS)
    assert all(config.additional_preparation_steps == 1600000 for _, config in identity["configs"].values())
    assert not (cohort.output / "runs").exists()
    for seed, expected in before.items():
        assert sha256(cohort.original / "runs" / f"classical-{seed}" / "manifest.json") == expected
    selection = identity["selection"]
    assert selection["1101"]["artifacts"] != selection["1102"]["artifacts"]
    assert selection["1101"]["engine_artifacts"] == selection["1108"]["engine_artifacts"]


@pytest.mark.parametrize("kind", ["missing_seed", "incomplete_seed", "wrong_phase", "changed_dt", "wrong_assessment", "production_adoption"])
def test_incomplete_or_changed_cohort_cannot_publish_registration(cohort, kind):
    directory = cohort.original / "runs" / "classical-1108"
    if kind == "missing_seed":
        shutil.rmtree(directory)
    elif kind in ("incomplete_seed", "changed_dt", "wrong_phase"):
        manifest = schedule.read_json(directory / "manifest.json")
        if kind == "incomplete_seed":
            manifest["complete"] = False
        elif kind == "changed_dt":
            manifest["config"]["dt_fs"] = .125
        else:
            record = manifest["chunks"][0]
            atomic_npz(directory / record["file"], **phase_arrays(schedule.ORIGIN, phase=0))
            record["sha256"] = sha256(directory / record["file"])
        write_json(directory / "manifest.json", manifest)
        update_assessment(cohort)
    elif kind == "wrong_assessment":
        assessment = schedule.read_json(cohort.assessment)
        assessment["chains"].pop()
        write_json(cohort.assessment, assessment)
        update_assessment(cohort)
    else:
        adoption = schedule.read_json(cohort.adoption)
        adoption["production_authorized"] = True
        write_json(cohort.adoption, adoption)
    with pytest.raises((ValueError, FileNotFoundError)):
        schedule.register(cohort.original, cohort.output, cohort.proposal, cohort.adoption)
    assert not cohort.output.exists()


@pytest.mark.parametrize("change", ["source", "selected_checkpoint", "model", "config", "proposal"])
def test_changed_registered_input_prevents_any_launch(registered, monkeypatch, change):
    if change == "source":
        registered.source.write_text("Changed frozen fixture source")
    elif change == "selected_checkpoint":
        (registered.original / "runs/classical-1108/chunk-400000.npz").write_bytes(b"changed")
    elif change == "model":
        (registered.original / "runs/classical-1108/model.xml").write_text("changed model")
    elif change == "proposal":
        registered.proposal.write_text("{}")
    else:
        path = registered.output / "configs/seed-1108.json"
        data = schedule.read_json(path);data["additional_preparation_steps"] -= 4000;write_json(path, data)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("changed input launched a child"))
    with pytest.raises(ValueError):
        schedule.run_campaign(registered.registration, wall_seconds=10)


@pytest.mark.parametrize("lease", ["coordinator.lock", "seed-1108.lock"])
def test_one_writer_leases_prevent_every_launch(registered, monkeypatch, lease):
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("held lease launched a child"))
    with schedule.exclusive_lease(registered.output / "control" / lease):
        with pytest.raises(RuntimeError, match="already held"):
            schedule.run_campaign(registered.registration, wall_seconds=10)


def test_all_fixed_seeds_run_once_with_inherited_leases_and_no_nve_command(registered, monkeypatch):
    calls = []
    class Child:
        def __init__(self, command, **kwargs):
            self.seed = seed_from_command(command);self.pid = self.seed;self.done = False
            assert command[1].endswith("native_preparation_extension.py")
            assert len(set(kwargs["pass_fds"])) == 6
            assert "--config" in command and "--resume" not in command
            assert all(kwargs["env"][key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"))
            calls.append(self.seed)
        def wait(self):
            if not self.done:
                make_branch(registered, self.seed);self.done = True
            return 0
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    schedule.run_campaign(registered.registration, wall_seconds=20, workers=4)
    assert sorted(calls) == list(schedule.SEEDS)
    for first, second in schedule.QUEUES.values():
        assert calls.index(first) < calls.index(second)
    for seed in schedule.SEEDS:
        receipt = schedule.read_json(registered.output / "control/prefixes" / f"seed-{seed}.json")
        assert receipt["complete"] and receipt["committed_steps"] == 2000000
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("completed cohort reran"))
    schedule.run_campaign(registered.registration, wall_seconds=10)


def test_partial_branch_resumes_without_config_or_new_seed(registered, monkeypatch):
    for seed in schedule.SEEDS:
        make_branch(registered, seed, step=800000 if seed == 1101 else schedule.TARGET)
    calls = []
    class Child:
        def __init__(self, command, **kwargs):
            assert "--resume" in command and "--config" not in command
            self.seed = seed_from_command(command);self.pid = self.seed;self.done = False;calls.append(self.seed)
        def wait(self):
            if not self.done:
                directory = registered.output / "runs" / f"preparation-{self.seed}"
                manifest = schedule.read_json(directory / "manifest.json")
                path = directory / "chunk-000002.npz"
                atomic_npz(path, **phase_arrays(schedule.TARGET))
                manifest["chunks"].append({"file": path.name, "sha256": sha256(path), "step": schedule.TARGET, "observations": 0})
                manifest.update(committed_steps=schedule.TARGET, complete=True, refresh_count=4000, next_refresh_step=schedule.TARGET)
                write_json(directory / "manifest.json", manifest);self.done = True
            return 0
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    schedule.run_campaign(registered.registration, wall_seconds=20)
    assert calls == [1101]


@pytest.mark.parametrize("corrupt", ["replace_prefix", "remove_branch", "NVE_phase"])
def test_old_prefix_or_preparation_phase_failure_is_retained_before_dispatch(registered, monkeypatch, corrupt):
    directory, manifest = make_branch(registered, 1101, step=800000)
    identity = schedule.load_registration(registered.registration)
    state, chunks = schedule.branch_prefix(identity, 1101)
    write_json(registered.output / "control/prefixes/seed-1101.json",
               {"registration_sha256": identity["sha256"], "seed": 1101, **state, "chunks": chunks})
    if corrupt == "replace_prefix":
        make_branch(registered, 1101, step=800000, prefix_replacement=True)
    elif corrupt == "remove_branch":
        shutil.rmtree(directory)
    else:
        manifest["production_started"] = True;write_json(directory / "manifest.json", manifest)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("corrupt branch launched a child"))
    with pytest.raises(RuntimeError, match="preflight failed"):
        schedule.run_campaign(registered.registration, wall_seconds=10)
    assert (registered.output / "control/FAILURE.json").exists()


def test_stop_marker_performs_no_propagation(registered, monkeypatch):
    control = registered.output / "control";control.mkdir()
    (control / "STOP_AFTER_CURRENT").write_text("Fixture scheduling pause")
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("stop marker ignored"))
    schedule.run_campaign(registered.registration, wall_seconds=10)
    assert not (registered.output / "runs").exists()


def test_post_spawn_journal_failure_latches_and_drains_before_return(registered, monkeypatch):
    waited, spawned = [], []
    original = schedule.Journal.__call__
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = seed_from_command(command);spawned.append(self.pid)
        def wait(self):
            assert (registered.output / "control/FAILURE.json").exists();waited.append(self.pid);return 0
        def terminate(self):
            pytest.fail("native children must not be terminated")
        kill = terminate
    def journal(self, event, **values):
        if event == "preparation_invocation_started":
            raise OSError("Injected journal failure after successful spawn")
        return original(self, event, **values)
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    monkeypatch.setattr(schedule.Journal, "__call__", journal)
    with pytest.raises(RuntimeError, match="stopped after an error"):
        schedule.run_campaign(registered.registration, wall_seconds=10)
    assert len(spawned) == 1 and waited == spawned


def test_one_failed_child_drains_other_inflight_children_and_blocks_queue_successors(registered, monkeypatch):
    all_started = threading.Barrier(4)
    failure_visible = threading.Event()
    spawned, waited = [], []
    journal_original = schedule.Journal.__call__
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = seed_from_command(command);self.done = False;spawned.append(self.pid)
        def wait(self):
            if not self.done:
                all_started.wait(timeout=10)
                if self.pid != 1101:
                    assert failure_visible.wait(timeout=10)
                    make_branch(registered, self.pid)
                waited.append(self.pid);self.done = True
            return 17 if self.pid == 1101 else 0
        def terminate(self):
            pytest.fail("in-flight children must drain normally")
        kill = terminate
    def journal(self, event, **values):
        result = journal_original(self, event, **values)
        if event == "preparation_stopped_on_error":
            failure_visible.set()
        return result
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    monkeypatch.setattr(schedule.Journal, "__call__", journal)
    with pytest.raises(RuntimeError, match="stopped after an error"):
        schedule.run_campaign(registered.registration, wall_seconds=20)
    assert sorted(spawned) == sorted(waited) == [1101, 1102, 1103, 1104]
    assert schedule.read_json(registered.output / "control/FAILURE.json")["seed"] == 1101
    with pytest.raises(RuntimeError, match="retained preparation failure"):
        schedule.run_campaign(registered.registration, wall_seconds=20)
    assert len(spawned) == 4


@pytest.mark.parametrize("lease", ["four-worker-control/coordinator.lock", "scheduler-odd.lock",
                                    "scheduler-even.lock", "four-worker-control/seed-1108.lock"])
def test_original_pool_must_release_every_legacy_and_seed_lease_before_registration(cohort, lease):
    with schedule.exclusive_lease(cohort.original / lease):
        with pytest.raises(RuntimeError, match="already held"):
            schedule.register(cohort.original, cohort.output, cohort.proposal, cohort.adoption)
    assert not cohort.output.exists()


@pytest.mark.parametrize("kind", ["nonzero_exit", "pool_not_drained", "later_coordinator_started"])
def test_original_exit_proof_must_be_normal_and_the_final_journal_event(cohort, kind):
    proof = schedule.read_json(cohort.exit_proof)
    if kind == "nonzero_exit":
        proof["process_exit_code"] = 17
    elif kind == "pool_not_drained":
        proof["pool_drained"] = False
    else:
        journal = cohort.original / "four-worker-control/events.jsonl"
        with journal.open("a") as stream:
            stream.write(json.dumps({"event": "coordinator_started", "run_id": "another-invocation"}) + "\n")
        proof["journal_sha256"] = sha256(journal)
    write_json(cohort.exit_proof, proof)
    adoption = schedule.read_json(cohort.adoption)
    adoption["original_coordinator_exit_proof"]["sha256"] = sha256(cohort.exit_proof)
    write_json(cohort.adoption, adoption)
    with pytest.raises(ValueError, match="normal exit"):
        schedule.register(cohort.original, cohort.output, cohort.proposal, cohort.adoption)
    assert not cohort.output.exists()


def test_diffusion_or_pass_flags_never_select_registration_budget_or_seeds(cohort):
    report = schedule.read_json(cohort.assessment)
    report["diffusion_VACF_angstrom2_ps"] = {"mean": 12345.0}
    report["numeric_predeclared_checks_passed"] = False
    write_json(cohort.assessment, report)
    update_assessment(cohort)
    path = schedule.register(cohort.original, cohort.output, cohort.proposal, cohort.adoption)
    identity = schedule.load_registration(path)
    assert sorted(identity["configs"]) == list(schedule.SEEDS)
    assert all(config.additional_preparation_steps == schedule.EXTRA for _, config in identity["configs"].values())


def test_validation_failure_is_latched_while_launch_lock_is_held(registered, monkeypatch):
    def fail(identity):
        raise ValueError("Injected changed registration before spawn")
    monkeypatch.setattr(schedule, "verify_runtime", fail)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("validation failure did not stop all queues"))
    with pytest.raises(RuntimeError, match="stopped after an error"):
        schedule.run_campaign(registered.registration, wall_seconds=10)
    assert (registered.output / "control/FAILURE.json").exists()


def test_native_prefix_replacement_stops_without_changing_any_other_seed(registered, monkeypatch):
    make_branch(registered, 1101, step=800000)
    calls = []
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = seed_from_command(command);calls.append(self.pid);self.done = False
        def wait(self):
            if not self.done:
                make_branch(registered, self.pid, prefix_replacement=True);self.done = True
            return 0
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="stopped after an error"):
        schedule.run_campaign(registered.registration, workers=1, wall_seconds=10)
    assert calls == [1101]
    assert (registered.output / "runs/preparation-1101/manifest.json").exists()
    assert not (registered.output / "runs/preparation-1105").exists()


@pytest.mark.parametrize("signal_number,exceptional_shutdown", [(2, False), (15, False), (15, True)])
def test_real_coordinator_signal_stops_admission_and_drains_real_stub_child(registered, monkeypatch, signal_number, exceptional_shutdown):
    import multiprocessing
    import os
    import signal
    import subprocess
    import time

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("the isolated POSIX signal fixture requires fork")
    marker = registered.output / "stub-child-started.json"
    release = registered.output / "release-stub-child"
    result_path = registered.output / "isolated-signal-result.json"
    real_popen = subprocess.Popen
    script = (
        "import json,os,sys,time\n"
        "from pathlib import Path\n"
        "marker,release=map(Path,sys.argv[1:])\n"
        "marker.write_text(json.dumps({'pid':os.getpid()}))\n"
        "deadline=time.monotonic()+5\n"
        "while not release.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "sys.exit(0 if release.exists() else 23)\n"
    )

    def isolated_coordinator():
        calls = []
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        class Child:
            def __init__(self, command, **kwargs):
                self.seed = seed_from_command(command)
                calls.append(self.seed)
                self.process = real_popen([sys.executable, "-c", script, str(marker), str(release)], **kwargs)
                self.pid = self.process.pid
                self.done = False
            def wait(self):
                code = self.process.wait()
                if code == 0 and not self.done:
                    make_branch(registered, self.seed)
                    self.done = True
                return code
        schedule.subprocess.Popen = Child
        if exceptional_shutdown:
            def fail_main_wait(*args, **kwargs):
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                raise RuntimeError("Injected main-thread error before exceptional pool drain")
            schedule.wait = fail_main_wait
        try:
            outcome = schedule.run_campaign(registered.registration, workers=1, wall_seconds=10)
            result = {"outcome": outcome, "calls": calls, "raised": None}
        except BaseException as error:
            result = {"outcome": None, "calls": calls, "raised": type(error).__name__}
        result["original_handlers_restored"] = all(signal.getsignal(number) == handler for number, handler in previous.items())
        write_json(result_path, result)

    process = multiprocessing.get_context("fork").Process(target=isolated_coordinator)
    process.start()
    try:
        deadline = time.monotonic() + 4
        wait_for = registered.output / ("control/FAILURE.json" if exceptional_shutdown else marker.name)
        while not wait_for.exists() and process.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists() and wait_for.exists()
        child_pid = schedule.read_json(marker)["pid"]
        os.kill(process.pid, signal_number)  # Only this isolated fixture coordinator, never the process group.
        if not exceptional_shutdown:
            stop_path = registered.output / "control/STOP_AFTER_CURRENT"
            while not stop_path.exists() and process.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            assert stop_path.exists()
            assert schedule.read_json(stop_path)["signal"] == signal_number
        time.sleep(.05)
        assert process.is_alive()
        os.kill(child_pid, 0)  # The real stub child survives the coordinator-only signal.
        for lease in (registered.original / "four-worker-control/coordinator.lock",
                      registered.output / "control/seed-1101.lock"):
            with pytest.raises(RuntimeError, match="already held"):
                with schedule.exclusive_lease(lease):
                    pass
    finally:
        release.write_text("Normal exit permitted for this non-physical fixture")
        process.join(timeout=10)
    assert not process.is_alive() and process.exitcode == 0
    result = schedule.read_json(result_path)
    assert result["calls"] == [1101] and result["original_handlers_restored"] is True
    if exceptional_shutdown:
        assert result["raised"] == "RuntimeError"
    else:
        assert result["raised"] is None
        assert result["outcome"] == {"stop_signal": signal_number, "exit_code": 128 + signal_number}
    assert schedule.read_json(registered.output / "runs/preparation-1101/manifest.json")["complete"] is True
    assert not (registered.output / "runs/preparation-1105").exists()



