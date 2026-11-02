#!/usr/bin/env python3
"""Registered, preparation-only continuation of the complete fixed water cohort.

Registration reads all eight completed original runs before publishing anything.
It binds their exact pre-NVE 100 ps checkpoints, the reviewed proposal, a separate
preparation-only adoption record, and the frozen runner. Execution adds400ps to
every chain, with inherited one-writer leases and durable failure-stop records.
No diffusion-based branch, production phase, seed substitution, or NVE launch
exists here. The implementation can be tested with metadata fixtures; no live
adoption or long continuation is implied by creating this source file.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water import native_preparation_extension as extension
from benchmarks.water import native_reference as native
from benchmarks.water.campaign import atomic_json, sha256
from benchmarks.water.native_campaign_schedule_v2 import Journal, exclusive_lease

SCHEMA = "pyrpmd.water-preparation-cohort-registration.v1"
ADOPTION_SCHEMA = "pyrpmd.water-preparation-only-adoption.v1"
RUNNER = "benchmarks/water/native_preparation_extension.py"
HELPERS = "benchmarks/water/native_campaign_schedule_v2.py"
LEGACY_HELPERS = "benchmarks/water/native_campaign_schedule.py"
RUNNER_SHA256 = "7442d986efc6298f33cbe2b3e7c27eef6a27d3a4fa9861bc818bf70c763e8a4c"
HELPERS_SHA256 = "af1258f9118c4de296c90edb18c57aceb07a007d564652b4f5db6e851cece75e"
LEGACY_HELPERS_SHA256 = "4c17aaf4a9a26a6367dfbce6864a367883990a8e24f3e0ff290301b6a53e5239"
PROPOSAL_SHA256 = "e17bd58c303e7f14f1023a458cd59a004c83185a6864f1587ec93d2787737b12"
SEEDS = tuple(range(1101, 1109))
QUEUES = {"a": [1101, 1105], "b": [1102, 1106], "c": [1103, 1107], "d": [1104, 1108]}
ORIGIN = 400000
EXTRA = 1600000
TARGET = ORIGIN + EXTRA
SCALAR_THRESHOLDS = {"temperature_K": 5.0, "potential_kcal_mol_water": .05,
                     "OH_angstrom": .001, "HOH_degrees": .1}


def read_json(path):
    return json.loads(Path(path).read_text())


def coordinator_identity():
    if (sha256(ROOT / RUNNER) != RUNNER_SHA256 or sha256(ROOT / HELPERS) != HELPERS_SHA256
            or sha256(ROOT / LEGACY_HELPERS) != LEGACY_HELPERS_SHA256):
        raise ValueError("the reviewed continuation runner or inherited-lease helpers changed")
    return {**extension.source_identity(), HELPERS: HELPERS_SHA256, LEGACY_HELPERS: LEGACY_HELPERS_SHA256,
            "benchmarks/water/native_preparation_coordinator.py": sha256(Path(__file__))}


def validate_proposal(path):
    path = Path(path).resolve()
    proposal = read_json(path)
    if (sha256(path) != PROPOSAL_SHA256
            or proposal.get("schema") != "pyrpmd.water-equilibration-extension-proposal.v2"
            or proposal["fixed_seeds"] != list(SEEDS)
            or proposal["fixed_extension"]["additional_preparation_ps"] != 400.0
            or proposal["fixed_extension"]["additional_steps_per_seed"] != EXTRA
            or proposal["fixed_extension"]["production_automatically_started"] is not False
            or proposal["stationarity"]["each_seed_fixed_half_mean_absolute_thresholds"] != SCALAR_THRESHOLDS
            or proposal["structural_diagnostics"]["RDF_simultaneous_uncertainty_gate"] is not None):
        raise ValueError("the fixed v2 preparation-only proposal or its unchanged scalar criteria differ")
    return proposal


def validate_adoption(path, proposal_hash):
    """A later explicit scientific decision, never inferred from D or pass flags."""
    path = Path(path).resolve()
    adoption = read_json(path)
    if (adoption.get("schema") != ADOPTION_SCHEMA
            or adoption.get("decision") != "adopt_fixed_additional_preparation_only"
            or adoption.get("fixed_seeds") != list(SEEDS)
            or adoption.get("proposal_sha256") != proposal_hash
            or adoption.get("additional_preparation_ps") != 400.0
            or adoption.get("production_authorized") is not False
            or adoption.get("RDF_and_coordination_hard_gates_adopted") is not False
            or adoption.get("original_scalar_thresholds_changed") is not False
            or adoption.get("diffusion_used_to_select_budget_or_chains") is not False
            or not isinstance(adoption.get("reviewed_at_utc"), str) or not adoption["reviewed_at_utc"]
            or not isinstance(adoption.get("review_record"), str) or not adoption["review_record"]):
        raise ValueError("an explicit fixed all-eight preparation-only adoption record is required")
    assessment_path = Path(adoption["original_cohort_assessment"]["path"]).resolve()
    assessment = read_json(assessment_path)
    rows = assessment.get("chains", [])
    if (sha256(assessment_path) != adoption["original_cohort_assessment"]["sha256"]
            or assessment.get("schema") != "pyrpmd.water-native-chain-analysis.v1"
            or assessment.get("all_predeclared_chains_complete") is not True
            or len(rows) != len(SEEDS) or sorted(row["seed"] for row in rows) != list(SEEDS)
            or any(row.get("complete") is not True for row in rows)):
        raise ValueError("adoption requires the preserved complete original eight-chain assessment")
    # Only completion and source identities are used. A D value or eligibility
    # boolean never selects a branch, seed, duration or thermostat here.
    return adoption, {row["seed"]: row["manifest_sha256"] for row in rows}


def validate_original_exit(original_root, adoption):
    """Bind observed normal process exit and the final drained-pool event."""
    original_root = Path(original_root).resolve()
    reference = adoption["original_coordinator_exit_proof"]
    proof_path = Path(reference["path"]).resolve()
    proof = read_json(proof_path)
    journal_path = original_root / "four-worker-control" / "events.jsonl"
    journal_bytes = journal_path.read_bytes()
    rows = [json.loads(line) for line in journal_bytes.splitlines() if line.strip()]
    if (sha256(proof_path) != reference["sha256"]
            or proof.get("schema") != "pyrpmd.water-original-coordinator-normal-exit.v1"
            or type(proof.get("process_exit_code")) is not int or proof["process_exit_code"] != 0
            or proof.get("pool_drained") is not True
            or proof.get("scheduler_source_sha256") != HELPERS_SHA256
            or proof.get("original_campaign_registration_sha256") != sha256(original_root / "four-worker-registration.json")
            or proof.get("journal_sha256") != hashlib.sha256(journal_bytes).hexdigest()
            or not rows or rows[-1] != proof.get("coordinator_finished_event")
            or rows[-1].get("event") != "coordinator_finished" or rows[-1].get("failed") is not False
            or not isinstance(rows[-1].get("run_id"), str) or not rows[-1]["run_id"]
            or not any(row.get("event") == "coordinator_started" and row.get("run_id") == rows[-1]["run_id"] for row in rows)):
        raise ValueError("the original coordinator must have a bound normal exit and final drained-pool event")
    return proof


@contextmanager
def original_exclusion_leases(original_root):
    """Prove/reassert release of every old writer lease without changing data."""
    root = Path(original_root).resolve()
    paths = [root / "four-worker-control/coordinator.lock", root / "scheduler-odd.lock", root / "scheduler-even.lock"]
    paths += [root / "four-worker-control" / f"seed-{seed}.lock" for seed in SEEDS]
    if any(not path.is_file() for path in paths):
        raise ValueError("the original coordinator/legacy/seed lease records must already exist")
    with ExitStack() as stack:
        handles = [stack.enter_context(exclusive_lease(path)) for path in paths]
        yield {"global": handles[:3], "seeds": dict(zip(SEEDS, handles[3:]))}


def _checkpoint_phase(path, *, expected_step):
    with np.load(path, allow_pickle=False) as data:
        required = {"q", "p", "state_step", "active_segment", "box_angstrom", "rng_json",
                    "openmm_checkpoint", "engine_positions_nm", "engine_half_step_velocities_nm_ps",
                    "engine_forces_kj_mol_nm"}
        if not required.issubset(data.files) or int(data["state_step"]) != expected_step or int(data["active_segment"]) != -1:
            raise ValueError("the exact checkpoint must retain v2 raw state and the pre-NVE preparation phase")


def register(original_root, output, proposal_path, adoption_path):
    """Capture all eight sources, then atomically publish a registration last.

    No Context is created, no random number is consumed, and no dynamics run.
    This operation is not called on the live cohort by the review tests.
    """
    # Acquiring all eleven original leases proves the old pool released its
    # writers. Keep them until the new registration has been published.
    with original_exclusion_leases(original_root):
        return _register_locked(original_root, output, proposal_path, adoption_path)


def _register_locked(original_root, output, proposal_path, adoption_path):
    original_root, output = Path(original_root).resolve(), Path(output).resolve()
    proposal_path, adoption_path = Path(proposal_path).resolve(), Path(adoption_path).resolve()
    identity = coordinator_identity()
    proposal = validate_proposal(proposal_path)
    adoption, assessed_manifests = validate_adoption(adoption_path, sha256(proposal_path))
    validate_original_exit(original_root, adoption)
    adoption_hash = sha256(adoption_path)
    if output == original_root or original_root in output.parents or output in original_root.parents:
        raise ValueError("the registered continuation output must be separate from the original cohort")
    selection, configs = {}, {}
    for seed in SEEDS:
        directory = original_root / "runs" / f"classical-{seed}"
        config = extension.prepare_config(str(directory), additional_preparation_ps=400.0)
        source, original, base, chosen, _ = extension._load_source(config)
        expected = native.Config(seed=seed, initial_checkpoint=base.initial_checkpoint)
        if (asdict(base) != asdict(expected) or base.seed != seed or base.equilibration_steps != ORIGIN
                or config.additional_preparation_steps != EXTRA or config.checkpoint_steps != 4000
                or original.get("checkpoint_contract_version") != 2 or not base.initial_checkpoint
                or not isinstance(original.get("initial_checkpoint_sha256"), str)
                or len(original["initial_checkpoint_sha256"]) != 64
                or assessed_manifests[seed] != config.source_manifest_sha256):
            raise ValueError("original seed/model/duration/checkpoint differs from the fixed assessed cohort")
        _checkpoint_phase(source / chosen["file"], expected_step=ORIGIN)
        selection[str(seed)] = {"source_run": str(source), "manifest_sha256": config.source_manifest_sha256,
                                "checkpoint": dict(chosen), "base_config": asdict(base),
                                "source_sha256": original["source_sha256"], "environment": original["environment"],
                                "box_angstrom": original["box_angstrom"],
                                "artifacts": original["artifacts"],
                                "engine_artifacts": {name: original["artifacts"][name]
                                                     for name in ("model.xml", "integrator.xml")},
                                "initial_checkpoint_sha256": original.get("initial_checkpoint_sha256")}
        configs[str(seed)] = asdict(config)
    first = selection[str(SEEDS[0])]
    for row in selection.values():
        if any(row[key] != first[key] for key in ("source_sha256", "environment", "box_angstrom", "engine_artifacts", "initial_checkpoint_sha256")):
            raise ValueError("all eight sources must share the unchanged registered model/environment/initial geometry")
    if (coordinator_identity() != identity or sha256(adoption_path) != adoption_hash
            or sha256(proposal_path) != PROPOSAL_SHA256):
        raise RuntimeError("source/adoption/proposal changed during all-eight checkpoint selection")
    for row in selection.values():
        directory = Path(row["source_run"])
        if (sha256(directory / "manifest.json") != row["manifest_sha256"]
                or sha256(directory / row["checkpoint"]["file"]) != row["checkpoint"]["sha256"]):
            raise RuntimeError("an original checkpoint changed during all-eight selection")
    registration_lease = output.with_name(output.name + "-registration.lock")
    with exclusive_lease(registration_lease):
        if output.exists() and any(output.iterdir()):
            raise ValueError("registration output must be new/empty; never overwrite an adopted cohort")
        output.mkdir(parents=True, exist_ok=True)
        (output / "configs").mkdir()
        shutil.copyfile(adoption_path, output / "adoption.json")
        atomic_json(output / "source-selection.json", {"schema": "pyrpmd.water-all-eight-preparation-source-selection.v1",
                    "fixed_seeds": list(SEEDS), "source_step": ORIGIN, "pre_NVE": True,
                    "selected": selection, "created_utc": datetime.now(timezone.utc).isoformat()})
        config_sha256 = {}
        for seed in SEEDS:
            path = output / "configs" / f"seed-{seed}.json"
            atomic_json(path, configs[str(seed)])
            config_sha256[str(seed)] = sha256(path)
        registration = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "output_root": str(output), "original_root": str(original_root),
                        "source_sha256": identity, "environment": native.environment_identity(),
                        "proposal_path": str(proposal_path), "proposal_sha256": sha256(proposal_path),
                        "adoption_sha256": sha256(output / "adoption.json"),
                        "source_selection_sha256": sha256(output / "source-selection.json"),
                        "config_sha256": config_sha256, "fixed_seeds": list(SEEDS), "queues": QUEUES,
                        "source_step": ORIGIN, "additional_preparation_steps": EXTRA,
                        "target_preparation_step": TARGET, "maximum_workers": 4,
                        "preparation_only": True, "production_authorized": False,
                        "RDF_and_coordination_hard_gates_adopted": False,
                        "diagnostics_are_diffusion_evidence": False,
                        "original_scalar_thresholds": proposal["stationarity"]["each_seed_fixed_half_mean_absolute_thresholds"]}
        atomic_json(output / "registration.json", registration)
    return output / "registration.json"


def load_registration(path):
    path = Path(path).resolve()
    registration = read_json(path)
    output = Path(registration["output_root"]).resolve()
    if (registration.get("schema") != SCHEMA or path.parent != output
            or registration.get("fixed_seeds") != list(SEEDS) or registration.get("queues") != QUEUES
            or registration.get("source_step") != ORIGIN or registration.get("additional_preparation_steps") != EXTRA
            or registration.get("target_preparation_step") != TARGET or registration.get("maximum_workers") != 4
            or registration.get("preparation_only") is not True or registration.get("production_authorized") is not False
            or registration.get("RDF_and_coordination_hard_gates_adopted") is not False
            or registration.get("diagnostics_are_diffusion_evidence") is not False
            or registration.get("original_scalar_thresholds") != SCALAR_THRESHOLDS
            or registration.get("source_sha256") != coordinator_identity()
            or registration.get("environment") != native.environment_identity()):
        raise ValueError("registered cohort/source/phase/physical settings differ from the fixed preparation contract")
    validate_proposal(registration["proposal_path"])
    if (registration["proposal_sha256"] != sha256(registration["proposal_path"])
            or registration["adoption_sha256"] != sha256(output / "adoption.json")
            or registration["source_selection_sha256"] != sha256(output / "source-selection.json")):
        raise ValueError("proposal/adoption/source-selection record changed")
    adoption, assessed = validate_adoption(output / "adoption.json", registration["proposal_sha256"])
    validate_original_exit(registration["original_root"], adoption)
    receipt = read_json(output / "source-selection.json")
    if (receipt.get("fixed_seeds") != list(SEEDS) or receipt.get("source_step") != ORIGIN
            or receipt.get("pre_NVE") is not True or set(receipt["selected"]) != {str(seed) for seed in SEEDS}
            or set(registration["config_sha256"]) != {str(seed) for seed in SEEDS}):
        raise ValueError("source selection must contain exactly the original eight checkpoints")
    configs = {}
    for seed in SEEDS:
        row = receipt["selected"][str(seed)]
        directory = Path(row["source_run"]).resolve()
        expected_directory = Path(registration["original_root"]).resolve() / "runs" / f"classical-{seed}"
        config_path = output / "configs" / f"seed-{seed}.json"
        config = extension.Config(**read_json(config_path)).validate()
        if (directory != expected_directory or Path(config.source_run).resolve() != directory
                or sha256(config_path) != registration["config_sha256"][str(seed)]
                or config.source_manifest_sha256 != row["manifest_sha256"]
                or config.source_checkpoint_sha256 != row["checkpoint"]["sha256"]
                or config.additional_preparation_steps != EXTRA or config.checkpoint_steps != 4000
                or sha256(directory / "manifest.json") != row["manifest_sha256"]
                or assessed[seed] != row["manifest_sha256"]
                or Path(row["checkpoint"]["file"]).name != row["checkpoint"]["file"]
                or row["checkpoint"]["step"] != ORIGIN
                or sha256(directory / row["checkpoint"]["file"]) != row["checkpoint"]["sha256"]):
            raise ValueError("registered exact original checkpoint/configuration changed")
        original = read_json(directory / "manifest.json")
        expected_base = asdict(native.Config(seed=seed, initial_checkpoint=row["base_config"]["initial_checkpoint"]))
        if (original.get("complete") is not True or original["committed_steps"] != 800000
                or original["config"] != expected_base or row["base_config"] != expected_base
                or original["source_sha256"] != native.source_identity()
                or original["environment"] != native.environment_identity()
                or original["source_sha256"] != row["source_sha256"]
                or original["environment"] != row["environment"] or original["artifacts"] != row["artifacts"]
                or original["box_angstrom"] != row["box_angstrom"]
                or original.get("checkpoint_contract_version") != 2
                or row["checkpoint"] not in original["chunks"]):
            raise ValueError("the bound original model/source/complete-cohort identity differs")
        if row["engine_artifacts"] != {name: original["artifacts"][name] for name in ("model.xml", "integrator.xml")}:
            raise ValueError("registered engine identities differ from the original artifacts")
        for name, artifact in row["artifacts"].items():
            if Path(name).name != name or sha256(directory / name) != artifact["sha256"]:
                raise ValueError("original serialized model/integrator or historical artifact changed")
        _checkpoint_phase(directory / row["checkpoint"]["file"], expected_step=ORIGIN)
        configs[seed] = (config_path, config)
    return {"path": path, "sha256": sha256(path), "registration": registration,
            "selection": receipt["selected"], "output": output, "configs": configs}


def verify_runtime(identity):
    checked = load_registration(identity["path"])
    if checked["sha256"] != identity["sha256"]:
        raise RuntimeError("the registered preparation cohort changed during execution")


def branch_prefix(identity, seed):
    """Hash and validate only preparation state; never repair/replace history."""
    output = identity["output"]
    directory = output / "runs" / f"preparation-{seed}"
    path = directory / "manifest.json"
    prefix_path = output / "control" / "prefixes" / f"seed-{seed}.json"
    if not path.exists():
        if prefix_path.exists() and read_json(prefix_path)["chunks"]:
            raise RuntimeError("a previously committed preparation branch is missing")
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("uncommitted continuation files require review, not replacement")
        return {"committed_steps": ORIGIN, "complete": False, "manifest_sha256": None}, []
    raw = path.read_bytes()
    manifest = json.loads(raw)
    config = identity["configs"][seed][1]
    selected = identity["selection"][str(seed)]
    if (manifest.get("schema") != extension.SCHEMA or manifest["config"] != asdict(config)
            or manifest["source_sha256"] != extension.source_identity()
            or manifest["base_config"] != selected["base_config"]
            or manifest["environment"] != native.environment_identity()
            or manifest.get("preparation_only") is not True or manifest.get("production_started") is not False
            or manifest["source_preparation_step"] != ORIGIN or manifest["target_preparation_step"] != TARGET
            or manifest["artifacts"] != selected["engine_artifacts"]):
        raise ValueError("existing branch differs from its registered preparation-only identity")
    native.verify_manifest(directory, manifest)
    chunks = manifest["chunks"]
    steps = [row["step"] for row in chunks]
    step = manifest["committed_steps"]
    refresh = native.Config().sampling_refresh_steps
    if (not steps or steps[0] != ORIGIN or steps[-1] != step or step > TARGET or step < ORIGIN
            or any(right <= left for left, right in zip(steps, steps[1:]))
            or manifest["complete"] is not (step == TARGET)
            or manifest["refresh_count"] != (step - ORIGIN + refresh - 1) // refresh
            or manifest["next_refresh_step"] != extension._next_refresh(step, refresh)):
        raise ValueError("existing branch has inconsistent committed steps or refresh phase")
    _checkpoint_phase(directory / chunks[-1]["file"], expected_step=step)
    with np.load(directory / chunks[-1]["file"], allow_pickle=False) as data:
        if int(data["next_refresh_step"]) != manifest["next_refresh_step"]:
            raise ValueError("checkpoint and branch manifest refresh phases differ")
    if prefix_path.exists():
        prior = read_json(prefix_path)
        if (prior["registration_sha256"] != identity["sha256"] or prior["seed"] != seed
                or chunks[:len(prior["chunks"])] != prior["chunks"]
                or step < prior["committed_steps"]):
            raise RuntimeError("previously committed preparation prefix changed")
    return {"committed_steps": step, "complete": manifest["complete"],
            "manifest_sha256": hashlib.sha256(raw).hexdigest()}, chunks


@contextmanager
def stop_signal_handlers(requests):
    """Only record requests in handlers; lock and filesystem work run normally."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("the signal-aware preparation coordinator must run in the main thread")
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    def request(number, frame):
        # Python list append avoids acquiring our locks inside a signal handler.
        requests.append(number)
    try:
        for number in previous:
            signal.signal(number, request)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_campaign(registration_path, *, workers=4, wall_seconds=7200, invocation_seconds=1800):
    if (workers not in (1, 2, 4) or isinstance(workers, bool)
            or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 7200
            or not math.isfinite(invocation_seconds) or not 0 < invocation_seconds <= 1800):
        raise ValueError("use1/2/4 workers, outer budget at most7200s and native calls at most1800s")
    identity = load_registration(registration_path)
    output = identity["output"]
    control = output / "control"
    control.mkdir(exist_ok=True)
    failure = control / "FAILURE.json"
    if failure.exists():
        raise RuntimeError("a retained preparation failure requires review; automatic replacement is prohibited")
    journal = Journal(control / "events.jsonl")
    stopped = threading.Event()
    launch_lock = threading.RLock()
    errors = []
    signal_requests = []
    started = time.perf_counter()

    def latch(error, queue, seed):
        with launch_lock:
            stopped.set()
            if not any(previous is error for previous in errors):
                errors.append(error)
            if not failure.exists():
                with failure.open("x") as stream:
                    json.dump({"utc": datetime.now(timezone.utc).isoformat(), "queue": queue, "seed": seed,
                               "registration_sha256": identity["sha256"], "error_type": type(error).__name__,
                               "message": str(error), "no_seed_or_data_replacement": True}, stream)
                    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())

    def fail(error, queue, seed):
        latch(error, queue, seed)
        journal("preparation_stopped_on_error", queue=queue, seed=seed,
                error_type=type(error).__name__, message=str(error), in_flight_children_drain_normally=True)

    with ExitStack() as stack:
        original_leases = stack.enter_context(original_exclusion_leases(identity["registration"]["original_root"]))
        coordinator_lease = stack.enter_context(exclusive_lease(control / "coordinator.lock"))
        seed_leases = {seed: stack.enter_context(exclusive_lease(control / f"seed-{seed}.lock")) for seed in SEEDS}
        if failure.exists():
            raise RuntimeError("a retained preparation failure requires review")
        for seed in SEEDS:
            try:
                branch_prefix(identity, seed)
            except BaseException as error:
                fail(error, "preflight", seed)
                raise RuntimeError("preparation preflight failed; existing data and failure retained") from error
        (control / "prefixes").mkdir(exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        journal("preparation_coordinator_started", registration_sha256=identity["sha256"],
                queues=QUEUES, active_worker_limit=workers, outer_wall_seconds=wall_seconds,
                invocation_wall_seconds=invocation_seconds, production_authorized=False)

        def worker(queue, seeds):
            seed = None
            try:
                for seed in seeds:
                    while True:
                        with ExitStack() as invocation:
                            with launch_lock:
                                try:
                                    remaining = wall_seconds - (time.perf_counter() - started)
                                    if stopped.is_set() or signal_requests or (control / "STOP_AFTER_CURRENT").exists() or remaining <= 0:
                                        journal("preparation_queue_stopped", queue=queue, seed=seed,
                                                reason="signal_request" if signal_requests else
                                                "failure" if stopped.is_set() else
                                                "operator_request" if (control / "STOP_AFTER_CURRENT").exists() else "outer_budget")
                                        return
                                    verify_runtime(identity)
                                    before, prefix = branch_prefix(identity, seed)
                                    if before["complete"]:
                                        journal("preparation_chain_complete", queue=queue, seed=seed, committed=before)
                                        break
                                    atomic_json(control / "prefixes" / f"seed-{seed}.json",
                                                {"registration_sha256": identity["sha256"], "seed": seed,
                                                 **before, "chunks": prefix, "last_native_exit_code": None})
                                    directory = output / "runs" / f"preparation-{seed}"
                                    command = [sys.executable, str(ROOT / RUNNER), "--output", str(directory),
                                               "--wall-seconds", str(min(invocation_seconds, remaining))]
                                    command += ["--resume"] if before["manifest_sha256"] else ["--config", str(identity["configs"][seed][0])]
                                    journal("preparation_invocation_prepared", queue=queue, seed=seed, committed=before,
                                            command=command, config_sha256=identity["registration"]["config_sha256"][str(seed)])
                                    log = invocation.enter_context((output / "logs" / f"seed-{seed}.log").open("a"))
                                    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                                       MKL_NUM_THREADS="1", OPENMM_CPU_THREADS="1", NUMEXPR_NUM_THREADS="1")
                                    # A signal may arrive during the source/hash
                                    # checks. Recheck immediately before admission.
                                    if signal_requests or stopped.is_set():
                                        journal("preparation_queue_stopped", queue=queue, seed=seed,
                                                reason="signal_request" if signal_requests else "failure")
                                        return
                                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                                             stderr=subprocess.STDOUT,
                                                             pass_fds=tuple(handle.fileno() for handle in
                                                                 original_leases["global"] +
                                                                 [original_leases["seeds"][seed], coordinator_lease, seed_leases[seed]]))
                                    invocation.callback(child.wait)  # Own draining before even a journal can fail.
                                    journal("preparation_invocation_started", queue=queue, seed=seed, child_pid=child.pid)
                                except BaseException as error:
                                    latch(error, queue, seed)  # Latch before releasing the launch lock.
                                    raise
                            code = child.wait()
                        if code != 0:
                            with launch_lock:
                                stopped.set()
                        after, updated = branch_prefix(identity, seed)
                        if updated[:len(prefix)] != prefix:
                            raise RuntimeError("a native invocation replaced the committed preparation prefix")
                        atomic_json(control / "prefixes" / f"seed-{seed}.json",
                                    {"registration_sha256": identity["sha256"], "seed": seed, **after,
                                     "chunks": updated, "last_native_exit_code": code})
                        journal("preparation_invocation_finished", queue=queue, seed=seed, child_pid=child.pid,
                                native_exit_code=code, committed=after)
                        if code != 0:
                            raise RuntimeError(f"preparation seed{seed} failed with exit{code}; state and logs retained")
                        if after["committed_steps"] <= before["committed_steps"]:
                            raise RuntimeError("a successful preparation invocation made no progress")
                journal("preparation_queue_complete", queue=queue)
            except BaseException as error:
                fail(error, queue, seed)

        with stop_signal_handlers(signal_requests):
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="water-preparation") as pool:
                futures = [pool.submit(worker, queue, seeds) for queue, seeds in QUEUES.items()]
                pending = set(futures)
                signal_recorded = False
                try:
                    while pending:
                        done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        if signal_requests and not signal_recorded:
                            with launch_lock:
                                stopped.set()
                                atomic_json(control / "STOP_AFTER_CURRENT",
                                            {"reason": "coordinator_signal", "signal": signal_requests[0],
                                             "registration_sha256": identity["sha256"],
                                             "utc": datetime.now(timezone.utc).isoformat()})
                                journal("preparation_signal_stop_requested", signal=signal_requests[0],
                                        admission_stopped=True, in_flight_children_drain_normally=True)
                                signal_recorded = True
                        for future in done:
                            future.result()
                except BaseException as error:
                    # Latch before the pool context's blocking shutdown. A
                    # main-thread error must never admit more queue successors.
                    fail(error, "coordinator", None)
                    raise
        journal("preparation_coordinator_finished", failed=bool(errors),
                elapsed_wall_seconds=time.perf_counter()-started, production_authorized=False,
                stop_signal=signal_requests[0] if signal_requests else None)
    if errors:
        raise RuntimeError("preparation cohort stopped after an error; all started children drained normally") from errors[0]
    return {"stop_signal": signal_requests[0] if signal_requests else None,
            "exit_code": 128 + signal_requests[0] if signal_requests else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("register", help="read/hash all eight completed sources; no dynamics")
    create.add_argument("--original-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--proposal", type=Path, required=True)
    create.add_argument("--adoption-record", type=Path, required=True)
    execute = subparsers.add_parser("run", help="execute only the registered preparation stage")
    execute.add_argument("--registration", type=Path, required=True)
    execute.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    execute.add_argument("--wall-seconds", type=float, default=7200)
    execute.add_argument("--invocation-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.action == "register":
        print(register(args.original_root, args.output, args.proposal, args.adoption_record))
    else:
        result = run_campaign(args.registration, workers=args.workers, wall_seconds=args.wall_seconds,
                              invocation_seconds=args.invocation_seconds)
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
