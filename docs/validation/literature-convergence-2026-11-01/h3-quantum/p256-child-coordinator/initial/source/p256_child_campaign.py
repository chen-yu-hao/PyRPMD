"""One-attempt coordinator for the fixed complete P256 child cohort.

Scheduling, identity and native-step accounting only. No rate analyzer is called.
The physical branch producer and the reviewed process dispatcher are unchanged.
"""
from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

DESIGN = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-fixed-eight-chain-design/proposal.json"
DESIGN_SHA = "c668c15fd89ca0a535afc4082521e3f42ba9e84f5736b5b481bc1d7c376d791d"
BASELINE_SHA = "62ce4fea23a55955cf404ae8aa00a2c70204bf20dd48e74de520169a83724720"
SCHEDULER_PATH = ROOT / "benchmarks/h3/p256_parent_campaign.py"
SCHEDULER_SHA = "7934c1d7f695ce43337d11171cbad8c46ff95311490b755bf4191a56e00c0da3"
PRODUCER_SHA = "9c5c5106d8392ca577f8ea6b54f6e1828dcabb8f86a0e01b732c2e626aa15692"
ANALYZER_SHA = "9e2fdacddcf2ef5b825e0eb0b65789e5f045a8505a8fbd09fe21b704ed4936b6"
OUTPUT = ROOT.parent / "literature-convergence-quantum/p256-fixed-eight-chain-v1/recrossing"
SEEDS = (825601, 825603, 825607, 825609, 825613, 825617, 825621, 825623)
BRANCHES = ((0, .1), (0, .05), (1, .1), (1, .05))
PARENTS, BEADS, PAIRS, HORIZON, CHUNK_PARENTS = 200, 256, 2, 200, 8
EXPECTED_NATIVE_STEPS = 76_800_000
WALL_SECONDS = 36_000
FINALIZATION_RESERVE_SECONDS = 120
AUTH_SCHEMA = "pyrpmd.h3.p256-child-execution-authorization.v1"
REG_SCHEMA = "pyrpmd.h3.p256-child-coordinator-registration.v1"
COMPLETE_SCHEMA = "pyrpmd.h3.p256-child-cohort-completion.v1"
PRODUCER_SCHEMA = "pyrpmd.h3.p256-surface-recrossing.v1"
TOKEN_ENV = "PYRPMD_P256_CHILD_COORDINATOR_TOKEN"
PROCESS_TOKEN = uuid.uuid4().hex
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def identity(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def shared():
    require(sha(SCHEDULER_PATH) == SCHEDULER_SHA, "reviewed process dispatcher changed")
    module = importlib.import_module("benchmarks.h3.p256_parent_campaign")
    require(Path(module.__file__).resolve() == SCHEDULER_PATH.resolve(), "dispatcher import route differs")
    return module


def producer():
    require(sha(ROOT / "benchmarks/h3/p256_surface_recrossing.py") == PRODUCER_SHA
            and sha(ROOT / "benchmarks/h3/p256_recrossing_analyze.py") == ANALYZER_SHA,
            "reviewed P256 producer or analyzer changed")
    module = importlib.import_module("benchmarks.h3.p256_surface_recrossing")
    require(Path(module.__file__).resolve() == (ROOT / "benchmarks/h3/p256_surface_recrossing.py").resolve(),
            "child producer import route differs")
    return module


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_identity(row, guards=None):
    path = Path(row["path"]).resolve()
    actual = identity(path)
    require(actual == row, f"bound input identity changed: {path}")
    if guards is not None:
        guards[str(path)] = row["sha256"]
    return path


def runtime_identity():
    import numpy as np
    import scipy
    binary = Path(importlib.import_module("pyrpmd._native").__file__).resolve()
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "python_executable": identity(sys.executable),
            "actual_native_module": identity(binary)}


def source_inventory(specification, branch_producer):
    names = set(specification["parent_generation"]["scientific_source_identity"]["source_sha256"])
    names.update(name for name in specification["child_experiment_definition"]["PES_and_all_scientific_dependencies"]
                 if not name.startswith("loaded_native:"))
    names.update(("benchmarks/h3/p256_child_campaign.py", "benchmarks/h3/p256_parent_campaign.py",
                  "benchmarks/h3/p256_surface_recrossing.py", "benchmarks/h3/p256_recrossing_analyze.py",
                  "benchmarks/h3/quantum_recrossing_analyze.py"))
    values = {name: identity(ROOT / name) for name in sorted(names)}
    binary = Path(importlib.import_module("pyrpmd._native").__file__).resolve()
    values["loaded_native:" + binary.name] = identity(binary)
    physical = branch_producer.source_identity()
    expected = specification["child_experiment_definition"]["PES_and_all_scientific_dependencies"]
    require(all(physical.get(name) == digest for name, digest in expected.items()), "frozen physical source differs")
    require(all(values[name]["sha256"] == digest for name, digest in physical.items()),
            "actual producer/native imports differ from the source inventory")
    return values, physical


def validate_authorization(authorization, *, actual_sources, actual_runtime, available_cpus):
    require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("parent_adoption") is True,
            "explicit reviewed child execution adoption required")
    require(authorization.get("design_sha256") == DESIGN_SHA
            and authorization.get("P128_baseline_sha256") == BASELINE_SHA,
            "fixed design or complete P128 baseline differs")
    require(authorization.get("source_files") == actual_sources and authorization.get("runtime") == actual_runtime,
            "execution source or runtime was not frozen in authorization")
    require(Path(authorization["output"]).resolve() == OUTPUT.resolve(), "separate child output root differs")
    require(authorization.get("seed_order") == list(SEEDS)
            and authorization.get("branch_order") == [[batch, dt] for batch, dt in BRANCHES],
            "complete fixed seed or branch order differs")
    require(authorization.get("whole_campaign_wall_seconds") == WALL_SECONDS
            and authorization.get("finalization_reserve_seconds") == FINALIZATION_RESERVE_SECONDS
            and authorization.get("wall_budget_origin") == "root explicit child-campaign choice; not the parent-generation proposal",
            "separate child campaign wall allocation differs")
    workers, cpus = authorization.get("workers"), authorization.get("cpu_affinity")
    require(type(workers) is int and 1 <= workers <= 2 and isinstance(cpus, list)
            and len(cpus) == len(set(cpus)) == workers
            and all(type(cpu) is int and cpu in available_cpus for cpu in cpus),
            "one or two distinct available single-CPU workers required")
    require(authorization.get("automatic_resume") is False and authorization.get("automatic_analysis") is False,
            "this coordinator permits neither automatic resume nor rate analysis")


def preflight(authorization_path):
    """Read-only complete-input checks; never register or construct a potential."""
    authorization_path = Path(authorization_path).resolve()
    authorization = read_json(authorization_path)
    require(sha(DESIGN) == DESIGN_SHA, "fixed eight-chain design changed")
    specification = read_json(DESIGN)
    branch_producer = producer()
    require(tuple(branch_producer.SEEDS) == SEEDS and branch_producer.DESIGN_SHA == DESIGN_SHA
            and branch_producer.BASELINE_SHA == BASELINE_SHA, "producer fixed experiment differs")
    sources, physical = source_inventory(specification, branch_producer)
    actual_runtime = runtime_identity()
    validate_authorization(authorization, actual_sources=sources, actual_runtime=actual_runtime,
                           available_cpus=os.sched_getaffinity(0))
    guards = {str(DESIGN): DESIGN_SHA, str(authorization_path): sha(authorization_path)}
    for row in sources.values():
        check_identity(row, guards)
    parent_path = check_identity(authorization["parent_manifest"], guards)
    require(parent_path == (branch_producer.PARENTS / "parents-complete.json").resolve(), "parent manifest route differs")
    manifest = branch_producer.verify_manifest(branch_producer.PARENTS, specification)
    guards[str(branch_producer.PARENTS / "registration.json")] = manifest["registration_sha256"]
    for row in manifest["parents"]:
        for item in row["files"].values():
            check_identity(item, guards)
    baseline_path = check_identity(authorization["P128_complete_four_pair_report"], guards)
    require(baseline_path == branch_producer.BASELINE.resolve() and sha(baseline_path) == BASELINE_SHA,
            "complete P128 baseline route/hash differs")
    baseline = read_json(baseline_path)
    require(tuple(baseline["completed_seeds"]) == tuple(baseline["expected_seeds"]) == tuple(branch_producer.P128_SEEDS)
            and baseline["criteria"]["complete_registered_cohort"] is True
            and baseline["pair_batches"] == [0, 1] and baseline["horizon_fs"] == HORIZON,
            "complete matched P128 four-pair baseline required")
    for name in ("library", "library_build_receipt"):
        path = check_identity(authorization[name], guards)
        expected_key = "external_library" if name == "library" else "external_library_build_receipt"
        expected = specification["parent_generation"][expected_key]
        require(str(path) == expected["path"] and authorization[name]["sha256"] == expected["sha256"]
                and authorization[name]["bytes"] == expected["bytes"], "fixed external model/build identity differs")
    for key in ("synthetic_gate", "independent_coordinator_review", "scientific_review", "resource_decision"):
        check_identity(authorization[key], guards)
    decision = read_json(authorization["resource_decision"]["path"])
    require(decision.get("safe_to_launch_child_workers") is True
            and decision.get("prerequisite_queues_drained") is True
            and type(decision.get("child_workers")) is int
            and decision["child_workers"] == authorization["workers"]
            and type(decision.get("declared_project_active_workers")) is int
            and authorization["workers"] <= decision["declared_project_active_workers"] <= 8,
            "actual child resource/queue-drain decision required")
    require(isinstance(decision.get("evidence"), list) and decision["evidence"], "actual queue-drain evidence required")
    for row in decision["evidence"]:
        check_identity(row, guards)
    shared().verify_guards(guards)
    return {"authorization": authorization, "authorization_path": str(authorization_path),
            "specification": specification, "manifest": manifest, "guards": guards,
            "physical_source_identity": physical, "runtime": actual_runtime, "output": str(OUTPUT.resolve())}


def validate_protocol(protocol, context):
    authorization = context["authorization"]
    expected = {"schema": PRODUCER_SCHEMA, "design_sha256": DESIGN_SHA, "seeds": list(SEEDS),
                "beads": BEADS, "parents_per_seed": PARENTS, "horizon_fs": HORIZON,
                "momentum_pairs_per_batch": PAIRS, "registered_primary_pair_batches": [0, 1],
                "momentum_pair_indices": [0, 1, 2, 3], "children_per_pair": 2,
                "timestep_fs": [.1, .05], "primary_timestep_fs": .05,
                "parent_completed_sweeps": list(range(2100, 22001, 100)),
                "parent_manifest_sha256": authorization["parent_manifest"]["sha256"],
                "parent_registration_sha256": context["manifest"]["registration_sha256"],
                "source_sha256": context["physical_source_identity"]}
    require(all(protocol.get(key) == value for key, value in expected.items()), "registered child protocol differs")
    require(protocol["P128_complete_four_pair_report"] == {
        "path": authorization["P128_complete_four_pair_report"]["path"], "sha256": BASELINE_SHA},
        "registered baseline differs")


def branch_directory(output, seed, timestep, batch):
    base = Path(output) if batch == 0 else Path(output) / "extra-pair-batch-1"
    return base / f"seed-{seed}" / f"dt-{timestep:g}" / "horizon-200"


def parent_identity(context, seed):
    row = next(row for row in context["manifest"]["parents"] if row["seed"] == seed)
    return {"checkpoint_sha256": row["checkpoint_sha256"],
            "settings_sha256": row["files"]["settings.json"]["sha256"],
            "report_sha256": row["files"]["report.json"]["sha256"], "seed": seed,
            "BAR_log_TS_quantum_classical_ratio": row["BAR_log_TS_quantum_classical_ratio"],
            "parent_manifest_sha256": context["authorization"]["parent_manifest"]["sha256"],
            "parent_registration_sha256": context["manifest"]["registration_sha256"],
            "P256_design_sha256": DESIGN_SHA}


def validate_chunk(arrays, *, indices, settings_fingerprint, timestep, batch):
    """Schema, native counts and pairing identities only; no flux/rate inference."""
    import numpy as np
    n = len(indices)
    shapes = {"parent_index": (n,), "parent_completed_sweeps": (n,), "channel": (n,),
              "momentum_pair_index": (PAIRS,), "time_fs": (HORIZON,),
              "surface_values": (n, PAIRS, 2, HORIZON), "minimum_pair_at_1fs": (n, PAIRS, 2, HORIZON+1),
              "energy_extrema": (n, PAIRS, 2, 3), "initial_p": (n, PAIRS, BEADS, 3, 3),
              "final_q": (n, PAIRS, 2, BEADS, 3, 3), "final_p": (n, PAIRS, 2, BEADS, 3, 3),
              "final_step": (n, PAIRS, 2), "final_time_au": (n, PAIRS, 2),
              "flux_metric": (n,), "velocity": (n, PAIRS), "numerator": (n, PAIRS, HORIZON),
              "denominator": (n, PAIRS)}
    require(set(arrays) == set(shapes) | {"fingerprint", "native_steps_this_stage", "previous_chunk_sha256"},
            "child raw payload schema differs")
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        require(value.shape == shape and value.dtype.kind in "iuf" and np.isfinite(value).all(),
                f"invalid child array shape/value: {name}")
    require(str(arrays["fingerprint"].item()) == settings_fingerprint
            and str(arrays["previous_chunk_sha256"].item()) == "", "chunk identity or unregistered extension differs")
    require(np.array_equal(arrays["parent_index"], indices)
            and np.array_equal(arrays["parent_completed_sweeps"], 2000+(indices+1)*100)
            and np.array_equal(arrays["momentum_pair_index"], np.arange(PAIRS*batch, PAIRS*(batch+1)))
            and np.array_equal(arrays["time_fs"], np.arange(1, HORIZON+1)), "chunk fixed observation identity differs")
    require(np.isin(arrays["channel"], [0, 1]).all() and np.all(arrays["flux_metric"] > 0)
            and np.all(arrays["denominator"] >= 0), "invalid channel or flux metadata")
    steps = round(HORIZON/timestep)
    native = np.asarray(arrays["native_steps_this_stage"])
    require(native.shape == () and native.dtype.kind in "iu" and int(native) == n*PAIRS*2*steps
            and arrays["final_step"].dtype.kind in "iu" and np.all(arrays["final_step"] == steps),
            "actual native/final step counts differ")
    return {"native_steps": int(native),
            "initial_p_sha256": hashlib.sha256(arrays["initial_p"].tobytes(order="C")).hexdigest(),
            "channel_sha256": hashlib.sha256(arrays["channel"].tobytes(order="C")).hexdigest()}


def read_child_chunk(raw):
    from pyrpmd.checkpoint import load_checkpoint
    with load_checkpoint(io.BytesIO(raw), expected_kind="h3.quantum-surface-children", require_integrity=True) as saved:
        return {name: saved[name] for name in saved.files}


def verify_branch(output, context, protocol, seed, timestep, batch):
    import numpy as np
    require(seed in SEEDS and (batch, timestep) in BRANCHES, "unregistered branch")
    directory = branch_directory(output, seed, timestep, batch)
    settings_path, report_path = directory/"settings.json", directory/"report.json"
    require(directory.resolve().is_relative_to(Path(output).resolve())
            and not settings_path.is_symlink() and not report_path.is_symlink(), "child branch route differs")
    before = {"settings": identity(settings_path), "report": identity(report_path)}
    settings, report = read_json(settings_path), read_json(report_path)
    expected = {"schema": PRODUCER_SCHEMA, "phase": "primary", "seed": seed, "beads": BEADS,
                "parents": PARENTS, "timestep_fs": timestep, "horizon_fs": HORIZON, "start_horizon_fs": 0,
                "pair_batch": batch, "momentum_pair_indices": list(range(PAIRS*batch, PAIRS*(batch+1))),
                "protocol_sha256": fingerprint(protocol), "parent_identity": parent_identity(context, seed),
                "source_sha256": context["physical_source_identity"], "chunk_parents": CHUNK_PARENTS}
    require(settings == expected, "branch settings differ from complete fixed parent/protocol identity")
    count = PARENTS*PAIRS*2*round(HORIZON/timestep)
    report_expected = {"schema": PRODUCER_SCHEMA, "phase": "primary", "seed": seed, "pair_batch": batch,
                       "timestep_fs": timestep, "horizon_fs": HORIZON, "complete": True,
                       "parents_completed": PARENTS, "pairs_per_parent": PAIRS,
                       "momentum_pair_indices": expected["momentum_pair_indices"], "source_unchanged": True,
                       "native_steps_this_stage": count, "unique_native_steps_all_stages": count,
                       "python_steps": 0, "settings_sha256": before["settings"]["sha256"],
                       "parent_identity": expected["parent_identity"]}
    require(all(report.get(key) == value for key, value in report_expected.items()), "incomplete or changed branch report")
    names = [f"chunk-{left:04d}.npz" for left in range(0, PARENTS, CHUNK_PARENTS)]
    inventory = report["chunks"]
    require([row["file"] for row in inventory] == names and len(set(names)) == len(inventory),
            "child chunk inventory must be the exact ordered complete set")
    require({path.name for path in directory.glob("chunk-*.npz")} == set(names), "unregistered or missing child chunks")
    chunks, total = [], 0
    for left, item in zip(range(0, PARENTS, CHUNK_PARENTS), inventory):
        path = directory/item["file"]
        require(not path.is_symlink(), "child chunk cannot be a symlink")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == item["sha256"], "child chunk changed after report commit")
        arrays = read_child_chunk(raw)
        checked = validate_chunk(arrays, indices=np.arange(left, min(left+CHUNK_PARENTS, PARENTS)),
                                 settings_fingerprint=fingerprint(settings), timestep=timestep, batch=batch)
        total += checked["native_steps"]
        chunks.append({"path": str(path.resolve()), "bytes": len(raw), "sha256": digest, **checked})
    require(total == count and identity(settings_path) == before["settings"]
            and identity(report_path) == before["report"], "branch changed during verification")
    return {"seed": seed, "pair_batch": batch, "timestep_fs": timestep, "complete": True,
            "parents": PARENTS, "pairs_per_parent": PAIRS, "native_steps": total, "python_steps": 0,
            "settings": before["settings"], "report": before["report"], "chunks": chunks}


def verify_seed(output, context, protocol, seed):
    rows = [verify_branch(output, context, protocol, seed, dt, batch) for batch, dt in BRANCHES]
    for batch in (0, 1):
        coarse, fine = rows[2*batch:2*batch+2]
        require([(x["initial_p_sha256"], x["channel_sha256"]) for x in coarse["chunks"]]
                == [(x["initial_p_sha256"], x["channel_sha256"]) for x in fine["chunks"]],
                "paired timesteps have different parent/channel/Maxwell draws")
    require(sum(row["native_steps"] for row in rows) == EXPECTED_NATIVE_STEPS//len(SEEDS),
            "per-seed native accounting differs")
    return {"seed": seed, "complete": True, "branches": rows,
            "native_steps": sum(row["native_steps"] for row in rows), "rate_analysis_performed": False}


def verify_complete_cohort(rows, output):
    require([row["seed"] for row in rows] == list(SEEDS), "full fixed eight-seed order required")
    reports, settings_files, chunks = [], [], []
    total = 0
    for seed, row in zip(SEEDS, rows):
        require(row["complete"] is True and row["rate_analysis_performed"] is False
                and [(item["pair_batch"], item["timestep_fs"]) for item in row["branches"]] == list(BRANCHES),
                "complete fixed branch matrix required")
        seed_steps = 0
        for (batch, dt), branch in zip(BRANCHES, row["branches"]):
            directory = branch_directory(output, seed, dt, batch).resolve()
            require(branch["seed"] == seed and branch["complete"] is True
                    and branch["parents"] == PARENTS and branch["pairs_per_parent"] == PAIRS
                    and branch["python_steps"] == 0
                    and branch["native_steps"] == PARENTS*PAIRS*2*round(HORIZON/dt),
                    "branch native counts or cohort identity differ")
            require(Path(branch["report"]["path"]).resolve() == directory/"report.json"
                    and Path(branch["settings"]["path"]).resolve() == directory/"settings.json",
                    "completed branch report/settings route differs")
            expected_paths = [str(directory/f"chunk-{left:04d}.npz") for left in range(0, PARENTS, CHUNK_PARENTS)]
            require([item["path"] for item in branch["chunks"]] == expected_paths,
                    "completed branch has a missing, repeated or foreign chunk")
            reports.append(branch["report"]["path"]); settings_files.append(branch["settings"]["path"])
            chunks.extend(expected_paths)
            seed_steps += branch["native_steps"]
        require(row["native_steps"] == seed_steps == EXPECTED_NATIVE_STEPS//len(SEEDS), "seed native step total differs")
        total += seed_steps
    require(len(set(reports)) == len(set(settings_files)) == 32
            and len(chunks) == len(set(chunks)) == 800 and total == EXPECTED_NATIVE_STEPS,
            "complete 32-branch/800-chunk/76.8M-step inventory required")
    return {"branches": len(reports), "chunks": len(chunks), "native_steps": total}


@contextmanager
def worker_interrupts():
    previous = {}
    def stop(number, frame):
        raise InterruptedError(f"coordinator-owned worker interrupted by signal {number}")
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.getsignal(number)
        signal.signal(number, stop)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def inherited_token(token):
    old = os.environ.get(TOKEN_ENV)
    os.environ[TOKEN_ENV] = token
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = old


def worker(registration_path, attempt_path, seed, cpu):
    start = time.monotonic()
    registration_path, attempt_path = Path(registration_path).resolve(), Path(attempt_path).resolve()
    registration, plan = read_json(registration_path), read_json(attempt_path/"plan.json")
    require(registration.get("schema") == REG_SCHEMA and plan["registration_sha256"] == sha(registration_path),
            "worker registration differs")
    require(seed in SEEDS and cpu in plan["cpu_affinity"] and plan["coordinator_pid"] == os.getppid()
            and plan["coordinator_process_token"] == os.environ.get(TOKEN_ENV), "worker requires its owning live coordinator")
    context = preflight(registration["authorization_path"])
    output = Path(context["output"])
    require(registration_path == output/"coordinator-registration.json"
            and attempt_path.parent == output/"coordinator-attempts"
            and registration["authorization_sha256"] == sha(registration["authorization_path"])
            and registration["source_sha256"] == sha(__file__)
            and registration["seed_order"] == list(SEEDS),
            "worker campaign route or authorization differs")
    expected_guards = dict(context["guards"], **{str(registration_path): sha(registration_path),
                                               str(output/"protocol.json"): sha(output/"protocol.json")})
    require(plan["source_guards"] == expected_guards
            and registration["protocol_sha256"] == sha(output/"protocol.json")
            and plan["seed_order"] == list(SEEDS)
            and plan["branch_order"] == [[batch, dt] for batch, dt in BRANCHES]
            and plan["cpu_affinity"] == context["authorization"]["cpu_affinity"]
            and plan["workers"] == context["authorization"]["workers"]
            and plan["whole_campaign_wall_seconds"] == WALL_SECONDS
            and plan["finalization_reserve_seconds"] == FINALIZATION_RESERVE_SECONDS,
            "worker attempt/source/CPU/count contract differs")
    shared().verify_guards(plan["source_guards"])
    shared().atomic_new_json(attempt_path/f"seed-{seed}-worker-claim.json", {
        "seed": seed, "pid": os.getpid(), "parent_pid": os.getppid(), "process_token": PROCESS_TOKEN,
        "coordinator_process_token": plan["coordinator_process_token"], "created_utc": now()})
    os.sched_setaffinity(0, {cpu})
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "worker threads must be one")
    raw_protocol = read_json(output/"protocol.json")
    protocol = dict(raw_protocol); protocol.pop("created_before_primary_children_utc")
    validate_protocol(protocol, context)
    calls, failure, result = [], None, None
    try:
        with worker_interrupts():
            branch_producer = producer()
            for batch, dt in BRANCHES:
                shared().verify_guards(plan["source_guards"])
                branch = branch_directory(output, seed, dt, batch)
                require(not branch.exists(), "no implicit branch resume or repeated dispatch is permitted")
                entered = time.monotonic()
                branch_producer.run_seed(output, context["authorization"]["library"]["path"], seed, dt,
                                         horizon=HORIZON, pair_batch=batch)
                checked = verify_branch(output, context, protocol, seed, dt, batch)
                calls.append({"batch": batch, "timestep_fs": dt, "wall_seconds": time.monotonic()-entered,
                              "native_steps": checked["native_steps"], "report": checked["report"]})
                shared().atomic_new_json(attempt_path/f"seed-{seed}-batch-{batch}-dt-{dt:g}.json", calls[-1])
            result = verify_seed(output, context, protocol, seed)
            shared().verify_guards(plan["source_guards"])
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        shared().atomic_new_json(attempt_path/f"seed-{seed}-execution.json", {
            "seed": seed, "pid": os.getpid(), "CPU_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "whole_worker_call_wall_seconds": time.monotonic()-start,
            "whole_process_user_CPU_seconds": usage.ru_utime, "whole_process_system_CPU_seconds": usage.ru_stime,
            "peak_RSS_KiB": usage.ru_maxrss, "completed_branches": calls, "complete": result is not None,
            "native_steps": result["native_steps"] if result else None, "failure": failure,
            "created_utc": now(), "rate_analysis_performed": False})
    return 0


def worker_command(registration, attempt, seed, cpu):
    return [sys.executable, str(Path(__file__).resolve()), "_worker", "--registration", str(registration),
            "--attempt", str(attempt), "--seed", str(seed), "--cpu", str(cpu)]


def run_campaign(authorization_path):
    start = ENTRY_MONOTONIC
    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    context = preflight(authorization_path)
    authorization, output = context["authorization"], Path(context["output"])
    deadline = start + WALL_SECONDS
    with shared().campaign_lock(output) as lease:
        require(not output.exists() or not any(output.iterdir()), "one-attempt campaign output must be new/empty; no automatic resume")
        require(time.monotonic() < deadline-FINALIZATION_RESERVE_SECONDS, "preflight exhausted the campaign wall budget")
        try:
            protocol = producer().register(output, authorization["library"]["path"])
            validate_protocol(protocol, context)
            raw_protocol = read_json(output/"protocol.json")
            raw_protocol.pop("created_before_primary_children_utc")
            require(raw_protocol == protocol, "coordinator protocol registration differs")
        except BaseException as error:
            shared().atomic_new_json(output/"coordinator-pre-dispatch-failure.json", {
                "created_utc": now(), "error_type": type(error).__name__, "message": str(error),
                "workers_launched": 0, "automatic_retry": False, "authorization": identity(context["authorization_path"])})
            raise
        registration = output/"coordinator-registration.json"
        shared().atomic_new_json(registration, {"schema": REG_SCHEMA, "created_utc": now(),
            "authorization_path": context["authorization_path"], "authorization_sha256": sha(context["authorization_path"]),
            "output": str(output), "source_sha256": sha(__file__), "seed_order": list(SEEDS),
            "protocol_sha256": sha(output/"protocol.json"), "parent_manifest": authorization["parent_manifest"],
            "P128_baseline_sha256": BASELINE_SHA, "automatic_resume": False, "automatic_analysis": False})
        attempt = output/"coordinator-attempts"/uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        guards = dict(context["guards"], **{str(registration): sha(registration),
                                          str(output/"protocol.json"): sha(output/"protocol.json")})
        plan = {"schema": "pyrpmd.h3.p256-child-coordinator-attempt.v1", "created_utc": now(),
                "coordinator_pid": os.getpid(), "coordinator_process_token": PROCESS_TOKEN,
                "registration_sha256": sha(registration), "source_guards": guards,
                "workers": authorization["workers"], "cpu_affinity": authorization["cpu_affinity"],
                "whole_campaign_wall_seconds": WALL_SECONDS, "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
                "wall_budget_origin": authorization["wall_budget_origin"], "seed_order": list(SEEDS),
                "branch_order": [[batch, dt] for batch, dt in BRANCHES], "script_argv": sys.argv}
        shared().atomic_new_json(attempt/"plan.json", plan)
        with inherited_token(PROCESS_TOKEN):
            outcome = shared().dispatch_jobs(SEEDS,
                lambda seed, cpu: worker_command(registration, attempt, seed, cpu),
                lambda seed: verify_seed(output, context, protocol, seed),
                cpus=authorization["cpu_affinity"], deadline=deadline-FINALIZATION_RESERVE_SECONDS,
                attempt=attempt, lease=lease, guard=lambda: shared().verify_guards(guards))
        errors, final = list(outcome["errors"]), []
        if not errors:
            try:
                require(set(outcome["results"]) == set(SEEDS), "all eight fixed child seeds must finish")
                shared().verify_guards(guards)
                final = [verify_seed(output, context, protocol, seed) for seed in SEEDS]
                verify_complete_cohort(final, output)
                require(time.monotonic() <= deadline, "whole campaign deadline includes final raw verification")
            except BaseException as error:
                errors.append(f"final audit: {type(error).__name__}: {error}")
        execution = []
        for seed in SEEDS:
            path = attempt/f"seed-{seed}-execution.json"
            execution.append(identity(path) if path.is_file() else {
                "seed": seed, "status": "not_launched" if seed not in outcome["launched_seeds"] else "unavailable_after_failure_or_bounded_termination"})
            if not errors:
                try:
                    receipt = read_json(path)
                    require(receipt["seed"] == seed and receipt["complete"] is True and receipt["failure"] is None
                            and receipt["native_steps"] == EXPECTED_NATIVE_STEPS//len(SEEDS)
                            and [(row["batch"], row["timestep_fs"]) for row in receipt["completed_branches"]] == list(BRANCHES)
                            and len(receipt["CPU_affinity"]) == 1
                            and receipt["CPU_affinity"][0] in authorization["cpu_affinity"]
                            and receipt["thread_environment"] == {name: "1" for name in THREAD_VARIABLES},
                            "worker execution receipt differs")
                    for key in ("whole_worker_call_wall_seconds", "whole_process_user_CPU_seconds", "whole_process_system_CPU_seconds"):
                        value = receipt[key]
                        require(isinstance(value, (int, float)) and 0 <= value < float("inf"), "invalid measured worker resource receipt")
                except BaseException as error:
                    errors.append(f"worker accounting: {type(error).__name__}: {error}")
        complete = not errors and len(final) == len(SEEDS) and outcome["all_workers_drained"]
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = {"schema": COMPLETE_SCHEMA, "created_utc": now(), "complete_registered_cohort": complete,
                  "eligible_for_separate_full_cohort_analysis": complete, "rate_analysis_performed": False,
                  "seed_order": list(SEEDS), "branch_order": [[batch, dt] for batch, dt in BRANCHES],
                  "expected_branch_count": 32, "expected_chunk_count": 800, "expected_native_steps": EXPECTED_NATIVE_STEPS,
                  "native_steps": sum(row["native_steps"] for row in final) if complete else None,
                  "protocol": identity(output/"protocol.json"), "registration": identity(registration),
                  "attempt_plan": identity(attempt/"plan.json"), "authorization": identity(context["authorization_path"]),
                  "parent_manifest": authorization["parent_manifest"], "P128_baseline_sha256": BASELINE_SHA,
                  "source_files": authorization["source_files"], "runtime": context["runtime"],
                  "seeds": final if complete else [], "partial_verified_results": list(outcome["results"].values()) if errors else [],
                  "errors": errors, "all_owned_workers_drained": outcome["all_workers_drained"],
                  "launched_seeds": outcome["launched_seeds"], "unstarted_seeds": outcome["unstarted_seeds"],
                  "execution_receipts": execution, "whole_campaign_wall_seconds": time.monotonic()-start,
                  "child_user_CPU_seconds": usage.ru_utime-child_usage_before.ru_utime,
                  "child_system_CPU_seconds": usage.ru_stime-child_usage_before.ru_stime,
                  "coordinator_user_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                  "coordinator_system_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  "automatic_resume": False, "scientific_convergence_claimed": False}
        shared().atomic_new_json(attempt/"result.json", result)
        if complete:
            shared().atomic_new_json(output/"children-complete.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "run"):
        command = sub.add_parser(action)
        command.add_argument("--authorization", required=True, type=Path)
    child = sub.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("--registration", required=True, type=Path)
    child.add_argument("--attempt", required=True, type=Path)
    child.add_argument("--seed", required=True, type=int)
    child.add_argument("--cpu", required=True, type=int)
    args = parser.parse_args()
    if args.action == "_worker":
        return worker(args.registration, args.attempt, args.seed, args.cpu)
    if args.action == "preflight":
        value = preflight(args.authorization)
        print(json.dumps({"preflight_passed": True, "guards": len(value["guards"]), "workers_launched": 0,
                          "PES_evaluations": 0, "output_root_created": False}))
        return 0
    value = run_campaign(args.authorization)
    print(json.dumps({key: value[key] for key in ("complete_registered_cohort", "errors", "native_steps", "launched_seeds")}))
    return 0 if value["complete_registered_cohort"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
