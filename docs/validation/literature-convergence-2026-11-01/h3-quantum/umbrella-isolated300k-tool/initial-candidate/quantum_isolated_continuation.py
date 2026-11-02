"""One bounded eight-seed300000 continuation from an exact isolated100000 restore.

Only preparation, provenance and process ownership are new. Every scientific
step is the unchanged quantum_sampling_campaign.one_seed entrypoint. Original
data, verified prior extractions and old leases are never written here.
"""
from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()
import resource
import signal
import sys


def install_worker_limits(owner_deadline, *, cpu_seconds=21600, wall_seconds=21600):
    """Worker-only bootstrap, before NumPy/PES import; survives owner death."""
    import math
    if not math.isfinite(owner_deadline) or owner_deadline <= time.monotonic():
        raise ValueError("worker absolute deadline has already expired")
    if type(cpu_seconds) is not int or cpu_seconds < 1 or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("positive worker limits required")
    old = resource.getrlimit(resource.RLIMIT_CPU)
    if old[1] != resource.RLIM_INFINITY and old[1] < cpu_seconds:
        raise ValueError("inherited CPU hard limit is smaller than the registered worker cap")
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    deadline = min(owner_deadline, time.monotonic()+wall_seconds)
    # Default termination avoids serializing a mid-sweep state from Python's
    # finally clause. The last completed atomic checkpoint remains preserved.
    for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(number, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.000001, deadline-time.monotonic()))
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    return {"owner_work_deadline": owner_deadline, "worker_wall_deadline": deadline,
            "RLIMIT_CPU": list(resource.getrlimit(resource.RLIMIT_CPU)),
            "alarm_unblocked": signal.SIGALRM not in signal.pthread_sigmask(signal.SIG_BLOCK, set()),
            "installed_before_scientific_imports": True}


BOOTSTRAP = None
if __name__ == "__main__" and "--_worker" in sys.argv:
    # Install an independently expiring worker budget before all other imports.
    BOOTSTRAP = install_worker_limits(float(sys.argv[sys.argv.index("--worker-deadline")+1]))

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True

POLICY = ROOT/"docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-300000-proposal/implementation-policy.json"
POLICY_SHA = "6a3b93f045acd421507b407dd4b2b3c36600591c0488c4437b04957a3e326409"
PROPOSAL_SHA = "65568f44ab61a40207c4b7bb3f72ba9e228eade37ca152fc753cb5612aa67e7a"
TESTS = ROOT/"tests/test_h3_quantum_isolated_continuation.py"
DISPATCH = ROOT/"benchmarks/h3/p256_parent_campaign.py"
DISPATCH_SHA = "7934c1d7f695ce43337d11171cbad8c46ff95311490b755bf4191a56e00c0da3"
ALARM = ROOT/"benchmarks/h3/p256_complete_analysis.py"
ALARM_SHA = "67e4e3eb1b7a7cc875fd14c74829c71be265a8171c516b7b10af5dca461d7b7e"
INSPECTOR = ROOT/"benchmarks/h3/quantum_sampling_scheduler.py"
INSPECTOR_SHA = "1fa4d02c5a9ed5884e81fd0de1203cec47570d9a514a512d636524583e59ed98"
DRIVER = ROOT/"benchmarks/h3/quantum_sampling_campaign.py"
DRIVER_SHA = "29a70de601253ba2262fd1be3a65e65d4bc7cef26652cbe2ff2d30ccccf9df50"
SEEDS = (211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
AUTH_SCHEMA = "pyrpmd.h3.isolated300k-root-adoption.v1"
REG_SCHEMA = "pyrpmd.h3.isolated300k-registration.v1"
ATTEMPT_SCHEMA = "pyrpmd.h3.isolated300k-attempt.v1"
COMPLETE_SCHEMA = "pyrpmd.h3.isolated300k-completion.v1"
TOKEN_ENV = "PYRPMD_ISOLATED300K_OWNER_TOKEN"
PROCESS_TOKEN = uuid.uuid4().hex


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path, check=lambda: None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular non-symlink input required")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while True:
            check(); data = stream.read(1024*1024)
            if not data:
                break
            value.update(data)
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(fields(before) == fields(after) == fields(path.stat()), "input changed during hashing")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": value.hexdigest()}


def bound(row, check=lambda: None):
    require(identity(row["path"], check) == row, "bound input identity changed: "+row["path"])
    return Path(row["path"])


def read_bound(row, check=lambda: None):
    path = bound(row, check); data = path.read_bytes()
    require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "JSON changed after hashing")
    return json.loads(data)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def verify_guards(rows, check=lambda: None):
    for row in rows:
        bound(row, check)


def module_at(path, digest):
    require(identity(path)["sha256"] == digest, "frozen helper changed")
    name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    value = importlib.import_module(name)
    require(Path(value.__file__).resolve() == path.resolve(), "import route differs")
    return value


def runtime():
    return {"executable": str(Path(sys.executable)), "python": sys.version,
            "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy")}


def validate_runtime(expected):
    actual = runtime()
    require(all(actual[name] == expected[name] for name in actual), "registered original runtime differs")
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "single-thread environment required")
    return actual


def validate_cpus(cpus):
    require(isinstance(cpus, list) and len(cpus) in (2, 4)
            and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
            and len(set(cpus)) == len(cpus), "exactly2or4 distinct fixed CPU slots required")
    return cpus


def storage_admission(storage, adoption, free_bytes):
    reservations = adoption.get("remaining_external_reservations")
    require(isinstance(reservations, list) and len(reservations) >= 2
            and all(isinstance(row, dict) for row in reservations)
            and {row.get("name") for row in reservations} >= {"water", "P256_children"}
            and len({row.get("name") for row in reservations}) == len(reservations)
            and all(type(row.get("remaining_bytes")) is int and row["remaining_bytes"] >= 0
                    and isinstance(row.get("basis"), str) and bool(row["basis"]) for row in reservations),
            "root must explicitly account for remaining water and active P256 growth")
    reserved = adoption.get("remaining_other_reserved_bytes")
    require(type(reserved) is int and reserved == sum(row["remaining_bytes"] for row in reservations)
            and isinstance(adoption.get("other_reservation_basis"), str)
            and bool(adoption["other_reservation_basis"]), "root must bind current remaining external storage reservation")
    required = (storage["mandatory_free_reserve_bytes"]
                +storage["MC_new_working_future_archive_and_fresh_replay_allowance_bytes"]+reserved)
    require(free_bytes >= required, "insufficient space for fixed own/other reservations and4GiB free reserve")
    return required


def validate_terminal(observation):
    result = observation.get("tool_result", {})
    require(observation.get("schema") == "pyrpmd.root-actual-exec-observation.v1"
            and type(observation.get("exec_session_id")) is int and observation["exec_session_id"] == 46558
            and type(result.get("exit_code")) is int and result["exit_code"] == 0
            and bool(result.get("chunk_id")) and bool(observation.get("observed_utc")),
            "actual successful100k archive session exit required")


def validate_receipts(adoption, check=lambda: None):
    review = read_bound(adoption["independent_review"], check)
    require(review.get("approved_for_fixed_isolated300000") is True
            and review.get("source_sha256") == identity(__file__, check)["sha256"]
            and review.get("tests_sha256") == identity(TESTS, check)["sha256"]
            and review.get("policy_sha256") == POLICY_SHA,
            "approving current-source/current-test independent review required")
    gate = read_bound(adoption["test_gate"], check)
    require(type(gate.get("returncode")) is int and gate["returncode"] == 0
            and gate.get("failures") == gate.get("errors") == gate.get("skipped") == 0
            and type(gate.get("passed")) is int and gate["passed"] > 0
            and gate.get("source_sha256") == identity(__file__, check)["sha256"]
            and gate.get("tests_sha256") == identity(TESTS, check)["sha256"],
            "successful exact-source inert gate required")


def load_context(adoption_path, check=lambda: None):
    require(identity(POLICY, check)["sha256"] == POLICY_SHA, "fixed implementation policy changed")
    policy = json.loads(POLICY.read_text())
    proposal = read_bound(policy["proposal"], check)
    require(policy["proposal"]["sha256"] == PROPOSAL_SHA, "original scientific proposal changed")
    read_bound(policy["resource_estimates"], check)
    adoption_path = Path(adoption_path).resolve(); adoption = json.loads(adoption_path.read_text())
    require(adoption.get("schema") == AUTH_SCHEMA and adoption.get("adopted") is True
            and adoption.get("scope") == "restore_and_run_fixed300000"
            and adoption.get("source_sha256") == identity(__file__, check)["sha256"]
            and adoption.get("policy_sha256") == POLICY_SHA,
            "new exact root adoption of restore and fixed continuation required")
    validate_receipts(adoption, check)
    cpus = validate_cpus(adoption.get("CPU_affinity"))
    require(adoption.get("workers") == len(cpus)
            and adoption.get("sampling_envelope") == policy["sampling_envelope"]
            and adoption.get("storage_revision") == policy["storage_revision"], "resource contract differs")
    require(adoption.get("no_original_or_other_writer_running_or_pending") is True
            and adoption.get("original_and_verified_evidence_remain_read_only") is True
            and adoption.get("only_new_registered_working_copy_may_change") is True
            and adoption.get("no_automatic_retry_or_extension") is True,
            "explicit root no-overlap and read-only evidence promises required")
    actual_runtime = validate_runtime(proposal["runtime"])
    route = proposal["new_isolated_working_copy"]
    output = Path(route["root"])
    require(output.is_absolute() and output.resolve() == output
            and output.parent == WORK/"literature-convergence-quantum"
            and Path(route["source_code_root"]).resolve() == ROOT,
            "fixed new working route/source root required")
    require(Path(route["fresh_archive_restore"]) == output/"restored"
            and Path(route["actual_sampling_campaign"]) == output/"restored/data/campaign-run",
            "new campaign must be inside the new isolated restore")
    for forbidden in (WORK/"literature-convergence-quantum/campaign-run",
                      WORK/"h3-quantum-umbrella-stage-100000-restored-v2"):
        require(not output.is_relative_to(forbidden) and not forbidden.is_relative_to(output), "old evidence cannot be a working destination")
    evidence = proposal["required_successful_prior_archive"]
    execution = read_bound(evidence["archive_execution"], check)
    validate_terminal(read_bound(evidence["root_actual_normal_exit"], check))
    require(execution.get("complete_archive") is True and execution.get("exact_scientific_replay") is True
            and execution.get("replay_invocations") == 1 and execution.get("owned_replay_drained") is True
            and execution.get("all8_state_RNG_and_prefixes_equal") is True
            and execution.get("all_original_archived_extracted_bytes_verified") is True
            and execution.get("archive") == evidence["archive"]
            and execution.get("member_inventory") == evidence["member_inventory"],
            "new successful exact100k archive/replay identities required")
    members = read_bound(evidence["member_inventory"], check)
    require(members.get("archive") == evidence["archive"] and members.get("member_count") == 116
            and members.get("raw_NPZ_count") == 33, "registered100k member inventory differs")
    protocol = read_bound(proposal["fixed_scientific_contract"]["protocol"], check)
    require(tuple(protocol["independent_ladder_seeds"]) == SEEDS
            and protocol["fixed_refinement_prefixes"] == [30000, 100000, 300000]
            and protocol["warmup_sweeps"] == 5000 and protocol["beads_primary"] == 128,
            "registered fixed physics/cohort differs")
    source_rows = [*proposal["protected_source_files"], identity(__file__), identity(TESTS),
                   identity(POLICY), policy["proposal"], policy["resource_estimates"],
                   adoption["independent_review"], adoption["test_gate"], identity(adoption_path)]
    for path, expected in ((DISPATCH, DISPATCH_SHA), (ALARM, ALARM_SHA), (INSPECTOR, INSPECTOR_SHA), (DRIVER, DRIVER_SHA)):
        row = identity(path, check); require(row["sha256"] == expected, "frozen implementation changed")
        source_rows.append(row)
    library = identity(route["library"]["path"], check)
    require(library["sha256"] == route["library"]["sha256"], "exact BKMP2 model changed")
    source_rows.append(library)
    verify_guards(source_rows, check)
    immutable = [evidence[name] for name in ("archive", "member_inventory", "archive_execution", "root_actual_normal_exit",
                    "root_adoption", "compact_publication_inventory", "independent_completed_archive_review")]
    verify_guards(immutable, check)
    return {"policy": policy, "proposal": proposal, "adoption": adoption, "adoption_path": str(adoption_path),
            "output": output, "campaign": Path(route["actual_sampling_campaign"]), "cpus": cpus,
            "protocol": protocol, "source_guards": source_rows, "immutable_guards": immutable,
            "members": members, "runtime": actual_runtime, "library": library}


def safe_member(name):
    value = PurePosixPath(name)
    require(not value.is_absolute() and ".." not in value.parts and "\\" not in name
            and value.as_posix() == name and name not in ("", "."), "unsafe archive member")
    return value


def extract_archive(archive, inventory, destination, check=lambda: None):
    """Copy exact archived bytes; never link to either prior data tree."""
    bound(archive, check)
    expected = {row["member"]: row for row in inventory["members"]}
    require(len(expected) == len(inventory["members"]) == inventory["member_count"], "duplicate archive member declaration")
    for name in expected: safe_member(name)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    observed = []
    with tarfile.open(archive["path"], "r|gz") as stream:
        seen = set()
        for item in stream:
            check(); safe_member(item.name)
            require(item.name in expected and item.name not in seen and item.isfile() and not item.issparse(),
                    "unexpected/duplicate/nonregular archive member")
            row = expected[item.name]
            require(item.size == row["source"]["bytes"], "archive member size differs")
            target = destination/item.name; target.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with stream.extractfile(item) as source, target.open("xb") as output:
                while True:
                    check(); block = source.read(1024*1024)
                    if not block: break
                    output.write(block); digest.update(block); count += len(block)
                output.flush(); os.fsync(output.fileno())
            require(count == row["source"]["bytes"] and digest.hexdigest() == row["source"]["sha256"], "extracted bytes differ")
            target.chmod(0o600)  # Only this new owner-writable copy; bytes unchanged.
            require(target.stat().st_nlink == 1 and not target.is_symlink(), "working copy must not alias old files")
            observed.append({"member": item.name, "copy": identity(target, check), "archive_source": row["source"]})
            seen.add(item.name)
    require(seen == set(expected), "missing archive members")
    require(not (destination/"data/campaign-run/.quantum-scheduler-lock").exists(), "old lease cannot enter new working tree")
    return sorted(observed, key=lambda row: row["member"])


def verify_seed(inspector, campaign, prior, protocol, seed, *, complete):
    original = next(row for row in prior["seeds"] if row["seed"] == seed)
    row = inspector.inspect_seed(campaign, original, 300000, protocol, prior["protocol"]["sha256"],
                                 prior["source_sha256"]["benchmarks/h3/quantum_sampling_campaign.py"])
    if complete:
        require(row["completed_sweeps"] == 305000 and row["arrays"]["samples"]["shape"] == [300000, 48]
                and row["arrays"]["diagnostics"]["shape"] == [30000, 48, 7]
                and [x["production_sweeps"] for x in row["stage_receipts"]] == [30000, 100000, 300000],
                "every complete seed requires both300k state and receipt plus full observations")
    else:
        require(row == original and row["completed_sweeps"] == 105000,
                "relocated100k checkpoint/settings/RNG/full payload identity differs")
    return row


def verify_complete(rows):
    require([row["seed"] for row in rows] == list(SEEDS), "complete fixed eight-seed order required")
    for row in rows:
        require(row["completed_sweeps"] == 305000 and row["arrays"]["samples"]["shape"] == [300000, 48]
                and row["arrays"]["diagnostics"]["shape"] == [30000, 48, 7]
                and [x["production_sweeps"] for x in row["stage_receipts"]] == [30000, 100000, 300000],
                "incomplete seed cannot contribute to global completion")
    return {"seeds": list(SEEDS), "additional_sweeps_all8": 1600000,
            "retained_production_sweeps_every_seed": 300000, "retained_scalar_CV_observations": 115200000}


def prepare(context, check):
    proposal, output = context["proposal"], context["output"]
    evidence = proposal["required_successful_prior_archive"]
    copied = extract_archive(evidence["archive"], context["members"], output/"restored", check)
    prior_path = context["campaign"]/"stage-100000.json"
    require(identity(prior_path, check)["sha256"] == proposal["new_isolated_working_copy"]["expected_prior_scheduler_input_sha256"],
            "exact original100k full snapshot required")
    inspector = module_at(INSPECTOR, INSPECTOR_SHA)
    prior = inspector.load_snapshot(prior_path, identity(prior_path)["sha256"])
    require(prior["production_sweeps"] == 100000 and prior["independent_seeds"] == list(SEEDS), "not a full100k scheduler snapshot")
    decision = dict(proposal["decision"])
    decision.pop("completed_cohort_report"); decision.pop("completed_cohort_execution")
    report = output/"restored/data/full-cohort-stage-100000/campaign-comparison.json"
    execution = report.parent/"execution.json"
    decision.update(protocol_sha256=prior["protocol"]["sha256"], completed_cohort_report_path=str(report),
        completed_cohort_report_sha256=identity(report, check)["sha256"],
        completed_cohort_execution_sha256=identity(execution, check)["sha256"])
    require(decision["completed_cohort_report_sha256"] == proposal["decision"]["completed_cohort_report"]["sha256"]
            and decision["completed_cohort_execution_sha256"] == proposal["decision"]["completed_cohort_execution"]["sha256"],
            "copied completed trigger evidence differs")
    decision_path = output/"continuation-decision.json"; write_new(decision_path, decision)
    protocol_path = output/"restored/reference/protocol.json"
    # Direct preflight is read-only. The old scheduler --check-only command would
    # create a directory lease and is never called on an immutable prior root.
    actual_prior, protocol, _, initial = inspector.preflight(context["campaign"], context["library"]["path"],
        protocol_path, prior_path, identity(prior_path)["sha256"], decision_path, identity(decision_path)["sha256"], 300000)
    require(actual_prior == prior and protocol == context["protocol"], "copied preflight protocol/snapshot differs")
    for seed, row in zip(SEEDS, initial):
        check(); require(row == verify_seed(inspector, context["campaign"], prior, protocol, seed, complete=False), "initial seed differs")
    immutable_working = [identity(prior_path), identity(protocol_path), identity(decision_path)]
    for row in prior["seeds"]:
        for stage in row["stage_receipts"]:
            for key in ("receipt", "state"):
                immutable_working.append(identity(context["campaign"]/stage[key]["path"], check))
    registration = {"schema": REG_SCHEMA, "created_utc": now(), "source": identity(__file__), "tests": identity(TESTS),
        "policy": identity(POLICY), "adoption": identity(context["adoption_path"]),
        "campaign": str(context["campaign"]), "output": str(output), "prior_snapshot": identity(prior_path),
        "protocol": identity(protocol_path), "decision": identity(decision_path), "seed_order": list(SEEDS),
        "target_stage": 300000, "initial_seeds": initial, "initial_members": copied,
        "source_guards": context["source_guards"], "immutable_archive_guards": context["immutable_guards"],
        "immutable_working_prefix_guards": immutable_working, "library": context["library"],
        "runtime": context["runtime"], "archive": evidence["archive"], "member_inventory": evidence["member_inventory"],
        "exact_prior_archive_replay": evidence["archive_execution"], "old_evidence_changed": False,
        "restored_lease_present": False, "physical_steps_so_far": 0}
    write_new(output/"registration.json", registration)
    return registration, prior


class AdmissionAlarmMask:
    """Keep owner SIGALRM pending across frozen Popen-to-active registration."""
    def __init__(self):
        self.previous = None

    def acquire(self):
        require(self.previous is None, "nested worker admission alarm mask")
        self.previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})

    def release(self):
        if self.previous is not None:
            old, self.previous = self.previous, None
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


class ResourceWatchdog:
    """Observe free space and only owned PIDs; stop this owner on a breach.

    Memory/free-space are observations, not OS quotas. The extra checkpoint
    buffer allowance triggers a stop before the mandatory free-reserve floor.
    """
    def __init__(self, policy, cpus, *, disk_path=WORK):
        self.policy, self.cpus, self.disk_path = policy, cpus, disk_path
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.owned, self.reason, self.maximum_observed_RSS = {}, None, 0
        self.minimum_observed_free = None
        self.thread = None

    @staticmethod
    def process_observation(pid):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return int(fields[19]), int(fields[21])*os.sysconf("SC_PAGE_SIZE")
        except FileNotFoundError:
            return None

    def event(self, row):
        with self.lock:
            if row["event"] == "worker_started":
                observed = self.process_observation(row["pid"])
                self.owned[row["seed"]] = {"pid": row["pid"], "start_ticks": None if observed is None else observed[0]}
            elif row["event"] == "worker_reaped":
                self.owned.pop(row["seed"], None)

    def observe(self):
        free = shutil.disk_usage(self.disk_path).free
        current = self.process_observation(os.getpid())
        total = 0 if current is None else current[1]
        with self.lock:
            owned = list(self.owned.values())
        for row in owned:
            observed = self.process_observation(row["pid"])
            # A PID/start-time mismatch is never treated as an owned process.
            if observed is not None and observed[0] == row["start_ticks"]:
                total += observed[1]
        self.maximum_observed_RSS = max(self.maximum_observed_RSS, total)
        self.minimum_observed_free = free if self.minimum_observed_free is None else min(self.minimum_observed_free, free)
        storage, limits = self.policy["storage_revision"], self.policy["sampling_envelope"]
        threshold = storage["mandatory_free_reserve_bytes"]+len(self.cpus)*storage["per_active_checkpoint_temporary_buffer_allowance_bytes"]
        if free < threshold:
            return "filesystem free space below reserved floor plus checkpoint buffers"
        if total > limits["planned_total_observed_RSS_limit_bytes"]:
            return "observed owned-process RSS exceeds registered planning envelope"
        if time.process_time() >= limits["owner_observed_CPU_limit_seconds"]:
            return "observed owner CPU budget exhausted"
        return None

    def start(self):
        def watch():
            # Inherited blocked SIGALRM leaves the main thread as the sole
            # recipient, including while its admission mask is held.
            while not self.stop_event.wait(self.policy["sampling_envelope"]["metadata_watchdog_interval_seconds"]):
                try:
                    reason = self.observe()
                except BaseException as error:
                    reason = f"resource watchdog failed: {type(error).__name__}: {error}"
                if reason:
                    self.reason = reason
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            self.thread = threading.Thread(target=watch, daemon=True)
            self.thread.start()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            require(not self.thread.is_alive(), "owned resource watchdog did not stop")

    def receipt(self):
        return {"reason": self.reason, "maximum_observed_owned_RSS_bytes": self.maximum_observed_RSS,
                "minimum_observed_filesystem_free_bytes": self.minimum_observed_free,
                "observation_interval_seconds": self.policy["sampling_envelope"]["metadata_watchdog_interval_seconds"],
                "OS_memory_or_disk_quota_claimed": False}


def dispatch_owned(dispatcher, command_for, verify_result, *, cpus, deadline, attempt,
                   lease, guard, monitor):
    """Extend frozen dispatch only with alarm-safe ownership and observation.

    The caller keeps its work-alarm and deferred stop handlers installed until
    this function returns or reports an unconfirmed drain. Tests supply inert
    commands to this same scheduling layer; the production call fixes one_seed.
    """
    mask = AdmissionAlarmMask()
    old_token = os.environ.get(TOKEN_ENV); os.environ[TOKEN_ENV] = PROCESS_TOKEN
    try:
        monitor.start()
        def command(seed, cpu):
            value = command_for(seed, cpu)
            mask.acquire()
            return value
        def event(row):
            if row["event"] == "worker_started":
                mask.release()  # Frozen dispatch has already recorded active[seed].
            monitor.event(row)
        return dispatcher.dispatch_jobs(SEEDS, command, verify_result, cpus=cpus,
            deadline=deadline, attempt=attempt, lease=lease, guard=guard, event_hook=event)
    finally:
        # Even a failed monitor join must restore the pending alarm and token.
        # An unconfirmed worker drain leaves the caller's lease.release false.
        try:
            monitor.close()
        finally:
            if old_token is None: os.environ.pop(TOKEN_ENV, None)
            else: os.environ[TOKEN_ENV] = old_token
            mask.release()


@contextmanager
def exclusive_owner(output):
    output = Path(output)
    lock = output.parent/("."+output.name+"-continuation-lock")
    lock.mkdir()
    lease = {"release": True, "path": str(lock)}
    try:
        write_new(lock/"owner.json", {"created_utc": now(), "PID": os.getpid(),
                  "process_token": PROCESS_TOKEN, "new_working_root": str(output)})
        yield lease
    finally:
        if lease["release"]:
            (lock/"owner.json").unlink(missing_ok=True)
            lock.rmdir()


def check_worker_claim(registration, registration_path, plan, attempt_path, seed, cpu, bootstrap):
    require(registration.get("schema") == REG_SCHEMA and plan.get("schema") == ATTEMPT_SCHEMA
            and plan.get("registration") == identity(registration_path)
            and plan.get("source") == identity(__file__) and registration.get("source") == identity(__file__),
            "worker registration/source differs")
    require(type(seed) is int and seed in SEEDS and type(cpu) is int and cpu in plan["CPU_affinity"]
            and plan.get("owner_PID") == os.getppid()
            and isinstance(plan.get("owner_process_token"), str) and bool(plan["owner_process_token"])
            and plan["owner_process_token"] == os.environ.get(TOKEN_ENV)
            and plan.get("seed_order") == registration.get("seed_order") == list(SEEDS),
            "private worker requires its current owning coordinator and fixed seed/CPU")
    validate_cpus(plan["CPU_affinity"])
    adoption = read_bound(registration["adoption"])
    require(plan["CPU_affinity"] == adoption["CPU_affinity"]
            and plan.get("fixed_worker_count") == len(plan["CPU_affinity"])
            and plan.get("mid_run_resize") is False
            and plan.get("worker_CPU_hard_limit_seconds") == 21600
            and plan.get("worker_wall_hard_limit_seconds") == 21600
            and math.isfinite(plan["work_deadline_monotonic"])
            and plan["whole_deadline_monotonic"]-plan["work_deadline_monotonic"] == 120,
            "private worker budget/CPU contract differs")
    output = Path(registration["output"])
    require(registration_path == output/"registration.json" and attempt_path == output/"attempt"
            and registration["campaign"] == str(output/"restored/data/campaign-run")
            and plan.get("target_stage") == registration.get("target_stage") == 300000,
            "private worker output routes or stage differ")
    require(bootstrap is not None and bootstrap["installed_before_scientific_imports"] is True
            and bootstrap["alarm_unblocked"] is True and bootstrap["RLIMIT_CPU"] == [21600, 21600]
            and bootstrap["owner_work_deadline"] == plan["work_deadline_monotonic"],
            "independent worker kernel limits must precede scientific imports")


def worker(registration_path, attempt_path, seed, cpu):
    registration_path, attempt_path = Path(registration_path).resolve(), Path(attempt_path).resolve()
    registration, plan = json.loads(registration_path.read_text()), json.loads((attempt_path/"plan.json").read_text())
    check_worker_claim(registration, registration_path, plan, attempt_path, seed, cpu, BOOTSTRAP)
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "private worker needs one-thread environment")
    os.sched_setaffinity(0, {cpu})
    require(os.sched_getaffinity(0) == {cpu}, "worker CPU pin failed")
    validate_runtime(registration["runtime"])
    claim = attempt_path/f"seed-{seed}-once.json"
    write_new(claim, {"created_utc": now(), "seed": seed, "CPU": cpu, "PID": os.getpid(),
        "parent_PID": os.getppid(), "owner_token": plan["owner_process_token"],
        "registration": identity(registration_path), "attempt": identity(attempt_path/"plan.json"),
        "kernel_limits": BOOTSTRAP, "automatic_retry": False})
    guards = registration["source_guards"]+registration["immutable_working_prefix_guards"]
    verify_guards(guards)
    inspector = module_at(INSPECTOR, INSPECTOR_SHA)
    prior = inspector.load_snapshot(registration["prior_snapshot"]["path"], registration["prior_snapshot"]["sha256"])
    protocol = read_bound(registration["protocol"])
    before = verify_seed(inspector, registration["campaign"], prior, protocol, seed, complete=False)
    require(before == next(row for row in registration["initial_seeds"] if row["seed"] == seed), "initial worker state differs")
    inspector.verify_imported_sources({row["path"]: row["sha256"] for row in registration["source_guards"]})
    driver = module_at(DRIVER, DRIVER_SHA)
    job = (Path(registration["campaign"]), Path(registration["library"]["path"]),
           Path(registration["protocol"]["path"]), seed, 300000)
    returned = driver.one_seed(job)
    actual = verify_seed(inspector, registration["campaign"], prior, protocol, seed, complete=True)
    stage_path = Path(registration["campaign"])/f"seed-{seed}/stages/300000/stage.json"
    require(returned == json.loads(stage_path.read_text()), "frozen worker return/committed stage receipt differ")
    verify_guards(guards)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"schema": "pyrpmd.h3.isolated300k-worker-completion.v1", "created_utc": now(),
        "seed": seed, "complete": True, "source": identity(__file__), "registration": identity(registration_path),
        "attempt": identity(attempt_path/"plan.json"), "CPU": cpu, "PID": os.getpid(), "parent_PID": os.getppid(),
        "source_guards_unchanged": True, "entrypoint": "benchmarks.h3.quantum_sampling_campaign.one_seed",
        "entrypoint_return": returned, "checked_seed": actual, "kernel_limits": BOOTSTRAP,
        "worker_CPU_seconds_before_report": usage.ru_utime+usage.ru_stime,
        "worker_peak_RSS_bytes": usage.ru_maxrss*1024, "scientific_analysis_performed": False}
    write_new(attempt_path/f"seed-{seed}-result.json", result)
    return result


def verify_worker_result(registration, prior, attempt, seed):
    path = attempt/f"seed-{seed}-result.json"
    result = json.loads(path.read_text())
    require(result.get("schema") == "pyrpmd.h3.isolated300k-worker-completion.v1"
            and result.get("seed") == seed and result.get("complete") is True
            and result.get("source_guards_unchanged") is True
            and result.get("source") == identity(__file__)
            and result.get("registration") == identity(Path(registration["output"])/"registration.json")
            and result.get("attempt") == identity(attempt/"plan.json")
            and result.get("scientific_analysis_performed") is False,
            "exact complete private worker result required")
    plan = json.loads((attempt/"plan.json").read_text())
    claim = json.loads((attempt/f"seed-{seed}-once.json").read_text())
    require(claim.get("seed") == seed and result.get("CPU") == claim.get("CPU")
            and claim["CPU"] in plan["CPU_affinity"]
            and result.get("PID") == claim.get("PID")
            and result.get("parent_PID") == claim.get("parent_PID") == plan["owner_PID"]
            and claim.get("owner_token") == plan["owner_process_token"]
            and claim.get("registration") == result["registration"]
            and claim.get("attempt") == result["attempt"]
            and result.get("kernel_limits") == claim.get("kernel_limits")
            and claim["kernel_limits"]["RLIMIT_CPU"] == [21600, 21600]
            and claim["kernel_limits"]["owner_work_deadline"] == plan["work_deadline_monotonic"]
            and claim["kernel_limits"]["installed_before_scientific_imports"] is True
            and claim["kernel_limits"]["alarm_unblocked"] is True,
            "worker result/once claim/kernel budget differ")
    inspector = module_at(INSPECTOR, INSPECTOR_SHA)
    protocol = read_bound(registration["protocol"])
    checked = verify_seed(inspector, registration["campaign"], prior, protocol, seed, complete=True)
    require(checked == result["checked_seed"], "worker completion/raw final snapshot differ")
    return {"seed": seed, "worker_result": identity(path), "checked_seed": checked}


def run(adoption_path):
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "one-thread owner environment required")
    # Install the preparation wall bound before archive hashing/import checks.
    # This small helper is stdlib-only; no worker exists during admission.
    alarms = module_at(ALARM, ALARM_SHA)
    def admission_check():
        require(time.monotonic() < ENTRY_MONOTONIC+900, "preparation admission wall budget exhausted")
        require(time.process_time() < 300, "observed preparation admission CPU budget exhausted")
    with alarms.work_alarm(ENTRY_MONOTONIC+900, {}):
        context = load_context(adoption_path, admission_check)
    policy, proposal, output = context["policy"], context["proposal"], context["output"]
    limits, storage = policy["sampling_envelope"], policy["storage_revision"]
    cpus = context["cpus"]
    free_before = shutil.disk_usage(WORK).free
    required_free = storage_admission(storage, context["adoption"], free_before)
    require(not output.exists() and not output.is_symlink(), "new isolated working destination must not exist")
    os.sched_setaffinity(0, set(cpus))
    dispatcher = module_at(DISPATCH, DISPATCH_SHA)
    hard_deadline = ENTRY_MONOTONIC+limits["whole_owner_wall_seconds"]
    work_deadline = hard_deadline-limits["drain_reserve_seconds"]
    monitor = ResourceWatchdog(policy, cpus)
    result, error = {"complete": False, "sampling_analysis_invocations": 0}, None
    outer_stop = None
    def check():
        require(time.monotonic() < work_deadline, "whole owner work deadline exhausted")
        require(time.process_time() < limits["owner_observed_CPU_limit_seconds"], "observed owner CPU budget exhausted")
        require(not monitor.reason, monitor.reason or "resource monitor failed")
        require(outer_stop is None or not outer_stop["reason"], "owner stop signal observed")
    with dispatcher.deferred_stop_signals() as outer_stop, exclusive_owner(output) as lease:
        output.mkdir(exist_ok=False)
        write_new(output/"RUN_ONCE.json", {"created_utc": now(), "source": identity(__file__), "policy": identity(POLICY),
            "adoption": identity(adoption_path), "PID": os.getpid(), "process_token": PROCESS_TOKEN,
            "CPU_affinity": cpus, "free_bytes_at_admission": free_before, "required_free_bytes": required_free,
            "whole_deadline_monotonic": hard_deadline, "automatic_retry_or_extension": False})
        attempt = output/"attempt"; attempt.mkdir()
        try:
            preparation_deadline = min(work_deadline, ENTRY_MONOTONIC+policy["preparation_envelope"]["wall_limit_seconds"])
            with alarms.work_alarm(preparation_deadline, {}):
                def preparation_check():
                    check()
                    require(time.process_time() < policy["preparation_envelope"]["observed_CPU_limit_seconds"],
                            "observed preparation CPU budget exhausted")
                registration, prior = prepare(context, preparation_check)
                verify_guards(context["source_guards"]+context["immutable_guards"], preparation_check)
            plan = {"schema": ATTEMPT_SCHEMA, "created_utc": now(), "registration": identity(output/"registration.json"),
                "source": identity(__file__), "seed_order": list(SEEDS), "target_stage": 300000,
                "owner_PID": os.getpid(), "owner_process_token": PROCESS_TOKEN, "CPU_affinity": cpus,
                "whole_deadline_monotonic": hard_deadline, "work_deadline_monotonic": work_deadline,
                "worker_CPU_hard_limit_seconds": 21600, "worker_wall_hard_limit_seconds": 21600,
                "fixed_worker_count": len(cpus), "mid_run_resize": False}
            write_new(attempt/"plan.json", plan)
            check(); monitor.reason = monitor.observe(); check()
            with alarms.work_alarm(work_deadline, {}):
                def guard():
                    check()
                    verify_guards(context["source_guards"]+registration["immutable_working_prefix_guards"], check)
                def command(seed, cpu):
                    check()
                    return [sys.executable, str(Path(__file__).resolve()), "--_worker",
                            "--registration", str(output/"registration.json"), "--attempt", str(attempt),
                            "--seed", str(seed), "--cpu", str(cpu), "--worker-deadline", repr(work_deadline)]
                dispatch = dispatch_owned(dispatcher, command,
                    lambda seed: verify_worker_result(registration, prior, attempt, seed),
                    cpus=cpus, deadline=work_deadline, attempt=attempt, lease=lease, guard=guard, monitor=monitor)
                require(dispatch["all_workers_drained"] is True and not dispatch["errors"]
                        and not dispatch["unstarted_seeds"] and dispatch["launched_seeds"] == list(SEEDS),
                        "incomplete/failing dispatch cannot publish global completion")
                check()
                rows = [dispatch["results"][seed]["checked_seed"] for seed in SEEDS]
                counts = verify_complete(rows)
                verify_guards(context["source_guards"]+context["immutable_guards"]+registration["immutable_working_prefix_guards"], check)
                inspector = module_at(INSPECTOR, INSPECTOR_SHA)
                for seed, expected in zip(SEEDS, rows):
                    check(); require(verify_seed(inspector, context["campaign"], prior, context["protocol"], seed, complete=True) == expected,
                                     "final complete checkpoint changed after worker verification")
                snapshot = {"schema": "pyrpmd.h3.quantum-umbrella-stage-snapshot.v1", "created_utc": now(),
                    "production_sweeps": 300000, "warmup_sweeps": 5000, "independent_seeds": list(SEEDS),
                    "campaign_directory": str(context["campaign"]), "protocol": prior["protocol"], "schedule": prior["schedule"],
                    "source_sha256": prior["source_sha256"], "model_sha256": prior["model_sha256"], "seeds": rows,
                    "prior_complete100000_archive": registration["archive"], "prior_snapshot_sha256": registration["prior_snapshot"]["sha256"],
                    "physical_converged": False}
                completion = {"schema": COMPLETE_SCHEMA, "created_utc": now(), "production": 300000,
                    "requested_seeds": list(SEEDS), "completed_seeds": list(SEEDS), "all_owned_workers_drained": True,
                    "source_guards_unchanged": True, "prior_prefix_and_state_files_unchanged": True,
                    "registration": identity(output/"registration.json"), "owner_source": identity(__file__),
                    "attempt": identity(attempt/"plan.json"), "archive_ancestry": registration["archive"],
                    "worker_results": [dispatch["results"][seed]["worker_result"] for seed in SEEDS],
                    "snapshot": snapshot, "counts": counts, "scientific_analysis_performed": False,
                    "physical_converged": False, "literature_reproduced": False, "automatic_extension": False}
                check(); write_new(context["campaign"]/"stage-300000.json", completion)
                result.update(complete=True, global_completion=identity(context["campaign"]/"stage-300000.json"),
                    all_owned_workers_drained=True, all8_fixed_work_and_prefix_checks_passed=True,
                    source_guards_unchanged=True, counts=counts)
        except BaseException as caught:
            error = caught
            result["failure"] = {"type": type(caught).__name__, "message": str(caught)}
            result["owned_draining_confirmed"] = lease["release"]
        finally:
            monitor.close()
            usage, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(schema="pyrpmd.h3.isolated300k-owner-execution.v1", created_utc=now(),
                source=identity(__file__), policy=identity(POLICY), elapsed_wall_seconds=time.monotonic()-ENTRY_MONOTONIC,
                own_process_CPU_seconds=usage.ru_utime+usage.ru_stime,
                reaped_children_CPU_seconds=children.ru_utime+children.ru_stime,
                resource_observations=monitor.receipt(), original_data_or_lease_modified=False,
                verified_prior_extraction_modified=False, automatic_retry_or_extension=False,
                scientific_source_writes_by_tool=False, numerical_analysis_performed=False,
                Git_mutated=False, upload=False, permanent_attempt_claim_retained=True)
            write_new(output/"execution.json", result)
    if error is not None:
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--worker-deadline", type=float)
    args = parser.parse_args()
    if args._worker:
        result = worker(args.registration, args.attempt, args.seed, args.cpu)
        print(json.dumps({"seed": result["seed"], "complete": result["complete"]}), flush=True)
    else:
        require(args.authorization is not None, "root authorization required")
        result = run(args.authorization)
        print(json.dumps({"complete": result["complete"], "global_completion": result.get("global_completion")}), flush=True)


if __name__ == "__main__":
    main()
