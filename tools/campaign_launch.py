"""Run validation-campaign jobs inside a lane's CPU allocation, with a ledger.

Several agents share one machine during a validation campaign
(``docs/tasks/campaign-c32/``). Each agent owns a *lane*: a fixed set of
logical CPUs, a core-hour budget and a shared campaign clock. This launcher is
the only supported way to start campaign jobs, so that

* every job is pinned to its lane's CPUs and runs with explicit BLAS/OpenMP
  thread counts, and a lane cannot start more concurrent slots than it owns;
* every job is recorded in an append-only JSONL ledger with its command,
  CPU time, peak memory, exit status and the source identity it ran against
  (git commit and cleanliness of the frozen PyRPMD tree and of the working
  directory, SHA-256 of the compiled extensions, the preregistration file);
* production work cannot start without a committed preregistration file and
  is stopped (SIGTERM, then SIGKILL) at the campaign production deadline, so
  that unfinished calculations stop at a checkpoint instead of overrunning.

Sub-commands::

    check   validate a lane plan (and, with --machine, this machine's CPUs)
    start   create the campaign clock once, binding the frozen commit
    run     run one job inside a lane: ``run ... -- COMMAND [ARGS...]``
    status  summarize ledger usage against each lane's budget

``check`` uses only the standard library and works on every platform. ``run``
requires Linux (``os.sched_setaffinity`` and ``fcntl``). The launcher neither
installs software nor modifies the source tree.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PLAN_SCHEMA = "pyrpmd.campaign-plan.v1"
START_SCHEMA = "pyrpmd.campaign-start.v1"
LEDGER_SCHEMA = "pyrpmd.campaign-ledger.v1"
KINDS = ("pilot", "production", "analysis", "test", "benchmark")
# Kinds that consume scientific budget and therefore stop at the deadline.
DEADLINE_KINDS = frozenset({"pilot", "production", "benchmark"})
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
TIMELINE_KEYS = ("phase0_end", "production_deadline", "campaign_end")
EXIT_REFUSED = 3


class PlanError(ValueError):
    """The plan or a launch request violates a campaign rule."""


# ---------------------------------------------------------------------------
# Plan validation (standard library only, platform independent)


def load_plan(path: str | os.PathLike) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        plan = json.load(stream)
    plan["_path"] = str(path.resolve())
    return plan


def plan_errors(plan: dict, *, plan_dir: Path | None = None) -> list[str]:
    """Return every rule violation in ``plan``; an empty list means valid."""
    errors: list[str] = []
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append(f"schema must be {PLAN_SCHEMA!r}")
    machine = plan.get("machine", {})
    ncpu = machine.get("logical_cpus")
    if not isinstance(ncpu, int) or isinstance(ncpu, bool) or ncpu < 1:
        errors.append("machine.logical_cpus must be a positive integer")
        ncpu = None
    timeline = plan.get("timeline_hours", {})
    values = []
    for key in TIMELINE_KEYS:
        value = timeline.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(f"timeline_hours.{key} must be a positive number")
        else:
            values.append(value)
    if len(values) == len(TIMELINE_KEYS) and values != sorted(values):
        errors.append("timeline_hours must satisfy phase0_end <= production_deadline <= campaign_end")
    window = None
    if len(values) == len(TIMELINE_KEYS):
        window = timeline["production_deadline"] - timeline["phase0_end"]

    lanes = plan.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        return errors + ["lanes must be a nonempty list"]
    ids = [lane.get("id") for lane in lanes]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        errors.append("lane ids must be unique nonempty strings")
    owner: dict[int, str] = {}
    phase0_lanes = []
    for lane in lanes:
        lid = lane.get("id", "?")
        cpus = lane.get("cpus")
        if (not isinstance(cpus, list) or not cpus
                or not all(isinstance(c, int) and not isinstance(c, bool) for c in cpus)):
            errors.append(f"{lid}: cpus must be a nonempty list of integers")
            continue
        if len(set(cpus)) != len(cpus):
            errors.append(f"{lid}: duplicate CPU ids")
        for cpu in cpus:
            if cpu < 0 or (ncpu is not None and cpu >= ncpu):
                errors.append(f"{lid}: CPU {cpu} outside 0..{(ncpu or 1) - 1}")
            if cpu in owner and owner[cpu] != lid:
                errors.append(f"{lid}: CPU {cpu} already belongs to {owner[cpu]}")
            owner.setdefault(cpu, lid)
        budget = lane.get("core_hours")
        planned = lane.get("planned_core_hours")
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
            errors.append(f"{lid}: core_hours must be positive")
        elif window is not None and budget > len(cpus) * window + 1e-9:
            errors.append(
                f"{lid}: core_hours {budget} exceeds {len(cpus)} CPUs x {window} h production window")
        if not isinstance(planned, (int, float)) or isinstance(planned, bool) or planned <= 0:
            errors.append(f"{lid}: planned_core_hours must be positive")
        elif isinstance(budget, (int, float)) and planned > budget:
            errors.append(f"{lid}: planned_core_hours exceeds core_hours")
        pilot = lane.get("pilot_fraction", 0.1)
        if not isinstance(pilot, (int, float)) or not 0 < pilot <= 0.25:
            errors.append(f"{lid}: pilot_fraction must be in (0, 0.25]")
        if lane.get("phase0_all_cpus"):
            phase0_lanes.append(lid)
        card = lane.get("card")
        if not isinstance(card, str) or not card:
            errors.append(f"{lid}: card is required")
        elif plan_dir is not None and not (plan_dir / card).is_file():
            errors.append(f"{lid}: card {card} does not exist")
        for key in ("title", "branch", "record"):
            if not isinstance(lane.get(key), str) or not lane.get(key):
                errors.append(f"{lid}: {key} is required")
    if ncpu is not None and len(owner) > ncpu:
        errors.append("lanes use more CPUs than the machine has")
    if len(phase0_lanes) != 1:
        errors.append("exactly one lane must set phase0_all_cpus (the gate lane)")
    reviewers = {}
    for lane in lanes:
        target = lane.get("cross_review")
        if target not in ids:
            errors.append(f"{lane.get('id')}: cross_review must name another lane")
        elif target == lane.get("id"):
            errors.append(f"{lane.get('id')}: a lane cannot review itself")
        else:
            reviewers.setdefault(target, []).append(lane.get("id"))
    for lid in ids:
        if len(reviewers.get(lid, [])) != 1:
            errors.append(f"{lid}: must be reviewed by exactly one other lane")
    return errors


def lane_by_id(plan: dict, lane_id: str) -> dict:
    for lane in plan["lanes"]:
        if lane["id"] == lane_id:
            return lane
    raise PlanError(f"unknown lane {lane_id!r}; plan lanes: {[l['id'] for l in plan['lanes']]}")


def machine_errors(plan: dict) -> list[str]:
    """Check that this process may use every planned CPU (Linux only)."""
    if not hasattr(os, "sched_getaffinity"):
        return ["this platform cannot report CPU affinity; run the campaign on Linux"]
    available = os.sched_getaffinity(0)
    needed = {cpu for lane in plan["lanes"] for cpu in lane["cpus"]}
    missing = sorted(needed - available)
    errors = [f"CPUs {missing} are not available to this process"] if missing else []
    siblings = _smt_siblings(needed & available)
    if siblings:
        errors.append(
            "planned CPUs share physical cores (SMT siblings): "
            + ", ".join(f"{a}+{b}" for a, b in siblings)
            + "; remap lanes.json to one logical CPU per physical core")
    return errors


def _smt_siblings(cpus: set[int]) -> list[tuple[int, int]]:
    pairs = set()
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            text = path.read_text().strip()
        except OSError:
            continue
        for other in _parse_cpu_list(text):
            if other != cpu and other in cpus:
                pairs.add(tuple(sorted((cpu, other))))
    return sorted(pairs)


def _parse_cpu_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return out


# ---------------------------------------------------------------------------
# Campaign clock and source identity


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def start_path(root: Path) -> Path:
    return root / "campaign_start.json"


def read_start(root: Path) -> dict:
    path = start_path(root)
    if not path.is_file():
        raise PlanError(f"campaign clock {path} does not exist; the gate lane runs `start` first")
    with path.open(encoding="utf-8") as stream:
        start = json.load(stream)
    if start.get("schema") != START_SCHEMA:
        raise PlanError(f"{path} has an unexpected schema")
    return start


def elapsed_hours(start: dict, now: datetime | None = None) -> float:
    began = datetime.fromisoformat(start["start_utc"])
    return ((now or _utc_now()) - began).total_seconds() / 3600.0


def deadline_utc(start: dict, plan: dict, key: str) -> datetime:
    began = datetime.fromisoformat(start["start_utc"])
    return began + timedelta(hours=float(plan["timeline_hours"][key]))


def _git(directory: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True,
                                text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_identity(directory: Path) -> dict:
    top = _git(directory, "rev-parse", "--show-toplevel")
    if top is None:
        return {"repository": None}
    status = _git(Path(top), "status", "--porcelain", "--untracked-files=normal")
    return {
        "repository": top,
        "commit": _git(Path(top), "rev-parse", "HEAD"),
        "clean": status == "",
        "status_lines": None if status is None else len(status.splitlines()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pyrpmd_identity() -> dict:
    """Locate the PyRPMD this interpreter imports, without importing it."""
    spec = importlib.util.find_spec("pyrpmd")
    if spec is None or not spec.origin:
        return {"package": None}
    package = Path(spec.origin).resolve().parent
    extensions = {}
    for path in sorted(package.glob("_*.so")) + sorted(package.glob("_*.pyd")):
        extensions[path.name] = _sha256(path)
    return {"package": str(package), "extensions": extensions, "git": git_identity(package)}


def prereg_identity(path: str | None, cwd: Path) -> dict | None:
    if path is None:
        return None
    file = (cwd / path).resolve()
    if not file.is_file():
        raise PlanError(f"preregistration file {file} does not exist")
    top = _git(file.parent, "rev-parse", "--show-toplevel")
    if top is None:
        raise PlanError(f"preregistration file {file} is not inside a git worktree")
    tracked = _git(Path(top), "ls-files", "--error-unmatch", str(file)) is not None
    changed = _git(Path(top), "status", "--porcelain", "--", str(file))
    if not tracked or changed:
        raise PlanError(f"preregistration file {file} must be committed before production")
    return {"path": str(file), "sha256": _sha256(file),
            "commit": _git(Path(top), "log", "-n", "1", "--format=%H", "--", str(file))}


# ---------------------------------------------------------------------------
# Lane slot accounting


@contextmanager
def _locked(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_runs(lane_dir: Path) -> list[dict]:
    runs = []
    for path in sorted((lane_dir / "active").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record["_path"] = str(path)
        record["alive"] = _alive(int(record.get("launcher_pid", -1)))
        runs.append(record)
    return runs


def _append_ledger(lane_dir: Path, record: dict) -> None:
    lane_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with (lane_dir / "ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_ledger(lane_dir: Path) -> list[dict]:
    path = lane_dir / "ledger.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Commands


def cmd_check(args) -> int:
    plan = load_plan(args.plan)
    errors = plan_errors(plan, plan_dir=Path(args.plan).resolve().parent)
    if not errors and args.machine:
        errors = machine_errors(plan)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    used = sum(len(lane["cpus"]) for lane in plan["lanes"])
    budget = sum(lane["core_hours"] for lane in plan["lanes"])
    planned = sum(lane["planned_core_hours"] for lane in plan["lanes"])
    print(f"plan ok: {len(plan['lanes'])} lanes, {used}/{plan['machine']['logical_cpus']} CPUs, "
          f"{planned:g} planned / {budget:g} budgeted core-hours")
    return 0


def cmd_start(args) -> int:
    plan = load_plan(args.plan)
    errors = plan_errors(plan, plan_dir=Path(args.plan).resolve().parent)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen = None if args.frozen_commit == "none" else args.frozen_commit
    identity = pyrpmd_identity()
    if frozen is not None:
        git = identity.get("git", {})
        if git.get("commit") != frozen or not git.get("clean"):
            print(f"error: imported PyRPMD tree {identity.get('package')} is at {git.get('commit')} "
                  f"(clean={git.get('clean')}), not a clean checkout of {frozen}", file=sys.stderr)
            return 1
    record = {
        "schema": START_SCHEMA,
        "campaign": plan.get("campaign"),
        "start_utc": _iso(_utc_now()),
        "frozen_commit": frozen,
        "plan_sha256": _sha256(Path(args.plan)),
        "pyrpmd": identity,
        "python": sys.executable,
        "host": os.uname().nodename if hasattr(os, "uname") else None,
    }
    path = start_path(root)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        print(f"error: {path} already exists; the campaign clock is started only once", file=sys.stderr)
        return 1
    print(json.dumps({key: record[key] for key in ("campaign", "start_utc", "frozen_commit")}))
    return 0


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _launch_checks(args, plan, start, lane, now) -> tuple[list[int], dict | None]:
    """Validate a launch request; return the job's CPUs and preregistration identity."""
    hours = elapsed_hours(start, now)
    timeline = plan["timeline_hours"]
    if hours >= timeline["campaign_end"]:
        raise PlanError("the campaign has ended")
    if args.kind in DEADLINE_KINDS and hours >= timeline["production_deadline"]:
        raise PlanError(f"{args.kind} jobs stop at the production deadline "
                        f"({timeline['production_deadline']} h); only analysis/test remain")
    phase0 = hours < timeline["phase0_end"]
    if phase0 and not lane.get("phase0_all_cpus"):
        raise PlanError(f"Phase 0 lasts {timeline['phase0_end']} h and belongs to the gate lane; "
                        "read the card and write the preregistration meanwhile")
    if args.kind == "production" and args.prereg is None:
        raise PlanError("production jobs need --prereg pointing to a committed preregistration file")
    if phase0 and lane.get("phase0_all_cpus"):
        cpus = sorted({cpu for item in plan["lanes"] for cpu in item["cpus"]})
    else:
        cpus = list(lane["cpus"])
    if args.slots < 1 or args.threads < 1:
        raise PlanError("--slots and --threads must be positive")
    if args.threads > args.slots:
        raise PlanError("--threads cannot exceed --slots (one slot per running thread)")
    if args.slots > len(cpus):
        raise PlanError(f"lane {lane['id']} owns {len(cpus)} CPUs; cannot run {args.slots} slots")
    if not hasattr(os, "sched_setaffinity"):
        raise PlanError("`run` needs Linux CPU affinity (os.sched_setaffinity)")
    missing = sorted(set(cpus) - os.sched_getaffinity(0))
    if missing:
        raise PlanError(f"CPUs {missing} are not available to this process")
    frozen = start.get("frozen_commit")
    if frozen is not None:
        identity = pyrpmd_identity()
        git = identity.get("git", {})
        if git.get("commit") != frozen or not git.get("clean"):
            raise PlanError(f"imported PyRPMD is at {git.get('commit')} (clean={git.get('clean')}), "
                            f"not the frozen commit {frozen}; use the campaign environment")
        recorded = start.get("pyrpmd", {})
        if args.alternate_build:
            if args.kind not in ("test", "benchmark"):
                raise PlanError("--alternate-build is only for test/benchmark jobs (e.g. the NumPy or FMA build)")
        elif recorded.get("package") and identity.get("package") != recorded["package"]:
            raise PlanError(f"imported PyRPMD {identity.get('package')} is not the campaign build "
                            f"{recorded['package']}; check PYTHONPATH and the interpreter")
        elif recorded.get("extensions") is not None and identity.get("extensions") != recorded["extensions"]:
            raise PlanError("compiled PyRPMD extensions differ from those recorded at the campaign start")
    prereg = prereg_identity(args.prereg, Path.cwd())
    if args.kind == "production":
        cwd_git = git_identity(Path.cwd())
        if cwd_git.get("repository") is not None and not cwd_git.get("clean"):
            raise PlanError("production must start from a clean lane worktree; commit drivers and inputs first")
    return cpus, prereg


def cmd_run(args) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        return _refuse("no command given; use `run ... -- COMMAND [ARGS...]`")
    plan = load_plan(args.plan)
    errors = plan_errors(plan, plan_dir=Path(args.plan).resolve().parent)
    if errors:
        return _refuse("invalid plan: " + "; ".join(errors))
    root = Path(args.root).resolve()
    now = _utc_now()
    try:
        start = read_start(root)
        lane = lane_by_id(plan, args.lane)
        cpus, prereg = _launch_checks(args, plan, start, lane, now)
    except PlanError as error:
        return _refuse(str(error))

    lane_dir = root / lane["id"]
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}-{_safe(args.name)}"
    env = dict(os.environ)
    for name in THREAD_VARIABLES:
        env[name] = str(args.threads)
    bindir = str(Path(sys.executable).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["C32_LANE"] = lane["id"]
    env["C32_RUN_ID"] = run_id
    env["C32_LANE_CPUS"] = ",".join(map(str, cpus))
    deadline = None
    if args.kind in DEADLINE_KINDS:
        deadline = deadline_utc(start, plan, "production_deadline")
    record = {
        "schema": LEDGER_SCHEMA,
        "event": "start",
        "run_id": run_id,
        "lane": lane["id"],
        "name": args.name,
        "kind": args.kind,
        "argv": command,
        "cwd": str(Path.cwd()),
        "cpus": cpus,
        "slots": args.slots,
        "threads": args.threads,
        "start_utc": _iso(now),
        "campaign_hour": round(elapsed_hours(start, now), 4),
        "deadline_utc": None if deadline is None else _iso(deadline),
        "python": sys.executable,
        "pyrpmd": pyrpmd_identity(),
        "cwd_git": git_identity(Path.cwd()),
        "prereg": prereg,
        "note": args.note,
        "alternate_build": bool(args.alternate_build),
        "stdout": str(lane_dir / "logs" / f"{run_id}.out"),
        "stderr": str(lane_dir / "logs" / f"{run_id}.err"),
    }
    if args.dry_run:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    active_file = lane_dir / "active" / f"{run_id}.json"
    waited = 0.0
    while True:
        with _locked(lane_dir / ".lock"):
            busy = sum(int(r.get("slots", 1)) for r in active_runs(lane_dir) if r["alive"])
            if busy + args.slots <= len(cpus):
                active_file.parent.mkdir(parents=True, exist_ok=True)
                active_file.write_text(json.dumps({"run_id": run_id, "launcher_pid": os.getpid(),
                                                   "slots": args.slots, "name": args.name}),
                                       encoding="utf-8")
                break
        if not args.wait:
            return _refuse(f"lane {lane['id']} has {busy} of {len(cpus)} slots busy; "
                           f"use --wait to queue this {args.slots}-slot job")
        if deadline is not None and _utc_now() >= deadline:
            return _refuse("production deadline reached while waiting for a free slot")
        time.sleep(args.poll_seconds)
        waited += args.poll_seconds

    record["queued_seconds"] = round(waited, 3)
    _append_ledger(lane_dir, record)
    (lane_dir / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[campaign] {run_id} on CPUs {record['cpus']} -> {record['stdout']}", file=sys.stderr)
    returncode, usage, wall, stopped = _supervise(command, env, cpus, record, deadline, args.grace_seconds)
    end = {
        "schema": LEDGER_SCHEMA,
        "event": "end",
        "run_id": run_id,
        "lane": lane["id"],
        "name": args.name,
        "kind": args.kind,
        "slots": args.slots,
        "end_utc": _iso(_utc_now()),
        "wall_seconds": round(wall, 3),
        "user_cpu_seconds": round(usage.ru_utime, 3) if usage else None,
        "system_cpu_seconds": round(usage.ru_stime, 3) if usage else None,
        "max_rss_kib": int(usage.ru_maxrss) if usage else None,
        "returncode": returncode,
        "stopped_at_deadline": stopped,
    }
    _append_ledger(lane_dir, end)
    try:
        active_file.unlink()
    except FileNotFoundError:
        pass
    return returncode


def _safe(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)
    return keep.strip("-")[:80] or "job"


def _supervise(command, env, cpus, record, deadline, grace):
    cpuset = set(cpus)

    def pin():
        os.sched_setaffinity(0, cpuset)

    began = time.monotonic()
    with open(record["stdout"], "wb") as out, open(record["stderr"], "wb") as err:
        try:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err,
                                     preexec_fn=pin, start_new_session=True)
        except OSError as error:
            err.write(f"[campaign] could not start: {error}\n".encode())
            return 127, None, time.monotonic() - began, False

        forwarded = {}

        def forward(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            forwarded[signum] = signal.signal(signum, forward)
        stopped = False
        kill_after = None
        try:
            # The supervisor is the only place that reaps the child, so the
            # rusage of the whole job (including waited-for descendants) is kept.
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid == child.pid:
                    break
                if deadline is not None and not stopped and _utc_now() >= deadline:
                    stopped = True
                    _signal_group(child.pid, signal.SIGTERM)
                    kill_after = time.monotonic() + grace
                elif kill_after is not None and time.monotonic() >= kill_after:
                    _signal_group(child.pid, signal.SIGKILL)
                    kill_after = None
                time.sleep(0.2)
        finally:
            for signum, handler in forwarded.items():
                signal.signal(signum, handler)
    returncode = os.waitstatus_to_exitcode(status)
    child.returncode = returncode
    return returncode, usage, time.monotonic() - began, stopped


def _signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def lane_usage(plan: dict, root: Path) -> list[dict]:
    rows = []
    for lane in plan["lanes"]:
        lane_dir = root / lane["id"]
        records = read_ledger(lane_dir)
        starts = {r["run_id"]: r for r in records if r.get("event") == "start"}
        ends = [r for r in records if r.get("event") == "end"]
        cpu_hours = slot_hours = pilot_slot_hours = 0.0
        failed = deadline_stops = 0
        for end in ends:
            cpu = (end.get("user_cpu_seconds") or 0.0) + (end.get("system_cpu_seconds") or 0.0)
            cpu_hours += cpu / 3600.0
            hours = end.get("wall_seconds", 0.0) * int(end.get("slots", 1)) / 3600.0
            slot_hours += hours
            if end.get("kind") == "pilot":
                pilot_slot_hours += hours
            failed += int(end.get("returncode", 0) != 0)
            deadline_stops += int(bool(end.get("stopped_at_deadline")))
        finished = {r["run_id"] for r in ends}
        unfinished = [run_id for run_id in starts if run_id not in finished]
        active = active_runs(lane_dir) if lane_dir.is_dir() else []
        alive = {r["run_id"] for r in active if r["alive"]}
        rows.append({
            "lane": lane["id"],
            "cpus": len(lane["cpus"]),
            "core_hours_budget": lane["core_hours"],
            "planned_core_hours": lane["planned_core_hours"],
            "cpu_hours_used": round(cpu_hours, 3),
            "slot_hours_used": round(slot_hours, 3),
            "pilot_slot_hours": round(pilot_slot_hours, 3),
            "pilot_limit_hours": round(lane.get("pilot_fraction", 0.1) * lane["core_hours"], 3),
            "runs_finished": len(ends),
            "runs_failed": failed,
            "runs_stopped_at_deadline": deadline_stops,
            "runs_active": sorted(alive),
            "runs_lost": sorted(run_id for run_id in unfinished if run_id not in alive),
        })
    return rows


def cmd_status(args) -> int:
    plan = load_plan(args.plan)
    root = Path(args.root).resolve()
    rows = lane_usage(plan, root)
    if args.lane:
        rows = [row for row in rows if row["lane"] == args.lane]
    clock = None
    try:
        clock = round(elapsed_hours(read_start(root)), 3)
    except PlanError:
        pass
    if args.json:
        print(json.dumps({"campaign_hour": clock, "lanes": rows}, indent=2, sort_keys=True))
        return 0
    print(f"campaign hour: {clock}")
    print(f"{'lane':<5}{'cpus':>5}{'slot-h':>9}{'cpu-h':>9}{'plan':>7}{'budget':>8}"
          f"{'pilot-h':>9}{'fail':>5}{'lost':>5}{'act':>5}")
    for row in rows:
        warn = " OVER" if row["slot_hours_used"] > row["core_hours_budget"] else ""
        if row["pilot_slot_hours"] > row["pilot_limit_hours"]:
            warn += " PILOT>LIMIT"
        print(f"{row['lane']:<5}{row['cpus']:>5}{row['slot_hours_used']:>9.1f}{row['cpu_hours_used']:>9.1f}"
              f"{row['planned_core_hours']:>7g}{row['core_hours_budget']:>8g}{row['pilot_slot_hours']:>9.1f}"
              f"{row['runs_failed']:>5}{len(row['runs_lost']):>5}{len(row['runs_active']):>5}{warn}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command_name", required=True)

    check = commands.add_parser("check", help="Validate a lane plan")
    check.add_argument("--plan", required=True)
    check.add_argument("--machine", action="store_true", help="Also check CPU availability and SMT siblings")
    check.set_defaults(handler=cmd_check)

    start = commands.add_parser("start", help="Start the campaign clock (once)")
    start.add_argument("--plan", required=True)
    start.add_argument("--root", required=True, help="Campaign data root ($C32_DATA)")
    start.add_argument("--frozen-commit", required=True,
                       help="Commit the imported PyRPMD tree must match, or 'none' to disable the check")
    start.set_defaults(handler=cmd_start)

    run = commands.add_parser("run", help="Run one job inside a lane: run [options] -- COMMAND...")
    run.add_argument("--plan", required=True)
    run.add_argument("--root", required=True, help="Campaign data root ($C32_DATA)")
    run.add_argument("--lane", required=True)
    run.add_argument("--name", required=True, help="Short job label recorded in the ledger")
    run.add_argument("--kind", choices=KINDS, required=True)
    run.add_argument("--slots", type=int, default=None,
                     help="CPUs the job occupies (workers x threads); default --threads")
    run.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP threads per process (default 1)")
    run.add_argument("--prereg", help="Committed preregistration file (required for production)")
    run.add_argument("--note", help="Free-text note recorded in the ledger")
    run.add_argument("--alternate-build", action="store_true",
                     help="Test/benchmark only: allow a different build of the same frozen commit")
    run.add_argument("--wait", action="store_true", help="Queue until enough lane slots are free")
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--grace-seconds", type=float, default=120.0,
                     help="Seconds between SIGTERM and SIGKILL at the deadline (default 120)")
    run.add_argument("--dry-run", action="store_true", help="Print the resolved launch record only")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=cmd_run)

    status = commands.add_parser("status", help="Summarize lane usage")
    status.add_argument("--plan", required=True)
    status.add_argument("--root", required=True)
    status.add_argument("--lane")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command_name", None) == "run" and args.slots is None:
        args.slots = args.threads
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
