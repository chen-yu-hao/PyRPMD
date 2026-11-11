"""Coordinator-owned, advisory progress snapshots and JSON-lines run events.

No MPI calls are made from the heartbeat thread. Numerical state, RNG streams,
checkpoint formats and restart identities are not changed by monitoring.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from .io import write_json
from .progress_readers import CheckpointProgress


SCHEMA = "pyrpmd.progress"
TERMINAL = frozenset(("completed", "failed", "interrupted", "partial"))


def _coordinator():
    # Evaluate on the calling thread, before starting the reporter. A forked or
    # spawned local worker must not mistake its MPI-free environment for rank 0.
    if multiprocessing.parent_process() is not None:
        return False
    from .parallel import is_root
    return is_root()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _interval(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("progress interval must be a finite positive number")
    return value


class RunProgress:
    """Use AFTER output initialization succeeds, around workflow execution.

    Only the coordinator touches progress.json/run.log.jsonl or emits events.
    Observability failures are reported once and never replace a science error.
    ``result`` may be set before exit to preserve an explicitly partial result.
    """

    def __init__(self, directory, config, plan=(), *, interval=2.0, checkpoint_directory=None):
        self.directory = Path(directory)
        self.config = config
        self.reader = CheckpointProgress(self.directory, config, plan, checkpoint_directory=checkpoint_directory)
        self.interval = _interval(interval)
        self.result = None
        self.enabled = False
        self._owner = os.getpid()
        self._stop = threading.Event()
        self._thread = None
        self._log = None
        self._warned = False
        self._previous = None
        self._last_event = 0.0
        self._baseline = None
        self._baseline_stage = None
        self._baseline_time = None

    def __enter__(self):
        self.enabled = _coordinator() and self.directory.is_dir()
        if not self.enabled:
            return self
        self.run_id = uuid.uuid4().hex
        self.started_at = _utc()
        self._start = time.monotonic()
        try:
            self._log = (self.directory / "run.log.jsonl").open("a", encoding="utf-8")
        except OSError as error:
            self._warning(error)
        self._publish("running", event="run.started")
        self._thread = threading.Thread(target=self._heartbeat, name="pyrpmd-progress", daemon=True)
        try:
            self._thread.start()
        except RuntimeError as error:
            self._thread = None
            self._warning(error)
        return self

    def _warning(self, error):
        if self._warned:
            return
        self._warned = True
        try:
            print(json.dumps({"event": "progress.warning", "time": _utc(),
                              "message": f"Progress reporting unavailable: {error}"}, allow_nan=False),
                  file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._publish("running")

    def _publish(self, status, *, event=None, error=None):
        if not self.enabled or os.getpid() != self._owner:
            return
        try:
            snapshot = self.reader.snapshot()
            now = time.monotonic()
            stage = snapshot["stage"]
            stage_record = next((item for item in snapshot["stages"] if item["stage"] == stage), None)
            done = stage_record["completed_steps"] if stage_record else None
            total = stage_record["total_steps"] if stage_record else None
            if (stage != self._baseline_stage or self._baseline_time is None
                    or (done is not None and (self._baseline is None or done < self._baseline))):
                self._baseline_stage, self._baseline = stage, done
                self._baseline_time = now
            advanced = None if done is None or self._baseline is None else done - self._baseline
            elapsed = now - self._baseline_time
            eta = None
            if (status == "running" and advanced is not None and advanced > 0
                    and elapsed > 0 and total is not None and total > done):
                eta = (total - done) * elapsed / advanced
            if status == "completed":
                eta = 0.0
            snapshot.update({"schema": SCHEMA, "schema_version": 1,
                             "run_id": self.run_id, "task": self.config.get("task"),
                             "output": str(self.directory.resolve()), "status": status,
                             "started_at": self.started_at, "updated_at": _utc(),
                             "updated_unix": time.time(), "elapsed_seconds": max(0.0, now - self._start),
                             "heartbeat_interval_seconds": self.interval,
                             "eta_seconds": eta, "eta_scope": "current_stage",
                             "resumed": bool(self.config.get("resume", False)),
                             "pid": self._owner})
            if error is not None:
                snapshot["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
            write_json(self.directory / "progress.json", snapshot)
            signature = (status, stage, snapshot["completed_steps"], snapshot["total_steps"],
                         tuple(snapshot["warnings"]))
            if event or (signature != self._previous and now - self._last_event >= self.interval):
                self._event(event or "run.progress", snapshot)
                self._last_event = now
                self._previous = signature
        except Exception as failure:
            self._warning(failure)

    def _event(self, event, snapshot):
        level = "ERROR" if snapshot["status"] == "failed" else (
            "WARNING" if snapshot["status"] in ("interrupted", "partial") else "INFO")
        record = {"event": event, "level": level, **snapshot}
        text = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if self._log is not None:
            try:
                self._log.write(text + "\n")
                self._log.flush()
            except (OSError, ValueError) as error:
                self._warning(error)
        try:
            print(text, file=sys.stderr, flush=True)
        except (OSError, ValueError) as error:
            self._warning(error)

    def __exit__(self, exc_type, error, traceback):
        if not self.enabled or os.getpid() != self._owner:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        status = "completed"
        if exc_type is not None:
            status = "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
        elif isinstance(self.result, dict) and self.result.get("status") in (
                "partial", "running", "collecting_flux", "sampling_interfaces"):
            status = "partial"
        self._publish(status, event=f"run.{status}", error=error)
        if self._log is not None:
            try:
                self._log.close()
            except OSError as failure:
                self._warning(failure)
        return False


def read_status(directory):
    """Read a snapshot without modifying the run or initializing its potential."""
    directory = Path(directory)
    path = directory / "progress.json"
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"No progress.json in {directory}; the run has not started or predates progress reporting") from error
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1):
        raise ValueError(f"Unsupported progress schema in {path}")
    try:
        timestamp = float(value["updated_unix"])
        interval = _interval(value["heartbeat_interval_seconds"])
        if not math.isfinite(timestamp):
            raise ValueError("invalid heartbeat timestamp")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid progress heartbeat in {path}") from error
    age = max(0.0, time.time() - timestamp)
    value["heartbeat_age_seconds"] = age
    value["stale"] = value.get("status") not in TERMINAL and age > max(30.0, 3 * interval)
    if value["stale"]:
        value["eta_seconds"] = None
        value["heartbeat_note"] = "Heartbeat is stale; this does not prove that a remote job has stopped"
    return value


def _line(value):
    done, total, percent = (value.get(key) for key in ("completed_steps", "total_steps", "percent"))
    counter = f"{done if done is not None else '?'} / {total if total is not None else '?'} steps"
    fraction = "?%" if percent is None else f"{percent:.1f}%"
    eta = value.get("eta_seconds")
    remaining = "unknown" if eta is None else f"{eta:.0f}s"
    stale = " [stale heartbeat]" if value.get("stale") else ""
    return f"{value['status']} | {value['stage']} | {counter} | {fraction} | stage ETA {remaining}{stale}"


def status_command(args):
    interval = _interval(args.interval)
    if not _coordinator():
        return 0
    tty = args.watch and not args.json and sys.stdout.isatty()
    try:
        while True:
            value = read_status(args.directory)
            if tty:
                print("\r\033[2K" + _line(value), end="", flush=True)
            else:
                print(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 indent=None if args.watch else 2), flush=True)
            if not args.watch or value.get("status") in TERMINAL:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 130
    finally:
        if tty:
            print(flush=True)


def register_parser(commands):
    parser = commands.add_parser("status", help="Read a run's committed progress without loading its PES")
    parser.add_argument("directory", metavar="RUN_DIR")
    parser.add_argument("--watch", action="store_true", help="Refresh until completion, failure, interruption or Ctrl-C")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2)")
    parser.add_argument("--json", action="store_true", help="Use JSON lines instead of a terminal watch display")
    parser.set_defaults(_status_handler=status_command)
    return parser
