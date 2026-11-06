"""Campaign lane plan and pinned launcher (docs/tasks/campaign-c32)."""

from __future__ import annotations

import copy
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "tasks" / "campaign-c32" / "lanes.json"
_SPEC = importlib.util.spec_from_file_location("campaign_launch", ROOT / "tools" / "campaign_launch.py")
launch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launch)

LINUX = hasattr(os, "sched_setaffinity") and sys.platform.startswith("linux")
needs_linux = pytest.mark.skipif(not LINUX, reason="the launcher's run command needs Linux CPU affinity")


def test_repository_plan_is_valid_and_uses_the_whole_machine():
    plan = launch.load_plan(PLAN)
    assert launch.plan_errors(plan, plan_dir=PLAN.parent) == []
    cpus = sorted(cpu for lane in plan["lanes"] for cpu in lane["cpus"])
    assert cpus == list(range(plan["machine"]["logical_cpus"]))
    window = plan["timeline_hours"]["production_deadline"] - plan["timeline_hours"]["phase0_end"]
    assert sum(lane["core_hours"] for lane in plan["lanes"]) <= len(cpus) * window
    assert launch.main(["check", "--plan", str(PLAN)]) == 0


def test_every_card_names_its_lane_cpus_and_branch():
    plan = launch.load_plan(PLAN)
    for lane in plan["lanes"]:
        text = (PLAN.parent / lane["card"]).read_text(encoding="utf-8")
        assert text.startswith("---\norphan: true\n---\n"), lane["card"]
        assert lane["branch"] in text, (lane["card"], lane["branch"])
        assert f"--lane {lane['id']}" in text, lane["card"]
        first, last = lane["cpus"][0], lane["cpus"][-1]
        assert f"{first}–{last}" in text or f"{first}-{last}" in text, lane["card"]


def _base_plan():
    return copy.deepcopy(launch.load_plan(PLAN))


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["lanes"][1]["cpus"].append(0), "already belongs to L0"),
    (lambda p: p["lanes"][1].__setitem__("core_hours", 10_000), "exceeds"),
    (lambda p: p["lanes"][1].__setitem__("planned_core_hours", 10_000), "planned_core_hours exceeds"),
    (lambda p: p["lanes"][1].__setitem__("cross_review", "L1"), "cannot review itself"),
    (lambda p: p["lanes"][2].__setitem__("phase0_all_cpus", True), "exactly one lane"),
    (lambda p: p["lanes"][1].__setitem__("card", "missing.md"), "does not exist"),
    (lambda p: p["lanes"][1]["cpus"].append(32), "outside"),
    (lambda p: p["timeline_hours"].__setitem__("phase0_end", 99), "phase0_end <= production_deadline"),
    (lambda p: p["lanes"][3].__setitem__("cross_review", "L2"), "reviewed by exactly one"),
    (lambda p: p["lanes"][1].__setitem__("pilot_fraction", 0.5), "pilot_fraction"),
])
def test_plan_rule_violations_are_reported(mutate, message):
    plan = _base_plan()
    mutate(plan)
    errors = launch.plan_errors(plan, plan_dir=PLAN.parent)
    assert any(message in error for error in errors), errors


# ---------------------------------------------------------------------------
# Launcher behaviour on a small temporary plan


def _cpus(count):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < count:
        pytest.skip(f"needs {count} available CPUs")
    return available[:count]


def _write_plan(tmp_path, cpus, *, phase0=1.0, deadline=40.0, end=50.0):
    gate, work = cpus[0], cpus[1:]
    for name in ("gate.md", "work.md"):
        (tmp_path / name).write_text("---\norphan: true\n---\n", encoding="utf-8")
    window = deadline - phase0
    plan = {
        "schema": launch.PLAN_SCHEMA,
        "campaign": "unit",
        "machine": {"logical_cpus": max(cpus) + 1},
        "timeline_hours": {"phase0_end": phase0, "production_deadline": deadline, "campaign_end": end},
        "lanes": [
            {"id": "G", "title": "gate", "card": "gate.md", "cpus": [gate], "phase0_all_cpus": True,
             "core_hours": window, "planned_core_hours": 1, "branch": "b/g", "record": "r/g",
             "cross_review": "W"},
            {"id": "W", "title": "work", "card": "work.md", "cpus": work,
             "core_hours": window * len(work), "planned_core_hours": 1, "branch": "b/w", "record": "r/w",
             "cross_review": "G"},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _write_start(root, hours_ago):
    root.mkdir(parents=True, exist_ok=True)
    began = launch._utc_now() - timedelta(hours=hours_ago)
    record = {"schema": launch.START_SCHEMA, "campaign": "unit", "start_utc": launch._iso(began),
              "frozen_commit": None}
    launch.start_path(root).write_text(json.dumps(record), encoding="utf-8")


_PROBE = ("import json, os; print(json.dumps({'cpus': sorted(os.sched_getaffinity(0)), "
          "'omp': os.environ['OMP_NUM_THREADS'], 'blas': os.environ['OPENBLAS_NUM_THREADS'], "
          "'lane': os.environ['C32_LANE']}))")


def _run(plan, root, *extra, command=None):
    command = command or [sys.executable, "-c", _PROBE]
    return launch.main(["run", "--plan", str(plan), "--root", str(root), *extra, "--", *command])


@needs_linux
def test_start_creates_the_clock_once(tmp_path, capsys):
    plan = _write_plan(tmp_path, _cpus(2))
    root = tmp_path / "data"
    args = ["start", "--plan", str(plan), "--root", str(root), "--frozen-commit", "none"]
    assert launch.main(args) == 0
    assert launch.read_start(root)["frozen_commit"] is None
    assert launch.main(args) == 1
    assert "already exists" in capsys.readouterr().err


@needs_linux
def test_run_pins_cpus_sets_threads_and_writes_the_ledger(tmp_path):
    cpus = _cpus(3)
    plan = _write_plan(tmp_path, cpus)
    root = tmp_path / "data"
    _write_start(root, hours_ago=2)
    assert _run(plan, root, "--lane", "W", "--name", "probe", "--kind", "pilot", "--threads", "2") == 0
    records = launch.read_ledger(root / "W")
    assert [r["event"] for r in records] == ["start", "end"]
    start, end = records
    assert start["cpus"] == cpus[1:] and start["slots"] == 2 and start["threads"] == 2
    assert end["returncode"] == 0 and end["stopped_at_deadline"] is False
    assert end["user_cpu_seconds"] is not None and end["max_rss_kib"] > 0
    probe = json.loads(Path(start["stdout"]).read_text(encoding="utf-8"))
    assert probe == {"cpus": cpus[1:], "omp": "2", "blas": "2", "lane": "W"}
    assert not list((root / "W" / "active").glob("*.json"))

    row = {r["lane"]: r for r in launch.lane_usage(launch.load_plan(plan), root)}["W"]
    assert row["runs_finished"] == 1 and row["runs_failed"] == 0 and row["runs_lost"] == []
    assert row["pilot_slot_hours"] == row["slot_hours_used"]


@needs_linux
def test_phase0_belongs_to_the_gate_lane(tmp_path, capsys):
    cpus = _cpus(2)
    plan = _write_plan(tmp_path, cpus)
    root = tmp_path / "data"
    _write_start(root, hours_ago=0)
    assert _run(plan, root, "--lane", "W", "--name", "early", "--kind", "pilot") == launch.EXIT_REFUSED
    assert "Phase 0" in capsys.readouterr().err
    assert _run(plan, root, "--lane", "G", "--name", "gate", "--kind", "test") == 0
    start = launch.read_ledger(root / "G")[0]
    assert start["cpus"] == sorted(cpus)


@needs_linux
def test_slot_limits_and_concurrency_are_enforced(tmp_path, capsys):
    cpus = _cpus(3)
    plan = _write_plan(tmp_path, cpus)
    root = tmp_path / "data"
    _write_start(root, hours_ago=2)
    assert _run(plan, root, "--lane", "W", "--name", "big", "--kind", "pilot", "--slots", "3") == launch.EXIT_REFUSED
    assert "owns 2 CPUs" in capsys.readouterr().err
    assert _run(plan, root, "--lane", "W", "--name", "bad", "--kind", "pilot",
                "--slots", "1", "--threads", "2") == launch.EXIT_REFUSED
    active = root / "W" / "active"
    active.mkdir(parents=True)
    (active / "busy.json").write_text(json.dumps({"run_id": "busy", "launcher_pid": os.getpid(), "slots": 2}),
                                      encoding="utf-8")
    assert _run(plan, root, "--lane", "W", "--name", "queued", "--kind", "pilot") == launch.EXIT_REFUSED
    assert "slots busy" in capsys.readouterr().err
    # A stale registration from a dead launcher no longer blocks the lane.
    (active / "busy.json").write_text(json.dumps({"run_id": "busy", "launcher_pid": 2**22 + 12345, "slots": 2}),
                                      encoding="utf-8")
    assert _run(plan, root, "--lane", "W", "--name", "free", "--kind", "pilot") == 0


@needs_linux
def test_deadline_stops_budgeted_work_but_not_analysis(tmp_path, monkeypatch):
    cpus = _cpus(2)
    plan = _write_plan(tmp_path, cpus)
    root = tmp_path / "data"
    _write_start(root, hours_ago=2)
    soon = launch._utc_now() + timedelta(seconds=1)
    monkeypatch.setattr(launch, "deadline_utc", lambda start, plan, key: soon)
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    code = _run(plan, root, "--lane", "W", "--name", "long", "--kind", "pilot",
                "--grace-seconds", "0.5", command=sleeper)
    end = launch.read_ledger(root / "W")[-1]
    assert end["stopped_at_deadline"] is True and code != 0
    assert end["wall_seconds"] < 60

    _write_start(root, hours_ago=45)  # after the 40 h production deadline, before the 50 h end
    assert _run(plan, root, "--lane", "W", "--name", "late", "--kind", "pilot") == launch.EXIT_REFUSED
    assert _run(plan, root, "--lane", "W", "--name", "analysis", "--kind", "analysis") == 0


@needs_linux
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_production_requires_a_committed_preregistration_and_clean_worktree(tmp_path, monkeypatch, capsys):
    cpus = _cpus(2)
    plan = _write_plan(tmp_path, cpus)
    root = tmp_path / "data"
    _write_start(root, hours_ago=2)
    lane_repo = tmp_path / "lane"
    lane_repo.mkdir()
    git = ["git", "-C", str(lane_repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    prereg = lane_repo / "prereg.json"
    prereg.write_text(json.dumps({"observable": "unit"}), encoding="utf-8")
    monkeypatch.chdir(lane_repo)

    assert _run(plan, root, "--lane", "W", "--name", "p", "--kind", "production") == launch.EXIT_REFUSED
    assert "--prereg" in capsys.readouterr().err
    assert _run(plan, root, "--lane", "W", "--name", "p", "--kind", "production",
                "--prereg", "prereg.json") == launch.EXIT_REFUSED
    assert "must be committed" in capsys.readouterr().err

    subprocess.run([*git, "add", "prereg.json"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "prereg"], check=True)
    (lane_repo / "scratch.txt").write_text("uncommitted driver", encoding="utf-8")
    assert _run(plan, root, "--lane", "W", "--name", "p", "--kind", "production",
                "--prereg", "prereg.json") == launch.EXIT_REFUSED
    assert "clean lane worktree" in capsys.readouterr().err

    (lane_repo / "scratch.txt").unlink()
    assert _run(plan, root, "--lane", "W", "--name", "p", "--kind", "production",
                "--prereg", "prereg.json") == 0
    start = launch.read_ledger(root / "W")[0]
    assert start["prereg"]["sha256"] == launch._sha256(prereg)
    assert start["cwd_git"]["clean"] is True


@needs_linux
def test_frozen_build_identity_is_enforced(tmp_path, monkeypatch, capsys):
    plan = _write_plan(tmp_path, _cpus(2))
    root = tmp_path / "data"
    _write_start(root, hours_ago=2)
    start = json.loads(launch.start_path(root).read_text(encoding="utf-8"))
    start.update(frozen_commit="f" * 40, pyrpmd={"package": "/campaign/src/pyrpmd", "extensions": {"_native.so": "0"}})
    launch.start_path(root).write_text(json.dumps(start), encoding="utf-8")
    other = {"package": "/lane/src/pyrpmd", "extensions": {},
             "git": {"commit": "f" * 40, "clean": True}}
    monkeypatch.setattr(launch, "pyrpmd_identity", lambda: other)
    assert _run(plan, root, "--lane", "W", "--name", "x", "--kind", "pilot") == launch.EXIT_REFUSED
    assert "not the campaign build" in capsys.readouterr().err
    assert _run(plan, root, "--lane", "W", "--name", "x", "--kind", "pilot",
                "--alternate-build") == launch.EXIT_REFUSED
    assert "only for test/benchmark" in capsys.readouterr().err
    assert _run(plan, root, "--lane", "W", "--name", "numpy-suite", "--kind", "test", "--alternate-build") == 0
    assert launch.read_ledger(root / "W")[0]["alternate_build"] is True
    other["git"] = {"commit": "e" * 40, "clean": True}
    assert _run(plan, root, "--lane", "W", "--name", "y", "--kind", "test",
                "--alternate-build") == launch.EXIT_REFUSED
    assert "not the frozen commit" in capsys.readouterr().err


@needs_linux
def test_dry_run_reports_without_executing(tmp_path, capsys):
    plan = _write_plan(tmp_path, _cpus(2))
    root = tmp_path / "data"
    _write_start(root, hours_ago=2)
    assert _run(plan, root, "--lane", "W", "--name", "dry", "--kind", "pilot", "--dry-run") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["lane"] == "W" and shown["event"] == "start"
    assert not (root / "W" / "ledger.jsonl").exists()
