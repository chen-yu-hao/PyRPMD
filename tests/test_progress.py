"""Read-only status, atomic reporting, restart ETA and coordinator ownership."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import threading
import time
from unittest import mock

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez
from pyrpmd.io import write_json
from pyrpmd.progress import RunProgress, read_status, register_parser, status_command
from pyrpmd.progress_readers import CheckpointProgress


def _config(**extra):
    return {"task": "pimd", "sampling": {"equilibration_steps": 10,
            "production_steps": 90, "replicas": 1}, **extra}


def _checkpoint(directory, steps, replica=0):
    path = Path(directory) / "checkpoints" / f"materials_{replica:04d}.npz"
    atomic_savez(path, elapsed=steps, q=np.zeros((2, 1, 3)))
    return path


def _args(directory, **changes):
    return argparse.Namespace(directory=str(directory), watch=False, interval=.01, json=True, **changes)


def _worker(directory, replica, output):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with RunProgress(directory, _config()) as reporter:
            _checkpoint(directory, 100, replica)
    output.put((reporter.enabled, stdout.getvalue(), stderr.getvalue()))


def test_materials_counts_and_atomic_replacement_cache(tmp_path):
    reader = CheckpointProgress(tmp_path, _config())
    assert reader.snapshot()["completed_steps"] == 0
    _checkpoint(tmp_path, 15)
    value = reader.snapshot()
    assert value["stage"] == "production"
    assert value["completed_steps"] == 15
    assert value["total_steps"] == 100
    assert value["percent"] == 15
    _checkpoint(tmp_path, 25)
    assert reader.snapshot()["completed_steps"] == 25
    with mock.patch("pyrpmd.progress_readers._clock", side_effect=AssertionError("unchanged checkpoint reread")):
        assert reader.snapshot()["completed_steps"] == 25


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_clocks_are_unknown_not_zero(tmp_path, bad):
    _checkpoint(tmp_path, bad)
    value = CheckpointProgress(tmp_path, _config()).snapshot()
    assert value["completed_steps"] is None
    assert value["percent"] is None
    assert value["warnings"]


def test_corrupt_and_object_checkpoints_are_not_trusted(tmp_path):
    path = tmp_path / "checkpoints" / "materials_0000.npz"
    path.parent.mkdir()
    path.write_bytes(b"not a zip")
    reader = CheckpointProgress(tmp_path, _config())
    assert reader.snapshot()["completed_steps"] is None
    np.savez(path, elapsed=np.array({"untrusted": True}, dtype=object))
    assert reader.snapshot()["completed_steps"] is None


def test_does_not_load_coordinate_arrays(tmp_path):
    _checkpoint(tmp_path, 10)
    original = np.load

    def selective(source, **kwargs):
        # Only the small scalar NPY member is passed to NumPy, not the NPZ.
        assert isinstance(source, io.BytesIO)
        assert source.getbuffer().nbytes < 1024
        assert kwargs["allow_pickle"] is False
        return original(source, **kwargs)

    with mock.patch("pyrpmd.progress_readers.np.load", side_effect=selective):
        assert CheckpointProgress(tmp_path, _config()).snapshot()["completed_steps"] == 10


def test_native_journal_and_torch_batch_clocks(tmp_path):
    cfg = _config()
    cfg["sampling"].update(engine="native", replicas=2)
    for replica in range(2):
        write_json(tmp_path / "samples" / f"native_pimd_{replica:04d}" / "journal.json",
                   {"blocks": [{"end": 25}]})
    assert CheckpointProgress(tmp_path, cfg).snapshot()["completed_steps"] == 50
    cfg["sampling"]["engine"] = "torch"
    atomic_savez(tmp_path / "checkpoints" / "materials_torch.npz", step=25)
    value = CheckpointProgress(tmp_path, cfg).snapshot()
    assert (value["completed_steps"], value["total_steps"], value["percent"]) == (50, 200, 25)


def test_rate_counts_signed_child_pair_once(tmp_path):
    plan = [{"stage": stage, "planned_steps": total} for stage, total in
            (("umbrella", 100), ("recrossing parent", 20), ("recrossing child", 60))]
    atomic_savez(tmp_path / "umbrella/checkpoints/window_00000_replica_0000.npz", elapsed=100)
    atomic_savez(tmp_path / "checkpoints/parent_0000.npz", elapsed=20)
    atomic_savez(tmp_path / "checkpoints/child_0000_00000_0000.npz", numerator=np.zeros(15))
    value = CheckpointProgress(tmp_path, {"task": "rate"}, plan).snapshot()
    assert value["stage"] == "recrossing child"
    assert value["completed_steps"] == 150
    assert value["total_steps"] == 180
    assert value["percent"] == pytest.approx(100 * 150 / 180)


def test_extension_reads_new_generation_not_old_children(tmp_path):
    plan = [{"stage": stage, "planned_steps": total} for stage, total in
            (("umbrella", 100), ("recrossing parent", 20), ("recrossing child", 60))]
    atomic_savez(tmp_path / "checkpoints/child_0000_00000_0000.npz", numerator=np.zeros(30))
    generation = tmp_path / "extensions/rate/generation"
    value = CheckpointProgress(tmp_path, {"task": "rate"}, plan, checkpoint_directory=generation).snapshot()
    assert value["completed_steps"] == 0


def test_drpmd_records_actual_steps_not_early_stop_budget(tmp_path):
    atomic_savez(tmp_path / "trajectories/trajectory-00000000.checkpoint.npz",
                 metadata=json.dumps({"done": 25, "status": "product"}))
    cfg = {"task": "drpmd", "collision": {"max_steps": 100, "trajectories": 1}}
    value = CheckpointProgress(tmp_path, cfg).snapshot()
    assert value["completed_steps"] == 25
    assert value["percent"] == 25
    assert "early stopping" in value["scope"]


@pytest.mark.parametrize("config", [
    {"task": "network"},
    {"task": "fes", "adaptive_umbrella": {"enabled": True}},
    {"task": "pimd", "tempering": {"mode": "temperature"}},
    {"task": "fes", "fes": {"method": "metadynamics"}, "metadynamics": {"shared_walkers": True}},
])
def test_unknown_budgets_are_not_fabricated(tmp_path, config):
    value = CheckpointProgress(tmp_path, config).snapshot()
    assert value["total_steps"] is None
    assert value["percent"] is None


def test_monitor_lifecycle_leaves_stdout_clean(tmp_path, capsys):
    _checkpoint(tmp_path, 15)
    with RunProgress(tmp_path, _config(), interval=.01) as reporter:
        _checkpoint(tmp_path, 100)
        reporter.result = {"status": "complete", "converged": False}
    value = read_status(tmp_path)
    assert value["status"] == "completed"
    assert value["percent"] == 100
    assert value["eta_seconds"] == 0
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    records = [json.loads(line) for line in stderr.splitlines()]
    assert records[0]["event"] == "run.started"
    assert records[-1]["event"] == "run.completed"
    log = [json.loads(line) for line in (tmp_path / "run.log.jsonl").read_text().splitlines()]
    assert records == log


@pytest.mark.parametrize("error,state", [(RuntimeError("PES failed"), "failed"),
                                          (KeyboardInterrupt(), "interrupted")])
def test_terminal_error_preserves_original_exception(tmp_path, error, state):
    with pytest.raises(type(error)) as caught:
        with RunProgress(tmp_path, _config()):
            raise error
    assert caught.value is error
    assert read_status(tmp_path)["status"] == state


def test_partial_does_not_become_completed(tmp_path):
    with RunProgress(tmp_path, _config()) as reporter:
        reporter.result = {"status": "partial"}
    assert read_status(tmp_path)["status"] == "partial"


def test_reporting_failure_does_not_mask_scientific_failure(tmp_path, capsys):
    original = RuntimeError("original PES failure")
    with mock.patch("pyrpmd.progress.write_json", side_effect=OSError("disk full")):
        with pytest.raises(RuntimeError) as caught:
            with RunProgress(tmp_path, _config()):
                raise original
    assert caught.value is original
    assert capsys.readouterr().err.count('"event": "progress.warning"') == 1


def test_resume_eta_uses_new_work_and_resets_for_stage(tmp_path):
    _checkpoint(tmp_path, 50)
    with mock.patch("pyrpmd.progress.time.monotonic", return_value=100):
        with RunProgress(tmp_path, _config(resume=True), interval=100) as reporter:
            assert read_status(tmp_path)["eta_seconds"] is None
            _checkpoint(tmp_path, 60)
            with mock.patch("pyrpmd.progress.time.monotonic", return_value=110):
                reporter._publish("running")
            assert read_status(tmp_path)["eta_seconds"] == pytest.approx(40)
            assert read_status(tmp_path)["resumed"]


def test_atomic_snapshots_during_continuous_reads(tmp_path):
    failures = []
    stop = threading.Event()
    with RunProgress(tmp_path, _config(), interval=.005) as reporter:
        def read():
            while not stop.is_set():
                try:
                    read_status(tmp_path)
                except Exception as error:
                    failures.append(error)
        thread = threading.Thread(target=read)
        thread.start()
        try:
            for step in range(1, 20):
                _checkpoint(tmp_path, step)
                reporter._stop.wait(.005)
        finally:
            stop.set()
            thread.join()
    assert failures == []


def test_nonroot_mpi_and_local_workers_do_not_report(tmp_path, capsys):
    with mock.patch("pyrpmd.parallel.is_root", return_value=False):
        with RunProgress(tmp_path, _config()) as reporter:
            assert not reporter.enabled
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr() == ("", "")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [context.Process(target=_worker, args=(str(tmp_path), replica, queue)) for replica in range(2)]
    cfg = _config()
    cfg["sampling"]["replicas"] = 2
    with RunProgress(tmp_path, cfg, interval=.01):
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
            if worker.is_alive():
                worker.terminate()
                worker.join()
                pytest.fail("progress worker did not terminate")
            assert worker.exitcode == 0
        assert [queue.get(timeout=5) for _ in workers] == [(False, "", ""), (False, "", "")]
    queue.close()
    assert read_status(tmp_path)["completed_steps"] == 200
    log = [json.loads(line) for line in (tmp_path / "run.log.jsonl").read_text().splitlines()]
    assert sum(item["event"] == "run.started" for item in log) == 1
    assert sum(item["event"] == "run.completed" for item in log) == 1


def test_readonly_cli_and_watch_terminal_exit(tmp_path, capsys):
    with RunProgress(tmp_path, _config()):
        pass
    capsys.readouterr()
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert status_command(_args(tmp_path)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    args = _args(tmp_path)
    args.watch = True
    assert status_command(args) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert "\x1b" not in output
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}


def test_watch_refresh_and_ctrl_c(tmp_path, capsys):
    with RunProgress(tmp_path, _config()):
        value = read_status(tmp_path)
    args = _args(tmp_path)
    args.watch = True
    with mock.patch("pyrpmd.progress.read_status", return_value=value), \
            mock.patch("pyrpmd.progress.time.sleep", side_effect=KeyboardInterrupt):
        assert status_command(args) == 130
    assert "Traceback" not in capsys.readouterr().err


def test_missing_invalid_and_stale_snapshots(tmp_path):
    with pytest.raises(ValueError, match="No progress.json"):
        read_status(tmp_path)
    write_json(tmp_path / "progress.json", {"schema": "foreign"})
    with pytest.raises(ValueError, match="Unsupported progress schema"):
        read_status(tmp_path)
    with RunProgress(tmp_path, _config()):
        value = read_status(tmp_path)
    value["updated_unix"] = time.time() - 300
    write_json(tmp_path / "progress.json", value)
    status = read_status(tmp_path)
    assert status["stale"]
    assert status["status"] == "running"
    assert status["eta_seconds"] is None


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_watch_intervals(interval, tmp_path):
    args = _args(tmp_path)
    args.interval = interval
    with pytest.raises(ValueError, match="finite positive"):
        status_command(args)


def test_status_registration_and_real_runner(tmp_path, capsys):
    from pyrpmd.cli import main
    from pyrpmd.runner import run, resume_run
    from test_runner_ux import small_fes
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["status", str(tmp_path), "--watch"]).watch
    directory = tmp_path / "run"
    run(small_fes(), output=directory)
    value = read_status(directory)
    assert value["status"] == "completed"
    assert (value["completed_steps"], value["total_steps"]) == (22, 22)
    checkpoint_bytes = {p: p.read_bytes() for p in (directory / "checkpoints").glob("*.npz")}
    resume_run(directory)
    assert read_status(directory)["resumed"]
    assert all(p.read_bytes() == data for p, data in checkpoint_bytes.items())
    capsys.readouterr()
    assert main(["status", str(directory), "--watch", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="Output directory is not empty"):
        run(small_fes(), output=directory)
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_monitor_never_initializes_a_missing_run_directory(tmp_path):
    missing = tmp_path / "missing"
    with RunProgress(missing, _config()) as reporter:
        assert not reporter.enabled
    assert not missing.exists()


def test_mpi_rank_check_never_runs_on_heartbeat_thread(tmp_path):
    owner = threading.current_thread()

    def is_root():
        assert threading.current_thread() is owner
        return True

    with mock.patch("pyrpmd.parallel.is_root", side_effect=is_root) as check:
        with RunProgress(tmp_path, _config(), interval=.01):
            time.sleep(.03)
        assert check.call_count == 1


def test_disabled_adaptive_options_keep_fixed_budget(tmp_path):
    cfg = {"task": "fes", "adaptive_umbrella": {"enabled": False}}
    plan = [{"stage": "umbrella", "planned_steps": 100}]
    value = CheckpointProgress(tmp_path, cfg, plan).snapshot()
    assert value["total_steps"] == 100


def test_process_fes_has_one_coordinator_log(tmp_path):
    from pyrpmd.runner import run
    from test_runner_ux import small_fes

    cfg = small_fes()
    cfg["sampling"]["replicas"] = 2
    cfg["parallel"] = {"backend": "process", "trajectory_workers": 2}
    directory = tmp_path / "process-fes"
    run(cfg, output=directory)
    value = read_status(directory)
    assert value["completed_steps"] == value["total_steps"] == 44
    log = [json.loads(row) for row in (directory / "run.log.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "run.started" for row in log) == 1
    assert len({row["pid"] for row in log}) == 1


def test_native_pimd_reports_real_journal_clock(tmp_path):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.runner import run
    from test_native_pimd_preflight import job

    cfg = job()
    cfg["sampling"].update(equilibration_steps=2, production_steps=4, replicas=2)
    cfg["output"] = {"checkpoint_steps": 2}
    directory = tmp_path / "native-pimd"
    run(cfg, output=directory)
    value = read_status(directory)
    assert value["completed_steps"] == value["total_steps"] == 12
    assert value["percent"] == 100


def test_drpmd_effective_budget_survives_completed_extension_resume(tmp_path):
    from pyrpmd.runner import run_resolved, resume_run, extend_run
    from test_drpmd_extend import configuration

    cfg, system = configuration(tmp_path)
    directory = tmp_path / "drpmd"
    run_resolved(cfg, system, output=directory)
    extend_run(directory, max_time=8 * cfg["dt_au"])
    assert read_status(directory)["completed_steps"] == 16
    resume_run(directory)
    value = read_status(directory)
    assert value["completed_steps"] == value["total_steps"] == 16
    assert value["percent"] == 100


def test_eta_recovers_after_an_unreadable_clock(tmp_path):
    _checkpoint(tmp_path, -1)
    with mock.patch("pyrpmd.progress.time.monotonic", return_value=100):
        with RunProgress(tmp_path, _config(), interval=100) as reporter:
            assert read_status(tmp_path)["completed_steps"] is None
            _checkpoint(tmp_path, 15)
            with mock.patch("pyrpmd.progress.time.monotonic", return_value=110):
                reporter._publish("running")
            assert read_status(tmp_path)["eta_seconds"] is None
            _checkpoint(tmp_path, 25)
            with mock.patch("pyrpmd.progress.time.monotonic", return_value=120):
                reporter._publish("running")
            assert read_status(tmp_path)["eta_seconds"] == pytest.approx(75)
