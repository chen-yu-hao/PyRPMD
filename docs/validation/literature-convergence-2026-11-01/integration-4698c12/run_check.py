"""Check the frozen H3 methods on the externally integrated main branch.

This is a bounded compatibility check, not a new statistical rate campaign.
Run from the repository root with PYTHONPATH=src:. and one numerical thread.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-replay", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    evidence = root / "docs/validation/literature-convergence-2026-11-01"
    output = Path(__file__).resolve().parent
    runtime = {}
    for name in ("pyrpmd.dynamics", "pyrpmd.propagation", "pyrpmd.cv",
                 "pyrpmd.checkpoint", "pyrpmd.constants", "pyrpmd.io", "pyrpmd._native"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(root / "src"):
            raise ValueError(f"Compatibility check loaded another checkout: {path}")
        runtime[name] = {"path": str(path.relative_to(root)), "sha256": digest(path)}
    tests = json.loads((evidence / "h3-method-test-receipt.json").read_text())["test_files"]
    tests += ["tests/test_h3_surface_partition_controls.py"]
    command = [sys.executable, "-m", "pytest", "-q", *tests,
               "--junitxml=" + str(output / "h3-method-tests.xml")]
    with (output / "h3-method-tests.log").open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError("Integrated H3 method tests failed; inspect the retained log")
    from benchmarks.h3.quantum_recrossing_validate import validate
    from pyrpmd.checkpoint import load_checkpoint
    replay = validate(output / "native-replay", args.library, args.parent)
    comparisons = []
    for dt in (.1, .05):
        name = f"observations-dt-{dt:g}.npz"
        old_path, new_path = args.old_replay / name, output / "native-replay" / name
        with load_checkpoint(old_path, expected_kind="h3.quantum-child-replay-observations",
                             require_integrity=True) as old, \
             load_checkpoint(new_path, expected_kind="h3.quantum-child-replay-observations",
                             require_integrity=True) as new:
            values = {}
            for key in ("time_fs", "continuous", "restored", "final_q", "final_p"):
                a, b = np.asarray(old[key]), np.asarray(new[key])
                values[key] = {"bitwise_equal": bool(np.array_equal(a, b)),
                               "maximum_absolute_difference": float(np.max(np.abs(a - b)))}
        comparisons.append({"timestep_fs": dt, "old_sha256": digest(old_path),
                            "new_sha256": digest(new_path), "arrays": values})
    suite = ET.parse(output / "h3-method-tests.xml").getroot().find("testsuite")
    unchanged = all(digest(root / row["path"]) == row["sha256"] for row in runtime.values())
    exact = all(v["bitwise_equal"] for row in comparisons for v in row["arrays"].values())
    report = {
        "schema": "pyrpmd.h3.external-main-integration-check.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_commit": "4698c1202d0726bce7d63742e7cd31a6f137a847",
        "frozen_experiment_commit": "3d096d5cfc5709f14a1ad5e3d184b108eef9f20e",
        "scope": "Method/API tests and one excluded native H3 calibration at two time steps; not a rerun of the primary statistical campaigns or the complete project suite.",
        "python": platform.python_version(), "numpy": np.__version__,
        "runtime": runtime, "source_unchanged_during_check": unchanged,
        "test_command": command, "test_files_sha256": {p: digest(root / p) for p in tests},
        "test_summary": dict(suite.attrib), "junit_sha256": digest(output / "h3-method-tests.xml"),
        "old_replay_report_sha256": digest(args.old_replay / "native-restart-validation.json"),
        "new_replay_report_sha256": digest(output / "native-replay/native-restart-validation.json"),
        "old_new_native_comparisons": comparisons,
        "all_compared_trajectory_arrays_bitwise_equal": exact,
        "new_native_restart_exact": all(all(row[k] for k in
            ("final_q_bitwise_equal", "final_p_bitwise_equal", "all_observations_bitwise_equal", "step_equal", "time_equal"))
            for row in replay["rows"]),
        "driver_sha256": digest(__file__),
    }
    (output / "compatibility.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"method_tests": dict(suite.attrib), "old_new_native_arrays_equal": exact,
                      "runtime_source_unchanged": unchanged}), flush=True)
    if not exact or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
