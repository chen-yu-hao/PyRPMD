"""Bounded HCH4 integration replay; keep the registered NVE failure visible."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pyrpmd
from pyrpmd import _native
from benchmarks.hch4 import pes, verify_native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
LIBRARY = Path("/workspace/scratch/a5e4820daedd/literature-convergence-hch4/build/libcbe2008.so")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (HERE / "receipt.json").exists() or (HERE / "workflow").exists():
        raise ValueError("This integration receipt already exists; use a fresh evidence directory")
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYRPMD_CBE2008_LIBRARY=str(LIBRARY))
    paths = {"pyrpmd": pyrpmd.__file__, "native": _native.__file__,
             "HCH4_provider": pes.__file__, "HCH4_probe": verify_native.__file__}
    for path in paths.values():
        assert Path(path).resolve().is_relative_to(ROOT), path
    guarded = sorted(set((ROOT / "src/pyrpmd").rglob("*.py"))
                     | set((ROOT / "benchmarks/hch4").glob("*.py"))
                     | {Path(_native.__file__), ROOT / "benchmarks/hch4/reproduction_protocol.json",
                        ROOT / "benchmarks/hch4/wrapper.f90", ROOT / "tests/test_hch4_native.py",
                        ROOT / "tests/test_hch4_pes.py"})
    before = {str(p.relative_to(ROOT)): sha256(p) for p in guarded}
    head = lambda: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {"schema": "pyrpmd.hch4.integration-replay.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "git_head_at_start": head(), "evidence_label": "integration-4698c12",
              "label_note": "Directory label denotes the integration base; actual execution HEAD and imported bytes are recorded separately.",
              "root": str(ROOT), "python_executable": sys.executable, "python": platform.python_version(),
              "numpy": np.__version__, "imports": paths, "native_sha256": sha256(_native.__file__),
              "PES_library": str(LIBRARY), "PES_library_sha256": sha256(LIBRARY),
              "environment": {k: env[k] for k in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYRPMD_CBE2008_LIBRARY")},
              "runner_sha256": sha256(__file__), "commands": [], "rate_computed": False,
              "long_sampling_launched": False}
    commands = [
        ("standard-tests", ["-m", "pytest", "tests/test_hch4_native.py", "tests/test_hch4_pes.py",
                            "-q", "-p", "no:cacheprovider", "--junitxml=" + str(HERE / "standard-tests.xml")]),
        ("native-verification", ["-m", "benchmarks.hch4.verify_native", "--library", str(LIBRARY),
                                 "--output", str(HERE / "native-verification.json")]),
        ("workflow", ["-m", "benchmarks.hch4.verify_workflow", "--library", str(LIBRARY),
                       "--output", str(HERE / "workflow")]),
    ]
    for name, arguments in commands:
        started = time.perf_counter()
        used = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_before = used.ru_utime + used.ru_stime
        with (HERE / f"{name}.log").open("w") as log:
            run = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, check=False, timeout=180)
        used = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["commands"].append({"name": name, "argv": [sys.executable, *arguments],
                                   "exit_code": run.returncode, "wall_seconds": time.perf_counter() - started,
                                   "CPU_seconds": used.ru_utime + used.ru_stime - cpu_before,
                                   "log": f"{name}.log"})
    probe = json.loads((HERE / "native-verification.json").read_text())
    workflow = json.loads((HERE / "workflow/report.json").read_text())
    oldpath = ROOT / "docs/validation/literature-convergence-2026-11-01/other-reactions/hch4/native/verification.json"
    old = json.loads(oldpath.read_text())
    old_spans = {(r["beads"], r["dt_fs"]): r["energy_span_per_bead_hartree"] for r in old["nve_checks"]}
    deltas = [{"beads": r["beads"], "dt_fs": r["dt_fs"],
               "old_span_hartree_per_bead": old_spans[(r["beads"], r["dt_fs"])],
               "new_span_hartree_per_bead": r["energy_span_per_bead_hartree"],
               "difference": r["energy_span_per_bead_hartree"] - old_spans[(r["beads"], r["dt_fs"])],
               } for r in probe["nve_checks"]]
    after = {str(p.relative_to(ROOT)): sha256(p) for p in guarded}
    result.update(git_head_at_end=head(), source_files=before, source_files_unchanged=before == after,
                  native_probe_gates=probe["gates"], native_probe_overall_passed=probe["passed"],
                  workflow_passed=workflow["passed"], workflow_cases=len(workflow["records"]),
                  original_probe_sha256=sha256(oldpath), NVE_comparison_with_original=deltas,
                  known_failure="P128 dt=0.1 fs fixed non-equilibrium 20 fs NVE energy span exceeds unchanged 5e-5 Ha/bead gate. Native/Python parity is a separate passing check.")
    result["integration_checks_passed"] = bool(
        [r["exit_code"] for r in result["commands"]] == [0, 1, 0]
        and probe["gates"]["native_python_all_paths"]
        and probe["gates"]["static_equations_and_channels"]
        and probe["gates"]["P1_short_NVE"]
        and not probe["gates"]["P128_short_NVE"]
        and workflow["passed"] and before == after
        and max(abs(r["difference"]) for r in deltas) < 1e-12)
    result["total_subprocess_CPU_seconds"] = sum(r["CPU_seconds"] for r in result["commands"])
    (HERE / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("integration_checks_passed", "git_head_at_start", "git_head_at_end",
                                            "native_sha256", "native_probe_gates", "workflow_passed",
                                            "total_subprocess_CPU_seconds", "NVE_comparison_with_original")}, indent=2))
    if not result["integration_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
