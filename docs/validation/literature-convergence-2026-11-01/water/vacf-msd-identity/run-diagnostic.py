"""Bounded offline execution of the post hoc complete-cohort identity check."""
import time
START = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
WATER = HERE.parent
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
os.sched_setaffinity(0, {5})
os.nice(10)
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.setitimer(signal.ITIMER_REAL, 120)
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT))
from benchmarks.water import vacf_msd_identity as diagnostic

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

guards = {name: sha(ROOT/name) for name in diagnostic.SOURCE_NAMES}
report = diagnostic.diagnose(WATER/"original-fixed-cohort/analysis", WATER/"protocol.json",
    WATER/"production-stability-preregistration.json", WATER/"original-fixed-cohort/original-normal-exit.json",
    HERE/"analysis")
assert guards == {name: sha(ROOT/name) for name in guards}
usage = resource.getrusage(resource.RUSAGE_SELF)
receipt = {"schema": "pyrpmd.water-vacf-msd-identity-execution.v1", "all_source_guards_unchanged": guards,
           "report_sha256": sha(HERE/"analysis/report.json"), "affinity": sorted(os.sched_getaffinity(0)),
           "nice": os.nice(0), "CPU_seconds": usage.ru_utime+usage.ru_stime,
           "wall_seconds": time.monotonic()-START, "peak_RSS_KiB": usage.ru_maxrss,
           "CPU_hard_limit_seconds": 60, "wall_limit_seconds": 120,
           "physical_steps": 0, "PES_evaluations": 0}
with (HERE/"execution.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True); stream.write("\n")
print(json.dumps({"execution": receipt, "short_minus_late_contrast": report["short_minus_late_contrast"]}, indent=2))
