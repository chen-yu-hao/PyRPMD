"""Time the tutorial dRPMD calculation with 1, 2 and 4 trajectory worker processes.

Usage: python paper/scripts/parallel.py EXCHANGE_DIRECTORY RUNS_DIRECTORY > parallel.json
EXCHANGE_DIRECTORY contains drpmd.inp and build/libexchange.so.
"""
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

work, runs = Path(sys.argv[1]), Path(sys.argv[2])
base = (work / "drpmd.inp").read_text()
serial = 'backend = "serial"\ntrajectory_workers = 1'
assert serial in base
rows, digests = [], set()
for workers in (1, 2, 4):
    name = work / f"drpmd_w{workers}.inp"
    name.write_text(base.replace(serial, f'backend = "process"\nmode = "serial"\ntrajectory_workers = {workers}'))
    times = []
    for repeat in range(3):
        out = runs / f"par_w{workers}"
        shutil.rmtree(out, ignore_errors=True)
        start = time.perf_counter()
        subprocess.run([sys.executable, "-m", "pyrpmd", "run", name.name, "--output", str(out)],
                       cwd=work, check=True, stdout=subprocess.DEVNULL)
        times.append(time.perf_counter() - start)
    digests.add(hashlib.sha256((out / "results/channels.csv").read_bytes()).hexdigest())
    rows.append({"workers": workers, "wall_seconds": statistics.median(times), "repeats": times})
json.dump({"calculation": "dRPMD, 4096 collisions, 32 parents, P=4, process backend",
           "identical_results_across_workers": len(digests) == 1, "rows": rows}, sys.stdout, indent=1)
