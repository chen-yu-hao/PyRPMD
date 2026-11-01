"""Run the repository suite while verifying production/test source immutability."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def main():
    output = Path(__file__).resolve().parent
    repo = output.parents[2]

    def source_hashes():
        suffixes = {".py", ".cpp", ".hpp", ".h", ".f90", ".c"}
        return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ("src", "tests", "native")
                for p in sorted((repo / folder).rglob("*"))
                if p.is_file() and p.suffix in suffixes}

    before = source_hashes()
    (output / "full-suite-source-manifest.json").write_text(json.dumps(before, indent=2) + "\n")
    packages = ("numpy", "scipy", "pytest", "h5py", "torch", "ase", "mpi4py")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONPATH=str(repo / "src"))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="p63-") as temp:
        command = [sys.executable, "-m", "pytest", "-q", "--basetemp=" + temp,
                   "--junitxml=" + str(output / "full-suite.xml")]
        with (output / "full-suite.log").open("w") as log:
            completed = subprocess.run(command, cwd=repo, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
    after = source_hashes()
    report = {
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "command": command, "python": sys.version, "platform": platform.platform(),
        "dependencies": versions, "thread_limits": {k: environment[k] for k in
                                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "source_files": len(before), "source_unchanged": before == after,
        "changed_sources": sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)),
        "returncode": completed.returncode, "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "full-suite-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return completed.returncode or (1 if before != after else 0)


if __name__ == "__main__":
    raise SystemExit(main())
