"""Build a byte-pinned, unmodified RPMDrate CBE/PES-2008 source with a C ABI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text())
DEFAULT_LIBRARY = HERE / "build" / "libcbe2008.so"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source(path, record):
    path = Path(path)
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if (len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]
            or blob != record["git_blob_sha1"]):
        raise ValueError(f"CBE/PES-2008 source checksum mismatch: {path}")
    return path


def fetch_sources(destination, source_dir=None):
    """Fetch only the two compile inputs, or verify/copy an exact offline pair."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for record in SOURCE["files"]:
        if not record["compile"]:
            continue
        target = destination / record["name"]
        if source_dir is not None:
            original = check_source(Path(source_dir) / record["name"], record)
            if original.resolve() != target.resolve():
                shutil.copyfile(original, target)
        elif not target.exists():
            request = urllib.request.Request(record["url"], headers={"User-Agent": "PyRPMD-HCH4-benchmark/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(record["bytes"] + 1)
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            try:
                check_source(temporary, record)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        outputs.append(check_source(target, record))
    return outputs


def build(output=DEFAULT_LIBRARY, *, source_dir=None, compiler=None):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = fetch_sources(output.parent, source_dir)
    command = shlex.split(compiler or os.environ.get("FC", "gfortran"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError("GNU Fortran is required; supply --fc or FC")
    version = subprocess.run(command + ["--version"], check=True, capture_output=True, text=True).stdout.strip()
    if "GNU Fortran" not in version:
        raise RuntimeError("This benchmark build currently supports GNU Fortran")
    # utility.f has legacy sequence labels after column 72. Extending fixed
    # lines would interpret those labels as code and is not equivalent.
    flags = ["-O2", "-fPIC", "-shared", "-std=legacy", "-ffixed-line-length-72"]
    wrapper = HERE / "wrapper.f90"
    with tempfile.TemporaryDirectory(prefix="cbe-build-", dir=output.parent) as temporary:
        candidate = Path(temporary) / output.name
        invocation = command + flags + [str(p) for p in inputs] + [str(wrapper), "-o", str(candidate)]
        result = subprocess.run(invocation, check=False, capture_output=True, text=True, cwd=temporary, timeout=120)
        if result.returncode:
            failure = {"schema": "pyrpmd.hch4.build-failure.v1", "command": invocation,
                       "returncode": result.returncode, "stdout": result.stdout,
                       "stderr": result.stderr, "builder_sha256": sha256(Path(__file__))}
            output.with_suffix(".failed-build.json").write_text(json.dumps(failure, indent=2) + "\n")
            result.check_returncode()
        candidate.replace(output)
    report = {
        "schema": "pyrpmd.hch4.build.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "source_files": {p.name: {"path": str(p), "sha256": sha256(p)} for p in inputs},
        "wrapper_sha256": sha256(wrapper),
        "builder_sha256": sha256(Path(__file__)),
        "library": str(output), "library_sha256": sha256(output),
        "compiler": version, "command": invocation, "flags": flags,
        "compiler_stdout": result.stdout, "compiler_stderr": result.stderr,
        "platform": platform.platform(),
        "scope": "Build only; run verify.py before interpreting energy/force results. No rates are computed.",
    }
    output.with_suffix(".build.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--source-dir", type=Path, help="Offline directory containing the exact two pinned .f files")
    parser.add_argument("--fc", help="GNU Fortran executable, default FC or gfortran")
    args = parser.parse_args()
    report = build(args.output, source_dir=args.source_dir, compiler=args.fc)
    print(json.dumps({"library": report["library"], "library_sha256": report["library_sha256"],
                      "compiler": report["compiler"].splitlines()[0]}, indent=2))


if __name__ == "__main__":
    main()
