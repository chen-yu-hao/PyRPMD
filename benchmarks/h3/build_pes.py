"""Fetch a checksum-pinned real BKMP2 PES and compile its C-interoperable binding."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text())
DEFAULT_LIBRARY = HERE / "build" / "libbkmp2.so"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_source(path):
    path = Path(path)
    if path.stat().st_size != SOURCE["bytes"] or sha256(path) != SOURCE["sha256"]:
        raise ValueError(f"BKMP2 source checksum mismatch: {path}")
    return path


def fetch_source(destination, source=None):
    destination = Path(destination)
    if source is not None:
        original = checked_source(source)
        if original.resolve() != destination.resolve():
            shutil.copyfile(original, destination)
    elif not destination.exists():
        request = urllib.request.Request(
            SOURCE["url"], headers={"User-Agent": "PyRPMD-H3-benchmark/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(SOURCE["bytes"] + 1)
        if (
            len(content) != SOURCE["bytes"]
            or hashlib.sha256(content).hexdigest() != SOURCE["sha256"]
        ):
            raise ValueError(
                "Downloaded BKMP2 data do not match the pinned source; refusing to compile"
            )
        temporary = destination.with_suffix(".download")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return checked_source(destination)


def build(output=DEFAULT_LIBRARY, *, source=None, compiler=None, extra_flags=None):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = fetch_source(output.parent / "h3_bkmp2.f", source)
    wrapper = HERE / "wrapper.f90"
    command = shlex.split(compiler or os.environ.get("FC", "gfortran"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError(
            "GNU Fortran is required: install gfortran or set FC to its executable"
        )
    version = subprocess.run(
        command + ["--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if "GNU Fortran" not in version:
        raise RuntimeError("This reproducible benchmark currently requires GNU Fortran")
    flags = shlex.split(
        extra_flags if extra_flags is not None else os.environ.get("FFLAGS", "")
    )
    flags += [
        "-O3",
        "-fPIC",
        "-shared",
        "-std=legacy",
        "-ffixed-line-length-none",
        "-fdefault-real-8",
        "-fdefault-double-8",
    ]
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    compile_command = (
        command + flags + [str(source_path), str(wrapper), "-o", str(temporary)]
    )
    try:
        completed = subprocess.run(
            compile_command, check=True, capture_output=True, text=True
        )
        # Independent author-published saddle energy guards kind/ABI mistakes.
        lib = ctypes.CDLL(str(temporary))
        radial = lib.pyrpmd_bkmp2_radial
        radial.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [
            ctypes.POINTER(ctypes.c_int32)
        ]
        radial.restype = None
        r = (ctypes.c_double * 3)(1.757, 1.757, 3.514)
        e = ctypes.c_double()
        g = (ctypes.c_double * 3)()
        status = ctypes.c_int32()
        radial(r, ctypes.byref(e), g, ctypes.byref(status))
        if status.value or abs(e.value + 0.159176000) > 5e-10:
            raise RuntimeError(
                f"Compiled BKMP2 failed the independent saddle oracle: {e.value}, status={status.value}"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": "pyrpmd.h3.bkmp2-build.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "source_file": str(source_path),
        "wrapper_sha256": sha256(wrapper),
        "library": str(output),
        "library_sha256": sha256(output),
        "compiler": version,
        "command": compile_command,
        "flags": flags,
        "platform": platform.platform(),
        "compiler_stdout": completed.stdout,
        "compiler_stderr": completed.stderr,
        "author_saddle_energy_hartree": e.value,
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--source", type=Path, help="offline copy of the exact pinned source"
    )
    parser.add_argument(
        "--fc", help="GNU Fortran executable; defaults to FC or gfortran"
    )
    parser.add_argument(
        "--fflags", help="additional compiler flags, before required precision flags"
    )
    args = parser.parse_args()
    report = build(
        args.output, source=args.source, compiler=args.fc, extra_flags=args.fflags
    )
    print(
        json.dumps(
            {
                "library": report["library"],
                "source_sha256": SOURCE["sha256"],
                "library_sha256": report["library_sha256"],
                "compiler": report["compiler"].splitlines()[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
