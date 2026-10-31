"""Fetch checksum-pinned BrHCl_LEPS1 and build its atomic-unit PyRPMD C ABI."""

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
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text())
DEFAULT_LIBRARY = HERE / "build" / "libbrhcl.so"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_source(path):
    path = Path(path)
    if path.stat().st_size != SOURCE["bytes"] or sha256(path) != SOURCE["sha256"]:
        raise ValueError(f"BrHCl_LEPS1 source checksum mismatch: {path}")
    return path


def fetch_source(destination, source=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        original = checked_source(source)
        if original.resolve() != destination.resolve():
            shutil.copyfile(original, destination)
    elif not destination.exists():
        request = urllib.request.Request(
            SOURCE["url"], headers={"User-Agent": "PyRPMD-BrHCl-benchmark/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(SOURCE["bytes"] + 1)
        if len(content) != SOURCE["bytes"] or hashlib.sha256(content).hexdigest() != SOURCE["sha256"]:
            raise ValueError("Downloaded BrHCl source does not match the pinned source")
        temporary = destination.with_suffix(".download")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return checked_source(destination)


def _compiler(command, default):
    command = shlex.split(command or default)
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError(f"Compiler unavailable: {command or default}")
    version = subprocess.run(command + ["--version"], check=True, capture_output=True, text=True).stdout.strip()
    return command, version


def build(output=DEFAULT_LIBRARY, *, source=None, compiler=None, cxx=None):
    """Build with ordinary DOUBLE PRECISION kinds; no implicit kind promotion."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = fetch_source(output.parent / "BrHCl_LEPS1.f", source)
    fc, fc_version = _compiler(compiler, os.environ.get("FC", "gfortran"))
    cc, cc_version = _compiler(cxx, os.environ.get("CXX", "g++"))
    if "GNU Fortran" not in fc_version:
        raise RuntimeError("This reproducible legacy-source build requires GNU Fortran")
    # Do not inherit FFLAGS: real/integer promotion silently changes COMMON ABI.
    if any("default-real" in arg or "default-double" in arg or "default-integer" in arg for arg in fc):
        raise ValueError("Kind-promotion flags are not supported for the fixed legacy COMMON ABI")
    wrapper = HERE / "wrapper.f90"
    cpp = HERE / "wrapper.cpp"
    exports = HERE / "exports.map"
    commands, outputs = [], []
    with tempfile.TemporaryDirectory(prefix="brhcl-build-", dir=output.parent) as directory:
        directory = Path(directory)
        objects = [directory / name for name in ("legacy.o", "fortran.o", "cabi.o")]
        temporary = directory / output.name
        commands = [
            fc + ["-O3", "-fPIC", "-std=legacy", "-ffixed-line-length-none", "-c", str(source_path), "-o", str(objects[0])],
            fc + ["-O3", "-fPIC", "-ffree-line-length-none", "-c", str(wrapper), "-o", str(objects[1])],
            cc + ["-O3", "-fPIC", "-std=c++17", "-pthread", "-I" + str(HERE.parents[1] / "native/include"), "-c", str(cpp), "-o", str(objects[2])],
            fc + ["-shared", *map(str, objects), "-lstdc++", "-pthread", "-Wl,--version-script=" + str(exports), "-o", str(temporary)],
        ]
        for command in commands:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            outputs.append({"stdout": result.stdout, "stderr": result.stderr})
        library = ctypes.CDLL(str(temporary))
        radial = library.pyrpmd_brhcl_radial
        radial.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.POINTER(ctypes.c_int32)]
        radial.restype = None
        checks = []
        # Independent asymptotic dimer energies from the published parameters.
        for index, name in enumerate(("HBr+Cl", "HCl+Br", "BrCl+H")):
            r = [99.0] * 3
            r[index] = SOURCE["morse_equilibrium_angstrom"][index] / SOURCE["legacy_angstrom_per_bohr"]
            expected = (SOURCE["morse_dissociation_kcal_mol"][1] - SOURCE["morse_dissociation_kcal_mol"][index]) / SOURCE["legacy_kcal_mol_per_hartree"]
            energy = ctypes.c_double()
            gradient = (ctypes.c_double * 3)()
            status = ctypes.c_int32()
            radial((ctypes.c_double * 3)(*r), ctypes.byref(energy), gradient, ctypes.byref(status))
            if status.value or abs(energy.value - expected) > 1e-9:
                raise RuntimeError(f"Compiled BrHCl failed {name} oracle: {energy.value} vs {expected}, status={status.value}")
            checks.append({"arrangement": name, "distances_bohr": r, "energy_hartree": energy.value, "reference_hartree": expected,
                           "absolute_error_hartree": abs(energy.value - expected),
                           "tolerance_hartree": 1e-9, "note": "The unchanged source uses rounded sqrt(2)=1.41421356."})
        temporary.replace(output)
    report = {
        "schema": "pyrpmd.brhcl.leps1-build.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE, "source_file": str(source_path),
        "wrapper_sha256": {path.name: sha256(path) for path in (wrapper, cpp, exports)},
        "library": str(output), "library_sha256": sha256(output),
        "fortran_compiler": fc_version, "cxx_compiler": cc_version,
        "commands": commands, "compiler_output": outputs,
        "platform": platform.platform(), "asymptotic_oracles": checks,
        "precision": "ordinary DOUBLE PRECISION; no real/integer kind promotion",
    }
    output.with_suffix(".build.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--source", type=Path, help="offline copy of the exact pinned upstream source")
    parser.add_argument("--fc", help="GNU Fortran command, defaults to FC/gfortran")
    parser.add_argument("--cxx", help="C++17 compiler command, defaults to CXX/g++")
    args = parser.parse_args()
    report = build(args.output, source=args.source, compiler=args.fc, cxx=args.cxx)
    print(json.dumps({key: report[key] for key in ("library", "library_sha256", "asymptotic_oracles")}, indent=2))


if __name__ == "__main__":
    main()
