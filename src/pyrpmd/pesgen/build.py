"""Source-pinned optional Caracal build and immutable native PES bundles.

No compiler or network action occurs on import. Builds run only on explicit calls.
The upstream checkout is copied, never edited in place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

from .reference import atomic_json, canonical_hash, file_hash

UPSTREAM_COMMIT = "7f95ce9818b7f6d81d7cb3b634f360373c090820"
PATCH_VERSION = "pyrpmd-caracal-safety-1"
_NATIVE = Path(__file__).with_name("_native")
# Set from the pinned source, not from generated binaries or build products.
SOURCE_DIGEST = "2e55993184aa52a7afb11a71b8555379fb2dba957cebf3a8df5e0d3effc2f90c"


def source_digest(root):
    root = Path(root)
    paths = [root / "Makefile"] + sorted(p for p in (root / "src").rglob("*")
             if p.is_file() and p.suffix.lower() in {".f", ".f90", ".c", ".h", ".inc", ".py"})
    return canonical_hash({p.relative_to(root).as_posix(): file_hash(p) for p in paths})


def run_command(command, cwd, log, timeout=1800):
    """Run an explicit argv, never a shell; preserve diagnostics on any failure."""
    if not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a nonempty string argv")
    with open(log, "w") as stream:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            raise RuntimeError(f"cannot start command; see {log}: {exc}") from exc
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if code:
        tail = Path(log).read_text(errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({code}); see {log}\n{tail}")


def patch_source(root):
    """Small audited corrections; no force-field equations are replaced.

    Called on a fresh, fully source-hashed copy. Full diffs remain in the build.
    """
    root = Path(root)
    changes = {
        "gethirsh.f90": [
            ("ok=.false.", "ok=.false.\ncsum=0.d0"),
            ("if(index(a80,'HIRSHFELD ANALYSIS').ne.0) then",
             "if(index(a80,'HIRSHFELD ANALYSIS').ne.0) then\n         ok=.true.\n         csum=0.d0"),
        ],
        "main_gen.f90": [
            ("      call docm5(n,at,.true.,xyz,chir,q)",
             "      chrg=dble(nint(sum(chir)))\n      call docm5(n,at,.true.,xyz,chir,q)"),
            ("     ! stop 'no charges!'", "     stop 'missing Hirshfeld charge section'"),
        ],
        "ff_clean.f90": [
            ("integer::iat,i,j,k,l,m,new,ii(6,ndim),minus,plus",
             "integer::iat,i,j,k,l,m,new,minus,plus\ninteger,allocatable::ii(:,:)"),
            ("allocate(vv(2+3*ntterm,ndim))", "allocate(vv(2+3*ntterm,ndim),ii(6,ndim))"),
            ("deallocate(vv)", "deallocate(vv,ii)"),
        ],
    }
    import difflib
    patches = []
    for name, replacements in changes.items():
        path = root / "src" / name
        old = path.read_text(); new = old
        for before, after in replacements:
            if new.count(before) != 1:
                raise ValueError(f"unexpected upstream source for patch: {name}")
            new = new.replace(before, after)
        path.write_text(new)
        patches += difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile=f"a/src/{name}", tofile=f"b/src/{name}")
    (root / "pyrpmd-safety.patch").write_text("".join(patches))


def compile_worker(root, output, *, fortran="mpifort", link_flags=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if link_flags is None:
        link_flags = ["-llapack", "-lblas", "-lfftw3"] + shlex.split(subprocess.check_output(
            ["python3-config", "--embed", "--ldflags"], text=True))
    command = shlex.split(fortran) + ["-O2", "-fopenmp", "-fno-align-commons", "-I", str(root / "src"),
               str(_NATIVE / "worker.f90"), str(root / "src/libcaracal.a"),
               "-Wl,--no-as-needed", *link_flags, "-o", str(output)]
    run_command(command, root, output.with_suffix(".build.log"))
    return command


def build_native(source, destination, *, fortran="mpifort", c_compiler="mpicc", cxx="c++"):
    """Build pinned Caracal + isolated worker on Linux. Requires MPI/FFTW/BLAS.

    The destination must be new. Compilation is serial because upstream module
    dependencies are not safe under parallel make.
    """
    if not sys.platform.startswith("linux"):
        raise ValueError("the native process-isolation bridge currently requires Linux")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("build destination must be outside the upstream checkout")
    if source_digest(source) != SOURCE_DIGEST:
        raise ValueError(f"Caracal sources do not match pinned commit {UPSTREAM_COMMIT}")
    if destination.exists():
        raise FileExistsError("native build destination must be new")
    destination.mkdir(parents=True)
    root = destination / "Caracal"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns(".git", "*.o", "*.mod", "*.a", "*.x"))
    patch_source(root)
    (root / "bin").mkdir(exist_ok=True)
    flags = "-cpp -fopenmp -fPIC -fno-align-commons -DPARALLEL -fallow-argument-mismatch -O2 -I/usr/include"
    libraries = ["-llapack", "-lblas", "-lfftw3"] + shlex.split(subprocess.check_output(
        ["python3-config", "--embed", "--ldflags"], text=True))
    command = ["make", "-f", "../Makefile", f"FC={fortran}", f"CC={c_compiler}",
               f"BINDIR={root / 'bin'}", f"FFLAGS={flags}",
               "LINKFLAGS=-fopenmp -Wl,--no-as-needed " + shlex.join(libraries)]
    run_command(command, root / "src", destination / "caracal-build.log")
    worker = destination / "caracal_worker.x"
    worker_command = compile_worker(root, worker, fortran=fortran, link_flags=libraries)
    record = {"schema": "pyrpmd.caracal-build/1", "upstream_commit": UPSTREAM_COMMIT,
              "source_digest": SOURCE_DIGEST, "patch_version": PATCH_VERSION,
              "patch_sha256": file_hash(root / "pyrpmd-safety.patch"),
              "commands": [command, worker_command], "cxx": cxx,
              "worker_sha256": file_hash(worker),
              "worker_source_sha256": file_hash(_NATIVE / "worker.f90"),
              "executables": {name: file_hash(root / "bin" / f"{name}.x")
                              for name in ("qmdffgen", "evbopt", "explore")}}
    atomic_json(destination / "build.json", record)
    return destination


def load_build(path):
    path = Path(path).resolve()
    record = json.loads((path / "build.json").read_text())
    if (record.get("schema") != "pyrpmd.caracal-build/1" or record.get("source_digest") != SOURCE_DIGEST
            or record.get("patch_version") != PATCH_VERSION
            or record.get("worker_source_sha256") != file_hash(_NATIVE / "worker.f90")):
        raise ValueError("native build schema/source/patch mismatch")
    if file_hash(path / "caracal_worker.x") != record.get("worker_sha256"):
        raise ValueError("native worker integrity mismatch")
    for name in ("qmdffgen", "evbopt", "explore"):
        if file_hash(path / "Caracal/bin" / f"{name}.x") != record.get("executables", {}).get(name):
            raise ValueError(f"native executable integrity mismatch: {name}")
    return record


def bundle_native(build, files, symbols, destination, *, charge=0, spin=0, timeout=120.0, provenance=None, force_mode="analytic", force_step=1e-4):
    """Compile a C ABI v1 library with embedded text model data and fixed identity.

    A persistent Fortran worker owns all upstream global state. Native C++
    propagation calls this library directly; no Python is called per time step.
    """
    build, destination = Path(build).resolve(), Path(destination).resolve()
    import math
    if force_mode not in ("analytic", "fd") or not math.isfinite(force_step) or not 1e-7 <= force_step <= 1e-2:
        raise ValueError("force_mode must be analytic/fd and step in [1e-7,1e-2] bohr")
    record = load_build(build)
    if destination.exists():
        raise FileExistsError("bundle destination must be new")
    if not 0 < timeout <= 3600 or len(symbols) < 2:
        raise ValueError("invalid timeout/atom count")
    if isinstance(charge, bool) or not isinstance(charge, int) or isinstance(spin, bool) or not isinstance(spin, int) or spin < 0:
        raise ValueError("charge/spin must be integers and spin nonnegative")
    import re
    for name, text in files.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,39}", name) or not isinstance(text, str) or "\0" in text:
            raise ValueError("embedded files require short names and NUL-free text")
    if not {"model.key", "start.xyz"} <= set(files):
        raise ValueError("native model requires model.key and start.xyz")
    from ..system import _Z
    numbers = [_Z[s] for s in symbols]
    identity = {"natoms": len(symbols), "symbols": list(symbols), "atomic_numbers": numbers,
                "charge": charge, "multiplicity": spin + 1}
    destination.mkdir(parents=True)
    worker = destination / "caracal_worker.x"
    shutil.copy2(build / "caracal_worker.x", worker)
    data = worker.read_bytes(); fnv = 14695981039346656037
    for b in data:
        fnv = ((fnv ^ b) * 1099511628211) & ((1 << 64) - 1)
    metadata = {"schema": "pyrpmd.pes/1", "abi_version": 1, "name": "Caracal fitted PES",
                "natoms": len(symbols), "atomic_numbers": numbers, "layout": "B,N,3:C", "dtype": "float64",
                "position_unit": "bohr", "energy_unit": "hartree", "force_unit": "hartree/bohr",
                "boundary_conditions": "nonperiodic", "state": {"charge": charge, "multiplicity": spin+1},
                "capabilities": {"stateful": False, "thread_safety": "serialized_per_context", "native_batch": True, "force_kind": "analytic" if force_mode == "analytic" else "finite_difference", "requires_bead_affinity": False, "execution_device": "cpu"},
                "force_step_bohr": force_step if force_mode == "fd" else None,
                "execution": "persistent isolated native worker; serial frames per context",
                "model_sha256": canonical_hash(files), "worker_sha256": record["worker_sha256"],
                "upstream_commit": UPSTREAM_COMMIT, "patch_version": PATCH_VERSION,
                "provenance": provenance or {}, "validated_physical_accuracy": False}
    def literal(text):
        # JSON escaping is also a valid C++ string literal for our ASCII metadata/files.
        return json.dumps(text, ensure_ascii=True)
    header = "#include <cstdint>\n"
    header += "static const char FORCE_MODE[]="+literal(force_mode)+", FORCE_STEP[]="+literal(str(force_step))+";\n"
    header += f"static const int64_t NATOMS={len(symbols)}, TIMEOUT_MS={int(timeout*1000)};\n"
    header += f"static const uint64_t WORKER_FNV={fnv}ULL, WORKER_BYTES={len(data)}ULL;\n"
    header += "static const struct {const char* key; const char* value;} IDENTITIES[]={\n"
    header += ",\n".join("{"+literal(k)+","+literal(json.dumps(v, separators=(',', ':')))+"}" for k,v in identity.items()) + "};\n"
    header += "static const struct {const char* name; const char* content;} MODEL_FILES[]={\n"
    header += ",\n".join("{"+literal(k)+","+literal(v)+"}" for k,v in sorted(files.items()))+"};\n"
    header += "static const char METADATA[]="+literal(json.dumps(metadata, sort_keys=True))+";\n"
    with tempfile.TemporaryDirectory(dir=destination, prefix="compile-") as td:
        td = Path(td); (td / "model_data.h").write_text(header)
        cmd = shlex.split(record["cxx"]) + ["-std=c++17", "-O2", "-fPIC", "-shared", "-pthread",
                "-I", str(_NATIVE), "-I", str(td), str(_NATIVE / "bridge.cpp"), "-ldl", "-o", str(destination / "libpes.so")]
        run_command(cmd, td, destination / "bridge-build.log")
    # Preserve parameters and license/provenance as well as the embedded copy.
    model = destination / "model"; model.mkdir()
    for name, text in files.items():
        (model / name).write_text(text)
    notices = destination / "third-party-notices"; notices.mkdir()
    upstream = build / "Caracal"
    shutil.copy2(upstream / "README.md", notices / "CARACAL-README.md")
    for source in upstream.rglob("*"):
        if source.is_file() and any(word in source.name.lower() for word in ("license", "copying", "notice")):
            relative = source.relative_to(upstream)
            target = notices / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (notices / "SOURCE.txt").write_text("Caracal upstream commit " + UPSTREAM_COMMIT +
        "\nBundled source/header notices remain in the separately supplied Caracal checkout.\n" +
        "This binary bundle does not relicense upstream or bundled third-party components.\n")
    atomic_json(destination / "bundle.json", {"schema": "pyrpmd.caracal-bundle/1", "metadata": metadata,
                "build": record, "library_sha256": file_hash(destination / "libpes.so"),
                "files": {k: canonical_hash(v) for k,v in files.items()}})
    return destination / "libpes.so"
