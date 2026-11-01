"""Build a static, nb=1 interface to exact author-attributed EAM4 source.

No network access. Obtain the explicitly public Cambridge archive separately.
The third-party source, extracted suffix, compiler outputs and binary must stay
outside this repository. Only the OPTIM-dependent front wrapper is excluded;
the coefficient-bearing suffix is copied byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ARCHIVE_URL = "https://www-wales.ch.cam.ac.uk/svn.tar.bz2"
ARCHIVE_SHA256 = "8463fbbbab3f613c1daa021df99cf89791be3ca61c7f3b9c2f5307e484bf8029"
ARCHIVE_BYTES = 122758544
MEMBER = "./wales/OPTIM/source/nimetpot.f90"
SOURCE_SHA256 = "b294b106cac20038eb941037b63f35dcce374ed40559d68217f5751b90d514a6"
SOURCE_BYTES = 43812
SUFFIX_MARKER = b"subroutine eamnih_forces_umb("
ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_source(source_file: Path | None, source_archive: Path | None) -> bytes:
    if (source_file is None) == (source_archive is None):
        raise ValueError("provide exactly one source file or source archive")
    if source_file is not None:
        content = source_file.read_bytes()
    else:
        assert source_archive is not None
        if source_archive.stat().st_size != ARCHIVE_BYTES or sha256_file(source_archive) != ARCHIVE_SHA256:
            raise ValueError("archive length/SHA256 does not match the inspected public release")
        content = None
        # Never extract paths or symlinks from the third-party archive.
        with tarfile.open(source_archive, "r|bz2") as archive:
            for member in archive:
                if member.name == MEMBER:
                    if not member.isfile() or member.size != SOURCE_BYTES:
                        raise ValueError("unexpected candidate source member")
                    content = archive.extractfile(member).read()
                    break
        if content is None:
            raise ValueError("pinned source member is absent")
    if len(content) != SOURCE_BYTES or hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("source length/SHA256 does not match the inspected EAM4 implementation")
    return content


def suffix(content: bytes) -> bytes:
    if content.count(SUFFIX_MARKER) != 1:
        raise ValueError("ambiguous standalone routine boundary")
    return content[content.index(SUFFIX_MARKER):]


def build(source_file: Path | None, source_archive: Path | None, output: Path, compiler: str) -> dict:
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("third-party source and binary output must be outside the repository")
    if output.exists() and any(output.iterdir()):
        raise ValueError("build output must be a new or empty directory")
    content = checked_source(source_file, source_archive)
    tail = suffix(content)
    output.mkdir(parents=True, exist_ok=True)
    original = output / "nimetpot.original.f90"
    standalone = output / "eam4.original-suffix.f90"
    original.write_bytes(content)
    standalone.write_bytes(tail)
    bridge_source = Path(__file__).with_name("eam4_bridge.f90")
    bridge = output / "eam4_bridge.f90"
    shutil.copyfile(bridge_source, bridge)
    library = output / "libeam4_static.so"
    flags = ["-O2", "-std=f2008", "-fPIC", "-shared", "-ffree-line-length-none",
             "-fcheck=bounds", "-frecursive", "-fno-fast-math"]
    command = [compiler, *flags, str(standalone), str(bridge), "-o", str(library)]
    before = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
    (output / "compile.stdout.txt").write_text(completed.stdout)
    (output / "compile.stderr.txt").write_text(completed.stderr)
    report = {
        "schema": "pyrpmd.materials-reference.eam4-static-build.v1",
        "scope": "Unmodified EAM4 coefficient routines; nb=1 static checks only; no paper diffusion reproduction",
        "archive": {"url": ARCHIVE_URL, "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_BYTES, "member": MEMBER},
        "original_source": {"path": str(original), "sha256": SOURCE_SHA256, "bytes": len(content)},
        "suffix": {"path": str(standalone), "sha256": hashlib.sha256(tail).hexdigest(), "bytes": len(tail),
                   "start_byte_zero_based": len(content)-len(tail), "transformation": "unchanged byte suffix only"},
        "source_files": {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
                         str(bridge_source.relative_to(ROOT)): sha256_file(bridge_source)},
        "compiler_version": subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                           timeout=5, check=True).stdout,
        "command": command, "returncode": completed.returncode, "elapsed_seconds": time.monotonic()-before,
        "compile_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "compile_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "energy_convention": "Original fixed--fixed Ni pair-energy constant omitted; no comparisons across fixed partitions",
        "derivative_convention": "dV/dq for H and mobile Ni; original zero fixed-Ni outputs are not physical reaction forces",
        "units": {"q": "angstrom", "energy": "eV", "gradient": "eV/angstrom"},
        "license_source": "https://www-wales.ch.cam.ac.uk/OPTIM/ (public distribution described as GPL)",
    }
    if completed.returncode == 0:
        report["library"] = {"path": str(library), "bytes": library.stat().st_size, "sha256": sha256_file(library)}
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    if completed.returncode:
        raise RuntimeError(f"compile failed; preserved logs/report in {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-file", type=Path)
    inputs.add_argument("--source-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fc", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_file, args.source_archive, args.output, args.fc), indent=2))


if __name__ == "__main__":
    main()
