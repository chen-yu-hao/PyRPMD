"""Verify archived review evidence and run new regressions from an installed wheel.

Build first with ``python -m build --no-isolation --outdir <dist>``; then run
``python docs/validation/pr63-independent-review/verify_distribution.py <dist>``.
No source-tree import is used by the isolated wheel subprocess.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    output = Path(__file__).resolve().parent
    repo = output.parents[2]
    dist = Path(sys.argv[1]).resolve()
    wheels = sorted(dist.glob("*.whl"))
    sources = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("use a directory containing exactly one wheel and one sdist")
    wheel, sdist = wheels[0], sources[0]
    production = [
        "src/pyrpmd/potentials/contraction.py", "src/pyrpmd/replica_exchange.py",
        "src/pyrpmd/workflows/nonadiabatic_rate.py", "src/pyrpmd/workflows/spectroscopy.py",
    ]
    tests = ["test_spectroscopy_trace_identity.py", "test_replica_exchange_composition_review.py",
             "test_tier2_nonadiabatic_ml_review.py"]
    suffixes = {".py", ".md", ".json", ".csv", ".npz", ".pt", ".png", ".pdf"}
    expected = [repo / name for name in production] + [repo / "tests" / name for name in tests]
    expected += [path for path in sorted(output.rglob("*")) if path.is_file()
                 and path.suffix in suffixes and "__pycache__" not in path.parts
                 and path.name not in {"package-verification.json", "delivery-manifest.json"}]
    verified = {}
    with tarfile.open(sdist) as archive:
        root = archive.getnames()[0].split("/")[0]
        for path in expected:
            name = path.relative_to(repo).as_posix()
            stream = archive.extractfile(root + "/" + name)
            if stream is None or stream.read() != path.read_bytes():
                raise AssertionError(f"sdist mismatch: {name}")
            verified[name] = sha256(path.read_bytes())
    with zipfile.ZipFile(wheel) as archive:
        for name in production:
            if archive.read(name.removeprefix("src/")) != (repo / name).read_bytes():
                raise AssertionError(f"wheel source mismatch: {name}")
        native = [name for name in archive.namelist() if name.startswith("pyrpmd/")
                  and name.endswith(".so")]
        if not any("_native." in name for name in native) or not any("_thermal." in name for name in native):
            raise AssertionError("native propagation/thermal extensions absent from wheel")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="pr63-wheel-", dir=repo.parent) as temp:
        work = Path(temp)
        installed = work / "installed"
        install = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                   "--target", str(installed), str(wheel)]
        with (output / "package-install.log").open("w") as log:
            subprocess.run(install, cwd=work, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        testdir = work / "tests"
        testdir.mkdir()
        for name in tests + ["test_spectroscopy_workflow.py"]:
            shutil.copy2(repo / "tests" / name, testdir / name)
        code = '''import importlib, pathlib, sys
target = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
for name in ("pyrpmd", "pyrpmd._native", "pyrpmd._thermal", "pyrpmd.replica_exchange",
             "pyrpmd.workflows.spectroscopy", "pyrpmd.workflows.nonadiabatic_rate"):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(target), (name, str(path))
    print("installed", name, path.relative_to(target))
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
'''
        command = [sys.executable, "-I", "-c", code, str(installed), "-q", "-c", "/dev/null",
                   "--confcutdir=" + str(work), "--junitxml=" + str(output / "package-wheel-tests.xml")]
        command += [str(testdir / name) for name in tests]
        with (output / "package-wheel-tests.log").open("w") as log:
            result = subprocess.run(command, cwd=work, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
    report = {
        "wheel": {"name": wheel.name, "sha256": sha256(wheel.read_bytes())},
        "sdist": {"name": sdist.name, "sha256": sha256(sdist.read_bytes())},
        "source_tree": str(repo), "verified_sdist_files": verified,
        "verified_wheel_sources": production, "wheel_native_extensions": native,
        "raw_npz_files": sum(name.endswith(".npz") for name in verified),
        "torchscript_models": sum(name.endswith(".pt") for name in verified),
        "isolated_wheel_test_returncode": result.returncode,
        "isolation": "python -I; installed target first; import origins asserted; tests copied outside checkout",
        "test_files": tests, "python": sys.version,
    }
    (output / "package-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "verified_sdist_files"}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
