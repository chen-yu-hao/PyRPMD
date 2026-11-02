"""Run the sealed integrated method selection using one isolated interpreter.

Build products stay outside the checkout. Temporary, ABI-specific symlinks make
those exact products discoverable by this interpreter and every child process;
the existing CPython 3.12 binaries are never replaced. No test/source is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timezone
from xml.etree import ElementTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def environment():
    return dict(
        python=sys.version, executable=sys.executable, prefix=sys.prefix,
        platform=platform.platform(), abi=sysconfig.get_config_var("SOABI"),
        extensions=sysconfig.get_config_var("EXT_SUFFIX"),
        distributions={d.metadata["Name"]: d.version for d in metadata.distributions()},
        affinity=sorted(os.sched_getaffinity(0)),
        environment={k: os.environ.get(k) for k in [
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "OPENMM_CPU_THREADS", "TMPDIR", "PYRPMD_CBE2008_LIBRARY",
        ]},
    )


def runtime_snapshot(repo, build):
    import pyrpmd
    assert Path(pyrpmd.__file__).resolve() == repo / "src/pyrpmd/__init__.py"
    native = {}
    for name in ["_native", "_thermal", "_electronic"]:
        module = importlib.import_module("pyrpmd." + name)
        path = Path(module.__file__)
        expected = build / "lib/pyrpmd" / (name + sysconfig.get_config_var("EXT_SUFFIX"))
        assert path.resolve() == expected.resolve(), (path, expected)
        native[name] = dict(path=str(path), resolved=str(path.resolve()), sha256=sha(path))
    import openmm
    assert metadata.version("OpenMM") == "8.6.1"
    return dict(
        environment=environment(), native=native,
        pyrpmd_import=str(Path(pyrpmd.__file__).resolve()),
        openmm=dict(version=metadata.version("OpenMM"), path=openmm.__file__,
                    platforms=[openmm.Platform.getPlatform(i).getName()
                               for i in range(openmm.Platform.getNumPlatforms())]),
    )


def source_guards(before, repo):
    roots = {"integrated": repo, "original": Path("/workspace/scratch/a5e4820daedd/PyRPMD")}
    checks = {}
    for label, root in roots.items():
        changed = [p for p, digest in before[label].items()
                   if not (root / p).is_file() or sha(root / p) != digest]
        checks[label] = dict(files_checked=len(before[label]), changed=changed)
    changed = [p for p, digest in before["external_models"].items()
               if not Path(p).is_file() or sha(p) != digest]
    checks["external_models"] = dict(files_checked=len(before["external_models"]), changed=changed)
    checks["all_unchanged"] = all(not v["changed"] for v in checks.values())
    return checks


class RuntimeAudit:
    def __init__(self, repo, build, output):
        self.repo, self.build, self.output = repo, build, output

    def pytest_sessionstart(self, session):
        write(self.output / "pytest-runtime-start.json", runtime_snapshot(self.repo, self.build))

    def pytest_sessionfinish(self, session, exitstatus):
        snapshot = runtime_snapshot(self.repo, self.build)
        snapshot["pytest_exitstatus"] = int(exitstatus)
        imported = {}
        for name, module in sorted(sys.modules.items()):
            if not name.startswith(("pyrpmd", "benchmarks.h3", "benchmarks.hch4", "benchmarks.water")):
                continue
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file():
                imported[name] = dict(path=str(Path(path).resolve()), sha256=sha(path))
        snapshot["actual_loaded_project_modules"] = imported
        write(self.output / "pytest-runtime-finish.json", snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["py313", "py314"], required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    repo, build = args.repo.resolve(), args.work.resolve()
    if args.probe:
        print(json.dumps(runtime_snapshot(repo, build), sort_keys=True))
        return 0
    evidence = Path(__file__).resolve().parent
    output = evidence / args.label
    selection = json.loads((evidence / "selection.json").read_text())
    before = json.loads((evidence / "guard-before.json").read_text())
    assert Path.cwd() == repo and os.environ["PYTHONPATH"] == "src:."
    assert len(os.sched_getaffinity(0)) == 1
    assert source_guards(before, repo)["all_unchanged"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == selection["integrated_head"]
    assert all(sha(repo / p) == d for p, d in selection["selected_test_files_sha256"].items())
    assert sys.version_info[:2] == ((3, 13) if args.label == "py313" else (3, 14))
    installed = environment()
    write(output / "environment-after-install.json", installed)
    baseline = json.loads((output / "environment-before.json").read_text())["distributions"]
    assert all(installed["distributions"].get(n) == v for n, v in baseline.items())
    added = {n: v for n, v in installed["distributions"].items() if n not in baseline}
    assert {n.lower(): v for n, v in added.items()} == {"openmm": "8.6.1"}, added
    build_command = [sys.executable, "setup.py", "build_ext", "--parallel", "1", "--force",
                     "--build-temp", str(build / "objects"), "--build-lib", str(build / "lib")]
    started = time.perf_counter()
    report = dict(schema="pyrpmd.integrated-python-abi-method-gate.v1",
                  started_utc=datetime.now(timezone.utc).isoformat(), label=args.label,
                  source_head=selection["integrated_head"], driver_sha256=sha(__file__),
                  selected_tests_sha256=sha(evidence / "selection.json"),
                  guarded_files_before_sha256=sha(evidence / "guard-before.json"),
                  build_command=build_command, installed_only=added)
    links = []
    exit_code = 1
    try:
        with (output / "native-build.log").open("w") as stream:
            result = subprocess.run(build_command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
        report["build_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("native extension build failed; see native-build.log")
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        assert "312" not in suffix
        report["native_products"] = {}
        for name in ["_native", "_thermal", "_electronic"]:
            product = build / "lib/pyrpmd" / (name + suffix)
            link = repo / "src/pyrpmd" / product.name
            assert product.is_file() and not link.exists() and not link.is_symlink()
            link.symlink_to(product)
            links.append((link, product))
            report["native_products"][name] = dict(path=str(product), sha256=sha(product),
                                                    bytes=product.stat().st_size, temporary_import_link=str(link))
        # A separately spawned interpreter must see the same new ABI binaries.
        child_command = [sys.executable, str(Path(__file__).resolve()), "--label", args.label,
                         "--repo", str(repo), "--work", str(build), "--probe"]
        child = subprocess.run(child_command, cwd=repo, capture_output=True, text=True)
        (output / "child-import.log").write_text(child.stderr)
        assert child.returncode == 0, child.stderr
        write(output / "child-import.json", json.loads(child.stdout))
        report["child_import_command"] = child_command
        import pytest
        files = [p for g in selection["groups"].values() for p in g["files"]]
        pytest_args = ["-q", "-p", "no:cacheprovider", *files,
                       "--basetemp=" + str(build / "pytest-tmp"),
                       "--junitxml=" + str(output / "pytest.xml")]
        report["pytest_arguments"] = pytest_args
        exit_code = int(pytest.main(pytest_args, plugins=[RuntimeAudit(repo, build, output)]))
        report["pytest_exit_code"] = exit_code
        xml = ElementTree.parse(output / "pytest.xml")
        report["junit_counts"] = {k: sum(int(s.attrib.get(k, "0")) for s in xml.iter("testsuite"))
                                  for k in ["tests", "failures", "errors", "skipped"]}
        case_ids = [case.attrib.get("classname", "") + "::" + case.attrib["name"]
                    for case in xml.iter("testcase")]
        report["distinct_case_ids"] = len(set(case_ids))
        report["case_ids"] = case_ids
        report["selection_count_matches"] = len(set(case_ids)) == selection["expected_distinct_tests_per_interpreter"]
        report["junit_sha256"] = sha(output / "pytest.xml")
    except BaseException as error:
        report["driver_error"] = repr(error)
        raise
    finally:
        for link, product in reversed(links):
            assert link.is_symlink() and link.resolve() == product.resolve()
            link.unlink()
        report["temporary_ABI_links_removed"] = True
        report["guard_after"] = source_guards(before, repo)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = (exit_code == 0 and report.get("selection_count_matches", False)
                            and report["guard_after"]["all_unchanged"]
                            and report.get("junit_counts", {}).get("skipped", 1) == 0)
        write(output / "compatibility.json", report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
