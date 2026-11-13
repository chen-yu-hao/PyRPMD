"""Reproduce the T52 structural checks against a pinned external checkout.

No RPMDrate package or PES is loaded. Raw CLI logs and converted files go to
the requested new output directory; summary.json is the compact evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from pyrpmd.constants import BOHR_ANGSTROM
from pyrpmd.input import JobSpec
from pyrpmd.rpmdrate_import import UPSTREAM_COMMIT
from pyrpmd.units import parse_quantity


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "H+H2": {"fragments": [[0, 1], [2]], "forming_bonds": [[1, 2]], "breaking_bonds": [[0, 1]],
             "equivalent_paths": [[1, 0, 2]], "rinf_angstrom": 15.8754, "windows": 110, "chains": 200},
    "H+CH4": {"fragments": [[0, 1, 2, 3, 4], [5]], "forming_bonds": [[3, 5]], "breaking_bonds": [[1, 3]],
              "equivalent_paths": [[0, 1, 2, 4, 3, 5], [0, 1, 3, 2, 4, 5], [3, 1, 2, 0, 4, 5]],
              "rinf_angstrom": 15.8754, "windows": 110, "chains": 200},
    "H+C2H6": {"fragments": [list(range(8)), [8]], "forming_bonds": [[7, 8]], "breaking_bonds": [[4, 7]],
               "equivalent_paths": [[0, 1, 2, 3, 4, 7, 6, 5, 8], [0, 1, 2, 3, 4, 5, 7, 6, 8]],
               "rinf_angstrom": 7.9377, "windows": 106, "chains": 150},
    "OH+CH4": {"fragments": [[0, 1, 2, 3, 4], [5, 6]], "forming_bonds": [[3, 5]], "breaking_bonds": [[1, 3]],
               "equivalent_paths": [[0, 1, 2, 4, 3, 5, 6], [0, 1, 3, 2, 4, 5, 6], [3, 1, 2, 0, 4, 5, 6]],
               "rinf_angstrom": 7.9377, "windows": 110, "chains": 200},
}


def cli(log, *args, success=True):
    result = subprocess.run([sys.executable, "-m", "pyrpmd", *map(str, args)],
                            text=True, capture_output=True, timeout=120)
    log.write_text(result.stdout + result.stderr)
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0, "Unsupported upstream settings were accepted silently"
    return result.returncode


def geometry_literal(path):
    # Independent static reading of the literal TS, without using the importer.
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "transitionState":
            value = next(keyword.value for keyword in node.keywords if keyword.arg == "geometry")
            rows, unit = ast.literal_eval(value)
            assert unit in ("bohr", "angstrom")
            return np.asarray(rows) / (1 if unit == "bohr" else BOHR_ANGSTROM)
    raise AssertionError("No literal transition-state geometry")


def topology(reaction, permutation):
    def sets(rows):
        return sorted(sorted(permutation[i] for i in row) for row in rows)
    inverse = np.argsort(permutation)
    paths = [[permutation[path[inverse[i]]] for i in range(len(permutation))]
             for path in reaction["equivalent_paths"]]
    return {"fragments": sets(reaction["fragments"]), "forming_bonds": sets(reaction["forming_bonds"]),
            "breaking_bonds": sets(reaction["breaking_bonds"]), "equivalent_paths": sorted(paths)}


def validate(upstream, output):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    assert revision == UPSTREAM_COMMIT, f"Expected {UPSTREAM_COMMIT}, found {revision}"
    if output.exists():
        raise ValueError("Use a new validation output directory")
    output.mkdir(parents=True)
    summary = {"upstream_commit": revision,
               "pyrpmd_base": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               "environment": {"hostname": platform.node(), "platform": platform.platform(),
                               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                               "cpu_count": os.cpu_count(), "affinity_count": len(os.sched_getaffinity(0)),
                               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                               "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
                "PES_evaluations": 0,
               "external_dynamics": "未实测 (not run)", "cases": {}}
    for name, expected in CASES.items():
        source = upstream / "examples" / name / "input.py"
        target = output / name
        args = ["import-rpmdrate", source, "--temperature", 1000, "--beads", 4, "--output", target]
        strict_status = cli(output / f"{name}-strict.log", *args, success=False)
        assert not target.exists()
        cli(output / f"{name}-import.log", *args, "--allow-unmapped")
        valid = cli(output / f"{name}-validate.log", "validate", target / "rate.inp")
        preview = cli(output / f"{name}-prepare.log", "prepare", target / "rate.inp")
        report = json.loads((target / "import_report.json").read_text())
        config, system = JobSpec.from_file(target / "rate.inp").resolve()
        reaction = config["reaction"]
        for key in ("fragments", "forming_bonds", "breaking_bonds", "equivalent_paths"):
            assert reaction[key] == expected[key], (name, key, reaction[key], expected[key])
        assert reaction["symmetry_factor"] == 1
        rinf = parse_quantity(reaction["rinf"], "length")
        np.testing.assert_allclose(rinf, expected["rinf_angstrom"] / BOHR_ANGSTROM, rtol=1e-14)
        np.testing.assert_allclose(system.geometry, geometry_literal(source), rtol=1e-14, atol=1e-14)
        windows = preview["umbrella_preparation"]["window_plan"]
        assert len(windows) == expected["windows"]
        stiffness = np.array([window["stiffness_au"][0][0] for window in windows])
        np.testing.assert_allclose(stiffness, 0.1 * 1000, rtol=1e-15)
        for window in windows:
            assert (window["chains"], window["equilibration_steps"], window["production_steps"],
                    window["sample_stride_steps"]) == (expected["chains"], 200000, 1000000, 1)
        rc = preview["recrossing_preparation"]
        assert (rc["planned_configurations"], rc["planned_pairs"], rc["planned_trajectories"]) == (1000, 50000, 100000)
        assert valid["valid"] and not preview["ready"]
        summary["cases"][name] = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "strict_exit": strict_status, "strict_output_files": 0,
            "allow_unmapped_validate": True, "static_prepare": True,
            "natoms": system.natoms, "windows": len(windows), "chains_per_window": expected["chains"],
            "equilibration_steps": 200000, "production_steps": 1000000,
            "rinf_bohr": rinf, "fragments": reaction["fragments"], "forming_bonds": reaction["forming_bonds"],
            "breaking_bonds": reaction["breaking_bonds"], "equivalent_paths": reaction["equivalent_paths"],
            "geometry_max_abs_error_bohr": float(np.max(np.abs(system.geometry - geometry_literal(source)))),
            "kforce_hartree": 100.0, "stiffness_max_abs_error_hartree": float(np.max(np.abs(stiffness - 100))),
            "parent_configurations": 1000, "child_pairs": 50000, "child_trajectories": 100000,
            "unmapped": report["unmapped"],
        }
        print(f"{name}: {len(windows)} windows, strict refusal, validate/prepare passed", flush=True)

    imported, system = JobSpec.from_file(output / "H+H2/rate.inp").resolve()
    handwritten, reference = JobSpec.from_file(Path(__file__).with_name("h3-reference.inp")).resolve()
    for key in ("fragments", "forming_bonds", "breaking_bonds", "equivalent_paths", "symmetry_factor"):
        assert imported["reaction"][key] == handwritten["reaction"][key]
    np.testing.assert_array_equal(system.geometry, reference.geometry)
    np.testing.assert_allclose(parse_quantity(imported["reaction"]["rinf"], "length"),
                               parse_quantity(handwritten["reaction"]["rinf"], "length"), rtol=1e-14)
    summary["h3_handwritten_identity"] = "passed (lengths compared in bohr)"
    benchmark = ROOT / "benchmarks" / "h3" / "rpmd.py"
    module_spec = importlib.util.spec_from_file_location("t52_h3_benchmark", benchmark)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    standard, _ = module.configuration(stage="rate", output=output / "unused-benchmark-output")
    convergence, _ = JobSpec.from_file(benchmark.parent / "convergence_inputs" / "base.inp").resolve()
    comparison = topology(imported["reaction"], [2, 1, 0])
    for config in (standard, convergence):
        assert comparison == topology(config["reaction"], [0, 1, 2])
    summary["h3_bkmp2_topology"] = {"configuration_rate": True, "convergence_base": True,
                                   "upstream_to_benchmark_permutation": [2, 1, 0],
                                   "benchmark_rinf_bohr": parse_quantity(standard["reaction"]["rinf"], "length"),
                                   "geometry_and_rinf_are_not_identity_comparisons": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"H3 handwritten identity and both BKMP2 topology comparisons passed; {output / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.upstream.resolve(), args.output.resolve())
