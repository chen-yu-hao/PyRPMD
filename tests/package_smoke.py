"""Exercise an installed distribution from outside its source tree.

Run with the Python belonging to the independent wheel/sdist install. The
source-root argument supplies examples only; it is never added to sys.path.
This script is used by package CI as well as local release verification.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if Path.cwd().resolve().is_relative_to(source):
        raise RuntimeError("Run package smoke from outside the extracted source tree")
    import pyrpmd
    from pyrpmd import _native
    from pyrpmd.dynamics import Dynamics, State, INTEGRATORS, native_capabilities
    from pyrpmd.potentials import HarmonicPotential

    installed = Path(pyrpmd.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()) or installed.is_relative_to(source):
        raise RuntimeError(f"Imported source/editable tree instead of independent installation: {installed}")
    version = importlib.metadata.version("PyRPMD")
    if pyrpmd.__version__ != version:
        raise RuntimeError("Installed metadata and Python package versions disagree")
    modules = [
        "pyrpmd.checkpoint", "pyrpmd.reactions", "pyrpmd.workflows.network",
        "pyrpmd.workflows.adaptive_umbrella", "pyrpmd.workflows.metadynamics",
        "pyrpmd.analysis.metadynamics",
    ]
    for module in modules:
        importlib.import_module(module)
    rng = np.random.default_rng(221)
    state = State(rng.normal(scale=0.1, size=(5, 1, 3)), rng.normal(scale=0.1, size=(5, 1, 3)))
    errors = {}
    for scheme in INTEGRATORS:
        states = []
        for native in (False, "required"):
            dynamics = Dynamics(HarmonicPotential(k=0.2), [1.7], 3.0, 5, 0.003,
                                integrator=scheme, native=native)
            trial = state.copy()
            for _ in range(10):
                dynamics.step(trial)
            states.append(trial)
        difference = max(float(np.max(np.abs(states[0].q - states[1].q))),
                         float(np.max(np.abs(states[0].p - states[1].p))))
        if difference > 2e-11:
            raise AssertionError(f"Installed native/Python {scheme} parity failed: {difference}")
        errors[scheme] = difference

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def command(label, arguments):
        completed = subprocess.run(arguments, cwd=output, env=environment,
                                   text=True, capture_output=True, timeout=300)
        log = output / f"{label}.log"
        log.write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{label} failed ({completed.returncode}):\n" +
                               "\n".join((completed.stdout + completed.stderr).splitlines()[-35:]))
        return completed.stdout

    executable = Path(sys.executable).parent / ("pyrpmd.exe" if os.name == "nt" else "pyrpmd")
    console_version = command("console-version", [str(executable), "--version"]).strip()
    info = json.loads(command("info", [sys.executable, "-m", "pyrpmd", "info"]))
    example_paths = {
        "network": "examples/network/two_entrances.py",
        "adaptive": "examples/fes/adaptive_harmonic.inp",
        "meta_standard": "examples/metadynamics/standard.inp",
        "meta_well_tempered": "examples/metadynamics/well_tempered_2d.inp",
    }
    examples = {}
    for name, relative in example_paths.items():
        example = source / relative
        if not example.is_file():
            raise AssertionError(f"Source distribution is missing {relative}")
        command(f"{name}-validate", [sys.executable, "-m", "pyrpmd", "validate", str(example)])
        directory = output / name
        command(f"{name}-run", [sys.executable, "-m", "pyrpmd", "run", str(example), "--output", str(directory)])
        summary = json.loads((directory / "results" / "summary.json").read_text())
        command(f"{name}-resume", [sys.executable, "-m", "pyrpmd", "resume", str(directory)])
        resumed = json.loads((directory / "results" / "summary.json").read_text())
        if summary != resumed:
            raise AssertionError(f"{name} changed its completed result on resume")
        if name == "network":
            if summary["completed_paths"] != 4 or summary["total_rate_cm3_s"] is not None:
                raise AssertionError("Network example did not complete all separate paths")
            paths = [path for row in summary["matrix"].values() for cell in row.values()
                     for path in cell["transition_states"].values()]
            if any(path["result"].get("child_trajectories") != 4 for path in paths):
                raise AssertionError("Network example did not propagate its actual paired children")
        examples[name] = {"task": summary["task"], "status": summary.get("status"),
                          "converged": summary.get("converged", False), "resume_identical": True}
    # Exercise public dRPMD entry points with real P>1 parent sampling and
    # collision propagation. Low-level parity alone cannot catch a workflow
    # that accidentally rejects (or ignores) a newly selected integrator.
    from pyrpmd.constants import KB_AU

    drpmd_workflows = {}
    for scheme in ("normal-mode-bab", "velocity-verlet"):
        directory = output / f"drpmd-{scheme}"
        job = {
            "task": "drpmd", "temperature": 1 / KB_AU, "beads": 3,
            "seed": 11, "native": "required",
            "system": {
                "symbols": ["H", "H"], "masses": [1.0, 1.0],
                "geometry": [[0, 0, -2], [0, 0, 2]], "fragments": [[0], [1]],
            },
            "potential": {"provider": "free"},
            "dynamics": {"timestep": 0.05, "integrator": scheme,
                         "loop_backend": "auto", "block_steps": 3},
            "sampling": {
                "equilibration_steps": 4, "production_steps": 4,
                "interval_steps": 2, "andersen_interval": 2, "parents": 1,
            },
            "collision": {"bmax": 0.5, "trajectories": 2, "max_steps": 200},
            "outcomes": {"bond_cutoffs": {"H-H": 0.3}, "persistence_steps": 2},
            "output": {"checkpoint_steps": 10},
        }
        pyrpmd.run(job, output=directory)
        summary = json.loads((directory / "results" / "summary.json").read_text())
        pyrpmd.resume_run(directory, execution={"loop_backend": "auto", "block_steps": 7})
        resumed = json.loads((directory / "results" / "summary.json").read_text())
        if summary != resumed:
            raise AssertionError(f"dRPMD {scheme} changed its completed result on resume")
        resolved = json.loads((directory / "input" / "resolved.json").read_text())
        if resolved["dynamics"]["integrator"] != scheme or resolved["sampling"]["integrator"] != scheme + "-andersen":
            raise AssertionError(f"dRPMD {scheme} did not retain its selected sampling/production method")
        records = summary["records"]
        if len(records) != 2 or summary["parent_thermal_steps"] != {"0": 8}:
            raise AssertionError(f"dRPMD {scheme} did not execute the actual parent/child workflow")
        if any(r["status"] == "failed" or
               r["propagation"]["native_steps"] + r["propagation"]["python_steps"] <= 0
               for r in records):
            raise AssertionError(f"dRPMD {scheme} did not propagate its actual children")
        drpmd_workflows[scheme] = {
            "beads": 3, "trajectories": len(records),
            "parent_thermal_steps": summary["parent_thermal_steps"],
            "child_native_steps": [r["propagation"]["native_steps"] for r in records],
            "child_python_steps": [r["propagation"]["python_steps"] for r in records],
            # FreePotential supplies Python forces. Auto uses the Python loop
            # with required native substeps, not an invented fused PES binding.
            "driver": "auto", "native_substeps": "required",
            "resume_identical": True,
        }
    report = {
        "version": version, "python": sys.version, "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "h5py")},
        "prefix": sys.prefix, "package": str(installed), "native_module": _native.__file__,
        "native_capabilities": native_capabilities(), "console_version": console_version,
        "cli_info": info, "imported_modules": modules, "integrator_max_abs_errors": errors,
        "examples": examples, "drpmd_workflows": drpmd_workflows,
        "scope": "Installed-distribution and finite-sampling workflow validation, not physical convergence or universal wheel compatibility.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"version": version, "package": str(installed), "examples": examples,
                      "drpmd_workflows": drpmd_workflows}, indent=2))


if __name__ == "__main__":
    main()
