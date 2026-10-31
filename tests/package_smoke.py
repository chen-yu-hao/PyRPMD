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
from unittest.mock import patch

import numpy as np


def native_release_checks(source, output):
    """Exercise new native APIs in the installed wheel, including large rings."""
    from pyrpmd import _native
    from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State
    from pyrpmd.potentials import CABIPotential, HarmonicPotential

    for marker in ("DRPMD_OUTCOMES_API", "BLOCK_ENERGY_CHECK_API", "SAMPLING_FREE_API"):
        if getattr(_native, marker, 0) < 1:
            raise AssertionError(f"Installed native module is missing {marker}")
    rng = np.random.default_rng(981)
    errors = {}
    for beads in (5, 32, 256):
        initial = State(rng.normal(scale=0.1, size=(beads, 1, 3)),
                        rng.normal(scale=0.1, size=(beads, 1, 3)))
        for scheme in ("cayley-bcb", "normal-mode-bab"):
            states, generators = [], []
            with patch.object(_native, "sample_free", wraps=_native.sample_free) as sampler:
                for native in (False, "required"):
                    dynamics = Dynamics(HarmonicPotential(k=0.2), [1.7], 3., beads,
                                        0.003, native=native, integrator=scheme, seed=712)
                    trial = initial.copy()
                    for _ in range(4):
                        dynamics.sample_step(trial)
                    states.append(trial)
                    generators.append(dynamics.rng.bit_generator.state)
                assert sampler.call_count == 4, (beads, scheme, sampler.call_count)
            np.testing.assert_allclose(states[0].q, states[1].q, rtol=2e-11, atol=2e-11)
            np.testing.assert_allclose(states[0].p, states[1].p, rtol=3e-11, atol=2e-9)
            assert generators[0] == generators[1]
            assert states[0].step == states[1].step == 4
            assert states[0].time == states[1].time
            errors[f"pile-{scheme}-{beads}"] = float(np.max(np.abs(states[0].p - states[1].p)))
        if beads >= 32:
            states = []
            for native in (False, "required"):
                integrator = CayleyIntegrator([1.7], 3., beads, 0.003,
                                              scheme="normal-mode-bab", native=native)
                trial = initial.copy()
                for _ in range(20):
                    integrator.free_step(trial.q, trial.p)
                states.append(trial)
            np.testing.assert_allclose(states[0].q, states[1].q, rtol=2e-11, atol=2e-11)
            np.testing.assert_allclose(states[0].p, states[1].p, rtol=3e-11, atol=2e-9)
            errors[f"fft-{beads}"] = float(np.max(np.abs(states[0].p - states[1].p)))

    library = output / "harmonic.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-I", str(source / "native/include"),
                    str(source / "native/examples/harmonic_pes.c"), "-o", str(library)], check=True)
    potential = CABIPotential(library)
    try:
        dynamics = Dynamics(potential, [1.7], 3., 32, 0.003, native="required")
        states, results = [], []
        for backend in ("python", "native"):
            trial = State(np.full((32, 1, 3), 0.1), np.full((32, 1, 3), 0.2))
            result = dynamics.propagate_block(trial, 4, backend=backend,
                                             observe=[0, 1, 0, 1], check_energy=True)
            assert result.completed == 4 and result.backend == backend
            np.testing.assert_array_equal(result.steps, [2, 4])
            assert np.isfinite(result.energies).all()
            states.append(trial)
            results.append(result)
        np.testing.assert_allclose(states[0].q, states[1].q, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(states[0].p, states[1].p, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(results[0].energies, results[1].energies, rtol=2e-11)
    finally:
        potential.close()
    return errors


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
    from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State, INTEGRATORS, native_capabilities
    from pyrpmd.potentials import HarmonicPotential

    installed = Path(pyrpmd.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()) or installed.is_relative_to(source):
        raise RuntimeError(f"Imported source/editable tree instead of independent installation: {installed}")
    version = importlib.metadata.version("PyRPMD")
    if pyrpmd.__version__ != version:
        raise RuntimeError("Installed metadata and Python package versions disagree")
    native_release = native_release_checks(source, output)
    modules = [
        "pyrpmd.checkpoint", "pyrpmd.reactions", "pyrpmd.workflows.network",
        "pyrpmd.workflows.adaptive_umbrella", "pyrpmd.workflows.metadynamics",
        "pyrpmd.analysis.metadynamics",
        "pyrpmd.pcet", "pyrpmd.pcet_input", "pyrpmd.workflows.pcet",
        "pyrpmd.pesgen.jaxpip_data", "pyrpmd.pesgen.jaxpip_fit",
        "pyrpmd.pesgen.jaxpip_export",
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

    # A fresh distribution must contain the new native APIs. Source tests allow
    # an older installed extension to fall back; a release artifact must not
    # silently pass this check by exercising only that fallback.
    native_api = {name: getattr(_native, name, 0)
                  for name in ("SAMPLING_FREE_API", "BLOCK_ENERGY_CHECK_API")}
    if any(value < 1 for value in native_api.values()) or not callable(getattr(_native, "sample_free", None)):
        raise AssertionError(f"Installed extension is missing the optimized native APIs: {native_api}")
    sampling_errors = {}
    for scheme in ("cayley-bcb", "normal-mode-bab"):
        initial = State(rng.normal(scale=0.1, size=(32, 2, 3)),
                        rng.normal(scale=0.1, size=(32, 2, 3)))
        states, generators = [], []
        for native in (False, "required"):
            dynamics = Dynamics(HarmonicPotential(k=0.2), [1.7, 0.9], 3., 32, 0.003,
                                integrator=scheme, native=native, seed=381)
            trial = initial.copy()
            for _ in range(4):
                dynamics.sample_step(trial)
            states.append(trial)
            generators.append(dynamics.rng.bit_generator.state)
            if native == "required" and dynamics._cached_pile_coefficients is None:
                raise AssertionError("Installed PILE-L unexpectedly used the reference fallback")
        np.testing.assert_allclose(states[0].q, states[1].q, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(states[0].p, states[1].p, rtol=2e-11, atol=2e-10)
        if generators[0] != generators[1] or (states[0].step, states[0].time) != (states[1].step, states[1].time):
            raise AssertionError(f"Installed {scheme} sampling changed RNG or the state clock")
        sampling_errors[scheme] = {
            "q": float(np.max(np.abs(states[0].q - states[1].q))),
            "p": float(np.max(np.abs(states[0].p - states[1].p))),
            "rng_equal": True, "clock_equal": True,
        }
    initial = State(rng.normal(scale=0.1, size=(64, 2, 3)),
                    rng.normal(scale=0.1, size=(64, 2, 3)))
    free_states = []
    for native in (False, "required"):
        integrator = CayleyIntegrator([1.7, 0.9], 3., 64, 0.003,
                                     scheme="normal-mode-bab", native=native)
        trial = initial.copy()
        for _ in range(8):
            integrator.free_step(trial.q, trial.p)
        free_states.append(trial)
    np.testing.assert_allclose(free_states[0].q, free_states[1].q, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(free_states[0].p, free_states[1].p, rtol=3e-11, atol=2e-9)
    large_ring_errors = {name: float(np.max(np.abs(getattr(free_states[0], name) - getattr(free_states[1], name))))
                         for name in ("q", "p")}

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
    # A wheel user must be able to start without a source checkout or separately
    # downloaded example. Exercise the packaged template from an unrelated cwd.
    starter = output / "first-project"
    command("init", [sys.executable, "-m", "pyrpmd", "init", str(starter)])
    starter_input = starter / "quickstart.inp"
    command("init-validate", [sys.executable, "-m", "pyrpmd", "validate",
                              str(starter_input), "--check-pes"])
    command("init-run", [sys.executable, "-m", "pyrpmd", "run", str(starter_input)])
    starter_run = starter / "run"
    starter_summary = json.loads(command(
        "init-analyze", [sys.executable, "-m", "pyrpmd", "analyze", str(starter_run)]))
    for key, expected in (("task", "fes"), ("samples", 120), ("streams", 6), ("windows", 3)):
        if starter_summary.get(key) != expected:
            raise AssertionError(f"Installed starter has unexpected {key}: {starter_summary.get(key)!r}")
    command("init-resume", [sys.executable, "-m", "pyrpmd", "resume", str(starter_run)])
    starter_resumed = json.loads((starter_run / "results/summary.json").read_text())
    if starter_summary != starter_resumed:
        raise AssertionError("Installed starter changed its completed result on resume")
    starter_result = {"task": starter_summary["task"], "samples": starter_summary["samples"],
                      "streams": starter_summary["streams"], "windows": starter_summary["windows"],
                      "resume_identical": True}
    example_paths = {
        "network": "examples/network/two_entrances.py",
        "adaptive": "examples/fes/adaptive_harmonic.inp",
        "meta_standard": "examples/metadynamics/standard.inp",
        "meta_well_tempered": "examples/metadynamics/well_tempered_2d.inp",
        "pcet": "examples/pcet/harmonic_scan.inp",
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
        if name == "pcet":
            if summary["completed_scan_points"] != 18 or summary["normalization"] != "local_first_order":
                raise AssertionError("PCET example did not evaluate its full local-rate scan")
            if summary["rate_unit"] != "s^-1" or not summary["numerical_convergence"]:
                raise AssertionError("PCET example failed its rate-unit or numerical-convergence checks")
            if not summary["physical_applicability_unverified"]:
                raise AssertionError("PCET teaching model must retain its applicability limitation")
            if len(summary["kinetic_isotope_effects"]) != 9:
                raise AssertionError("PCET example did not compare both isotopes at every condition")
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
                         for name in ("numpy", "scipy", "h5py", "threadpoolctl")},
        "prefix": sys.prefix, "package": str(installed), "native_module": _native.__file__,
        "native_capabilities": native_capabilities(), "console_version": console_version,
        "cli_info": info, "imported_modules": modules, "integrator_max_abs_errors": errors,
        "native_api": native_api, "sampling_max_abs_errors": sampling_errors,
        "large_ring_max_abs_errors": large_ring_errors,
        "native_release_max_abs_errors": native_release,
        "starter_project": starter_result,
        "examples": examples, "drpmd_workflows": drpmd_workflows,
        "scope": "Installed-distribution and finite-sampling workflow validation, not physical convergence or universal wheel compatibility.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"version": version, "package": str(installed), "examples": examples,
                      "drpmd_workflows": drpmd_workflows}, indent=2))


if __name__ == "__main__":
    main()
