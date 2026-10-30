"""Reproducible checkout comparison for the October 2026 efficiency changes.

Each checkout/repetition uses a fresh interpreter, identical single-CPU and
BLAS settings, the same real BKMP2 binary, and new workflow output directories.
Only the operation is timed; imports, configuration, verification, and evidence
serialization are excluded. --smoke performs reduced functional checks without
collecting timings. See the adjacent README for scope and interpretation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


CASES = ("bkmp2_drpmd", "bkmp2_fes", "pimd_pile_l", "com_gradient_n3",
         "com_gradient_n1000", "nrpmd_mint_s16", "fes_bootstrap", "normal_mode_n32_p256")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
BASELINE_COMMIT = "6fde29d2039c01921f00393bf17830b939af131b"
TOLERANCES = {
    "bkmp2_drpmd": {"rtol": 1e-10, "atol": 2e-9},
    "bkmp2_fes": {"rtol": 2e-11, "atol": 2e-10},
    "pimd_pile_l": {"rtol": 2e-11, "atol": 2e-10},
    "com_gradient_n3": {"rtol": 2e-14, "atol": 2e-14},
    "com_gradient_n1000": {"rtol": 2e-14, "atol": 2e-14},
    "nrpmd_mint_s16": {"rtol": 2e-11, "atol": 2e-11},
    "fes_bootstrap": {"rtol": 2e-11, "atol": 2e-12},
    "normal_mode_n32_p256": {"rtol": 2e-11, "atol": 2e-11},
}
# The large-ring checks use the same q/p bounds and candidate energy bound as
# test_native_sampling.test_large_normal_mode_free_map_matches_fft_preserves_energy_and_reverses
# (40 free steps there, 32 here). The pinned legacy dense transform accumulates
# 3.74e-12 relative energy drift in this 32-step problem; its separate bound is
# explicit and does not relax the candidate's independent conservation check.
NORMAL_MODE_ENERGY_LIMITS = {"baseline": 5e-12, "candidate": 2e-12}
ARRAY_TOLERANCES = {
    "normal_mode_n32_p256": {
        "final_p": {"rtol": 3e-11, "atol": 2e-9},
        "reference_p": {"rtol": 3e-11, "atol": 2e-9},
        "free_ring_energy": {"rtol": 5e-12, "atol": 0.},
    },
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _source_hashes(root):
    paths = [root / "pyproject.toml", root / "setup.py"]
    paths += list((root / "src/pyrpmd").rglob("*.py"))
    paths += [path for path in (root / "native").rglob("*")
              if path.suffix in (".h", ".hpp", ".cpp", ".c")]
    paths += [path for path in (root / "benchmarks/h3").iterdir()
              if path.suffix in (".py", ".inp", ".json", ".f90")]
    return {str(path.relative_to(root)): _sha256(path) for path in sorted(paths)}


def _manifest(root, library):
    import h5py
    import numpy as np
    import scipy
    import threadpoolctl
    import pyrpmd
    from pyrpmd import _native

    for module in (pyrpmd, _native):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise AssertionError(f"Imported {module.__name__} outside selected checkout: {module.__file__}")
    provenance = json.loads(library.with_suffix(".build.json").read_text())
    library_hash = _sha256(library)
    if provenance["library_sha256"] != library_hash:
        raise AssertionError("BKMP2 binary does not match its build provenance")
    pools = threadpoolctl.threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools if pool["user_api"] == "blas"):
        raise AssertionError("A loaded BLAS pool is not restricted to one thread")
    return {
        "checkout": str(root), "commit": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--short"),
        "harness_sha256": _sha256(__file__),
        "sources_sha256": _source_hashes(root),
        "python_package": str(Path(pyrpmd.__file__).resolve()),
        "native_extension": str(Path(_native.__file__).resolve()),
        "native_extension_sha256": _sha256(_native.__file__),
        "native_openmp": bool(_native.has_openmp()),
        "native_pile_l_entrypoint": hasattr(_native, "sample_free"),
        "versions": {"python": sys.version, "pyrpmd": pyrpmd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "h5py": h5py.__version__, "threadpoolctl": threadpoolctl.__version__},
        "executable": sys.executable, "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "threadpools": pools,
        "pes_library": str(library), "pes_library_sha256": library_hash,
        "pes_source_sha256": provenance["source"]["sha256"],
        "pes_build_provenance_sha256": _sha256(library.with_suffix(".build.json")),
        "pes_compiler": provenance["compiler"], "pes_flags": provenance["flags"],
    }


def _identity(manifest):
    return {key: manifest[key] for key in (
        "commit", "sources_sha256", "native_extension_sha256", "pes_library_sha256",
        "pes_build_provenance_sha256", "harness_sha256", "versions", "affinity", "thread_environment")}


def _check_imports(root):
    for name, module in list(sys.modules.items()):
        if name == "pyrpmd" or name.startswith(("pyrpmd.", "benchmarks.h3.", "_pyrpmd_user_")):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise AssertionError(f"Mixed-checkout import: {name} from {filename}")


def _digest(arrays, signature):
    import numpy as np

    digest = hashlib.sha256(json.dumps(_plain(signature), sort_keys=True, allow_nan=False).encode())
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        if array.dtype.hasobject:
            raise AssertionError(f"Object array is not admissible evidence: {key}")
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fes_evidence(data):
    import numpy as np

    arrays = {f"analysis/{key}": value for key, value in data.items()
              if isinstance(value, np.ndarray)}
    arrays.update({f"analysis/edges_{i}": edge for i, edge in enumerate(data["edges"])})
    arrays["analysis/weight_ess"] = np.asarray(data["weight_ess"])
    arrays["analysis/bootstrap_solver_residuals"] = np.asarray(
        [record["solver_residual"] for record in data["bootstrap_diagnostics"]])
    signature = {key: data[key] for key in (
        "connected", "bootstrap_replicates", "bootstrap_requested", "bootstrap_attempted",
        "bootstrap_reference_bin", "bootstrap_block_plan", "measure", "energy_unit")}
    signature["bootstrap_decisions"] = [{key: record[key] for key in ("draw", "valid", "reason")}
                                         for record in data["bootstrap_diagnostics"]]
    return signature, arrays


def _drpmd_case(directory, library, smoke):
    import numpy as np
    from benchmarks.h3.drpmd_optimization import _configuration, _workflow_arrays, _workflow_signature
    from pyrpmd.workflows.drpmd import run_drpmd

    args = SimpleNamespace(library=library, trajectories=2 if smoke else 8,
                           max_time_fs=2. if smoke else 200., checkpoint_steps=500,
                           trajectory_stride=10)
    config, system = _configuration(args, 1)
    settings = {"potential": "real BKMP2", "beads": 16, "trajectories": args.trajectories,
                "parents": 2, "max_time_fs": args.max_time_fs, "equilibration_fs": 5.,
                "parent_interval_fs": 2., "check_interval": 1, "trajectory_stride": 10,
                "checkpoint_steps": 500, "loop_backend": "native", "block_steps": 256}

    def collect(result):
        if len(result["records"]) != args.trajectories or any(
                record["status"] == "failed" for record in result["records"]):
            raise AssertionError("dRPMD did not finish the requested scientific work")
        arrays = _workflow_arrays(directory, args.trajectories)
        signature = _workflow_signature(result)
        signature["parents"] = []
        for path in sorted((directory / "checkpoints").glob("parent-*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(saved["metadata"].item())
                signature["parents"].append({key: metadata[key] for key in
                                             ("parent_id", "step", "time", "rng", "burn_done", "next_index")})
                for key in ("q", "p"):
                    arrays[f"parent_{metadata['parent_id']}/{key}"] = saved[key]
        steps = sum(record["steps"] for record in result["records"])
        native_steps = sum(record["propagation"]["native_steps"] for record in result["records"])
        if native_steps != steps:
            raise AssertionError("The dRPMD workload did not use the requested native loop")
        work = {"trajectories": args.trajectories, "accepted_child_steps": steps,
                "parent_thermal_steps": result["parent_thermal_steps"],
                "saved_trajectory_frames": sum(len(arrays[f"trajectory_{i}_steps"])
                                                for i in range(args.trajectories))}
        return signature, arrays, work, {"native_child_steps": native_steps}

    return lambda: run_drpmd(config, system, directory), collect, settings, "complete production workflow including initialization, sampling, I/O and analysis"


def _fes_case(directory, library, smoke):
    import numpy as np
    from benchmarks.h3.rpmd import configuration
    from pyrpmd.workflows.fes import run_fes

    config, system = configuration("fes2d", directory, workers=1, library=library)
    neq, nprod = (4, 20) if smoke else (40, 1000)
    config["dynamics"].update(loop_backend="native", block_steps=256)
    config["sampling"].update(equilibration_steps=neq, production_steps=nprod,
                              replicas=1, sample_interval=5, andersen_interval=25)
    config["output"]["checkpoint_steps"] = 500
    config["analysis"].update(bootstrap=0, min_count=1, bins=[8, 8])
    config["umbrella"]["grid"] = [{"start": 1.65, "stop": 1.85, "count": 2}] * 2
    settings = {"potential": "real BKMP2", "beads": config["beads"], "windows": 4,
                "equilibration_steps": neq, "production_steps": nprod, "sample_interval": 5,
                "andersen_interval": 25, "checkpoint_steps": 500, "bootstrap": 0,
                "loop_backend": "native", "block_steps": 256, "bins": [8, 8]}

    def collect(result):
        signature, arrays = _fes_evidence(result["data"])
        signature["streams"] = []
        if len(result["streams"]) != 4:
            raise AssertionError("FES stream count changed")
        native_steps = 0
        for index, stream in enumerate(result["streams"]):
            if len(stream["samples"]) != nprod // 5:
                raise AssertionError("FES recorded fewer samples than requested")
            for key in ("samples", "energies", "last_q", "last_p", "center", "stiffness"):
                arrays[f"stream_{index}/{key}"] = stream[key]
            with np.load(stream["checkpoint"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["values"], stream["samples"])
                np.testing.assert_array_equal(saved["energies"], stream["energies"])
                if int(saved["elapsed"]) != neq + nprod:
                    raise AssertionError("FES did not reach the requested phase clock")
                signature["streams"].append({"window": stream["window"], "replica": stream["replica"],
                    "step": int(saved["step"]), "time": float(saved["time"]),
                    "elapsed": int(saved["elapsed"]), "rng": json.loads(str(saved["rng"]))})
            native_steps += stream["propagation"]["native_steps"]
        if native_steps != 4 * (neq + nprod):
            raise AssertionError("FES did not use the requested native loop")
        work = {"streams": 4, "accepted_steps": 4 * (neq + nprod), "samples": 4 * (nprod // 5)}
        if int(result["summary"]["samples"]) != work["samples"]:
            raise AssertionError("FES summary and saved sample count disagree")
        return signature, arrays, work, {"native_steps": native_steps}

    return lambda: run_fes(config, system, directory, return_data=True), collect, settings, "complete production workflow including initialization, sampling, I/O and analysis"


def _pimd_case(directory, library, smoke):
    import numpy as np
    from pyrpmd.system import System
    from pyrpmd.workflows.materials import run_materials

    atoms, beads = 32, 32
    neq, nprod, interval = (4, 20, 2) if smoke else (200, 800, 10)
    system = System(tuple(["H"] * atoms), np.ones(atoms), np.zeros((atoms, 3)))
    config = {"task": "pimd", "beta": 4., "beads": beads, "dt_au": .02,
              "seed": 6703, "native": True,
              "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 1.}},
              "sampling": {"equilibration_steps": neq, "production_steps": nprod,
                           "sample_interval": interval, "replicas": 1,
                           "thermostat": "pile-l", "gamma_centroid": 1.},
              "dynamics": {"loop_backend": "auto", "integrator": "cayley-bcb"},
              "output": {"checkpoint_steps": 500}}
    settings = {"potential": "synthetic harmonic", "atoms": atoms, "beads": beads,
                "equilibration_steps": neq, "production_steps": nprod, "sample_interval": interval,
                "thermostat": "pile-l", "gamma_centroid": 1., "beta": 4., "dt_au": .02,
                "seed": 6703, "loop_backend": "auto", "integrator": "cayley-bcb", "checkpoint_steps": 500}

    def collect(result):
        if len(result["streams"]) != 1 or result["summary"]["status"] != "complete":
            raise AssertionError("PIMD did not complete its one requested replica")
        stream = result["streams"][0]
        if stream["elapsed"] != neq + nprod or len(stream["sample_steps"]) != nprod // interval + 1:
            raise AssertionError("PIMD phase clock or scientific sample count changed")
        arrays = {key: stream[key] for key in ("positions", "velocities", "observables",
                  "radii_gyration_squared", "sample_steps", "times")}
        arrays.update(final_q=stream["state"].q, final_p=stream["state"].p)
        signature = {"step": stream["state"].step, "time": stream["state"].time,
                     "elapsed": stream["elapsed"], "rng": stream["rng"], "complete": stream["complete"]}
        return signature, arrays, {"accepted_steps": neq + nprod, "samples": len(stream["sample_steps"]),
                                  "replicas": 1}, {"propagation": stream["propagation"]}

    return lambda: run_materials(config, system, directory, return_data=True), collect, settings, "complete production workflow including initialization, PILE-L sampling, checkpoint/trajectory I/O and analysis"


def _com_case(name, smoke):
    import numpy as np
    from pyrpmd.cv import COMDistance

    atoms = int(name.rsplit("n", 1)[1])
    calls = 1 if smoke else (4000 if atoms == 3 else 16)
    rng = np.random.default_rng(716 + atoms)
    masses = rng.uniform(1., 3., atoms)
    q = rng.normal(size=(atoms, 3))
    q[:atoms // 2, 0] += 3.
    cv = COMDistance([list(range(atoms // 2)), list(range(atoms // 2, atoms))], masses)

    def run():
        for _ in range(calls):
            value, gradient = cv.centroid_value_gradient(q)
        return value, gradient

    def collect(result):
        return {}, {"input/q": q, "input/masses": masses,
                    "value": np.asarray(result[0]), "gradient": result[1]}, {"calls": calls, "atoms": atoms}, {}

    return run, collect, {"atoms": atoms, "calls": calls, "operation": "COMDistance.centroid_value_gradient"}, "actual checkout COM gradient API; fixed repeated coordinates, no construction or evidence serialization"


def _mint_case(smoke):
    import numpy as np
    from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState, _mint

    beads, states, atoms, calls = 32, 16, 2, 2 if smoke else 32
    rng = np.random.default_rng(7316)
    matrix = rng.normal(scale=.2, size=(beads, states, states))
    matrix = .5 * (matrix + matrix.swapaxes(1, 2))
    gradient = rng.normal(scale=.1, size=(beads, states, states, atoms, 3))
    gradient = .5 * (gradient + gradient.swapaxes(1, 2))
    state = NRPMDState(rng.normal(size=(beads, atoms, 3)), rng.normal(size=(beads, atoms, 3)),
                      step=7, time=.21, mapping_q=rng.normal(size=(beads, states)),
                      mapping_p=rng.normal(size=(beads, states)))
    inputs = {f"input/{key}": getattr(state, key).copy() for key in ("q", "p", "mapping_q", "mapping_p")}
    inputs.update({"input/matrix": matrix, "input/gradient": gradient})
    result = DiabaticResult(matrix, gradient)
    history = {key: np.empty((calls, *getattr(state, key).shape)) for key in ("p", "mapping_q", "mapping_p")}

    def run():
        for index in range(calls):
            _mint(state, result, .017)
            for key, rows in history.items():
                rows[index] = getattr(state, key)

    def collect(unused):
        np.testing.assert_array_equal(state.q, inputs["input/q"])
        return {"step": state.step, "time": state.time}, {**inputs, **history}, {
            "calls": calls, "beads": beads, "states": states, "atoms": atoms}, {}

    settings = {"beads": beads, "states": states, "atoms": atoms, "calls": calls, "duration_au": .017,
                "operation": "actual _mint flow including the density basis contraction"}
    return run, collect, settings, "local MInt including eigensolve, contraction, nuclear impulse, mapping rotation and identical per-call state recording; no PES or trajectory workflow"


def _bootstrap_case(smoke):
    import numpy as np
    from pyrpmd.analysis.fes import reconstruct_fes

    rng = np.random.default_rng(1825)
    centers = np.array([[-.6, -.2], [.6, .3], [-.6, -.2]])
    stiffness = np.array([[[.6, .1], [.1, 1.2]], [[.4, -.08], [-.08, .9]], [[.6, .1], [.1, 1.2]]])
    counts = [100, 90, 80] if smoke else [1500, 1300, 1100]
    samples = [rng.normal(center, [1.4, .65], (count, 2)) for center, count in zip(centers, counts)]
    for index, sample in enumerate(samples):
        sample[:, 0] += (index - 1) * 4 * np.pi
    edges = [np.linspace(-np.pi, np.pi, 8), np.array([-1.5, -.3, .4, 1.5])]
    draws = 2 if smoke else 6

    def collect(result):
        signature, arrays = _fes_evidence(result)
        if not result["connected"] or result["bootstrap_attempted"] != draws or result["bootstrap_replicates"] < 2:
            raise AssertionError("Synthetic FES did not exercise a connected, valid bootstrap analysis")
        arrays.update({f"input/stream_{i}": sample for i, sample in enumerate(samples)})
        arrays.update({"input/centers": centers, "input/stiffness": stiffness})
        return signature, arrays, {"streams": 3, "samples": sum(counts), "samples_per_stream": counts,
                                  "bootstrap_requested": draws, "bootstrap_attempted": result["bootstrap_attempted"]}, {}

    settings = {"samples_per_stream": counts, "dimensions": 2, "periods": [2 * np.pi, None],
                "bootstrap": draws, "sample_seed": 1825, "bootstrap_seed": 915, "beta": 1.3,
                "min_count": 3, "duplicate_umbrella_states": True, "bins": [edge.tolist() for edge in edges]}
    return (lambda: reconstruct_fes(samples, centers, stiffness, 1.3, bins=edges,
                                    periods=(2 * np.pi, None), bootstrap=draws, seed=915, min_count=3)), collect, settings, "complete reconstruct_fes analysis including MBAR, statistical inefficiency, circular-block bootstrap, histogram and uncertainty; no sample generation or I/O"


def _normal_mode_case(smoke, variant):
    import numpy as np
    from pyrpmd.dynamics import CayleyIntegrator, State

    name = "normal_mode_n32_p256"
    energy_limit = NORMAL_MODE_ENERGY_LIMITS[variant]
    atoms, beads, calls, beta, dt = 32, 256, 2 if smoke else 32, 3.4, .003
    masses = np.linspace(1.1, 12., atoms)
    rng = np.random.default_rng(813256)
    original = State(rng.normal(0., .1, (beads, atoms, 3)),
                     rng.normal(size=(beads, atoms, 3)), step=7, time=.21)
    state = original.copy()
    integrator = CayleyIntegrator(masses, beta, beads, dt, threads=1,
                                 native=True, scheme="normal-mode-bab")
    if not integrator.native:
        raise AssertionError("The requested normal-mode free map is not native")

    def energy(value):
        difference = value.q - np.roll(value.q, 1, axis=0)
        return float(np.sum(value.p**2 / (2 * masses[None, :, None])) +
                     .5 * (beads / beta)**2 * np.sum(masses[None, :, None] * difference**2))

    initial_energy = energy(original)

    def run():
        for _ in range(calls):
            integrator.free_step(state.q, state.p)

    def collect(unused):
        # The existing Python/NumPy integrator is an untimed independent
        # execution path, not an alternative kernel substituted into timing.
        reference = original.copy()
        python = CayleyIntegrator(masses, beta, beads, dt, threads=1,
                                  native=False, scheme="normal-mode-bab")
        for _ in range(calls):
            python.free_step(reference.q, reference.p)
        np.testing.assert_allclose(state.q, reference.q, **TOLERANCES[name])
        np.testing.assert_allclose(state.p, reference.p, **ARRAY_TOLERANCES[name]["final_p"])
        expected_clock = (original.step, original.time)
        if (state.step, state.time) != expected_clock or (reference.step, reference.time) != expected_clock:
            raise AssertionError("free_step changed the state clock; only q/p should be mutated")
        final_energy, reference_energy = energy(state), energy(reference)
        relative_error = abs(final_energy / initial_energy - 1.)
        reference_relative_error = abs(reference_energy / initial_energy - 1.)
        if relative_error >= energy_limit or reference_relative_error >= 2e-12:
            raise AssertionError(f"Free-ring energy drift: native={relative_error:.6g} "
                                 f"(limit {energy_limit}), reference={reference_relative_error:.6g} (limit 2e-12)")
        signature = {"step": state.step, "time": state.time,
                     "clock_policy": "free_step mutates q/p only; the State clock is unchanged",
                     "integrated_duration_au": calls * dt}
        arrays = {"input/q": original.q, "input/p": original.p, "input/masses": masses,
                  "final_q": state.q, "final_p": state.p,
                  "reference_q": reference.q, "reference_p": reference.p,
                  "free_ring_energy": np.asarray([initial_energy, final_energy, reference_energy])}
        execution = {"native": True, "reference": "CayleyIntegrator(native=False, scheme='normal-mode-bab')",
                     "reference_is_outside_timing": True,
                     "reference_max_q_error": float(np.max(np.abs(state.q - reference.q))),
                     "reference_max_p_error": float(np.max(np.abs(state.p - reference.p))),
                     "energy_relative_error": relative_error,
                     "reference_energy_relative_error": reference_relative_error,
                     "energy_order": ["initial", "native_final", "python_reference_final"],
                     "energy_relative_error_limit": energy_limit,
                     "reference_energy_relative_error_limit": 2e-12}
        return signature, arrays, {"free_steps": calls, "atoms": atoms, "beads": beads}, execution

    settings = {"atoms": atoms, "beads": beads, "calls": calls, "beta": beta, "dt_au": dt,
                "seed": 813256, "mass_min": float(masses.min()), "mass_max": float(masses.max()),
                "scheme": "normal-mode-bab", "native": True, "threads": 1}
    return run, collect, settings, "actual CayleyIntegrator.free_step calls only; local free-ring kernel, no PES, thermostat, workflow I/O, reference propagation or energy diagnostics in timing"


def _case(name, directory, library, smoke, variant):
    if name == "bkmp2_drpmd":
        return _drpmd_case(directory, library, smoke)
    if name == "bkmp2_fes":
        return _fes_case(directory, library, smoke)
    if name == "pimd_pile_l":
        return _pimd_case(directory, library, smoke)
    if name.startswith("com_gradient_"):
        return _com_case(name, smoke)
    if name == "nrpmd_mint_s16":
        return _mint_case(smoke)
    if name == "fes_bootstrap":
        return _bootstrap_case(smoke)
    if name == "normal_mode_n32_p256":
        return _normal_mode_case(smoke, variant)
    raise ValueError(name)


def _worker(args):
    root = args.checkout.resolve()
    # Do not let the harness's own directory select a different checkout.
    sys.path = [str(root / "src"), str(root)] + [path for path in sys.path
                  if path and Path(path).resolve() != Path(__file__).resolve().parent]
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(root, args.library)
    report = {"variant": args.variant, "smoke": args.smoke, "manifest": manifest, "cases": {}}
    for name in args.cases:
        case_directory = args.output / name
        case_directory.mkdir()
        warm_digest = None
        if not args.smoke:
            warm, collect, settings, scope = _case(name, case_directory / "warmup", args.library, False, args.variant)
            signature, arrays, work, execution = collect(warm())
            warm_digest = _digest(arrays, signature)
            del arrays, warm, collect
        run, collect, settings, scope = _case(name, case_directory / "production", args.library, args.smoke, args.variant)
        if args.smoke:
            result, seconds = run(), None
        else:
            started = time.perf_counter()
            result = run()
            seconds = time.perf_counter() - started
        signature, arrays, work, execution = collect(result)
        digest = _digest(arrays, signature)
        if warm_digest is not None and warm_digest != digest:
            raise AssertionError(f"Warmup and fresh production outputs differ: {name}")
        np.savez(case_directory / "arrays.npz", **arrays)
        report["cases"][name] = {"seconds": seconds, "signature": _plain(signature),
                                "settings": _plain(settings), "work": _plain(work),
                                "execution": _plain(execution), "data_digest": digest,
                                "timed_scope": scope, "warmup_matched": warm_digest is not None}
        _write_json(args.output / "result.json", report)
        del arrays, result, run, collect
    _check_imports(root)
    after = _manifest(root, args.library)
    if _identity(manifest) != _identity(after):
        raise AssertionError("Sources, extension, PES or execution environment changed during the worker")
    report["manifest"]["threadpools_after"] = after["threadpools"]
    _write_json(args.output / "result.json", report)


def _compare(name, left_directory, right_directory):
    import numpy as np

    left = json.loads((left_directory / "result.json").read_text())["cases"][name]
    right = json.loads((right_directory / "result.json").read_text())["cases"][name]
    for field in ("settings", "work", "signature"):
        if left[field] != right[field]:
            raise AssertionError(f"{name}: {field} differ; matching result counts alone are insufficient")
    errors, exact = {}, True
    with (np.load(left_directory / name / "arrays.npz", allow_pickle=False) as a,
          np.load(right_directory / name / "arrays.npz", allow_pickle=False) as b):
        if set(a.files) != set(b.files):
            raise AssertionError(f"{name}: different saved numerical fields")
        for key in a.files:
            aa, bb = a[key], b[key]
            if aa.shape != bb.shape or aa.dtype != bb.dtype:
                raise AssertionError(f"{name}/{key}: array shape or dtype differs")
            same = np.array_equal(aa, bb, equal_nan=True)
            exact &= same
            leaf = key.rsplit("/", 1)[-1]
            require_exact = (aa.dtype.kind in "biu" or key.startswith("input/") or
                             leaf in ("times", "sample_steps") or leaf.endswith(("_steps", "_time_au")))
            if require_exact:
                np.testing.assert_array_equal(aa, bb, err_msg=f"{name}/{key}")
            else:
                tolerance = ARRAY_TOLERANCES.get(name, {}).get(key, TOLERANCES[name])
                np.testing.assert_allclose(aa, bb, equal_nan=True, err_msg=f"{name}/{key}", **tolerance)
            finite = np.isfinite(aa) & np.isfinite(bb)
            error = float(np.max(np.abs(aa[finite] - bb[finite]))) if finite.any() and aa.dtype.kind != "b" else 0.
            errors[key] = {"max_absolute_error": error, "bitwise_equal": bool(same),
                           "exact_required": require_exact, "shape": list(aa.shape),
                           "nonfinite_count": int((~np.isfinite(aa)).sum())}
    return {"settings_work_rng_clocks_and_discrete_decisions_equal": True,
            "recorded_numerical_fields_match": True, "all_arrays_bitwise_equal": bool(exact),
            "tolerances_for_nonexact_fields": TOLERANCES[name],
            "tolerances_for_named_fields": ARRAY_TOLERANCES.get(name, {}), "arrays": errors}


def _controller(args):
    if not args.smoke and args.repeats < 3:
        raise ValueError("Use at least three repetitions; the default is five")
    if not hasattr(os, "sched_getaffinity"):
        if not args.smoke:
            raise RuntimeError("Timing requires OS support for explicit single-CPU affinity")
    elif args.cpu is None:
        args.cpu = min(os.sched_getaffinity(0))
    elif args.cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside the allowed affinity set")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    if roots["baseline"] == roots["candidate"]:
        raise ValueError("Baseline and candidate must be different checkouts")
    if _git(roots["baseline"], "rev-parse", "HEAD") != args.baseline_commit:
        raise ValueError("Baseline commit differs from --baseline-commit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repeats = 1 if args.smoke else args.repeats
    report = {"schema": "pyrpmd.audit-optimization.v1", "passed": False,
              "mode": "smoke_functional_only" if args.smoke else "timed_comparison",
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "harness_sha256": _sha256(__file__), "repeats": repeats,
              "warmups_per_case_per_worker": 0 if args.smoke else 1,
              "physical_convergence_claim": False,
              "scope": "same-machine implementation comparison; numerical agreement of recorded fields does not establish physical convergence",
              "runs": [], "cases": {}, "manifests": {}}
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_ENV})
    environment.update(PYTHONNOUSERSITE="1", PYTHONHASHSEED="0")
    identities, digests = {}, {}
    directories = {name: [] for name in roots}
    seconds = {case: {name: [] for name in roots} for case in args.cases}
    verifications = {case: [] for case in args.cases}
    try:
        for repeat in range(repeats):
            order = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
            for variant in order:
                directory = output / f"repeat-{repeat:02d}" / variant
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--checkout", str(roots[variant]), "--variant", variant,
                           "--output", str(directory), "--library", str(args.library),
                           "--cases", *args.cases]
                if args.cpu is not None:
                    command += ["--cpu", str(args.cpu)]
                if args.smoke:
                    command += ["--smoke"]
                child_environment = environment.copy()
                child_environment["PYTHONPATH"] = os.pathsep.join([str(roots[variant] / "src"), str(roots[variant])])
                log = output / f"repeat-{repeat:02d}-{variant}.log"
                with log.open("w") as stream:
                    child = subprocess.run(command, cwd=roots[variant], env=child_environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
                if child.returncode:
                    raise RuntimeError(f"{variant} repeat {repeat} failed; see {log}")
                value = json.loads((directory / "result.json").read_text())
                identity = _identity(value["manifest"])
                if variant in identities and identities[variant] != identity:
                    raise AssertionError(f"{variant} changed sources/binaries/environment between repetitions")
                identities[variant] = identity
                report["manifests"][variant] = value["manifest"]
                directories[variant].append(directory)
                report["runs"].append({"repeat": repeat, "variant": variant, "command": command,
                                       "directory": str(directory), "log": str(log), "cases": value["cases"]})
                for case, row in value["cases"].items():
                    key = (variant, case)
                    if key in digests and digests[key] != row["data_digest"]:
                        raise AssertionError(f"{variant}/{case} produced different scientific arrays on repetition {repeat}")
                    digests[key] = row["data_digest"]
                    if not args.smoke:
                        seconds[case][variant].append(row["seconds"])
                _write_json(output / "report.json", report)
                print(json.dumps({"repeat": repeat, "variant": variant, "functional_only": args.smoke,
                                  "completed_cases": list(value["cases"])}), flush=True)
            baseline, candidate = directories["baseline"][-1], directories["candidate"][-1]
            for case in args.cases:
                verifications[case].append(_compare(case, baseline, candidate))
                row = {"validation_by_repeat": verifications[case], "seconds_raw": seconds[case]}
                if not args.smoke:
                    medians = {variant: statistics.median(times) for variant, times in seconds[case].items()}
                    row.update(median_seconds=medians, speedup=medians["baseline"] / medians["candidate"],
                               paired_speedups=[a / b for a, b in zip(seconds[case]["baseline"], seconds[case]["candidate"])],
                               candidate_batch_shorter_than_5ms=medians["candidate"] < .005)
                report["cases"][case] = row
            if report["manifests"]["baseline"]["pes_library_sha256"] != report["manifests"]["candidate"]["pes_library_sha256"]:
                raise AssertionError("The two checkouts used different PES binaries")
            _write_json(output / "report.json", report)
        report["passed"] = True
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "report.json", report)
    print(json.dumps({"passed": True, "mode": report["mode"], "report": str(output / "report.json")}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--smoke", action="store_true", help="reduced functional comparison; collect no timings")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.library = args.library.resolve()
    args.output = args.output.resolve()
    if len(args.cases) != len(set(args.cases)):
        parser.error("Each case may be selected only once")
    if args.worker:
        if args.checkout is None or args.variant is None:
            parser.error("Worker arguments are incomplete")
        _worker(args)
    else:
        if args.baseline is None:
            parser.error("--baseline is required")
        _controller(args)


if __name__ == "__main__":
    main()
