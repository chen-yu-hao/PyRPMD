#!/usr/bin/env python3
"""Measured mapping-flow and end-to-end NRPMD backend/batching comparisons.

Correctness is checked against NumPy scalar propagation of identical initial
states. Timing uses warmups and repeated medians; speedup is never a pass gate.
Missing backends are recorded as unavailable, never replaced by another one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd import __version__
from pyrpmd.batched_nonadiabatic import BatchedNRPMDState, BatchedNRPMDDynamics
from pyrpmd.constants import AMU_TO_AU
from pyrpmd.dynamics import State
from pyrpmd.electronic_backends import (
    electronic_backend_capabilities, electronic_mapping_flow,
    torch_electronic_mapping_flow, validate_electronic_backend,
)
from pyrpmd.io import write_json
from pyrpmd.nonadiabatic import NRPMDState, NRPMDDynamics
from pyrpmd.nonadiabatic_ensemble import density_matrix
from pyrpmd.potentials.multistate_external import NaH2LEPS5G
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling


ATOL, RTOL = 5e-10, 2e-11


def _hermitian(rng, shape, scale):
    a = rng.normal(size=shape)+1j*rng.normal(size=shape)
    return scale*.5*(a+a.swapaxes(-1, -2).conj())


def _case(name, count, beads, states, seed):
    rng = np.random.default_rng(seed)
    if name == "complex_lvc":
        potential = LinearVibronicCoupling(
            _hermitian(rng, (states, states), .02),
            _hermitian(rng, (3, states, states), .005), harmonic=[.1, .2, .15])
        masses = np.array([15.])
        q = rng.normal(scale=.03, size=(count, beads, 1, 3))
        p = rng.normal(scale=.05, size=q.shape)
        beta, dt = 20., .05
        description = "synthetic complex Hermitian linear-vibronic model; not an electronic-structure fit"
    else:
        potential = NaH2LEPS5G()
        states = 2
        masses = np.array([22.98976928, 1.00782503223, 1.00782503223])*AMU_TO_AU
        geometry = np.array([[3.2, .8, 0.], [-.85, 0., 0.], [.85, 0., 0.]])
        q = geometry[None, None]+rng.normal(scale=.002, size=(count, beads, 3, 3))
        p = rng.normal(scale=.05, size=q.shape)
        beta, dt = 1000., .1
        description = "published NaH2 5G molecular PES with actual Na/H masses"
    initial = tuple(NRPMDState.from_nuclear(State(x, y), states, 1, rng) for x, y in zip(q, p))
    hashes = {}
    for key in ("q", "p", "mapping_q", "mapping_p"):
        array = np.ascontiguousarray(np.stack([getattr(state, key) for state in initial]))
        hashes[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                       "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return {"name": name, "description": description, "potential": potential,
            "masses": masses, "initial": initial, "beta": beta, "dt": dt,
            "count": count, "beads": beads, "states": states, "atoms": len(masses),
            "model_identity": potential.to_dict(), "initial_arrays": hashes}


def _device(backend, torch_device):
    return torch_device if backend == "torch" else "cpu"


def _synchronize(backend, device):
    if backend == "torch" and device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def _timing(function, *, warmups, repeats, inner=1, synchronize=lambda: None):
    for _ in range(warmups):
        function()
        synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter_ns()
        for _ in range(inner):
            function()
        synchronize()
        samples.append((time.perf_counter_ns()-started)*1e-9/inner)
    return {"median_seconds": statistics.median(samples), "samples_seconds": samples,
            "min_seconds": min(samples), "max_seconds": max(samples),
            "warmups": warmups, "repeats": repeats, "inner_calls_per_sample": inner}


def _errors(actual, expected):
    errors = {key: float(np.max(np.abs(actual[key]-value))) for key, value in expected.items()}
    passed = all(np.allclose(actual[key], value, rtol=RTOL, atol=ATOL) for key, value in expected.items())
    if not passed:
        raise AssertionError(f"backend/batch numerical disagreement: {errors}")
    return {"passed": passed, "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
            "max_absolute_errors": errors}


def _kernel_inputs(case):
    state = BatchedNRPMDState.from_states(case["initial"])
    evaluated = case["potential"].evaluate(state.q.reshape(-1, case["atoms"], 3))
    leading = (case["count"], case["beads"], case["states"], case["states"])
    return (state.mapping_q+1j*state.mapping_p, evaluated.matrix.reshape(leading),
            evaluated.gradients.reshape(*leading, case["atoms"], 3))


def kernel_benchmark(case, available, args):
    z, matrix, gradients = _kernel_inputs(case)
    duration = .5*case["dt"]
    reference = electronic_mapping_flow(z, matrix, gradients, duration, backend="numpy", threads=args.threads)
    expected = dict(z=reference[0], delta_p=reference[1])
    rows = []
    for backend in available:
        device = _device(backend, args.torch_device)
        function = lambda: electronic_mapping_flow(z, matrix, gradients, duration, backend=backend,
                                                   device=device, threads=args.threads)
        output = function()
        check = _errors(dict(z=output[0], delta_p=output[1]), expected)
        row = {"backend": backend, "device": device, "api": "host_array",
               "includes_pes": False, "input_preparation_in_timing": False,
               "host_device_transfers_in_timing": backend == "torch" and device.startswith("cuda"),
               "checks": check, **_timing(function, warmups=args.warmups, repeats=args.repeats,
                                          inner=args.kernel_inner,
                                          synchronize=lambda: _synchronize(backend, device))}
        rows.append(row)
        if backend == "torch":
            import torch
            tensors = tuple(torch.as_tensor(a, dtype=torch.complex128, device=device) for a in (z, matrix, gradients))
            resident = lambda: torch_electronic_mapping_flow(*tensors, duration)
            result = resident()
            check = _errors(dict(z=result[0].cpu().numpy(), delta_p=result[1].cpu().numpy()), expected)
            rows.append({"backend": "torch", "device": device, "api": "tensor_resident",
                         "includes_pes": False, "input_preparation_in_timing": False,
                         "host_device_transfers_in_timing": False,
                         "checks": check, **_timing(resident, warmups=args.warmups, repeats=args.repeats,
                                                    inner=args.kernel_inner,
                                                    synchronize=lambda: _synchronize("torch", device))})
    baseline = next(r["median_seconds"] for r in rows if r["backend"] == "numpy")
    for row in rows:
        row["speedup_vs_numpy_host_array"] = baseline/row["median_seconds"]
        row["frames_per_second"] = case["count"]*case["beads"]/row["median_seconds"]
    return {"case": case["name"], "trajectory_batch": case["count"], "beads": case["beads"],
            "electronic_states": case["states"], "atoms": case["atoms"],
            "matrix_input_dtype": str(matrix.dtype), "duration_atomic": duration,
            "rows": rows}


def _driver(case, backend, args, batched):
    cls = BatchedNRPMDDynamics if batched else NRPMDDynamics
    return cls(case["potential"], case["masses"], case["beta"], case["beads"], case["dt"], case["states"],
               native="required" if args.ring_native else False, threads=args.threads,
               electronic_backend=backend, electronic_device=_device(backend, args.torch_device))


def _observations(states, drivers, *, batched):
    if batched:
        return {"q": states.q, "p": states.p, "mapping_q": states.mapping_q, "mapping_p": states.mapping_p,
                "populations": drivers.populations(states), "density": drivers.density_matrix(states),
                "energy": drivers.energy(states), "step": states.step, "time": states.time}
    return {"q": np.stack([s.q for s in states]), "p": np.stack([s.p for s in states]),
            "mapping_q": np.stack([s.mapping_q for s in states]), "mapping_p": np.stack([s.mapping_p for s in states]),
            "populations": np.stack([d.populations(s) for d, s in zip(drivers, states)]),
            "density": np.stack([density_matrix(s) for s in states]),
            "energy": np.array([d.energy(s) for d, s in zip(drivers, states)]),
            "step": np.array([s.step for s in states]), "time": np.array([s.time for s in states])}


def _step_runner(case, backend, args, batched):
    drivers = (_driver(case, backend, args, True) if batched else
               tuple(_driver(case, backend, args, False) for _ in case["initial"]))

    def prepare():
        return (BatchedNRPMDState.from_states(case["initial"]) if batched else
                tuple(s.copy() for s in case["initial"]))

    def propagate(states):
        for _ in range(args.steps):
            if batched:
                drivers.step(states)
            else:
                for driver, state in zip(drivers, states):
                    driver.step(state)
        return states

    return drivers, prepare, propagate


def end_to_end_benchmark(case, available, args):
    reference_drivers, prepare, propagate = _step_runner(case, "numpy", args, False)
    reference = _observations(propagate(prepare()), reference_drivers, batched=False)
    rows = []
    for backend in available:
        device = _device(backend, args.torch_device)
        for batched in (False, True):
            drivers, prepare, propagate = _step_runner(case, backend, args, batched)
            trial = propagate(prepare())
            check = _errors(_observations(trial, drivers, batched=batched), reference)
            for _ in range(args.warmups):
                propagate(prepare())
                _synchronize(backend, device)
            samples = []
            for _ in range(args.repeats):
                state = prepare()  # identical state, copied outside the timed interval
                _synchronize(backend, device)
                started = time.perf_counter_ns()
                propagate(state)
                _synchronize(backend, device)
                samples.append((time.perf_counter_ns()-started)*1e-9)
            final_check = _errors(_observations(state, drivers, batched=batched), reference)
            rows.append({"backend": backend, "device": device,
                         "driver": "BatchedNRPMDDynamics" if batched else "NRPMDDynamics_scalar_loop",
                         "includes_pes": True, "includes_nuclear_free_ring": True,
                         "includes_observation_or_io": False,
                         "state_reset_and_driver_construction_in_timing": False,
                         "electronic_host_array_api": True,
                         "ring_native": args.ring_native,
                         "steps": args.steps, "checks": final_check,
                         "initial_comparison_passed": check["passed"],
                         "median_seconds": statistics.median(samples), "samples_seconds": samples,
                         "min_seconds": min(samples), "max_seconds": max(samples),
                         "warmups": args.warmups, "repeats": args.repeats,
                         "pes_calls_per_step": 2 if batched else 2*case["count"],
                         "pes_frames_per_step": 2*case["count"]*case["beads"]})
    baseline = next(row["median_seconds"] for row in rows if row["backend"] == "numpy"
                    and row["driver"] == "NRPMDDynamics_scalar_loop")
    for row in rows:
        own_scalar = next(r["median_seconds"] for r in rows if r["backend"] == row["backend"]
                          and r["driver"] == "NRPMDDynamics_scalar_loop")
        row["speedup_vs_numpy_scalar"] = baseline/row["median_seconds"]
        row["speedup_vs_same_backend_scalar"] = own_scalar/row["median_seconds"]
        row["trajectory_steps_per_second"] = case["count"]*args.steps/row["median_seconds"]
    return {"case": case["name"], "description": case["description"],
            "trajectory_batch": case["count"], "beads": case["beads"], "electronic_states": case["states"],
            "atoms": case["atoms"], "beta_hartree_inverse": case["beta"], "dt_atomic": case["dt"],
            "masses_electron_mass": case["masses"].tolist(),
            "model_identity": case["model_identity"], "initial_arrays": case["initial_arrays"], "rows": rows}


def _code_identity(args):
    """Pin the loaded checkout and compiled extension, not just version 0.x."""
    root = Path(__file__).resolve().parents[2]
    paths = ["benchmarks/nonadiabatic/electronic_backends.py", "native/electronic.cpp",
             "src/pyrpmd/electronic_backends.py", "src/pyrpmd/batched_nonadiabatic.py",
             "src/pyrpmd/nonadiabatic.py", "src/pyrpmd/nonadiabatic_ensemble.py",
             "src/pyrpmd/dynamics.py",
             "src/pyrpmd/potentials/nonadiabatic_models.py", "src/pyrpmd/potentials/nah2_leps.py"]
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    binaries = {}
    for name in ("pyrpmd._electronic", "pyrpmd._native"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            binary = Path(spec.origin)
            binaries[name] = {"file": binary.name, "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    return {"source_sha256": sources, "available_native_binaries": binaries}


def _environment(args, capabilities):
    versions = {"pyrpmd": __version__, "numpy": np.__version__, "python": platform.python_version()}
    for package in ("scipy", "torch", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    result = {"software": versions, "platform": platform.platform(), "machine": platform.machine(),
              "cpu_model": cpu, "logical_cpus": os.cpu_count(),
              "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "requested_threads": args.threads, "loaded_threadpools_during_measurement": threadpool_info(),
              "electronic_capabilities": capabilities, "torch_device_requested": args.torch_device}
    if capabilities["torch"]:
        import torch
        result["torch"] = {"intraop_threads": torch.get_num_threads(),
                           "interop_threads": torch.get_num_interop_threads(),
                           "cuda_runtime": torch.version.cuda}
        if args.torch_device.startswith("cuda") and torch.cuda.is_available():
            result["torch"]["device_name"] = torch.cuda.get_device_name(args.torch_device)
    return result


def benchmark(args):
    identity = _code_identity(args)
    capabilities = electronic_backend_capabilities()
    available, unavailable = [], []
    requested = list(dict.fromkeys(["numpy", *args.backends]))
    for backend in requested:
        try:
            validate_electronic_backend(backend, _device(backend, args.torch_device))
        except (RuntimeError, ValueError) as error:
            unavailable.append({"backend": backend, "device": _device(backend, args.torch_device),
                                "status": "unavailable", "reason": str(error)})
        else:
            available.append(backend)
    torch_threads = None
    if "torch" in available:
        import torch
        torch_threads = torch.get_num_threads()
        torch.set_num_threads(args.threads)
    try:
        with threadpool_limits(limits=args.threads):
            cases = [_case(name, args.batch, args.beads, args.states, args.seed+i)
                     for i, name in enumerate(("complex_lvc", "nah2_5g"))]
            kernels = [kernel_benchmark(case, available, args) for case in cases]
            propagation = [end_to_end_benchmark(case, available, args) for case in cases]
            environment = _environment(args, capabilities)
    finally:
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
    if identity != _code_identity(args):
        raise RuntimeError("benchmark source or compiled extension changed during measurement; rerun on a stable checkout")
    return {"format": "PyRPMD.electronic_backend_benchmark.v1", "correctness_passed": True,
            "speedup_is_acceptance_gate": False, "environment": environment, "code_identity": identity,
            "settings": {key: value for key, value in vars(args).items() if key != "output"},
            "timing": "perf_counter_ns wall time; explicit CUDA synchronization; median of independent timed repeats",
            "precision": "float64/complex128", "unavailable_backends": unavailable,
            "kernel_cases": kernels, "end_to_end_cases": propagation,
            "scope": "electronic kernel and integration steps; excludes physical validation, I/O, sampling and interprocess scheduling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=("numpy", "native", "torch"), default=["numpy", "native", "torch"])
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--beads", type=int, default=8)
    parser.add_argument("--states", type=int, default=4, help="synthetic complex model state count; molecular NaH2 remains two-state")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ring-native", action="store_true", help="require native free-ring kernels for both compared drivers")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--kernel-inner", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, default=Path("results/electronic_backends.json"))
    args = parser.parse_args()
    if any(getattr(args, key) < 1 for key in ("batch", "beads", "steps", "threads", "warmups", "repeats", "kernel_inner")):
        parser.error("batch, beads, steps, threads, warmups, repeats and kernel-inner must be positive")
    if args.states < 2 or args.seed < 0:
        parser.error("states must be >= 2 and seed nonnegative")
    if args.threads > 1 and not args.ring_native:
        parser.error("threads>1 for these dynamics constructors requires --ring-native with OpenMP support")
    result = benchmark(args)
    write_json(args.output, result)
    print(json.dumps({"correctness_passed": result["correctness_passed"],
                      "output": str(args.output), "unavailable_backends": result["unavailable_backends"],
                      "end_to_end": [{"case": case["case"], "rows": [
                          {key: row[key] for key in ("backend", "driver", "median_seconds", "speedup_vs_numpy_scalar")}
                          for row in case["rows"]]} for case in result["end_to_end_cases"]]}, indent=2))


if __name__ == "__main__":
    main()
