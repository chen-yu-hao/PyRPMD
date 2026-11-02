#!/usr/bin/env python3
"""Measure real MACE inference: separate ASE frames versus one graph batch.

The small model is initialized locally, never trained, and never downloaded.
Timings include neighbour construction, unit conversion and NumPy transfers;
model construction is excluded.  The benchmark checks numerical agreement but
does not assess physical accuracy or converged molecular dynamics.

Run from an installed checkout with the ``mace-batched`` extra:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
      examples/potentials/benchmark_batched_ml.py --output mace-batch.json
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __package__:
    from ._random_mace import MODEL_DESCRIPTION, make_random_mace
else:
    from _random_mace import MODEL_DESCRIPTION, make_random_mace


def _hardware(torch, device):
    cpu_name = platform.processor() or None
    if cpu_name == platform.machine():
        # Virtualized runtimes sometimes expose the instruction set here,
        # without identifying the underlying processor model.
        cpu_name = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = None
    return {
        "cpu_model": cpu_name,
        "cpu_model_note": None if cpu_name else "host CPU model is not exposed by this environment",
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": available_cpus,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _geometry(atoms, frames, seed=42061):
    from pyrpmd.potentials.python import BOHR_ANGSTROM

    side = int(np.ceil(atoms ** (1 / 3)))
    lattice = np.stack(np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), axis=-1)
    base = lattice.reshape(-1, 3)[:atoms] * 1.9
    rng = np.random.default_rng(seed)
    q = (base + rng.normal(scale=0.025, size=(frames, atoms, 3))) / BOHR_ANGSTROM
    cell = np.eye(3) * (side * 1.9) / BOHR_ANGSTROM
    symbols = ["O" if i % 4 == 0 else "H" for i in range(atoms)]
    return q, cell, symbols


def _relative_norm(actual, reference):
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny))


def run(*, beads=(8, 32, 64), atoms=(64, 512), trajectories=1,
        repeats=3, warmup=2, threads=1, dtype="float64", device="auto"):
    """Return per-step timings and actual framework forward-call counts."""
    import torch
    from threadpoolctl import threadpool_limits
    from mace.calculators import MACECalculator
    from pyrpmd.potentials.adapters import ModelASEPotential
    from pyrpmd.potentials.batched_ml import MACEBatchedPotential

    for name, value, minimum in (("trajectories", trajectories, 1), ("repeats", repeats, 3),
                                 ("warmup", warmup, 2), ("threads", threads, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, values in (("beads", beads), ("atoms", atoms)):
        if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError(f"{name} must be a nonempty list of positive integers")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if device == "auto":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
    else:
        requested = torch.device(device)
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("only cpu and cuda devices are supported")
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable")
        devices = [requested]

    torch.set_num_threads(threads)
    base_model = make_random_mace(dtype=dtype)
    versions = {name: importlib.metadata.version(name) for name in (
        "mace-torch", "torch", "e3nn", "ase", "vesin-torch", "numpy",
    )}
    tolerance = {"rtol": 1e-10, "atol": 2e-13} if dtype == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    rows = []
    with threadpool_limits(limits=threads):
        for target in devices:
            def synchronize():
                if target.type == "cuda":
                    torch.cuda.synchronize(target)

            for n in atoms:
                for p in beads:
                    frames = p * trajectories
                    q, cell, symbols = _geometry(n, frames)
                    batch_model, scalar_model = deepcopy(base_model), deepcopy(base_model)
                    counters = {"scalar": 0, "batch": 0}

                    def count_scalar(_, __):
                        counters["scalar"] += 1

                    def count_batch(_, __):
                        counters["batch"] += 1

                    handles = [scalar_model.register_forward_pre_hook(count_scalar),
                               batch_model.register_forward_pre_hook(count_batch)]
                    calculator = MACECalculator(
                        models=scalar_model, device=str(target), default_dtype=dtype,
                        compute_stress=True,
                    )
                    scalar = ModelASEPotential(symbols, calculator, provider="mace",
                                               device=str(target), cell=cell, pbc=[True] * 3)
                    batch = MACEBatchedPotential(symbols, batch_model, device=str(target),
                                                 dtype=dtype, cell=cell, pbc=[True] * 3,
                                                 virial=True)
                    paths = {"scalar": scalar, "batch": batch}
                    elapsed = {"scalar": [], "batch": []}
                    outputs = {}
                    try:
                        for _ in range(warmup):
                            for name, provider in paths.items():
                                if name == "scalar":
                                    # Also benchmark an actual model call for
                                    # an explicitly requested one-frame case.
                                    calculator.reset()
                                provider.evaluate(q)
                                synchronize()
                        counters.update(scalar=0, batch=0)
                        for repeat in range(repeats):
                            order = ("scalar", "batch") if repeat % 2 == 0 else ("batch", "scalar")
                            for name in order:
                                if name == "scalar":
                                    calculator.reset()
                                synchronize()
                                start = time.perf_counter()
                                outputs[name] = paths[name].evaluate(q)
                                synchronize()
                                elapsed[name].append(time.perf_counter() - start)
                        expected, actual = outputs["scalar"], outputs["batch"]
                        np.testing.assert_allclose(actual.energy, expected.energy, **tolerance)
                        np.testing.assert_allclose(actual.forces, expected.forces, **tolerance)
                        np.testing.assert_allclose(actual.virial, expected.virial, **tolerance)
                        if counters != {"scalar": repeats * frames, "batch": repeats}:
                            raise RuntimeError(f"unexpected model forward counts: {counters}")
                        scalar_median = float(np.median(elapsed["scalar"]))
                        batch_median = float(np.median(elapsed["batch"]))
                        rows.append({
                            "device": str(target), "beads": p, "trajectories": trajectories,
                            "frames": frames, "atoms": n,
                            "scalar_seconds": elapsed["scalar"], "batch_seconds": elapsed["batch"],
                            "scalar_median_ms_per_step": 1000 * scalar_median,
                            "batch_median_ms_per_step": 1000 * batch_median,
                            "speedup": scalar_median / batch_median,
                            "scalar_forwards_per_step": counters["scalar"] // repeats,
                            "batch_forwards_per_step": counters["batch"] // repeats,
                            "energy_relative_norm_error": _relative_norm(actual.energy, expected.energy),
                            "forces_relative_norm_error": _relative_norm(actual.forces, expected.forces),
                            "virial_relative_norm_error": _relative_norm(actual.virial, expected.virial),
                            "maximum_force_error_hartree_per_bohr": float(np.max(np.abs(actual.forces - expected.forces))),
                        })
                        print(f"{target}: P={p}, N={n}: {1000 * scalar_median:.3f} -> "
                              f"{1000 * batch_median:.3f} ms ({scalar_median / batch_median:.2f}x)", file=sys.stderr)
                    finally:
                        scalar.close()
                        batch.close()
                        for handle in handles:
                            handle.remove()
    return {
        "benchmark": "real untrained MACE: ASE per-frame versus one native graph batch",
        "physical_accuracy_claim": False,
        "hardware": [_hardware(torch, target) for target in devices],
        "versions": versions,
        "python": platform.python_version(),
        "model": MODEL_DESCRIPTION,
        "settings": {"dtype": dtype, "threads": threads, "torch_threads": torch.get_num_threads(),
                     "warmup": warmup, "repeats": repeats, "virial": True,
                     "pbc": [True, True, True], "numpy_boundary_included": True,
                     "neighbor_construction_included": True,
                     "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "tolerance": tolerance,
        "cuda_available": torch.cuda.is_available(),
        "cuda_measured": any(target.type == "cuda" for target in devices),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beads", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--atoms", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--device", default="auto", help="cpu, cuda:N, or auto (CPU plus CUDA if available)")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    # Some optional frameworks print initialization messages to stdout.
    with redirect_stdout(sys.stderr):
        result = run(**args)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
