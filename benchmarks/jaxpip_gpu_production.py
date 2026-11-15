#!/usr/bin/env python3
"""Measure a checked JaxPIP fit on a real GPU production path.

This benchmark deliberately measures only the direct JaxPIP -> DLPack ->
TorchDynamics route. It reports compilation separately from steady-state block
time, varies the actual ``K*P`` frame batch, and records allocated/reserved
CUDA memory. It does not claim chemical accuracy; the fit manifest must carry
its own test-set acceptance unless ``--allow-unvalidated`` is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_directory", type=Path,
                   help="checked pyrpmd.jaxpip-fit/1 directory")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--beads", type=int, default=16)
    p.add_argument("--trajectories", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=16)
    p.add_argument("--batch-repeats", type=int, default=30)
    p.add_argument("--max-frames-per-call", type=int)
    p.add_argument("--allow-unvalidated", action="store_true")
    return p


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nvidia_memory_mib():
    """Read total device usage, including JAX's allocator, when available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return [int(value.strip()) for value in output.splitlines() if value.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def run(args):
    if args.beads < 1 or args.steps < 1 or args.warmup_steps < 0 or args.batch_repeats < 1:
        raise ValueError("beads, steps and batch-repeats must be positive; warmup-steps is nonnegative")
    if any(k < 1 for k in args.trajectories):
        raise ValueError("trajectory counts must be positive")
    import torch
    from pyrpmd.potentials.jaxpip import JaxPIPPotential
    from pyrpmd.torch_backend import TorchDynamics, TorchState

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device} is unavailable")
    # The fitted manifest supplies atom count and ordered composition. The
    # probe geometry is intentionally deterministic and lives in model units.
    from pyrpmd.pesgen.jaxpip_fit import load_fit
    manifest = load_fit(args.model_directory)
    contract = manifest["contract"]
    natoms = len(contract["atomic_numbers"])
    settings = manifest.get("settings", {})
    position_unit = settings.get("position_unit", "bohr")
    rng = np.random.default_rng(20261115)
    probe = rng.normal(0.0, 0.05, (natoms, 3)).astype(np.float64)
    # Keep the probe in a compact molecular-scale cloud. The benchmark is a
    # runtime test; model-domain quality remains the fit/test-set contract.
    q = np.broadcast_to(probe, (max(args.trajectories) * args.beads, natoms, 3)).copy()
    provider = JaxPIPPotential.from_spec({
        "model_directory": str(args.model_directory),
        "device": args.device,
        "max_frames_per_call": args.max_frames_per_call,
        "allow_unvalidated": args.allow_unvalidated,
    })
    try:
        adapter = provider.as_torch_pes()
        compile_start = time.perf_counter()
        provider.warmup_torch(args.beads, natoms)
        _sync(torch, device)
        compile_seconds = time.perf_counter() - compile_start
        evaluation = []
        for trajectories in args.trajectories:
            frames = trajectories * args.beads
            positions = torch.as_tensor(q[:frames], dtype=torch.float64,
                                        device=device).contiguous()
            adapter.evaluate_tensor(positions)
            _sync(torch, device)
            start = time.perf_counter()
            for _ in range(args.batch_repeats):
                adapter.evaluate_tensor(positions)
            _sync(torch, device)
            seconds = (time.perf_counter() - start) / args.batch_repeats
            evaluation.append({
                "trajectories": trajectories, "beads": args.beads,
                "frames": frames, "seconds_per_batch": seconds,
                "frames_per_second": frames / seconds,
            })

        dynamics = []
        masses = np.ones(natoms, dtype=np.float64)
        beta = 1.0 / (3.166811563e-6 * 300.0)
        for trajectories in args.trajectories:
            frames = trajectories * args.beads
            q0 = torch.as_tensor(q[:frames], dtype=torch.float64, device=device)
            q0 = q0.reshape(trajectories, args.beads, natoms, 3)
            p0 = torch.zeros_like(q0)
            engine = TorchDynamics(adapter, masses, beta, args.beads, 0.1,
                                   integrator="cayley-bcb", seed=11)
            state = TorchState(q0, p0)
            engine.propagate(state, args.warmup_steps)
            _sync(torch, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            nvidia_before = _nvidia_memory_mib()
            start = time.perf_counter()
            engine.propagate(state, args.steps)
            _sync(torch, device)
            seconds = time.perf_counter() - start
            row = {
                "trajectories": trajectories, "beads": args.beads,
                "steps": args.steps, "seconds": seconds,
                "steps_per_second": args.steps / seconds,
                "frames_per_second": frames * args.steps / seconds,
                "force_evaluations": dict(engine.force_evaluations),
                "finite": bool(torch.isfinite(state.q).all().item()
                               and torch.isfinite(state.p).all().item()),
            }
            if device.type == "cuda":
                row.update(
                    cuda_allocated_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    cuda_reserved_peak_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                    nvidia_memory_before_mib=nvidia_before[device.index or 0] if nvidia_before else None,
                    nvidia_memory_after_mib=(_nvidia_memory_mib() or [None])[device.index or 0],
                )
            dynamics.append(row)
        result = {
            "schema": "pyrpmd.jaxpip-gpu-benchmark/1",
            "model_directory": str(args.model_directory.resolve()),
            "model_manifest": {"fingerprint": manifest.get("fingerprint"),
                               "within_requested_test_bounds": manifest.get("within_requested_test_bounds"),
                               "contract": contract, "settings": settings},
            "runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "device": str(device),
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                        "jax": provider.metadata["jax_version"],
                        "jaxpip": provider.metadata["jaxpip_version"],
                        "dtype": "float64", "compile_seconds": compile_seconds,
                        "max_frames_per_call": args.max_frames_per_call},
            "settings": vars(args) | {"model_directory": str(args.model_directory)},
            "evaluation": evaluation, "dynamics": dynamics,
            "limitations": [
                "timings include JAX/Torch bridge execution after one explicit warmup",
                "this benchmark measures runtime throughput and memory, not chemical accuracy",
                "rate convergence still requires the real US/AUS/recrossing production input",
            ],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return result
    finally:
        provider.close()


if __name__ == "__main__":
    run(_parser().parse_args())
