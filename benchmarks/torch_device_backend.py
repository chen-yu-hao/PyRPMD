"""Synthetic model benchmark, not a chemistry/GPU speedup claim.

Run from a checkout with PyRPMD's gpu extra installed:
  PYTHONPATH=src python benchmarks/torch_device_backend.py --device cuda

The frame-loop and batched cases use the same Hamiltonian. The RPC/MTS case
changes the Hamiltonian and timestep splitting; compare its cost, not its
trajectory, and converge the approximation before any scientific use.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from pyrpmd.torch_backend import TorchPES, TorchRPCTerm, TorchSplitPotential, TorchState, TorchDynamics


class Model(torch.nn.Module):
    metadata = {"batch_independent": True}

    def __init__(self, width):
        super().__init__()
        # Identical weights across benchmark cases without changing caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(3, width), torch.nn.SiLU(),
                torch.nn.Linear(width, width), torch.nn.SiLU(),
                torch.nn.Linear(width, 1),
            )
        self.calls = self.frames = 0

    def forward(self, q):
        self.calls += 1
        self.frames += q.shape[0]
        return self.network(q).sum((1, 2)) + 0.01 * q.square().sum((1, 2))


class FrameLoop(torch.nn.Module):
    metadata = {"batch_independent": True}

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, q):
        return torch.cat([self.model(frame.unsqueeze(0)) for frame in q])


class Zero(torch.nn.Module):
    metadata = {"batch_independent": True}

    def forward(self, q):
        return 0.0 * q.sum((1, 2))


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beads", type=positive, default=32)
    parser.add_argument("--trajectories", type=positive, default=16)
    parser.add_argument("--atoms", type=positive, default=12)
    parser.add_argument("--steps", type=positive, default=20)
    parser.add_argument("--width", type=positive, default=64)
    parser.add_argument("--contracted-beads", type=positive, default=4)
    parser.add_argument("--inner-steps", type=positive, default=4)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--threads", type=positive, default=1)
    args = parser.parse_args()
    if args.contracted_beads > args.beads:
        parser.error("contracted-beads cannot exceed beads")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(29)
    shape = (args.trajectories, args.beads, args.atoms, 3)
    q0 = (0.1 * torch.randn(shape, generator=generator, dtype=torch.float64)).to(args.device)
    p0 = torch.zeros_like(q0)
    rows, endpoints = [], []
    for name in ("frame_loop", "batched", "rpc_mts"):
        model = Model(args.width)
        pes = TorchPES(FrameLoop(model) if name == "frame_loop" else model, device=args.device)
        potential = pes
        inner_steps = 1
        if name == "rpc_mts":
            fast = TorchRPCTerm(TorchPES(Zero(), device=args.device), args.beads)
            slow = TorchRPCTerm(pes, args.beads, args.contracted_beads)
            potential = TorchSplitPotential([fast], [slow])
            inner_steps = args.inner_steps
        dyn = TorchDynamics(potential, [1.] * args.atoms, 100., args.beads, .02,
                            inner_steps=inner_steps)
        dyn.propagate(TorchState(q0, p0), 2)  # Warmup, outside measured blocks.
        timings = []
        for _ in range(args.repeats):
            state = TorchState(q0, p0)
            model.calls = model.frames = 0
            sync(args.device)
            start = time.perf_counter()
            dyn.propagate(state, args.steps)
            sync(args.device)
            timings.append(time.perf_counter() - start)
        endpoints.append((state.q.clone(), state.p.clone()))
        rows.append({"case": name, "seconds": timings,
                     "median_seconds": statistics.median(timings),
                     "expensive_model_calls_per_block": model.calls,
                     "expensive_frames_per_block": model.frames,
                     "approximate_hamiltonian": name == "rpc_mts" and args.contracted_beads != args.beads})
    for i in (0, 1):
        torch.testing.assert_close(endpoints[0][i], endpoints[1][i], rtol=1e-10, atol=1e-10)
    print(json.dumps({"torch": torch.__version__, "settings": vars(args),
                      "batched_matches_frame_loop": True, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
