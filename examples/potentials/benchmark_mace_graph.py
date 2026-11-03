#!/usr/bin/env python3
"""Measure CPU graph construction before/after the T04 MACE copy refinement.

This benchmark excludes model inference, autograd and public input validation.
It checks every graph field against the implementation published at ef7c468.
The model is initialized locally; no weights are downloaded.

Run from an installed checkout with the ``mace-batched`` extra:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
      examples/potentials/benchmark_mace_graph.py --output mace-graph.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

if __package__:
    from ._random_mace import make_random_mace
    from .benchmark_batched_ml import _geometry
else:
    from _random_mace import make_random_mace
    from benchmark_batched_ml import _geometry


REFERENCE_COMMIT = "ef7c468481715e4fa6a5eb35148d2094ad9aee31"


def _published_graph(provider, q):
    """The graph assembly operations from the reference commit, unchanged."""
    torch = provider.torch
    frames, atoms = len(q), len(provider.symbols)
    positions = (q.detach() / provider.length_factor).contiguous()
    cell, flags = provider._cell_tensor(), provider._pbc_tensor()
    indices, shifts = [], []
    for frame in range(frames):
        i, j, shift = provider._neighbors.compute(
            positions[frame], cell, flags, quantities="ijS", copy=True,
        )
        indices.append(torch.stack((i, j)) + frame * atoms)
        shifts.append(shift)
    unit_shifts = torch.cat(shifts).to(dtype=provider.dtype)
    graph = provider._template(frames)
    graph.update(
        positions=positions.reshape(-1, 3), edge_index=torch.cat(indices, dim=1),
        unit_shifts=unit_shifts, shifts=unit_shifts @ cell,
        cell=cell.expand(frames, 3, 3).reshape(-1, 3), pbc=flags.expand(frames, 3),
    )
    return graph


def _memory_record(torch, function):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
    ) as profile:
        result = function()
    events = profile.key_averages()
    # Keep graph storage alive until profiling finishes, just as for timings.
    del result
    return {
        "torch_allocation_traffic_bytes": sum(
            max(0, event.self_cpu_memory_usage) for event in events
        ),
        "torch_clone_calls": sum(event.count for event in events if event.key == "aten::clone"),
        "torch_stack_calls": sum(event.count for event in events if event.key == "aten::stack"),
        "note": (
            "Sum of positive self_cpu_memory_usage over Torch profiler operator groups; "
            "not peak process RSS, and excludes native Vesin allocations."
        ),
    }


def run(*, atoms=(64, 512), frames=(8, 32, 64), dtypes=("float64", "float32"),
        warmup=3, repeats=15, threads=1):
    """Return alternating CPU timings and a separate B=32 allocation profile."""
    import torch
    from threadpoolctl import threadpool_limits
    from pyrpmd.potentials import MACEBatchedPotential

    for name, values in (("atoms", atoms), ("frames", frames)):
        if not values or any(type(value) is not int or value < 1 for value in values):
            raise ValueError(f"{name} must contain positive integers")
    if not dtypes or any(dtype not in ("float64", "float32") for dtype in dtypes):
        raise ValueError("dtypes must contain float64 and/or float32")
    for name, value, minimum in (("warmup", warmup, 1), ("repeats", repeats, 3),
                                 ("threads", threads, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    torch.set_num_threads(threads)
    rows = []
    with threadpool_limits(limits=threads):
        for dtype in dtypes:
            model = make_random_mace(dtype=dtype)
            for atom_count in atoms:
                for frame_count in frames:
                    q, cell, symbols = _geometry(atom_count, frame_count)
                    with MACEBatchedPotential(
                        symbols, model, dtype=dtype, cell=cell, pbc=[True] * 3,
                    ) as provider:
                        q = torch.tensor(q, dtype=getattr(torch, dtype))
                        functions = {
                            "published": lambda: _published_graph(provider, q),
                            "refined": lambda: provider._graph(q),
                        }
                        reference, refined = functions["published"](), functions["refined"]()
                        if not refined["edge_index"].is_contiguous():
                            raise AssertionError("refined edge_index must be contiguous")
                        if reference.keys() != refined.keys() or any(
                            not torch.equal(value, refined[key]) for key, value in reference.items()
                        ):
                            raise AssertionError("graph fields differ from the reference")
                        for _ in range(warmup):
                            for function in functions.values():
                                function()
                        timings = {name: [] for name in functions}
                        for repetition in range(repeats):
                            order = list(functions)
                            if repetition % 2:
                                order.reverse()
                            for name in order:
                                start = time.perf_counter()
                                result = functions[name]()
                                timings[name].append(1000 * (time.perf_counter() - start))
                        del result
                        medians = {name: statistics.median(values) for name, values in timings.items()}
                        allocation = {
                            name: _memory_record(torch, function) for name, function in functions.items()
                        } if frame_count == 32 else None
                        rows.append({
                            "dtype": dtype, "atoms": atom_count, "frames": frame_count,
                            "edges": reference["edge_index"].shape[1],
                            "all_graph_fields_bit_identical": True, "median_ms": medians,
                            "graph_stage_speedup": medians["published"] / medians["refined"],
                            "timings_ms": timings, "allocation_profile": allocation,
                        })
    return {
        "scope": "MACE graph construction only, no model inference/autograd or public validation",
        "reference_commit": REFERENCE_COMMIT,
        "refined_variant": "owned pairs+offset then transpose; final edge_index concatenation is contiguous",
        "device": "cpu",
        "hardware": {"machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "mace-torch", "vesin-torch",
        )},
        "threads": threads, "warmup": warmup, "repeats": repeats,
        "gpu_measured": False, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--dtypes", nargs="+", choices=["float64", "float32"],
                        default=["float64", "float32"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("mace-graph-timing.json"))
    args = parser.parse_args()
    output = args.output
    options = vars(args)
    del options["output"]
    result = run(**options)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f"{row['dtype']} N={row['atoms']} B={row['frames']}: "
              f"graph-stage ratio {row['graph_stage_speedup']:.3f}")
    print(f"Full CPU graph-stage record: {output}")


if __name__ == "__main__":
    main()
