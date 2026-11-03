"""Diagnose T19 LAMMPS batch performance without weakening force correctness.

The default FCC cases reproduce T05's periodic-face grid (offset=0) alongside
an otherwise identical translated grid (offset=0.173). Timing and instrumentation
use separate passes. A pinned --baseline-ref compares the actual old adapter,
and rounds alternate their mode order to reduce systematic shared-host drift.

Example from a checkout with the LAMMPS Python/shared-library dependencies::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
      benchmarks/efficiency/t19_lammps.py --baseline-ref <commit> \
      --atoms 256 2048 4096 --beads 32 --output lammps-t19.json

The optional --beads 8 32 64 matrix can be expensive: persistent mode creates
one complete LAMMPS instance per bead. These are force-only CPU measurements;
they neither measure CUDA nor establish physical convergence.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.efficiency.force_engine_adapters import (
    coordinate_sequence, environment, errors, legacy_provider,
)
from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM
from pyrpmd.potentials.lammps import LAMMPSPotential


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _cells(atoms):
    if atoms % 4:
        raise ValueError("FCC atom counts must be positive multiples of four")
    count = atoms//4
    a = max(value for value in range(1, int(round(count**(1/3)))+2)
            if count % value == 0 and value**3 <= count)
    remainder = count//a
    b = max(value for value in range(1, int(remainder**.5)+1) if remainder % value == 0)
    return np.array([a, b, remainder//b], dtype=int)


def _write_model(directory, atoms, offset):
    lattice = 5.26
    cells = _cells(atoms)
    lengths = cells*lattice
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.stack(np.meshgrid(*(np.arange(int(c)) for c in cells), indexing="ij"), axis=-1)
    q = ((grid.reshape(-1, 1, 3)+basis[None])*lattice).reshape(-1, 3)
    q = (q+offset*lattice) % lengths
    if len(q) != atoms:
        raise AssertionError("FCC grid did not produce the requested atom count")
    data = directory/"lj.data"
    atom_lines = "\n".join(f"{i+1} 1 {r[0]:.17g} {r[1]:.17g} {r[2]:.17g}"
                           for i, r in enumerate(q))
    box_lines = "\n".join(f"0 {length:.17g} {axis}lo {axis}hi"
                          for axis, length in zip("xyz", lengths))
    data.write_text(f"T19 FCC LJ\n\n{atoms} atoms\n1 atom types\n\n{box_lines}\n\n"
                    f"Masses\n\n1 39.948\n\nAtoms # atomic\n\n{atom_lines}\n")
    cutoff = min(8.5, .45*float(np.min(lengths)))
    script = directory/"lj.in"
    script.write_text(
        'units metal\natom_style atomic\nboundary p p p\n'
        'atom_modify map array\nread_data ${pyrpmd_model_dir}/lj.data\n'
        f'pair_style lj/cut {cutoff:.17g}\npair_coeff 1 1 0.0103 3.4\n'
        'neighbor 2.0 bin\n')
    options = dict(model=script, symbols=["Ar"]*atoms, type_map=["Ar"],
                   cell=np.diag(lengths)*ANGSTROM_TO_AU, pbc=[True]*3,
                   asset_files=[data])
    geometry = {"cell_angstrom": lengths.tolist(), "fcc_cells": cells.tolist(),
                "cutoff_angstrom": cutoff, "neighbor_skin_angstrom": 2.0,
                "lattice_angstrom": lattice, "epsilon_ev": .0103, "sigma_angstrom": 3.4}
    return options, q*ANGSTROM_TO_AU, geometry


def _counters(provider):
    return dict(provider.batch_stats)


def _counter_delta(before, after):
    return {name: after[name]-before[name] for name in ("neighbor_rebuilds", "neighbor_reuses")}


def _compare(result, reference, maxima):
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(result, name), getattr(reference, name),
                                   rtol=2e-10, atol=1e-12,
                                   err_msg=f"LAMMPS {name} disagrees with full setup")
    for name, value in errors(result, reference).items():
        maxima[name] = max(maxima.get(name, 0.0), value)


def _measure(make_provider, frames, references, warmup):
    started = time.perf_counter()
    provider = make_provider()
    construction = time.perf_counter()-started
    wall, cpu, maximum = [], [], {}
    try:
        started = time.perf_counter()
        first = provider.evaluate(frames[0])
        first_seconds = time.perf_counter()-started
        _compare(first, references[0], maximum)
        before = None
        for index, (frame, reference) in enumerate(zip(frames, references)):
            if index == warmup:
                before = _counters(provider)
            t0, c0 = time.perf_counter(), time.process_time()
            result = provider.evaluate(frame)
            c1, t1 = time.process_time(), time.perf_counter()
            if index >= warmup:
                wall.append(t1-t0)
                cpu.append(c1-c0)
            _compare(result, reference, maximum)
        after = _counters(provider)
        return {"construction_seconds": construction, "first_evaluation_seconds": first_seconds,
                "wall_seconds": wall, "cpu_seconds": cpu, "median_seconds": statistics.median(wall),
                "median_cpu_seconds": statistics.median(cpu), "instances": after["instances"],
                "timed_counters": _counter_delta(before, after), "error_against_full_setup": maximum}
    finally:
        provider.close()


def _profile(make_provider, frames, references, warmup, count):
    """Disjoint method timers; this pass is excluded from published medians."""
    phases = {name: {"seconds": 0.0, "calls": 0} for name in (
        "engine_validation", "atom_validation", "box_reads", "force_sweep")}
    maximum = {}
    with make_provider() as provider:
        provider.evaluate(frames[0])  # Allocate every persistent slot first.
        for frame in frames[:warmup]:
            provider.evaluate(frame)

        def timer(function, phase):
            @wraps(function)
            def measured(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    phases[phase]["seconds"] += time.perf_counter()-started
                    phases[phase]["calls"] += 1
            return measured

        total = 0.0
        before = _counters(provider)
        with ExitStack() as stack:
            for slot in [provider, *provider._replicas]:
                for method, phase in (("_check_engine", "engine_validation"),
                                      ("_validate_atoms", "atom_validation"), ("_box_data", "box_reads")):
                    stack.enter_context(patch.object(slot, method, timer(getattr(slot, method), phase)))
                stack.enter_context(patch.object(slot.engine, "command", timer(slot.engine.command, "force_sweep")))
            used = min(count, len(frames)-warmup)
            for frame, reference in zip(frames[warmup:warmup+used], references[warmup:warmup+used]):
                started = time.perf_counter()
                result = provider.evaluate(frame)
                total += time.perf_counter()-started
                _compare(result, reference, maximum)
        phases["other_adapter_work"] = {
            "seconds": max(0.0, total-sum(value["seconds"] for value in phases.values())),
            "calls": used,
        }
        for record in phases.values():
            record["seconds_per_batch"] = record["seconds"]/used
            record["fraction_of_profiled_wall"] = record["seconds"]/total
        return {"evaluations": used, "total_seconds": total, "phases": phases,
                "timed_counters": _counter_delta(before, _counters(provider)),
                "interpretation": "Instrumented inclusive methods are disjoint; run includes native force/setup work. This is a separate pass, not a timing speedup denominator."}


def _case(atoms, beads, offset, args, previous):
    with tempfile.TemporaryDirectory(prefix="pyrpmd-t19-lammps-") as tmp:
        options, q, geometry = _write_model(Path(tmp), atoms, offset)
        frames = coordinate_sequence(q, beads, args.warmup+args.repeats, args.seed)
        # Every candidate uses exactly the same reference configurations and
        # pure full-setup forces, including ghost and reciprocal-space virials.
        with previous(**options) as reference_provider:
            references = [reference_provider.evaluate(frame) for frame in frames]
        factories = {}
        if args.baseline_ref:
            factories.update(baseline_serial=lambda: previous(**options),
                             baseline_persistent=lambda: previous(**options, batch_mode="persistent",
                                                                  neighbor_skin=2*ANGSTROM_TO_AU))
        factories.update(serial=lambda: LAMMPSPotential(**options),
                         persistent=lambda: LAMMPSPotential(**options, batch_mode="persistent",
                                                             neighbor_skin=2*ANGSTROM_TO_AU))
        rounds = {name: [] for name in factories}
        orders = []
        for round_index in range(args.rounds):
            order = list(factories)
            if round_index % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                rounds[name].append(_measure(factories[name], frames, references, args.warmup))
        modes = {}
        for name, values in rounds.items():
            median = statistics.median(value["median_seconds"] for value in values)
            maximum = {}
            for value in values:
                for key, error in value["error_against_full_setup"].items():
                    maximum[key] = max(maximum.get(key, 0.0), error)
            modes[name] = {"median_seconds": median,
                           "median_cpu_seconds": statistics.median(value["median_cpu_seconds"] for value in values),
                           "rounds": values, "error_against_full_setup": maximum,
                           "profile": _profile(factories[name], frames, references, args.warmup,
                                               args.profile_evaluations)}
        for name, value in modes.items():
            value["speedup_over_current_serial"] = modes["serial"]["median_seconds"]/value["median_seconds"]
            if args.baseline_ref:
                baseline = "baseline_persistent" if name.endswith("persistent") else "baseline_serial"
                value["speedup_over_previous_same_mode"] = modes[baseline]["median_seconds"]/value["median_seconds"]
        return {"physical_atoms": atoms, "beads": beads, "offset_lattice_fraction": offset,
                "geometry": geometry, "round_orders": orders, "modes": modes}


def _save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=_positive, nargs="+", default=[256, 2048, 4096])
    parser.add_argument("--beads", type=_positive, nargs="+", default=[32])
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, .173])
    parser.add_argument("--warmup", type=_positive, default=2)
    parser.add_argument("--repeats", type=_positive, default=7)
    parser.add_argument("--rounds", type=_positive, default=3)
    parser.add_argument("--profile-evaluations", type=_positive, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--baseline-ref", help="Pinned Git commit containing the adapter to compare")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(n % 4 for n in args.atoms):
        parser.error("--atoms must contain multiples of four for an FCC grid")
    if not all(np.isfinite(offset) for offset in args.offsets):
        parser.error("--offsets must be finite")
    previous = legacy_provider("lammps", LAMMPSPotential, args.baseline_ref)
    adapter_path = ROOT/"src/pyrpmd/potentials/lammps.py"
    report = {"schema": "pyrpmd.t19.lammps-profile.v1", "complete": False,
              "physical_converged": False, "environment": environment(), "baseline_ref": args.baseline_ref,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {"adapter": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                                "benchmark": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "timing_unit": "seconds per complete evaluate([P,N,3]) call",
              "summary_statistic": "median of per-round medians; unprofiled passes only",
              "warmup": args.warmup, "timed_evaluations_per_round": args.repeats, "rounds": args.rounds,
              "seed": args.seed, "planned_cases": len(args.atoms)*len(args.beads)*len(args.offsets),
              "comparison_tolerance": {"rtol": 2e-10, "atol_atomic_units": 1e-12}, "cases": []}
    if args.baseline_ref:
        baseline_source = subprocess.check_output(
            ["git", "show", f"{args.baseline_ref}:src/pyrpmd/potentials/lammps.py"], cwd=ROOT)
        report["source_sha256"]["baseline_adapter"] = hashlib.sha256(baseline_source).hexdigest()
    _save(args.output, report)
    try:
        for atoms in args.atoms:
            for beads in args.beads:
                for offset in args.offsets:
                    case = _case(atoms, beads, offset, args, previous)
                    report["cases"].append(case)
                    _save(args.output, report)
                    print(json.dumps({"atoms": atoms, "beads": beads, "offset": offset,
                                      "median_seconds": {k: v["median_seconds"] for k, v in case["modes"].items()}}), flush=True)
        report["complete"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _save(args.output, report)


if __name__ == "__main__":
    main()
