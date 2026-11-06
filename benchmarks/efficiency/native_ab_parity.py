"""Bit-for-bit A/B check of the installed ``pyrpmd._native`` against a reference build.

Build the reference extension from another checkout (for example the merge
base) and pass its ``_native*.so``. Both modules then advance identical inputs
through ``free_step``, ``advance`` (propagate_block), ``advance_sampling``
(PILE-L blocks) and ``run_block`` (Dynamics.propagate), and every returned
state, record, energy and status is compared with exact equality.

The default PES is the built-in native harmonic potential on the H+CH4
geometry with the published four-channel RPMDrate coordinate. ``--cbe-library``
uses a locally built CBE/PES-2008 library instead (see benchmarks/hch4).

    python benchmarks/efficiency/native_ab_parity.py --reference /path/to/_native.so
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyrpmd import _native as candidate
from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias, WallBias
from pyrpmd.block_propagation import _native_plan, bias_descriptors, cv_descriptors
from pyrpmd.constants import FS_TO_AU, KB_AU
from pyrpmd.cv import CVStack, Distance
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials._builtin_native import resolve_native_binding
from pyrpmd.potentials.native import descriptor_binding
from pyrpmd.workflows.rate import project_momenta, project_positions

from benchmarks.efficiency.native_step_overhead import setup


def load(path):
    spec = importlib.util.spec_from_file_location("_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    return a == b


class Comparison:
    def __init__(self, reference):
        self.reference, self.count, self.failures = reference, 0, []

    def check(self, name, run):
        outputs = [run(module) for module in (self.reference, candidate)]
        self.count += 1
        if len(outputs[0]) != len(outputs[1]) or not all(same(a, b) for a, b in zip(*outputs)):
            self.failures.append(name)


def start(geometry, masses, beta, beads, seed):
    rng = np.random.default_rng(seed)
    q = np.repeat(geometry[None], beads, axis=0) + rng.normal(scale=2e-3, size=(beads, 6, 3))
    return State(q, rng.normal(size=q.shape) * np.sqrt(masses[None, :, None] * beads / beta))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, type=Path, help="reference _native extension")
    parser.add_argument("--cbe-library", type=Path, help="optional CBE/PES-2008 library")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    geometry, masses, coordinate = setup()
    beta = 1 / (KB_AU * 300.0)
    if args.cbe_library:
        from benchmarks.hch4.pes import CBE2008Potential
        from pyrpmd.potentials import PythonPotential
        provider = PythonPotential(CBE2008Potential(args.cbe_library))
    else:
        provider = HarmonicPotential(k=1e-3)
    ab = Comparison(load(args.reference))

    for scheme in ("cayley-bcb", "normal-mode-bab", "velocity-verlet"):
        for beads in (1, 2, 3, 5, 8, 31, 32, 64, 128):
            for threads in (1, 2):
                dt = 0.5 if scheme == "velocity-verlet" else 4.0
                rng = np.random.default_rng(beads)
                q0, p0 = rng.normal(size=(beads, 6, 3)), 30 * rng.normal(size=(beads, 6, 3))

                def free(module):
                    try:
                        context = module.create(masses.tolist(), 1000.0, beads, dt, threads, scheme)
                    except (ValueError, RuntimeError) as error:
                        return (str(error),)
                    q, p = q0.copy(), p0.copy()
                    for _ in range(5):
                        module.free_step(context, q, p)
                    return q, p
                ab.check(f"free_step {scheme} P{beads} t{threads}", free)

    uf = CompositeBias(UmbrellaBias(coordinate, [0.9], 30.0), FluxMetricBias(coordinate, masses, beta))
    stack = CVStack([coordinate])
    mixed = CompositeBias(UmbrellaBias(stack, [0.9], 30.0), FluxMetricBias(coordinate, masses, beta),
                          WallBias(Distance((1, 5)), 0.01, lower=0.5))
    for beads in (1, 2, 3, 8, 32):
        dyn = Dynamics(provider, masses, beta, beads, 0.1 * FS_TO_AU, native="required",
                       threads=1, loop_backend="native", seed=1)
        state = start(geometry, masses, beta, beads, 11 + beads)
        steps = 23
        mask = (np.arange(1, steps + 1) % 4 == 0).astype(np.uint8)

        def block(bias, cv, extra, *, constraint=None, fragments=(), first=state, states=True):
            def run(module):
                binding, options = _native_plan(dyn, bias, cv, constraint, fragments, 1e-10, True, first.time)
                options.update(extra)
                options["step_energies"] = np.full(steps, np.nan)
                if options.get("previous"):
                    options["previous"] = np.full((2, beads, 6, 3), np.nan)
                width = 2 + 36 + (cv.dimension if cv is not None else 0)
                records = np.full((int(mask.sum()), width), np.nan)
                snapshots = np.full((int(mask.sum()), 2, beads, 6, 3), np.nan) if states else None
                context = module.create(masses.tolist(), beta, beads, dyn.dt, 1, "cayley-bcb")
                q, p = first.q.copy(), first.p.copy()
                with binding.acquire() as descriptor:
                    result = module.advance(context, descriptor, q, p, options, mask, records, snapshots)
                previous = options.get("previous")
                return (result[:4], q, p, records, snapshots, options["step_energies"],
                        previous if isinstance(previous, np.ndarray) else None)
            return run

        for reuse in (False, True):
            for energy in (False, True):
                extra = {"reuse_forces": reuse, "check_energy": energy}
                tag = f"P{beads} reuse={reuse} check_energy={energy}"
                ab.check(f"advance unbiased {tag}", block(None, coordinate, extra))
                ab.check(f"advance umbrella+flux {tag}", block(uf, coordinate, extra))
                ab.check(f"advance stack+wall {tag}", block(mixed, stack, extra, states=False))
                ab.check(f"advance other observation {tag}", block(uf, Distance((0, 5)), extra, states=False))
                ab.check(f"advance previous {tag}", block(uf, coordinate, dict(extra, previous=True), states=False))
        surface = coordinate.dividing_surface(0.999)
        parent = state.copy()
        parent.q[:] = project_positions(parent.q, surface, masses)
        project_momenta(parent.p, parent.q, surface, masses)
        ab.check(f"advance centroid RATTLE P{beads}",
                 block(None, surface, {"reuse_forces": True, "check_energy": True}, constraint=surface, first=parent))
        weights = masses[:5]
        fragment = [(np.arange(5), weights, weights.sum(),
                     np.average(state.q.mean(0)[:5], axis=0, weights=weights))]
        ab.check(f"advance fragments P{beads}", block(uf, coordinate, {"reuse_forces": True}, fragments=fragment))
        refresh = {"offsets": np.array([0, 7, 15], np.int64),
                   "momenta": np.stack([dyn.sample_momenta() for _ in range(3)]), "projection": "none"}
        ab.check(f"advance refresh P{beads}", block(uf, coordinate, {"reuse_forces": True, "refresh": refresh}, states=False))

        binding = descriptor_binding(resolve_native_binding(provider, 6))
        noise = np.random.default_rng(beads).normal(size=(17, beads, 6, 3))
        for fixed_atoms in ((), (1,)):
            for bias, cv in ((None, coordinate), (uf, coordinate), (mixed, Distance((0, 5)))):
                def sampling(module, bias=bias, cv=cv, fixed_atoms=fixed_atoms):
                    fixed = np.zeros(6, np.uint8);p0 = state.p.copy()
                    for atom in fixed_atoms:
                        fixed[atom] = 1;p0[:, atom] = 0
                    sampled = (np.arange(17) % 3 == 2).astype(np.uint8)
                    memo = {}
                    spec = {"beta": beta, "time": 0.0, "cvs": cv_descriptors(cv, 6, memo),
                            "biases": bias_descriptors(bias, dyn, memo), "record_energy": True,
                            "check_energy": True, "reuse_forces": True, "step_energies": np.full(17, np.nan),
                            "energy_states": np.full((17, 2, beads, 6, 3), np.nan)}
                    context = module.create(masses.tolist(), beta, beads, dyn.dt, 1, "cayley-bcb")
                    q, p = state.q.copy(), p0
                    records = np.full((int(sampled.sum()), 2 + 36 + 1), np.nan)
                    with binding.acquire() as descriptor:
                        result = module.advance_sampling(context, descriptor, q, p, noise, np.full(beads, .9),
                                                         np.full(beads, .05), fixed, sampled, records, spec)
                    return result, q, p, records, spec["step_energies"], spec["energy_states"]
                ab.check(f"advance_sampling P{beads} fixed={fixed_atoms} bias={bias is not None}", sampling)

        for bias in (None, uf):
            def whole(module, bias=bias):
                run = Dynamics(provider, masses, beta, beads, 0.1 * FS_TO_AU, native="required",
                               threads=1, loop_backend="native", seed=1)
                run.integrator._native = module
                run.integrator._context = module.create(masses.tolist(), beta, beads, run.dt, 1, "cayley-bcb")
                current = state.copy()
                result = run.propagate(current, 32, bias=bias, observe=coordinate, sample_interval=8, backend="native")
                return (current.q, current.p, result.energies, result.values, result.initial_energy,
                        result.minimum_energy, result.maximum_energy)
            ab.check(f"run_block P{beads} bias={bias is not None}", whole)

    report = {"schema": "pyrpmd.native-ab-parity.v1", "pes": "CBE/PES-2008" if args.cbe_library else "builtin harmonic",
              "comparisons": ab.count, "mismatches": ab.failures,
              "reference": str(args.reference), "candidate": candidate.__file__}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if ab.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
