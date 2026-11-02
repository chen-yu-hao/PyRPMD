"""Reproducible force-only T05 timing and accuracy comparisons.

Run from a source checkout with the optional engines installed::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
      python benchmarks/efficiency/force_engine_adapters.py --output timings.json

The water grid is deliberately unequilibrated: this measures evaluator cost,
not liquid-water convergence. Context/instance creation is reported separately
from warmed evaluation. All modes see exactly the same coordinate sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM


def legacy_provider(engine, current, ref):
    """Load the actual historical adapter, retaining normal package imports."""
    if ref is None:
        return current
    path = f"src/pyrpmd/potentials/{engine}.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)
    module = types.ModuleType(f"pyrpmd.potentials._benchmark_legacy_{engine}")
    module.__package__ = "pyrpmd.potentials"
    exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
    return getattr(module, current.__name__)


def environment():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENMM_CPU_THREADS")},
        "versions": {},
        "nvidia_smi": shutil.which("nvidia-smi"),
        "nvidia_device_nodes": [str(path) for path in Path("/dev").glob("nvidia*")],
    }
    if hasattr(os, "sched_getaffinity"):
        info["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    for package in ("numpy", "scipy", "pyrpmd", "openmm", "lammps", "mpi4py", "mpich"):
        try:
            info["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["versions"][package] = None
    for key, path in (("cpu_model", "/proc/cpuinfo"),
                      ("cpu_quota", "/sys/fs/cgroup/cpu.max"),
                      ("memory_limit_bytes", "/sys/fs/cgroup/memory.max")):
        try:
            value = Path(path).read_text()
            if key == "cpu_model":
                value = next(line.split(":", 1)[1].strip()
                             for line in value.splitlines()
                             if line.startswith("model name"))
            info[key] = value.strip()
        except (OSError, StopIteration):
            info[key] = "unavailable"
    try:
        import openmm
        info["openmm_platforms"] = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())]
    except ImportError:
        info["openmm_platforms"] = []
    return info


def coordinate_sequence(q, beads, count, seed):
    rng = np.random.default_rng(seed)
    initial = q[None] + rng.normal(scale=0.002, size=(beads, *q.shape))
    velocity = rng.normal(scale=0.0004, size=initial.shape)
    return [np.ascontiguousarray(initial + i * velocity) for i in range(count)]


def errors(actual, reference):
    report = {}
    for name in ("energy", "forces", "virial"):
        current = getattr(actual, name)
        expected = getattr(reference, name)
        if current is None and expected is None:
            continue
        delta = float(np.max(np.abs(current - expected)))
        scale = float(np.max(np.abs(expected)))
        report[f"{name}_max_abs_error"] = delta
        report[f"{name}_relative_linf_error"] = delta / max(scale, 1e-30)
    return report


def measure(make_provider, frames, warmup, reference=None):
    started = time.perf_counter()
    provider = make_provider()
    construction = time.perf_counter() - started
    records, results, maximum = [], [], {}
    try:
        first = time.perf_counter()
        provider.evaluate(frames[0])
        cold = time.perf_counter() - first
        for index, frame in enumerate(frames):
            started = time.perf_counter()
            result = provider.evaluate(frame)
            elapsed = time.perf_counter() - started
            if index >= warmup:
                records.append(elapsed)
            if reference is None:
                results.append(result)
            else:
                for key, value in errors(result, reference[index]).items():
                    maximum[key] = max(maximum.get(key, 0.0), value)
        record = {
            "construction_seconds": construction,
            "first_evaluation_seconds": cold,
            "evaluation_seconds": records,
            "median_seconds": statistics.median(records),
            "minimum_seconds": min(records),
            "p95_seconds": float(np.percentile(records, 95)),
            "error_against_serial": maximum,
            "provider_metadata": provider.metadata,
        }
        if hasattr(provider, "batch_stats"):
            record["batch_stats"] = dict(provider.batch_stats)
        return record, results
    finally:
        provider.close()


class _RPMDCopyProbe:
    """Benchmark-only copy-API control; never advances the integrator.

    The q-TIP4P/F factory has three-parent average virtual sites. Populate
    those explicitly, since copy getState does not recompute virtual sites.
    This deliberately is not a general production System adapter.
    """

    def __init__(self, constructor, options, beads):
        import openmm as mm
        from openmm import unit

        self.mm, self.unit = mm, unit
        self.context = self.integrator = None
        with constructor(**options) as template:
            self.system = template.system
            self.physical = template.particle_indices.copy()
            self.metadata = dict(template.metadata, benchmark_only=True,
                                 evaluation="RPMDIntegrator copy API without step", copies=beads)
            backend = template.context.getPlatform()
            properties = template.metadata["platform_properties"].copy()
        self.virtual = np.array([i for i in range(self.system.getNumParticles())
                                 if self.system.isVirtualSite(i)], dtype=int)
        sites = [self.system.getVirtualSite(int(i)) for i in self.virtual]
        if any(not isinstance(site, mm.ThreeParticleAverageSite) for site in sites):
            raise ValueError("RPMD copy probe only supports three-parent average virtual sites")
        self.parents = np.array([[site.getParticle(i) for i in range(3)] for site in sites])
        self.weights = np.array([[site.getWeight(i) for i in range(3)] for site in sites])
        self.beads = beads
        self.integrator = mm.RPMDIntegrator(beads, 300., 0., .001)
        self.integrator.setApplyThermostat(False)
        self.context = mm.Context(self.system, self.integrator, backend, properties)
        velocity = np.zeros((self.system.getNumParticles(), 3)) * unit.nanometer / unit.picosecond
        for copy in range(beads):
            self.integrator.setVelocities(copy, velocity)

    def evaluate(self, x):
        from pyrpmd.potentials.base import PESResult, validate_result
        from pyrpmd.potentials.openmm import BOHR_NM, KJ_MOL_TO_AU

        if len(x) != self.beads:
            raise ValueError("copy probe batch size must match its fixed copy count")
        positions = np.zeros((len(x), self.system.getNumParticles(), 3))
        positions[:, self.physical] = x * BOHR_NM
        if len(self.virtual):
            positions[:, self.virtual] = np.einsum(
                "bska,sk->bsa", positions[:, self.parents], self.weights)
        for copy in range(self.beads):
            self.integrator.setPositions(copy, positions[copy] * self.unit.nanometer)
        energies, forces = [], []
        for copy in range(self.beads):
            state = self.integrator.getState(copy, getEnergy=True, getForces=True)
            energies.append(state.getPotentialEnergy().value_in_unit(
                self.unit.kilojoule_per_mole) * KJ_MOL_TO_AU)
            raw = state.getForces(asNumpy=True).value_in_unit(
                self.unit.kilojoule_per_mole / self.unit.nanometer)
            forces.append(np.asarray(raw)[self.physical] * KJ_MOL_TO_AU * BOHR_NM)
        return validate_result(PESResult(energies, forces), x.shape)

    def close(self):
        self.context = None
        self.integrator = None


def water_case(cells, args):
    from benchmarks.water.qtip4pf import initial_grid, make_system
    from pyrpmd.potentials.openmm import OpenMMPotential

    q, side = initial_grid(cells_per_axis=cells, seed=args.seed)
    molecules = len(q) // 3
    cutoff = min(9.0, 0.45 * side)
    model = make_system(molecules=molecules, box_angstrom=side,
                        cutoff_angstrom=cutoff, ewald_tolerance=1e-6)
    LegacyOpenMM = legacy_provider("openmm", OpenMMPotential, args.legacy_ref)
    options = dict(model=model, symbols=["O", "H", "H"] * molecules,
                   cell=np.eye(3) * side * ANGSTROM_TO_AU, pbc=[True] * 3,
                   device=args.device)
    if args.device == "cpu":
        options.update(platform="CPU", properties={"Threads": str(args.cpu_threads)})
    frames = coordinate_sequence(q, args.beads, args.warmup + args.repeats, args.seed)
    baseline, reference = measure(lambda: LegacyOpenMM(**options), frames, args.warmup)
    modes = {"serial": baseline}
    print(f"OpenMM N={len(q)} serial Threads={args.cpu_threads}: "
          f"{baseline['median_seconds']:.6f} s/batch", flush=True)
    if args.device == "cpu":
        for threads in args.serial_cpu_threads:
            if threads == args.cpu_threads:
                continue
            threaded_options = dict(options, properties={"Threads": str(threads)})
            record, _ = measure(lambda: LegacyOpenMM(**threaded_options),
                                frames, args.warmup, reference)
            record["speedup_over_serial"] = baseline["median_seconds"] / record["median_seconds"]
            modes[f"serial_threads_{threads}"] = record
            print(f"OpenMM N={len(q)} serial Threads={threads}: "
                  f"{record['median_seconds']:.6f} s/batch", flush=True)
    for workers in args.workers:
        record, _ = measure(
            lambda: OpenMMPotential(**options, batch_mode="contexts", max_workers=workers),
            frames, args.warmup, reference)
        record["speedup_over_serial"] = baseline["median_seconds"] / record["median_seconds"]
        modes[f"contexts_workers_{workers}"] = record
        print(f"OpenMM N={len(q)} Context workers={workers}: "
              f"{record['median_seconds']:.6f} s/batch", flush=True)
    if args.rpmd_copy:
        record, _ = measure(lambda: _RPMDCopyProbe(LegacyOpenMM, options, args.beads),
                            frames, args.warmup, reference)
        record["speedup_over_serial"] = baseline["median_seconds"] / record["median_seconds"]
        modes["rpmd_copy_probe"] = record
        print(f"OpenMM N={len(q)} RPMD copy probe: "
              f"{record['median_seconds']:.6f} s/batch", flush=True)
    return {
        "engine": "openmm", "model": "q-TIP4P/F", "physical_atoms": len(q),
        "engine_particles_per_copy": model.getNumParticles(), "molecules": molecules,
        "beads": args.beads, "cell_angstrom": side, "cutoff_angstrom": cutoff,
        "ewald_tolerance": 1e-6, "device": args.device,
        "cpu_threads_per_context": args.cpu_threads if args.device == "cpu" else None,
        "modes": modes,
    }


def lj_case(cells, args):
    from pyrpmd.potentials.lammps import LAMMPSPotential

    LegacyLAMMPS = legacy_provider("lammps", LAMMPSPotential, args.legacy_ref)
    lattice = 5.26
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    indices = np.stack(np.meshgrid(*([np.arange(cells)] * 3), indexing="ij"), axis=-1)
    q = ((indices.reshape(-1, 1, 3) + basis[None]) * lattice).reshape(-1, 3)
    q *= ANGSTROM_TO_AU
    side = lattice * cells
    # Explicit atom IDs reproduce this exact q order independently of LAMMPS'
    # create_atoms/lattice ordering or processor decomposition.
    with tempfile.TemporaryDirectory(prefix="pyrpmd-lj-benchmark-") as directory:
        directory = Path(directory)
        data = directory / "lj.data"
        atoms = "\n".join(f"{i+1} 1 {r[0]:.17g} {r[1]:.17g} {r[2]:.17g}"
                          for i, r in enumerate(q * BOHR_ANGSTROM))
        data.write_text(
            f"LJ benchmark\n\n{len(q)} atoms\n1 atom types\n\n"
            f"0 {side:.17g} xlo xhi\n0 {side:.17g} ylo yhi\n0 {side:.17g} zlo zhi\n\n"
            f"Masses\n\n1 39.948\n\nAtoms # atomic\n\n{atoms}\n")
        script = directory / "lj.in"
        script.write_text(
            'units metal\natom_style atomic\nboundary p p p\n'
            'atom_modify map array\nread_data ${pyrpmd_model_dir}/lj.data\n'
            'pair_style lj/cut 8.5\npair_coeff 1 1 0.0103 3.4\n'
            'neighbor 2.0 bin\n')
        options = dict(model=script, symbols=["Ar"] * len(q), type_map=["Ar"],
                       cell=np.eye(3) * side * ANGSTROM_TO_AU, pbc=[True] * 3,
                       asset_files=[data])
        frames = coordinate_sequence(q, args.beads, args.warmup + args.repeats, args.seed)
        baseline, reference = measure(lambda: LegacyLAMMPS(**options), frames, args.warmup)
        modes = {"serial": baseline}
        if args.legacy_ref is not None:
            serial_current, _ = measure(lambda: LAMMPSPotential(**options),
                                        frames, args.warmup, reference)
            modes["serial_current"] = serial_current
        persistent, _ = measure(
            lambda: LAMMPSPotential(**options, batch_mode="persistent",
                                    neighbor_skin=2.0 * ANGSTROM_TO_AU),
            frames, args.warmup, reference)
        persistent["speedup_over_serial"] = baseline["median_seconds"] / persistent["median_seconds"]
        # Independent analytic LJ checks live in the provider tests; this is a
        # many-atom, changing-coordinate cross-mode comparison.
        modes["persistent"] = persistent
        for name, record in modes.items():
            if name != "serial" and max(
                    record["error_against_serial"][f"{key}_relative_linf_error"]
                    for key in ("energy", "forces", "virial")) > 1e-10:
                raise AssertionError(f"LAMMPS {name} result exceeds 1e-10 relative error")
        return {
            "engine": "lammps", "model": "Lennard-Jones FCC", "physical_atoms": len(q),
            "beads": args.beads, "cell_angstrom": side, "cutoff_angstrom": 8.5,
            "neighbor_skin_angstrom": 2.0, "epsilon_ev": .0103, "sigma_angstrom": 3.4,
            "modes": modes,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("all", "openmm", "lammps"), default="all")
    parser.add_argument("--water-cells", nargs="+", type=int, default=[4, 9])
    parser.add_argument("--lj-cells", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--serial-cpu-threads", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--legacy-ref", help="compare against adapters from this local git commit/ref")
    parser.add_argument("--rpmd-copy", action="store_true",
                        help="also measure the force-only RPMD copy API on the water model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.beads, args.cpu_threads, args.warmup, args.repeats,
           *args.workers, *args.serial_cpu_threads, *args.water_cells, *args.lj_cells) <= 0:
        parser.error("sizes, thread counts, warmup and repeats must be positive")
    report = {
        "schema": "pyrpmd.force-engine-benchmark.v1", "environment": environment(),
        "warmup_evaluations": args.warmup, "timed_evaluations": args.repeats,
        "seed": args.seed, "timing_unit": "seconds per complete batch evaluation",
        "physical_converged": False, "complete": False, "cases": [],
        "legacy_ref": args.legacy_ref,
        "rpmd_copy_probe": args.rpmd_copy,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(),
                                       ROOT / "src/pyrpmd/potentials/openmm.py",
                                       ROOT / "src/pyrpmd/potentials/lammps.py",
                                       ROOT / "benchmarks/water/qtip4pf.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    if args.engine in ("all", "openmm"):
        tasks.extend((water_case, cells) for cells in args.water_cells)
    if args.engine in ("all", "lammps"):
        tasks.extend((lj_case, cells) for cells in args.lj_cells)
    report["planned_cases"] = len(tasks)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for function, cells in tasks:
        case = function(cells, args)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"engine": case["engine"], "atoms": case["physical_atoms"],
                          "median_seconds": {k: v["median_seconds"]
                                             for k, v in case["modes"].items()}}), flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
