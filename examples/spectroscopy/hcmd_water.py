"""Train and run an explicit intramolecular h-CMD molecular partition.

This example reuses fqcmd_water.py's illustrative noninteracting quartic water
surface. It does not introduce another f-QCMD implementation or claim liquid
spectral accuracy. Molecules in the first explicit partition use bead-mean
internal-coordinate targets; the others use Cartesian-centroid targets. Every
IBI iteration samples their combined corrected potential.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
      examples/spectroscopy/hcmd_water.py --output runs/hcmd-water

The saved hcmd-input.py can subsequently be passed to ``pyrpmd validate``,
``pyrpmd run`` and the usual spectroscopy resume/reanalysis commands. The
default is a short research demonstration; physical_converged remains false.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import FS_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import atomic_write_text, file_sha256
from pyrpmd.potentials.base import check_potential
from pyrpmd.potentials.python import import_object
from pyrpmd.qcmd import (
    FQCMDModel, HCMDModel, HCMDPotential, TriatomicQuasiCentroid,
    fqcmd_base_identity, hcmd_coordinate_histograms, update_hcmd_model,
)
from pyrpmd.spectroscopy_input import resolve_spectroscopy_job
from pyrpmd.workflows.spectroscopy import run_spectroscopy


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(text):
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, default=Path("runs/hcmd-water"))
    result.add_argument("--molecules", type=_positive_int, default=8)
    result.add_argument("--fqcmd-molecules", type=_positive_int, default=4)
    result.add_argument("--beads", type=_positive_int, default=16)
    result.add_argument("--temperature", type=_positive_float, default=300.)
    result.add_argument("--dt-fs", type=_positive_float, default=.1)
    result.add_argument("--seed", type=_positive_int, default=1729)
    result.add_argument("--warmup", type=_positive_int, default=1000)
    result.add_argument("--sample-steps", type=_positive_int, default=4000)
    result.add_argument("--sample-every", type=_positive_int, default=10)
    result.add_argument("--ibi-iterations", type=_positive_int, default=3)
    result.add_argument("--bins", type=_positive_int, default=80)
    result.add_argument("--production-steps", type=_positive_int, default=10000)
    result.add_argument("--train-only", action="store_true")
    return result


def _sample_counts(potential, model, options, *, beads, seed, start=None):
    dynamics = Dynamics(potential, model.geometry.masses, model.beta, beads,
                        options.dt_fs * FS_TO_AU, seed=seed, native=False,
                        loop_backend="python", integrator="normal-mode-bab")
    if start is None:
        raise ValueError("an explicit initial geometry is required")
    q = np.broadcast_to(start, (beads, *start.shape[-2:])).copy()
    state = State(q, dynamics.sample_momenta())
    counts = [np.zeros(len(edge) - 1, dtype=np.int64) for edge in model.edges]
    started = time.perf_counter()
    for step in range(options.warmup + options.sample_steps):
        dynamics.sample_step(state, gamma_centroid=.05 / FS_TO_AU)
        sample = step + 1 - options.warmup
        if sample > 0 and sample % options.sample_every == 0:
            observed = hcmd_coordinate_histograms(
                state.q[None] if beads > 1 else state.q, model, bead_average=beads > 1)
            for total, incoming in zip(counts, observed):
                total += incoming
    return tuple(counts), state.q.copy(), time.perf_counter() - started


def run(options):
    if options.fqcmd_molecules >= options.molecules:
        raise ValueError("both explicit molecular partitions must be nonempty")
    if options.sample_steps < options.sample_every:
        raise ValueError("sample_steps must save at least one training frame")
    if options.bins < 4:
        raise ValueError("at least four bins are required for each spline")
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("fqcmd_water.py").resolve()
    base = import_object(str(source) + ":build_potential")()
    raw = {
        "task": "spectroscopy", "beads": 1, "temperature": options.temperature,
        "seed": options.seed + 20000,
        "system": {"symbols": ["O", "H", "H"] * options.molecules,
                   "geometry_unit": "bohr", "geometry": base.equilibrium(options.molecules).tolist()},
        "potential": {"provider": "python", "factory": str(source) + ":build_potential"},
        "properties": {"provider": "python", "factory": str(source) + ":build_properties"},
        "dynamics": {"timestep": f"{options.dt_fs} fs", "integrator": "normal-mode-bab"},
        "sampling": {"equilibration_steps": options.warmup,
                     "production_steps": options.production_steps, "sample_interval": 2,
                     "replicas": 2, "gamma_centroid": "0.05 fs^-1"},
        "spectroscopy": {"method": "h-cmd", "hcmd_model_file": "hcmd-model.npz", "spectra": ["ir"]},
        "output": {"directory": "production", "checkpoint_steps": 100},
    }
    config, system = resolve_spectroscopy_job(raw, base_directory=options.output)
    geometry = TriatomicQuasiCentroid(np.arange(system.natoms).reshape(-1, 3), system.masses)
    partitions = {"f-qcmd": list(range(options.fqcmd_molecules)),
                  "f-cmd": list(range(options.fqcmd_molecules, options.molecules))}
    groups = ([[0, 0, 1]] * options.fqcmd_molecules
              + [[2, 2, 3]] * (options.molecules - options.fqcmd_molecules))
    grids = [np.linspace(.5, 3.5, options.bins + 1), np.linspace(.15, np.pi - .15, options.bins + 1)]
    edges = grids + grids
    scientific_settings = {key: value for key, value in vars(options).items() if key not in ("output", "train_only")}
    provenance = {"training": "joint h-CMD intramolecular IBI from common unconstrained PIMD",
                  "settings": scientific_settings, "source_sha256": file_sha256(source),
                  "training_script_sha256": file_sha256(Path(__file__)),
                  "thermostat": "PILE-L; gamma_centroid=0.05 fs^-1",
                  "ibi_damping": .5, "ibi_regularization": .01, "ibi_polynomial_degree": 2,
                  "physical_converged": False}
    correction = FQCMDModel(geometry, groups, edges, [np.zeros(len(edge) - 1) for edge in edges],
                            beta=config["beta"], base_identity=fqcmd_base_identity(config["potential"], system),
                            provenance=provenance)
    model = HCMDModel(correction, partitions)
    target, _, target_seconds = _sample_counts(base, model, options, beads=options.beads,
                                               seed=options.seed, start=system.geometry)
    print(f"h-CMD target: partitions={partitions}; P={options.beads}; {target_seconds:.2f}s", flush=True)
    collected = {f"target_{i}": value for i, value in enumerate(target)}
    records = []
    start = system.geometry
    for iteration in range(options.ibi_iterations + 1):
        seed = options.seed + 10000 + iteration
        sampled, final, seconds = _sample_counts(HCMDPotential(base, model), model, options, beads=1,
                                                  seed=seed, start=start)
        start = final[0]
        error = [float(.5 * np.sum(np.abs(a / a.sum() - b / b.sum()))) for a, b in zip(target, sampled)]
        record = {"iteration": iteration, "seed": seed, "sampling_wall_seconds": seconds,
                  "marginal_total_variation": error, "sampled_model_sha256": model.fingerprint}
        collected.update({f"iteration_{iteration}_sampled_{i}": value for i, value in enumerate(sampled)})
        if iteration < options.ibi_iterations:
            model, diagnostics = update_hcmd_model(model, target, sampled, damping=.5,
                                                    regularization=.01, polynomial_degree=2)
            record["update_diagnostics"] = diagnostics
        records.append(record)
        print(f"h-CMD IBI {iteration}: marginal TV={error}; {seconds:.2f}s", flush=True)
    model.save(options.output / "hcmd-model.npz")
    atomic_write_text(options.output / "hcmd-input.py", "def build():\n    return " + repr(raw) + "\n")
    atomic_savez(options.output / "training-distributions.npz", kind="PyRPMD.hcmd-water-training.v1",
                 **collected, **{f"edges_{i}": edge for i, edge in enumerate(model.edges)})
    report = {"example": "illustrative-quartic-water-hcmd", "settings": scientific_settings,
              "partitions": partitions, "model_fingerprint": model.fingerprint,
              "model_file_sha256": file_sha256(options.output / "hcmd-model.npz"),
              "target_wall_seconds": target_seconds, "ibi": records,
              "force_check": check_potential(HCMDPotential(base, model), system.geometry[None]),
              "physical_converged": False, "converged": False,
              "limitations": ["Short illustrative run with no quantum spectral reference.",
                              "No intermolecular correction; nonperiodic bent triatomics only.",
                              "PIMD bead/time-step/sampling and IBI basis/regularization remain unconverged.",
                              "Production evaluates an effective geometry property surface, not a conditional average."]}
    if not options.train_only:
        report["production"] = run_spectroscopy(config, system, options.output / "production")
    atomic_write_text(options.output / "report.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    base.close()
    return report


if __name__ == "__main__":
    run(parser().parse_args())
