#!/usr/bin/env python3
"""Independent source, electrostatic, platform, cell and timestep water checks.

Optional --author-source is the unmodified i-PI Fortran file named below. It
is compiled locally, never silently patched. Its parameters and Ewald accuracy
differ from the printed Table I; both comparisons are reported separately.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import Config, FixedWaterCache, atomic_json, atomic_npz, create_dynamics
from benchmarks.water.qtip4pf import PARAMETERS, initial_grid, make_system
from benchmarks.water.pilot import verify_cluster
from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM, FS_TO_AU, KCAL_MOL_TO_AU
from pyrpmd.dynamics import State
from pyrpmd.potentials.openmm import BOHR_NM, KJ_MOL_TO_AU, OpenMMPotential

AUTHOR_SOURCE = {
    "url": "https://raw.githubusercontent.com/i-pi/i-pi/93162d280d15fd954da6dc3580a5ed3503e207ac/drivers/f90/pes/qtip4pf.f90",
    "sha256": "911047fc1133662723e5d07342214af82e2b60f49de17ef2d0f4074519ec5297",
    "attribution": "Source header credits S. Habershon, D. Manolopoulos, T. Markland; dated 23 June 2014",
}
AUTHOR_PARAMETERS = {
    "epsilon_kcal_mol": 2.95147e-4 / KCAL_MOL_TO_AU,
    "sigma_angstrom": 5.96946 * BOHR_ANGSTROM,
    "q_M_e": 1.1128, "gamma": 0.73612,
    "D_r_kcal_mol": .185 / KCAL_MOL_TO_AU,
    "alpha_r_per_angstrom": 1.21 / BOHR_ANGSTROM,
    "r_eq_angstrom": 1.78 * BOHR_ANGSTROM,
    "k_theta_kcal_mol_rad2": .14 / KCAL_MOL_TO_AU,
    "theta_eq_degrees": 107.4,
}


def source_parameter_system(molecules, box, tolerance):
    """Explicit audit variant using the author source's rounded atomic values."""
    model = make_system(molecules=molecules, box_angstrom=box, ewald_tolerance=tolerance)
    bond, angle, nonbonded = model.getForces()
    for i, value in enumerate((.185 / KJ_MOL_TO_AU, 1.21 / BOHR_NM, 1.78 * BOHR_NM)):
        bond.setGlobalParameterDefaultValue(i, value)
    for i in range(angle.getNumAngles()):
        a, b, c, theta, _ = angle.getAngleParameters(i)
        angle.setAngleParameters(i, a, b, c, theta, .14 / KJ_MOL_TO_AU)
    for i in range(nonbonded.getNumParticles()):
        charge, _, _ = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, charge, 5.96946 * BOHR_NM,
                                       2.95147e-4 / KJ_MOL_TO_AU if i % 4 == 0 else 0)
    return model


def difference(energy, force, reference_energy, reference_force, molecules):
    delta = np.asarray(force) - np.asarray(reference_force)
    return {"energy_difference_hartree": float(np.asarray(energy) - np.asarray(reference_energy)),
            "energy_difference_kcal_mol_per_water": float((np.asarray(energy) - reference_energy) / (molecules * KCAL_MOL_TO_AU)),
            "force_max_error_hartree_per_bohr": float(np.max(abs(delta))),
            "force_rms_error_hartree_per_bohr": float(np.sqrt(np.mean(delta**2))),
            "force_relative_rms_error": float(np.linalg.norm(delta) / np.linalg.norm(reference_force))}


def author_calculator(source):
    source = Path(source).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != AUTHOR_SOURCE["sha256"]:
        raise ValueError("author Fortran source does not match the pinned SHA256")
    directory = tempfile.TemporaryDirectory(prefix="qtip4pf-oracle-")
    binary = Path(directory.name) / "oracle.so"
    subprocess.run(["gfortran", "-O2", "-shared", "-fPIC", str(source), "-o", str(binary)], check=True)
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)

    def evaluate(q, box, *, intramolecular=False):
        positions = np.array(q, dtype=float, order="F")
        force = np.zeros_like(positions, order="F")
        cell = np.full(3, box * ANGSTROM_TO_AU)
        count, energy = ctypes.c_int(len(positions)), ctypes.c_double()
        virial = np.zeros((3, 3), order="F")
        if intramolecular:
            library.qtip4pf_sr_(positions.ctypes.data_as(pointer), ctypes.byref(count),
                               force.ctypes.data_as(pointer), ctypes.byref(energy), virial.ctypes.data_as(pointer))
        else:
            library.qtip4pf_(cell.ctypes.data_as(pointer), positions.ctypes.data_as(pointer),
                            ctypes.byref(count), force.ctypes.data_as(pointer), ctypes.byref(energy),
                            virial.ctypes.data_as(pointer))
        return energy.value, force.copy()
    return directory, evaluate


def evaluate(model, q, box, platform="Reference", threads=1):
    n = len(q) // 3
    with OpenMMPotential(model, ["O", "H", "H"] * n,
                         cell=np.eye(3) * box * ANGSTROM_TO_AU, pbc=[True] * 3,
                         platform=platform, properties={"Threads": str(threads), "DeterministicForces": "true"} if platform == "CPU" else {}) as provider:
        provider.evaluate(q[None])
        start = time.perf_counter()
        result = provider.evaluate(q[None])
        elapsed = time.perf_counter() - start
        return float(result.energy[0]), result.forces[0].copy(), elapsed


def timestep_scan(q, box, *, beads=1, duration_fs=20, timesteps=(.5, .25, .125)):
    results = []
    initial = None
    for dt in timesteps:
        config = Config(beads=beads, dt_fs=dt, seed=927, threads=1)
        provider, dynamics = create_dynamics(config, box)
        try:
            if initial is None:
                initial = State(np.repeat(q[None], beads, axis=0), dynamics.sample_momenta())
            state = initial.copy()
            energies = [dynamics.energy(state) / beads]
            start = time.perf_counter()
            for _ in range(round(duration_fs / dt)):
                dynamics.step(state)
                energies.append(dynamics.energy(state) / beads)
            elapsed = time.perf_counter() - start
            delta = np.asarray(energies) - energies[0]
            endpoint = state.copy()
            state.p *= -1
            for _ in range(round(duration_fs / dt)):
                dynamics.step(state)
            results.append({"beads": beads, "dt_fs": dt, "duration_fs": duration_fs,
                            "max_energy_change_per_water_kcal_mol": float(np.max(abs(delta)) / (216 * KCAL_MOL_TO_AU)),
                            "rms_energy_change_per_water_kcal_mol": float(np.sqrt(np.mean(delta**2)) / (216 * KCAL_MOL_TO_AU)),
                            "reversibility_position_max_bohr": float(np.max(abs(state.q - initial.q))),
                            "reversibility_momentum_relative_norm": float(np.linalg.norm(state.p + initial.p) / np.linalg.norm(initial.p)),
                            "force_evaluation_seconds_per_step": elapsed / (len(energies) - 1),
                            "finite": bool(np.isfinite(endpoint.q).all() and np.isfinite(endpoint.p).all())})
        finally:
            provider.close()
    return results


def run(output, *, snapshot=None, author_source=None, dynamics=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    q, box = initial_grid(cells_per_axis=6)
    snapshot_hash = None
    if snapshot:
        snapshot_hash = hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()
        with np.load(snapshot, allow_pickle=False) as data:
            q, box = data["q"].mean(axis=0), float(data["box_angstrom"])
    if q.shape != (648, 3):
        raise ValueError("source-size numerical audit expects 216 OHH molecules")
    atomic_npz(output / "input.npz", q=q, box_angstrom=box)
    n = 216
    import openmm
    report = {"schema": "pyrpmd.water-numerics.v1", "molecules": n, "box_angstrom": box,
              "snapshot_sha256": snapshot_hash, "openmm_version": openmm.__version__,
              "platforms": [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())],
              "parameter_comparison": {k: {"table_I": v, "author_source_atomic_units_converted": AUTHOR_PARAMETERS[k],
                                             "relative_difference": (AUTHOR_PARAMETERS[k] - v) / v}
                                       for k, v in PARAMETERS.items()},
              "cluster_oracle": verify_cluster(), "electrostatics": [],
              "literature_reproduced": False, "physical_converged": False}
    reference = make_system(molecules=n, box_angstrom=box, ewald_tolerance=1e-9)
    reference.getForce(2).setNonbondedMethod(openmm.NonbondedForce.Ewald)
    e0, f0, seconds = evaluate(reference, q, box)
    report["reference"] = {"method": "direct Ewald", "platform": "Reference", "tolerance": 1e-9,
                           "energy_hartree": e0, "elapsed_evaluation_seconds": seconds}
    atomic_npz(output / "direct-ewald-oracle.npz", energy=e0, force=f0)
    for tolerance in (1e-4, 1e-5, 1e-6, 1e-7):
        values = {}
        for name in ("CPU", "Reference"):
            model = make_system(molecules=n, box_angstrom=box, ewald_tolerance=tolerance)
            e, f, seconds = evaluate(model, q, box, platform=name)
            values[name] = (e, f)
            report["electrostatics"].append({"method": "PME", "platform": name, "tolerance": tolerance,
                                            "evaluation_seconds": seconds, **difference(e, f, e0, f0, n)})
        e, f = values["CPU"]
        e_ref, f_ref = values["Reference"]
        report.setdefault("platform_comparison", []).append({"tolerance": tolerance, **difference(e, f, e_ref, f_ref, n)})
    # Constant-volume dispersion tails must change the energy only.
    model = make_system(molecules=n, box_angstrom=box, ewald_tolerance=1e-6, dispersion_correction=True)
    e_tail, f_tail, _ = evaluate(model, q, box)
    model = make_system(molecules=n, box_angstrom=box, ewald_tolerance=1e-6)
    e_plain, f_plain, _ = evaluate(model, q, box)
    report["dispersion_tail"] = difference(e_tail, f_tail, e_plain, f_plain, n)
    # Whole molecule images preserve virtual-site geometry.
    shifted = q.copy()
    shifted[:3] += [box * ANGSTROM_TO_AU, -box * ANGSTROM_TO_AU, 0]
    e_image, f_image, _ = evaluate(model, shifted, box)
    report["whole_molecule_image"] = difference(e_image, f_image, e_plain, f_plain, n)
    # Cell setter and independent context must describe the same Hamiltonian.
    cell = np.eye(3) * box * ANGSTROM_TO_AU
    with OpenMMPotential(model, ["O", "H", "H"] * n, cell=cell, pbc=[True]*3, platform="Reference") as provider:
        provider.set_cell(cell * 1.01)
        changed = provider.evaluate((q * 1.01)[None])
    changed_model = make_system(molecules=n, box_angstrom=box * 1.01, ewald_tolerance=1e-6)
    e_new, f_new, _ = evaluate(changed_model, q * 1.01, box * 1.01)
    report["cell_update"] = difference(changed.energy[0], changed.forces[0], e_new, f_new, n)
    if author_source:
        temporary, calculator = author_calculator(author_source)
        try:
            author_e, author_f = calculator(q, box)
            match_model = source_parameter_system(n, box, 1e-8)
            match_e, match_f, _ = evaluate(match_model, q, box)
            monomer = q[:3].copy()
            intra_e, intra_f = calculator(monomer, box, intramolecular=True)
            with OpenMMPotential(source_parameter_system(1, None, 1e-8), ["O", "H", "H"], platform="Reference") as provider:
                intra = provider.evaluate(monomer[None])
            report["author_fortran"] = {"source": AUTHOR_SOURCE,
                "printed_table_vs_unmodified_source": difference(e0, f0, author_e, author_f, n),
                "source_parameters_vs_unmodified_source": difference(match_e, match_f, author_e, author_f, n),
                "intramolecular_source_parameters": difference(intra.energy[0], intra.forces[0], intra_e, intra_f, 1),
                "limitations": ["Unmodified source uses an explicit low-precision erfc approximation and finite direct Ewald sums.",
                                "Printed Table I and source atomic-unit parameters are not numerically identical.",
                                "No bitwise or high-precision identity is expected; source virial was not used."]}
        finally:
            temporary.cleanup()
    if dynamics:
        report["timestep"] = timestep_scan(q, box)
        report["timestep"] += timestep_scan(q, box, beads=32, duration_fs=2, timesteps=(.25, .125))
    report["scope"] = "One source-size configuration and short real-mass trajectories; not ensemble/bead/cell-size convergence."
    report["analysis_source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "benchmarks/water/numerics.py", "benchmarks/water/qtip4pf.py", "benchmarks/water/campaign.py")}
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--author-source", type=Path)
    parser.add_argument("--skip-dynamics", action="store_true")
    args = parser.parse_args()
    report = run(args.output, snapshot=args.snapshot, author_source=args.author_source,
                 dynamics=not args.skip_dynamics)
    print(json.dumps({"electrostatic_cases": len(report["electrostatics"]),
                      "cluster_oracle_passed": report["cluster_oracle"]["passed"],
                      "physical_converged": False}, indent=2))


if __name__ == "__main__":
    main()
