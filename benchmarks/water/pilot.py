#!/usr/bin/env python3
"""Run a small q-TIP4P/F/PME RPMD pilot; never labels it a liquid reproduction.

From the checkout root:
  python benchmarks/water/pilot.py --output runs/water-pilot
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.qtip4pf import (
    H_MASS_DA, O_MASS_DA, PARAMETERS, SOURCE, cluster_energy,
    dipoles, initial_grid, make_system,
)
from pyrpmd.analysis.diffusion import analyze_diffusion
from pyrpmd.analysis.spectroscopy import relative_ir_spectrum
from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import write_json
from pyrpmd.potentials.openmm import OpenMMPotential


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_identity():
    files = ["benchmarks/water/qtip4pf.py", "benchmarks/water/pilot.py",
             "src/pyrpmd/potentials/openmm.py", "src/pyrpmd/potentials/base.py",
             "src/pyrpmd/dynamics.py", "src/pyrpmd/constants.py", "src/pyrpmd/geometry.py",
             "src/pyrpmd/analysis/diffusion.py", "src/pyrpmd/analysis/spectroscopy.py"]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                             stderr=subprocess.DEVNULL, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"base_revision": revision, "working_tree_dirty": dirty,
            "sha256": {f: _hash(ROOT / f) for f in files}}


def verify_cluster():
    """Force checks differentiate independent NumPy equations, not OpenMM."""
    grid, _ = initial_grid()
    q = grid[:6][None].copy()
    q[:, 1, 0] += .07
    with OpenMMPotential(make_system(molecules=2), ["O", "H", "H"] * 2,
                         platform="Reference") as provider:
        result = provider.evaluate(q)
        fd = np.empty_like(q)
        for atom in range(6):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[:, atom, axis] += 2e-5
                minus[:, atom, axis] -= 2e-5
                fd[:, atom, axis] = -(cluster_energy(plus) - cluster_energy(minus)) / 4e-5
        energy_error = float(np.max(abs(result.energy - cluster_energy(q))))
        force_error = float(np.max(abs(result.forces - fd)))
    return {"oracle": "independent NumPy Eqs. (1)-(4); finite-difference all 18 coordinates",
            "max_energy_error_hartree": energy_error,
            "max_force_error_hartree_per_bohr": force_error,
            "energy_tolerance_hartree": 2e-10,
            "force_tolerance_hartree_per_bohr": 2e-9,
            "passed": energy_error < 2e-10 and force_error < 2e-9}


def run(output, *, beads=4, equilibration_steps=64, production_steps=256, seed=541):
    for key, value, minimum in (("beads", beads, 1), ("equilibration_steps", equilibration_steps, 1),
                                ("production_steps", production_steps, 16)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_at_start = _source_identity()
    verification = verify_cluster()
    if not verification["passed"]:
        raise AssertionError(f"q-TIP4P/F force oracle failed: {verification}")
    q0, box = initial_grid(seed=seed)
    cells = np.eye(3) * box * ANGSTROM_TO_AU
    molecule_mass = np.array([O_MASS_DA, H_MASS_DA, H_MASS_DA]) * AMU_TO_AU
    masses = np.tile(molecule_mass, 8)
    beta, dt = 1 / (KB_AU * 298), .05 * FS_TO_AU
    model = make_system(molecules=8, box_angstrom=box, cutoff_angstrom=2.5,
                        ewald_tolerance=1e-6)
    from openmm import XmlSerializer
    (output / "model.xml").write_text(XmlSerializer.serialize(model), encoding="utf-8")
    positions, velocities, mu, energy = [], [], [], []
    with OpenMMPotential(model, ["O", "H", "H"] * 8, cell=cells,
                         pbc=[True] * 3, platform="Reference") as provider:
        provider_metadata = dict(provider.metadata)
        dynamics = Dynamics(provider, masses, beta, beads, dt, seed=seed, native=False,
                            loop_backend="python", integrator="normal-mode-bab")
        state = State(np.repeat(q0[None], beads, axis=0), dynamics.sample_momenta())
        for _ in range(equilibration_steps):
            dynamics.sample_step(state, gamma_centroid=1 / (10 * FS_TO_AU))
        # Remove the total centroid COM motion once; the NVE map conserves it.
        com_velocity = state.p.mean(axis=0).sum(axis=0) / masses.sum()
        state.p -= masses[None, :, None] * com_velocity
        for step in range(production_steps + 1):
            if step:
                dynamics.step(state)
            centroid = state.q.mean(axis=0).reshape(8, 3, 3)
            centroid_v = (state.p.mean(axis=0) / masses[:, None]).reshape(8, 3, 3)
            weights = molecule_mass[:, None] / molecule_mass.sum()
            positions.append((centroid * weights).sum(axis=1))
            velocities.append((centroid_v * weights).sum(axis=1))
            mu.append(dipoles(state.q))
            energy.append(dynamics.energy(state))
    positions, velocities, mu, energy = map(np.asarray, (positions, velocities, mu, energy))
    times = np.arange(production_steps + 1) * dt
    trajectory = output / "trajectory.npz"
    np.savez_compressed(trajectory, times_au=times, molecular_centroid_positions_bohr=positions,
                        molecular_centroid_velocities_au=velocities, bead_dipoles_e_bohr=mu,
                        ring_polymer_hamiltonian_hartree=energy, cell_bohr=cells,
                        initial_geometry_bohr=q0, final_q_bohr=state.q, final_p_au=state.p)
    diffusion = analyze_diffusion(positions, times, velocities=velocities, bootstrap=0)
    spectrum = relative_ir_spectrum(mu, dt, beta=beta, normalize=False, window="hann")
    write_json(output / "diffusion-diagnostic.json", diffusion)
    np.savez_compressed(output / "spectrum-diagnostic.npz",
                        wavenumber_cm1=spectrum["wavenumber_cm1"],
                        relative_ir=spectrum["relative_ir"])
    energy_drift = float(np.max(abs(energy - energy[0])) / beads)
    finite = bool(all(np.isfinite(x).all() for x in (positions, velocities, mu, energy)))
    source_at_end = _source_identity()
    report = {
        "schema": "pyrpmd.qtip4pf-pilot.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "model/PME/virtual-site/integrator/observable software validation",
        "physical_converged": False, "literature_reproduced": False,
        "liquid_diffusion_angstrom2_ps": None, "liquid_ir_validated": False,
        "model_source": SOURCE, "parameters": PARAMETERS,
        "conditions": {"temperature_K": 298, "molecules": 8, "density_g_cm3": .997,
                       "box_angstrom": box, "beads": beads, "timestep_fs": .05,
                       "equilibration_steps": equilibration_steps,
                       "production_steps": production_steps, "production_duration_fs": production_steps * .05,
                       "cutoff_angstrom": 2.5, "electrostatics": "PME", "ewald_tolerance": 1e-6,
                       "dispersion_correction": False, "independent_replicas": 1, "seed": seed,
                       "sampling": "PILE-L", "production": "NVE RPMD", "constraints": []},
        "provider": provider_metadata, "cluster_verification": verification,
        "finite": finite, "max_energy_drift_per_bead_hartree": energy_drift,
        "energy_drift_tolerance_per_bead_hartree": 1e-5,
        "software_checks_passed": bool(finite and verification["passed"] and energy_drift < 1e-5),
        "diffusion_diagnostic_status": diffusion["status"],
        "limitations": [
            "Unrelaxed eight-molecule starting grid; short thermostat segment does not establish equilibrium liquid.",
            "Tiny cell and 2.5-A cutoff deliberately differ from the paper's 216 molecules and 9-A cutoff.",
            "One replica and a few femtoseconds cannot establish a diffusive regime or resolved liquid IR spectrum.",
            "Saved IR is an RPMD diagnostic; the original paper's liquid IR uses PA-CMD, a different method.",
            "No literature agreement, timestep/bead/electrostatic convergence or GPU performance is claimed."],
        "source_at_start": source_at_start, "source_at_end": source_at_end,
        "source_hashes_unchanged_during_run": source_at_start["sha256"] == source_at_end["sha256"],
        "reproduce": "python benchmarks/water/pilot.py --output NEW_DIRECTORY "
                     f"--beads {beads} --equilibration-steps {equilibration_steps} "
                     f"--production-steps {production_steps} --seed {seed}",
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "openmm")},
        "artifacts": {p.name: {"sha256": _hash(p), "bytes": p.stat().st_size}
                      for p in output.iterdir() if p.is_file()},
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--equilibration-steps", type=int, default=64)
    parser.add_argument("--production-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=541)
    args = parser.parse_args()
    report = run(args.output, beads=args.beads, equilibration_steps=args.equilibration_steps,
                 production_steps=args.production_steps, seed=args.seed)
    print(json.dumps({k: report[k] for k in ("software_checks_passed", "physical_converged",
          "max_energy_drift_per_bead_hartree", "diffusion_diagnostic_status", "elapsed_seconds")}, indent=2))
    raise SystemExit(0 if report["software_checks_passed"] else 1)


if __name__ == "__main__":
    main()
