"""Opt-in, source-pinned real Ni-H EAM integration pilot.

Run from a source checkout with ``python benchmarks/materials/ni_h_eam.py
--model /path/NiAlH_jea.eam.alloy --output /new/path``.  Add ``--download``
to fetch the pinned model if absent.  No potential table is redistributed.
This small slab/short trajectory is a workflow validation, not a measured or
converged diffusion coefficient and not a reproduction of Suleimanov (2012).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

import numpy as np


MODEL_COMMIT = "5f0288a471841037872e673f93abe6af41490a6a"
MODEL_FILENAME = "NiAlH_jea.eam.alloy"
MODEL_URL = f"https://raw.githubusercontent.com/lammps/lammps/{MODEL_COMMIT}/potentials/{MODEL_FILENAME}"
MODEL_SHA256 = "545b0f79207ae519b7f355451812acd999bd6ed095f40a6bcfa9aa5258a38388"
MODEL_BYTES = 291035
NVE_ENERGY_TOLERANCE_PER_BEAD = 1e-4  # Hartree, finite-step integration gate only.
PROVENANCE = {
    "model": "Angelo-Moody-Baskes Ni-Al-H EAM; only Ni and H species used",
    "filename": MODEL_FILENAME,
    "repository": "https://github.com/lammps/lammps",
    "commit": MODEL_COMMIT,
    "url": MODEL_URL,
    "sha256": MODEL_SHA256,
    "bytes": MODEL_BYTES,
    "citation": "J. E. Angelo, N. R. Moody and M. I. Baskes, Trapping of hydrogen to lattice defects in nickel, Modelling Simul. Mater. Sci. Eng. 3, 289-307 (1995).",
    "doi": "10.1088/0965-0393/3/3/001",
    "registry": "https://www.ctcms.nist.gov/potentials/entry/1995--Angelo-J-E-Moody-N-R-Baskes-M-I--Ni-Al-H/",
    "format_documentation": "https://docs.lammps.org/pair_eam.html",
    "units": "DYNAMO setfl, LAMMPS metal units (angstrom, eV)",
    "redistributed": False,
}


def obtain_model(path, *, download=False):
    """Validate existing bytes, or explicitly opt into an integrity-pinned fetch."""
    path = Path(path).resolve()
    if path.is_file():
        data = path.read_bytes()
    elif download:
        with urlopen(MODEL_URL, timeout=30) as response:
            # A bounded read prevents replacing a model with an arbitrary response.
            data = response.read(MODEL_BYTES + 1)
    else:
        raise FileNotFoundError(f"No model at {path}; supply the pinned file or explicitly pass --download")
    if len(data) != MODEL_BYTES or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Ni-H potential bytes do not match the pinned SHA256/length")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation leaves any concurrently supplied file untouched.
        with path.open("xb") as handle:
            handle.write(data)
    return path


def make_slab():
    """27 movable Ni atoms, one H at a Ni(100) hollow; 1/9 surface coverage."""
    from ase.build import fcc100, add_adsorbate

    atoms = fcc100("Ni", size=(3, 3, 3), a=3.52, vacuum=8.0, periodic=False)
    add_adsorbate(atoms, "H", 0.9, "hollow")
    # No frozen layers: the entire small slab is part of the simulated system.
    return atoms


def job(model, *, isotope="H", beads=1, quick=False, seed=7201):
    atoms = make_slab()
    symbols = atoms.get_chemical_symbols()
    symbols[-1] = isotope
    return {
        "task": "diffusion", "preset": "pilot", "temperature": "600 K",
        "beads": beads, "seed": seed,
        "system": {"symbols": symbols, "geometry": atoms.positions.tolist(),
                   "geometry_unit": "angstrom", "cell": atoms.cell[:].tolist(),
                   "cell_unit": "angstrom", "pbc": atoms.pbc.tolist()},
        "potential": {"provider": "ase", "factory": "ase.calculators.eam:EAM",
                      "params": {"potential": str(Path(model).resolve())}},
        "dynamics": {"timestep": "0.1 fs", "integrator": "cayley-bcb", "loop_backend": "auto"},
        "sampling": {"thermostat": "pile-l", "gamma_centroid": "0.1 fs^-1",
                     "equilibration_steps": 4 if quick else 64,
                     "production_steps": 16 if quick else 160,
                     "sample_interval": 2, "replicas": 1},
        "diffusion": {"atom_indices": [len(atoms) - 1], "axes": [0, 1],
                      "reference_atoms": list(range(len(atoms) - 1)), "bootstrap": 0},
        "parallel": {"backend": "serial"},
        "output": {"checkpoint_steps": 32},
    }


def verify_adapter(model):
    """Compare independent direct-ASE calls, cell translations and force FD."""
    from ase.calculators.eam import EAM
    from pyrpmd.constants import ANGSTROM_TO_AU, EV_TO_AU, BOHR_ANGSTROM
    from pyrpmd.input import JobSpec
    from pyrpmd.potentials import make_potential

    config, system = JobSpec.from_mapping(job(model)).resolve()
    adapter = make_potential(config["potential"], system)
    atoms = make_slab()
    atoms.positions[-1] += [0.13, -0.07, 0.04]
    atoms.calc = EAM(potential=str(model))
    q = atoms.positions * ANGSTROM_TO_AU
    direct_energy = atoms.get_potential_energy()
    direct_force = atoms.get_forces()
    expected_force = direct_force * EV_TO_AU * BOHR_ANGSTROM
    result = adapter.evaluate(q[None])
    energy_error = float(abs(result.energy[0] - direct_energy * EV_TO_AU))
    force_error = float(np.max(np.abs(result.forces[0] - expected_force)))
    translated = q.copy()
    translated[-1] += system.cell[0] - 2 * system.cell[1]
    image = adapter.evaluate(translated[None])
    translation_energy_error = float(abs(result.energy[0] - image.energy[0]))
    translation_force_error = float(np.max(np.abs(result.forces - image.forces)))
    numeric = []
    original = atoms.positions.copy()
    epsilon = 1e-5  # angstrom; independently differentiate ASE's energy.
    for atom, axis in ((len(atoms) - 1, 0), (len(atoms) - 1, 2), (len(atoms) - 2, 2)):
        atoms.positions[:] = original
        atoms.positions[atom, axis] += epsilon
        plus = atoms.get_potential_energy()
        atoms.positions[atom, axis] -= 2 * epsilon
        minus = atoms.get_potential_energy()
        finite_difference = -(plus - minus) / (2 * epsilon)
        numeric.append({"atom": atom, "axis": axis, "force_eV_per_angstrom": float(direct_force[atom, axis]),
                        "finite_difference_eV_per_angstrom": float(finite_difference),
                        "absolute_error_eV_per_angstrom": float(abs(direct_force[atom, axis] - finite_difference))})
    adapter.close()
    maximum_fd_error = max(value["absolute_error_eV_per_angstrom"] for value in numeric)
    passed = (energy_error < 1e-10 and force_error < 1e-10 and
              translation_energy_error < 1e-10 and translation_force_error < 1e-10 and
              maximum_fd_error < 1e-5)
    return {"passed": passed, "natoms": len(atoms), "cell_angstrom": atoms.cell[:].tolist(),
            "pbc": atoms.pbc.tolist(), "direct_energy_eV": float(direct_energy),
            "energy_error_hartree": energy_error, "force_error_hartree_per_bohr": force_error,
            "translation_energy_error_hartree": translation_energy_error,
            "translation_force_error_hartree_per_bohr": translation_force_error,
            "force_finite_differences": numeric}


def _source_identity():
    import pyrpmd
    root = Path(__file__).resolve().parents[2]
    files = ["src/pyrpmd/system.py", "src/pyrpmd/geometry.py", "src/pyrpmd/input.py",
             "src/pyrpmd/runner.py", "src/pyrpmd/constants.py", "src/pyrpmd/units.py",
             "src/pyrpmd/dynamics.py", "src/pyrpmd/propagation.py", "native/cayley.cpp",
             "src/pyrpmd/potentials/base.py", "src/pyrpmd/potentials/python.py",
             "src/pyrpmd/potentials/optional.py",
             "src/pyrpmd/workflows/materials.py", "src/pyrpmd/trajectory.py",
             "src/pyrpmd/analysis/diffusion.py",
             "benchmarks/materials/ni_h_eam.py"]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"base_revision": revision, "working_tree_dirty": dirty, "pyrpmd_version": pyrpmd.__version__,
            "sha256": {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
                       for path in files if (root / path).is_file()}}


def _trajectory_chunks(path):
    """Read the exported committed history in either supported storage format."""
    path = Path(path)
    fields = ["positions", "velocities", "observables", "radii_gyration_squared"]
    if path.suffix == ".json":
        from pyrpmd.trajectory import iter_chunks

        manifest = json.loads(path.read_text())
        if manifest.get("format") != "PyRPMD.materials-trajectory-stream.v1":
            raise ValueError("Unknown materials trajectory manifest format")
        yield from iter_chunks(path.parent, manifest["trajectory"], fields=fields)
    else:
        with np.load(path, allow_pickle=False) as trajectory:
            yield {name: trajectory[name] for name in fields}


def _trajectory_diagnostics(path):
    """Accumulate finite-state, NVE and H-ring checks without collecting frames."""
    from pyrpmd.workflows.materials import OBSERVABLE_NAMES

    initial_energy = None
    drift, radius_sum, frames, finite = 0.0, 0.0, 0, True
    energy_column = OBSERVABLE_NAMES.index("ring_polymer_hamiltonian")
    for chunk in _trajectory_chunks(path):
        observables = chunk["observables"]
        if not len(observables):
            continue
        hamiltonian = observables[:, energy_column]
        if initial_energy is None:
            initial_energy = float(hamiltonian[0])
        drift = max(drift, float(np.max(np.abs(hamiltonian - initial_energy))))
        radius_sum += float(chunk["radii_gyration_squared"][:, -1].sum())
        frames += len(observables)
        finite = finite and all(np.isfinite(chunk[name]).all() for name in chunk)
    if not frames:
        raise ValueError("Materials pilot trajectory contains no saved frames")
    return drift, radius_sum / frames, bool(finite)


def run_pilot(model, output, *, quick=False):
    from pyrpmd import run
    from pyrpmd.io import write_json

    model = obtain_model(model)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty pilot output directory")
    output.mkdir(parents=True, exist_ok=True)
    verification = verify_adapter(model)
    if not verification["passed"]:
        raise AssertionError(f"Real EAM adapter validation failed: {verification}")
    report = {
        "schema": "pyrpmd.materials-eam-pilot", "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "potential": PROVENANCE, "source_at_start": _source_identity(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "ase", "h5py")},
        "scope": "Real PES/interface, finite-step sampling and NVE propagation smoke validation",
        "physical_converged": False, "equilibration_converged": False,
        "reproduces_suleimanov_2012": False,
        "limitations": ["27-atom three-layer Ni slab, no fixed layers, one H at 1/9 surface coverage.",
                        "Short records and four beads are not a convergence study.",
                        "One replica per isotope/bead case provides no independent diffusion uncertainty.",
                        "No quantum enhancement, isotope transport ratio or macroscopic diffusion is inferred.",
                        "Absolute H/Ni(100) accuracy of this historical EAM is not established by these tests."],
        "adapter_validation": verification, "cases": [],
    }
    for isotope, beads in (("H", 1), ("D", 1), ("H", 4), ("D", 4)):
        case_name = f"{isotope.lower()}-p{beads}"
        start = time.perf_counter()
        summary = run(job(model, isotope=isotope, beads=beads, quick=quick), output=output / case_name)
        drift, radius, finite = _trajectory_diagnostics(summary["files"]["trajectories"][0])
        ring_check = radius == 0.0 if beads == 1 else radius > 0.0
        energy_check = drift / beads < NVE_ENERGY_TOLERANCE_PER_BEAD
        saved_system = json.loads((output / case_name / "input/system.json").read_text())
        case = {"name": case_name, "isotope": isotope, "beads": beads,
                "hydrogen_mass_electron_masses": saved_system["masses"][-1],
                "temperature_K": 600.0, "dt_fs": 0.1,
                "equilibration_steps": 4 if quick else 64, "production_steps": 16 if quick else 160,
                "samples": summary["samples_per_replica"][0],
                "workflow_status": summary["status"], "analysis_status": summary["analysis_status"],
                "physical_converged": False, "finite_state_and_observables": finite,
                "max_H_P_drift_hartree": drift,
                "max_H_P_drift_per_bead_hartree": drift / beads,
                "energy_drift_tolerance_per_bead_hartree": NVE_ENERGY_TOLERANCE_PER_BEAD,
                "energy_drift_check": energy_check,
                "mean_H_radius_of_gyration_squared_bohr2": radius,
                "ring_spread_check": bool(ring_check),
                "elapsed_seconds": time.perf_counter() - start,
                "summary_relative_path": f"{case_name}/results/summary.json"}
        report["cases"].append(case)
        write_json(output / "report.json", report)
        print(json.dumps(case, allow_nan=False), flush=True)
        if not finite or not energy_check or not ring_check or summary["status"] != "complete":
            raise AssertionError("Real EAM propagation failed finite-state, ring-spread or finite-step energy checks")
    report["source_at_end"] = _source_identity()
    report["source_files_changed_during_run"] = [path for path, sha in report["source_at_start"]["sha256"].items()
                                                if report["source_at_end"]["sha256"].get(path) != sha]
    report["implementation_checks_passed"] = verification["passed"] and all(
        c["finite_state_and_observables"] and c["energy_drift_check"] and c["ring_spread_check"] and
        c["workflow_status"] == "complete" for c in report["cases"])
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Explicitly fetch the pinned model if absent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Only 4 equilibration + 16 production steps per case")
    args = parser.parse_args()
    model = obtain_model(args.model, download=args.download)
    run_pilot(model, args.output, quick=args.quick)


if __name__ == "__main__":
    main()
