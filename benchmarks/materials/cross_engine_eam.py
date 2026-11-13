"""Compare independent ASE/LAMMPS interpolation of the pinned real Ni-H EAM.

This verifies a provider and its physical virial, not the Ni(100) literature
diffusion coefficient. The historical EAM table is supplied separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.materials.ni_h_eam import PROVENANCE, make_slab, obtain_model
from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.potentials.lammps import LAMMPSPotential


def _script(path, atoms, model):
    model = str(Path(model).resolve())
    if any(c in model for c in ('"', '\n', '\r', '$')):
        raise ValueError("unsupported characters in model path")
    matrix = atoms.cell[:]
    lengths = np.diag(matrix)
    fractional = atoms.positions @ np.linalg.inv(matrix)
    fractional[:, atoms.pbc] -= np.floor(fractional[:, atoms.pbc])
    positions = fractional @ matrix
    lines = ["units metal", "atom_style atomic", "boundary p p f",
             "atom_modify map array",
             "region box prism 0 %.17g 0 %.17g 0 %.17g %.17g %.17g %.17g" %
             (*lengths, matrix[1, 0], matrix[2, 0], matrix[2, 1]),
             "create_box 2 box"]
    for i, q in enumerate(positions):
        lines.append("create_atoms %d single %.17g %.17g %.17g units box" %
                     ((1 if i < len(atoms)-1 else 2), *q))
    lines += ["mass 1 58.6934", "mass 2 1.00794", "pair_style eam/alloy",
              f'pair_coeff * * "{model}" Ni H']
    path.write_text("\n".join(lines)+"\n")


def audit(model):
    from ase.calculators.eam import EAM

    model = obtain_model(model)
    atoms = make_slab()
    atoms.calc = EAM(potential=str(model))
    base = atoms.positions.copy()
    rng = np.random.default_rng(20261101)
    configurations = [base.copy()]
    for _ in range(11):
        q = base+rng.normal(0, .025, base.shape)
        q[-1] += rng.normal(0, .12, 3)
        configurations.append(q)
    rows = []
    with tempfile.TemporaryDirectory(prefix="pyrpmd-eam-") as temporary:
        script = Path(temporary)/"model.in"
        _script(script, atoms, model)
        cell = atoms.cell[:]*ANGSTROM_TO_AU
        with LAMMPSPotential(atoms.get_chemical_symbols(), script,
                             type_map=["Ni", "H"], cell=cell,
                             pbc=atoms.pbc, asset_files=[model]) as provider:
            for i, coordinates in enumerate(configurations):
                atoms.positions = coordinates
                q = coordinates[None]*ANGSTROM_TO_AU
                before = q.copy()
                result = provider.evaluate(q)
                assert np.array_equal(q, before), "provider changed unwrapped coordinates"
                energy = result.energy[0]/EV_TO_AU
                force = result.forces[0]/(EV_TO_AU*BOHR_ANGSTROM)
                virial = result.virial[0]/EV_TO_AU
                ase_energy = atoms.get_potential_energy()
                ase_force = atoms.get_forces()
                ase_virial = -atoms.get_stress(voigt=False)*atoms.get_volume()
                rows.append({"case": i, "coordinates_angstrom": coordinates.tolist(),
                             "lammps_energy_eV": float(energy), "ase_energy_eV": ase_energy,
                             "absolute_energy_difference_eV": float(abs(energy-ase_energy)),
                             "max_force_difference_eV_per_angstrom": float(np.max(abs(force-ase_force))),
                             "max_virial_difference_eV": float(np.max(abs(virial-ase_virial)))})
            # One configuration, six independent Cartesian energy derivatives.
            q = configurations[1][None]*ANGSTROM_TO_AU
            reference = provider.evaluate(q)
            force = reference.forces[0]/(EV_TO_AU*BOHR_ANGSTROM)
            fd = []
            for atom in (0, len(atoms)-1):
                for axis in range(3):
                    estimates = []
                    for h in (1e-4, 5e-5):
                        plus, minus = q.copy(), q.copy()
                        plus[0, atom, axis] += h*ANGSTROM_TO_AU
                        minus[0, atom, axis] -= h*ANGSTROM_TO_AU
                        ep, em = provider.evaluate(plus).energy[0], provider.evaluate(minus).energy[0]
                        estimates.append(float(-(ep-em)/(2*h*EV_TO_AU)))
                    fd.append({"atom": atom, "axis": axis, "energy_derivatives_eV_per_angstrom": estimates,
                               "reported_force_eV_per_angstrom": float(force[atom, axis]),
                               "absolute_error_eV_per_angstrom": abs(estimates[-1]-force[atom, axis])})
            # Homogeneous cell/coordinate strain checks include many-body EAM
            # virials rather than a potentially wrong sum(r*f) under PBC.
            strain = []
            for a, b in ((0, 0), (1, 1), (2, 2), (1, 0)):
                h = 2e-5
                energies = []
                for sign in (1, -1):
                    transform = np.eye(3)
                    transform[a, b] += sign*h
                    # Partial-PBC slabs intentionally do not support variable-
                    # cell dynamics. Build a fresh fixed-cell provider at each
                    # strain; do not weaken that guard for this derivative.
                    strained = atoms.copy()
                    strained.cell = atoms.cell[:] @ transform
                    strained.positions = q[0] * BOHR_ANGSTROM @ transform
                    strain_script = Path(temporary)/f"strain-{a}-{b}-{sign}.in"
                    _script(strain_script, strained, model)
                    with LAMMPSPotential(atoms.get_chemical_symbols(), strain_script,
                                         type_map=["Ni", "H"], cell=cell@transform,
                                         pbc=atoms.pbc, asset_files=[model]) as strained_provider:
                        energies.append(float(strained_provider.evaluate(q@transform).energy[0]/EV_TO_AU))
                derivative = -(energies[0]-energies[1])/(2*h)
                expected = float(reference.virial[0, a, b]/EV_TO_AU)
                strain.append({"indices": [a, b], "energy_derivative_eV": derivative,
                               "reported_virial_eV": expected, "absolute_error_eV": abs(derivative-expected)})
            shifted = q.copy()
            shifted[:, -1] += 3*cell[0]-2*cell[1]
            translated = provider.evaluate(shifted)
            translation_error = float(np.max(abs(translated.forces-reference.forces)))
            metadata = dict(provider.metadata)
    maxima = {key: max(row[key] for row in rows) for key in
              ("absolute_energy_difference_eV", "max_force_difference_eV_per_angstrom", "max_virial_difference_eV")}
    maxima["finite_difference_force_error_eV_per_angstrom"] = max(row["absolute_error_eV_per_angstrom"] for row in fd)
    maxima["strain_virial_error_eV"] = max(row["absolute_error_eV"] for row in strain)
    tolerances = {"absolute_energy_difference_eV": 2e-4,
                  "max_force_difference_eV_per_angstrom": 5e-4,
                  "max_virial_difference_eV": 5e-3,
                  "finite_difference_force_error_eV_per_angstrom": 5e-4,
                  "strain_virial_error_eV": 5e-3}
    return {"schema": "pyrpmd.cross-engine-eam.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "same tabulated EAM, independent engines, finite differences, periodic physical virial",
            "geometry": "27 Ni atoms in 3x3x3 Ni(100) slab, one H; a=3.52A; no frozen layers",
            "seed": 20261101, "cases": rows, "model": PROVENANCE,
            "provider": metadata, "maxima": maxima, "tolerances": tolerances,
            "force_derivatives": fd, "strain_derivatives": strain,
            "periodic_force_error_hartree_per_bohr": translation_error,
            "software_checks_passed": all(maxima[k] < v for k, v in tolerances.items()) and translation_error < 1e-11,
             "literature_reproduced": False,
            "limitations": ["ASE and LAMMPS interpolate the tabulated functions differently; byte-identical floating-point forces are not expected.",
                            "No diffusivity, equilibrium sampling or match to the 2012 Ni(100) PES/lattice has been established.",
                            "Tolerance values are engineering checks for table interpolation and finite differences, not uncertainties of a literature diffusion coefficient."],
            "versions": {n: importlib.metadata.version(n) for n in ("numpy", "ase", "lammps")},
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                              ("benchmarks/materials/cross_engine_eam.py", "benchmarks/materials/ni_h_eam.py",
                               "src/pyrpmd/potentials/lammps.py", "src/pyrpmd/constants.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("maxima", "software_checks_passed")}, indent=2))
    raise SystemExit(0 if report["software_checks_passed"] else 1)


if __name__ == "__main__":
    main()
