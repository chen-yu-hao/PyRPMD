#!/usr/bin/env python3
"""Compare the NumPy port with the independently compiled, pinned POTLIB kernel.

Requires gfortran only for this check; the shipped Python PES does not require
a compiler or ChemPotPy installation. The original .f is authenticated before
the unused ChemPotPy wrapper is excluded. No fit equation or constant is edited.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from pyrpmd.io import write_json
from pyrpmd.potentials.multistate_external import NaH2LEPS5G


HERE = Path(__file__).resolve().parent


def comparison_geometries(count=64, seed=20261101):
    rng = np.random.default_rng(seed)
    q = np.zeros((count, 3, 3))
    bond = rng.uniform(1.1, 2.8, count)
    radius = rng.uniform(2.5, 12., count)
    angle = rng.uniform(.08, np.pi-.08, count)
    q[:, 0, 0] = radius*np.cos(angle)
    q[:, 0, 1] = radius*np.sin(angle)
    q[:, 1, 0], q[:, 2, 0] = -.5*bond, .5*bond
    for frame in q:
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        frame[:] = frame@rotation+rng.uniform(-3., 3., 3)
    return q


class OriginalFortran:
    def __init__(self, compiler, directory):
        source = HERE/"vendor/chempotpy/NaH2_LEPS_5G_1992_DPEM.f"
        data = source.read_bytes()
        expected = NaH2LEPS5G().to_dict()["source"]["sha256"]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("original Fortran source hash mismatch")
        text = data.decode("utf-8")
        start = re.search(r"^      subroutine prepot\s*$", text, re.MULTILINE)
        if start is None:
            raise ValueError("pinned source does not contain the expected native POTLIB kernel")
        # The prefix contains the unused angstrom/eV wrapper and a LAPACK
        # diagonalizer. The native bohr/hartree PREPOT/POT and all fit routines
        # below this boundary are copied byte-for-byte into the compilation.
        core = Path(directory)/"nah2_5g_original.f"
        core.write_text(text[start.start():], encoding="utf-8")
        library = Path(directory)/"nah2_5g_original.so"
        self.command = [compiler, "-O2", "-fPIC", "-shared", "-std=legacy",
                        "-ffixed-line-length-72", str(core), "-o", str(library)]
        subprocess.run(self.command, check=True, capture_output=True, text=True)
        self.library = ctypes.CDLL(str(library))
        self.library.prepot_.argtypes = []
        self.library.prepot_.restype = None
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        self.library.pot_.argtypes = [array, array, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.library.pot_.restype = None
        self.library.prepot_()

    def matrix(self, q):
        distances = np.asfortranarray(np.stack([
            np.linalg.norm(q[:, 0]-q[:, 1], axis=-1),
            np.linalg.norm(q[:, 1]-q[:, 2], axis=-1),
            np.linalg.norm(q[:, 0]-q[:, 2], axis=-1),
        ], axis=1))
        nt = ctypes.c_int(len(q))
        values = []
        for element in (1, 2, 3):
            energy = np.empty(len(q), dtype=np.float64)
            surface = ctypes.c_int(element)
            self.library.pot_(distances, energy, ctypes.byref(nt), ctypes.byref(surface))
            values.append(energy)
        result = np.empty((len(q), 2, 2))
        result[:, 0, 0], result[:, 1, 1] = values[0], values[2]
        result[:, 0, 1] = result[:, 1, 0] = values[1]
        return result


def compare(compiler="gfortran", count=64, step=2e-4):
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError("gfortran is required for the independent original-source comparison; "
                           "the local molecular simulation remains available without it")
    q = comparison_geometries(count)
    potential = NaH2LEPS5G()
    actual = potential.evaluate(q)
    with tempfile.TemporaryDirectory(prefix="pyrpmd-nah2-reference-") as directory:
        original = OriginalFortran(executable, directory)
        expected = original.matrix(q)
        numerical = np.empty_like(actual.gradients)
        for atom in range(3):
            for axis in range(3):
                values = []
                for amount in (2., 1., -1., -2.):
                    displaced = q.copy()
                    displaced[:, atom, axis] += amount*step
                    values.append(original.matrix(displaced))
                numerical[..., atom, axis] = (-values[0]+8*values[1]-8*values[2]+values[3])/(12*step)
    matrix_error = float(np.max(np.abs(actual.matrix-expected)))
    gradient_error = float(np.max(np.abs(actual.gradients-numerical)))
    eigenvalue_error = float(np.max(np.abs(np.linalg.eigvalsh(actual.matrix)-np.linalg.eigvalsh(expected))))
    accepted = matrix_error < 3e-12 and eigenvalue_error < 3e-12 and gradient_error < 3e-8
    return {
        "format": "PyRPMD.NaH2.original_fortran_comparison.v1",
        "passed": accepted, 
        "source": potential.to_dict()["source"],
        "compiler": subprocess.run([executable, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
        "compilation_flags": ["-O2", "-fPIC", "-shared", "-std=legacy", "-ffixed-line-length-72"],
        "geometries": count, "geometry_seed": 20261101,
        "gradient_reference": "five-point Cartesian differences of independently compiled original POT energies",
        "difference_step_bohr": step,
        "max_matrix_error_hartree": matrix_error,
        "max_eigenvalue_error_hartree": eigenvalue_error,
        "max_gradient_error_hartree_per_bohr": gradient_error,
        "thresholds": {"matrix_hartree": 3e-12, "eigenvalues_hartree": 3e-12,
                       "gradient_hartree_per_bohr": 3e-8},
        "scope": "implementation equivalence; not validation of chemical accuracy or nonadiabatic dynamics",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/nah2_fortran_comparison.json"))
    args = parser.parse_args()
    if args.count < 4:
        parser.error("--count must be >= 4")
    result = compare(args.compiler, args.count)
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["passed"]:
        raise SystemExit("NaH2 port failed the independent original-source check")


if __name__ == "__main__":
    main()
