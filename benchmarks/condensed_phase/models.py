"""Independent periodic solid models for condensed-phase NQE validation.

The harmonic crystal is an analytically soluble *model* solid. The switched
Lennard-Jones FCC solid is anharmonic; neither is presented as an ab-initio
material or a fit to a measured equation of state. Reduced units have hbar=1.
"""
from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.geometry import validate_cell_pbc
from pyrpmd.native_ensemble import NativePeriodicPES
from pyrpmd.potentials.base import PESResult, coordinates

HERE = Path(__file__).resolve().parent


def fcc(cells=2, density=1.0):
    """Unrelaxed FCC lattice; four atoms per conventional cubic unit cell."""
    side = (4 / density)**(1 / 3)
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    q = np.concatenate([(np.asarray(x) + basis + .125) * side
                        for x in itertools.product(range(cells), repeat=3)])
    return q, np.eye(3) * (cells * side)


def build_lj(directory):
    """Build the checked-in, dependency-free C++ potential into scratch output."""
    source = HERE / "lj.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    path = Path(directory) / ("condensed_lj_" + digest + ".so")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["c++", "-std=c++11", "-O3", "-shared", "-fPIC",
                        str(source), "-o", str(path)], check=True)
    return path


class LennardJonesSolid(NativePeriodicPES):
    """Full periodic image sum of a C2-switched LJ potential, sigma=epsilon=1."""

    closed = False

    def __init__(self, library, atoms, cell, *, component=0):
        if component not in (0, 1, 2):
            raise ValueError("component must be full=0, repulsive=1 or attractive=2")
        super().__init__(library, "condensed_lj_v1", atoms, context=component,
                         owner=self if component else None,
                         identity=f"LJ sigma=epsilon=1 switch=2.0 cutoff=2.5 part={component}")
        self.bind_system({"cell": cell, "pbc": [True]*3})
        self.metadata.update(batch_independent=True, virial=True)

    def set_cell(self, cell):
        self.cell, flags = validate_cell_pbc(cell, [True]*3)
        self.pbc = np.ascontiguousarray(flags, dtype=np.int32)

    def close(self):
        self.closed = True


class HarmonicCrystal:
    """27 coupled atoms, periodic nearest-neighbour graph, Gruneisen EOS.

    U = U0(V) + 1/2 sum_a u_a.T K(V) u_a,
    K(V)=(V/V0)^(-2 gamma)*(pin I + coupling graph-Laplacian).
    u_i=q_i-f_i h are displacements about fractional lattice sites.
    A small positive pin fixes all translation zero modes, making the full
    Gaussian partition integral and finite-size reference unambiguous. The
    model has 3N dispersive vibrational modes, not one isolated oscillator.
    """

    metadata = {"batch_independent": True, "supports_pbc": True, "virial": True}

    def __init__(self, cells=3, spacing=3., mass=1., coupling=.12, pin=.06,
                 gamma=1.5, bulk=.06):
        self.cells = cells
        self.fractional = (np.array(list(itertools.product(range(cells), repeat=3))) + .5) / cells
        self.natoms = len(self.fractional)
        self.reference_cell = np.eye(3) * cells * spacing
        self.reference_volume = float(np.linalg.det(self.reference_cell))
        self.mass, self.gamma, self.bulk = mass, gamma, bulk
        self.masses = np.full(self.natoms, mass)
        self.base_matrix = np.eye(self.natoms) * pin
        mapping = {tuple(x): i for i, x in enumerate(itertools.product(range(cells), repeat=3))}
        for lattice, i in mapping.items():
            for axis in range(3):
                neighbour = list(lattice)
                neighbour[axis] = (neighbour[axis] + 1) % cells
                j = mapping[tuple(neighbour)]
                self.base_matrix[i, i] += coupling
                self.base_matrix[j, j] += coupling
                self.base_matrix[i, j] -= coupling
                self.base_matrix[j, i] -= coupling
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.base_matrix)
        self.set_cell(self.reference_cell)

    def set_cell(self, cell):
        self.cell, _ = validate_cell_pbc(cell, [True]*3)
        self.volume = float(np.linalg.det(self.cell))
        self.centers = self.fractional @ self.cell
        self.scale = (self.volume/self.reference_volume)**(-2*self.gamma)
        self.matrix = self.base_matrix*self.scale

    def elastic(self, volume=None):
        v = self.volume if volume is None else volume
        return self.bulk/(2*self.reference_volume)*(v-self.reference_volume)**2

    def evaluate(self, positions):
        q = coordinates(positions)
        if q.shape[1] != self.natoms:
            raise ValueError("crystal atom count mismatch")
        u = q - self.centers
        f = -np.einsum("ij,bja->bia", self.matrix, u)
        vib = -.5*np.sum(u*f, axis=(1, 2))
        scalar = (2*self.gamma*vib
                  - self.volume*self.bulk*(self.volume-self.reference_volume)/self.reference_volume)
        w = np.einsum("bia,bic->bac", f, u) + scalar[:, None, None]*np.eye(3)
        return PESResult(vib+self.elastic(), f, virial=w)

    def quantum_reference(self, beta, volume=None):
        v = self.volume if volume is None else volume
        frequencies = np.sqrt(self.eigenvalues/self.mass)*(v/self.reference_volume)**(-self.gamma)
        x = beta*frequencies/2
        vib_energy = 3*np.sum(.5*frequencies/np.tanh(x))
        cv = 3*np.sum((x/np.sinh(x))**2)
        pressure = self.gamma*vib_energy/v - self.bulk*(v-self.reference_volume)/self.reference_volume
        logz = -beta*self.elastic(v) - 3*np.sum(np.log(2*np.sinh(x)))
        return dict(energy=vib_energy+self.elastic(v), heat_capacity_kB=cv,
                    pressure=pressure, log_partition=logz)

    def close(self):
        pass


def piglet_parameters(beta):
    """Temperature-scaled official i-PI four-bead parameter set; no refitting."""
    raw = json.loads((HERE/"piglet_4_750K.json").read_text())
    source_beta = 1/(KB_AU*raw["temperature_K"])
    return {"A": np.asarray(raw["A"])*source_beta/beta,
            "C": np.asarray(raw["C"]), "beta": beta, "beads": 4,
            "provenance": raw["provenance"] + f" source blob {raw['source_blob_sha']}; beta_target={beta:.17g}"}

