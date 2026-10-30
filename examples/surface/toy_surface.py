"""Analytic periodic corrugated surface for software validation, not a Cu PES.

Atom 0 is an adsorbate. All remaining atoms define a rigid reference plane.
E = A/2*(1-cos(2*pi*x/Lx)) + B/2*(1-cos(2*pi*y/Ly))
    + kz/2*(z-z0)**2, relative to the substrate's geometric center.
This is an intentionally simple, periodic model with exact analytic forces.
"""

import numpy as np
from pyrpmd.potentials import PESResult


class CorrugatedSurface:
    metadata = {"supports_pbc": True, "thread_safety": "context_serial", "execution_device": "cpu"}

    def __init__(self, barrier=0.003, transverse_barrier=0.003, height=3.0, vertical_k=0.02):
        self.barrier = float(barrier)
        self.transverse_barrier = float(transverse_barrier)
        self.height = float(height)
        self.vertical_k = float(vertical_k)
        if not np.isfinite([barrier, transverse_barrier, height, vertical_k]).all() or min(barrier, transverse_barrier, vertical_k) <= 0:
            raise ValueError("toy surface parameters must be finite with positive barriers/stiffness")
        self.cell = None

    def bind_system(self, system):
        if system.natoms < 2 or system.pbc != (True, True, False):
            raise ValueError("toy surface needs an adsorbate, a substrate and pbc=[true,true,false]")
        if not np.allclose(system.cell, np.diag(np.diag(system.cell))) or np.any(np.diag(system.cell) <= 0):
            raise ValueError("toy surface uses an orthorhombic positive cell")
        if set(system.fixed_atoms) != set(range(1, system.natoms)):
            raise ValueError("toy surface requires all substrate atoms fixed and atom 0 mobile")
        self.cell = system.cell.copy()

    def evaluate(self, q):
        if self.cell is None:
            raise ValueError("bind_system must supply the periodic cell first")
        substrate = q[:, 1:].mean(axis=1)
        xyz = q[:, 0] - substrate
        frequencies = 2 * np.pi / np.diag(self.cell)[:2]
        phase = xyz[:, :2] * frequencies
        amplitudes = np.array([self.barrier, self.transverse_barrier])
        dz = xyz[:, 2] - self.height
        energy = (0.5 * amplitudes * (1 - np.cos(phase))).sum(axis=1)
        energy += 0.5 * self.vertical_k * dz**2
        gradient = np.column_stack([
            0.5 * amplitudes * frequencies * np.sin(phase), self.vertical_k * dz,
        ])
        forces = np.empty_like(q)
        forces[:, 0] = -gradient
        forces[:, 1:] = gradient[:, None] / (q.shape[1] - 1)
        return PESResult(energy, forces)


def build(**params):
    return CorrugatedSurface(**params)
