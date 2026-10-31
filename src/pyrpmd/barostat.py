"""Isotropic centroid Bussi--Zykova--Parrinello pressure control.

Equations (4)--(7) of Ceriotti, More and Manolopoulos, Comput. Phys.
Commun. 185, 1019 (2014), doi:10.1016/j.cpc.2013.10.027, use the same
sum-beads Hamiltonian and beta/P convention as :mod:`pyrpmd.dynamics`.
Only centroid coordinates and momenta couple to the isotropic cell piston.
This module composes their BZP generator with the existing fixed-cell
integrators using a symmetric split; it does not copy an external integrator.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from functools import partial

import numpy as np

from .geometry import validate_cell_pbc
from .potentials.base import PotentialError, validate_result
from .validation import finite_real

_real = partial(finite_real, message='{name} must be a finite real{bound}')


def validate_bzp_snapshot(snapshot, reference_cell):
    """Validate piston state without constructing or contacting a potential.

    Completed-run reanalysis and storage migration use this same validation
    as live restart. Return a normalized independent JSON-compatible value.
    """
    if (not isinstance(snapshot, Mapping)
            or set(snapshot) != {"method", "cell", "momentum"}
            or snapshot["method"] != "bzp"):
        raise ValueError("invalid BZP checkpoint state")
    cell, _ = validate_cell_pbc(snapshot["cell"], [True, True, True])
    reference, _ = validate_cell_pbc(reference_cell, [True, True, True])
    scale = float(np.sum(cell * reference) / np.sum(reference**2))
    if (not np.isfinite(scale) or scale <= 0
            or not np.allclose(cell, scale * reference, rtol=1e-11, atol=1e-12)):
        raise ValueError("BZP checkpoint cell is not a positive isotropic scaling of the reference cell")
    volume = abs(np.linalg.det(cell))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("BZP checkpoint volume must be positive and finite")
    momentum = _real(snapshot["momentum"], "piston momentum")
    return {"method": "bzp", "cell": cell.tolist(), "momentum": momentum}


class BZPBarostat:
    """Isotropic NPT PIMD with a Langevin cell piston, in atomic units.

    ``potential`` must support ``set_cell(cell_bohr)`` and return a physical
    per-bead configurational virial ``W = -V * stress``. The virial includes
    explicit cell derivatives and excludes kinetic and ring-spring terms.
    A missing virial is an error; Cartesian ``q dot F`` is not substituted.

    The system has a fully periodic, constant-shape 3D cell and all atoms are
    mobile. Cell vectors are rows. The input System remains the immutable run
    reference; the evolving cell lives here and in the provider. No COM
    position or momentum constraint is imposed.

    ``tau_au`` sets piston mass ``3*N*tau_au**2/beta``; by default its friction
    is ``1/tau_au``. Particle thermostatting remains explicit in :meth:`step`.
    ``friction_au=0`` and ``thermostat=None`` expose the deterministic map for
    numerical checks, not an NPT sampling recommendation.
    """

    def __init__(self, potential, system, beta, nbeads, pressure_au, tau_au,
                 friction_au=None):
        self.beta = _real(beta, "beta", positive=True)
        if (isinstance(nbeads, (bool, np.bool_))
                or not isinstance(nbeads, (int, np.integer)) or nbeads < 1):
            raise ValueError("nbeads must be a positive integer")
        self.nbeads = int(nbeads)
        self.pressure_au = _real(pressure_au, "pressure_au")
        self.tau_au = _real(tau_au, "tau_au", positive=True)
        self.friction_au = _real(1.0 / self.tau_au if friction_au is None else friction_au,
                                 "friction_au", nonnegative=True)
        get = system.get if isinstance(system, Mapping) else lambda k, d=None: getattr(system, k, d)
        self._cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        if self._cell is None or not pbc.all():
            raise ValueError("BZP requires a fully periodic 3D cell")
        if get("fixed_atoms", ()):
            raise ValueError("BZP requires all atoms mobile; fixed atoms need fixed-cell sampling")
        masses = np.asarray(get("masses"))
        if (masses.ndim != 1 or not len(masses) or masses.dtype.kind not in "iuf"
                or not np.isfinite(masses).all() or np.any(masses <= 0)):
            raise ValueError("BZP requires positive finite atomic masses")
        self.masses = np.array(masses, dtype=float, copy=True)
        self.masses.flags.writeable = False
        self.mass = _real(3.0 * len(masses) * self.tau_au**2 / self.beta,
                          "piston mass", positive=True)
        self._reference_cell = self._cell.copy()
        self.potential = potential
        if not callable(getattr(potential, "set_cell", None)):
            raise PotentialError("BZP requires a PES with explicit set_cell(cell_bohr) support")
        self.momentum = 0.0
        self._set_cell(self._cell)

    @property
    def cell(self):
        """An independent snapshot of the current row-vector cell in bohr."""
        return self._cell.copy()

    @property
    def volume(self):
        return float(abs(np.linalg.det(self._cell)))

    def _set_cell(self, cell):
        cell, _ = validate_cell_pbc(cell, [True, True, True])
        volume = abs(np.linalg.det(cell))
        if not np.isfinite(volume) or volume <= 0:
            raise FloatingPointError("BZP cell has a nonpositive or nonfinite volume")
        self.potential.set_cell(cell.copy())
        self._cell = cell

    def _check_state(self, state):
        shape = (self.nbeads, len(self.masses), 3)
        if (state.q.shape != shape or state.p.shape != shape
                or not np.isfinite(state.q).all() or not np.isfinite(state.p).all()):
            raise ValueError("BZP state shape/finiteness disagrees with the system")
        _real(self.momentum, "piston momentum")

    def _pressure_numerator(self, state):
        self._check_state(state)
        result = validate_result(self.potential.evaluate(state.q), state.q.shape)
        if result.virial is None:
            raise PotentialError("BZP requires per-bead physical configurational virials from the PES")
        centroid_p = state.p.mean(axis=0)
        offsets = state.q - state.q.mean(axis=0, keepdims=True)
        # p_0 = sqrt(P)*p_centroid for orthonormal normal modes. Thus this
        # equals 2*K_0 + Tr(sum_j W_j) - sum_ij (q_ij-qbar_i)*F_ij.
        twice_kinetic = self.nbeads * np.sum(centroid_p**2 / self.masses[:, None])
        virial = np.trace(result.virial, axis1=1, axis2=2).sum()
        correction = np.sum(offsets * result.forces)
        numerator = float(twice_kinetic + virial - correction)
        if not np.isfinite(numerator):
            raise FloatingPointError("nonfinite BZP centroid-virial pressure")
        return numerator

    def pressure(self, state):
        """Instantaneous centroid-virial pressure, hartree/bohr**3."""
        return self._pressure_numerator(state) / (3.0 * self.nbeads * self.volume)

    def snapshot(self):
        """JSON-compatible cell/piston state; particle RNG belongs to Dynamics."""
        return {"method": "bzp", "cell": self._cell.tolist(), "momentum": float(self.momentum)}

    def restore(self, snapshot):
        """Restore a checkpoint after validating fixed-shape isotropic scaling."""
        saved = validate_bzp_snapshot(snapshot, self._reference_cell)
        self._set_cell(saved["cell"])
        self.momentum = saved["momentum"]

    def _piston_kick(self, state, dt):
        # The +3*P/beta correction corresponds to unconstrained COM and to
        # dV measure. Omitting it yields the wrong NPT volume distribution.
        force = (self._pressure_numerator(state)
                 - 3.0 * self.nbeads * self.pressure_au * self.volume
                 + 3.0 * self.nbeads / self.beta)
        self.momentum += dt * force
        if not np.isfinite(self.momentum):
            raise FloatingPointError("nonfinite BZP piston momentum")

    def _dilate(self, state, dt):
        exponent = dt * self.momentum / self.mass
        # Bounds keep both Cartesian state and determinant representable.
        # Reject the step; never clip pressure or the accepted cell change.
        if not np.isfinite(exponent) or abs(exponent) > 100.0:
            raise FloatingPointError("BZP cell change overflow; reduce timestep or increase time_constant")
        scale = math.exp(exponent)
        centroid_q, centroid_p = state.q.mean(axis=0), state.p.mean(axis=0)
        self._set_cell(self._cell * scale)
        state.q += centroid_q[None] * math.expm1(exponent)
        state.p += centroid_p[None] * math.expm1(-exponent)

    def _thermostat(self, rng, dt):
        if self.friction_au == 0.0:
            return
        decay = math.exp(-self.friction_au * dt)
        amplitude = math.sqrt(self.nbeads * self.mass / self.beta
                              * (-math.expm1(-2.0 * self.friction_au * dt)))
        self.momentum = decay * self.momentum + amplitude * float(rng.normal())

    def step(self, state, dynamics, *, thermostat="pile-l", gamma_centroid=1.0,
             andersen_interval=10):
        """Advance one pressure-controlled step, restoring every state on error.

        The palindrome is piston OU/2, pressure kick/2, dilation/2,
        existing fixed-volume propagator, dilation/2, pressure kick/2,
        piston OU/2. The exact dilation preserves each internal bead offset.
        PILE-L retains the selected BCOCB/BAOAB particle map. Andersen uses
        the existing global collision clock. No biased/constraint dynamics
        or COM projection is accepted by this API.
        """
        if (dynamics.potential is not self.potential or dynamics.beta != self.beta
                or dynamics.nbeads != self.nbeads
                or not np.array_equal(dynamics.masses, self.masses)):
            raise ValueError("BZP and Dynamics must share the potential, masses, beta and bead count")
        if dynamics.fixed_atoms:
            raise ValueError("BZP requires all atoms mobile")
        if dynamics.loop_backend == "native":
            raise ValueError("BZP requires loop_backend='auto' or 'python'")
        if thermostat not in ("pile-l", "andersen", None):
            raise ValueError("BZP particle thermostat must be pile-l or andersen")
        if (isinstance(andersen_interval, (bool, np.bool_))
                or not isinstance(andersen_interval, (int, np.integer)) or andersen_interval < 1):
            raise ValueError("andersen_interval must be a positive integer")
        dt = _real(dynamics.dt, "BZP timestep", positive=True)
        self._check_state(state)
        saved_state = state.copy()
        saved_piston = self.snapshot()
        saved_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
        saved_stats = copy.deepcopy(dynamics.propagation_stats)
        try:
            self._thermostat(dynamics.rng, dt / 2)
            self._piston_kick(state, dt / 2)
            self._dilate(state, dt / 2)
            if thermostat == "pile-l":
                dynamics.sample_step(state, gamma_centroid=gamma_centroid)
            else:
                if thermostat == "andersen" and state.step % andersen_interval == 0:
                    state.p[:] = dynamics.sample_momenta()
                dynamics.step(state)
            self._dilate(state, dt / 2)
            self._piston_kick(state, dt / 2)
            self._thermostat(dynamics.rng, dt / 2)
            self._check_state(state)
            dynamics._record_python_steps(1)
            if dynamics.loop_backend != "python":
                reasons = dynamics.propagation_stats["fallback_reasons"]
                reason = "BZP pressure control uses the Python propagation loop"
                reasons[reason] = reasons.get(reason, 0) + 1
        except BaseException:
            state.q[:], state.p[:] = saved_state.q, saved_state.p
            state.step, state.time = saved_state.step, saved_state.time
            dynamics.rng.bit_generator.state = saved_rng
            dynamics.propagation_stats = saved_stats
            self.restore(saved_piston)
            raise
        return state

    def extended_energy(self, state, dynamics):
        """Deterministic extended invariant (thermostat heat is not included)."""
        return float(dynamics.energy(state) + self.nbeads * self.pressure_au * self.volume
                     + self.momentum**2 / (2.0 * self.mass)
                     - self.nbeads / self.beta * math.log(self.volume))
