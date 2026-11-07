"""Online joint RPMD--quantum-subsystem propagation.

This module provides a small, deterministic prototype for *online* mixed
propagation.  A ring-polymer environment is propagated with velocity Verlet
while a one-dimensional quantum subsystem is propagated on the same time
step by a Fourier split operator.  The quantum potential is evaluated at the
ring-polymer centroid and its gradient contributes an Ehrenfest force to the
classical beads.

The wavefunction propagation is exact for the supplied finite grid and
frozen potential during one split-operator step (up to the usual
second-order time-step and finite-grid errors).  The complete coupled
trajectory is a mixed quantum--classical approximation: it is not an exact
entangled system+bath propagator, and it does not provide a generally exact
reaction rate.  ``physical_converged`` is therefore always ``False`` until
independent grid, time-step, bead, and sampling checks have been performed by
the caller.

The API intentionally takes potential/force callables instead of coupling to
one particular PyRPMD engine.  This keeps the reference workflow useful for
small model systems and permits a production RPMD driver to feed an online
centroid and receive the quantum force.  All quantities use atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


ArrayLike = Any
QuantumPotential = Callable[[np.ndarray, np.ndarray], np.ndarray]
QuantumGradient = Callable[[np.ndarray, np.ndarray], np.ndarray]
BathForce = Callable[[np.ndarray], np.ndarray]
BathPotential = Callable[[np.ndarray], float]

_EPS = np.finfo(float).eps


def _readonly(value: ArrayLike, *, dtype=None) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    frozen = np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)
    frozen.flags.writeable = False
    return frozen


def _grid(value: ArrayLike) -> tuple[np.ndarray, float]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size < 4 or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("grid must be a finite one-dimensional vector with at least four points")
    grid = np.array(raw, dtype=float, copy=True)
    diffs = np.diff(grid)
    dx = float(diffs[0])
    if dx <= 0 or not np.allclose(diffs, dx, rtol=1e-10, atol=0.0):
        raise ValueError("grid must be strictly increasing and uniformly spaced")
    return _readonly(grid), dx


def _scalar(value: Any, name: str, *, positive=False, nonnegative=False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    out = float(raw)
    if positive and out <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and out < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _integer(value: Any, name: str, *, nonnegative=False, positive=False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if positive and value < 1:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


@dataclass(frozen=True)
class JointPropagationResult:
    """Recorded trajectory and quantum state from :class:`OnlineQuantumRPMD`.

    ``positions`` and ``momenta`` have shape ``[frame, bead, bath_coordinate]``;
    ``wavefunctions`` has shape ``[frame, grid]`` and uses the convention
    ``dx*sum(abs(psi)**2)=1``.  ``quantum_force`` is the force on the centroid
    bath coordinate before it is distributed equally over the beads.  The
    energy fields are diagnostics for the mixed propagation and are not
    proofs of convergence.
    """

    time: np.ndarray
    positions: np.ndarray
    momenta: np.ndarray
    wavefunctions: np.ndarray
    quantum_position: np.ndarray
    quantum_momentum: np.ndarray
    quantum_energy: np.ndarray
    quantum_force: np.ndarray
    total_energy: np.ndarray
    wavefunction_norm: np.ndarray
    diagnostics: Mapping[str, Any]
    physical_converged: bool = False

    def __post_init__(self):
        for name in (
            "time", "positions", "momenta", "wavefunctions", "quantum_position",
            "quantum_momentum", "quantum_energy", "quantum_force", "total_energy",
            "wavefunction_norm",
        ):
            value = np.asarray(getattr(self, name))
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, _readonly(value))
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "physical_converged", bool(self.physical_converged))

    @property
    def final_positions(self) -> np.ndarray:
        return self.positions[-1]

    @property
    def final_momenta(self) -> np.ndarray:
        return self.momenta[-1]

    @property
    def final_wavefunction(self) -> np.ndarray:
        return self.wavefunctions[-1]


class OnlineQuantumRPMD:
    """Jointly propagate an RPMD environment and a quantum coordinate.

    Parameters
    ----------
    grid, quantum_mass, dt:
        Uniform periodic quantum-coordinate grid, mass and time step.
    beta:
        Inverse temperature.  The ring-polymer spring frequency is
        ``P / beta`` for ``P=len(positions)`` in atomic units.
    positions, momenta, bath_masses:
        Initial environment ring-polymer state.  Arrays are shaped
        ``[bead, bath_coordinate]`` (a one-bead trajectory is valid).
    quantum_potential:
        Callable ``V = quantum_potential(grid, centroid)`` returning a real
        ``[grid]`` potential for the quantum coordinate.  ``centroid`` is a
        one-dimensional bath-coordinate vector.
    quantum_gradient:
        Optional callable returning ``dV/dcentroid`` with shape
        ``[bath_coordinate, grid]``.  When omitted, a centered finite
        difference of ``quantum_potential`` is used.  The gradient is only
        used to provide the Ehrenfest force to the RPMD beads.
    bath_force:
        Optional callable returning ``-dV_bath/dq`` for each bead, with the
        same shape as ``positions``.  It is divided by ``P`` internally,
        consistent with the standard RPMD Hamiltonian.
    bath_potential:
        Optional callable returning the physical bath potential for one bead.
        It is used for an energy diagnostic only.
    wavefunction:
        Initial normalized quantum wavefunction with ``dx*sum(abs(psi)**2)=1``.

    Notes
    -----
    The coupling is evaluated at the centroid, so the quantum force is shared
    equally among the beads.  This is an explicit closure choice and is
    recorded in ``diagnostics``.  It is useful as an online reference
    implementation, not a general exact quantum dynamics algorithm.
    """

    def __init__(
        self,
        grid: ArrayLike,
        quantum_mass: float,
        beta: float,
        dt: float,
        positions: ArrayLike,
        momenta: ArrayLike,
        bath_masses: ArrayLike,
        quantum_potential: QuantumPotential,
        *,
        quantum_gradient: QuantumGradient | None = None,
        bath_force: BathForce | None = None,
        bath_potential: BathPotential | None = None,
        wavefunction: ArrayLike | None = None,
        finite_difference_step: float = 1e-5,
    ):
        self.grid, self.dx = _grid(grid)
        self.quantum_mass = _scalar(quantum_mass, "quantum_mass", positive=True)
        self.beta = _scalar(beta, "beta", positive=True)
        self.dt = _scalar(dt, "dt", positive=True)
        self.fd_step = _scalar(finite_difference_step, "finite_difference_step", positive=True)

        q = np.asarray(positions)
        p = np.asarray(momenta)
        m = np.asarray(bath_masses)
        if (q.ndim != 2 or q.shape[0] < 1 or q.shape[1] < 1 or q.dtype.kind not in "fiu"
                or not np.isfinite(q).all()):
            raise ValueError("positions must be a finite [bead,bath_coordinate] array")
        if p.shape != q.shape or p.dtype.kind not in "fiu" or not np.isfinite(p).all():
            raise ValueError("momenta must have the same finite shape as positions")
        if m.ndim != 1 or m.shape[0] != q.shape[1] or m.dtype.kind not in "fiu":
            raise ValueError("bath_masses must match the bath-coordinate axis")
        if not np.isfinite(m).all() or np.any(m <= 0):
            raise ValueError("bath_masses must be positive and finite")
        if not callable(quantum_potential):
            raise TypeError("quantum_potential must be callable")
        if quantum_gradient is not None and not callable(quantum_gradient):
            raise TypeError("quantum_gradient must be callable or None")
        if bath_force is not None and not callable(bath_force):
            raise TypeError("bath_force must be callable or None")
        if bath_potential is not None and not callable(bath_potential):
            raise TypeError("bath_potential must be callable or None")

        self.positions = np.array(q, dtype=float, copy=True)
        self.momenta = np.array(p, dtype=float, copy=True)
        self.bath_masses = np.array(m, dtype=float, copy=True)
        self.quantum_potential = quantum_potential
        self.quantum_gradient = quantum_gradient
        self.bath_force = bath_force
        self.bath_potential = bath_potential
        self.nbeads, self.nbath = self.positions.shape
        self.spring_frequency = self.nbeads / self.beta
        if wavefunction is None:
            raise ValueError("wavefunction is required; provide a normalized finite-grid state")
        psi = np.asarray(wavefunction)
        if psi.ndim != 1 or psi.shape != self.grid.shape or psi.dtype.kind not in "fciu" or not np.isfinite(psi).all():
            raise ValueError("wavefunction must be a finite vector with the same shape as grid")
        self.wavefunction = np.array(psi, dtype=complex, copy=True)
        norm = self._norm(self.wavefunction)
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("wavefunction must have unit dx-weighted norm")
        self._kinetic = (2.0 * np.pi * np.fft.fftfreq(len(self.grid), d=self.dx)) ** 2 / (2.0 * self.quantum_mass)
        phase = -1j * self.dt * self._kinetic
        if not np.isfinite(phase).all():
            raise ValueError("quantum grid, mass and dt produce non-finite kinetic phases")
        self._kinetic_phase = np.exp(phase)
        self.time = 0.0
        self.diagnostics = {
            "algorithm": "online_rpmd_quantum_ehrenfest",
            "quantum_propagator": "periodic_fourier_strang",
            "quantum_coordinate_count": 1,
            "environment_beads": self.nbeads,
            "centroid_coupling": True,
            "quantum_force_distribution": "equal_over_beads",
            "quantum_gradient_source": "analytic" if quantum_gradient is not None else "finite_difference",
            "beta": float(self.beta),
            "dt": float(self.dt),
            "quantum_mass": float(self.quantum_mass),
            "bath_masses": self.bath_masses.tolist(),
            "spring_frequency": float(self.spring_frequency),
            "physical_converged": False,
            "exact_scope": "conditional quantum subsystem along an RPMD trajectory",
            "rate_claim": "no generally exact flux-side rate",
        }
        # Validate callbacks before mutating any state in the first step.
        self._potential(self._centroid())
        self._bath_force(self.positions)
        self._quantum_gradient(self._centroid())

    def _norm(self, psi: np.ndarray) -> float:
        return float(self.dx * np.sum(np.abs(psi) ** 2))

    def _centroid(self) -> np.ndarray:
        return np.mean(self.positions, axis=0)

    def _potential(self, centroid: np.ndarray) -> np.ndarray:
        value = np.asarray(self.quantum_potential(self.grid, np.asarray(centroid, dtype=float)))
        if value.shape != self.grid.shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("quantum_potential must return a finite real vector with shape grid")
        return np.array(value, dtype=float, copy=False)

    def _quantum_gradient(self, centroid: np.ndarray) -> np.ndarray:
        if self.quantum_gradient is not None:
            value = np.asarray(self.quantum_gradient(self.grid, np.asarray(centroid, dtype=float)))
        else:
            value = np.empty((self.nbath, len(self.grid)), dtype=float)
            for axis in range(self.nbath):
                step = self.fd_step * max(1.0, abs(float(centroid[axis])))
                plus, minus = centroid.copy(), centroid.copy()
                plus[axis] += step
                minus[axis] -= step
                value[axis] = (self._potential(plus) - self._potential(minus)) / (2.0 * step)
        if value.shape != (self.nbath, len(self.grid)) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("quantum_gradient must return finite [bath_coordinate,grid] values")
        return np.array(value, dtype=float, copy=False)

    def _bath_force(self, positions: np.ndarray) -> np.ndarray:
        if self.bath_force is None:
            return np.zeros_like(positions)
        value = np.asarray(self.bath_force(np.array(positions, dtype=float, copy=True)))
        if value.shape != positions.shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("bath_force must return a finite array with positions' shape")
        return np.array(value, dtype=float, copy=False)

    def _quantum_force(self, psi: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        probability = np.abs(psi) ** 2
        gradient = self._quantum_gradient(centroid)
        # Force is -<dV/dq>; dx is required because psi samples are not
        # normalized under the unweighted Euclidean dot product.
        return -self.dx * np.einsum("x,ax->a", probability, gradient)

    def _spring_force(self, positions: np.ndarray) -> np.ndarray:
        if self.nbeads == 1:
            return np.zeros_like(positions)
        neighbors = np.roll(positions, 1, axis=0) + np.roll(positions, -1, axis=0) - 2.0 * positions
        return self.bath_masses[None, :] * self.spring_frequency ** 2 * neighbors

    def _forces(self, positions: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centroid = np.mean(positions, axis=0)
        qforce = self._quantum_force(psi, centroid)
        # H_P has (1/P) sum_j V_bath(q_j), hence physical and quantum forces
        # are divided by P when combined with the spring force.
        force = self._spring_force(positions) + self._bath_force(positions) / self.nbeads
        force += qforce[None, :] / self.nbeads
        return force, qforce

    def _propagate_quantum(self, psi: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        potential = self._potential(centroid)
        half = np.exp(-0.5j * self.dt * potential)
        evolved = half * psi
        evolved = np.fft.ifft(np.fft.fft(evolved, norm="ortho") * self._kinetic_phase, norm="ortho")
        evolved = half * evolved
        if not np.isfinite(evolved).all():
            raise FloatingPointError("quantum wavefunction propagation produced non-finite values")
        return evolved

    def _quantum_observables(self, psi: np.ndarray, centroid: np.ndarray):
        probability = np.abs(psi) ** 2
        xmean = self.dx * np.sum(probability * self.grid)
        x2 = self.dx * np.sum(probability * self.grid ** 2)
        kinetic_psi = np.fft.ifft(np.fft.fft(psi, norm="ortho") * self._kinetic, norm="ortho")
        kinetic = self.dx * np.vdot(psi, kinetic_psi).real
        potential = self._potential(centroid)
        vmean = self.dx * np.sum(probability * potential)
        energy = float(kinetic + vmean)
        return float(xmean), float(x2 - xmean * xmean), energy

    def _bath_energy(self) -> float:
        kinetic = 0.5 * np.sum(self.momenta ** 2 / self.bath_masses[None, :])
        spring = 0.5 * np.sum(self.bath_masses[None, :] * self.spring_frequency ** 2
                               * (self.positions - np.roll(self.positions, 1, axis=0)) ** 2)
        if self.bath_potential is None:
            physical = 0.0
        else:
            physical = float(np.mean([self.bath_potential(np.array(row, dtype=float, copy=True)) for row in self.positions]))
            if not np.isfinite(physical):
                raise ValueError("bath_potential returned a non-finite value")
        return float(kinetic + spring + physical)

    def step(self) -> None:
        """Advance the coupled state by one symmetric mixed-dynamics step.

        The update is transactional: if a callback or numerical check fails,
        all phase-space, wavefunction and clock values are restored to their
        pre-step state before the exception is re-raised.
        """
        old_positions = self.positions.copy()
        old_momenta = self.momenta.copy()
        old_wavefunction = self.wavefunction.copy()
        old_time = self.time
        try:
            forces, _ = self._forces(self.positions, self.wavefunction)
            self.momenta += 0.5 * self.dt * forces
            self.positions += self.dt * self.momenta / self.bath_masses[None, :]
            midpoint = 0.5 * (old_positions.mean(axis=0) + self.positions.mean(axis=0))
            self.wavefunction = self._propagate_quantum(self.wavefunction, midpoint)
            forces, _ = self._forces(self.positions, self.wavefunction)
            self.momenta += 0.5 * self.dt * forces
            self.time += self.dt
            if not np.isfinite(self.positions).all() or not np.isfinite(self.momenta).all():
                raise FloatingPointError("RPMD propagation produced non-finite phase-space values")
            norm = self._norm(self.wavefunction)
            if not np.isfinite(norm) or abs(norm - 1.0) > 5e-8:
                raise FloatingPointError(f"quantum norm drifted to {norm:.6g}")
        except Exception:
            self.positions[...] = old_positions
            self.momenta[...] = old_momenta
            self.wavefunction = old_wavefunction
            self.time = old_time
            raise

    def _record(self, times, positions, momenta, wavefunctions, xmean, xmomentum,
                qenergy, qforce, total, norms):
        centroid = self._centroid()
        x, _, energy = self._quantum_observables(self.wavefunction, centroid)
        # p_x = <psi|(-i d/dx)|psi>, evaluated in the periodic Fourier basis.
        momentum_wave = np.fft.ifft(np.fft.fft(self.wavefunction, norm="ortho")
                                    * (2.0 * np.pi * np.fft.fftfreq(len(self.grid), d=self.dx)), norm="ortho")
        pm = self.dx * np.vdot(self.wavefunction, momentum_wave).real
        qf = self._quantum_force(self.wavefunction, centroid)
        times.append(self.time)
        positions.append(self.positions.copy())
        momenta.append(self.momenta.copy())
        wavefunctions.append(self.wavefunction.copy())
        xmean.append(x)
        xmomentum.append(float(pm))
        qenergy.append(energy)
        qforce.append(qf)
        total.append(self._bath_energy() + energy)
        norms.append(self._norm(self.wavefunction))

    def propagate(self, steps: int, *, sample_stride: int = 1) -> JointPropagationResult:
        """Advance ``steps`` and return frames at ``sample_stride`` intervals.

        The current state is included as frame zero.  The object remains at the
        final state, so repeated calls continue the same online trajectory.
        """
        steps = _integer(steps, "steps", nonnegative=True)
        stride = _integer(sample_stride, "sample_stride", positive=True)
        indices = list(range(0, steps + 1, stride))
        if indices[-1] != steps:
            indices.append(steps)
        times, positions, momenta, wavefunctions = [], [], [], []
        xmean, xmomentum, qenergy, qforce, total, norms = [], [], [], [], [], []
        relative = 0
        for index in indices:
            while relative < index:
                self.step()
                relative += 1
            self._record(times, positions, momenta, wavefunctions, xmean, xmomentum,
                         qenergy, qforce, total, norms)
        diagnostics = dict(self.diagnostics)
        diagnostics.update({
            "steps": steps,
            "sample_stride": stride,
            "final_time": float(self.time),
            "max_norm_error": float(np.max(np.abs(np.asarray(norms) - 1.0))),
            "max_force_norm": float(np.max(np.linalg.norm(np.asarray(qforce), axis=1))),
        })
        return JointPropagationResult(
            np.asarray(times), np.asarray(positions), np.asarray(momenta), np.asarray(wavefunctions),
            np.asarray(xmean), np.asarray(xmomentum), np.asarray(qenergy), np.asarray(qforce),
            np.asarray(total), np.asarray(norms), diagnostics, physical_converged=False,
        )


# Descriptive alias used by callers who prefer the longer workflow name.
JointRPMDQuantumPropagator = OnlineQuantumRPMD

__all__ = ["JointPropagationResult", "OnlineQuantumRPMD", "JointRPMDQuantumPropagator"]
