"""Force-based equilibrium actions for the main :class:`Dynamics` engine.

These are sampling Hamiltonians, not Born--Oppenheimer potentials and not
real-time RPMD approximations.  They use the engine's sum-beads/beta-over-P
convention.  All arrays and imaginary times are in atomic units.

References: Kapil et al., JCP 145, 234103 (2016), doi:10.1063/1.4971438;
Feldman and Hirshberg, JCP 159, 154107 (2023), doi:10.1063/5.0173749;
Sarsa et al., JCP 113, 1366 (2000), doi:10.1063/1.481926.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp

from .path_integral import _integer, _masses, _positive
from .potentials.base import Potential, PotentialError, PESResult, coordinates, validate_result
from .potentials.contraction import _component


def _ring_terms(q, masses, beta):
    delta = q - np.roll(q, -1, axis=0)
    coefficient = masses[None, :, None] * (len(q) / beta) ** 2
    energy = 0.5 * float(np.sum(coefficient * delta * delta))
    force = -coefficient * (2 * q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0))
    return energy, force


class _SamplingPotential(Potential):
    """Borrow a physical provider; its lifecycle remains with the caller."""

    equilibrium_only = True

    def __init__(self, potential, masses, beta, nbeads):
        self.potential = _component(potential, "physical potential")
        self.masses = _masses(masses)
        self.beta = _positive(beta, "beta")
        self.nbeads = _integer(nbeads, "nbeads")
        self.metadata = {
            "provider": type(self).__name__, "batch_independent": False,
            "ring_polymer_coupled": True, "trajectory_batch_safe": False,
            "equilibrium_only": True, "virial": False,
            "energy_convention": "sum_beads_beta_over_P_sampling_action",
            "thread_safety": "context_serial",
        }

    def _q(self, positions):
        q = coordinates(positions)
        if q.shape != (self.nbeads, len(self.masses), 3):
            raise ValueError("sampling potential requires one complete ring with matching P,N")
        return q


class SuzukiChinPotential(_SamplingPotential):
    """Alternating Suzuki--Chin potential with symmetric finite-difference forces.

    Requires an even bead count.  ``fd_step`` is the RMS Cartesian displacement
    in bohr, normalized independently on each force-bearing bead.  Physical
    observables use even beads for alpha=0; averaging the modified potential
    is not an energy estimator.  :meth:`estimators` supplies thermodynamic
    energy, pressure and scaled-coordinate heat-capacity ingredients for all
    alpha.  When the physical provider supplies strain virials, the effective
    action virial is differentiated with the same displaced force calls.
    """

    def __init__(self, potential, masses, beta, nbeads, *, alpha=0.0, fd_step=1e-4):
        super().__init__(potential, masses, beta, nbeads)
        if self.nbeads % 2:
            raise ValueError("Suzuki--Chin requires an even bead count")
        if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
            raise ValueError("alpha must be finite and in [0,1]")
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1]")
        self.fd_step = _positive(fd_step, "fd_step")
        self.weights = np.where(np.arange(nbeads) % 2 == 0, 2 / 3, 4 / 3)
        d = np.where(np.arange(nbeads) % 2 == 0, self.alpha / 6, (1 - self.alpha) / 12)
        self.coefficients = self.weights * d * (self.beta / self.nbeads) ** 2
        self.weights.flags.writeable = self.coefficients.flags.writeable = False
        self.metadata.pop("virial", None)
        self.metadata.update({
            "action": "suzuki_chin",
            "virial_requires": "physical_provider_configurational_virials",
            "virial_convention": "negative_affine_strain_derivative_of_sampling_potential",
            "pressure_estimator": "thermodynamic_centroid_virial",
        })

    def set_cell(self, cell):
        """Forward a validated periodic cell; physical-provider ownership stays external."""
        from .geometry import validate_cell_pbc
        cell, _ = validate_cell_pbc(cell, [True, True, True])
        if cell is None:
            raise ValueError("Suzuki--Chin cell must be a finite nonsingular 3D cell")
        setter = getattr(self.potential, "set_cell", None)
        if not callable(setter):
            raise PotentialError("Suzuki--Chin NPT requires physical set_cell(cell_bohr) support")
        setter(cell.copy())
        self.cell = cell.copy()

    def evaluate(self, positions):
        q = self._q(positions)
        return self._evaluate_sc(q)[0]

    def _evaluate_sc(self, q):
        """Evaluate one or more whole rings as independent physical frames."""
        count = len(q) // self.nbeads
        if count < 1 or len(q) != count * self.nbeads:
            raise ValueError("Suzuki--Chin evaluation requires complete rings")
        weights = np.tile(self.weights, count)
        coefficients = np.tile(self.coefficients, count)
        result = validate_result(self.potential.evaluate(q), q.shape)
        physical_energy, physical_force = result.energy.copy(), result.forces.copy()
        physical_virial = None if result.virial is None else result.virial.copy()
        acceleration = physical_force / self.masses[None, :, None]
        correction = coefficients * np.sum(physical_force * acceleration, axis=(1, 2))
        energy = weights * physical_energy + correction
        forces = weights[:, None, None] * physical_force
        virial = None if physical_virial is None else weights[:, None, None] * physical_virial
        rms = np.sqrt(np.mean(acceleration * acceleration, axis=(1, 2)))
        active = (coefficients != 0) & (rms > 0)
        if np.any(active):
            direction = acceleration[active]
            h = self.fd_step / rms[active]
            plus = q[active] + h[:, None, None] * direction
            minus = q[active] - h[:, None, None] * direction
            rp = validate_result(self.potential.evaluate(plus), plus.shape)
            fp = rp.forces.copy()
            wp = None if rp.virial is None else rp.virial.copy()
            rm = validate_result(self.potential.evaluate(minus), minus.shape)
            fm = rm.forces
            # dF/dq @ (F/m) = - Hess(V) @ (F/m).
            projected = (fp - fm) / (2 * h[:, None, None])
            forces[active] -= 2 * coefficients[active, None, None] * projected
            if virial is not None:
                if wp is None or rm.virial is None:
                    raise PotentialError("Suzuki--Chin displaced physical evaluations must preserve virial support")
                # W_G = -d(sum F^2/m)/d(strain) = 2*(F tensor F/m - D_(F/m) W).
                # The tensor term differentiates Cartesian vector components;
                # omitting it gives a wrong pressure even for a harmonic PES.
                projected_virial = (wp - rm.virial) / (2 * h[:, None, None])
                force_tensor = np.einsum("bna,bnc->bac", physical_force[active], direction)
                virial[active] += 2 * coefficients[active, None, None] * (force_tensor - projected_virial)
        effective = validate_result(PESResult(energy, forces, virial=virial), q.shape)
        physical = PESResult(physical_energy, physical_force, virial=physical_virial)
        return effective, physical, correction

    def estimators(self, positions, *, volume=None, finite_difference_step=1e-3,
                   require_pressure=True):
        """Physical SC observables and heat-capacity curvature for one ring.

        ``volume`` is in bohr^3; pressure requires a physical provider virial.
        Set ``require_pressure=False`` for NVT energy/heat-capacity analysis
        when a periodic physical provider does not supply virials.
        The beta derivative holds the cell fixed.  Heat capacities require
        an ensemble of these ingredients, not the fluctuation of the sampling
        Hamiltonian; use ``suzuki_chin_thermodynamics`` to combine samples.
        """
        from .suzuki_chin import suzuki_chin_estimators
        return suzuki_chin_estimators(self, positions, volume=volume,
                                      finite_difference_step=finite_difference_step,
                                      require_pressure=require_pressure)

    def estimators_batch(self, positions, *, volumes=None, finite_difference_step=1e-3,
                         require_pressure=True):
        """Vectorized fixed-cell estimator columns for paths shaped [sample,P,N,3]."""
        from .suzuki_chin import suzuki_chin_estimators_batch
        return suzuki_chin_estimators_batch(self, positions, volumes=volumes,
                                            finite_difference_step=finite_difference_step,
                                            require_pressure=require_pressure)


@dataclass(frozen=True)
class ExchangeResult:
    energy: float
    forces: np.ndarray
    connections: np.ndarray


def bosonic_exchange(positions, mass, beta):
    """Feldman--Hirshberg spring energy/forces in O(N² + PN) work.

    One group of equal-mass indistinguishable particles, no periodic images.
    The recurrence samples representative permutations, as in the paper; its
    pointwise potential is NOT the factorial sum over every permutation.
    ``connections[i,j]`` is the probability of joining last bead i to first j.
    The physical potential must be invariant to exchange of these particles.
    """
    q = coordinates(positions)
    mass, beta = _positive(mass, "mass"), _positive(beta, "beta")
    p, n, _ = q.shape
    coefficient = mass * (p / beta) ** 2
    beta_p = beta / p
    interior_delta = q[1:] - q[:-1]
    interior = 0.5 * coefficient * float(np.sum(interior_delta ** 2))
    # Internal springs occur in every representative permutation.  Removing
    # their common offset before log recurrences prevents cancellation at low T.
    link = 0.5 * coefficient * np.sum((q[-1, :, None] - q[0, None, :]) ** 2, axis=2)
    adjacent = np.concatenate(([0.0], np.cumsum(np.diag(link, k=1))))
    cycle = np.zeros((n, n))
    for end in range(n):
        starts = np.arange(end + 1)
        cycle[starts, end] = adjacent[end] - adjacent[starts] + link[end, starts]
    prefix = np.zeros(n + 1)
    for end in range(1, n + 1):
        prefix[end] = logsumexp(prefix[:end] - beta_p * cycle[:end, end - 1]) - math.log(end)
    suffix = np.zeros(n + 1)
    log_indices = np.log(np.arange(1, n + 1))
    for start in range(n - 1, -1, -1):
        suffix[start] = logsumexp(-beta_p * cycle[start, start:] + suffix[start + 1:] - log_indices[start:])
    connections = np.zeros((n, n))
    for end in range(n):
        starts = np.arange(end + 1)
        connections[end, starts] = np.exp(prefix[starts] + suffix[end + 1]
            - beta_p * cycle[starts, end] - math.log(end + 1) - prefix[n])
        if end + 1 < n:
            exponent = prefix[end + 1] + suffix[end + 1] - prefix[n]
            connections[end, end + 1] = -np.expm1(min(0.0, exponent))
    force = np.zeros_like(q)
    force[:-1] += coefficient * interior_delta
    force[1:] -= coefficient * interior_delta
    force[-1] -= coefficient * (q[-1] - connections @ q[0])
    force[0] -= coefficient * (q[0] - connections.T @ q[-1])
    energy = interior - float(prefix[n]) / beta_p
    if not math.isfinite(energy) or not np.isfinite(force).all():
        raise ValueError("bosonic spring overflow; check mass, beta and coordinates")
    return ExchangeResult(energy, force, connections)


class BosonicExchangePotential(_SamplingPotential):
    """Replace the main engine's distinguishable springs by bosonic springs.

    ``groups`` are disjoint lists of identical equal-mass particle indices.
    Other particles remain distinguishable.  The correction subtracts the
    ordinary ring springs exactly once; Dynamics adds those springs as usual.
    No minimum-image shortcut is used: periodic exchange needs winding sums
    and is deliberately rejected by the input workflow.
    """

    def __init__(self, potential, masses, beta, nbeads, *, groups=None):
        super().__init__(potential, masses, beta, nbeads)
        if groups is None:
            groups = [list(range(len(self.masses)))]
        if not isinstance(groups, (list, tuple)) or not groups:
            raise ValueError("groups must be a nonempty sequence")
        self.groups, seen = [], set()
        for group in groups:
            values = list(group)
            if not values or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                    or not 0 <= i < len(self.masses) for i in values):
                raise ValueError("boson groups must contain in-range integer indices")
            if len(set(values)) != len(values) or seen.intersection(values):
                raise ValueError("boson groups must be disjoint without repeated particles")
            ids = np.asarray(values, dtype=int)
            if not np.all(self.masses[ids] == self.masses[ids[0]]):
                raise ValueError("each boson group requires exactly equal masses")
            self.groups.append(ids)
            seen.update(values)

    def evaluate(self, positions):
        q = self._q(positions)
        result = validate_result(self.potential.evaluate(q), q.shape)
        energy, force = result.energy.copy(), result.forces.copy()
        for ids in self.groups:
            ring, ordinary = _ring_terms(q[:, ids], self.masses[ids], self.beta)
            exchanged = bosonic_exchange(q[:, ids], self.masses[ids[0]], self.beta)
            energy += (exchanged.energy - ring) / self.nbeads
            force[:, ids] += exchanged.forces - ordinary
        return validate_result(PESResult(energy, force), q.shape)


class GaussianTrialWavefunction:
    """Positive trial amplitude exp[-sum_i alpha_i |q_i-center_i|² / 2]."""

    def __init__(self, alpha, center=0.0):
        self.alpha = np.asarray(alpha, dtype=float)
        self.center = np.asarray(center, dtype=float)
        if (not np.isfinite(self.alpha).all() or np.any(self.alpha <= 0)
                or not np.isfinite(self.center).all()):
            raise ValueError("trial alpha must be positive; alpha and center must be finite")

    def evaluate(self, positions):
        q = coordinates(positions)
        alpha = self.alpha[:, None] if self.alpha.ndim == 1 else self.alpha
        displacement = q - self.center
        gradient = np.broadcast_to(-alpha * displacement, q.shape).copy()
        log_amplitude = -0.5 * np.sum(alpha * displacement ** 2, axis=(1, 2))
        return log_amplitude, gradient


class PIGSPotential(_SamplingPotential):
    """Primitive open-chain ground-state projection with positive endpoint trial.

    For L links and P=L+1 beads, tau=projection_time/L and the auxiliary
    engine beta=P*tau.  The density is Psi_T(q0) exp[-S_primitive] Psi_T(qL).
    Its midpoint has the pure projected distribution; endpoints do not.
    ``trial.evaluate([B,N,3])`` returns log amplitude[B], its gradient[B,N,3].
    """

    def __init__(self, potential, masses, nbeads, projection_time, trial):
        nbeads = _integer(nbeads, "nbeads", 3)
        if nbeads % 2 != 1:
            raise ValueError("PIGS requires odd nbeads (an even number of links)")
        self.projection_time = _positive(projection_time, "projection_time")
        self.tau = self.projection_time / (nbeads - 1)
        super().__init__(potential, masses, nbeads * self.tau, nbeads)
        if not callable(getattr(trial, "evaluate", None)):
            raise TypeError("trial must implement evaluate returning log amplitude and gradient")
        self.trial = trial
        self.weights = np.ones(nbeads)
        self.weights[[0, -1]] = 0.5
        self.metadata["topology"] = "fully_open_ground_state_projection"

    def evaluate(self, positions):
        q = self._q(positions)
        physical = validate_result(self.potential.evaluate(q), q.shape)
        energy = self.weights * physical.energy
        force = self.weights[:, None, None] * physical.forces
        log_psi, gradient = self.trial.evaluate(q[[0, -1]])
        log_psi, gradient = np.asarray(log_psi), np.asarray(gradient)
        if (log_psi.dtype.kind not in "fiu" or gradient.dtype.kind not in "fiu"
                or log_psi.shape != (2,) or gradient.shape != (2, q.shape[1], 3)
                or not np.isfinite(log_psi).all() or not np.isfinite(gradient).all()):
            raise ValueError("trial must return finite log amplitude[2], gradient[2,N,3]")
        energy[[0, -1]] -= log_psi / self.tau
        force[[0, -1]] += gradient / self.tau
        # Remove only the closing link from the engine's periodic spring.
        delta = q[-1] - q[0]
        coefficient = self.masses[:, None] / self.tau ** 2
        closing = 0.5 * float(np.sum(coefficient * delta ** 2))
        energy -= closing / self.nbeads
        force[-1] += coefficient * delta
        force[0] -= coefficient * delta
        return validate_result(PESResult(energy, force), q.shape)

    def action(self, positions):
        q = self._q(positions)
        energy = float(self.evaluate(q).energy.sum()) + _ring_terms(q, self.masses, self.beta)[0]
        return self.tau * energy


__all__ = ["SuzukiChinPotential", "BosonicExchangePotential", "bosonic_exchange",
           "ExchangeResult", "GaussianTrialWavefunction", "PIGSPotential"]
