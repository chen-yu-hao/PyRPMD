"""Original two-state kinetically constrained RPMD model dynamics.

Menzeleev, Bell and Miller, JCP 140, 064103 (2014),
https://arxiv.org/abs/1401.3033, Eqs. 7, 19, 25--32 and Appendix B.
The original primitive transfer matrix, finite-b restraint and centroid
kinetic constraint are explicit.  Calibration and convergence parameters must
be supplied; this is not the revised 2016 or 2026 KC-RPMD formulation.
"""

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp

from ._matrix_path import ScaledProduct, integrator_signature, real_diabatic
from .dynamics import CayleyIntegrator, State
from .nonadiabatic import _positive_integer
from .potentials.base import PotentialError, coordinates


@dataclass(frozen=True)
class KCCalibration:
    """Crossing-ensemble <|grad w|>_c and <|K|^-1>_c in atomic units.

The ensemble is delta(w)*K**2*exp(-beta*V0), as in the original paper.
These are model-specific measured/analytic inputs, not fit-free constants.
"""
    mean_gap_gradient: float
    mean_inverse_coupling: float

    def __post_init__(self):
        for key in ("mean_gap_gradient", "mean_inverse_coupling"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError("KC calibration averages must be finite positive scalars")
            object.__setattr__(self, key, float(value))

    def auxiliary_mass(self, beta):
        """Physical y mass in the mean-bead Hamiltonian, with hbar=1."""
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        with np.errstate(over="ignore", invalid="ignore"):
            value = (np.float64(beta)**3 / (2*np.pi**3)
                     * (self.mean_gap_gradient / self.mean_inverse_coupling)**2)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("KC auxiliary mass is not representable")
        return float(value)


@dataclass(frozen=True)
class KCRPMDResult:
    potential_energy: float
    nuclear_forces: np.ndarray
    auxiliary_force: float
    log_electronic_weight: float
    sector_weights: np.ndarray


class KCRPMDPotential:
    """Whole-ring KC potential in the real-mass, sum-bead convention.

Only a real two-state diabatic basis with strictly positive coupling is
supported.  A constant electronic basis sign flip can change negative constant
coupling to positive.  Sign changes and complex coupling are rejected.
"""

    def __init__(self, potential, beta, nbeads, *, calibration, a, b):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.nbeads = _positive_integer(nbeads, "nbeads")
        if self.nbeads < 2:
            raise ValueError("primitive KC-RPMD requires at least two beads for a kink pair")
        if not isinstance(calibration, KCCalibration):
            raise ValueError("calibration must be KCCalibration")
        if not np.isfinite(a) or not 0 < a <= 1:
            raise ValueError("original KC constraint parameter a must satisfy 0 < a <= 1")
        if not np.isfinite(b) or b <= 0:
            raise ValueError("KC restraint parameter b must be finite and positive")
        self.potential, self.beta = potential, float(beta)
        self.calibration, self.a, self.b = calibration, float(a), float(b)

    def _kink_weight(self, result):
        """Positive augmented transfer path; no subtraction of nearly equal traces."""
        matrix, gradient = result.matrix, result.gradients
        tau = self.beta / self.nbeads
        coupling = matrix[:, 0, 1]
        if np.any(coupling <= 0):
            raise PotentialError("original KC-RPMD requires strictly positive diabatic coupling")
        if np.any(tau * coupling >= 1):
            raise PotentialError("KC primitive transfer requires beta*K/P < 1; increase nbeads")
        common = 0.5 * (matrix[:, 0, 0] + matrix[:, 1, 1])
        gcommon = 0.5 * (gradient[:, 0, 0] + gradient[:, 1, 1])
        diagonal = np.stack((matrix[:, 0, 0] - common, matrix[:, 1, 1] - common), axis=1)
        gdiag = np.stack((gradient[:, 0, 0] - gcommon,
                          gradient[:, 1, 1] - gcommon), axis=1)
        log_scales = np.max(-tau * diagonal, axis=1)
        row = np.exp(-tau * diagonal - log_scales[:, None])
        transfer = np.zeros((self.nbeads, 2, 2))
        transfer[:, 0, 0], transfer[:, 1, 1] = row[:, 0], row[:, 1]
        transfer[:, 0, 1] = tau * coupling * row[:, 0]
        transfer[:, 1, 0] = tau * coupling * row[:, 1]
        derivative = -tau * gdiag[:, :, None] * transfer[..., None, None]
        derivative[:, 0, 1] += tau * row[:, 0, None, None] * gradient[:, 0, 1]
        derivative[:, 1, 0] += tau * row[:, 1, None, None] * gradient[:, 0, 1]
        # Flags distinguish paths that have/have not switched electronic state.
        # The upper-right product block sums all switched paths.  Its diagonal
        # closes the electronic path and therefore contains >=2 switches.
        augmented = np.zeros((self.nbeads, 4, 4))
        d_augmented = np.zeros((self.nbeads, 4, 4, *gradient.shape[3:]))
        for index in (0, 1):
            augmented[:, index, index] = transfer[:, index, index]
            d_augmented[:, index, index] = derivative[:, index, index]
            other = 1 - index
            augmented[:, index, other + 2] = transfer[:, index, other]
            d_augmented[:, index, other + 2] = derivative[:, index, other]
        augmented[:, 2:, 2:] = transfer
        d_augmented[:, 2:, 2:] = derivative
        product = ScaledProduct(augmented)
        kink = float(product.matrix[0, 2] + product.matrix[1, 3])
        if not np.isfinite(kink) or kink <= 0:
            raise PotentialError("KC kink weight is numerically unresolved")
        dlog_kink = np.empty_like(gcommon)
        for bead in range(self.nbeads):
            insertion = product.insertion(bead, d_augmented[bead])
            dlog_kink[bead] = (insertion[0, 2] + insertion[1, 3]) / kink
        log_kink = np.log(kink) + product.log_scale + log_scales.sum()
        pure_logs = -tau * diagonal.sum(axis=0)
        pure_gradients = -tau * gdiag
        return common, gcommon, float(log_kink), dlog_kink, pure_logs, pure_gradients

    def evaluate(self, q, y):
        q = coordinates(q)
        if len(q) != self.nbeads:
            raise ValueError("KC-RPMD potential requires the entire configured ring")
        if isinstance(y, bool) or not np.isscalar(y) or not np.isfinite(y):
            raise ValueError("y must be a finite real scalar")
        result = real_diabatic(self.potential, q, 2)
        common, gcommon, log_kink, dlog_kink, pure_logs, pure_gradients = self._kink_weight(result)
        center = real_diabatic(self.potential, q.mean(axis=0, keepdims=True), 2)
        k = float(center.matrix[0, 0, 1])
        if k <= 0:
            raise PotentialError("KC centroid coupling must be strictly positive")
        gap = float(center.matrix[0, 0, 0] - center.matrix[0, 1, 1])
        w = gap / k
        dw = ((center.gradients[0, 0, 0] - center.gradients[0, 1, 1]) / k
              - gap / k**2 * center.gradients[0, 0, 1]) / self.nbeads
        penalty_log = 0.5 * np.log(self.a / np.pi) + np.log(self.calibration.mean_gap_gradient) - self.a * w*w
        dpenalty = -2 * self.a * w * dw
        # Original finite-b square-well restraint, evaluated in log space.
        sectors = np.array([-1.0, 0.0, 1.0])
        displacement = float(y) - sectors
        argument = 2 * self.b * (np.abs(displacement) - 0.5)
        log_restraint = -np.logaddexp(0, argument)
        dlog_restraint = -2 * self.b * np.sign(displacement) * expit(argument)
        sector_logs = np.array([pure_logs[0], log_kink + penalty_log, pure_logs[1]]) + log_restraint
        log_total = float(logsumexp(sector_logs))
        weights = np.exp(sector_logs - log_total)
        dlog_total = (weights[0] * pure_gradients[:, 0]
                      + weights[1] * (dlog_kink + dpenalty)
                      + weights[2] * pure_gradients[:, 1])
        coefficient = self.nbeads / self.beta
        energy = float(common.sum() - coefficient * log_total)
        forces = -gcommon + coefficient * dlog_total
        force_y = float(coefficient * np.dot(weights, dlog_restraint))
        log_weight = log_total - self.beta / self.nbeads * float(common.sum())
        if (not np.isfinite(energy) or not np.isfinite(forces).all()
                or not np.isfinite(force_y) or not np.isfinite(log_weight)):
            raise PotentialError("KC-RPMD potential or gradient overflow")
        return KCRPMDResult(energy, forces, force_y, float(log_weight), weights)


@dataclass
class KCRPMDState(State):
    y: float = 0.0
    p_y: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        for key in ("y", "p_y"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError("y and p_y must be finite real scalars")
            setattr(self, key, float(value))

    def copy(self):
        return KCRPMDState(self.q, self.p, self.step, self.time, self.y, self.p_y)


class KCRPMDDynamics:
    """Reversible force/free-ring splitting of the original KC Hamiltonian.

``p_y`` is the sum-convention momentum P*m_y*dy/dt.  Thus the auxiliary
mass used by this integrator is P times the published mean-convention mass.
Energy and all nuclear momenta use PyRPMD's ordinary sum-bead convention.
"""

    def __init__(self, potential, masses, beta, nbeads, dt, *, calibration, a, b,
                 native=True, integrator="cayley-bcb", threads=1):
        self.potential = KCRPMDPotential(potential, beta, nbeads,
                                        calibration=calibration, a=a, b=b)
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt,
                                          native=native, scheme=integrator, threads=threads)
        self._integrator_signature = integrator_signature(self.integrator)
        self.auxiliary_mass = nbeads * calibration.auxiliary_mass(beta)

    def _check(self, state):
        if not isinstance(state, KCRPMDState):
            raise ValueError("KC-RPMD dynamics requires KCRPMDState")
        self.integrator._check(state.q, state.p)
        if (not np.isfinite(state.y) or not np.isfinite(state.p_y)
                or isinstance(state.step, bool) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid auxiliary state or clock")
        if (integrator_signature(self.integrator) != self._integrator_signature
                or self.potential.beta != self.integrator.beta
                or self.potential.nbeads != self.integrator.nbeads
                or self.auxiliary_mass != self.integrator.nbeads * self.potential.calibration.auxiliary_mass(self.integrator.beta)):
            raise ValueError("KC-RPMD dynamics parameters are immutable")

    def step(self, state):
        self._check(state)
        trial = state.copy()
        dt = self.integrator.dt
        before = self.potential.evaluate(trial.q, trial.y)
        trial.p += 0.5 * dt * before.nuclear_forces
        trial.p_y += 0.5 * dt * before.auxiliary_force
        self.integrator.free_step(trial.q, trial.p)
        trial.y += dt * trial.p_y / self.auxiliary_mass
        after = self.potential.evaluate(trial.q, trial.y)
        trial.p += 0.5 * dt * after.nuclear_forces
        trial.p_y += 0.5 * dt * after.auxiliary_force
        trial.step += 1
        trial.time += dt
        self._check(trial)
        state.q[:], state.p[:] = trial.q, trial.p
        state.y, state.p_y = trial.y, trial.p_y
        state.step, state.time = trial.step, trial.time
        return state

    def _spring_energy(self, state):
        difference = state.q - np.roll(state.q, 1, axis=0)
        return 0.5 * (self.integrator.nbeads / self.integrator.beta)**2 * np.sum(
            self.integrator.masses[None, :, None] * difference**2)

    def energy(self, state):
        self._check(state)
        kinetic = np.sum(state.p**2 / (2 * self.integrator.masses[None, :, None]))
        kinetic += state.p_y**2 / (2 * self.auxiliary_mass)
        return float(kinetic + self._spring_energy(state)
                     + self.potential.evaluate(state.q, state.y).potential_energy)

    def log_sampling_density(self, state):
        """Coordinate/y equilibrium log density, omitting Gaussian momenta."""
        self._check(state)
        return -self.integrator.beta / self.integrator.nbeads * (
            self._spring_energy(state) + self.potential.evaluate(state.q, state.y).potential_energy)


class KCRPMDSampler:
    """Metropolis sampling of the specified finite-P, finite-b KC distribution."""

    def __init__(self, dynamics, *, position_step, auxiliary_step, seed=0):
        if not isinstance(dynamics, KCRPMDDynamics):
            raise ValueError("dynamics must be KCRPMDDynamics")
        for value in (position_step, auxiliary_step):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("proposal steps must be finite and positive")
        self.dynamics = dynamics
        self.position_step, self.auxiliary_step = float(position_step), float(auxiliary_step)
        self.rng = np.random.default_rng(seed)
        self.attempted = self.accepted = 0

    def step(self, state):
        self.dynamics._check(state)
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            old = self.dynamics.log_sampling_density(state)
            trial = state.copy()
            trial.q += self.rng.normal(scale=self.position_step, size=state.q.shape)
            trial.y += self.rng.normal(scale=self.auxiliary_step)
            new = self.dynamics.log_sampling_density(trial)
            accepted = np.log(self.rng.random()) < min(0.0, new - old)
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        self.attempted += 1
        if accepted:
            state.q[:] = trial.q
            state.y = trial.y
            self.accepted += 1
        return bool(accepted)

    def draw_momenta(self, state):
        self.dynamics._check(state)
        integrator = self.dynamics.integrator
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            thermal_scale = np.sqrt(integrator.nbeads / integrator.beta)
            momentum = self.rng.normal(size=state.p.shape) * np.sqrt(integrator.masses[None, :, None]) * thermal_scale
            auxiliary = float(self.rng.normal() * np.sqrt(self.dynamics.auxiliary_mass) * thermal_scale)
            if not np.isfinite(momentum).all() or not np.isfinite(auxiliary):
                raise ValueError("sampled KC momenta overflow")
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        state.p[:] = momentum
        state.p_y = auxiliary
        return state

