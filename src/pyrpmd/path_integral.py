"""Equilibrium Monte Carlo for primitive and Takahashi--Imada paths.

Coordinates, masses and energies use atomic units; temperature uses kelvin.
These samplers generate Boltzmann (distinguishable-nucleus) equilibrium paths,
not real-time trajectories.  The Takahashi--Imada action has fourth-order
accuracy for a *closed partition-function trace*, not arbitrary coordinate
histograms or off-diagonal density matrices.  It is deliberately not a PES
adapter and cannot be passed to an RPMD propagator.

References: Takahashi and Imada, JPSJ 53, 3765 (1984),
https://doi.org/10.1143/JPSJ.53.3765; Ceperley, RMP 67, 279 (1995),
https://doi.org/10.1103/RevModPhys.67.279.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.fft import dct, idct

from .constants import KB_AU
from .potentials.base import coordinates, validate_result
from .units import parse_quantity
from .validation import checked_integer as _integer
from .validation import finite_positive

_positive = partial(finite_positive, message='{name} must be a positive finite number')


def _masses(value):
    mass = np.asarray(value)
    if mass.dtype.kind not in "fiu" or mass.ndim != 1 or not len(mass):
        raise ValueError("masses_au must be a nonempty real vector")
    mass = np.array(mass, dtype=float, copy=True)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("masses_au must be positive and finite")
    mass.flags.writeable = False
    return mass


def _method(value):
    if not isinstance(value, str) or value not in ("primitive", "ti", "takahashi_imada"):
        raise ValueError("method must be 'primitive' or 'takahashi_imada'")
    return "takahashi_imada" if value == "ti" else value


@dataclass(frozen=True)
class ActionTerms:
    """Dimensionless closed-path action and derivatives at fixed coordinates."""

    spring: float
    potential: float
    correction: float
    beta: float
    dof_times_beads: int

    @property
    def total(self):
        return self.spring + self.potential + self.correction

    @property
    def energy(self):
        """-d(log weight)/d beta, including the free-particle normalization."""
        return (0.5 * self.dof_times_beads - self.spring + self.potential + 3 * self.correction) / self.beta

    @property
    def second_beta_derivative(self):
        """d²(action)/d beta²; needed in the heat-capacity estimator."""
        return (2 * self.spring + 6 * self.correction) / self.beta**2


class PathIntegralAction:
    """Closed, unconstrained Cartesian path action for a temperature-independent PES.

    ``masses_au`` are electron masses, ``temperature_K`` is a number in kelvin
    or an explicit unit string, and the PES receives ``[P, N, 3]`` in bohr.
    Nuclear exchange, constraints, periodic winding and temperature-dependent
    effective PESs are outside this implementation's normalization convention.
    All atoms and all three Cartesian components are sampled.
    """

    equilibrium_only = True
    topology = "closed"

    def __init__(self, potential, masses_au, temperature_K, *, method="primitive"):
        if not callable(getattr(potential, "evaluate", None)):
            raise TypeError("potential must implement evaluate(q)")
        self.potential = potential
        self.masses_au = _masses(masses_au)
        self.temperature_K = _positive(parse_quantity(temperature_K, "temperature"), "temperature_K")
        self.beta = 1 / (KB_AU * self.temperature_K)
        self.method = _method(method)

    def _coordinates(self, q):
        q = coordinates(q)
        if q.shape[1] != len(self.masses_au):
            raise ValueError("path atom count does not match masses_au")
        return q

    def evaluate_batch(self, paths):
        """Evaluate independent closed paths ``[samples, P, N, 3]`` in one PES call.

        The returned energy and second derivative include the explicit beta
        dependence of the force-squared correction.  Naive potential averages
        or fluctuations of a fictitious ring Hamiltonian are not substitutes.
        """
        q = np.asarray(paths)
        if q.dtype.kind not in "fiu" or q.ndim != 4 or q.shape[-1] != 3 or min(q.shape) < 1:
            raise ValueError("paths must be a nonempty real [samples,P,N,3] array")
        if q.shape[2] != len(self.masses_au) or not np.isfinite(q).all():
            raise ValueError("paths must be finite and match masses_au")
        q = np.asarray(q, dtype=float)
        nsamples, beads, natoms, _ = q.shape
        flat = np.ascontiguousarray(q.reshape(nsamples * beads, natoms, 3))
        result = validate_result(self.potential.evaluate(flat), flat.shape)
        delta = q - np.roll(q, 1, axis=1)
        spring = beads / (2 * self.beta) * np.sum(self.masses_au[None, None, :, None] * delta**2, axis=(1, 2, 3))
        potential = self.beta * result.energy.reshape(nsamples, beads).mean(axis=1)
        correction = np.zeros(nsamples)
        if self.method == "takahashi_imada":
            force = result.forces.reshape(q.shape)
            correction = self.beta**3 / (24 * beads**3) * np.sum(force**2 / self.masses_au[None, None, :, None], axis=(1, 2, 3))
        energy = (1.5 * natoms * beads - spring + potential + 3 * correction) / self.beta
        second = (2 * spring + 6 * correction) / self.beta**2
        values = {"spring_action": spring, "potential_action": potential,
                  "correction_action": correction, "energy_estimator": energy,
                  "second_beta_derivative": second}
        if not all(np.isfinite(x).all() for x in values.values()):
            raise ValueError("path action/derivative overflow; check temperature, masses and coordinates")
        return values

    def evaluate(self, q):
        q = self._coordinates(q)
        terms = self.evaluate_batch(q[None])
        return ActionTerms(float(terms["spring_action"][0]),
                           float(terms["potential_action"][0]),
                           float(terms["correction_action"][0]),
                           self.beta, 3 * q.shape[0] * q.shape[1])


@dataclass
class PathIntegralSamples:
    """Closed equilibrium samples and provenance; potential ownership stays with caller."""

    positions: np.ndarray
    action: PathIntegralAction
    terms: dict[str, np.ndarray]
    acceptance: dict[str, float]
    attempts: int
    burn_in: int
    stride: int


def _ring_noise(rng, shape, beta, masses):
    beads = shape[0]
    if beads == 1:
        return np.zeros(shape)
    mode = np.fft.rfft(rng.normal(size=shape), axis=0, norm="ortho")
    mode[0] = 0
    eig = 4 * np.sin(np.pi * np.arange(1, len(mode)) / beads)**2
    mode[1:] *= np.sqrt(beta / (beads * eig[:, None, None] * masses[None, :, None]))
    return np.fft.irfft(mode, n=beads, axis=0, norm="ortho")


def _sampling_options(steps, burn_in, stride, centroid_step_bohr, internal_mix):
    steps = _integer(steps, "steps")
    burn_in = _integer(burn_in, "burn_in", 0)
    stride = _integer(stride, "stride")
    if steps < stride:
        raise ValueError("steps must be >= stride to save at least one sample")
    centroid_step = _positive(centroid_step_bohr, "centroid_step_bohr")
    mix = _positive(internal_mix, "internal_mix")
    if mix > 1:
        raise ValueError("internal_mix must be in (0,1]")
    return steps, burn_in, stride, centroid_step, mix


def sample_path_integral(action, initial_positions, *, steps, burn_in=1000, stride=10,
                         centroid_step_bohr=0.1, internal_mix=0.25, seed=None):
    """Metropolis sampling with centroid and free-ring-preserving pCN moves.

    One attempted move is made per step, alternating centroid and internal-mode
    moves.  ``steps`` counts production attempts, after ``burn_in`` attempts.
    The spring contribution cancels from the pCN proposal ratio analytically;
    acceptance uses only the physical plus TI potential action.  No dynamical
    time or physical momentum is assigned to these configurations.
    """
    if not isinstance(action, PathIntegralAction) or action.topology != "closed":
        raise TypeError("action must be a closed PathIntegralAction")
    steps, burn_in, stride, width, mix = _sampling_options(steps, burn_in, stride, centroid_step_bohr, internal_mix)
    q = action._coordinates(initial_positions).copy()
    current = action.evaluate(q)
    rng = np.random.default_rng(seed)
    saved, saved_terms = [], []
    accepted = {"centroid": 0, "internal": 0}
    attempted = {"centroid": 0, "internal": 0}
    shrink = math.sqrt(1 - mix**2)
    for step in range(burn_in + steps):
        centroid_move = step % 2 == 0 or len(q) == 1
        kind = "centroid" if centroid_move else "internal"
        if centroid_move:
            proposal = q + rng.normal(scale=width, size=(1, q.shape[1], 3))
        else:
            center = q.mean(axis=0, keepdims=True)
            proposal = center + shrink * (q - center) + mix * _ring_noise(rng, q.shape, action.beta, action.masses_au)
        candidate = action.evaluate(proposal)
        log_ratio = current.potential + current.correction - candidate.potential - candidate.correction
        attempted[kind] += 1
        if math.log(max(rng.random(), np.finfo(float).tiny)) < min(0.0, log_ratio):
            q, current = proposal, candidate
            accepted[kind] += 1
        if step >= burn_in and (step - burn_in + 1) % stride == 0:
            saved.append(q.copy())
            saved_terms.append(current)
    terms = {"spring_action": np.array([x.spring for x in saved_terms]),
             "potential_action": np.array([x.potential for x in saved_terms]),
             "correction_action": np.array([x.correction for x in saved_terms]),
             "energy_estimator": np.array([x.energy for x in saved_terms]),
             "second_beta_derivative": np.array([x.second_beta_derivative for x in saved_terms])}
    return PathIntegralSamples(np.array(saved), action, terms,
                               {key: accepted[key] / attempted[key] if attempted[key] else 0.0 for key in attempted},
                               steps, burn_in, stride)


class OpenPathAction:
    """Primitive action with one Cartesian component of one atom opened.

    A path has ``[P+1,N,3]`` entries (P imaginary-time links).  Its first and
    last configurations agree except for ``[atom,axis]``.  This constrained
    off-diagonal density matrix yields the *one-dimensional marginal* of that
    atom's momentum along that axis.  Opening all three coordinates and using
    only one displacement component would give a different observable.
    """

    equilibrium_only = True
    topology = "one_component_open"
    method = "primitive"

    def __init__(self, potential, masses_au, temperature_K, *, atom=0, axis=0, method="primitive"):
        if method != "primitive":
            raise ValueError("open paths require primitive: TI is fourth order only for closed traces")
        closed = PathIntegralAction(potential, masses_au, temperature_K)
        self.potential, self.masses_au = closed.potential, closed.masses_au
        self.temperature_K, self.beta = closed.temperature_K, closed.beta
        self.atom = _integer(atom, "atom", 0)
        self.axis = _integer(axis, "axis", 0)
        if self.atom >= len(self.masses_au) or self.axis > 2:
            raise ValueError("open component must have atom in range and axis in {0,1,2}")
        self._closed_mask = np.ones((len(self.masses_au), 3), dtype=bool)
        self._closed_mask[self.atom, self.axis] = False

    def _coordinates(self, q):
        q = coordinates(q)
        if len(q) < 2 or q.shape[1] != len(self.masses_au):
            raise ValueError("open paths require [P+1,N,3], P>=1, matching masses")
        if not np.array_equal(q[0][self._closed_mask], q[-1][self._closed_mask]):
            raise ValueError("all endpoint components except the declared open component must coincide")
        return q

    def evaluate(self, q):
        q = self._coordinates(q)
        beads = len(q) - 1
        result = validate_result(self.potential.evaluate(q), q.shape)
        spring = beads / (2 * self.beta) * np.sum(self.masses_au[None, :, None] * np.diff(q, axis=0)**2)
        potential = self.beta / beads * (result.energy.sum() - 0.5 * (result.energy[0] + result.energy[-1]))
        if not np.isfinite(spring + potential):
            raise ValueError("open-path action overflow")
        return float(spring), float(potential)


@dataclass
class OpenPathSamples:
    positions: np.ndarray
    displacements: np.ndarray
    action: OpenPathAction
    acceptance: dict[str, float]
    attempts: int
    burn_in: int
    stride: int


def sample_open_path(action, initial_positions, *, steps, burn_in=1000, stride=10,
                     centroid_step_bohr=0.1, internal_mix=0.25, seed=None):
    """Sample a one-component-open density matrix using free-chain pCN moves.

    The opened coordinate uses the Neumann-chain cosine modes, including a
    freely translated centroid; closed components use periodic ring modes.
    Both proposal kernels preserve their free kinetic Gaussian distributions.
    """
    if not isinstance(action, OpenPathAction):
        raise TypeError("action must be an OpenPathAction")
    steps, burn_in, stride, width, mix = _sampling_options(steps, burn_in, stride, centroid_step_bohr, internal_mix)
    q = action._coordinates(initial_positions).copy()
    current = action.evaluate(q)
    beads, nopen = len(q) - 1, len(q)
    rng = np.random.default_rng(seed)
    shrink = math.sqrt(1 - mix**2)
    eig = 4 * np.sin(np.pi * np.arange(1, nopen) / (2 * nopen))**2
    open_scale = np.sqrt(action.beta / (beads * action.masses_au[action.atom] * eig))
    accepted = {"centroid": 0, "internal": 0}
    attempted = {"centroid": 0, "internal": 0}
    saved = []
    for step in range(burn_in + steps):
        kind = "centroid" if step % 2 == 0 else "internal"
        if kind == "centroid":
            proposal = q + rng.normal(scale=width, size=(1, q.shape[1], 3))
        else:
            center = q[:-1].mean(axis=0, keepdims=True)
            proposal = q.copy()
            proposal[:-1] = center + shrink * (q[:-1] - center) + mix * _ring_noise(rng, q[:-1].shape, action.beta, action.masses_au)
            proposal[-1] = proposal[0]
            opened = q[:, action.atom, action.axis]
            noise = dct(rng.normal(size=nopen), type=2, norm="ortho")
            noise[0] = 0
            noise[1:] *= open_scale
            proposal[:, action.atom, action.axis] = opened.mean() + shrink * (opened - opened.mean()) + mix * idct(noise, type=2, norm="ortho")
        candidate = action.evaluate(proposal)
        attempted[kind] += 1
        if math.log(max(rng.random(), np.finfo(float).tiny)) < min(0.0, current[1] - candidate[1]):
            q, current = proposal, candidate
            accepted[kind] += 1
        if step >= burn_in and (step - burn_in + 1) % stride == 0:
            saved.append(q.copy())
    positions = np.array(saved)
    displacements = positions[:, -1, action.atom, action.axis] - positions[:, 0, action.atom, action.axis]
    return OpenPathSamples(positions, displacements, action,
                           {key: accepted[key] / attempted[key] if attempted[key] else 0.0 for key in attempted}, steps, burn_in, stride)
