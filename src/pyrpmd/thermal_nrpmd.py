"""SEO-projected thermal NRPMD sampling and finite-time rate estimators.

This module implements the real-matrix equilibrium distribution of Richardson
and Thoss, JCP 139, 031102 (2013), Eqs. 4--14.  It is different from focused
electronic-state preparation.  The distribution is signed, and its absolute
value is sampled by Metropolis Monte Carlo.  NRPMD real-time propagation does
not in general preserve this distribution; a finite-time rate curve is not a
proof of a kinetic plateau or of physical convergence.

All quantities are atomic units.  The full physical diabatic matrix is split
smoothly into V0 = Tr(V)/S and a traceless matrix for propagation.  This split
is essential when the thermally sampled mapping norm fluctuates.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .dynamics import CayleyIntegrator
from .nonadiabatic import (
    DiabaticResult,
    NRPMDDynamics,
    NRPMDState,
    _mint,
    _positive_integer,
    validate_diabatic_result,
)
from .potentials.base import PotentialError, coordinates
from .validation import finite_positive as _finite_positive
from .validation import nonnegative_integer as _nonnegative_integer


@dataclass(frozen=True)
class _SplitMatrix:
    scalar: np.ndarray
    scalar_gradient: np.ndarray
    mapping: DiabaticResult
    half_boltzmann: np.ndarray
    log_weight_shift: float


def _split_matrix(potential, q, beta, nstates, *, boltzmann=True):
    result = validate_diabatic_result(potential.evaluate(q), q.shape, nstates)
    # The 2013 X/P-position projection is a signed distribution for real
    # matrices.  Extending it to SOC requires complex phase reweighting.
    if np.any(np.imag(result.matrix) != 0) or np.any(np.imag(result.gradients) != 0):
        raise PotentialError("thermal NRPMD sampling requires real symmetric matrices; complex phase sampling is not implemented")
    matrix = np.array(result.matrix.real, dtype=float, copy=True)
    gradients = np.array(result.gradients.real, dtype=float, copy=True)
    scalar = np.trace(matrix, axis1=1, axis2=2) / nstates
    scalar_gradient = np.einsum("biiac->bac", gradients) / nstates
    identity = np.eye(nstates)
    matrix -= scalar[:, None, None] * identity
    gradients -= scalar_gradient[:, None, None] * identity[None, :, :, None, None]
    if not boltzmann:
        # Real-time MInt diagonalizes the mapping matrix itself.  Do not
        # duplicate that eigensolve or build unused thermal exponentials.
        return _SplitMatrix(scalar, scalar_gradient, DiabaticResult(matrix, gradients),
                            np.empty((0, nstates, nstates)), 0.0)
    energies, vectors = np.linalg.eigh(matrix)
    # Shift each exponential by its minimum eigenvalue.  The omitted scalar
    # factor enters W twice per bead and is restored in logarithmic form.
    lower = energies[:, 0]
    beta_bead = beta / len(q)
    exponential = np.exp(-0.5 * beta_bead * (energies - lower[:, None]))
    half = (vectors * exponential[:, None, :]) @ vectors.swapaxes(1, 2)
    return _SplitMatrix(
        scalar, scalar_gradient, DiabaticResult(matrix, gradients), half,
        float(-beta_bead * lower.sum()),
    )


@dataclass(frozen=True)
class ThermalWeight:
    """Unnormalized coordinate/mapping log density and SEO population estimator.

    ``log_density`` excludes the independent Maxwell nuclear momenta and an
    irrelevant constant.  ``sign`` belongs to the initial statistical weight,
    and must not be recomputed along the real-time trajectory.
    """

    log_density: float
    sign: float
    electronic_populations: np.ndarray
    log_abs_mapping_weight: float


def _weight(split, q, x, y, masses, beta):
    mx = np.einsum("bij,bj->bi", split.half_boltzmann, x)
    previous_y = np.roll(y, 1, axis=0)
    left = np.einsum("bi,bi->b", previous_y, mx)
    right = np.einsum("bi,bi->b", y, mx)
    if np.any(left == 0) or np.any(right == 0):
        return ThermalWeight(-math.inf, 0.0, np.full(x.shape[1], np.nan), -math.inf)
    log_abs = float(np.log(abs(left)).sum() + np.log(abs(right)).sum() + split.log_weight_shift)
    sign = float(np.prod(np.sign(left)) * np.prod(np.sign(right)))
    # Projector insertion in the left half-link.  The denominator must use
    # y[b-1], not y[b]; consequently the population estimators sum to unity.
    population = np.mean(previous_y * mx / left[:, None], axis=0)
    dq = q - np.roll(q, -1, axis=0)
    spring = 0.5 * (len(q) / beta) ** 2 * np.sum(masses[None, :, None] * dq**2)
    gaussian = np.sum(x*x + y*y)
    log_density = log_abs - gaussian - beta / len(q) * (spring + split.scalar.sum())
    if not np.isfinite(log_density) or not np.isfinite(population).all():
        raise FloatingPointError("thermal NRPMD statistical weight overflowed")
    population.flags.writeable = False
    return ThermalWeight(float(log_density), sign, population, log_abs)


def thermal_weight(potential, state, masses, beta, nstates=None):
    """Evaluate the genuine SEO-projected signed thermal weight of a state.

    This is not ``exp(-beta/P * NRPMDDynamics.energy(state))``.  The latter
    distribution can be nonnormalizable because of inverted mapping surfaces.
    """
    if not isinstance(state, NRPMDState):
        raise ValueError("thermal_weight requires NRPMDState")
    nstates = state.mapping_q.shape[1] if nstates is None else _positive_integer(nstates, "nstates")
    checker = CayleyIntegrator(masses, beta, len(state.q), 0, native=False)
    checker._check(state.q, state.p)
    if state.mapping_q.shape != (len(state.q), nstates) or state.mapping_p.shape != state.mapping_q.shape:
        raise ValueError("mapping shape mismatch")
    for array in (state.mapping_q, state.mapping_p):
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("mapping variables must be finite real arrays")
    split = _split_matrix(potential, state.q, checker.beta, nstates)
    return _weight(split, state.q, state.mapping_q, state.mapping_p, checker.masses, checker.beta)


class ThermalNRPMDDynamics(NRPMDDynamics):
    """NRPMD with explicit scalar/traceless splitting for thermal mapping states.

    The scalar force is applied once, independently of the fluctuating mapping
    norm.  With one electronic state, nuclear motion is exactly ordinary RPMD.
    No additional electronic thermostat or population clipping is applied.
    """

    def _thermal_evaluate(self, q):
        return _split_matrix(self.potential, q, self.beta, self.nstates, boltzmann=False)

    def step(self, state):
        self._check(state)
        trial = state.copy()
        half = 0.5 * self.dt
        first = self._thermal_evaluate(trial.q)
        trial.p -= half * first.scalar_gradient
        _mint(trial, first.mapping, half)
        self.integrator.free_step(trial.q, trial.p)
        second = self._thermal_evaluate(trial.q)
        _mint(trial, second.mapping, half)
        trial.p -= half * second.scalar_gradient
        trial.step += 1
        trial.time += self.dt
        self._check(trial)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            getattr(state, key)[:] = getattr(trial, key)
        state.step, state.time = trial.step, trial.time
        return state

    def energy(self, state):
        """Sum-bead propagation Hamiltonian, not the thermal sampling density."""
        self._check(state)
        result = self._thermal_evaluate(state.q)
        z = state.mapping_q + 1j * state.mapping_p
        electronic = 0.5 * np.einsum("bi,bij,bj->", z.conj(), result.mapping.matrix, z).real
        kinetic = np.sum(state.p**2 / (2 * self.masses[None, :, None]))
        dq = state.q - np.roll(state.q, -1, axis=0)
        spring = 0.5 * (self.nbeads / self.beta)**2 * np.sum(self.masses[None, :, None] * dq**2)
        return float(result.scalar.sum() + electronic + kinetic + spring)

    def population_flux(self, state):
        """Analytical time derivative of the bead-averaged MMST populations."""
        self._check(state)
        matrix = self._thermal_evaluate(state.q).mapping.matrix
        z = state.mapping_q + 1j * state.mapping_p
        return np.mean(np.imag(z.conj() * np.einsum("bij,bj->bi", matrix, z)), axis=0)


@dataclass(frozen=True)
class ThermalEnsemble:
    """Saved thermal parents from one chain, with their initial signed weights.

    Samples remain correlated even though each recorded nuclear momentum is
    drawn independently.  Use independent chains and explicit blocks when
    interpreting uncertainties.  Stored states are copied before propagation.
    """

    states: tuple[NRPMDState, ...]
    signs: np.ndarray
    electronic_populations: np.ndarray
    masses: np.ndarray
    beta: float
    potential: object = field(repr=False, compare=False)
    sweeps: np.ndarray
    sampling_diagnostics: dict

    def __post_init__(self):
        if not self.states:
            raise ValueError("a thermal ensemble must contain at least one sample")
        if not all(isinstance(state, NRPMDState) for state in self.states):
            raise ValueError("thermal parents must be NRPMDState objects")
        copies = tuple(state.copy() for state in self.states)
        shape = copies[0].q.shape
        mapping_shape = copies[0].mapping_q.shape
        if any(state.q.shape != shape or state.mapping_q.shape != mapping_shape for state in copies):
            raise ValueError("thermal parent shapes must agree")
        checker = CayleyIntegrator(self.masses, self.beta, shape[0], 0, native=False)
        if checker.shape != shape:
            raise ValueError("thermal parent masses/shape mismatch")
        signs = np.array(self.signs, dtype=float, copy=True)
        populations = np.array(self.electronic_populations, dtype=float, copy=True)
        sweeps = np.array(self.sweeps, copy=True)
        count = len(copies)
        if signs.shape != (count,) or not np.all(np.isin(signs, [-1., 1.])):
            raise ValueError("thermal signs must be one +1/-1 value per sample")
        if populations.shape != (count, mapping_shape[1]) or not np.isfinite(populations).all():
            raise ValueError("invalid SEO population estimators")
        if not np.allclose(populations.sum(axis=1), 1, atol=1e-9, rtol=1e-9):
            raise ValueError("SEO population estimators must sum to one")
        if sweeps.shape != (count,) or sweeps.dtype.kind not in "iu" or np.any(sweeps < 0):
            raise ValueError("sweeps must be nonnegative integer sample indices")
        for value in (signs, populations, sweeps):
            value.flags.writeable = False
        object.__setattr__(self, "states", copies)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "electronic_populations", populations)
        object.__setattr__(self, "sweeps", sweeps)
        object.__setattr__(self, "masses", checker.masses)
        object.__setattr__(self, "beta", checker.beta)
        object.__setattr__(self, "sampling_diagnostics", copy.deepcopy(self.sampling_diagnostics))


class ThermalNRPMDSampler:
    """Metropolis sampler of the absolute SEO-projected thermal distribution.

    A sweep proposes one whole-ring centroid displacement, P local nuclear
    moves, and P Gaussian independence refreshes of the mapping variables.
    Physical time does not advance in sampling.  A potential failure rolls
    back the complete sweep, including the RNG state and acceptance counters.
    """

    def __init__(self, potential, masses, beta, q, nstates, seed=0,
                 nuclear_step=0.1, centroid_step=0.1):
        self.q = np.array(coordinates(q), copy=True)
        self.nstates = _positive_integer(nstates, "nstates")
        self._checker = CayleyIntegrator(masses, beta, len(self.q), 0, native=False)
        if self._checker.shape != self.q.shape:
            raise ValueError("q and masses shape mismatch")
        self.masses = self._checker.masses
        self.beta = self._checker.beta
        self.potential = potential
        self.nuclear_step = _finite_positive(nuclear_step, "nuclear_step")
        self.centroid_step = _finite_positive(centroid_step, "centroid_step")
        self.rng = np.random.default_rng(seed)
        self.sweeps = 0
        self.attempted = {"nuclear": 0, "centroid": 0, "mapping": 0}
        self.accepted = dict.fromkeys(self.attempted, 0)
        self._split = _split_matrix(potential, self.q, self.beta, self.nstates)
        for _ in range(100):
            self.x = self.rng.normal(scale=np.sqrt(0.5), size=(len(self.q), self.nstates))
            self.y = self.rng.normal(scale=np.sqrt(0.5), size=self.x.shape)
            self._weight = _weight(self._split, self.q, self.x, self.y, self.masses, self.beta)
            if self._weight.sign:
                break
        else:
            raise ValueError("could not initialize a nonzero SEO thermal weight")

    def _accept(self, delta, kind):
        self.attempted[kind] += 1
        if np.log(max(self.rng.random(), np.finfo(float).tiny)) < min(0.0, delta):
            self.accepted[kind] += 1
            return True
        return False

    def _nuclear_move(self, kind, bead=None):
        q = self.q.copy()
        if bead is None:
            q += self.rng.normal(scale=self.centroid_step, size=q.shape[1:])
        else:
            q[bead] += self.rng.normal(scale=self.nuclear_step, size=q.shape[1:])
        split = _split_matrix(self.potential, q, self.beta, self.nstates)
        weight = _weight(split, q, self.x, self.y, self.masses, self.beta)
        if self._accept(weight.log_density - self._weight.log_density, kind):
            self.q, self._split, self._weight = q, split, weight

    def sweep(self):
        before = (self.q, self.x, self.y, self._split, self._weight, self.sweeps,
                  self.attempted.copy(), self.accepted.copy(), copy.deepcopy(self.rng.bit_generator.state))
        try:
            self._nuclear_move("centroid")
            for bead in self.rng.permutation(len(self.q)):
                self._nuclear_move("nuclear", bead)
            for bead in self.rng.permutation(len(self.q)):
                x, y = self.x.copy(), self.y.copy()
                x[bead] = self.rng.normal(scale=np.sqrt(0.5), size=self.nstates)
                y[bead] = self.rng.normal(scale=np.sqrt(0.5), size=self.nstates)
                weight = _weight(self._split, self.q, x, y, self.masses, self.beta)
                # Independence-Hastings correction cancels the Gaussian
                # mapping density.  Omitting this term biases thermal states.
                gaussian_change = np.sum(x[bead]**2 + y[bead]**2 - self.x[bead]**2 - self.y[bead]**2)
                delta = weight.log_density - self._weight.log_density + gaussian_change
                if self._accept(delta, "mapping"):
                    self.x, self.y, self._weight = x, y, weight
            self.sweeps += 1
        except BaseException:
            (self.q, self.x, self.y, self._split, self._weight, self.sweeps,
             self.attempted, self.accepted, rng_state) = before
            self.rng.bit_generator.state = rng_state
            raise
        return self

    def snapshot(self):
        """Return a detached continuation state; no provider is serialized."""
        return {
            "q": self.q.copy(), "mapping_q": self.x.copy(), "mapping_p": self.y.copy(),
            "beta": self.beta, "masses": self.masses.copy(), "nstates": self.nstates,
            "nuclear_step": self.nuclear_step, "centroid_step": self.centroid_step,
            "sweeps": self.sweeps, "attempted": self.attempted.copy(),
            "accepted": self.accepted.copy(), "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, snapshot):
        """Restore matching parameters transactionally, reevaluating the PES."""
        if (snapshot["beta"] != self.beta or snapshot["nstates"] != self.nstates
                or not np.array_equal(snapshot["masses"], self.masses)
                or snapshot["nuclear_step"] != self.nuclear_step
                or snapshot["centroid_step"] != self.centroid_step):
            raise ValueError("thermal sampler snapshot parameter mismatch")
        state = NRPMDState(snapshot["q"], np.zeros_like(self.q),
                           mapping_q=snapshot["mapping_q"], mapping_p=snapshot["mapping_p"])
        if state.q.shape != self.q.shape or state.mapping_q.shape != self.x.shape:
            raise ValueError("thermal sampler snapshot shape mismatch")
        sweeps = _nonnegative_integer(snapshot["sweeps"], "sweeps")
        accepted, attempted = {}, {}
        for kind in self.attempted:
            attempted[kind] = _nonnegative_integer(snapshot["attempted"][kind], "attempted")
            accepted[kind] = _nonnegative_integer(snapshot["accepted"][kind], "accepted")
            if accepted[kind] > attempted[kind]:
                raise ValueError("accepted moves exceed attempted moves")
        generator = type(self.rng.bit_generator)()
        generator.state = copy.deepcopy(snapshot["rng"])
        split = _split_matrix(self.potential, state.q, self.beta, self.nstates)
        weight = _weight(split, state.q, state.mapping_q, state.mapping_p, self.masses, self.beta)
        if not weight.sign:
            raise ValueError("thermal sampler snapshot lies on a weight node")
        self.q, self.x, self.y = state.q, state.mapping_q, state.mapping_p
        self._split, self._weight = split, weight
        self.sweeps, self.attempted, self.accepted = sweeps, attempted, accepted
        self.rng.bit_generator.state = generator.state
        return self

    def sample(self, count, *, burn_in=0, stride=1):
        count = _positive_integer(count, "count")
        burn_in = _nonnegative_integer(burn_in, "burn_in")
        stride = _positive_integer(stride, "stride")
        for _ in range(burn_in):
            self.sweep()
        states, signs, populations, sweeps = [], [], [], []
        sigma = np.sqrt(self.masses[None, :, None] * len(self.q) / self.beta)
        for _ in range(count):
            for _ in range(stride):
                self.sweep()
            p = self.rng.normal(size=self.q.shape) * sigma
            states.append(NRPMDState(self.q, p, mapping_q=self.x, mapping_p=self.y))
            signs.append(self._weight.sign)
            populations.append(self._weight.electronic_populations)
            sweeps.append(self.sweeps)
        return ThermalEnsemble(
            tuple(states), signs, populations, self.masses, self.beta,
            self.potential, sweeps,
            {"sweeps": self.sweeps, "burn_in_requested": burn_in, "sample_stride": stride,
             "attempted": self.attempted.copy(), "accepted": self.accepted.copy(),
             "distribution": "absolute_SEO_projected_trace", "potential_split": "trace",
             "independent_chains": 1, "physical_converged": False},
        )


@dataclass(frozen=True)
class ThermalRateCorrelation:
    """Per-parent signed rate contributions and explicit normalization.

    ``rate_curve`` is in inverse atomic time.  It is the ratio of sums of the
    signed numerator and reactant denominator, preserving their covariance.
    This object deliberately does not automatically label a tail a rate.
    """

    time: np.ndarray
    numerator: np.ndarray
    reactant_weight: np.ndarray
    signs: np.ndarray
    estimator: str
    min_average_sign: float = 0.05
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        arrays = [np.array(getattr(self, key), dtype=float, copy=True)
                  for key in ("time", "numerator", "reactant_weight", "signs")]
        time, numerator, denominator, signs = arrays
        if (time.ndim != 1 or not len(time) or np.any(np.diff(time) <= 0)
                or numerator.shape != (len(signs), len(time))
                or denominator.shape != signs.shape or signs.ndim != 1 or not len(signs)
                or not np.all(np.isin(signs, [-1., 1.]))
                or not all(np.isfinite(value).all() for value in arrays)):
            raise ValueError("invalid thermal correlation arrays")
        threshold = _finite_positive(self.min_average_sign, "min_average_sign")
        if threshold > 1:
            raise ValueError("min_average_sign must not exceed one")
        for key, value in zip(("time", "numerator", "reactant_weight", "signs"), arrays):
            value.flags.writeable = False
            object.__setattr__(self, key, value)
        object.__setattr__(self, "min_average_sign", threshold)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @property
    def average_sign(self):
        return float(self.signs.mean())

    @property
    def unavailable_reasons(self):
        reasons = []
        if self.average_sign < self.min_average_sign:
            reasons.append("average_sign_below_threshold")
        den = float(self.reactant_weight.mean())
        if den <= 0:
            reasons.append("nonpositive_signed_reactant_normalization")
        elif den > self.average_sign + 1e-10:
            reasons.append("signed_reactant_population_outside_zero_to_one")
        return reasons

    @property
    def rate_curve(self):
        if self.unavailable_reasons:
            return None
        return self.numerator.sum(axis=0) / self.reactant_weight.sum()

    def summarize(self, *, start=None, stop=None, block_size=None, relative_tolerance=0.1):
        """Report an explicitly selected candidate plateau and block error.

        A supplied block size is an assumption about parent-chain mixing, not
        an automatic convergence certificate.  At least two complete blocks
        are required.  The last incomplete block is discarded only from the
        uncertainty calculation; the point estimate uses every parent.
        """
        tolerance = _finite_positive(relative_tolerance, "relative_tolerance")
        result = {
            "estimator": self.estimator, "time_unit": "atomic_time", "rate_unit": "atomic_time^-1",
            "samples": len(self.signs), "average_sign": self.average_sign,
            "signed_reactant_weight": float(self.reactant_weight.mean()),
            "reactant_population": (float(self.reactant_weight.sum()/self.signs.sum())
                                    if self.signs.sum() > 0 else None),
            "rate_curve_available": not self.unavailable_reasons,
            "unavailable_reasons": self.unavailable_reasons,
            "candidate_rate": None, "standard_error": None, "plateau_detected": False,
            "physical_converged": False,
        }
        if (start is None) != (stop is None):
            raise ValueError("both plateau start and stop must be supplied")
        if start is None:
            return result
        start, stop = float(start), float(stop)
        if not np.isfinite(start+stop) or start < self.time[0] or stop > self.time[-1] or stop <= start:
            raise ValueError("invalid plateau time interval")
        indices = np.flatnonzero((self.time >= start) & (self.time <= stop))
        if len(indices) < 4:
            raise ValueError("a plateau interval needs at least four time points")
        result["plateau_interval"] = [start, stop]
        curve = self.rate_curve
        if curve is None:
            return result
        tail = curve[indices]
        estimate = float(tail.mean())
        result["candidate_rate"] = estimate
        result["maximum_relative_excursion"] = (float(np.max(abs(tail-estimate))/abs(estimate))
                                                if estimate != 0 else None)
        # Check the entire interval, so equal half-means cannot certify a
        # strongly oscillating curve.  Statistical uncertainty is separate.
        stable = estimate > 0 and np.max(abs(tail-estimate)) <= tolerance*estimate
        result["interval_shape_stable"] = bool(stable)
        if block_size is None:
            result["uncertainty_reason"] = "parent_correlation_blocks_not_supplied"
            return result
        block_size = _positive_integer(block_size, "block_size")
        blocks = len(self.signs)//block_size
        result.update(block_size=block_size, complete_blocks=blocks,
                      uncertainty_scope="conditional_on_supplied_single_chain_blocks")
        if blocks < 2:
            result["uncertainty_reason"] = "fewer_than_two_complete_blocks"
            return result
        keep = blocks*block_size
        num = self.numerator[:keep, indices].mean(axis=1).reshape(blocks, block_size).sum(axis=1)
        den = self.reactant_weight[:keep].reshape(blocks, block_size).sum(axis=1)
        leave_den = den.sum()-den
        leave_sign = self.signs[:keep].reshape(blocks, block_size).sum(axis=1)
        leave_sign = (leave_sign.sum()-leave_sign)/((blocks-1)*block_size)
        if np.any(leave_den <= 0) or np.any(leave_sign < self.min_average_sign):
            result["uncertainty_reason"] = "jackknife_contains_unresolved_sign_or_reactant_normalization"
            return result
        jackknife = (num.sum()-num)/leave_den
        error = float(np.sqrt((blocks-1)/blocks*np.sum((jackknife-jackknife.mean())**2)))
        result["standard_error"] = error
        # More than two blocks and precision are deliberately required for a
        # diagnostic plateau label.  This still says nothing about bead/dt,
        # basin, sign or independent-chain convergence.
        result["plateau_detected"] = bool(stable and blocks >= 8 and 2*error <= tolerance*estimate)
        return result


def _check_correlation(ensemble, dynamics, steps, stride):
    if not isinstance(ensemble, ThermalEnsemble) or not isinstance(dynamics, ThermalNRPMDDynamics):
        raise ValueError("thermal correlations require ThermalEnsemble and ThermalNRPMDDynamics")
    if (ensemble.potential is not dynamics.potential or ensemble.beta != dynamics.beta
            or not np.array_equal(ensemble.masses, dynamics.masses)
            or ensemble.states[0].q.shape != dynamics.integrator.shape
            or ensemble.states[0].mapping_q.shape[1] != dynamics.nstates):
        raise ValueError("thermal ensemble and dynamics parameters/provider must match")
    if dynamics.dt <= 0:
        raise ValueError("correlation dynamics require positive dt")
    steps = _nonnegative_integer(steps, "steps")
    stride = _positive_integer(stride, "stride")
    indices = np.arange(0, steps+1, stride)
    if indices[-1] != steps:
        indices = np.append(indices, steps)
    return indices


def _surface_value(surface, state, masses, width):
    value, gradient = surface(state.q.mean(axis=0))
    if np.asarray(value).ndim != 0 or np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError("surface must return a finite real scalar value")
    gradient = np.asarray(gradient)
    if gradient.shape != state.q.shape[1:] or gradient.dtype.kind not in "fiu" or not np.isfinite(gradient).all():
        raise ValueError("surface gradient must be finite real [N,3]")
    tangent = np.tanh(float(value)/width)
    side = 0.5*(1+tangent)
    velocity = state.p.mean(axis=0)/masses[:, None]
    flux = 0.5*(1-tangent*tangent)/width * np.sum(gradient*velocity)
    return float(side), float(flux)


def thermal_flux_side(ensemble, dynamics, surface, *, steps, stride=1,
                      smoothing_width=0.1, min_average_sign=0.05):
    """Smoothed nuclear flux-side rate curve with thermal electronic preparation.

    ``surface(centroid[N,3])`` returns ``(s, ds/dcentroid)``; s>0 is product.
    h=(1+tanh(s/width))/2 and F=dh/dt.  The reported curve is
    <sign*F(0)*h(t)> / <sign*(1-h(0))>.  The width, dividing surface, numerical
    propagation and a separated kinetic plateau require independent checks.
    """
    indices = _check_correlation(ensemble, dynamics, steps, stride)
    width = _finite_positive(smoothing_width, "smoothing_width")
    numerator = np.empty((len(ensemble.states), len(indices)))
    denominator = np.empty(len(ensemble.states))
    for sample, (parent, sign) in enumerate(zip(ensemble.states, ensemble.signs)):
        state = parent.copy()
        side, flux = _surface_value(surface, state, dynamics.masses, width)
        denominator[sample] = sign*(1-side)
        for frame, target in enumerate(indices):
            for _ in range(int(target-(indices[frame-1] if frame else 0))):
                dynamics.step(state)
            final_side, _ = _surface_value(surface, state, dynamics.masses, width)
            numerator[sample, frame] = sign*flux*final_side
    return ThermalRateCorrelation(
        indices*dynamics.dt, numerator, denominator, ensemble.signs,
        "thermal_nrpmd_smoothed_nuclear_flux_side", min_average_sign,
        {"smoothing_width": width, "potential_split": "trace",
         "sampling": ensemble.sampling_diagnostics},
    )


def thermal_population_rate(ensemble, dynamics, reactant_state, product_state, *,
                            steps, stride=1, min_average_sign=0.05):
    """Electronic thermal rate curve from d C_RP(t)/dt divided by p_R.

    C_RP uses the SEO projector insertion A_R initially and the signed MMST
    population B_P at time t.  The derivative is evaluated analytically from
    the mapping equations.  Population estimators are never clipped.  Use
    between distinct fixed diabatic states, not instantaneous adiabatic labels.
    """
    indices = _check_correlation(ensemble, dynamics, steps, stride)
    for label, value in (("reactant_state", reactant_state), ("product_state", product_state)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < dynamics.nstates:
            raise ValueError(f"{label} must be an electronic state index")
    if reactant_state == product_state:
        raise ValueError("reactant_state and product_state must be distinct")
    numerator = np.empty((len(ensemble.states), len(indices)))
    population_correlation = np.empty_like(numerator)
    denominator = ensemble.signs*ensemble.electronic_populations[:, reactant_state]
    for sample, parent in enumerate(ensemble.states):
        state = parent.copy()
        for frame, target in enumerate(indices):
            for _ in range(int(target-(indices[frame-1] if frame else 0))):
                dynamics.step(state)
            numerator[sample, frame] = denominator[sample]*dynamics.population_flux(state)[product_state]
            population_correlation[sample, frame] = denominator[sample]*state.populations()[product_state]
    return ThermalRateCorrelation(
        indices*dynamics.dt, numerator, denominator, ensemble.signs,
        "thermal_nrpmd_electronic_population_flux", min_average_sign,
        {"reactant_state": int(reactant_state), "product_state": int(product_state),
         "population_correlation_signed_samples": population_correlation,
         "potential_split": "trace", "sampling": ensemble.sampling_diagnostics},
    )
