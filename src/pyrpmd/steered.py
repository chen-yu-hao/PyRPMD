"""Steered centroid dynamics and nonequilibrium work free energies.

The restraint is a physical energy W(s,lambda); the ring Hamiltonian receives
P*W. Reported work, heat and shadow work are physical energies (hartree), so
the exponential estimator uses beta, never beta/P. A symmetric protocol
switch surrounds a reversible, volume-preserving MD map. Optional exact
momentum OU half steps surround that map. The deterministic map's energy
error is recorded as shadow work and included in the free-energy work.

References: Jarzynski, PRL 78, 2690 (1997), doi:10.1103/PhysRevLett.78.2690;
Crooks, PRE 60, 2721 (1999), doi:10.1103/PhysRevE.60.2721; Bennett, Journal
of Computational Physics 22, 245 (1976), doi:10.1016/0021-9991(76)90078-4; Sivak, Chodera and Crooks,
PRX 3, 011007 (2013), doi:10.1103/PhysRevX.3.011007.

The estimated difference is between the *restrained endpoint Hamiltonians*.
It is not an unbiased CV PMF. Independent equilibrium initial conditions and
adequate work-distribution overlap remain the caller's scientific obligation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, log_expit, logsumexp

from .checkpoint import load_checkpoint, save_metadata_npz
from .dynamics import Dynamics, State
from .io import jsonable
from .validation import checked_integer, finite_real


def _array(value, name, *, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must be a real numerical array" +
                         (f" with {ndim} dimensions" if ndim is not None else ""))
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or not result.size:
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _works(work, name="work"):
    return _array(work, name, ndim=1)


def _json(value):
    return jsonable(value, callables=False, dataclasses="asdict", to_dict="ignore")


@dataclass
class WorkEstimate:
    """Endpoint free-energy estimate with explicitly conditional diagnostics.

    Uncertainty assumes independent pulling trajectories. Bootstrap error and
    bias diagnostics cannot detect work tails absent from the supplied data.
    ``converged`` refers only to the numerical estimator; physical validity
    requires independent studies of the model and sampling.
    """

    delta_free_energy: float
    standard_error: float
    method: str
    beta: float
    diagnostics: dict = field(default_factory=dict)
    converged: bool = True
    energy_unit: str = "hartree"
    target: str = "restrained_endpoint_free_energy_difference"

    def to_dict(self):
        from .io import report_jsonable

        return report_jsonable(self, dataclasses="asdict", to_dict="ignore")


def _jarzynski_value(work, beta):
    # Shifting by the minimum protects even a large common energy offset.
    offset = float(np.min(work))
    with np.errstate(over="ignore"):
        log_weights = -beta * (work - offset)
    normalizer = logsumexp(log_weights)
    return offset - (normalizer - np.log(len(work))) / beta, log_weights - normalizer


def jarzynski(work, beta, *, bootstrap=200, seed=0):
    """Stable -log(mean(exp(-beta*work)))/beta for independent pulls.

    ``work`` is work done *on* the system, including any numerical shadow
    work. The leading-order bias/variance and second-cumulant estimate are
    diagnostics, not automatic corrections or proofs of tail convergence.
    """
    work = _works(work)
    beta = finite_real(beta, "beta", positive=True)
    bootstrap = checked_integer(bootstrap, "bootstrap", minimum=0)
    if bootstrap == 1:
        raise ValueError("bootstrap must be zero or at least two")
    estimate, log_weights = _jarzynski_value(work, beta)
    n = len(work)
    weights = np.exp(log_weights)
    ess = float(np.exp(-logsumexp(2.0 * log_weights)))
    variance = float(np.var(work, ddof=1)) if n > 1 else np.nan
    relative_variance = max(0.0, (n / ess - 1) / (n - 1)) if n > 1 else np.nan
    delta_variance = relative_variance / beta**2
    error = float(np.sqrt(delta_variance))
    bootstrap_bias = np.nan
    if bootstrap and n > 1:
        rng = np.random.default_rng(seed)
        estimates = np.asarray([
            _jarzynski_value(work[rng.integers(n, size=n)], beta)[0]
            for _ in range(bootstrap)
        ])
        error = float(estimates.std(ddof=1))
        bootstrap_bias = float(estimates.mean() - estimate)
    warnings = [
        "Equilibrium initial conditions, independent pulls and unsampled low-work tails are not verified.",
        "The target is the restrained endpoint free-energy difference, not an unbiased CV PMF.",
    ]
    if n < 2:
        warnings.append("One trajectory cannot provide an uncertainty estimate.")
    if ess < max(10, 0.1 * n):
        warnings.append("Exponential work weights have low effective sample size; rare low-work trajectories dominate.")
    return WorkEstimate(float(estimate), error, "jarzynski", beta, {
        "samples": n, "work_mean": float(np.mean(work)), "work_variance": variance,
        "mean_dissipated_work": float(np.mean(work) - estimate),
        "weight_effective_sample_size": ess, "largest_normalized_weight": float(weights.max()),
        "second_cumulant_delta_free_energy": float(np.mean(work) - 0.5 * beta * variance),
        "delta_method_variance": float(delta_variance),
        "leading_order_positive_bias": float(relative_variance / (2 * beta)),
        "bootstrap_bias": bootstrap_bias, "bootstrap_replicates": bootstrap if n > 1 else 0,
        "standard_error_kind": "independent_trajectory_bootstrap" if bootstrap else "delta_method",
        "uncertainty_scope": "conditional on observed work distribution; missing tails and initialization bias excluded",
        "warnings": warnings,
    })


def _bar_value(forward, reverse, beta):
    # Reverse work uses its native B->A sign. Center all dimensionless values
    # to retain accuracy for an arbitrary common free-energy offset.
    offset = float(0.5 * np.min(forward) + 0.5 * np.max(-reverse))
    with np.errstate(over="ignore", invalid="ignore"):
        f = beta * (forward - offset)
        r = beta * (reverse + offset)
    if not np.isfinite(f).all() or not np.isfinite(r).all():
        raise ValueError("dimensionless work range overflows; rescale units or beta")
    ratio = float(np.log(len(forward) / len(reverse)))

    def residual(value):
        # Equality of LOG sums stays well conditioned even when both sums
        # underflow in a pair of badly separated work ensembles.
        return float(logsumexp(log_expit(value - f - ratio)) -
                     logsumexp(log_expit(-value - r + ratio)))

    lo = min(float(f.min()), float((-r).min())) - abs(ratio) - 2.0
    hi = max(float(f.max()), float((-r).max())) + abs(ratio) + 2.0
    root = brentq(residual, lo, hi, xtol=1e-12, rtol=1e-13)
    a, b = expit(root - f - ratio), expit(-root - r + ratio)
    overlap = float((a.sum() + b.sum()) * 0.5 *
                    (1.0 / len(forward) + 1.0 / len(reverse)))
    return offset + root / beta, overlap, a, b


def crooks(forward_work, reverse_work, beta, *, bootstrap=200, seed=0):
    """Crooks/Bennett bidirectional estimator, including unequal counts.

    Forward and reverse pulls must use time-reversed protocols, identical
    temperatures and independent equilibrium starts at their own endpoints.
    Reverse works retain the B->A sign; do not negate them before this call.
    A tiny overlap leaves a numerical root but not a reliable physical result.
    """
    forward = _works(forward_work, "forward_work")
    reverse = _works(reverse_work, "reverse_work")
    beta = finite_real(beta, "beta", positive=True)
    bootstrap = checked_integer(bootstrap, "bootstrap", minimum=0)
    if bootstrap == 1:
        raise ValueError("bootstrap must be zero or at least two")
    estimate, overlap, a, b = _bar_value(forward, reverse, beta)
    curvature = float((np.sum(a * (1 - a)) + np.sum(b * (1 - b))) / (len(a) + len(b)))
    resolved = overlap >= 0.03 and curvature > np.finfo(float).eps
    # Delta-method variance of the two independent sample sums divided by
    # the squared derivative of the BAR estimating equation.
    variance = np.nan
    scale = max(float(a.max()), float(b.max()))
    if min(len(forward), len(reverse)) > 1 and scale > 0:
        aa, bb = a / scale, b / scale
        denominator = beta * float(np.sum(aa * (1 - a)) + np.sum(bb * (1 - b)))
        if denominator > 0:
            variance = float((len(a) * aa.var(ddof=1) + len(b) * bb.var(ddof=1)) / denominator**2)
    error = float(np.sqrt(variance))
    bootstrap_bias = np.nan
    if bootstrap and min(len(forward), len(reverse)) > 1:
        rng = np.random.default_rng(seed)
        estimates = np.asarray([
            _bar_value(forward[rng.integers(len(forward), size=len(forward))],
                       reverse[rng.integers(len(reverse), size=len(reverse))], beta)[0]
            for _ in range(bootstrap)
        ])
        error = float(estimates.std(ddof=1))
        bootstrap_bias = float(estimates.mean() - estimate)
    jf = jarzynski(forward, beta, bootstrap=0)
    jr = jarzynski(reverse, beta, bootstrap=0)
    warnings = [
        "Crooks requires reverse protocols and independent equilibrium initial conditions at both endpoints.",
        "Observed-work bootstrap cannot diagnose unsampled tails or initialization bias.",
    ]
    if overlap < 0.03:
        warnings.append("Forward/reverse work overlap is poor; the BAR root is not evidence of free-energy convergence.")
    if curvature <= np.finfo(float).eps:
        warnings.append("BAR score curvature is unresolved; saturated work ensembles cannot support a reliable root or error bar.")
    if overlap > 1.0:
        warnings.append("Empirical overlap exceeds its population upper bound; inspect forward/reverse consistency and sampling error.")
    if min(len(forward), len(reverse)) < 2:
        warnings.append("Both directions need multiple trajectories to estimate uncertainty.")
    # Underflow-separated ensembles must not acquire an apparently perfect
    # uncertainty merely because every bootstrap repeats the same empty gap.
    if overlap < np.finfo(float).eps or curvature <= np.finfo(float).eps:
        error = np.nan
    return WorkEstimate(float(estimate), error, "crooks_bar", beta, {
        "forward_samples": len(forward), "reverse_samples": len(reverse),
        "overlap": overlap, "overlap_resolved": bool(resolved),
        "bar_score_curvature_per_sample": curvature,
        "forward_jarzynski": jf.delta_free_energy,
        "reverse_jarzynski_in_forward_sign": -jr.delta_free_energy,
        "forward_reverse_hysteresis": jf.delta_free_energy + jr.delta_free_energy,
        "forward_mean_dissipated_work": float(np.mean(forward) - estimate),
        "reverse_mean_dissipated_work": float(np.mean(reverse) + estimate),
        "forward_work_variance": jf.diagnostics["work_variance"],
        "reverse_work_variance": jr.diagnostics["work_variance"],
        "delta_method_variance": variance, "bootstrap_bias": bootstrap_bias,
        "bootstrap_replicates": bootstrap if min(len(forward), len(reverse)) > 1 else 0,
        "standard_error_kind": "independent_trajectory_bootstrap" if bootstrap else "delta_method",
        "warnings": warnings,
    }, converged=bool(curvature > np.finfo(float).eps))


class SteeredBias:
    """Harmonic centroid restraint with an explicit, immutable center schedule.

    ``centers`` has shape [steps+1,D]. ``stiffness`` is a scalar, diagonal
    vector or positive-definite matrix in hartree/(CV unit_i*CV unit_j).
    Force evaluation is pure; only SteeredMD advances the protocol center.
    Periodic CVs are rejected rather than guessing a winding convention.
    """

    history_dependent = True

    def __init__(self, cv, centers, stiffness):
        self.cv = cv
        self.dimension = checked_integer(cv.dimension, "CV dimension")
        components = getattr(cv, "cvs", (cv,))
        if any(getattr(component, "estimator", None) != "centroid" for component in components):
            raise ValueError("steering requires explicit centroid CVs")
        if len(cv.periods) != self.dimension or any(period is not None for period in cv.periods):
            raise ValueError("steering requires nonperiodic CVs; winding protocols are not implemented")
        self.centers = _array(centers, "centers", ndim=2)
        if self.centers.shape[1] != self.dimension or len(self.centers) < 2:
            raise ValueError("centers must have shape [steps+1,D] with at least one step")
        k = _array(stiffness, "stiffness")
        if k.ndim == 0:
            k = np.eye(self.dimension) * float(k)
        elif k.ndim == 1:
            k = np.diag(k)
        if (k.shape != (self.dimension, self.dimension) or
                not np.allclose(k, k.T, rtol=0, atol=1e-14) or np.linalg.eigvalsh(k).min() <= 0):
            raise ValueError("stiffness must be symmetric positive definite and match the CV dimension")
        self.stiffness = k
        self.centers.flags.writeable = self.stiffness.flags.writeable = False
        self._center = self.centers[0].copy()

    @property
    def center(self):
        return self._center.copy()

    def physical_energy(self, values, *, center=None):
        values = _array(values, "CV values")
        if values.shape[-1:] != (self.dimension,):
            raise ValueError("CV values must end with the steering dimension")
        center = self._center if center is None else _array(center, "center", ndim=1)
        if center.shape != (self.dimension,):
            raise ValueError("center must contain one value per CV")
        delta = values - center
        return 0.5 * np.einsum("...i,ij,...j->...", delta, self.stiffness, delta)

    def energy_force(self, q):
        q = _array(q, "coordinates", ndim=3)
        if q.shape[-1] != 3:
            raise ValueError("coordinates must have shape [P,N,3]")
        result = self.cv.value_and_jacobian(q)
        value, jacobian = _array(result.value, "CV values"), _array(result.jacobian, "CV Jacobian")
        if value.shape != (self.dimension,) or jacobian.shape != (self.dimension,) + q.shape:
            raise ValueError("CV contract requires value[D] and jacobian[D,P,N,3]")
        delta = value - self._center
        return (len(q) * 0.5 * float(delta @ self.stiffness @ delta),
                -len(q) * np.einsum("d,dpnc->pnc", self.stiffness @ delta, jacobian))


class SteeredMD:
    """A resumable symmetric steering protocol with explicit work accounting.

    Owns a copy of ``state`` and uses the supplied Dynamics RNG. Dynamics must
    have no configured thermostat/barostat/constraints/force splitting; this
    driver owns exact OU momentum refreshes and the work/heat split. Native
    free-ring maps are allowed; a native-only fused loop is not.

    Finite equilibration uses Metropolis-corrected fixed-center proposals,
    preserving the intended canonical distribution without asserting that the
    requested equilibration duration is sufficient. For a checkpoint, provide
    Dynamics.potential_identity and a semantic cv_identity for custom CVs.
    """

    _format = "PyRPMD.steered.v1"

    def __init__(self, dynamics, state, bias, *, thermostat="langevin", gamma=1.0,
                 equilibration_steps=0, initial_equilibrium=False, cv_identity=None):
        if type(dynamics) is not Dynamics or not isinstance(bias, SteeredBias):
            raise ValueError("SteeredMD requires Dynamics and SteeredBias")
        if dynamics.plan.active or dynamics.bond_constraints or dynamics.force_splitting is not None:
            raise ValueError("SteeredMD owns its bath and requires no barostat, constraints or force splitting")
        if dynamics.loop_backend == "native":
            raise ValueError("steering requires loop_backend='auto' or 'python'")
        finite_real(dynamics.dt, "steering timestep", positive=True)
        if thermostat not in ("none", "langevin"):
            raise ValueError("steering thermostat must be none or langevin")
        gamma = finite_real(gamma, "gamma", nonnegative=True)
        if thermostat == "langevin" and gamma <= 0:
            raise ValueError("Langevin steering requires positive gamma")
        if not isinstance(initial_equilibrium, bool):
            raise ValueError("initial_equilibrium must be boolean")
        if cv_identity is not None and (not isinstance(cv_identity, str) or not cv_identity.strip()):
            raise ValueError("cv_identity must be a nonempty semantic identity string")
        self.dynamics, self.bias, self.state = dynamics, bias, state.copy()
        self.bias._center = self.bias.centers[0].copy()
        dynamics.integrator._check(self.state.q, self.state.p)
        dynamics._check_fixed(self.state.q, self.state.p)
        bias.energy_force(self.state.q)
        self.thermostat, self.gamma = thermostat, gamma
        self.equilibration_steps = checked_integer(equilibration_steps, "equilibration_steps", minimum=0)
        if self.equilibration_steps and thermostat == "none":
            raise ValueError("equilibration requires the Langevin bath")
        self.initial_equilibrium, self.cv_identity = initial_equilibrium, cv_identity
        self.equilibrated = self.elapsed = self.equilibration_accepted = 0
        self._initial_clock = (self.state.step, self.state.time)
        self._records = [self._record(0.0, 0.0, 0.0)]
        self._identity = self._current_identity()

    def _current_identity(self):
        if self.cv_identity is None:
            from .workflows.sampling import _checkpoint_cv_identity

            cv = _checkpoint_cv_identity(self.bias.cv)
        else:
            cv = {"explicit_cv_identity": self.cv_identity, "dimension": self.bias.dimension,
                  "units": self.bias.cv.units, "periods": self.bias.cv.periods}
        return _json({"algorithm": "symmetric_steering_shadow_work_v1",
                      "dynamics": self.dynamics.operator_state_dict()["composition"]["identity"],
                      "cv": cv, "centers": self.bias.centers, "stiffness": self.bias.stiffness,
                      "thermostat": self.thermostat, "gamma": self.gamma,
                      "equilibration_steps": self.equilibration_steps,
                      "initial_equilibrium": self.initial_equilibrium})

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self._identity, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @property
    def complete(self):
        return self.elapsed == len(self.bias.centers) - 1

    def _record(self, protocol, shadow, heat):
        values = _array(self.bias.cv.value(self.state.q), "CV values", ndim=1)
        if values.shape != (self.bias.dimension,):
            raise ValueError("CV value must have shape [D]")
        return [self.state.time, *values, protocol, shadow, heat]

    def _kinetic(self):
        return float(np.sum(self.state.p**2 / (2 * self.dynamics.masses[None, :, None])) /
                     self.dynamics.nbeads)

    def _bath(self):
        if self.thermostat == "none":
            return 0.0
        before = self._kinetic()
        c = np.exp(-0.5 * self.gamma * self.dynamics.dt)
        variance = -np.expm1(-self.gamma * self.dynamics.dt)
        self.state.p[:] = c * self.state.p + np.sqrt(
            variance * self.dynamics.nbeads * self.dynamics.masses[None, :, None] / self.dynamics.beta
        ) * self.dynamics.rng.normal(size=self.state.p.shape)
        if self.dynamics.fixed_atoms:
            self.state.p[:, self.dynamics._fixed_indices] = 0.0
        return self._kinetic() - before

    def _switch(self, center):
        value = self.bias.cv.value(self.state.q)
        work = float(self.bias.physical_energy(value, center=center) - self.bias.physical_energy(value))
        self.bias._center = np.array(center, copy=True)
        return work

    def step(self):
        """Commit one complete equilibration or pulling step, or roll it back."""
        if self.complete:
            raise ValueError("steering protocol is already complete")
        old_state = self.state.copy()
        old_rng = copy.deepcopy(self.dynamics.rng.bit_generator.state)
        old_center = self.bias.center
        try:
            if self.equilibrated < self.equilibration_steps:
                self._bath()
                proposal_start = self.state.copy()
                before = self.dynamics.energy(self.state, self.bias) / self.dynamics.nbeads
                self.dynamics.step(self.state, self.bias)
                after = self.dynamics.energy(self.state, self.bias) / self.dynamics.nbeads
                accepted = np.log(self.dynamics.rng.random()) < -self.dynamics.beta * (after - before)
                if not accepted:
                    self.state.q[:] = proposal_start.q
                    self.state.p[:] = -proposal_start.p
                self._bath()
                record = self._record(0.0, 0.0, 0.0)
            else:
                start = self.bias.centers[self.elapsed]
                end = self.bias.centers[self.elapsed + 1]
                protocol = self._switch(0.5 * start + 0.5 * end)
                heat = self._bath()
                before = self.dynamics.energy(self.state, self.bias) / self.dynamics.nbeads
                self.dynamics.step(self.state, self.bias)
                shadow = self.dynamics.energy(self.state, self.bias) / self.dynamics.nbeads - before
                heat += self._bath()
                protocol += self._switch(end)
                previous = self._records[-1][-3:]
                record = self._record(previous[0] + protocol, previous[1] + shadow, previous[2] + heat)
            if not np.isfinite(record).all():
                raise FloatingPointError("nonfinite steering work or observable")
        except BaseException:
            self.state.q[:], self.state.p[:] = old_state.q, old_state.p
            self.state.step, self.state.time = old_state.step, old_state.time
            self.dynamics.rng.bit_generator.state = old_rng
            self.bias._center = old_center
            raise
        if self.equilibrated < self.equilibration_steps:
            self.equilibrated += 1
            self.equilibration_accepted += int(accepted)
            self._records[0] = record
        else:
            self.elapsed += 1
            self._records.append(record)
        self.dynamics._record_python_steps(1)
        return self.state

    def run(self, steps=None, *, checkpoint=None, checkpoint_interval=1000):
        """Advance a prefix or the remaining protocol, saving complete steps."""
        if checkpoint is not None and self.dynamics.potential_identity is None:
            raise ValueError("checkpointed steering requires an explicit potential_identity")
        if self._current_identity() != self._identity:
            raise ValueError("steering scientific parameters changed after construction")
        remaining = self.equilibration_steps - self.equilibrated + len(self.bias.centers) - 1 - self.elapsed
        steps = remaining if steps is None else checked_integer(steps, "steps", minimum=0, maximum=remaining)
        checkpoint_interval = checked_integer(checkpoint_interval, "checkpoint_interval")
        for _ in range(steps):
            self.step()
            if checkpoint is not None and (self.equilibrated + self.elapsed) % checkpoint_interval == 0:
                self.save(checkpoint)
        if checkpoint is not None:
            self.save(checkpoint)
        return self.result()

    def result(self):
        records = np.asarray(self._records)
        return {"time": records[:, 0].copy(), "cv_values": records[:, 1:-3].copy(),
                "centers": self.bias.centers[:self.elapsed + 1].copy(),
                "protocol_work": records[:, -3].copy(), "shadow_work": records[:, -2].copy(),
                "total_work": (records[:, -3] + records[:, -2]).copy(),
                "heat": records[:, -1].copy(), "elapsed": self.elapsed,
                "equilibrated": self.equilibrated, "equilibration_accepted": self.equilibration_accepted,
                "complete": self.complete, "state": self.state.copy(),
                "propagation": {"python_steps": self.equilibrated + self.elapsed, "native_steps": 0,
                                "native_free_ring": self.dynamics.native,
                                "scheduling_reason": "symmetric steering and shadow-work accounting use the Python scheduler"},
                "initial_equilibrium_declared": self.initial_equilibrium, "energy_unit": "hartree"}

    def save(self, path):
        """Atomically save protocol prefix, RNG, identity and physical state."""
        if self.dynamics.potential_identity is None:
            raise ValueError("checkpointed steering requires an explicit potential_identity")
        if self._current_identity() != self._identity:
            raise ValueError("steering scientific parameters changed after construction")
        metadata = {"format": self._format, "identity": self._identity, "fingerprint": self.fingerprint,
                    "operators": self.dynamics.operator_state_dict(),
                    "equilibrated": self.equilibrated, "elapsed": self.elapsed,
                    "equilibration_accepted": self.equilibration_accepted,
                    "step": self.state.step, "time": self.state.time,
                    "initial_clock": self._initial_clock}
        save_metadata_npz(path, metadata, q=self.state.q, p=self.state.p, records=np.asarray(self._records))

    def load(self, path):
        """Validate an entire prefix before atomically replacing live state."""
        if self.dynamics.potential_identity is None:
            raise ValueError("checkpointed steering requires an explicit potential_identity")
        if self._current_identity() != self._identity:
            raise ValueError("steering scientific parameters changed after construction")
        with load_checkpoint(path, expected_kind=self._format, require_integrity=True) as data:
            if set(data.files) != {"metadata", "q", "p", "records"}:
                raise ValueError("invalid steering checkpoint payload")
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("identity") != self._identity or metadata.get("fingerprint") != self.fingerprint:
                raise ValueError("steering checkpoint scientific identity differs")
            state = State(data["q"], data["p"], metadata["step"], metadata["time"])
            self.dynamics.integrator._check(state.q, state.p)
            self.dynamics._check_fixed(state.q, state.p)
            eq = checked_integer(metadata["equilibrated"], "checkpoint equilibrated", minimum=0,
                                 maximum=self.equilibration_steps)
            elapsed = checked_integer(metadata["elapsed"], "checkpoint elapsed", minimum=0,
                                      maximum=len(self.bias.centers) - 1)
            accepted = checked_integer(metadata["equilibration_accepted"], "checkpoint accepted", minimum=0, maximum=eq)
            clock = metadata["initial_clock"]
            if not isinstance(clock, list) or len(clock) != 2:
                raise ValueError("invalid steering initial clock")
            first_step = checked_integer(clock[0], "initial step", minimum=0)
            first_time = finite_real(clock[1], "initial time")
            if elapsed and eq != self.equilibration_steps:
                raise ValueError("steering checkpoint started pulling before equilibration completed")
            expected_time = first_time + (eq + elapsed) * self.dynamics.dt
            if state.step != first_step + eq + elapsed or not np.isclose(state.time, expected_time, rtol=1e-12, atol=1e-12):
                raise ValueError("steering checkpoint clock disagrees with completed steps")
            records = _array(data["records"], "checkpoint records", ndim=2)
            if (records.shape != (elapsed + 1, self.bias.dimension + 4) or
                    np.any(records[0, -3:] != 0.0) or
                    not np.allclose(records[:, 0], first_time + (eq + np.arange(elapsed + 1)) * self.dynamics.dt,
                                    rtol=1e-12, atol=1e-12) or
                    not np.array_equal(records[-1, 1:-3], self.bias.cv.value(state.q))):
                raise ValueError("steering checkpoint records disagree with the physical state")
            operators = metadata["operators"]
        # Dynamics validates and restores its operators transactionally.
        self.dynamics.load_operator_state_dict(operators)
        self.state = state
        self.equilibrated, self.elapsed, self.equilibration_accepted = eq, elapsed, accepted
        self._initial_clock = (first_step, first_time)
        self._records = records.tolist()
        self.bias._center = self.bias.centers[elapsed].copy()
        return self


__all__ = ["SteeredBias", "SteeredMD", "WorkEstimate", "jarzynski", "crooks"]
