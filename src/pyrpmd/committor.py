"""Independent canonical momentum shots for a fixed full-ring RPMD configuration.

This computes a configurational first-hit committor for the supplied RPMD
dynamics, NOT an exact quantum committor or a rate coefficient.
"""
from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import binomtest

from .dynamics import (
    CayleyIntegrator, Dynamics, State,
    _BLOCK_DYNAMICS_METHODS, _BLOCK_INTEGRATOR_METHODS,
)


def _integer(x, name, minimum=1):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Integral) or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(x)


def _finite(x, name):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def committor_statistics(n_a, n_b, n_unresolved, *, confidence_level=0.95):
    """Bounds use all shots; unresolved outcomes are NEVER discarded.

    Two exact-binomial limits, with alpha/2 allocated to each tail, give a
    conservative coverage envelope for eventual B-before-A probability.
    The envelope does not include model, bead or time-discretization errors.
    """
    a, b, u = (_integer(x, name, 0) for x, name in (
        (n_a, "n_a"), (n_b, "n_b"), (n_unresolved, "n_unresolved")))
    n = a + b + u
    if n == 0:
        raise ValueError("at least one shot is required")
    level = _finite(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    lower = binomtest(b, n).proportion_ci(level, method="exact").low
    upper = binomtest(b + u, n).proportion_ci(level, method="exact").high
    return {
        "n_shots": n, "n_A": a, "n_B": b, "n_unresolved": u,
        "p_B": b / n if u == 0 else None,
        "p_B_by_deadline": b / n,
        "p_B_given_resolved": b / (a + b) if a + b else None,
        "unresolved_fraction": u / n,
        "empirical_p_B_bounds": [b / n, (b + u) / n],
        "p_B_confidence_envelope": [float(lower), float(upper)],
        "confidence_level": level,
        "interval_method": "clopper_pearson_censoring_envelope",
        "converged": False,
    }


@dataclass(frozen=True)
class CommittorShot:
    outcome: str
    steps: int
    final_value: float
    final_state: State


@dataclass(frozen=True)
class CommittorResult:
    configuration: np.ndarray
    outcomes: np.ndarray
    steps: np.ndarray
    final_values: np.ndarray
    rng_state: dict
    beta: float
    dt: float
    max_steps: int
    basin_a: float
    basin_b: float
    initial_basin: str | None

    def __post_init__(self):
        q = np.asarray(self.configuration)
        out, steps = np.asarray(self.outcomes), np.asarray(self.steps)
        vals = np.asarray(self.final_values)
        if (q.dtype.kind not in "fiu" or q.ndim != 3 or q.shape[-1] != 3
                or min(q.shape) < 1 or not np.isfinite(q).all()):
            raise ValueError("configuration requires finite q[P,N,3]")
        if (out.dtype.kind not in "iu" or out.ndim != 1 or len(out) < 1
                or not np.isin(out, (-1, 0, 1)).all()
                or steps.dtype.kind not in "iu" or steps.shape != out.shape
                or vals.dtype.kind not in "fiu" or vals.shape != out.shape
                or not np.isfinite(vals).all()):
            raise ValueError("invalid committor shot records")
        count = _integer(self.max_steps, "max_steps")
        beta, dt = _finite(self.beta, "beta"), _finite(self.dt, "dt")
        a, b = _finite(self.basin_a, "basin_a"), _finite(self.basin_b, "basin_b")
        if beta <= 0 or dt <= 0 or a >= b:
            raise ValueError("invalid simulation parameters")
        classify = np.where(vals <= a, -1, np.where(vals >= b, 1, 0))
        if (not np.array_equal(classify, out) or np.any(steps < 0)
                or np.any(steps > count) or np.any(steps[out == 0] != count)):
            raise ValueError("shot outcomes and termination values disagree")
        if self.initial_basin is not None:
            if self.initial_basin not in ("A", "B"):
                raise ValueError("invalid initial basin")
            expected = -1 if self.initial_basin == "A" else 1
            if np.any(out != expected) or np.any(steps != 0):
                raise ValueError("initial basin should terminate immediately")
        elif np.any(steps == 0):
            raise ValueError("outside-basin shots need positive propagation steps")
        for name, data, dtype in (("configuration", q, float),
                                  ("outcomes", out, np.int8),
                                  ("steps", steps, np.int64),
                                  ("final_values", vals, float)):
            owned = np.array(data, dtype=dtype, copy=True, order="C")
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "rng_state", copy.deepcopy(self.rng_state))
        for name, value in (("beta", beta), ("dt", dt), ("max_steps", count),
                            ("basin_a", a), ("basin_b", b)):
            object.__setattr__(self, name, value)

    def summary(self, *, confidence_level=0.95):
        result = committor_statistics(
            int(np.sum(self.outcomes == -1)), int(np.sum(self.outcomes == 1)),
            int(np.sum(self.outcomes == 0)), confidence_level=confidence_level)
        if self.initial_basin is not None:
            exact = float(self.initial_basin == "B")
            result["p_B_confidence_envelope"] = [exact, exact]
            result["interval_method"] = "known_basin_boundary"
        result.update({
            "method": "fixed_full_ring_canonical_momenta_committor",
            "nbeads": len(self.configuration),
            "beta_au": self.beta, "dt_au": self.dt,
            "max_steps": self.max_steps, "horizon_au": self.max_steps * self.dt,
            "basin_a": self.basin_a, "basin_b": self.basin_b,
            "initial_basin": self.initial_basin,
        })
        return result


class CommittorSampler:
    """Independent momentum shots at one fixed complete ring geometry.

    For sum-beads H_P at beta/P, each mobile momentum component has variance
    m*P/beta. No thermostats, constraints, trajectory bias or COM subtraction.
    The first hit of s<=A or s>=B terminates at a completed integration step.
    """

    def __init__(self, dynamics, coordinate, *, basin_a, basin_b,
                 max_steps=10000, seed=0):
        if type(dynamics) is not Dynamics or type(dynamics.integrator) is not CayleyIntegrator:
            raise TypeError("committor requires the built-in deterministic Dynamics")
        self.dynamics = dynamics
        self.basin_a, self.basin_b = _finite(basin_a, "basin_a"), _finite(basin_b, "basin_b")
        if self.basin_a >= self.basin_b:
            raise ValueError("basin_a must be less than basin_b")
        self.max_steps = _integer(max_steps, "max_steps")
        if not math.isfinite(self.max_steps * dynamics.dt):
            raise ValueError("nonfinite simulation horizon")
        if callable(getattr(coordinate, "value", None)):
            if getattr(coordinate, "dimension", 1) != 1:
                raise ValueError("committor requires scalar CV")
            if any(p is not None for p in getattr(coordinate, "periods", (None,))):
                raise ValueError("committor requires ordered nonperiodic CV")
            self._coordinate = coordinate.value
        elif callable(coordinate):
            self._coordinate = coordinate
        else:
            raise TypeError("coordinate must be a scalar CV or callable on q")
        self.rng = np.random.default_rng(seed)
        if self.rng is dynamics.rng:
            raise ValueError("committor RNG must not be the Dynamics RNG")
        self._signature = self._dynamics_signature()
        self._check_dynamics()

    def _dynamics_signature(self):
        d = self.dynamics
        return (id(d.potential), d.beta, d.nbeads, d.dt, d.integrator_name,
                tuple(d.masses), d.fixed_atoms,
                None if d.fixed_positions is None else d.fixed_positions.tobytes())

    def _check_dynamics(self):
        d = self.dynamics
        d._check_parameters()
        if self._dynamics_signature() != self._signature:
            raise ValueError("Dynamics changed: create a new sampler")
        if d.dt <= 0 or len(d.fixed_atoms) == len(d.masses):
            raise ValueError("positive dt and mobile atoms are required")
        if d.plan.active:
            raise ValueError(
                "committor requires plain deterministic Dynamics: "
                "no thermostat, barostat or force splitting"
            )
        if d.bond_constraints:
            raise ValueError("committor does not support bond constraints")
        for obj, cls, methods in ((d, Dynamics, _BLOCK_DYNAMICS_METHODS),
                                  (d.integrator, CayleyIntegrator, _BLOCK_INTEGRATOR_METHODS)):
            if type(obj) is not cls or any(
                name in obj.__dict__ or getattr(cls, name) is not original
                for name, original in methods.items()
            ):
                raise TypeError("patched/thermostatted Dynamics is not supported")
        metadata = getattr(d.potential, "metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        caps = metadata.get("capabilities", {})
        caps = caps if isinstance(caps, dict) else {}
        if any(getattr(d.potential, key, False) or metadata.get(key, False) or caps.get(key, False)
               for key in ("history_dependent", "time_dependent", "stateful")):
            raise ValueError("committor requires time-independent, unbiased potential")

    def _value(self, q):
        view = q.view()
        view.flags.writeable = False
        value = np.asarray(self._coordinate(view))
        if value.dtype.kind not in "fiu" or value.size != 1 or not np.isfinite(value).all():
            raise ValueError("coordinate must return one finite real scalar")
        return float(value.reshape(-1)[0])

    def _basin(self, value):
        return "A" if value <= self.basin_a else "B" if value >= self.basin_b else None

    def _initial_state(self, q):
        state = State(q, np.zeros(np.shape(q)))
        self.dynamics.integrator._check(state.q, state.p)
        self.dynamics._check_fixed(state.q, state.p)
        return state

    def _momenta(self):
        d = self.dynamics
        p = self.rng.normal(size=d.integrator.shape) * np.sqrt(
            d.masses[None, :, None] * d.nbeads / d.beta)
        p[:, d.fixed_atoms, :] = 0.0
        return p

    def shoot(self, configuration):
        """Unbiased single shot; failed PES/CV calls do not become censoring."""
        self._check_dynamics()
        state = self._initial_state(configuration)
        value = self._value(state.q)
        initial = self._basin(value)
        if initial:
            return CommittorShot(initial, 0, value, state)
        saved = copy.deepcopy(self.rng.bit_generator.state)
        try:
            state.p[:] = self._momenta()
            for _ in range(self.max_steps):
                self.dynamics.step(state)
                value = self._value(state.q)
                basin = self._basin(value)
                if basin:
                    break
            return CommittorShot(basin or "unresolved", state.step, value, state)
        except Exception:
            self.rng.bit_generator.state = saved
            raise

    def sample(self, configuration, *, n_shots,
               on_shot: Callable[[int, CommittorShot], None] | None = None):
        """Store scalar records only. Callback may retain unresolved final states."""
        n = _integer(n_shots, "n_shots")
        if on_shot is not None and not callable(on_shot):
            raise TypeError("on_shot must be callable")
        self._check_dynamics()
        q = self._initial_state(configuration).q
        initial = self._basin(self._value(q))
        outcomes, steps, values = np.empty(n, np.int8), np.empty(n, np.int64), np.empty(n)
        for i in range(n):
            shot = self.shoot(q)
            outcomes[i] = {"A": -1, "B": 1, "unresolved": 0}[shot.outcome]
            steps[i], values[i] = shot.steps, shot.final_value
            if on_shot is not None:
                on_shot(i, shot)
        self._check_dynamics()
        d = self.dynamics
        return CommittorResult(q, outcomes, steps, values,
                               copy.deepcopy(self.rng.bit_generator.state),
                               d.beta, d.dt, self.max_steps,
                               self.basin_a, self.basin_b, initial)
