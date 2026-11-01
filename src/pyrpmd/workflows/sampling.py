"""Shared sampling validation and checkpoint identities; no propagation loop."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from ..dynamics import normalize_integrator
from ..units import parse_quantity


def validate_andersen_schedule(config, workflow=None):
    """Validate an explicitly requested collision clock without adding defaults.

    Omitting the option must leave historical resolved configs/fingerprints and
    RNG consumption unchanged. Public entrypoints pass their actual workflow
    name so a misleading ``task`` cannot silently enable an unsupported clock.
    """
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a table")
    if "andersen_schedule" not in sampling:
        return "fixed"
    kind = sampling["andersen_schedule"]
    if not isinstance(kind, str) or kind not in ("fixed", "geometric"):
        raise ValueError("sampling.andersen_schedule must be 'fixed' or 'geometric'")
    if sampling.get("thermostat", "andersen") != "andersen":
        raise ValueError("sampling.andersen_schedule requires thermostat='andersen'")
    workflow = config.get("task", "fes") if workflow is None else workflow
    if workflow not in ("fes", "rate", "adaptive_umbrella"):
        raise ValueError(
            "sampling.andersen_schedule is supported only by umbrella FES and gas-phase rate workflows"
        )
    exchange = config.get("replica_exchange", {})
    if not isinstance(exchange, Mapping):
        raise ValueError("replica_exchange must be a table")
    if exchange.get("enabled", bool(exchange)):
        raise ValueError("sampling.andersen_schedule is not supported with replica exchange")
    AndersenSchedule(sampling.get("andersen_interval", 10), kind)
    return kind


@dataclass
class AndersenSchedule:
    """A checkpointable global momentum-refresh clock, measured in MD steps.

    ``geometric`` uses independent waits N>=1 with P(N=n)=p(1-p)**(n-1)
    and p=1/interval. Its sole additional state is an absolute deadline; the
    caller already checkpoints the RNG used for momenta and waiting times.
    """

    interval: int
    kind: str = "fixed"
    next_collision_step: int = 0

    def __post_init__(self):
        if (isinstance(self.interval, (bool, np.bool_))
                or not isinstance(self.interval, (int, np.integer)) or self.interval < 1):
            raise ValueError("sampling.andersen_interval must be a positive integer")
        if self.kind not in ("fixed", "geometric"):
            raise ValueError("sampling.andersen_schedule must be 'fixed' or 'geometric'")
        self.interval = int(self.interval)
        if self.kind == "geometric" and self.interval > np.iinfo(np.int64).max:
            raise ValueError("geometric Andersen mean interval exceeds the checkpoint clock range")

    def restore(self, saved, step):
        keys = ("andersen_schedule", "andersen_next_collision_step")
        if self.kind == "fixed":
            if any(key in saved for key in keys):
                raise ValueError("fixed Andersen checkpoint contains an incompatible collision clock")
            return
        if not all(key in saved for key in keys):
            raise ValueError("geometric Andersen checkpoint is missing its collision clock")
        name = np.asarray(saved[keys[0]])
        deadline = np.asarray(saved[keys[1]])
        if (name.shape != () or str(name) != self.kind
                or deadline.shape != () or deadline.dtype.kind not in "iu"
                or int(deadline) < step or int(deadline) > np.iinfo(np.int64).max):
            raise ValueError("invalid geometric Andersen checkpoint collision clock")
        self.next_collision_step = int(deadline)

    def due(self, step):
        if self.kind == "fixed":
            return step % self.interval == 0
        if step > self.next_collision_step:
            raise ValueError("geometric Andersen collision deadline was skipped")
        return step == self.next_collision_step

    def refreshed(self, step, rng):
        if self.kind == "geometric":
            wait = int(rng.geometric(1.0 / self.interval))
            if wait < 1 or step + wait > np.iinfo(np.int64).max:
                raise ValueError("geometric Andersen collision clock overflow")
            self.next_collision_step = int(step) + wait

    def steps_until_collision(self, step):
        if self.kind == "fixed":
            return self.interval - step % self.interval
        remaining = self.next_collision_step - step
        if remaining <= 0:
            raise ValueError("geometric Andersen collision must refresh before propagation")
        return remaining

    def checkpoint_fields(self):
        if self.kind == "fixed":
            return {}
        return {"andersen_schedule": np.asarray(self.kind),
                "andersen_next_collision_step": np.asarray(self.next_collision_step, dtype=np.int64)}


def _checkpoint_cv_identity(cv):
    """Describe only mathematical built-ins whose method identities are known."""
    from ..analysis.metadynamics import _cv_signature
    from ..cv import (
        Angle, BondDifference, Cartesian, COMDistance, CVStack, Distance,
        RateSurface, RPMDrateCoordinate, _BUILTIN_CV_METHODS,
    )

    allowed = (Angle, BondDifference, Cartesian, COMDistance, CVStack, Distance,
               RateSurface, RPMDrateCoordinate)
    kind = type(cv)
    error = (
        "Checkpointed sampling requires unmodified supported built-in CV objects; "
        "use a serializable CV factory configuration for custom, overridden, "
        "periodic or surface CV objects"
    )
    if kind not in allowed:
        raise ValueError(error)
    methods = _BUILTIN_CV_METHODS[kind]
    if any(name in cv.__dict__ or getattr(kind, name, None) is not original
           for name, original in methods.items()):
        raise ValueError(error)
    if kind is CVStack:
        for component in cv.cvs:
            _checkpoint_cv_identity(component)
    elif kind is RateSurface:
        _checkpoint_cv_identity(cv.coordinate)
    elif kind is RPMDrateCoordinate:
        _checkpoint_cv_identity(cv.com)
    # Stack metadata is mutable and is not part of _cv_signature's recursive
    # tuple. Include it explicitly because periods affect bias displacements.
    return {
        "pyrpmd_builtin_cv": _cv_signature(cv),
        "dimension": cv.dimension,
        "units": tuple(cv.units),
        "periods": tuple(cv.periods),
    }


def checkpoint_cv_config(config):
    """Return a shallow canonical config for scientific checkpoint hashing.

    Direct built-in CV objects are replaced by semantic parameter identities
    *before* the fingerprint helper deep-copies the config. Object ``repr``
    would hash memory addresses and change across copies/processes/restarts.
    Existing mapping/factory specifications and their fingerprints stay
    unchanged. This result is for identity only, not propagation or input.
    """
    canonical = dict(config)
    specs = config.get("cvs")
    if isinstance(specs, (list, tuple)) and any(not isinstance(spec, Mapping) for spec in specs):
        identities = [spec if isinstance(spec, Mapping) else _checkpoint_cv_identity(spec)
                      for spec in specs]
        canonical["cvs"] = tuple(identities) if isinstance(specs, tuple) else identities
    return canonical


@dataclass(frozen=True)
class FESSampler:
    thermostat: str
    integrator: str
    gamma_centroid: float | None = None

    def step(self, dynamics, state, bias):
        if self.thermostat == "pile-l":
            return dynamics.sample_step(state, bias, method=self.integrator,
                                        gamma_centroid=self.gamma_centroid)
        return dynamics.step(state, bias)


def fes_sampler(config):
    """Validate resolved CLI inputs and direct workflow calls without mutating them.

    Andersen scheduling/RNG belongs to the caller. PILE-L reuses sample_step,
    including native free/OU/free and fixed-atom handling.
    """
    validate_andersen_schedule(config)
    sampling = config.get("sampling", {})
    dynamics = config.get("dynamics", {})
    scheme = normalize_integrator(dynamics.get("integrator", "cayley-bcb"))
    thermostat = sampling.get("thermostat", "andersen")
    if thermostat == "andersen":
        expected = scheme + "-andersen"
        if sampling.get("integrator", expected) != expected:
            raise ValueError("Andersen FES sampling must match the dynamics integrator")
        if "gamma_centroid" in sampling:
            raise ValueError("sampling.gamma_centroid requires PILE-L sampling")
        return FESSampler(thermostat, expected)
    if thermostat != "pile-l":
        raise ValueError("FES sampling.thermostat must be andersen or pile-l")
    if config.get("task", "fes") != "fes":
        raise ValueError("PILE-L umbrella sampling currently requires task='fes'")
    expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(scheme)
    if expected is None or sampling.get("integrator", expected) != expected:
        raise ValueError("PILE-L requires BCOCB with Cayley or BAOAB with normal-mode dynamics")
    if dynamics.get("loop_backend", "auto") == "native":
        raise ValueError("PILE-L sampling requires loop_backend='auto' or 'python'")
    gamma = parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("sampling.gamma_centroid must be positive and finite")
    return FESSampler(thermostat, expected, float(gamma))
