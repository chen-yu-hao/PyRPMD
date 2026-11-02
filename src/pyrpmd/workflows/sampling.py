"""Shared sampling validation and checkpoint identities; no propagation loop."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import copy
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
    if workflow == "fes":
        # Canonical FES dispatch also includes history-dependent MetaD/OPES.
        # Their aliases and canonical spellings must reject the same clocks.
        from .common import fes_method

        if fes_method(config) != "umbrella":
            workflow = fes_method(config)
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
    thermostat_spec: dict | None = None
    operator_checkpoint: bool = False

    def step(self, dynamics, state, bias):
        if self.thermostat_spec is not None:
            return dynamics.sample_step(state, bias,
                                        thermostat=copy.deepcopy(self.thermostat_spec))
        if self.thermostat == "pile-l":
            return dynamics.sample_step(state, bias, method=self.integrator,
                                        gamma_centroid=self.gamma_centroid)
        return dynamics.step(state, bias)


def thermostat_spec(sampling):
    """Resolve unit-aware sampling controls for the common propagation stack.

    The operator implementation validates the parameter schema; this adapter
    only converts workflow quantities to its atomic-unit contract. No mutable
    thermostat object or random stream is shared between independent tasks.
    """
    from ..engine import normalize_thermostat

    raw = sampling.get("thermostat_params", {})
    if not isinstance(raw, Mapping):
        raise ValueError("sampling.thermostat_params must be a table")
    parameters = copy.deepcopy(dict(raw))
    if "mode" in parameters:
        raise ValueError("set thermostat mode in sampling.thermostat, not thermostat_params")
    parameters["mode"] = sampling.get("thermostat", "andersen")
    if "gamma_centroid" in sampling:
        if "gamma_centroid" in parameters:
            raise ValueError("gamma_centroid was specified twice")
        parameters["gamma_centroid"] = sampling["gamma_centroid"]
    mode = parameters["mode"]
    if mode == "pile-l":
        parameters.setdefault("gamma_centroid", "0.01 fs^-1")
    if mode == "andersen":
        if "interval_steps" in parameters:
            if "interval" in parameters:
                raise ValueError("Andersen interval was specified twice")
            parameters["interval"] = parameters.pop("interval_steps")
        parameters.setdefault("interval", sampling.get("andersen_interval", 10))
    for name in ("gamma", "gamma_centroid"):
        if name in parameters:
            parameters[name] = parse_quantity(parameters[name], "frequency")
    for name in ("tau", "tdamp"):
        if name in parameters:
            parameters[name] = parse_quantity(parameters[name], "time")
    if mode in ("nhc", "nose-hoover-chain"):
        if "tdamp" not in parameters:
            parameters.setdefault("tau", parse_quantity("20 fs", "time"))
        # Phase-boundary reset is an existing MetaD policy, not an operator.
        reset = parameters.pop("reset_after_equilibration", False)
        if not isinstance(reset, bool) or reset:
            raise ValueError("composed NHC preserves its bath through equilibration; reset_after_equilibration must be false")
    if "gle" in parameters:
        if not isinstance(parameters["gle"], Mapping):
            raise ValueError("GLE parameters must be a table")
        gle = copy.deepcopy(dict(parameters["gle"]))
        from ..units import unit_factor
        factor = unit_factor(gle.pop("A_unit", "autime^-1"), "frequency")
        drift = np.asarray(gle.get("A"))
        if drift.dtype.kind not in "fiu":
            raise ValueError("GLE A must contain real numerical values")
        gle["A"] = np.asarray(drift, float) * factor
        if "beta" in gle:
            gle["beta"] = parse_quantity(gle["beta"], (0, -1, 0, 0, 0, 0))
        parameters["gle"] = gle
    return normalize_thermostat(parameters)


def fes_sampler(config):
    """Resolve canonical umbrella, MetaD and OPES sampling without mutation.

    Andersen scheduling/RNG belongs to the caller. Existing PILE-L inputs
    preserve their established NumPy stream. Other canonical thermostats use
    the common operator state, which the workflow checkpoints per stream.
    """
    validate_andersen_schedule(config)
    sampling = config.get("sampling", {})
    dynamics = config.get("dynamics", {})
    scheme = normalize_integrator(dynamics.get("integrator", "cayley-bcb"))
    if (dynamics.get("loop_backend", "auto") == "native"
            and config.get("native", dynamics.get("native", True)) is False):
        raise ValueError("native loop sampling cannot disable the native integrator")
    thermostat = sampling.get("thermostat", "andersen")
    if thermostat == "andersen":
        expected = scheme + "-andersen"
        if sampling.get("integrator", expected) != expected:
            raise ValueError("Andersen FES sampling must match the dynamics integrator")
        if "gamma_centroid" in sampling:
            raise ValueError("sampling.gamma_centroid requires PILE-L sampling")
        return FESSampler(thermostat, expected)
    if thermostat not in ("pile-l", "pile-g", "langevin", "svr", "gle",
                           "nm-gle", "nm-gle-g", "ffl", "nhc", "nose-hoover-chain"):
        raise ValueError("biased equilibrium sampling requires a canonical thermostat; none, TRPMD and PIGLET do not define its canonical measure")
    if config.get("task", "fes") not in ("fes", "metadynamics", "opes"):
        raise ValueError("canonical biased sampling requires task='fes', 'metadynamics' or 'opes'")
    nhc = thermostat in ("nhc", "nose-hoover-chain")
    expected = ({"cayley-bcb": "obcbo", "normal-mode-bab": "obabo", "velocity-verlet": "obabo"}
                if nhc else {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab", "velocity-verlet": "obabo"}).get(scheme)
    if thermostat == "pile-l" and scheme == "velocity-verlet":
        expected = None
    if expected is None or sampling.get("integrator", expected) != expected:
        if thermostat == "pile-l":
            raise ValueError("PILE-L requires BCOCB with Cayley or BAOAB with normal-mode dynamics")
        raise ValueError(f"{thermostat} sampling.integrator must be {expected}")
    if thermostat != "pile-l" or sampling.get("thermostat_params"):
        spec = thermostat_spec(sampling)
        if spec["mode"] in ("gle", "nm-gle", "nm-gle-g"):
            from ..native_ensemble import _gle_matrices
            # Check the canonical physical covariance before allocating output
            # or asking a provider for forces. Noncanonical GLE is not an MBAR
            # umbrella measure, even when its drift and diffusion are stable.
            _gle_matrices(spec["gle"], spec["mode"], int(config["beads"]),
                          float(config["beta"]), float(sampling.get("dt_au", config["dt_au"])))
        return FESSampler(spec["mode"], expected, thermostat_spec=spec,
                          operator_checkpoint=True)
    gamma = parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("sampling.gamma_centroid must be positive and finite")
    return FESSampler(thermostat, expected, float(gamma),
                      operator_checkpoint=dynamics.get("loop_backend", "auto") == "native")


def legacy_metadynamics_sampling(sampling):
    """Keep established MetaD maps, chain resets and random streams intact."""
    mode = sampling.get("thermostat", "andersen")
    parameters = sampling.get("thermostat_params", {})
    # Existing MetaD jobs may retain the NHC parameter table when switching
    # their thermostat to PILE-L. That table was ignored by the legacy path.
    composed_pile = isinstance(parameters, Mapping) and bool(
        set(parameters) & {"gamma_centroid", "pile_scale"})
    return mode in ("andersen", "nose-hoover-chain", "pile-l") and not (
        mode == "pile-l" and composed_pile)


def read_operator_checkpoint(data, sampler, elapsed, workflow):
    """Read optional legacy state, requiring bath state for composed sampling."""
    import json
    if "operator_state" not in data:
        if sampler is not None and sampler.operator_checkpoint and elapsed > 0:
            raise ValueError(f"{workflow} checkpoint is missing propagation operator state")
        return None
    saved = json.loads(str(data["operator_state"]))
    if not isinstance(saved, dict):
        raise ValueError(f"{workflow} checkpoint has invalid propagation operator state")
    if sampler is not None and sampler.operator_checkpoint and elapsed > 0 and not saved.get("sampling"):
        raise ValueError(f"{workflow} checkpoint is missing thermostat auxiliary state")
    return saved
