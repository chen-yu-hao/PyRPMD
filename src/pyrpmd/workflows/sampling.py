"""Shared sampling validation and checkpoint identities; no propagation loop."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from ..dynamics import normalize_integrator
from ..units import parse_quantity


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
