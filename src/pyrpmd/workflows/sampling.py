"""Shared FES thermostat validation; no independent propagation implementation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..dynamics import normalize_integrator
from ..units import parse_quantity


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
