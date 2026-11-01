"""Deterministic canonical sampling with an explicitly checkpointed heat bath.

Nosé--Hoover chains: Martyna, Klein & Tuckerman, JCP 97, 2635 (1992),
doi:10.1063/1.463940. Symmetric chain splitting with the three-stage
Suzuki--Yoshida composition. All quantities use atomic units. The target for
H_P is kT_P=P/beta, with 3*P*N_mobile degrees of freedom (no COM subtraction).
This global chain is a sampling thermostat, never part of real-time RPMD.
"""
from __future__ import annotations

import numbers

import numpy as np

from .units import parse_quantity


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def nhc_options(sampling):
    raw = sampling.get("thermostat_params", {})
    if not isinstance(raw, dict) or set(raw) - {"tdamp", "chain_length", "substeps", "reset_after_equilibration"}:
        raise ValueError("NHC thermostat_params accepts tdamp, chain_length, substeps and reset_after_equilibration")
    if not isinstance(raw.get("reset_after_equilibration", False), bool):
        raise ValueError("NHC reset_after_equilibration must be boolean")
    tau = parse_quantity(raw.get("tdamp", "20 fs"), "time")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("NHC tdamp must be finite and positive")
    return {"tdamp": tau, "chain_length": _integer(raw.get("chain_length", 3), "chain_length"),
            "substeps": _integer(raw.get("substeps", 1), "substeps")}


class NoseHooverChain:
    """One chain coupled to all mobile bead momenta of a Dynamics instance.

    ``step`` wraps a deterministic map symmetrically and restores both the
    physical state and chain on failure. ``state_dict`` is JSON serializable.
    Fixed atoms neither contribute to the bath mass nor get thermal momenta.
    """

    def __init__(self, dynamics, *, tdamp, chain_length=3, substeps=1):
        self.dynamics = dynamics
        self.length = _integer(chain_length, "chain_length")
        self.substeps = _integer(substeps, "substeps")
        self.tdamp = float(tdamp)
        if not np.isfinite(self.tdamp) or self.tdamp <= 0:
            raise ValueError("NHC tdamp must be finite and positive")
        self.mobile = np.ones(len(dynamics.masses), dtype=bool)
        self.mobile[list(dynamics.fixed_atoms)] = False
        self.dof = 3 * dynamics.nbeads * int(self.mobile.sum())
        if self.dof == 0:
            raise ValueError("NHC requires at least one mobile atom")
        self.kT = dynamics.nbeads / dynamics.beta
        self.Q = np.full(self.length, self.kT * self.tdamp**2)
        self.Q[0] *= self.dof
        self.eta = np.zeros(self.length)
        self.momentum = np.zeros(self.length)

    def state_dict(self):
        return {"eta": self.eta.tolist(), "momentum": self.momentum.tolist(),
                "dof": self.dof, "kT": self.kT, "tdamp": self.tdamp,
                "substeps": self.substeps}

    def load_state_dict(self, data):
        if (not isinstance(data, dict) or set(data) != {"eta", "momentum", "dof", "kT", "tdamp", "substeps"}
                or any(data[k] != getattr(self, k) for k in ("dof", "kT", "tdamp", "substeps"))):
            raise ValueError("NHC checkpoint parameters differ")
        eta, momentum = np.asarray(data["eta"], float), np.asarray(data["momentum"], float)
        if eta.shape != (self.length,) or momentum.shape != eta.shape or not np.isfinite([eta, momentum]).all():
            raise ValueError("NHC checkpoint variables must be finite and match chain_length")
        self.eta[:], self.momentum[:] = eta, momentum

    def energy(self):
        """Bath contribution to the conserved extended energy, in hartree."""
        return float(self.kT * (self.dof * self.eta[0] + self.eta[1:].sum())
                     + 0.5 * np.sum(self.momentum**2 / self.Q))

    def _kick_bath(self, p, j, h):
        damping = np.exp(-0.5 * h * self.momentum[j + 1] / self.Q[j + 1]) if j + 1 < self.length else 1.0
        force = (np.sum(p[:, self.mobile]**2 / self.dynamics.masses[None, self.mobile, None])
                 - self.dof * self.kT) if j == 0 else self.momentum[j - 1]**2 / self.Q[j - 1] - self.kT
        self.momentum[j] = damping * (damping * self.momentum[j] + h * force)

    def _bath(self, p, dt):
        w = 1.0 / (2.0 - 2.0**(1.0 / 3.0))
        for _ in range(self.substeps):
            for coefficient in (w, 1.0 - 2.0 * w, w):
                h = coefficient * dt / self.substeps
                for j in reversed(range(self.length)):
                    self._kick_bath(p, j, 0.5 * h)
                self.eta += h * self.momentum / self.Q
                p[:, self.mobile] *= np.exp(-h * self.momentum[0] / self.Q[0])
                for j in range(self.length):
                    self._kick_bath(p, j, 0.5 * h)

    def step(self, state, bias=None):
        dyn = self.dynamics
        dyn._check_parameters()
        dyn.integrator._check(state.q, state.p)
        dyn._check_fixed(state.q, state.p)
        if dyn.dt <= 0:
            raise ValueError("NHC sampling requires positive dt")
        previous, bath = state.copy(), self.state_dict()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self._bath(state.p, 0.5 * dyn.dt)
                dyn.step(state, bias)
                self._bath(state.p, 0.5 * dyn.dt)
                if not np.isfinite(state.p).all() or not np.isfinite([self.eta, self.momentum]).all():
                    raise FloatingPointError("nonfinite NHC state; reduce timestep or increase tdamp")
        except BaseException:
            state.q[:], state.p[:] = previous.q, previous.p
            state.step, state.time = previous.step, previous.time
            self.load_state_dict(bath)
            raise
        return state


class PythonNHCBath:
    """Pure-Python NHC operator for installations without ``_thermal``.

    This adapts the existing chain bath, without calling ``Dynamics.step``.
    Its distinct checkpoint format keeps a restored Python chain on the same
    implementation even if a compiled extension is subsequently installed.
    """

    checkpoint_format = "pyrpmd.python-nhc-bath.v1"

    def __init__(self, dynamics, q, duration, thermostat):
        from .engine import normalize_thermostat
        spec = normalize_thermostat(thermostat)
        if spec["mode"] != "nhc":
            raise ValueError("Python NHC bath requires thermostat mode nhc")
        if (isinstance(duration, (bool, np.bool_)) or not isinstance(duration, numbers.Real)
                or np.iscomplexobj(duration) or not np.isfinite(duration) or duration <= 0):
            raise ValueError("NHC bath duration must be positive and finite")
        self.dynamics = dynamics
        self.duration = float(duration)
        self.shape = (dynamics.nbeads, len(dynamics.masses), 3)
        self.q = self._coordinates(q)
        dynamics._check_fixed(self.q, np.zeros_like(self.q))
        self.q.flags.writeable = False
        self.chain = NoseHooverChain(
            dynamics, tdamp=spec.get("tau", 1.0),
            chain_length=spec.get("chain_length", 3), substeps=spec.get("substeps", 1),
        )
        self.identity = {
            "masses": dynamics.masses.tolist(), "beta": dynamics.beta,
            "beads": dynamics.nbeads, "fixed_atoms": list(dynamics.fixed_atoms),
            "duration": self.duration,
        }

    def _coordinates(self, q):
        values = np.asarray(q)
        if values.dtype.kind not in "fiu" or values.shape != self.shape or not np.isfinite(values).all():
            raise ValueError("NHC bath coordinates must be finite real [P,N,3] arrays")
        return np.array(values, dtype=np.float64, order="C", copy=True)

    def state(self):
        import copy
        return {"format": self.checkpoint_format, "identity": copy.deepcopy(self.identity),
                "q": self.q.copy(), "chain": self.chain.state_dict()}

    def restore(self, state):
        if (not isinstance(state, dict) or set(state) != {"format", "identity", "q", "chain"}
                or state["format"] != self.checkpoint_format or state["identity"] != self.identity):
            raise ValueError("Python NHC checkpoint parameters or format differ")
        q = self._coordinates(state["q"])
        self.dynamics._check_fixed(q, np.zeros_like(q))
        chain = state["chain"]
        if not isinstance(chain, dict):
            raise ValueError("invalid Python NHC chain checkpoint")
        for key in ("eta", "momentum"):
            value = np.asarray(chain.get(key))
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("NHC checkpoint chain variables must be finite real arrays")
        # The existing loader validates every parameter and both arrays before
        # committing either chain variable. Coordinates are a detached reference.
        self.chain.load_state_dict(chain)
        q.flags.writeable = False
        self.q = q

    def apply_thermostat(self, q, p, dt=None, *, step=None):
        """Apply a bath substep atomically; no random draw or force request."""
        duration = self.duration if dt is None else dt
        if (isinstance(duration, (bool, np.bool_)) or not isinstance(duration, numbers.Real)
                or np.iscomplexobj(duration) or not np.isfinite(duration) or duration <= 0):
            raise ValueError("NHC bath duration must be positive and finite")
        if step is not None and (isinstance(step, (bool, np.bool_))
                or not isinstance(step, (int, np.integer)) or step < 0):
            raise ValueError("NHC bath logical step must be a nonnegative integer")
        coordinates = self._coordinates(q)
        if (not isinstance(p, np.ndarray) or p.dtype != np.dtype(np.float64)
                or p.shape != self.shape or not p.flags.writeable or not np.isfinite(p).all()):
            raise ValueError("NHC bath momenta must be a writable finite float64 [P,N,3] array")
        if np.shares_memory(np.asarray(q), p):
            raise ValueError("NHC bath coordinates and momenta must not overlap")
        self.dynamics._check_parameters()
        self.dynamics._check_fixed(coordinates, p)
        previous = self.chain.state_dict()
        candidate = p.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self.chain._bath(candidate, float(duration))
                if not np.isfinite(candidate).all() or not np.isfinite([self.chain.eta, self.chain.momentum]).all():
                    raise FloatingPointError("nonfinite NHC state; reduce duration or increase tdamp")
            p[:] = candidate
        except BaseException:
            self.chain.load_state_dict(previous)
            raise
        return p
