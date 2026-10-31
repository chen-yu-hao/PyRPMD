"""Markovian centroid electronic-friction RPMD (EF-RPMD).

The physical friction tensor eta has units mass / atomic time and produces
force -eta @ velocity.  It is averaged over beads, then acts on the centroid
only.  Every bead receives the same correlated momentum increment.  The
electronic bath is at the nuclear temperature; its exact OU map satisfies
<p_centroid p_centroid.T> = M / beta, independently of the bead count.

This is the centroid approximation of Bi and Dou, JCP 160, 074110 (2024),
not the bath-renormalized, non-Markovian path-integral GLE of Trenins and
Rossi, PRL 134, 226201 (2025).  Internal modes receive no added thermostat.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .dynamics import Dynamics
from .potentials.base import PotentialError, coordinates
from .units import parse_quantity


@dataclass
class FrictionResult:
    """Symmetric positive-semidefinite eta[B,3*N,3*N], in electron-mass/autime.

    Cartesian order is atom-major: x0,y0,z0,x1,y1,z1,... .  Nuclear forces,
    inverse-time relaxation rates, and mass-weighted tensors are not eta.
    """
    tensor: np.ndarray
    status: np.ndarray | None = None
    messages: tuple[str, ...] | None = None


def validate_friction_result(result, shape):
    """Validate the entire tensor before any momentum or RNG is changed."""
    if not isinstance(result, FrictionResult):
        result = FrictionResult(result)
    value = np.asarray(result.tensor)
    if len(shape) != 3 or shape[-1] != 3 or min(shape) < 1:
        raise PotentialError("friction requires nonempty [B,N,3] coordinates")
    expected = (shape[0], 3 * shape[1], 3 * shape[1])
    if value.shape != expected or value.dtype.kind not in "fiu":
        raise PotentialError(f"friction tensor must be a real {expected} array")
    if not np.isfinite(value).all():
        raise PotentialError("friction tensor must be finite")
    status = np.zeros(shape[0], dtype=int) if result.status is None else np.asarray(result.status)
    if (status.shape != (shape[0],) or status.dtype.kind not in "iu"
            or np.any(status < 0)):
        raise PotentialError("friction status must be nonnegative integers [B]")
    messages = ("",) * shape[0] if result.messages is None else tuple(result.messages)
    if len(messages) != shape[0] or any(not isinstance(m, str) for m in messages):
        raise PotentialError("friction messages must contain one string per frame")
    if np.any(status):
        raise PotentialError(f"friction provider failed in frames {np.flatnonzero(status).tolist()}")
    value = np.array(value, dtype=float, order="C", copy=True)
    scale = np.max(np.abs(value), axis=(1, 2))
    tolerance = 256 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
    if np.any(np.max(np.abs(value - value.swapaxes(1, 2)), axis=(1, 2)) > tolerance):
        raise PotentialError("friction tensor must be symmetric")
    value = 0.5 * (value + value.swapaxes(1, 2))
    if np.any(np.linalg.eigvalsh(value)[:, 0] < -tolerance * value.shape[1]):
        raise PotentialError("friction tensor must be positive semidefinite")
    return FrictionResult(value, status.copy(), messages)


class ConstantElectronicFriction:
    """A constant physical eta tensor; use from_rates for inverse-time gamma.

    The constructor requires an explicit square [3*N,3*N] physical tensor.
    ``from_rates(gamma, masses)`` accepts one scalar rate or one rate per
    Cartesian component, including unit strings such as ``'0.01 fs^-1'``.
    """
    metadata = {"supports_pbc": True, "thread_safety": "stateless",
                "friction_units": "electron_mass/atomic_time"}

    def __init__(self, tensor):
        value = np.asarray(tensor)
        if value.ndim != 2 or value.shape[0] < 3 or value.shape[0] % 3:
            raise ValueError("constant friction needs a [3*N,3*N] physical tensor")
        self._tensor = validate_friction_result(value[None], (1, value.shape[0] // 3, 3)).tensor[0]
        self._tensor.flags.writeable = False

    @classmethod
    def from_rates(cls, gamma, masses):
        masses = np.asarray(masses)
        if (masses.dtype.kind not in "fiu" or masses.ndim != 1 or not len(masses)
                or not np.isfinite(masses).all() or np.any(masses <= 0)):
            raise ValueError("masses must be positive finite [N]")
        if isinstance(gamma, (str, int, float, np.number)):
            rates = np.full(3 * len(masses), parse_quantity(gamma, "frequency"))
        else:
            raw = np.asarray(gamma, dtype=object)
            if raw.shape not in ((len(masses), 3), (3 * len(masses),)):
                raise ValueError("gamma must be a scalar or [N,3]/[3*N] relaxation rates")
            rates = np.array([parse_quantity(v, "frequency") for v in raw.ravel()])
        if np.any(rates < 0) or not np.isfinite(rates).all():
            raise ValueError("gamma rates must be finite and nonnegative")
        return cls(np.diag(rates * np.repeat(masses, 3)))

    def bind_system(self, system):
        if self._tensor.shape != (3 * system.natoms, 3 * system.natoms):
            raise ValueError("constant friction atom count differs from system")

    def evaluate(self, q):
        q = coordinates(q)
        if self._tensor.shape[0] != 3 * q.shape[1]:
            raise PotentialError("friction tensor atom count mismatch")
        return FrictionResult(np.broadcast_to(self._tensor, (len(q), *self._tensor.shape)))

    def close(self):
        pass


_CONSTANT_FRICTION_EVALUATE = ConstantElectronicFriction.evaluate


class ElectronicFrictionDynamics(Dynamics):
    """O/2--B/2--free ring--B/2--O/2 physical EF-RPMD splitting.

    O is an exact mass-weighted Ornstein--Uhlenbeck centroid step with a
    frozen, bead-averaged tensor.  The ordinary real-mass nuclear propagator
    is reused unchanged.  Fixed atoms are projected out of the tensor.
    Call step/propagate for physical dynamics.  Prepare initial conditions
    with a separate Dynamics sampler on the same scalar/mean-force PES.
    Potential and friction providers are borrowed and remain caller-owned.
    """
    _stochastic_propagation = True
    def __init__(self, potential, masses, beta, nbeads, dt, friction, **kwargs):
        super().__init__(potential, masses, beta, nbeads, dt, **kwargs)
        if self.dt <= 0:
            raise ValueError("electronic friction requires positive dt")
        if self.loop_backend == "native":
            raise ValueError("electronic friction uses Python OU orchestration; use auto/python")
        if not callable(getattr(friction, "evaluate", None)):
            raise TypeError("friction must implement evaluate(q) returning FrictionResult")
        self._friction = friction
        self._constant_evaluate = friction.evaluate
        self._mobile = np.repeat(~np.isin(np.arange(len(self.masses)), self._fixed_indices), 3)
        if not self._mobile.any():
            raise ValueError("electronic friction requires at least one mobile atom")
        self._sqrt_mass = np.sqrt(np.repeat(self.masses, 3)[self._mobile])
        self._constant_eigen = None
        if (type(friction) is ConstantElectronicFriction and "evaluate" not in friction.__dict__
                and ConstantElectronicFriction.evaluate is _CONSTANT_FRICTION_EVALUATE):
            eta = friction.evaluate(np.zeros(self.integrator.shape)).tensor.mean(axis=0)
            self._constant_eigen = self._mass_weighted_eigen(eta)

    @property
    def friction(self):
        """Provider identity is immutable because constant eigensystems are cached."""
        return self._friction

    def _mass_weighted_eigen(self, eta):
        eta = eta[np.ix_(self._mobile, self._mobile)]
        gamma = eta / self._sqrt_mass[:, None] / self._sqrt_mass[None, :]
        rates, vectors = np.linalg.eigh(gamma)
        # Only the roundoff-negative eigenspace allowed by validation is clipped.
        return np.maximum(rates, 0), vectors

    def _centroid_ou(self, q, p, duration):
        if self._constant_eigen is None or self.friction.evaluate != self._constant_evaluate:
            result = validate_friction_result(self.friction.evaluate(q), q.shape)
            rates, vectors = self._mass_weighted_eigen(result.tensor.mean(axis=0))
        else:
            rates, vectors = self._constant_eigen
        if not np.any(rates):
            return  # Exact zero-friction limit also preserves the RNG stream.
        centroid = p.mean(axis=0).ravel()[self._mobile]
        weighted = centroid / self._sqrt_mass
        eigen_momentum = vectors.T @ weighted
        decay = np.exp(-rates * duration)
        amplitude = np.sqrt(-np.expm1(-2 * rates * duration) / self.beta)
        updated = vectors @ (decay * eigen_momentum + amplitude * self.rng.normal(size=rates.shape))
        increment = self._sqrt_mass * (updated - weighted)
        flat = p.reshape(self.nbeads, -1)
        flat[:, self._mobile] += increment

    def step(self, state, bias=None):
        self._check_parameters()
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)
        trial = state.copy()
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            self._centroid_ou(trial.q, trial.p, 0.5 * self.dt)
            Dynamics._propagate(self, trial, bias)
            self._centroid_ou(trial.q, trial.p, 0.5 * self.dt)
            self._project_fixed(trial.q, trial.p)
            self.integrator._check(trial.q, trial.p)
            self._check_fixed(trial.q, trial.p)
            if not np.isfinite(trial.time):
                raise FloatingPointError("nonfinite EF-RPMD integration clock")
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        state.q[:] = trial.q
        state.p[:] = trial.p
        state.step, state.time = trial.step, trial.time
        return state

    def _propagate(self, state, bias=None, fragment_data=()):
        if fragment_data:
            raise ValueError("EF-RPMD does not support additional fragment constraints")
        return self.step(state, bias)

    def propagate(self, state, steps, **options):
        # The public function also owns the RNG transaction, so direct imports
        # of pyrpmd.propagation.propagate have the same failure semantics.
        return super().propagate(state, steps, **options)

    def propagate_block(self, state, steps, **options):
        raise NotImplementedError("use step() or transactional propagate() for stochastic EF-RPMD")

    def sample_step(self, *args, **kwargs):
        raise ValueError("use EF-RPMD step(); prepare equilibrium with a separate Dynamics sampler")

    def thermalize(self, *args, **kwargs):
        raise ValueError("prepare equilibrium with a separate Dynamics sampler on the same PES")


_FIELDS = {"method", "provider", "gamma", "tensor", "factory", "params"}


def validate_electronic_friction(config, system):
    """Check the public opt-in, including direct resolved-workflow calls."""
    if "electronic_friction" not in config:
        return None
    spec = config["electronic_friction"]
    if not isinstance(spec, Mapping) or not spec or set(spec) - _FIELDS:
        raise ValueError("electronic_friction must be a nonempty table with known fields")
    if config.get("task") != "diffusion":
        raise ValueError("electronic_friction is supported by task='diffusion'; use the Python API for other protocols")
    if "nonadiabatic" in config:
        raise ValueError("electronic friction uses a scalar PES and cannot be combined with mapping dynamics")
    if spec.get("method", "centroid") != "centroid":
        raise ValueError("electronic_friction.method must be 'centroid' (Markovian EF-RPMD)")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise ValueError("electronic friction requires loop_backend='auto' or 'python'")
    provider = spec.get("provider", "constant")
    if provider == "constant":
        if "factory" in spec or "params" in spec or ("gamma" in spec) == ("tensor" in spec):
            raise ValueError("constant electronic_friction requires exactly one of gamma or tensor")
        constant = (ConstantElectronicFriction.from_rates(spec["gamma"], system.masses)
                    if "gamma" in spec else ConstantElectronicFriction(spec["tensor"]))
        constant.bind_system(system)
    elif provider == "python":
        if "gamma" in spec or "tensor" in spec or not isinstance(spec.get("factory"), str) or ":" not in spec["factory"]:
            raise ValueError("Python electronic_friction requires factory='module:object' and optional params")
        if not isinstance(spec.get("params", {}), Mapping):
            raise ValueError("electronic_friction.params must be a table")
    else:
        raise ValueError("electronic_friction.provider must be constant or python")
    if len(system.fixed_atoms) == system.natoms:
        raise ValueError("electronic friction requires mobile atoms")
    return dict(spec)


def make_electronic_friction(config, system):
    """Construct one provider per trajectory worker and bind its periodic cell."""
    spec = validate_electronic_friction(config, system)
    if spec is None:
        return None
    if spec.get("provider", "constant") == "constant":
        obj = (ConstantElectronicFriction.from_rates(spec["gamma"], system.masses)
               if "gamma" in spec else ConstantElectronicFriction(spec["tensor"]))
    else:
        from .potentials.python import import_object
        obj = import_object(spec["factory"])(**spec.get("params", {}))
    try:
        if not callable(getattr(obj, "evaluate", None)):
            raise TypeError("friction factory must return an object with evaluate(q)")
        bind = getattr(obj, "bind_system", None)
        if any(system.pbc) and (getattr(obj, "metadata", {}).get("supports_pbc") is not True or not callable(bind)):
            raise ValueError("periodic friction requires metadata['supports_pbc']=True and bind_system(system)")
        if callable(bind):
            bind(system)
        return obj
    except BaseException:
        close = getattr(obj, "close", None)
        if callable(close):
            close()
        raise
