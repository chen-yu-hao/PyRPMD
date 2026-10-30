"""Native block propagation and conservative Python compatibility dispatch.

The direct path accepts ONLY explicitly exported stateless native PES bindings
and exact supported built-in CV/bias types. A subclass is not silently treated
as its parent: overridden Python mathematics must remain in the Python path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import numbers
import numpy as np


class NativeUnavailableError(RuntimeError):
    """Requested native semantics cannot be provided without Python callbacks."""


@dataclass
class NativePESBinding:
    """Trusted process-local native function, context, lifetime owner and lock.

    kind=0: PyRPMD PES C ABI v1; kind=1: Fortran batch BIND(C) signature.
    Never pass a ctypes/Python callback: these functions run with the GIL freed.
    All ordinary evaluate/close calls must use the same lock. Recreated for each
    block, not cached on owner, to avoid owner/capsule reference cycles.
    """
    owner: object
    function: object
    lock: object
    natoms: int
    kind: int = 0
    context: object = 0
    is_open: object = None

    @contextmanager
    def acquire(self):
        from . import _native
        from .potentials.native import descriptor_binding
        with descriptor_binding(self).acquire() as (_, address, context, natoms):
            capsule = _native.bind_pes(
                self.kind, address, context, natoms, self.owner
            )
            yield capsule


def _atomic_binding(binding):
    """Accept both public bindings while retaining owner, lock and open check."""
    if isinstance(binding, NativePESBinding):
        return binding
    from .potentials.native import NativePESBinding as DescriptorBinding
    if isinstance(binding, DescriptorBinding):
        kinds = {"cabi-v1": 0, "fortran-batch-v1": 1}
        if binding.kind not in kinds:
            raise NativeUnavailableError("unsupported native PES binding version")
        return NativePESBinding(
            binding.owner, binding.function, binding.lock, binding.natoms,
            kind=kinds[binding.kind], context=binding.context, is_open=binding.is_open,
        )
    raise NativeUnavailableError("PES has no usable stateless native binding")


def _check_dynamics_methods(dynamics):
    from .dynamics import Dynamics, CayleyIntegrator
    methods = ("step", "_propagate", "_force", "energy", "_project_fragments", "_fragment_data")
    if type(dynamics) is not Dynamics or any(name in dynamics.__dict__ for name in methods):
        raise NativeUnavailableError("custom Dynamics methods require the Python path")
    integrator = dynamics.integrator
    if type(integrator) is not CayleyIntegrator or any(
        name in integrator.__dict__ for name in ("free_step", "kick", "_fourier_step", "spring_force")
    ):
        raise NativeUnavailableError("custom integrator methods require the Python path")


@dataclass
class PropagationResult:
    steps: np.ndarray
    energies: np.ndarray
    centroids: np.ndarray | None
    values: np.ndarray | None
    initial_energy: float
    minimum_energy: float
    maximum_energy: float
    backend: str
    fallback_reason: str | None = None


def _primitive(n, *, constant=0.0, terms=(), linear=None):
    return {"constant": float(constant),
            "linear": np.zeros(3*n).tolist() if linear is None else np.asarray(linear, float).reshape(-1).tolist(),
            "norms": [{"factor": float(a), "weights": np.asarray(w, float).tolist()} for a, w in terms]}


def _bond(n, atoms):
    a, b = atoms
    if a == b or min(a, b) < 0 or max(a, b) >= n:
        raise ValueError("distance requires two distinct in-range atoms")
    w = np.zeros(n); w[a] = 1.; w[b] = -1.
    return w


def _check_cv_methods(cv):
    methods = ("value", "value_and_jacobian", "centroid_value_gradient",
               "centroid_value_gradient_hessian", "surfaces")
    if any(k in getattr(cv, "__dict__", {}) for k in methods):
        raise NativeUnavailableError("overridden CV methods require Python")


def cv_descriptor(cv, n):
    from .cv import Distance, BondDifference, Cartesian, COMDistance, RPMDrateCoordinate, RateSurface
    if getattr(cv, "estimator", None) != "centroid":
        raise NativeUnavailableError("native CV path requires centroid estimator")
    if any(p is not None for p in cv.periods):
        raise NativeUnavailableError("periodic CV uses Python compatibility path")
    _check_cv_methods(cv)
    t = type(cv)
    if t is Distance:
        primitive = _primitive(n, terms=[(1., _bond(n, cv.atoms))])
    elif t is BondDifference:
        primitive = _primitive(n, terms=[(1., _bond(n, cv.bonds[0])), (-1., _bond(n, cv.bonds[1]))])
    elif t is COMDistance:
        if len(cv.weights) != n:
            raise ValueError("COM atom count mismatch")
        primitive = _primitive(n, terms=[(1., cv.weights)])
    elif t is Cartesian:
        if not 0 <= cv.atom < n:
            raise ValueError("Cartesian atom index out of range")
        g = np.zeros((n, 3)); g[cv.atom, cv.axis] = 1.
        primitive = _primitive(n, linear=g)
    elif t in (RPMDrateCoordinate, RateSurface):
        c = cv.coordinate if t is RateSurface else cv
        if type(c) is not RPMDrateCoordinate or type(c.com) is not COMDistance:
            raise NativeUnavailableError("custom rate coordinate requires Python")
        _check_cv_methods(c)
        _check_cv_methods(c.com)
        if len(c.com.weights) != n:
            raise ValueError("rate coordinate atom count mismatch")
        result = {"kind": 2 if t is RateSurface else 1,
                  "entry": _primitive(n, constant=c.rinf, terms=[(-1., c.com.weights)]),
                  "paths": [[_primitive(n, constant=ref, terms=[(-1., _bond(n, f)), (1., _bond(n, b))])
                             for ref, f, b in zip(c.reference_difference, forming, breaking)]
                            for forming, breaking in c.paths]}
        if t is RateSurface:
            result["xi"] = float(cv.xi)
        return result
    else:
        raise NativeUnavailableError(f"{t.__name__} requires Python CV evaluation")
    return {"kind": 0, "primitive": primitive}


def _cvs(cv, n):
    from .cv import CVStack
    _check_cv_methods(cv)
    return [cv_descriptor(c, n) for c in (cv.cvs if type(cv) is CVStack else (cv,))]


def _biases(bias, n, masses):
    from .bias import UmbrellaBias, FluxMetricBias, CompositeBias
    if bias is None:
        return []
    if "energy_force" in getattr(bias, "__dict__", {}):
        raise NativeUnavailableError("overridden bias requires Python")
    if type(bias) is CompositeBias:
        return [item for b in bias.biases for item in _biases(b, n, masses)]
    if type(bias) is UmbrellaBias:
        return [{"kind": 0, "cvs": _cvs(bias.cv, n), "center": bias.center.tolist(),
                 "stiffness": bias.stiffness.reshape(-1).tolist()}]
    if type(bias) is FluxMetricBias:
        if not np.array_equal(bias.masses, masses):
            raise NativeUnavailableError("metric masses differ from integrator")
        from .cv import RPMDrateCoordinate
        if type(bias.cv) is not RPMDrateCoordinate:
            raise NativeUnavailableError("native flux metric requires RPMDrateCoordinate")
        return [{"kind": 1, "cvs": _cvs(bias.cv, n), "beta": float(bias.beta)}]
    raise NativeUnavailableError(f"{type(bias).__name__} requires Python bias evaluation")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def propagate(dynamics, state, steps, *, bias=None, observe=None, sample_interval=1,
              record_centroids=True, backend=None, fragment_data=()):
    """Advance a transactional block with the explicitly selected ring map.

    Every physical step is checked for finite energy, even when observations are
    decimated. The final step is always sampled. On failure q/p/clock are unchanged
    for the ENTIRE block. No thermostat/RNG or constraint method is substituted.
    """
    steps = _integer(steps, "steps")
    stride = _integer(sample_interval, "sample_interval")
    backend = dynamics.loop_backend if backend is None else backend
    if backend not in ("auto", "native", "python"):
        raise ValueError("loop backend must be auto/native/python")
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    offsets = np.arange(stride, steps + 1, stride, dtype=np.int64)
    if not len(offsets) or offsets[-1] != steps:
        offsets = np.append(offsets, steps)
    rows, n = len(offsets), len(dynamics.masses)
    dimension = 0 if observe is None else _integer(observe.dimension, "CV dimension")
    es = np.empty(rows, dtype=float)
    cs = np.empty((rows, n, 3), dtype=float) if record_centroids else None
    vs = np.empty((rows, dimension), dtype=float) if observe is not None else None
    reason = None
    binding = None
    if backend != "python":
        try:
            _check_dynamics_methods(dynamics)
            if dynamics.integrator._native is None or not hasattr(dynamics.integrator._native, "run_block"):
                raise NativeUnavailableError(dynamics.integrator.native_fallback_reason or "fused extension unavailable or native=False")
            factory = getattr(dynamics.potential, "native_binding", None)
            if factory is None:
                raise NativeUnavailableError("PES has no explicit native binding")
            binding = _atomic_binding(factory())
            spec = {"observers": [] if observe is None else _cvs(observe, n),
                    "biases": _biases(bias, n, dynamics.masses),
                    "fragments": [{"atoms": ix.tolist(), "reference": ref.tolist()}
                                  for ix, _m, _total, ref in fragment_data],
                    "time": float(state.time)}
        except NativeUnavailableError as exc:
            if backend == "native":
                raise
            reason = str(exc)
            binding = None
    else:
        reason = "explicit Python loop backend"
    initial_step = state.step
    if binding is not None:
        q, p = state.q.copy(), state.p.copy()
        with binding.acquire() as capsule, dynamics.integrator._lock:
            initial, minimum, maximum, clock = dynamics.integrator._native.run_block(
                dynamics.integrator._context, capsule, q, p, steps, stride, es, cs, vs, spec
            )
        state.q[:] = q; state.p[:] = p
        state.step += steps; state.time = clock
        used = "native"
    else:
        trial = state.copy()
        initial = dynamics.energy(trial, bias)
        minimum, maximum = np.inf, -np.inf
        sample = 0
        for i in range(1, steps + 1):
            if fragment_data:
                dynamics._propagate(trial, bias, fragment_data)
            else:
                dynamics.step(trial, bias)
            e = dynamics.energy(trial, bias)
            if not np.isfinite(e):
                raise FloatingPointError("nonfinite ring-polymer energy")
            minimum, maximum = min(minimum, e), max(maximum, e)
            if i == offsets[sample]:
                es[sample] = e
                if cs is not None:
                    cs[sample] = trial.q.mean(axis=0)
                if vs is not None:
                    value = np.asarray(observe.value(trial.q), dtype=float)
                    if value.shape != (dimension,) or not np.isfinite(value).all():
                        raise ValueError("CV returned invalid observation")
                    vs[sample] = value
                sample += 1
        state.q[:] = trial.q; state.p[:] = trial.p
        state.step, state.time = trial.step, trial.time
        used = "python"
    dynamics.propagation_stats[used + "_steps"] += steps
    if reason:
        reasons = dynamics.propagation_stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + steps
    return PropagationResult(initial_step + offsets, es, cs, vs, float(initial),
                             float(minimum), float(maximum), used, reason)


def loop_options(config):
    """Integrator and execution choices; only block/backend are hash-exempt."""
    from .dynamics import normalize_integrator
    options = config.get("dynamics", {})
    return {"loop_backend": options.get("loop_backend", "auto"),
            "block_steps": options.get("block_steps", 256),
            "integrator": normalize_integrator(options.get("integrator", "cayley-bcb"))}


def native_supported(dynamics, *, bias=None, observe=None):
    if dynamics.loop_backend == "python":
        return False
    try:
        _check_dynamics_methods(dynamics)
        if dynamics.integrator._native is None or not hasattr(dynamics.integrator._native, "run_block"):
            raise NativeUnavailableError(dynamics.integrator.native_fallback_reason or "fused extension unavailable or native=False")
        factory = getattr(dynamics.potential, "native_binding", None)
        if factory is None:
            raise NativeUnavailableError("PES has no explicit stateless native binding")
        _atomic_binding(factory())
        n = len(dynamics.masses)
        _biases(bias, n, dynamics.masses)
        if observe is not None:
            _cvs(observe, n)
    except NativeUnavailableError:
        if dynamics.loop_backend == "native":
            raise
        return False
    return True


def scientific_config(config):
    """Remove block/backend tuning without dropping any scientific setting."""
    from copy import deepcopy
    result = deepcopy(config)
    if "dynamics" in result:
        result["dynamics"].pop("loop_backend", None)
        result["dynamics"].pop("block_steps", None)
        if not result["dynamics"]:
            result.pop("dynamics")
    return result


# The valid-prefix API is additive; propagate() retains whole-block rollback.
from .block_propagation import BlockResult, PropagationError, propagate_block
