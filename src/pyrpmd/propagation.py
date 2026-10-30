"""Chunked propagation with native fast paths and faithful Python fallbacks.

Only explicitly recognized, unmodified built-in CV/bias classes are lowered.
Arbitrary Python PES/CV/observables retain their Python execution path. Runtime
PES failures NEVER trigger a retry via another backend. Each completed step,
including its requested observations, is transactional.
"""
from dataclasses import dataclass
import numpy as np

from . import cv as cv_module
from .bias import UmbrellaBias, FluxMetricBias, CompositeBias
from .potentials.native import NativePESBinding


class NotNative(ValueError):
    """A capability is not available; this is not a dynamics failure."""


@dataclass
class BlockResult:
    requested: int
    completed: int
    backend: str
    fallback_reason: str | None
    steps: np.ndarray
    times: np.ndarray
    energies: np.ndarray
    centroids: np.ndarray
    centroid_momenta: np.ndarray
    values: np.ndarray
    q: np.ndarray | None
    p: np.ndarray | None
    error: str | None = None

    @property
    def success(self):
        return self.error is None and self.completed == self.requested


class PropagationError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"{result.backend} propagation failed after {result.completed}/{result.requested} steps: {result.error}")


def _pair(atoms, natoms):
    a, b = atoms
    if a == b or min(a, b) < 0 or max(a, b) >= natoms:
        raise ValueError("native CV atom indices are out of range")
    weights = np.zeros(natoms)
    weights[a], weights[b] = 1.0, -1.0
    return weights.tolist()


def cv_descriptor(cv, natoms):
    """A value/derivative-preserving lowering; subclasses deliberately fall back."""
    if any(name in getattr(cv, "__dict__", {}) for name in
           ("value", "value_and_jacobian", "centroid_value_gradient", "centroid_value_gradient_hessian")):
        raise NotNative("an instance overrides its CV evaluation")
    tag = {"estimator": getattr(cv, "estimator", "centroid")}
    if type(cv) is cv_module.Distance:
        return dict(tag, kind="norm", weights=_pair(cv.atoms, natoms))
    if type(cv) is cv_module.COMDistance:
        if len(cv.weights) != natoms:
            raise ValueError("COM CV size mismatch")
        return dict(tag, kind="norm", weights=cv.weights.tolist())
    if type(cv) is cv_module.Cartesian:
        if cv.atom < 0 or cv.atom >= natoms:
            raise ValueError("Cartesian CV atom out of range")
        weights = np.zeros((natoms, 3)); weights[cv.atom, cv.axis] = 1.0
        return dict(tag, kind="linear", weights=weights.ravel().tolist())
    if type(cv) is cv_module.BondDifference:
        return dict(tag, kind="difference", weights=_pair(cv.bonds[0], natoms), other=_pair(cv.bonds[1], natoms))
    if type(cv) is cv_module.RPMDrateCoordinate:
        return dict(tag, kind="rate", weights=cv.com.weights.tolist(), rinf=cv.rinf,
                    paths=[[{"forming": _pair(f, natoms), "breaking": _pair(b, natoms), "reference": r}
                            for f, b, r in zip(forming, breaking, cv.reference_difference)]
                           for forming, breaking in cv.paths])
    if type(cv) is cv_module.RateSurface:
        return dict(tag, kind="surface", coordinate=cv_descriptor(cv.coordinate, natoms), xi=cv.xi)
    raise NotNative(f"CV {type(cv).__name__} requires the Python path")


def cv_descriptors(cv, natoms):
    if cv is None:
        return []
    if type(cv) is cv_module.CVStack:
        if any(k in cv.__dict__ for k in ("value", "value_and_jacobian")):
            raise NotNative("CVStack evaluation is overridden")
        return [cv_descriptor(c, natoms) for c in cv.cvs]
    return [cv_descriptor(cv, natoms)]


def bias_descriptors(bias, dynamics):
    if bias is None:
        return []
    if "energy_force" in getattr(bias, "__dict__", {}):
        raise NotNative("bias evaluation is overridden")
    if type(bias) is CompositeBias:
        return [d for part in bias.biases for d in bias_descriptors(part, dynamics)]
    if type(bias) is UmbrellaBias:
        return [{"kind": "umbrella", "cvs": cv_descriptors(bias.cv, len(dynamics.masses)),
                 "center": bias.center.tolist(), "stiffness": bias.stiffness.ravel().tolist(),
                 "periods": [0.0 if p is None else float(p) for p in bias.cv.periods]}]
    if type(bias) is FluxMetricBias:
        if bias.beta != dynamics.beta or not np.array_equal(bias.masses, dynamics.masses):
            raise NotNative("flux metric masses/beta differ from the integrator")
        return [{"kind": "flux", "cvs": [cv_descriptor(bias.cv, len(dynamics.masses))]}]
    raise NotNative(f"bias {type(bias).__name__} requires the Python path")


def _native_plan(dynamics, bias, cv, constraint, fragment_data, tolerance, record_energy, clock):
    if not dynamics.native or not hasattr(dynamics.integrator._native, "advance"):
        raise NotNative("compiled fused kernel is not available")
    from .dynamics import Dynamics
    if type(dynamics) is not Dynamics or any(name in dynamics.__dict__ for name in ("step", "_propagate", "_force", "energy")):
        raise NotNative("custom Dynamics methods require the Python path")
    factory = getattr(dynamics.potential, "native_binding", None)
    binding = factory() if callable(factory) else None
    if binding is None:
        raise NotNative(f"PES {type(dynamics.potential).__name__} has no native binding")
    if not isinstance(binding, NativePESBinding) or binding.natoms != len(dynamics.masses):
        raise ValueError("invalid native binding or atom count")
    native_constraint = None if constraint is None else cv_descriptor(constraint, len(dynamics.masses))
    if constraint is not None and (getattr(constraint, "estimator", None) != "centroid" or bias is not None or fragment_data):
        raise NotNative("combined constraints/biases require the Python path")
    fragments = []
    for indices, masses, total, reference in fragment_data:
        weights = np.zeros(len(dynamics.masses));weights[indices] = masses
        fragments.append({"weights": weights.tolist(), "reference": reference.tolist()})
    return binding, {"beta": dynamics.beta, "time": clock, "cvs": cv_descriptors(cv, len(dynamics.masses)),
                     "biases": bias_descriptors(bias, dynamics), "constraint": native_constraint,
                     "constraint_tolerance": tolerance, "fragments": fragments, "record_energy": record_energy}


def propagate_block(dynamics, state, steps, *, bias=None, cv=None, constraint=None,
                    constraint_tolerance=1e-10, fragment_data=(), observe=None,
                    record_energy=True, record_states=False, backend=None, raise_on_error=True):
    """Advance at most 4096 BCB steps; return observations in requested step order.

    ``observe`` is a length-steps boolean mask (default: all steps). A failure
    leaves q/p/clock at the last successfully completed and observed step.
    ``PropagationError.result`` retains the valid prefix. No force cache survives
    a block, so changing bias, constraints, q/p or PES between blocks is safe.
    Blocks must end at thermostat/classifier/checkpoint boundaries, as the
    bundled workflows do. Sampling RNG remains in Python at those boundaries.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or not 0 <= steps <= 4096:
        raise ValueError("steps must be an integer in [0, 4096]")
    steps = int(steps)
    backend = backend or getattr(dynamics, "loop_backend", "auto")
    if backend not in ("auto", "native", "python"):
        raise ValueError("loop backend must be auto, native or python")
    if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0:
        raise ValueError("constraint tolerance must be positive and finite")
    if constraint is not None and (bias is not None or fragment_data):
        raise ValueError("constraint propagation cannot be combined with bias/fragment projections")
    dynamics._check_parameters();dynamics.integrator._check(state.q, state.p)
    if observe is None:
        mask = np.ones(steps, dtype=np.uint8)
    else:
        values = np.asarray(observe)
        if values.shape != (steps,) or values.dtype.kind not in "bui" or np.any((values != 0) & (values != 1)):
            raise ValueError("observe must be a boolean/binary mask with length steps")
        mask = np.array(values, dtype=np.uint8, order="C", copy=True)
    count = int(mask.sum());natoms = len(dynamics.masses);nd = natoms * 3
    dimension = int(cv.dimension) if cv is not None else 0
    if dimension < 0:
        raise ValueError("invalid observation dimension")
    width = 2 + 2 * nd + dimension
    memory = count * (width + (2 * state.q.size if record_states else 0)) * 8
    if memory > 256 * 1024 * 1024:
        raise ValueError("block observation buffers exceed 256 MiB; shorten the block")
    records = np.full((count, width), np.nan)
    snapshots = np.empty((count, 2, *state.q.shape)) if record_states else None
    start_step, start_time = state.step, state.time
    binding = options = None;reason = None
    if backend != "python":
        try:
            binding, options = _native_plan(dynamics, bias, cv, constraint, fragment_data,
                                            constraint_tolerance, record_energy, state.time)
        except NotNative as exc:
            reason = str(exc)
            if backend == "native":
                raise RuntimeError("native loop required but unsupported: " + reason) from exc
    selected = "native" if binding is not None else "python"
    completed = written = 0;error = None
    if steps and selected == "native":
        # The lease pins the owner and prevents concurrent context destruction.
        # The integrator lock protects all use of this Cayley context.
        with binding.acquire() as descriptor, dynamics.integrator._lock:
            completed, written, clock, message = dynamics.integrator._native.advance(
                dynamics.integrator._context, descriptor, state.q, state.p, options, mask, records, snapshots)
        state.step += completed;state.time = clock;error = message or None
    elif steps:
        for i in range(steps):
            previous = state.copy()
            try:
                if constraint is not None:
                    from .workflows.rate import constrained_step
                    constrained_step(dynamics, state, constraint, constraint_tolerance)
                elif fragment_data:
                    dynamics._propagate(state, bias, fragment_data)
                else:
                    if bias is None:
                        dynamics.step(state)
                    else:
                        dynamics.step(state, bias)
                if mask[i]:
                    row = records[written];row[0] = state.time
                    if record_energy:
                        row[1] = dynamics.energy(state, bias)
                        if not np.isfinite(row[1]):
                            raise FloatingPointError("nonfinite total ring energy")
                    row[2:2+nd] = state.q.mean(axis=0).ravel()
                    row[2+nd:2+2*nd] = state.p.mean(axis=0).ravel()
                    if cv is not None:
                        row[2+2*nd:] = cv.value(state.q)
                        if not np.isfinite(row[2+2*nd:]).all():
                            raise FloatingPointError("nonfinite CV observation")
                    if snapshots is not None:
                        snapshots[written, 0] = state.q;snapshots[written, 1] = state.p
            except Exception as exc:
                state.q[:] = previous.q;state.p[:] = previous.p
                state.step, state.time = previous.step, previous.time
                error = f"{type(exc).__name__}: {exc}"
                break
            completed += 1;written += int(mask[i])
    records = records[:written];snapshots = None if snapshots is None else snapshots[:written]
    indices = np.flatnonzero(mask[:completed]) + start_step + 1
    result = BlockResult(steps, completed, selected, reason, indices, records[:, 0], records[:, 1],
                         records[:, 2:2+nd].reshape(-1, natoms, 3),
                         records[:, 2+nd:2+2*nd].reshape(-1, natoms, 3), records[:, 2+2*nd:],
                         None if snapshots is None else snapshots[:, 0],
                         None if snapshots is None else snapshots[:, 1], error)
    stats = getattr(dynamics, "loop_stats", None)
    if stats is not None:
        stats[selected + "_steps"] += completed
        if reason:
            stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + completed
    if error and raise_on_error:
        raise PropagationError(result)
    return result


def execution_options(config):
    d = config.get("dynamics", {})
    return {"loop_backend": d.get("loop_backend", "auto"), "block_steps": d.get("block_steps", 128)}


def scientific_config(config):
    """Execution-only choices do not invalidate a physical checkpoint."""
    result = dict(config)
    if isinstance(result.get("dynamics"), dict):
        d = {k: v for k, v in result["dynamics"].items() if k not in ("loop_backend", "block_steps")}
        if d:
            result["dynamics"] = d
        else:
            result.pop("dynamics")
    return result
