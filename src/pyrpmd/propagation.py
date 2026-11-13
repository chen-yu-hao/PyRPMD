"""Native block propagation and conservative Python compatibility dispatch.

The direct path accepts ONLY explicitly exported stateless native PES bindings
and exact supported built-in CV/bias types. A subclass is not silently treated
as its parent: overridden Python mathematics must remain in the Python path.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .validation import checked_integer as _integer
from .potentials._builtin_native import resolve_native_binding


class NativeUnavailableError(RuntimeError):
    """Requested native semantics cannot be provided without Python callbacks."""


def empty_propagation_stats():
    """Return the JSON shape used for propagation accounting.

    Keeping construction in one place lets workflow summaries expose an
    explicit zero-count record even when a stage completed without advancing
    a trajectory (for example, a disconnected PMF).
    """
    return {"native_steps": 0, "python_steps": 0, "fallback_reasons": {}}


def combine_propagation_stats(records):
    """Combine per-stream propagation counters without mutating inputs."""
    result = empty_propagation_stats()
    for record in records or ():
        if not isinstance(record, Mapping):
            continue
        for key in ("native_steps", "python_steps"):
            value = record.get(key, 0)
            if isinstance(value, (int, float, np.integer, np.floating)):
                result[key] += int(value)
        if "untracked_steps" in record:
            value = record.get("untracked_steps")
            if isinstance(value, (int, float, np.integer, np.floating)):
                result["untracked_steps"] = result.get("untracked_steps", 0) + int(value)
        reasons = record.get("fallback_reasons", {})
        if isinstance(reasons, Mapping):
            for reason, count in reasons.items():
                if isinstance(reason, str) and isinstance(count, (int, float, np.integer, np.floating)):
                    result["fallback_reasons"][reason] = (
                        result["fallback_reasons"].get(reason, 0) + int(count)
                    )
    return result


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
    from .dynamics import (Dynamics, CayleyIntegrator, _BLOCK_DYNAMICS_METHODS,
                           _BLOCK_INTEGRATOR_METHODS)
    if getattr(dynamics, "bond_constraints", ()):
        raise NativeUnavailableError("bond constraints require the Python RATTLE propagation loop")
    if getattr(dynamics, "plan", None) is not None and dynamics.plan.active:
        raise NativeUnavailableError("configured operators require the composed propagation executor")
    if type(dynamics) is not Dynamics or not _original_methods(dynamics, _BLOCK_DYNAMICS_METHODS):
        raise NativeUnavailableError("custom Dynamics methods require the Python path")
    integrator = dynamics.integrator
    if type(integrator) is not CayleyIntegrator or not _original_methods(integrator, _BLOCK_INTEGRATOR_METHODS):
        raise NativeUnavailableError("custom integrator methods require the Python path")
    if (getattr(dynamics, "fixed_atoms", ()) and integrator._native is not None
            and getattr(integrator._native, "FIXED_ATOMS_API", 0) < 1):
        raise NativeUnavailableError("installed native extension lacks fixed-atom projection; rebuild it")


def fixed_descriptor(dynamics):
    """One constraint descriptor for both native block transaction policies."""
    if not dynamics.fixed_atoms:
        return {}
    return {"fixed_atoms": list(dynamics.fixed_atoms),
            "fixed_positions": dynamics.fixed_positions.reshape(-1).tolist()}


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


def _original_methods(obj, originals):
    return all(name not in obj.__dict__ and getattr(type(obj), name, None) is original
               for name, original in originals.items())


def _check_cv_methods(cv):
    from .cv import _BUILTIN_CV_METHODS
    methods = ("value", "value_and_jacobian", "centroid_value_gradient",
               "centroid_value_gradient_hessian", "centroid_value_gradient_hvp", "surfaces")
    if (any(k in getattr(cv, "__dict__", {}) for k in methods)
            or not _original_methods(cv, _BUILTIN_CV_METHODS.get(type(cv), {}))):
        raise NativeUnavailableError("overridden CV methods require Python")


def _check_bias_methods(bias):
    from .bias import _BUILTIN_BIAS_METHODS
    original = _BUILTIN_BIAS_METHODS.get(type(bias))
    if ("energy_force" in getattr(bias, "__dict__", {})
            or original is not None and getattr(type(bias), "energy_force", None) is not original):
        raise NativeUnavailableError("overridden bias requires Python")


def cv_descriptor(cv, n):
    from .cv import (
        BondDifference,
        Cartesian,
        COMDistance,
        Distance,
        RateSurface,
        RPMDrateCoordinate,
    )
    from .cv_periodic import FractionalCoordinate, LinearProjection
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
    elif t in (LinearProjection, FractionalCoordinate):
        # Both projection classes are affine functions of the Cartesian
        # centroid for a fixed cell.  Lower their constant coefficient and
        # full Cartesian gradient directly into the native primitive rather
        # than routing through Python CV evaluation.  Restricting this to the
        # exact built-in types keeps subclass/override semantics conservative.
        from .cv import CollectiveVariable, _AffineCoordinate
        overridden = any(name in getattr(cv, "__dict__", {}) for name in (
            "value", "value_and_jacobian", "centroid_value_gradient",
            "centroid_value_gradient_hessian", "centroid_value_gradient_hvp",
            "_affine_gradient"))
        inherited = {
            "value": CollectiveVariable.value,
            "value_and_jacobian": CollectiveVariable.value_and_jacobian,
            "centroid_value_gradient": _AffineCoordinate.centroid_value_gradient,
            "centroid_value_gradient_hessian": _AffineCoordinate.centroid_value_gradient_hessian,
            "centroid_value_gradient_hvp": _AffineCoordinate.centroid_value_gradient_hvp,
            "_affine_gradient": LinearProjection._affine_gradient,
        }
        overridden = overridden or any(getattr(type(cv), name, None) is not original
                                       for name, original in inherited.items())
        if overridden:
            raise NativeUnavailableError("overridden projection methods require Python")
        if getattr(type(cv), "_affine_gradient", None) is not LinearProjection._affine_gradient:
            raise NativeUnavailableError("overridden projection methods require Python")
        primitive = _primitive(n, constant=-float(cv.origin), linear=cv._affine_gradient())
    elif t in (RPMDrateCoordinate, RateSurface):
        c = cv.coordinate if t is RateSurface else cv
        if type(c) is not RPMDrateCoordinate or type(c.com) is not COMDistance:
            raise NativeUnavailableError("custom rate coordinate requires Python")
        _check_cv_methods(c)
        _check_cv_methods(c.com)
        if len(c.com.weights) != n:
            raise ValueError("rate coordinate atom count mismatch")
        result = {"kind": 2 if t is RateSurface else 1,
                  "regular_branch_only": c.regular_branch_only,
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
    from .bias import CompositeBias, FluxMetricBias, UmbrellaBias
    if bias is None:
        return []
    _check_bias_methods(bias)
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


def propagate(dynamics, state, steps, *, bias=None, observe=None, sample_interval=1,
              record_centroids=True, backend=None, fragment_data=()):
    """Advance a transactional block with the explicitly selected ring map.

    Every physical step is checked for finite energy, even when observations are
    decimated. The final step is always sampled. On failure q/p/clock are unchanged
    for the ENTIRE block. No thermostat/RNG or constraint method is substituted.
    The native loop starts each step from the previous step's endpoint PES/bias
    force, one PES call per step, unless the provider declares itself stateful,
    history- or time-dependent or fragment projections move q after that call.
    """
    kwargs = dict(bias=bias, observe=observe, sample_interval=sample_interval,
                  record_centroids=record_centroids, backend=backend, fragment_data=fragment_data)
    composed = getattr(dynamics, "plan", None) is not None and dynamics.plan.active
    if not composed and not getattr(dynamics, "_stochastic_propagation", False):
        return _propagate(dynamics, state, steps, **kwargs)
    operators = dynamics._operator_rollback_token() if composed else None
    rng_state = None if composed else copy.deepcopy(dynamics.rng.bit_generator.state)
    stats = copy.deepcopy(dynamics.propagation_stats)
    try:
        return _propagate(dynamics, state, steps, **kwargs)
    except BaseException:
        if operators is not None:
            dynamics._restore_operator_rollback_token(operators)
        else:
            dynamics.rng.bit_generator.state = rng_state
        dynamics.propagation_stats = stats
        raise


def _propagate(dynamics, state, steps, *, bias=None, observe=None, sample_interval=1,
               record_centroids=True, backend=None, fragment_data=()):
    steps = _integer(steps, "steps")
    stride = _integer(sample_interval, "sample_interval")
    backend = dynamics.loop_backend if backend is None else backend
    if backend not in ("auto", "native", "python"):
        raise ValueError("loop backend must be auto/native/python")
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    if getattr(dynamics, "plan", None) is not None and dynamics.plan.active:
        return _propagate_composed(dynamics, state, steps, bias=bias, observe=observe,
                                   sample_interval=stride, record_centroids=record_centroids,
                                   backend=backend, fragment_data=fragment_data)
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
            binding = _atomic_binding(resolve_native_binding(dynamics.potential, n))
            spec = {"observers": [] if observe is None else _cvs(observe, n),
                    "biases": _biases(bias, n, dynamics.masses),
                    "fragments": [{"atoms": ix.tolist(), "reference": ref.tolist()}
                                  for ix, _m, _total, ref in fragment_data],
                    "time": float(state.time)}
            from .block_propagation import _stateless_provider
            # Endpoint-force reuse is bitwise identical for a deterministic
            # provider; an older extension ignores this optional field.
            spec["reuse_forces"] = bool(not fragment_data and _stateless_provider(dynamics))
            spec.update(fixed_descriptor(dynamics))
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


_COMPOSED_NATIVE_BUFFER_BYTES = 256 * 1024**2


def _propagate_composed(dynamics, state, steps, *, bias, observe,
                        sample_interval, record_centroids, backend, fragment_data):
    """Lower a common operator plan to a block or to per-step callbacks.

    The enclosing propagate() owns whole-block bath/RNG rollback. Nuclear
    state is committed here only after every requested observation succeeds.
    """
    from .dynamics import _state_clock
    start_step, clock = _state_clock(state.step, state.time)
    component = dynamics._composition
    offsets = np.arange(sample_interval, steps + 1, sample_interval, dtype=np.int64)
    if not len(offsets) or offsets[-1] != steps:
        offsets = np.append(offsets, steps)
    rows, natoms = len(offsets), len(dynamics.masses)
    dimension = 0 if observe is None else _integer(observe.dimension, "CV dimension")
    if rows * (1 + 3 * natoms * bool(record_centroids) + dimension) * 8 > 256 * 1024**2:
        raise ValueError("block observation buffers exceed 256 MiB; shorten the block")
    energies = np.empty(rows)
    centroids = np.empty((rows, natoms, 3)) if record_centroids else None
    values = np.empty((rows, dimension)) if observe is not None else None
    if backend == "native" and observe is not None:
        raise NativeUnavailableError("composed native blocks require Python scheduling for CV observations")
    engine = component._prepare_native(state, bias, fragment_data, backend)
    minimum, maximum = np.inf, -np.inf
    if engine is not None and observe is None:
        # Energy extrema inspect every endpoint. Request only the public
        # positions, never unused momenta/radii; older extensions retain the
        # original observer. Bound temporary records independently of the
        # requested result size and the caller's block_steps setting.
        from .native_ensemble import _COLUMNS
        positions_only = getattr(engine._native, "COMMON_POSITION_OBSERVATIONS_API", 0) >= 1
        native_row_bytes = (len(_COLUMNS) + (3 * natoms * bool(record_centroids)
                                          if positions_only else 7 * natoms)) * 8
        native_chunk_limit = _COMPOSED_NATIVE_BUFFER_BYTES // native_row_bytes
        if not native_chunk_limit:
            raise ValueError("one native observation exceeds the block buffer budget")
        initial = engine._initial_energy(state.q, state.p, dynamics.masses, dynamics.beta)
        if not np.isfinite(initial):
            raise FloatingPointError("nonfinite ring-polymer energy")
        completed, row = 0, 0
        used = "native"
        while completed < steps:
            count = min(steps - completed, dynamics.block_steps, native_chunk_limit)
            # Request the common nuclear observations, not NativeEnsemble's
            # thermal estimators. This API owns whole-block rollback, so the
            # native call needs one snapshot, not a snapshot before each step.
            result = engine.run_prefix(count, record_energy=True, check_energy=True,
                                       record_centroids=record_centroids if positions_only else True,
                                       record_moments=not positions_only, _positions_only=positions_only,
                                       _outer_transaction=True, _atomic=True)
            if result["error"] is not None:
                raise RuntimeError(result["error"])
            column = result["columns"].index("ring_polymer_hamiltonian")
            samples = result["samples"][:, column]
            minimum, maximum = min(minimum, float(samples.min())), max(maximum, float(samples.max()))
            last = np.searchsorted(offsets, completed + count, side="right")
            selected = offsets[row:last] - completed - 1
            energies[row:last] = samples[selected]
            if centroids is not None:
                centroids[row:last] = result["centroids"][selected]
            # Match repeated accepted-step clock addition, including a caller's
            # nonzero time origin, without a Python propagation loop.
            clock = float(np.add.accumulate(np.r_[clock, np.full(count, dynamics.dt)])[-1])
            if not np.isfinite(clock):
                raise FloatingPointError("nonfinite integration clock")
            row, completed = last, completed + count
            if result.get("python_step_callbacks", 0):
                used = "python"
        final = engine._physical_state()
        component._native_state = final
        final_q, final_p = final["q"], final["p"]
        final_step, final_time = start_step + steps, clock
        if dynamics._initial_cell is not None:
            dynamics.cell = final["cell"].copy()
    else:
        trial = state.copy()
        initial = dynamics.energy(trial, bias)
        if not np.isfinite(initial):
            raise FloatingPointError("nonfinite ring-polymer energy")
        cache, row = {}, 0
        for i in range(1, steps + 1):
            component.step(trial, bias, fragment_data, cache=cache, backend=backend)
            energy = dynamics.energy(trial, bias)
            if not np.isfinite(energy):
                raise FloatingPointError("nonfinite ring-polymer energy")
            minimum, maximum = min(minimum, energy), max(maximum, energy)
            if i == offsets[row]:
                energies[row] = energy
                if centroids is not None:
                    centroids[row] = trial.q.mean(axis=0)
                if values is not None:
                    value = np.asarray(observe.value(trial.q), dtype=float)
                    if value.shape != (dimension,) or not np.isfinite(value).all():
                        raise ValueError("CV returned invalid observation")
                    values[row] = value
                row += 1
        used = "python"
        final_q, final_p, final_step, final_time = trial.q, trial.p, trial.step, trial.time
    state.q[:], state.p[:] = final_q, final_p
    state.step, state.time = final_step, final_time
    dynamics.last_step_backend = used
    dynamics.propagation_stats[used + "_steps"] += steps
    reason = None if used == "native" else "composed Python force/operator scheduling"
    if reason:
        reasons = dynamics.propagation_stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + steps
    return PropagationResult(start_step + offsets, energies, centroids, values,
                             float(initial), float(minimum), float(maximum), used, reason)


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
        if getattr(dynamics, "plan", None) is not None and dynamics.plan.active:
            component = dynamics._composition
            component.validate_binding()
            if (observe is not None or not dynamics.native
                    or not component._can_native(bias, ()) or not component._has_native_binding()):
                raise NativeUnavailableError("requested operators/observations require Python scheduling")
            return True
        _check_dynamics_methods(dynamics)
        if dynamics.integrator._native is None or not hasattr(dynamics.integrator._native, "run_block"):
            raise NativeUnavailableError(dynamics.integrator.native_fallback_reason or "fused extension unavailable or native=False")
        factory = getattr(dynamics.potential, "native_binding", None)
        if factory is None:
            raise NativeUnavailableError("PES has no explicit stateless native binding")
        _atomic_binding(resolve_native_binding(dynamics.potential, len(dynamics.masses)))
        n = len(dynamics.masses)
        _biases(bias, n, dynamics.masses)
        if observe is not None:
            _cvs(observe, n)
    except NativeUnavailableError:
        if dynamics.loop_backend == "native":
            raise
        return False
    return True


def native_support_reason(dynamics, *, bias=None, observe=None):
    """Return the first reason a dynamics object cannot use fused native steps.

    This is deliberately a read-only companion to :func:`native_supported`
    for validation and diagnostics.  It does not evaluate a PES or mutate a
    dynamics object.  ``None`` means the requested combination is native
    capable; an explicit Python backend reports its selection as a reason.
    """
    if dynamics.loop_backend == "python":
        return "explicit Python loop backend"
    try:
        if getattr(dynamics, "plan", None) is not None and dynamics.plan.active:
            component = dynamics._composition
            component.validate_binding()
            if (observe is not None or not dynamics.native
                    or not component._can_native(bias, ()) or not component._has_native_binding()):
                raise NativeUnavailableError("requested operators/observations require Python scheduling")
            return None
        _check_dynamics_methods(dynamics)
        if dynamics.integrator._native is None or not hasattr(dynamics.integrator._native, "run_block"):
            raise NativeUnavailableError(dynamics.integrator.native_fallback_reason or "fused extension unavailable or native=False")
        factory = getattr(dynamics.potential, "native_binding", None)
        if factory is None:
            raise NativeUnavailableError("PES has no explicit stateless native binding")
        _atomic_binding(resolve_native_binding(dynamics.potential, len(dynamics.masses)))
        n = len(dynamics.masses)
        _biases(bias, n, dynamics.masses)
        if observe is not None:
            _cvs(observe, n)
    except NativeUnavailableError as exc:
        return str(exc)
    return None


def scientific_config(config):
    """Remove execution/postprocessing controls from trajectory identity."""
    from copy import deepcopy
    result = deepcopy(config)
    if "dynamics" in result:
        result["dynamics"].pop("loop_backend", None)
        result["dynamics"].pop("block_steps", None)
        result["dynamics"].pop("electronic_backend", None)
        result["dynamics"].pop("electronic_device", None)
        if not result["dynamics"]:
            result.pop("dynamics")
    # These controls change resource ownership / launch grouping only, never
    # momenta, force definitions, estimator weights or the sampling measure.
    if "parallel" in result:
        result["parallel"].pop("reuse_potentials", None)
        result["parallel"].pop("trajectory_batch_size", None)
        if not result["parallel"]:
            result.pop("parallel")
    def strip_batch_contract(spec):
        if not isinstance(spec, dict):
            return
        kind = spec.get("provider", spec.get("type", "builtin"))
        if kind in ("ase", "mace", "nequip", "fairchem"):
            # The opt-in promises frame independence; it does not change the
            # calculator, Hamiltonian or units. Keep opaque model params intact.
            spec.pop("trajectory_batch_safe", None)
        elif kind == "committee":
            for member in spec.get("members", ()):
                strip_batch_contract(member)
        elif kind == "delta":
            strip_batch_contract(spec.get("reference"))
            strip_batch_contract(spec.get("correction"))
    strip_batch_contract(result.get("potential"))
    for parent in (result, result.get("rate", {})):
        recrossing = parent.get("recrossing")
        if isinstance(recrossing, dict) and "child_batch_pairs" in recrossing:
            recrossing.pop("child_batch_pairs")
            if not recrossing:
                parent.pop("recrossing")
    if (result.get("rate") == {}
            and "child_batch_pairs" in config.get("rate", {}).get("recrossing", {})):
        result.pop("rate")
    rate = result.get("rate", {})
    # Subspace calculations consume saved samples; they never alter the RPMD
    # Hamiltonian or trajectories. Keep their numerical identity separately.
    if "subspace_correction" in rate:
        rate.pop("subspace_correction")
        if not rate:
            result.pop("rate", None)
    request = rate.get("pmf")
    if isinstance(request, dict) and isinstance(request.get("dynamics"), dict):
        for key in ("loop_backend", "block_steps", "electronic_backend", "electronic_device"):
            request["dynamics"].pop(key, None)
        if not request["dynamics"]:
            request.pop("dynamics")
    stage = rate.get("pmf_stage")
    if isinstance(stage, dict) and isinstance(stage.get("config"), dict):
        # A saved PMF stage is a second simulation, with its own physical
        # clock and operators. Apply the same explicit execution exclusions
        # as a top-level run, without recursively dropping scientific keys.
        stage["config"] = scientific_config(stage["config"])
        for key in ("output", "parallel", "resume", "_resume", "_warnings",
                    "_input_path", "_base_directory"):
            stage["config"].pop(key, None)
    return result


# The valid-prefix API is additive; propagate() retains whole-block rollback.
from .block_propagation import BlockResult, PropagationError, propagate_block
