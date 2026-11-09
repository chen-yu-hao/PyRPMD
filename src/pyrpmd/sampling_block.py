"""Transactional native PILE-L/TRPMD sampling blocks.

The native entry point receives NumPy-generated noise and therefore never owns
or advances a random-number generator. This preserves the historical
one-step ``Dynamics.sample_step`` random stream.
"""
from __future__ import annotations

import copy
import numbers

import numpy as np

from .block_propagation import (
    BlockResult,
    NotNative,
    PropagationError,
    bias_descriptors,
    cv_descriptors,
    _stateless_provider,
)
from .propagation import NativeUnavailableError
from .potentials.native import descriptor_binding
from .potentials._builtin_native import resolve_native_binding
# Shared pre-drawn noise budget (R6), defined once by the momentum-refresh planner.
from .momentum_refresh import MAX_PREDRAWN_BYTES


MAX_SAMPLING_STEPS = 4096


def _mask(observe, steps):
    if observe is None:
        return np.ones(steps, dtype=np.uint8)
    values = np.asarray(observe)
    if (values.shape != (steps,) or values.dtype.kind not in "bui"
            or np.any((values != 0) & (values != 1))):
        raise ValueError("observe must be a boolean/binary mask with length steps")
    return np.array(values, dtype=np.uint8, order="C", copy=True)


def _sampling_parameters(dynamics, thermostat, gamma_centroid, friction_scale, method):
    if thermostat not in ("pile-l", "trpmd"):
        raise ValueError("thermostat must be pile-l or trpmd")
    if (isinstance(friction_scale, (bool, np.bool_))
            or not isinstance(friction_scale, numbers.Real)
            or not np.isfinite(friction_scale) or friction_scale < 0):
        raise ValueError("friction_scale must be a finite nonnegative number")
    friction_scale = float(friction_scale)
    if thermostat == "pile-l" and friction_scale != 1.0:
        raise ValueError("friction_scale is a TRPMD option; PILE-L retains gamma_k=2*omega_k")
    if gamma_centroid is None:
        gamma_centroid = 0.0 if thermostat == "trpmd" else 1.0
    if (isinstance(gamma_centroid, (bool, np.bool_))
            or not isinstance(gamma_centroid, numbers.Real)
            or not np.isfinite(gamma_centroid)):
        raise ValueError("gamma_centroid must be a finite real number")
    gamma_centroid = float(gamma_centroid)
    if thermostat == "trpmd" and gamma_centroid != 0.0:
        raise ValueError("TRPMD requires gamma_centroid=0")
    if thermostat == "pile-l" and gamma_centroid <= 0:
        raise ValueError("gamma_centroid must be positive")
    if dynamics.dt <= 0:
        raise ValueError("thermostatted sampling requires positive dt")
    expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(
        dynamics.integrator_name)
    if expected is None:
        raise ValueError("velocity-verlet sampling uses Andersen; BCOCB/BAOAB require their matching integrator")
    selected = expected if method is None else str(method).lower()
    if selected not in ("bcocb", "baoab"):
        raise ValueError("sampling method must be bcocb or baoab")
    if selected != expected and method is not None:
        raise ValueError("sampling method must match the configured Cayley integrator")
    return gamma_centroid, friction_scale, selected


def _eligibility(dynamics, bias, cv):
    """Return native objects without consuming RNG or changing state."""
    from .dynamics import (CayleyIntegrator, Dynamics, _PILE_DYNAMICS_METHODS,
                           _PILE_INTEGRATOR_METHODS)

    if dynamics.loop_backend == "python":
        raise NativeUnavailableError("explicit Python loop backend")
    if getattr(dynamics, "bond_constraints", ()):
        raise NativeUnavailableError("bond constraints require the Python sampling loop")
    if getattr(dynamics, "plan", None) is not None and dynamics.plan.active:
        raise NativeUnavailableError("configured operators require the Python sampling loop")
    if getattr(dynamics.plan, "sampling_integrator", None) is not None:
        raise NativeUnavailableError("composed sampling integrators require the Python path")
    if type(dynamics) is not Dynamics:
        raise NativeUnavailableError("custom Dynamics methods require the Python path")
    kernel = dynamics.integrator
    if type(kernel) is not CayleyIntegrator:
        raise NativeUnavailableError("custom integrators require the Python path")
    if not all(name not in dynamics.__dict__ and getattr(Dynamics, name, None) is original
               for name, original in _PILE_DYNAMICS_METHODS.items()):
        raise NativeUnavailableError("custom sampling methods require the Python path")
    if not all(name not in kernel.__dict__ and getattr(CayleyIntegrator, name, None) is original
               for name, original in _PILE_INTEGRATOR_METHODS.items()):
        raise NativeUnavailableError("custom sampling integrator methods require the Python path")
    if dynamics.nbeads > 8 and dynamics.nbeads & (dynamics.nbeads - 1):
        raise NativeUnavailableError("native sampling supports P<=8 or power-of-two bead counts")
    if dynamics.native is not True:
        raise NativeUnavailableError("compiled native sampling was not requested")
    native = getattr(kernel, "_native", None)
    if native is None or not hasattr(native, "advance_sampling"):
        raise NativeUnavailableError("compiled sampling block extension is unavailable")
    if getattr(native, "SAMPLING_BLOCK_API", 0) < 1:
        raise NativeUnavailableError("installed native extension lacks sampling block support; rebuild it")
    factory = getattr(dynamics.potential, "native_binding", None)
    if not callable(factory):
        raise NativeUnavailableError("PES has no explicit native binding")
    binding = descriptor_binding(resolve_native_binding(dynamics.potential, len(dynamics.masses)))
    # Descriptor lowering rejects Python callbacks before any random draw.
    memo = {}
    bias_specs = bias_descriptors(bias, dynamics, memo)
    cv_specs = [] if cv is None else cv_descriptors(cv, len(dynamics.masses), memo)
    return kernel, bias_specs, cv_specs, binding


def _result(start_step, requested, completed, backend, reason, records,
            mask, nd, ncv, error, step_energies):
    records = np.asarray(records, dtype=float)
    written = records.shape[0]
    if written:
        steps = start_step + np.flatnonzero(mask[:completed]) + 1
        times = records[:, 0].copy()
        energies = records[:, 1].copy()
        centroids = records[:, 2:2 + nd].reshape(written, nd // 3, 3).copy()
        momenta = records[:, 2 + nd:2 + 2 * nd].reshape(written, nd // 3, 3).copy()
        values = records[:, 2 + 2 * nd:].reshape(written, ncv).copy()
    else:
        steps = np.empty(0, dtype=np.int64)
        times = np.empty(0, dtype=float)
        energies = np.empty(0, dtype=float)
        centroids = np.empty((0, nd // 3, 3), dtype=float)
        momenta = np.empty((0, nd // 3, 3), dtype=float)
        values = np.empty((0, ncv), dtype=float)
    return BlockResult(
        requested, completed, backend, reason, steps, times, energies,
        centroids, momenta, values, None, None, error,
        step_energies=None if step_energies is None else np.asarray(step_energies)[:completed],
    )


def _draw_noise(dynamics, shape, scale):
    noise = dynamics.rng.normal(size=shape) * scale
    if len(dynamics._fixed_indices):
        noise[:, dynamics._fixed_indices, :] = 0.0
    return np.ascontiguousarray(noise, dtype=np.float64)


def _redraw(dynamics, shape, scale, count):
    for _ in range(count):
        _draw_noise(dynamics, shape, scale)


def _python_sampling_block(dynamics, state, steps, *, thermostat, method,
                           gamma_centroid, friction_scale, bias, cv, mask,
                           record_energy, check_energy, record_step_energies,
                           reason, raise_on_error):
    start_step = state.step
    nd = 3 * len(dynamics.masses)
    ncv = 0 if cv is None else int(cv.dimension)
    records = []
    step_energies = np.full(steps, np.nan) if record_step_energies else None
    completed = 0
    error = None
    if check_energy:
        try:
            initial = dynamics.energy(state, bias)
            if not np.isfinite(initial):
                raise FloatingPointError("nonfinite total ring energy")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    for index in range(steps if error is None else 0):
        q0, p0 = state.q.copy(), state.p.copy()
        step0, time0 = state.step, state.time
        rng0 = copy.deepcopy(dynamics.rng.bit_generator.state)
        try:
            if thermostat == "trpmd":
                dynamics.trpmd_step(state, bias, friction_scale=friction_scale, method=method)
            else:
                dynamics.sample_step(state, bias, method=method,
                                     gamma_centroid=gamma_centroid,
                                     thermostat="pile-l")
            need_energy = bool(check_energy or record_step_energies
                               or (record_energy and mask[index]))
            energy = np.nan
            if need_energy:
                energy = float(dynamics.energy(state, bias))
                if not np.isfinite(energy):
                    raise FloatingPointError("nonfinite total ring energy")
                if step_energies is not None:
                    step_energies[index] = energy
            if mask[index]:
                center = state.q.mean(axis=0)
                momentum = state.p.mean(axis=0)
                value = np.empty(ncv, dtype=float)
                if cv is not None:
                    value[:] = np.asarray(cv.value(state.q), dtype=float)
                    if value.shape != (ncv,) or not np.isfinite(value).all():
                        raise ValueError("CV returned invalid observation")
                records.append(np.r_[state.time, energy, center.ravel(), momentum.ravel(), value])
            completed += 1
        except Exception as exc:
            state.q[:], state.p[:] = q0, p0
            state.step, state.time = step0, time0
            dynamics.rng.bit_generator.state = rng0
            error = f"{type(exc).__name__}: {exc}"
            break
    dynamics.last_step_backend = "python"
    dynamics.propagation_stats["python_steps"] += completed
    reasons = dynamics.propagation_stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + completed
    width = 2 + 2 * nd + ncv
    array = np.asarray(records, dtype=float).reshape(-1, width)
    result = _result(start_step, steps, completed, "python", reason, array,
                     mask, nd, ncv, error, step_energies)
    if error and raise_on_error:
        raise PropagationError(result)
    return result


def propagate_sampling_block(
    dynamics, state, steps, *, thermostat="pile-l", method=None,
    gamma_centroid=None, friction_scale=1.0, bias=None, cv=None, observe=None,
    record_energy=True, check_energy=False, record_step_energies=False,
    reuse_forces=None, raise_on_error=True,
):
    """Advance PILE-L or TRPMD in a transactional native sampling block."""
    if (isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer))
            or not 0 <= steps <= MAX_SAMPLING_STEPS):
        raise ValueError("steps must be an integer in [0, 4096]")
    steps = int(steps)
    for value, name in ((record_energy, "record_energy"),
                        (check_energy, "check_energy"),
                        (record_step_energies, "record_step_energies"),
                        (raise_on_error, "raise_on_error")):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if reuse_forces is not None and not isinstance(reuse_forces, (bool, np.bool_)):
        raise ValueError("reuse_forces must be boolean or None")
    if reuse_forces is None:
        reuse_forces = _stateless_provider(dynamics)
    gamma_centroid, friction_scale, selected = _sampling_parameters(
        dynamics, thermostat, gamma_centroid, friction_scale, method)
    mask = _mask(observe, steps)
    if cv is not None and (isinstance(cv.dimension, (bool, np.bool_))
                           or not isinstance(cv.dimension, numbers.Integral)
                           or cv.dimension < 1):
        raise ValueError("observation dimension must be a positive integer")
    ncv = 0 if cv is None else int(cv.dimension)
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    start_step = state.step
    nd = 3 * len(dynamics.masses)
    empty = np.empty((0, 2 + 2 * nd + ncv), dtype=float)
    if steps == 0:
        return _result(start_step, 0, 0, "native", None, empty, mask, nd, ncv, None, None)

    reason = None
    if thermostat == "trpmd" and (dynamics.nbeads == 1 or friction_scale == 0.0):
        native_plan = None
        reason = "TRPMD has no stochastic ring modes"
    else:
        try:
            native_plan = _eligibility(dynamics, bias, cv)
        except (NativeUnavailableError, NotNative) as exc:
            native_plan = None
            reason = str(exc)
            if dynamics.loop_backend == "native":
                raise RuntimeError("native sampling block required but unsupported: " + reason) from exc
    if native_plan is None:
        return _python_sampling_block(
            dynamics, state, steps, thermostat=thermostat, method=selected,
            gamma_centroid=gamma_centroid, friction_scale=friction_scale,
            bias=bias, cv=cv, mask=mask, record_energy=bool(record_energy),
            check_energy=bool(check_energy),
            record_step_energies=bool(record_step_energies),
            reason=reason or "Python sampling fallback", raise_on_error=bool(raise_on_error),
        )

    kernel, bias_specs, cv_specs, binding = native_plan
    decay, amplitude, scale, fixed = dynamics._pile_coefficients(
        kernel, gamma_centroid, friction_scale)
    bytes_per_step = int(np.prod(state.p.shape, dtype=np.int64)) * 8
    if bytes_per_step <= 0:
        raise ValueError("invalid sampling state size")
    width = 2 + 2 * nd + ncv
    # Native energy checking remains in the GIL-free loop.  When an energy is
    # returned, retain the accepted q/p snapshots so the wrapper can apply the
    # exact Dynamics.energy NumPy reduction order before exposing the result.
    need_python_energy = bool(record_step_energies or (record_energy and np.any(mask)))
    total_bytes_per_step = (bytes_per_step + width * 8
                            + (8 if record_step_energies else 0)
                            + (2 * bytes_per_step if need_python_energy else 0))
    if total_bytes_per_step > 256 * 1024**2:
        raise ValueError("sampling block buffers exceed 256 MiB; shorten the block or observations")
    chunk_limit = max(1, min(steps, MAX_PREDRAWN_BYTES // bytes_per_step,
                             (256 * 1024**2) // total_bytes_per_step))
    records = []
    all_step_energies = np.full(steps, np.nan) if record_step_energies else None
    completed_total = 0
    error = None
    while completed_total < steps:
        chunk = min(chunk_limit, steps - completed_total)
        chunk_mask = np.ascontiguousarray(mask[completed_total:completed_total + chunk], dtype=np.uint8)
        rows = int(chunk_mask.sum())
        rng0 = copy.deepcopy(dynamics.rng.bit_generator.state)
        q0, p0 = state.q.copy(), state.p.copy()
        time0 = float(state.time)
        noise_block = np.empty((chunk, *state.p.shape), dtype=np.float64)
        for j in range(chunk):
            noise_block[j] = _draw_noise(dynamics, state.p.shape, scale)
        block_records = np.full((rows, width), np.nan, dtype=np.float64)
        block_step_energies = np.full(chunk, np.nan, dtype=np.float64) if record_step_energies else None
        energy_states = (np.full((chunk, 2, *state.q.shape), np.nan, dtype=np.float64)
                         if need_python_energy else None)
        spec = {
            "beta": float(dynamics.beta), "time": float(state.time),
            "cvs": cv_specs, "biases": bias_specs,
            "record_energy": bool(record_energy), "check_energy": bool(check_energy),
            "reuse_forces": bool(reuse_forces) if reuse_forces is not None else False,
            "step_energies": block_step_energies if block_step_energies is not None else None,
            "energy_states": energy_states if energy_states is not None else None,
        }
        try:
            with binding.acquire() as descriptor, kernel._lock:
                outcome = kernel._native.advance_sampling(
                    kernel._context, descriptor, state.q, state.p, noise_block,
                    decay, amplitude, fixed, chunk_mask, block_records, spec,
                )
            completed, written, clock, message = outcome[:4]
            completed, written = int(completed), int(written)
            if completed < 0 or completed > chunk or written < 0 or written > rows:
                raise RuntimeError("native sampling block returned an invalid prefix")
            if energy_states is not None and completed:
                from .dynamics import State
                observed_row = 0
                for index in range(completed):
                    if not (record_step_energies or chunk_mask[index]):
                        continue
                    snapshot = State(
                        energy_states[index, 0], energy_states[index, 1],
                        state.step + index + 1,
                        time0 + (index + 1) * dynamics.dt,
                    )
                    exact_energy = float(dynamics.energy(snapshot, bias))
                    if not np.isfinite(exact_energy):
                        raise FloatingPointError("nonfinite total ring energy")
                    if block_step_energies is not None:
                        block_step_energies[index] = exact_energy
                    if chunk_mask[index] and record_energy:
                        block_records[observed_row, 1] = exact_energy
                        observed_row += 1
            if block_step_energies is not None and all_step_energies is not None:
                all_step_energies[completed_total:completed_total + completed] = block_step_energies[:completed]
            if written:
                records.append(block_records[:written].copy())
            state.step += completed
            state.time = float(clock)
            completed_total += completed
            if message:
                error = str(message)
                if completed < chunk:
                    dynamics.rng.bit_generator.state = rng0
                    _redraw(dynamics, state.p.shape, scale, completed)
                break
            if completed != chunk:
                error = "native sampling block returned a short successful prefix"
                break
        except Exception:
            state.q[:], state.p[:] = q0, p0
            state.step, state.time = start_step + completed_total, time0
            dynamics.rng.bit_generator.state = rng0
            raise
    dynamics.last_step_backend = "native"
    dynamics.propagation_stats["native_steps"] += completed_total
    merged = np.concatenate(records, axis=0) if records else empty
    result = _result(start_step, steps, completed_total, "native", None, merged,
                     mask, nd, ncv, error, all_step_energies)
    if error and raise_on_error:
        raise PropagationError(result)
    return result
