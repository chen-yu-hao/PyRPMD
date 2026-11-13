"""Native centroid-friction blocks with the existing NumPy random stream.

The complete O/2--B/2--ring--B/2--O/2 loop runs in C++. Python supplies noise,
publishes requested observations and restores RNG state to the accepted prefix.
Only immutable constant diagonal friction and an explicitly native PES qualify.
"""
from __future__ import annotations

import copy
import numbers

import numpy as np

from .block_propagation import PropagationError, _stateless_provider
from .momentum_refresh import MAX_PREDRAWN_BYTES
from .potentials._builtin_native import resolve_native_binding
from .potentials.native import descriptor_binding
from .propagation import NativeUnavailableError, _original_methods, fixed_descriptor
from .sampling_block import _mask, _result


def native_friction_plan(dynamics):
    """Check native scope without advancing state, noise or the provider."""
    from .dynamics import CayleyIntegrator, _BLOCK_INTEGRATOR_METHODS
    from .electronic_friction import (
        ConstantElectronicFriction, ElectronicFrictionDynamics,
        _CONSTANT_FRICTION_EVALUATE, _EF_FUSED_METHODS,
    )

    if dynamics.loop_backend == "python":
        raise NativeUnavailableError("explicit Python loop backend")
    if type(dynamics) is not ElectronicFrictionDynamics or not _original_methods(dynamics, _EF_FUSED_METHODS):
        raise NativeUnavailableError("custom electronic-friction Dynamics methods require Python")
    if dynamics.bond_constraints or dynamics.plan.active:
        raise NativeUnavailableError("additional constraints or composed operators require Python friction")
    friction = dynamics.friction
    if (type(friction) is not ConstantElectronicFriction or "evaluate" in friction.__dict__
            or ConstantElectronicFriction.evaluate is not _CONSTANT_FRICTION_EVALUATE
            or dynamics._constant_eigen is None):
        raise NativeUnavailableError("position-dependent or overridden friction requires the Python OU path")
    if np.any(friction._tensor != np.diag(np.diag(friction._tensor))):
        raise NativeUnavailableError("off-diagonal electronic friction requires the Python OU eigensystem")
    kernel = dynamics.integrator
    if type(kernel) is not CayleyIntegrator or not _original_methods(kernel, _BLOCK_INTEGRATOR_METHODS):
        raise NativeUnavailableError("custom integrator methods require Python friction")
    if not dynamics.native or getattr(kernel._native, "CENTROID_FRICTION_API", 0) < 1:
        raise NativeUnavailableError("compiled centroid-friction block unavailable; rebuild the native extension")
    if not callable(getattr(dynamics.potential, "native_binding", None)):
        raise NativeUnavailableError("electronic-friction PES has no native binding")
    binding = descriptor_binding(resolve_native_binding(dynamics.potential, len(dynamics.masses)))
    rates, vectors = dynamics._constant_eigen
    order = np.argmax(vectors, axis=0)
    if not np.array_equal(vectors, np.eye(len(rates))[:, order]):
        raise NativeUnavailableError("non-diagonal friction eigenvectors require Python")
    coordinates = np.flatnonzero(dynamics._mobile)[order]
    decay = np.ones(3 * len(dynamics.masses))
    amplitude = np.zeros_like(decay)
    decay[coordinates] = np.exp(-rates * .5 * dynamics.dt)
    amplitude[coordinates] = np.sqrt(-np.expm1(-rates * dynamics.dt) / dynamics.beta)
    return binding, decay, amplitude, coordinates, bool(np.any(rates))


def propagate_friction_block(dynamics, state, steps, *, observe=None, record_energy=True,
                             check_energy=False, record_previous=False, raise_on_error=True):
    """Advance a native unbiased block, retaining only successfully accepted steps.

    Unsupported compositions raise before state or RNG mutation. The ordinary
    ``ElectronicFrictionDynamics.propagate`` remains the Python reference path.
    ``previous_rng_state`` accompanies the pre-final state for workflow rollback.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, numbers.Integral) or not 0 <= steps <= 4096:
        raise ValueError("steps must be an integer in [0, 4096]")
    for flag in (record_energy, check_energy, record_previous, raise_on_error):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("friction block recording options must be boolean")
    binding, decay, amplitude, coordinates, stochastic = native_friction_plan(dynamics)
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    mask = _mask(observe, steps)
    nd = 3 * len(dynamics.masses)
    if steps * 2 * nd * 8 > MAX_PREDRAWN_BYTES:
        raise ValueError("friction block noise exceeds the pre-drawn budget; shorten the block")
    start = state.copy()
    rng0 = copy.deepcopy(dynamics.rng.bit_generator.state)
    previous_rng = rng0
    records = np.empty((int(mask.sum()), 2 + 2 * nd))
    previous = np.empty((2, *state.q.shape)) if record_previous else None
    noise = np.zeros((steps, 2, nd))
    try:
        if stochastic and steps:
            # Sorted eigensystem coordinates retain the reference draw order,
            # including zero-rate modes when any other mode is damped.
            noise[:-1, :, coordinates] = dynamics.rng.normal(size=(steps - 1, 2, len(coordinates)))
            previous_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            noise[-1:, :, coordinates] = dynamics.rng.normal(size=(1, 2, len(coordinates)))
        options = {"beta": dynamics.beta, "time": state.time, "cvs": [], "biases": [],
                   "constraint": None, "constraint_tolerance": 1e-10, "fragments": [],
                   "record_energy": bool(record_energy), "check_energy": bool(check_energy),
                   "reuse_forces": _stateless_provider(dynamics), "previous": previous,
                   "centroid_friction": {"decay": decay.tolist(), "amplitude": amplitude.tolist(),
                                         "noise": noise}, **fixed_descriptor(dynamics)}
        kernel = dynamics.integrator
        with binding.acquire() as descriptor, kernel._lock:
            returned = kernel._native.advance(kernel._context, descriptor, state.q, state.p,
                                               options, mask, records, None)
        completed, written, clock, message = returned[:4]
        if not 0 <= completed <= steps or written != int(mask[:completed].sum()):
            raise RuntimeError("native friction returned an invalid accepted prefix")
        if completed != steps:
            dynamics.rng.bit_generator.state = rng0
            if stochastic:
                dynamics.rng.normal(size=(completed, 2, len(coordinates)))
        state.step = start.step + completed
        state.time = clock
    except BaseException:
        state.q[:], state.p[:] = start.q, start.p
        state.step, state.time = start.step, start.time
        dynamics.rng.bit_generator.state = rng0
        raise
    dynamics.last_step_backend = "native"
    dynamics.propagation_stats["native_steps"] += completed
    result = _result(start.step, steps, completed, "native", None, records[:written], mask, nd, 0,
                     message or None, None)
    if previous is not None and completed and not message:
        result.previous_q, result.previous_p = previous[0].copy(), previous[1].copy()
        result.previous_step = state.step - 1
        result.previous_time = returned[4]["previous_time"]
        result.previous_rng_state = previous_rng
    if message and raise_on_error:
        raise PropagationError(result)
    return result
