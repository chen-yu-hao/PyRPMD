"""Chunked propagation with native fast paths and faithful Python fallbacks.

Only explicitly recognized, unmodified built-in CV/bias classes are lowered.
Arbitrary Python PES/CV/observables retain their Python execution path. Runtime
PES failures NEVER trigger a retry via another backend. Each completed step,
including its requested observations, is transactional.
"""
from dataclasses import dataclass
import numbers
import numpy as np

from . import cv as cv_module
from .bias import UmbrellaBias, FluxMetricBias, CompositeBias
from .potentials.native import descriptor_binding


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
    outcome: dict | None = None
    previous_q: np.ndarray | None = None
    previous_p: np.ndarray | None = None
    previous_step: int | None = None
    previous_time: float | None = None
    diagnostics: dict[str, np.ndarray] | None = None

    @property
    def success(self):
        return self.error is None and (
            self.completed == self.requested
            or self.outcome is not None and (
                self.outcome["resolved"] is not None or self.outcome.get("pending_check", False)
            )
        )


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
    from .propagation import _check_cv_methods
    _check_cv_methods(cv)
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
        if type(cv.com) is not cv_module.COMDistance:
            raise NotNative("custom rate-coordinate COM requires the Python path")
        _check_cv_methods(cv.com)
        if len(cv.com.weights) != natoms:
            raise ValueError("rate-coordinate COM size mismatch")
        return dict(tag, kind="rate", weights=cv.com.weights.tolist(), rinf=cv.rinf,
                    paths=[[{"forming": _pair(f, natoms), "breaking": _pair(b, natoms), "reference": r}
                            for f, b, r in zip(forming, breaking, cv.reference_difference)]
                           for forming, breaking in cv.paths])
    if type(cv) is cv_module.RateSurface:
        if type(cv.coordinate) is not cv_module.RPMDrateCoordinate:
            raise NotNative("custom rate coordinate requires the Python path")
        return dict(tag, kind="surface", coordinate=cv_descriptor(cv.coordinate, natoms), xi=cv.xi)
    raise NotNative(f"CV {type(cv).__name__} requires the Python path")


def cv_descriptors(cv, natoms):
    if cv is None:
        return []
    if type(cv) is cv_module.CVStack:
        from .propagation import _check_cv_methods
        _check_cv_methods(cv)
        return [cv_descriptor(c, natoms) for c in cv.cvs]
    return [cv_descriptor(cv, natoms)]


def bias_descriptors(bias, dynamics):
    if bias is None:
        return []
    from .propagation import _check_bias_methods
    _check_bias_methods(bias)
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
    from .propagation import _check_dynamics_methods
    _check_dynamics_methods(dynamics)
    if not dynamics.native or not hasattr(dynamics.integrator._native, "advance"):
        raise NotNative(dynamics.integrator.native_fallback_reason or "compiled fused kernel is not available")
    factory = getattr(dynamics.potential, "native_binding", None)
    binding = factory() if callable(factory) else None
    if binding is None:
        raise NotNative(f"PES {type(dynamics.potential).__name__} has no native binding")
    binding = descriptor_binding(binding)
    if binding.natoms != len(dynamics.masses):
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


def native_outcome_supported(dynamics):
    """Whether plain blocks can evaluate the versioned outcome descriptors.

    The module marker also rejects an older installed extension, whose advance
    entry point would otherwise silently ignore new descriptor fields.
    """
    if getattr(dynamics, "loop_backend", "auto") == "python":
        return False
    native = getattr(dynamics.integrator, "_native", None)
    if getattr(native, "DRPMD_OUTCOMES_API", 0) < 1:
        return False
    from .propagation import NativeUnavailableError
    try:
        _native_plan(dynamics, None, None, None, (), 1e-10, True, 0.0)
    except (NotNative, NativeUnavailableError):
        return False
    return True


def _outcome_options(outcome, state, steps):
    if not isinstance(outcome, dict) or not isinstance(outcome.get("descriptor"), dict):
        raise ValueError("outcome requires a classifier descriptor")
    descriptor = outcome["descriptor"]
    labels = descriptor.get("channel_labels")
    if (not isinstance(labels, (list, tuple))
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("outcome channel_labels must be unique nonempty strings")

    def integer(name, default, minimum):
        value = outcome.get(name, default)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not minimum <= value <= np.iinfo(np.int64).max):
            raise ValueError(f"outcome {name} must be an integer >= {minimum}")
        return int(value)

    start = integer("start_step", state.step, 0)
    if start > np.iinfo(np.int64).max - steps:
        raise ValueError("outcome step counter overflow")
    check = integer("check_interval", 10, 1)
    persistence = integer("persistence_steps", 3 * check, 0)
    candidate, since = outcome.get("candidate"), outcome.get("candidate_since")
    code = -1
    if candidate is not None:
        if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
            raise ValueError("outcome candidate must be a normalized (status, channel) pair")
        if candidate[0] == "nonreactive" and candidate[1] is None:
            code = 0
        elif candidate[0] == "product" and candidate[1] in labels:
            code = labels.index(candidate[1]) + 1
        else:
            raise ValueError("outcome candidate is not a declared channel")
        if (isinstance(since, (bool, np.bool_)) or not isinstance(since, numbers.Integral)
                or not 0 <= since <= start):
            raise ValueError("outcome candidate_since must lie in the completed prefix")
        since = int(since)
    elif since is not None:
        raise ValueError("outcome without a candidate must have candidate_since=None")
    return {"descriptor": descriptor, "start_step": start, "check_interval": check,
            "persistence_steps": persistence, "candidate_code": code,
            "candidate_since": -1 if since is None else since}


def _outcome_result(spec, metadata):
    def label(code):
        if code < 0:
            return None
        if code == 0:
            return ("nonreactive", None)
        return ("product", spec["descriptor"]["channel_labels"][code - 1])
    since = metadata.get("candidate_since", spec["candidate_since"])
    return {
        "candidate": label(metadata.get("candidate_code", spec["candidate_code"])),
        "candidate_since": None if since < 0 else since,
        "resolved": label(metadata.get("resolved_code", -1)),
        "checks": int(metadata.get("checks", 0)),
        "pending_check": bool(metadata.get("pending_check", False)),
    }


ENERGY_PARTS = ("potential", "centroid_kinetic", "internal_kinetic", "spring")


def _diagnostic_options(diagnostics, dynamics):
    if diagnostics is None:
        return {}
    if not isinstance(diagnostics, dict) or set(diagnostics) - {"energy_parts", "ring_radii"}:
        raise ValueError("diagnostics must contain only energy_parts and ring_radii flags")
    if any(not isinstance(value, (bool, np.bool_)) for value in diagnostics.values()):
        raise ValueError("diagnostic flags must be boolean")
    selected = {name: True for name, value in diagnostics.items() if value}
    if selected.get("energy_parts"):
        from .dynamics import _BLOCK_DYNAMICS_METHODS
        # Reuse the ordinary energy evaluation without bypassing user-defined
        # energy semantics or silently assigning an energy offset to the PES.
        if getattr(dynamics.energy, "__func__", None) is not _BLOCK_DYNAMICS_METHODS["energy"]:
            raise ValueError("energy_parts diagnostics require the original Dynamics.energy method")
    return selected


def _nuclear_energy_terms(dynamics, state):
    q, p = state.q, state.p
    masses = dynamics.masses[None, :, None]
    beads = dynamics.nbeads
    kinetic = float(np.sum(p * p / (2 * masses)))
    dq = q - np.roll(q, -1, axis=0)
    spring = float(0.5 * (beads / dynamics.beta) ** 2 * np.sum(masses * dq * dq))
    center = p.mean(axis=0, keepdims=True)
    centroid = float(np.sum(center * center / (2 * masses)))
    internal = float(np.sum((p - center) ** 2 / (2 * masses))) / beads
    return kinetic, spring, centroid, internal


def _ring_radii(q, center=None):
    centered = q - (q.mean(axis=0) if center is None else center)
    radii = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=0))
    if not np.isfinite(radii).all():
        raise FloatingPointError("nonfinite ring radius diagnostics")
    return radii


def _python_energy_parts(dynamics, state, bias):
    """The same PES call and total-energy convention as Dynamics.energy.

    Internal here means noncentroid *bead* momenta, not molecular vibrations.
    Bias energy belongs to the recorded total but not to these physical parts.
    """
    from .potentials.base import validate_result
    q = state.q
    beads = dynamics.nbeads
    potential = float(np.sum(validate_result(dynamics.potential.evaluate(q), q.shape).energy))
    kinetic, spring, centroid, internal = _nuclear_energy_terms(dynamics, state)
    parts = np.array([potential / beads, centroid, internal, spring / beads])
    energy = potential + kinetic + spring
    if bias is not None:
        energy += float(bias.energy_force(q)[0])
    if not np.isfinite(parts).all() or not np.isfinite(energy):
        raise FloatingPointError("nonfinite nuclear energy diagnostics")
    return parts, energy


def state_diagnostics(dynamics, state, diagnostics, *, energy_per_bead=None):
    """Observe a single existing state using the block diagnostic conventions.

    Each selected value has no leading frame axis: energy_parts is shape (4,)
    and ring_radii is shape (natoms,). ``energy_per_bead``, when supplied, must
    be the *unbiased* H_P/P of this exact state. The potential component is then
    recovered by subtracting kinetic and spring energies, subject to floating
    point cancellation, without evaluating the PES. Otherwise energy_parts
    evaluates the PES once; ring_radii alone never evaluates it. This helper
    does not propagate, resample momenta, or mutate the state or RNG.
    """
    spec = _diagnostic_options(diagnostics, dynamics)
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    if energy_per_bead is not None and (
        isinstance(energy_per_bead, (bool, np.bool_))
        or not isinstance(energy_per_bead, numbers.Real)
        or not np.isfinite(energy_per_bead)
    ):
        raise ValueError("energy_per_bead must be a finite unbiased H_P/P")
    result = {}
    if spec.get("energy_parts"):
        if energy_per_bead is None:
            parts, _ = _python_energy_parts(dynamics, state, None)
        else:
            kinetic, spring, centroid, internal = _nuclear_energy_terms(dynamics, state)
            beads = dynamics.nbeads
            parts = np.array([float(energy_per_bead) - kinetic / beads - spring / beads,
                              centroid, internal, spring / beads])
            if not np.isfinite(parts).all():
                raise FloatingPointError("nonfinite nuclear energy diagnostics")
        result["energy_parts"] = parts
    if spec.get("ring_radii"):
        result["ring_radii"] = _ring_radii(state.q)
    return result


def propagate_block(dynamics, state, steps, *, bias=None, cv=None, constraint=None,
                    constraint_tolerance=1e-10, fragment_data=(), observe=None,
                    record_energy=True, record_states=False, backend=None, raise_on_error=True,
                    outcome=None, record_previous=False, reuse_forces=False,
                    check_energy=False, diagnostics=None):
    """Advance at most 4096 selected-integrator steps, observing in step order.

    ``observe`` is a length-steps boolean mask (default: all steps). A failure
    leaves q/p/clock at the last successfully completed step. Its requested
    observations are part of the same step transaction.
    ``PropagationError.result`` retains the valid prefix. No force cache survives
    a block, so changing bias, constraints, q/p or PES between blocks is safe.
    ``outcome`` opts into a supported native classifier and its persistence
    state machine; it can stop successfully before the requested block ends.
    A comparison within floating-point roundoff of a classification boundary
    yields ``outcome['pending_check']`` without updating persistence. The caller
    must run the original Python classifier at that step before continuing.
    Unsupported backends reject this option before mutating the state.
    ``record_previous`` retains only the state before the final accepted step
    of a successful block, allowing a Python classifier to reject that step.
    ``reuse_forces`` opts into block-local force reuse for ordinary native
    dynamics without bias, constraint or fragment projection. Other cases keep
    the reference PES evaluation schedule. No cache survives a block.
    ``check_energy`` validates the initial and every candidate step's total
    ring energy, including unobserved steps. Observations reuse that energy.
    Its default is False, preserving the original PES evaluation schedule.
    Older native extensions without this capability fall back in auto mode
    and are rejected in strict native mode before propagation.
    ``diagnostics`` optionally selects boolean ``energy_parts`` and
    ``ring_radii`` flags. Results use the same observed frames as ``steps``.
    Energy columns are ``ENERGY_PARTS``: physical potential/P, centroid kinetic,
    noncentroid bead kinetic/P, and ring spring/P, in hartree. Their sum is the
    unbiased ring Hamiltonian/P, not a quantum thermodynamic energy estimator.
    They require the original Dynamics.energy method. Ring radii are per-atom
    sqrt(mean_beads(|q-q_centroid|**2)), in bohr. These are bead diagnostics,
    not molecular normal-mode populations. Native observations use existing
    endpoint PES data, with no Python callback or extra unprojected PES call.
    Blocks must end at thermostat/Python-classifier/checkpoint boundaries.
    Sampling RNG remains in Python at those boundaries.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or not 0 <= steps <= 4096:
        raise ValueError("steps must be an integer in [0, 4096]")
    steps = int(steps)
    backend = backend or getattr(dynamics, "loop_backend", "auto")
    if backend not in ("auto", "native", "python"):
        raise ValueError("loop backend must be auto, native or python")
    if not isinstance(record_previous, (bool, np.bool_)) or not isinstance(reuse_forces, (bool, np.bool_)):
        raise ValueError("record_previous and reuse_forces must be boolean")
    if not isinstance(check_energy, (bool, np.bool_)):
        raise ValueError("check_energy must be boolean")
    diagnostic_spec = _diagnostic_options(diagnostics, dynamics)
    outcome_spec = None if outcome is None else _outcome_options(outcome, state, steps)
    if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0:
        raise ValueError("constraint tolerance must be positive and finite")
    if constraint is not None and (bias is not None or fragment_data):
        raise ValueError("constraint propagation cannot be combined with bias/fragment projections")
    dynamics._check_parameters();dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    if observe is None:
        mask = np.ones(steps, dtype=np.uint8)
    else:
        values = np.asarray(observe)
        if values.shape != (steps,) or values.dtype.kind not in "bui" or np.any((values != 0) & (values != 1)):
            raise ValueError("observe must be a boolean/binary mask with length steps")
        mask = np.array(values, dtype=np.uint8, order="C", copy=True)
    count = int(mask.sum());natoms = len(dynamics.masses);nd = natoms * 3
    dimension = 0
    if cv is not None:
        if (isinstance(cv.dimension, (bool, np.bool_))
                or not isinstance(cv.dimension, numbers.Integral) or cv.dimension < 1):
            raise ValueError("observation dimension must be a positive integer")
        dimension = int(cv.dimension)
    width = 2 + 2 * nd + dimension
    diagnostic_width = 4 * diagnostic_spec.get("energy_parts", False) + natoms * diagnostic_spec.get("ring_radii", False)
    memory = (count * (width + diagnostic_width + (2 * state.q.size if record_states else 0))
              + (2 * state.q.size if record_previous else 0)) * 8
    if memory > 256 * 1024 * 1024:
        raise ValueError("block observation buffers exceed 256 MiB; shorten the block")
    records = np.full((count, width), np.nan)
    snapshots = np.empty((count, 2, *state.q.shape)) if record_states else None
    previous_state = np.empty((2, *state.q.shape)) if record_previous else None
    diagnostic_arrays = {
        name: np.empty((count, 4 if name == "energy_parts" else natoms))
        for name in diagnostic_spec
    }
    start_step, start_time = state.step, state.time
    binding = options = None;reason = None
    if backend != "python":
        from .propagation import NativeUnavailableError
        try:
            binding, options = _native_plan(dynamics, bias, cv, constraint, fragment_data,
                                            constraint_tolerance, record_energy, state.time)
            if (outcome_spec is not None or record_previous) and getattr(
                    dynamics.integrator._native, "DRPMD_OUTCOMES_API", 0) < 1:
                binding = None
                raise NotNative("installed native extension lacks outcome/rollback support; rebuild it")
            if check_energy and getattr(dynamics.integrator._native, "BLOCK_ENERGY_CHECK_API", 0) < 1:
                binding = None
                raise NotNative("installed native extension lacks per-step energy checking; rebuild it")
            if diagnostic_spec and getattr(dynamics.integrator._native, "BLOCK_DIAGNOSTICS_API", 0) < 1:
                binding = None
                raise NotNative("installed native extension lacks nuclear diagnostics; rebuild it")
        except (NotNative, NativeUnavailableError) as exc:
            reason = str(exc)
            if backend == "native":
                raise RuntimeError("native loop required but unsupported: " + reason) from exc
    selected = "native" if binding is not None else "python"
    if outcome_spec is not None and selected != "native":
        raise RuntimeError("native outcomes require a supported native loop: " + (reason or "Python backend selected"))
    completed = written = 0;error = None
    metadata = {}
    previous_time = None
    if steps and selected == "native":
        # The lease pins the owner and prevents concurrent context destruction.
        # The integrator lock protects all use of this Cayley context.
        options.update(outcome=outcome_spec, previous=previous_state, reuse_forces=bool(reuse_forces))
        if check_energy:
            options["check_energy"] = True
        if diagnostic_arrays:
            options["diagnostics"] = diagnostic_arrays
        with binding.acquire() as descriptor, dynamics.integrator._lock:
            native_result = dynamics.integrator._native.advance(
                dynamics.integrator._context, descriptor, state.q, state.p, options, mask, records, snapshots)
        completed, written, clock, message = native_result[:4]
        if len(native_result) == 5:
            metadata = native_result[4]
        state.step += completed;state.time = clock;error = message or None
        previous_time = metadata.get("previous_time")
    elif steps:
        for i in range(steps):
            previous = state.copy()
            try:
                if check_energy and i == 0:
                    initial_energy = dynamics.energy(state, bias)
                    if not np.isfinite(initial_energy):
                        raise FloatingPointError("nonfinite total ring energy")
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
                dynamics._check_fixed(state.q, state.p)
                if diagnostic_spec.get("energy_parts") and mask[i]:
                    parts, energy = _python_energy_parts(dynamics, state, bias)
                    diagnostic_arrays["energy_parts"][written] = parts
                elif check_energy or (record_energy and mask[i]):
                    energy = dynamics.energy(state, bias)
                    if not np.isfinite(energy):
                        raise FloatingPointError("nonfinite total ring energy")
                if mask[i]:
                    row = records[written];row[0] = state.time
                    if record_energy:
                        row[1] = energy
                    row[2:2+nd] = state.q.mean(axis=0).ravel()
                    row[2+nd:2+2*nd] = state.p.mean(axis=0).ravel()
                    if diagnostic_spec.get("ring_radii"):
                        diagnostic_arrays["ring_radii"][written] = _ring_radii(
                            state.q, row[2:2+nd].reshape(natoms, 3))
                    if cv is not None:
                        value = np.asarray(cv.value(state.q), dtype=float)
                        if value.shape != (dimension,) or not np.isfinite(value).all():
                            raise ValueError("CV returned invalid observation")
                        row[2+2*nd:] = value
                    if snapshots is not None:
                        snapshots[written, 0] = state.q;snapshots[written, 1] = state.p
            except Exception as exc:
                state.q[:] = previous.q;state.p[:] = previous.p
                state.step, state.time = previous.step, previous.time
                error = f"{type(exc).__name__}: {exc}"
                break
            completed += 1;written += int(mask[i])
        if previous_state is not None and completed and error is None:
            previous_state[0] = previous.q;previous_state[1] = previous.p
            previous_time = previous.time
    records = records[:written];snapshots = None if snapshots is None else snapshots[:written]
    indices = np.flatnonzero(mask[:completed]) + start_step + 1
    result = BlockResult(steps, completed, selected, reason, indices, records[:, 0], records[:, 1],
                         records[:, 2:2+nd].reshape(-1, natoms, 3),
                         records[:, 2+nd:2+2*nd].reshape(-1, natoms, 3), records[:, 2+2*nd:],
                         None if snapshots is None else snapshots[:, 0],
                         None if snapshots is None else snapshots[:, 1], error,
                          outcome=None if outcome_spec is None else _outcome_result(outcome_spec, metadata),
                          diagnostics={name: values[:written] for name, values in diagnostic_arrays.items()}
                          if diagnostic_arrays else None)
    if previous_time is not None:
        result.previous_q, result.previous_p = previous_state
        result.previous_step = start_step + completed - 1
        result.previous_time = previous_time
    stats = getattr(dynamics, "loop_stats", None)
    if stats is not None:
        stats[selected + "_steps"] += completed
        if reason:
            stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + completed
    if error and raise_on_error:
        raise PropagationError(result)
    return result


def execution_options(config):
    from .propagation import loop_options
    return loop_options(config)


def scientific_config(config):
    """Execution-only choices do not invalidate a physical checkpoint."""
    from .propagation import scientific_config as shared_scientific_config
    return shared_scientific_config(config)
