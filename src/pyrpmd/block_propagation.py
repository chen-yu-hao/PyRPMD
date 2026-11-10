"""Chunked propagation with native fast paths and faithful Python fallbacks.

Only explicitly recognized, unmodified built-in CV/bias classes are lowered.
Arbitrary Python PES/CV/observables retain their Python execution path. Runtime
PES failures NEVER trigger a retry via another backend. Each completed step,
including its requested observations, is transactional.
"""
from dataclasses import dataclass
import copy
import json
import numbers
import numpy as np

from . import cv as cv_module
from .bias import (UmbrellaBias, FluxMetricBias, CompositeBias, WallBias,
                   MetadynamicsBias, GridMetadynamicsBias)
from .potentials.native import descriptor_binding
from .potentials._builtin_native import resolve_native_binding
from .validation import checked_integer


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
    # Total ring energy of every completed step (not only observed ones),
    # when requested with record_step_energies=True.
    step_energies: np.ndarray | None = None
    # Number of pre-drawn momentum refreshes that were committed.
    refreshes_completed: int = 0

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
    weights = [0.0] * natoms
    weights[a], weights[b] = 1.0, -1.0
    return weights


def cv_descriptor(cv, natoms, memo=None):
    """A value/derivative-preserving lowering; subclasses deliberately fall back.

    ``memo`` shares one lowering of the same CV object among the consumers of
    a single native plan (umbrella, flux metric and observation). The native
    loop then recognizes them as one coordinate by identity.
    """
    if memo is not None:
        key = (id(cv), natoms)
        if key in memo:
            return memo[key][1]
        descriptor = cv_descriptor(cv, natoms)
        memo[key] = (cv, descriptor)  # retain cv: its id stays unique
        return descriptor
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
                    regular_branch_only=cv.regular_branch_only,
                    paths=[[{"forming": _pair(f, natoms), "breaking": _pair(b, natoms), "reference": r}
                            for f, b, r in zip(forming, breaking, cv.reference_difference)]
                           for forming, breaking in cv.paths])
    if type(cv) is cv_module.RateSurface:
        if type(cv.coordinate) is not cv_module.RPMDrateCoordinate:
            raise NotNative("custom rate coordinate requires the Python path")
        return dict(tag, kind="surface", coordinate=cv_descriptor(cv.coordinate, natoms), xi=cv.xi)
    raise NotNative(f"CV {type(cv).__name__} requires the Python path")


def cv_descriptors(cv, natoms, memo=None):
    if cv is None:
        return []
    if type(cv) is cv_module.CVStack:
        from .propagation import _check_cv_methods
        _check_cv_methods(cv)
        return [cv_descriptor(c, natoms, memo) for c in cv.cvs]
    return [cv_descriptor(cv, natoms, memo)]


def bias_descriptors(bias, dynamics, memo=None):
    if bias is None:
        return []
    from .propagation import _check_bias_methods
    _check_bias_methods(bias)
    if type(bias) is CompositeBias:
        return [d for part in bias.biases for d in bias_descriptors(part, dynamics, memo)]
    if type(bias) is UmbrellaBias:
        return [{"kind": "umbrella", "cvs": cv_descriptors(bias.cv, len(dynamics.masses), memo),
                 "center": bias.center.tolist(), "stiffness": bias.stiffness.ravel().tolist(),
                 "periods": [0.0 if p is None else float(p) for p in bias.cv.periods]}]
    if type(bias) is FluxMetricBias:
        if bias.beta != dynamics.beta or not np.array_equal(bias.masses, dynamics.masses):
            raise NotNative("flux metric masses/beta differ from the integrator")
        return [{"kind": "flux", "cvs": [cv_descriptor(bias.cv, len(dynamics.masses), memo)]}]
    history_api = getattr(getattr(dynamics.integrator, "_native", None),
                          "NATIVE_HISTORY_BIAS_API", 0)
    if type(bias) is WallBias:
        if history_api < 1:
            raise NotNative("installed native extension lacks history-bias support; rebuild it")
        descriptor = dict(bias.native_descriptor())
        descriptor["cvs"] = cv_descriptors(bias.cv, len(dynamics.masses), memo)
        return [descriptor]
    if type(bias) is GridMetadynamicsBias:
        if history_api < 1:
            raise NotNative("installed native extension lacks history-bias support; rebuild it")
        descriptor = dict(bias.native_descriptor())
        descriptor["cvs"] = cv_descriptors(bias.cv, len(dynamics.masses), memo)
        return [descriptor]
    if type(bias) is MetadynamicsBias:
        if history_api < 1:
            raise NotNative("installed native extension lacks history-bias support; rebuild it")
        descriptor = dict(bias.native_descriptor())
        descriptor["cvs"] = cv_descriptors(bias.cv, len(dynamics.masses), memo)
        return [descriptor]
    raise NotNative(f"bias {type(bias).__name__} requires the Python path")


def _check_bias_snapshot(bias, descriptors):
    """Reject a history bias that changed while its native block was running."""
    if bias is None:
        if descriptors:
            raise RuntimeError("native bias snapshot has no corresponding bias")
        return
    index = 0

    def visit(current):
        nonlocal index
        if type(current) is CompositeBias:
            for part in current.biases:
                visit(part)
            return
        if index >= len(descriptors):
            raise RuntimeError("native bias snapshot is incomplete")
        descriptor = descriptors[index]
        index += 1
        if type(current) in (MetadynamicsBias, GridMetadynamicsBias):
            if descriptor.get("hill_count") != current.hill_count:
                raise RuntimeError("native bias snapshot expired during block")
            if type(current) is GridMetadynamicsBias:
                current_metadata = json.dumps(current.grid_state()["metadata"],
                                              sort_keys=True, separators=(",", ":"),
                                              allow_nan=False)
                if descriptor.get("grid_metadata") != current_metadata:
                    raise RuntimeError("native grid bias snapshot expired during block")

    visit(bias)
    if index != len(descriptors):
        raise RuntimeError("native bias snapshot has extra descriptors")


def _native_plan(dynamics, bias, cv, constraint, fragment_data, tolerance, record_energy, clock):
    from .propagation import _check_dynamics_methods, fixed_descriptor
    _check_dynamics_methods(dynamics)
    if not dynamics.native or not hasattr(dynamics.integrator._native, "advance"):
        raise NotNative(dynamics.integrator.native_fallback_reason or "compiled fused kernel is not available")
    factory = getattr(dynamics.potential, "native_binding", None)
    binding = resolve_native_binding(dynamics.potential, len(dynamics.masses)) if callable(factory) else None
    if binding is None:
        raise NotNative(f"PES {type(dynamics.potential).__name__} has no native binding")
    binding = descriptor_binding(binding)
    if binding.natoms != len(dynamics.masses):
        raise ValueError("invalid native binding or atom count")
    native_constraint = None if constraint is None else cv_descriptor(constraint, len(dynamics.masses))
    if constraint is not None and dynamics.fixed_atoms:
        raise NotNative("fixed-atom RATTLE uses the Python constraint operator")
    if constraint is not None and (getattr(constraint, "estimator", None) != "centroid" or bias is not None or fragment_data):
        raise NotNative("combined constraints/biases require the Python path")
    fragments = []
    for indices, masses, total, reference in fragment_data:
        weights = np.zeros(len(dynamics.masses));weights[indices] = masses
        fragments.append({"weights": weights.tolist(), "reference": reference.tolist()})
    memo = {}
    return binding, {"beta": dynamics.beta, "time": clock, "cvs": cv_descriptors(cv, len(dynamics.masses), memo),
                     "biases": bias_descriptors(bias, dynamics, memo), "constraint": native_constraint,
                     "constraint_tolerance": tolerance, "fragments": fragments, "record_energy": record_energy,
                     **fixed_descriptor(dynamics)}


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


def _refresh_inputs(refresh, steps, dynamics=None, *, projection=None):
    """Normalize a RefreshPlan or a low-level refresh mapping."""
    if refresh is None:
        return None, None, None
    plan = refresh if hasattr(refresh, "as_options") else None
    if plan is not None:
        if not hasattr(plan, "commit") or not hasattr(plan, "steps"):
            raise TypeError("refresh plan must provide steps, as_options and commit")
        if steps > int(plan.steps):
            steps = int(plan.steps)
        selected = plan.projection if projection is None else projection
        options = plan.as_options(selected)
    elif isinstance(refresh, dict):
        options = dict(refresh)
        if projection is not None:
            options["projection"] = projection
    else:
        raise TypeError("refresh must be a RefreshPlan or mapping")
    selected = options.get("projection", "none")
    if selected not in ("none", "centroid_constraint", "fragments"):
        raise ValueError("refresh projection must be none, centroid_constraint or fragments")
    raw_offsets = np.asarray(options.get("offsets"))
    raw_momenta = np.asarray(options.get("momenta"))
    if raw_offsets.dtype != np.dtype(np.int64) or raw_momenta.dtype != np.dtype(np.float64):
        raise ValueError("refresh offsets must be int64 and momenta must be float64")
    offsets = raw_offsets
    momenta = raw_momenta
    expected_shape = None if dynamics is None else tuple(dynamics.integrator.shape)
    if (offsets.ndim != 1 or momenta.ndim != 4 or momenta.shape[0] != offsets.size
            or (expected_shape is not None and tuple(momenta.shape[1:]) != expected_shape)):
        raise ValueError("refresh requires offsets [R] and momenta [R,P,N,3]")
    if offsets.size and (np.any(offsets < 0) or np.any(offsets[1:] <= offsets[:-1])
                         or int(offsets[-1]) >= steps):
        raise ValueError("refresh offsets must be strictly increasing and lie in the block")
    if momenta.size and not np.isfinite(momenta).all():
        raise ValueError("refresh momenta must be finite")
    options["offsets"] = np.ascontiguousarray(offsets, dtype=np.int64)
    options["momenta"] = np.ascontiguousarray(momenta, dtype=np.float64)
    options["projection"] = selected
    return plan, options, steps


def _apply_python_refresh(dynamics, state, momentum, projection, constraint, fragment_data):
    """Apply one pre-drawn refresh using the historical Python projections."""
    if projection == "none":
        state.p[:] = momentum
    elif projection == "centroid_constraint":
        if constraint is None:
            raise ValueError("centroid_constraint refresh requires a constraint CV")
        from .workflows.rate_constraints import project_momenta
        state.p[:] = momentum
        project_momenta(state.p, state.q, constraint, dynamics.masses)
    elif projection == "fragments":
        if not fragment_data:
            raise ValueError("fragments refresh requires fragment_data")
        state.p[:] = momentum
        dynamics._project_fragments(state.q, state.p, fragment_data)
    dynamics._project_fixed(state.q, state.p)


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
        return checked_integer(
            outcome.get(name, default), f"outcome {name}", minimum,
            np.iinfo(np.int64).max,
            message="{name} must be an integer >= {minimum}",
        )

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
    if beads == 1:
        # One bead: q - roll(q) and p - mean(p) are exactly zero, and the
        # centroid kinetic energy is the same reduction as the total one.
        return kinetic, 0.0, kinetic, 0.0
    dq = q - np.roll(q, -1, axis=0)
    spring = float(0.5 * (beads / dynamics.beta) ** 2 * np.sum(masses * dq * dq))
    center = p.mean(axis=0, keepdims=True)
    centroid = float(np.sum(center * center / (2 * masses)))
    internal = float(np.sum((p - center) ** 2 / (2 * masses))) / beads
    return kinetic, spring, centroid, internal


def _ring_radii(q, center=None):
    if len(q) == 1:
        return np.zeros(q.shape[1])
    centered = q - (q.mean(axis=0) if center is None else center)
    radii = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=0))
    if not np.isfinite(radii).all():
        raise FloatingPointError("nonfinite ring radius diagnostics")
    return radii


# Built-in biases whose energy_force is a pure function of q while a block
# runs (history/centre changes are explicit events between blocks). Keyed by
# the defining module and qualname, so checking never imports optional modules.
_PURE_BIASES = {
    "pyrpmd.bias": {"UmbrellaBias", "FluxMetricBias", "CompositeBias", "WallBias",
                    "MetadynamicsBias", "GridMetadynamicsBias"},
    "pyrpmd.steered": {"SteeredBias"},
    "pyrpmd.opes": {"OPESBias", "OPESExploreBias"},
    "pyrpmd.ves": {"VESBias"},
    "pyrpmd.abf": {"ABFBias"},
    "pyrpmd.workflows.tempering": {"_ScaledBias"},
}


def _pure_bias(bias):
    """Whether repeated energy_force calls at one q may share one result.

    Only exact, unpatched built-in classes qualify; wrappers qualify when
    every wrapped bias does. Their CVs are evaluated as the same pure
    mathematics. User-defined biases keep the reference call schedule.
    """
    kind = type(bias)
    method = getattr(kind, "energy_force", None)
    if (kind.__qualname__ not in _PURE_BIASES.get(kind.__module__, ())
            or "energy_force" in getattr(bias, "__dict__", {})
            or getattr(method, "__module__", None) not in _PURE_BIASES
            or getattr(method, "__qualname__", "").rpartition(".")[::2]
            not in {(name, "energy_force") for name in _PURE_BIASES[method.__module__]}):
        return False
    if kind.__qualname__ == "CompositeBias":
        return all(_pure_bias(part) for part in bias.biases)
    if kind.__qualname__ == "_ScaledBias":
        return _pure_bias(bias.bias)
    return True


def _fast_python_step_supported(dynamics, bias, constraint, fragment_data, composed):
    """Whether a plain step may reuse its endpoint PES/bias result in this block.

    Only the unmodified Dynamics/CayleyIntegrator mathematics without
    constraint, fragment projection or configured operators qualifies, with
    no bias or a pure built-in bias; it then performs exactly the same kicks,
    free step and projections as Dynamics._propagate. A provider declaring
    itself stateful keeps the reference evaluation schedule, since repeated
    calls may differ.
    """
    if composed or constraint is not None or fragment_data:
        return False
    if bias is not None and not _pure_bias(bias):
        return False
    from .dynamics import (Dynamics, CayleyIntegrator, _BLOCK_DYNAMICS_METHODS,
                           _BLOCK_INTEGRATOR_METHODS)
    from .propagation import _original_methods
    if (type(dynamics) is not Dynamics or dynamics.bond_constraints
            or type(dynamics.integrator) is not CayleyIntegrator
            or not _original_methods(dynamics, _BLOCK_DYNAMICS_METHODS)
            or not _original_methods(dynamics.integrator, _BLOCK_INTEGRATOR_METHODS)):
        return False
    return _stateless_provider(dynamics)


def _stateless_provider(dynamics):
    """Whether repeated PES calls at identical coordinates are interchangeable."""
    metadata = getattr(dynamics.potential, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    capabilities = metadata.get("capabilities", {})
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    return not any(source.get(key) is True for source in (metadata, capabilities)
                   for key in ("stateful", "history_dependent", "time_dependent"))


def _endpoint_force(dynamics, q, force, bias=None):
    """Write the biased physical force into a block-owned buffer.

    Same arithmetic as Dynamics._force. Returns (sum of PES energies, bias
    energy), the bias energy being None without a bias.
    """
    from .dynamics import evaluate_checked
    result = evaluate_checked(dynamics.potential, q)
    np.copyto(force, result.forces)
    bias_energy = None
    if bias is not None:
        bias_energy, extra = bias.energy_force(q)
        if np.shape(extra) != q.shape or not np.isfinite(extra).all():
            raise ValueError("bias returned invalid forces")
        force += extra
        bias_energy = float(bias_energy)
    if dynamics.fixed_atoms:
        force[:, dynamics._fixed_indices, :] = 0.0
    return float(np.sum(result.energy)), bias_energy


def _fast_python_step(dynamics, state, force, force_valid, bias=None):
    """Dynamics._propagate on the live arrays; the caller owns the rollback.

    Returns (summed PES energy, bias energy) at the new coordinates.
    ``force`` holds the endpoint force afterwards, so a following step of the
    same block may start from it.
    """
    from .dynamics import _state_clock
    step, time = _state_clock(state.step, state.time)
    next_time = time + dynamics.dt
    if not np.isfinite(next_time):
        raise FloatingPointError("nonfinite integration clock")
    q, p = state.q, state.p
    integrator = dynamics.integrator
    h = 0.5 * dynamics.dt
    if not force_valid:
        _endpoint_force(dynamics, q, force, bias)
    integrator.kick(p, force, h)
    dynamics._project_fixed(q, p)
    integrator.free_step(q, p)
    dynamics._project_fixed(q, p)
    energies = _endpoint_force(dynamics, q, force, bias)
    integrator.kick(p, force, h)
    dynamics._project_fixed(q, p)
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise FloatingPointError("nonfinite integration state")
    state.step, state.time = step + 1, next_time
    return energies


def _python_energy_parts(dynamics, state, bias):
    """The same PES call and total-energy convention as Dynamics.energy.

    Internal here means noncentroid *bead* momenta, not molecular vibrations.
    Bias energy belongs to the recorded total but not to these physical parts.
    """
    from .dynamics import evaluate_checked
    q = state.q
    return _energy_parts_from_potential(
        dynamics, state, bias, float(np.sum(evaluate_checked(dynamics.potential, q).energy)))


def _energy_parts_from_potential(dynamics, state, bias, potential, bias_energy=None):
    q = state.q
    beads = dynamics.nbeads
    kinetic, spring, centroid, internal = _nuclear_energy_terms(dynamics, state)
    parts = np.array([potential / beads, centroid, internal, spring / beads])
    energy = potential + kinetic + spring
    if bias is not None:
        energy += float(bias.energy_force(q)[0]) if bias_energy is None else bias_energy
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


def _composed_native_prefix(dynamics, state, steps, mask, record_energy,
                            check_energy, diagnostics, backend, raise_on_error):
    """Use the persistent native operator executor with per-step transactions."""
    component = dynamics._composition
    operators = dynamics._operator_rollback_token()
    initial_step, initial_time = state.step, state.time
    try:
        engine = component._prepare_native(state, backend=backend)
        if engine is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            clocks = np.add.accumulate(np.r_[float(initial_time), np.full(steps, dynamics.dt)])[1:]
        invalid = np.flatnonzero(~np.isfinite(clocks))
        allowed = int(invalid[0]) if len(invalid) else steps
        native = engine.run_prefix(
            allowed, observe=mask[:allowed], record_energy=record_energy,
            check_energy=check_energy,
            energy_parts=bool(diagnostics.get("energy_parts")),
            ring_radii=bool(diagnostics.get("ring_radii")), _outer_transaction=True)
        completed = native["completed"]
        error = native["error"]
        if completed == allowed and allowed < steps and error is None:
            error = "FloatingPointError: nonfinite integration clock"
        accepted = engine._physical_state()
        component._native_state = accepted
        columns, samples = native["columns"], native["samples"]
        selected = mask[:completed].astype(bool)
        energies = samples[:, columns.index("ring_polymer_hamiltonian")]
        centroid_q = native["centroids"]
        centroid_p = native["centroid_momenta"]
        arrays = {}
        if diagnostics.get("energy_parts"):
            arrays["energy_parts"] = native["energy_parts"]
        if diagnostics.get("ring_radii"):
            arrays["ring_radii"] = np.sqrt(native["radii_gyration_squared"])
        used = "native" if native.get("python_step_callbacks", 0) == 0 else "python"
        reason = None if used == "native" else "native operators with Python PES callbacks"
        result = BlockResult(steps, completed, used, reason,
                             initial_step + 1 + np.flatnonzero(selected), clocks[:completed][selected],
                             energies if record_energy else np.full(int(selected.sum()), np.nan),
                             centroid_q, centroid_p,
                             np.empty((int(selected.sum()), 0)), None, None, error,
                             diagnostics=arrays or None)
    except BaseException:
        dynamics._restore_operator_rollback_token(operators)
        raise
    if completed:
        state.q[:], state.p[:] = accepted["q"], accepted["p"]
        state.step, state.time = initial_step + completed, float(clocks[completed-1])
        if dynamics._initial_cell is not None:
            dynamics.cell = accepted["cell"].copy()
    elif error:
        # An unsuccessful first step cannot select/seed a new bath.
        dynamics._restore_operator_rollback_token(operators)
    dynamics.last_step_backend = used
    dynamics.propagation_stats[used + "_steps"] += completed
    if reason:
        reasons = dynamics.propagation_stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + completed
    if error and raise_on_error:
        raise PropagationError(result)
    return result


def propagate_block(dynamics, state, steps, *, bias=None, cv=None, constraint=None,
                    constraint_tolerance=1e-10, fragment_data=(), observe=None,
                    record_energy=True, record_states=False, backend=None, raise_on_error=True,
                    outcome=None, record_previous=False, reuse_forces=None,
                    check_energy=False, diagnostics=None, record_step_energies=False,
                    refresh=None):
    """Advance at most 4096 selected-integrator steps, observing in step order.

    ``observe`` is a length-steps boolean mask (default: all steps). A failure
    leaves q/p/clock at the last successfully completed step. Its requested
    observations are part of the same step transaction.
    ``PropagationError.result`` retains the valid prefix. Legacy and Python
    force caches are block-local. A composed native executor refreshes forces
    after q/p edits and requires its bound provider to remain immutable.
    ``outcome`` opts into a supported native classifier and its persistence
    state machine; it can stop successfully before the requested block ends.
    A comparison within floating-point roundoff of a classification boundary
    yields ``outcome['pending_check']`` without updating persistence. The caller
    must run the original Python classifier at that step before continuing.
    Unsupported backends reject this option before mutating the state.
    ``record_previous`` retains only the state before the final accepted step
    of a successful block, allowing a Python classifier to reject that step.
    ``reuse_forces`` controls block-local force reuse: each step starts from
    the previous step's endpoint PES/bias force at the same coordinates, i.e.
    one PES call per step instead of two. The results are identical for a
    deterministic provider. ``None`` (default) enables it unless the provider
    declares itself stateful, history- or time-dependent; ``False`` restores
    the two-call reference schedule. The native loop reuses forces with or
    without its pure umbrella/flux biases, centroid RATTLE and fixed atoms;
    fragment projection, which moves q after the force call, re-evaluates.
    No cache survives a block.
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
    ``record_step_energies`` returns ``BlockResult.step_energies``, the total
    ring energy of every completed step, independent of ``observe``. It lets
    a caller monitor conservation at every step while observing (centroids,
    diagnostics, CVs) only the frames it saves. Plain Python steps of a
    non-stateful PES, unbiased or with a pure built-in bias (umbrella, flux,
    wall, metadynamics, OPES, VES, ABF, steered, composites), take that energy
    from the step's endpoint PES/bias call,
    like the native loop; with ``reuse_forces`` they also start the next step
    from that endpoint force, approaching one PES call per step.
    Blocks must end at thermostat/Python-classifier/checkpoint boundaries.
    Legacy sampling RNG remains in Python at those boundaries; configured
    native baths carry their own checkpointed stream inside the executor.
    ``refresh`` accepts a :class:`pyrpmd.momentum_refresh.RefreshPlan` (or its
    low-level mapping) and installs pre-drawn Andersen momenta at the supplied
    offsets before the corresponding steps.  The plan's RNG replay callback is
    committed only for the successful refresh prefix reported by the block.
    """
    if getattr(dynamics, "_stochastic_propagation", False):
        raise ValueError("stochastic dynamics require step() or transactional propagate(), not propagate_block()")
    composed = getattr(dynamics, "plan", None) is not None and dynamics.plan.active
    if composed and constraint is not None:
        raise ValueError("configured baths/force splitting with RATTLE require a projected constraint operator")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or not 0 <= steps <= 4096:
        raise ValueError("steps must be an integer in [0, 4096]")
    steps = int(steps)
    requested_steps = steps
    backend = backend or getattr(dynamics, "loop_backend", "auto")
    if backend not in ("auto", "native", "python"):
        raise ValueError("loop backend must be auto, native or python")
    refresh_plan, refresh_options, planned_steps = _refresh_inputs(refresh, steps, dynamics)
    if refresh_plan is not None and refresh_plan.start_step != state.step:
        raise ValueError("refresh plan start_step must match state.step")
    if planned_steps is not None:
        steps = int(planned_steps)
    if refresh_options is not None:
        projection = refresh_options["projection"]
        if projection == "centroid_constraint" and constraint is None:
            raise ValueError("centroid_constraint refresh requires a constraint CV")
        if projection == "fragments" and not fragment_data:
            raise ValueError("fragments refresh requires fragment_data")
        if composed:
            raise RuntimeError("momentum refresh is incompatible with composed operators")
        if outcome is not None or record_previous:
            raise RuntimeError("momentum refresh is incompatible with outcome/rollback descriptors")
    if reuse_forces is None:
        reuse_forces = _stateless_provider(dynamics)
    if not isinstance(record_previous, (bool, np.bool_)) or not isinstance(reuse_forces, (bool, np.bool_)):
        raise ValueError("record_previous and reuse_forces must be boolean")
    if not isinstance(check_energy, (bool, np.bool_)):
        raise ValueError("check_energy must be boolean")
    if not isinstance(record_step_energies, (bool, np.bool_)):
        raise ValueError("record_step_energies must be boolean")
    if record_step_energies and composed:
        # The composed operator executor reports energies per observed frame.
        # Observe every step there, then keep only the requested frames.
        return _composed_with_step_energies(
            dynamics, state, steps, observe, bias=bias, cv=cv, constraint=constraint,
            constraint_tolerance=constraint_tolerance, fragment_data=fragment_data,
            record_energy=record_energy, record_states=record_states, backend=backend,
            raise_on_error=raise_on_error, outcome=outcome, record_previous=record_previous,
            reuse_forces=reuse_forces, check_energy=check_energy, diagnostics=diagnostics)
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
        if refresh_plan is not None and values.shape == (requested_steps,) and steps != requested_steps:
            values = values[:steps]
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
    if backend != "python" and not composed:
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
            if record_step_energies and getattr(dynamics.integrator._native, "BLOCK_STEP_ENERGIES_API", 0) < 1:
                binding = None
                raise NotNative("installed native extension lacks per-step energy records; rebuild it")
            if refresh_options is not None and getattr(
                    dynamics.integrator._native, "MOMENTUM_REFRESH_API", 0) < 1:
                binding = None
                raise NotNative("installed native extension lacks momentum refresh support; rebuild it")
        except (NotNative, NativeUnavailableError) as exc:
            reason = str(exc)
            if backend == "native":
                raise RuntimeError("native loop required but unsupported: " + reason) from exc
    selected = "native" if binding is not None else "python"
    if outcome_spec is not None and selected != "native":
        raise RuntimeError("native outcomes require a supported native loop: " + (reason or "Python backend selected"))
    completed = written = 0;error = None
    refreshes_completed = 0
    step_energies = np.full(steps, np.nan) if record_step_energies else None
    metadata = {}
    previous_time = None
    if (composed and steps and bias is None and cv is None and constraint is None
            and not fragment_data and not record_states and not record_previous and outcome is None):
        result = _composed_native_prefix(dynamics, state, steps, mask, record_energy,
                                         check_energy, diagnostic_spec, backend, raise_on_error)
        if result is not None:
            return result
    if composed and steps and backend == "native":
        raise RuntimeError("native composed prefix does not support these observation/constraint callbacks; use auto/python")
    if steps and selected == "native":
        # The lease pins the owner and prevents concurrent context destruction.
        # The integrator lock protects all use of this Cayley context.
        options.update(outcome=outcome_spec, previous=previous_state, reuse_forces=bool(reuse_forces))
        if check_energy:
            options["check_energy"] = True
        if diagnostic_arrays:
            options["diagnostics"] = diagnostic_arrays
        if step_energies is not None:
            options["step_energies"] = step_energies
        if refresh_options is not None:
            options["refresh"] = refresh_options
        with binding.acquire() as descriptor, dynamics.integrator._lock:
            try:
                native_result = dynamics.integrator._native.advance(
                    dynamics.integrator._context, descriptor, state.q, state.p, options, mask, records, snapshots)
            except BaseException:
                if refresh_plan is not None:
                    refresh_plan.commit(0)
                raise
        _check_bias_snapshot(bias, options["biases"])
        completed, written, clock, message = native_result[:4]
        if len(native_result) == 5:
            metadata = native_result[4]
        refreshes_completed = int(metadata.get("refreshes_completed", 0)) if refresh_options is not None else 0
        if refresh_plan is not None:
            refresh_plan.commit(refreshes_completed)
        state.step += completed;state.time = clock;error = message or None
        previous_time = metadata.get("previous_time")
    elif steps:
        cache = {}
        fast = _fast_python_step_supported(dynamics, bias, constraint, fragment_data, composed)
        reuse = fast and bool(reuse_forces)
        force = np.empty_like(state.q) if fast else None
        force_valid = False
        refresh_index = 0
        # Reused transaction buffers: a failed step restores exactly this state.
        backup_q = np.empty_like(state.q);backup_p = np.empty_like(state.p)
        backup_step = backup_time = None
        for i in range(steps):
            refreshed_this_step = False
            np.copyto(backup_q, state.q);np.copyto(backup_p, state.p)
            backup_step, backup_time = state.step, state.time
            operators = dynamics._operator_rollback_token() if composed else None
            previous_stats = copy.deepcopy(dynamics.propagation_stats) if composed else None
            try:
                if (refresh_options is not None and refresh_index < len(refresh_options["offsets"])
                        and int(refresh_options["offsets"][refresh_index]) == i):
                    _apply_python_refresh(
                        dynamics, state, refresh_options["momenta"][refresh_index],
                        refresh_options["projection"], constraint, fragment_data)
                    if refresh_options["projection"] == "fragments":
                        force_valid = False
                    refreshed_this_step = True
                if check_energy and i == 0:
                    initial_energy = dynamics.energy(state, bias)
                    if not np.isfinite(initial_energy):
                        raise FloatingPointError("nonfinite total ring energy")
                potential = bias_energy = None
                if fast:
                    potential, bias_energy = _fast_python_step(dynamics, state, force, force_valid, bias)
                    force_valid = reuse
                elif composed:
                    dynamics._composition.step(state, bias, fragment_data, cache=cache, backend=backend)
                elif constraint is not None:
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
                    if potential is None:
                        parts, energy = _python_energy_parts(dynamics, state, bias)
                    else:
                        parts, energy = _energy_parts_from_potential(
                            dynamics, state, bias, potential, bias_energy)
                    diagnostic_arrays["energy_parts"][written] = parts
                elif check_energy or step_energies is not None or (record_energy and mask[i]):
                    if potential is None:
                        energy = dynamics.energy(state, bias)
                    else:
                        kinetic, spring, _, _ = _nuclear_energy_terms(dynamics, state)
                        energy = potential + kinetic + spring
                        if bias_energy is not None:
                            energy += bias_energy
                    if not np.isfinite(energy):
                        raise FloatingPointError("nonfinite total ring energy")
                if step_energies is not None:
                    step_energies[i] = energy
                if mask[i]:
                    row = records[written];row[0] = state.time
                    if record_energy:
                        row[1] = energy
                    center = state.q[0] if dynamics.nbeads == 1 else state.q.mean(axis=0)
                    row[2:2+nd] = center.ravel()
                    row[2+nd:2+2*nd] = (state.p[0] if dynamics.nbeads == 1 else state.p.mean(axis=0)).ravel()
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
                force_valid = False
                if operators is not None:
                    dynamics._restore_operator_rollback_token(operators)
                    dynamics.propagation_stats = previous_stats
                state.q[:] = backup_q;state.p[:] = backup_p
                state.step, state.time = backup_step, backup_time
                error = f"{type(exc).__name__}: {exc}"
                break
            completed += 1;written += int(mask[i])
            if refreshed_this_step:
                refresh_index += 1
                refreshes_completed += 1
        if refresh_plan is not None:
            refresh_plan.commit(refreshes_completed)
        if previous_state is not None and completed and error is None:
            previous_state[0] = backup_q;previous_state[1] = backup_p
            previous_time = backup_time
    records = records[:written];snapshots = None if snapshots is None else snapshots[:written]
    indices = np.flatnonzero(mask[:completed]) + start_step + 1
    result = BlockResult(steps, completed, selected, reason, indices, records[:, 0], records[:, 1],
                         records[:, 2:2+nd].reshape(-1, natoms, 3),
                         records[:, 2+nd:2+2*nd].reshape(-1, natoms, 3), records[:, 2+2*nd:],
                         None if snapshots is None else snapshots[:, 0],
                         None if snapshots is None else snapshots[:, 1], error,
                          outcome=None if outcome_spec is None else _outcome_result(outcome_spec, metadata),
                          diagnostics={name: values[:written] for name, values in diagnostic_arrays.items()}
                          if diagnostic_arrays else None,
                          step_energies=None if step_energies is None else step_energies[:completed],
                          refreshes_completed=refreshes_completed)
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


def _composed_with_step_energies(dynamics, state, steps, observe, **options):
    raise_on_error = options.pop("raise_on_error")
    record_energy = options.pop("record_energy")
    if observe is None:
        keep = np.ones(steps, dtype=bool)
    else:
        keep = np.asarray(observe)
        if keep.shape != (steps,) or keep.dtype.kind not in "bui" or np.any((keep != 0) & (keep != 1)):
            raise ValueError("observe must be a boolean/binary mask with length steps")
        keep = keep.astype(bool)
    result = propagate_block(dynamics, state, steps, observe=None, record_energy=True,
                             raise_on_error=False, **options)
    selected = keep[:result.completed]
    result.step_energies = result.energies.copy()
    if not record_energy:
        result.energies = np.full(int(selected.sum()), np.nan)
    else:
        result.energies = result.energies[selected]
    for name in ("steps", "times", "centroids", "centroid_momenta", "values", "q", "p"):
        value = getattr(result, name)
        if value is not None:
            setattr(result, name, value[selected])
    if result.diagnostics:
        result.diagnostics = {name: value[selected] for name, value in result.diagnostics.items()}
    if result.error and raise_on_error:
        raise PropagationError(result)
    return result


def execution_options(config):
    from .propagation import loop_options
    return loop_options(config)


def scientific_config(config):
    """Execution-only choices do not invalidate a physical checkpoint."""
    from .propagation import scientific_config as shared_scientific_config
    return shared_scientific_config(config)
