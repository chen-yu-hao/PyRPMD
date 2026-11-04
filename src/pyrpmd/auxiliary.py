"""Auxiliary-degree-of-freedom operators for composed ring-polymer dynamics.

The protocol in this module is intentionally Python-first.  An auxiliary
operator owns non-nuclear state (mapping variables, spins, active surfaces,
memory kernels, ...), advances it at fixed nuclear coordinates, and exposes
its nuclear force, energy, persistence state and scientific identity.  The
nuclear propagation adapter remains responsible for transaction boundaries.
"""
from __future__ import annotations

import copy
import json
import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np


class AuxiliaryLayout(str, Enum):
    """How many auxiliary copies are attached to a ring polymer."""

    PER_RING = "per-ring"
    PER_BEAD = "per-bead"


class AuxiliarySplitPoint(str, Enum):
    """Validated insertion points in the nuclear operator composition."""

    SYMMETRIC_AROUND_FREE = "symmetric-around-free"


def _names(values, name):
    try:
        result = frozenset(str(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of names") from exc
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain nonempty names")
    return result


@dataclass(frozen=True)
class AuxiliaryCompatibility:
    """An explicit physical whitelist for one auxiliary Hamiltonian.

    ``thermostats`` describes baths acting during real-time propagation.
    ``preparation_thermostats`` is separate because freezing the auxiliary
    variables while preparing nuclear coordinates is a different Hamiltonian.
    """

    thermostats: frozenset[str] = field(default_factory=lambda: frozenset({"none"}))
    preparation_thermostats: frozenset[str] = field(
        default_factory=lambda: frozenset({"none"})
    )
    barostats: frozenset[str] = field(default_factory=lambda: frozenset({"none"}))
    force_splitting: frozenset[str] = field(default_factory=lambda: frozenset({"none"}))
    allow_pbc: bool = False
    allow_constraints: bool = False
    allow_bias: bool = False
    rationale: str = ""

    def __post_init__(self):
        object.__setattr__(self, "thermostats", _names(self.thermostats, "thermostats"))
        object.__setattr__(
            self,
            "preparation_thermostats",
            _names(self.preparation_thermostats, "preparation_thermostats"),
        )
        object.__setattr__(self, "barostats", _names(self.barostats, "barostats"))
        object.__setattr__(
            self, "force_splitting", _names(self.force_splitting, "force_splitting")
        )
        for name in ("allow_pbc", "allow_constraints", "allow_bias"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("an auxiliary compatibility whitelist requires a rationale")

    def validate(
        self,
        *,
        thermostat: str = "none",
        preparation_thermostat: str = "none",
        barostat: str = "none",
        force_splitting: str = "none",
        pbc: bool = False,
        constraints: bool = False,
        bias: bool = False,
        label: str = "auxiliary operator",
    ) -> None:
        """Reject every combination not explicitly listed, without side effects."""

        checks = (
            (thermostat, self.thermostats, "thermostat"),
            (preparation_thermostat, self.preparation_thermostats, "preparation thermostat"),
            (barostat, self.barostats, "barostat"),
            (force_splitting, self.force_splitting, "force splitting"),
        )
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(
                    f"{label} does not whitelist {name}={value!r}; "
                    f"allowed: {', '.join(sorted(allowed))}. {self.rationale}"
                )
        if pbc and not self.allow_pbc:
            raise ValueError(f"{label} does not whitelist periodic boundaries. {self.rationale}")
        if constraints and not self.allow_constraints:
            raise ValueError(f"{label} does not whitelist nuclear constraints. {self.rationale}")
        if bias and not self.allow_bias:
            raise ValueError(f"{label} does not whitelist a bias potential. {self.rationale}")


@runtime_checkable
class AuxiliaryOperator(Protocol):
    """Runtime-checkable contract for non-nuclear propagation state."""

    layout: AuxiliaryLayout
    nstates: int
    split_points: frozenset[AuxiliarySplitPoint]
    compatibility: AuxiliaryCompatibility

    @property
       def state_arrays(self) -> Mapping[str, np.ndarray]: ...

    def propagate(self, dt: float, q: np.ndarray) -> None: ...

    def nuclear_force(self, q: np.ndarray) -> np.ndarray: ...

    def energy(self, q: np.ndarray) -> float: ...

    def state_dict(self) -> dict: ...

    def restore_state_dict(self, state: Mapping[str, object]) -> None: ...

    def identity(self) -> dict: ...


def validate_auxiliary_operator(
    operator: AuxiliaryOperator, *, nbeads: int, metadata: bool = True,
    checkpoint: bool = False,
) -> None:
    """Validate protocol shape and metadata without propagation or RNG draws.

    ``checkpoint=True`` additionally verifies JSON persistence.  The hot
    propagation path deliberately performs only structural/array validation;
    serializing an O(P*S) state every timestep would turn validation into an
    accidental trajectory cost.
    """

    if not isinstance(operator, AuxiliaryOperator):
        raise TypeError("auxiliary operator does not satisfy AuxiliaryOperator")
    if isinstance(nbeads, (bool, np.bool_)) or not isinstance(nbeads, numbers.Integral) or nbeads < 1:
        raise ValueError("nbeads must be a positive integer")
    if not isinstance(operator.layout, AuxiliaryLayout):
        raise TypeError("auxiliary layout must be AuxiliaryLayout")
    if (isinstance(operator.nstates, (bool, np.bool_))
            or not isinstance(operator.nstates, numbers.Integral) or operator.nstates < 1):
        raise ValueError("auxiliary nstates must be a positive integer")
    if not isinstance(operator.split_points, frozenset) or not operator.split_points:
        raise ValueError("auxiliary split_points must be a nonempty frozenset")
    if any(not isinstance(point, AuxiliarySplitPoint) for point in operator.split_points):
        raise TypeError("auxiliary split_points must contain AuxiliarySplitPoint values")
    if not isinstance(operator.compatibility, AuxiliaryCompatibility):
        raise TypeError("auxiliary compatibility must be AuxiliaryCompatibility")
    arrays = operator.state_arrays
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("auxiliary state_arrays must be a nonempty mapping")
    expected = 1 if operator.layout is AuxiliaryLayout.PER_RING else int(nbeads)
    seen = []
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("auxiliary state array names must be nonempty strings")
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or not value.flags.c_contiguous or not value.flags.writeable
                or value.ndim < 1 or value.shape[0] != expected
                or not np.isfinite(value).all()):
            raise ValueError(
                f"auxiliary state array {name!r} must be writable C-contiguous "
                f"float64 with leading dimension {expected}"
            )
        if any(np.shares_memory(value, other) for other in seen):
            raise ValueError("auxiliary state arrays must not share memory")
        seen.append(value)
    if metadata or checkpoint:
        try:
            if metadata:
                json.dumps(operator.identity(), sort_keys=True, allow_nan=False)
            if checkpoint:
                json.dumps(operator.state_dict(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            label = "auxiliary identity and checkpoint state" if metadata and checkpoint else (
                "auxiliary checkpoint state" if checkpoint else "auxiliary identity"
            )
            raise ValueError(f"{label} must be finite JSON data") from exc


class AuxiliaryPropagationCore:
    """Transactional symmetric auxiliary/free-ring/auxiliary composition.

    The factory binds a fresh operator to a copied trial state.  This preserves
    the legacy all-or-nothing state commit while allowing an auxiliary exact
    flow (such as NRPMD MInt) to apply its integrated nuclear impulse.
    """

    checkpoint_version = 1

    def __init__(
        self,
        *,
        dt: float,
        nbeads: int,
        operator_factory: Callable[[object], AuxiliaryOperator],
        free_step: Callable[[np.ndarray, np.ndarray], None],
        validate_state: Callable[[object], None],
        state_fields: tuple[str, ...] = ("q", "p"),
        split_point: AuxiliarySplitPoint = AuxiliarySplitPoint.SYMMETRIC_AROUND_FREE,
    ):
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, numbers.Real):
            raise ValueError("auxiliary core dt must be a finite real number")
        self.dt = float(dt)
        if not math.isfinite(self.dt):
            raise ValueError("auxiliary core dt must be a finite real number")
        if isinstance(nbeads, (bool, np.bool_)) or not isinstance(nbeads, numbers.Integral) or nbeads < 1:
            raise ValueError("auxiliary core nbeads must be a positive integer")
        if not callable(operator_factory) or not callable(free_step) or not callable(validate_state):
            raise TypeError("auxiliary core callbacks must be callable")
        if (not isinstance(state_fields, tuple) or not state_fields
                or any(not isinstance(name, str) or not name for name in state_fields)):
            raise ValueError("state_fields must be a nonempty tuple of attribute names")
        if not isinstance(split_point, AuxiliarySplitPoint):
            raise TypeError("split_point must be AuxiliarySplitPoint")
        self.nbeads = int(nbeads)
        self.operator_factory = operator_factory
        self.free_step = free_step
        self.validate_state = validate_state
        self.state_fields = state_fields
        self.split_point = split_point

    def operator(self, state) -> AuxiliaryOperator:
        operator = self.operator_factory(state)
        validate_auxiliary_operator(operator, nbeads=self.nbeads, metadata=False)
        if self.split_point not in operator.split_points:
            raise ValueError(
                f"auxiliary operator does not allow split insertion {self.split_point.value!r}"
            )
        return operator

    def step(self, state):
        """Advance one complete reversible step and commit only on success."""

        self.validate_state(state)
        trial = state.copy()
        operator = self.operator(trial)
        operator.propagate(0.5 * self.dt, trial.q)
        self.free_step(trial.q, trial.p)
        operator.propagate(0.5 * self.dt, trial.q)
        next_time = float(trial.time) + self.dt
        if not math.isfinite(next_time):
            raise ValueError("nonfinite auxiliary propagation clock")
        trial.step = int(trial.step) + 1
        trial.time = next_time
        self.validate_state(trial)
        fields = self.state_fields + tuple(operator.state_arrays)
        if len(fields) != len(set(fields)):
            raise ValueError("nuclear and auxiliary commit fields overlap")
        for name in fields:
            getattr(state, name)[:] = getattr(trial, name)
        state.step, state.time = trial.step, trial.time
        return state

    def identity(self, state) -> dict:
        operator = self.operator(state)
        return {
            "schema": "pyrpmd.auxiliary-propagation-core",
            "version": self.checkpoint_version,
            "dt": self.dt,
            "nbeads": self.nbeads,
            "split_point": self.split_point.value,
            "operator": copy.deepcopy(operator.identity()),
        }


__all__ = [
    "AuxiliaryLayout",
    "AuxiliarySplitPoint",
    "AuxiliaryCompatibility",
    "AuxiliaryOperator",
    "AuxiliaryPropagationCore",
    "validate_auxiliary_operator",
]
