"""Planning and committing deterministic Andersen momentum refreshes.

The legacy Andersen path owns both the NumPy random generator and the
checkpointable collision clock. This module only borrows those objects while
planning a block; the caller-visible state is changed by :meth:`RefreshPlan.commit`
after the native or Python block reports how many refreshes actually committed.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import numbers

import numpy as np


MAX_PREDRAWN_BYTES = 64 * 1024**2


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass
class RefreshPlan:
    """A pre-drawn refresh block and its transactional submission function."""

    dynamics: object
    schedule: object
    start_step: int
    steps: int
    offsets: np.ndarray
    momenta: np.ndarray
    rng_state: dict
    schedule_state: int | None
    projection: str = "none"
    refreshes_committed: int = 0
    committed: bool = False

    def __post_init__(self):
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.momenta = np.asarray(self.momenta, dtype=np.float64)
        self.offsets.setflags(write=False)
        self.momenta.setflags(write=False)

    @property
    def refresh_count(self):
        return int(self.offsets.size)

    @property
    def max_steps(self):
        """Alias used by callers that treat a plan as a bounded block."""
        return self.steps

    def as_options(self, projection=None):
        """Return the native option mapping for this plan."""
        selected = self.projection if projection is None else projection
        if selected not in ("none", "centroid_constraint", "fragments"):
            raise ValueError("refresh projection must be none, centroid_constraint or fragments")
        return {"offsets": self.offsets, "momenta": self.momenta, "projection": selected}

    def commit(self, refreshes_completed):
        """Commit exactly the successful prefix reported by a propagation block.

        Replaying the old scalar calls from the planning snapshot is deliberate:
        it preserves the exact Generator state and geometric-clock draw order,
        including a block that failed immediately after a refresh.
        """
        count = _integer(refreshes_completed, "refreshes_completed")
        if count > self.refresh_count:
            raise ValueError("refreshes_completed exceeds the planned refresh count")
        if self.committed and count != self.refreshes_committed:
            raise ValueError("refresh plan was already committed with a different prefix")
        if self.committed and count == self.refreshes_committed:
            return
        rng = self.dynamics.rng
        old_state = copy.deepcopy(rng.bit_generator.state)
        old_deadline = getattr(self.schedule, "next_collision_step", None)
        try:
            rng.bit_generator.state = copy.deepcopy(self.rng_state)
            if self.schedule_state is not None:
                self.schedule.next_collision_step = int(self.schedule_state)
            for index in range(count):
                self.dynamics.sample_momenta()
                self.schedule.refreshed(self.start_step + int(self.offsets[index]), rng)
        except BaseException:
            rng.bit_generator.state = old_state
            if old_deadline is not None:
                self.schedule.next_collision_step = old_deadline
            raise
        self.refreshes_committed = count
        self.committed = True


def plan_refresh(dynamics, schedule, start_step, max_steps, *, budget=MAX_PREDRAWN_BYTES,
                 projection="none"):
    """Pre-draw an Andersen schedule without changing caller-visible state.

    ``offsets`` are relative to ``start_step`` and denote the step *before*
    which the replacement momentum is installed. Sampling remains scalar and
    follows ``Dynamics.sample_momenta`` and ``AndersenSchedule.refreshed`` in
    their historical order, so the plan can be committed by replaying only the
    successful refresh prefix.
    """
    start_step = _integer(start_step, "start_step")
    max_steps = _integer(max_steps, "max_steps")
    budget = _integer(budget, "budget", 1)
    if projection not in ("none", "centroid_constraint", "fragments"):
        raise ValueError("refresh projection must be none, centroid_constraint or fragments")
    if getattr(dynamics, "bond_constraints", None):
        raise ValueError("NP1 momentum refresh does not support bond constraints")
    if not hasattr(schedule, "due") or not hasattr(schedule, "refreshed"):
        raise TypeError("schedule must provide due(step) and refreshed(step, rng)")
    if not hasattr(dynamics, "rng") or not hasattr(dynamics, "sample_momenta"):
        raise TypeError("dynamics must provide rng and sample_momenta()")

    rng = dynamics.rng
    original_rng = copy.deepcopy(rng.bit_generator.state)
    original_deadline = getattr(schedule, "next_collision_step", None)
    temporary_schedule = copy.deepcopy(schedule)
    offsets = []
    samples = []
    planned_steps = max_steps
    try:
        rng.bit_generator.state = copy.deepcopy(original_rng)
        for offset in range(max_steps):
            absolute = start_step + offset
            if not temporary_schedule.due(absolute):
                continue
            sample = np.asarray(dynamics.sample_momenta(), dtype=np.float64)
            expected = tuple(getattr(dynamics.integrator, "shape", sample.shape))
            if sample.shape != expected or sample.ndim != 3 or not np.isfinite(sample).all():
                raise ValueError("sample_momenta returned an invalid finite [P,N,3] array")
            per_refresh = int(sample.nbytes + np.dtype(np.int64).itemsize)
            capacity = budget // per_refresh
            if capacity < 1:
                raise ValueError("momentum refresh exceeds the pre-drawn memory budget")
            if len(samples) >= capacity:
                planned_steps = offset
                break
            offsets.append(offset)
            samples.append(sample.copy())
            temporary_schedule.refreshed(absolute, rng)
            if len(samples) == capacity and offset + 1 < max_steps:
                planned_steps = offset + 1
                break
    finally:
        rng.bit_generator.state = original_rng
        if original_deadline is not None:
            schedule.next_collision_step = original_deadline

    momenta = np.stack(samples, axis=0) if samples else np.empty(
        (0,) + tuple(getattr(dynamics.integrator, "shape", ())), dtype=np.float64)
    return RefreshPlan(dynamics, schedule, start_step, int(planned_steps),
                       np.asarray(offsets, dtype=np.int64), momenta,
                       original_rng, original_deadline, projection)


__all__ = ["MAX_PREDRAWN_BYTES", "RefreshPlan", "plan_refresh"]
