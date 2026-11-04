"""Generic online excited-state provider framework."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from ._provider_config import _ProviderConfigurationMixin
from ._provider_persistence import _ProviderPersistenceMixin
from ._types import (
    BackendFailure, ExcitedStateFrameContext, ExcitedStateStatus,
)
from ._validation import _validate_backend_result
from .tracking import (
    StateTrackingError, align_states, path_derivative_couplings,
    time_derivative_coupling,
)


class ExcitedStateProvider(_ProviderConfigurationMixin, _ProviderPersistenceMixin):
    """Batch callback satisfying :class:`pyrpmd.adiabatic.AdiabaticResult`.

    A backend calculation is independent for every bead. Previous electronic
    state is accepted only through ``previous_context``; no hidden global last
    frame is used, so interleaved trajectories do not contaminate one another.
    """

    def evaluate(self, q, *, previous_q=None, previous_context=None):
        # Import lazily so the backend package can be inspected without loading
        # the full dynamics stack.
        from ...adiabatic import AdiabaticResult
        from ..base import coordinates

        coordinates_batch = coordinates(q)
        if coordinates_batch.shape[1] != len(self.symbols):
            raise ValueError(
                f"coordinates contain {coordinates_batch.shape[1]} atoms; expected {len(self.symbols)}"
            )
        previous_q, previous = self._validate_previous(
            coordinates_batch, previous_q, previous_context
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("excited-state provider is closed")
            call_id = self._calls
            self._calls += 1

        nbeads = len(coordinates_batch)
        energies = np.full((nbeads, self.nstates), np.nan)
        gradients = np.full(
            (nbeads, self.nstates, len(self.symbols), 3), np.nan
        )
        coupling_values: list[np.ndarray | None] = [None] * nbeads
        overlap_values: list[np.ndarray | None] = [None] * nbeads
        contexts: list[ExcitedStateFrameContext | None] = [None] * nbeads
        statuses = np.zeros(nbeads, dtype=np.int32)
        messages = [""] * nbeads
        backend_seconds = 0.0
        wall_start = time.perf_counter()
        call_path: Path | None = None
        bead_paths: tuple[Path, ...] = ()

        try:
            try:
                call_path, bead_paths = self._scratch.create_call(call_id, nbeads)
            except BackendFailure as exc:
                statuses.fill(int(exc.status))
                messages[:] = [str(exc)] * nbeads

            for bead, scratch in enumerate(bead_paths):
                old = None if previous is None else previous[bead]
                try:
                    started = time.perf_counter()
                    try:
                        result = self.backend.calculate(
                            coordinates_batch[bead],
                            scratch,
                            None if old is None else old.payload,
                        )
                    finally:
                        backend_seconds += time.perf_counter() - started
                    result = _validate_backend_result(
                        result, nstates=self.nstates, natoms=len(self.symbols)
                    )

                    if old is None:
                        current_energy = result.energies
                        current_gradient = result.gradients
                        current_coupling = result.derivative_couplings
                        current_strengths = result.oscillator_strengths
                        current_dipoles = result.transition_dipoles
                        permutation = np.arange(self.nstates, dtype=np.int64)
                        phases = np.ones(self.nstates)
                        tracked_overlap = None
                        tracking_diagnostics = None
                        if current_coupling is None and self.coupling_method != "analytic":
                            started = time.perf_counter()
                            try:
                                current_coupling = self.backend.initial_derivative_couplings(
                                    coordinates_batch[bead], result.payload, scratch
                                )
                            finally:
                                backend_seconds += time.perf_counter() - started
                            if current_coupling is not None:
                                current_coupling = np.asarray(current_coupling)
                    else:
                        started = time.perf_counter()
                        try:
                            raw_overlap = self._raw_overlap(result, old, scratch)
                        finally:
                            backend_seconds += time.perf_counter() - started
                        if raw_overlap is None:
                            if self.require_overlap or result.derivative_couplings is None:
                                raise BackendFailure(
                                    ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                                    "backend supplied neither an adjacent-frame overlap nor an overlap helper",
                                )
                            # Analytic NAC in a declared backend gauge can still
                            # be reported, but there is deliberately no fabricated
                            # overlap or phase tracking.
                            current_energy = result.energies
                            current_gradient = result.gradients
                            current_coupling = result.derivative_couplings
                            current_strengths = result.oscillator_strengths
                            current_dipoles = result.transition_dipoles
                            permutation = np.arange(self.nstates, dtype=np.int64)
                            phases = np.ones(self.nstates)
                            tracked_overlap = None
                            tracking_diagnostics = None
                        else:
                            try:
                                tracked = align_states(
                                    raw_overlap,
                                    result.energies,
                                    result.gradients,
                                    derivative_couplings=result.derivative_couplings,
                                    oscillator_strengths=result.oscillator_strengths,
                                    transition_dipoles=result.transition_dipoles,
                                    phase_overlap_tolerance=self.phase_overlap_tolerance,
                                    subspace_tolerance=self.subspace_tolerance,
                                    degeneracy_tolerance=self.degeneracy_tolerance,
                                    ambiguity_tolerance=self.ambiguity_tolerance,
                                )
                            except (StateTrackingError, ValueError) as exc:
                                raise BackendFailure(
                                    ExcitedStateStatus.STATE_TRACKING_FAILED, str(exc)
                                ) from exc
                            current_energy = tracked["energies"]
                            current_gradient = tracked["gradients"]
                            current_coupling = tracked["derivative_couplings"]
                            current_strengths = tracked["oscillator_strengths"]
                            current_dipoles = tracked["transition_dipoles"]
                            permutation = tracked["permutation"]
                            phases = tracked["phases"]
                            tracked_overlap = tracked["overlap"]
                            tracking_diagnostics = tracked["diagnostics"]
                            if current_coupling is None and self.coupling_method in ("hst", "npi"):
                                try:
                                    current_coupling = path_derivative_couplings(
                                        tracked_overlap,
                                        coordinates_batch[bead] - previous_q[bead],
                                        method=self.coupling_method,
                                        subspace_tolerance=self.subspace_tolerance,
                                    )
                                except (StateTrackingError, ValueError) as exc:
                                    raise BackendFailure(
                                        ExcitedStateStatus.COUPLING_UNAVAILABLE, str(exc)
                                    ) from exc

                    if current_coupling is not None:
                        current_coupling = np.asarray(current_coupling)
                        expected = (self.nstates, self.nstates, len(self.symbols), 3)
                        if (
                            current_coupling.shape != expected
                            or current_coupling.dtype.kind not in "fciu"
                            or not np.isfinite(current_coupling).all()
                            or not np.allclose(
                                current_coupling,
                                -current_coupling.swapaxes(0, 1).conj(),
                                rtol=0.0,
                                atol=1e-9,
                            )
                        ):
                            raise BackendFailure(
                                ExcitedStateStatus.COUPLING_UNAVAILABLE,
                                f"reported derivative couplings must be finite anti-Hermitian {expected}",
                            )
                    elif self.require_couplings:
                        raise BackendFailure(
                            ExcitedStateStatus.COUPLING_UNAVAILABLE,
                            "no analytic NAC or overlap-derived path coupling is available for this frame",
                        )

                    tdc = None
                    if tracked_overlap is not None and self.time_step is not None:
                        try:
                            tdc = time_derivative_coupling(
                                tracked_overlap,
                                self.time_step,
                                method=(
                                    "npi"
                                    if self.coupling_method == "analytic"
                                    else self.coupling_method
                                ),
                                subspace_tolerance=self.subspace_tolerance,
                            )
                        except (StateTrackingError, ValueError) as exc:
                            raise BackendFailure(
                                ExcitedStateStatus.COUPLING_UNAVAILABLE, str(exc)
                            ) from exc

                    energies[bead] = current_energy
                    gradients[bead] = current_gradient
                    coupling_values[bead] = current_coupling
                    overlap_values[bead] = tracked_overlap
                    diagnostic_data: dict[str, Any] = {
                        "backend": copy.deepcopy(dict(result.metadata)),
                        "coupling_source": (
                            "analytic"
                            if result.derivative_couplings is not None
                            else "overlap_path"
                            if current_coupling is not None and old is not None
                            else "backend_initial"
                            if current_coupling is not None
                            else "none"
                        ),
                    }
                    if tracking_diagnostics is not None:
                        diagnostic_data["tracking"] = {
                            "minimum_assigned_overlap": tracking_diagnostics.minimum_assigned_overlap,
                            "maximum_subspace_defect": tracking_diagnostics.maximum_subspace_defect,
                            "minimum_energy_gap": tracking_diagnostics.minimum_energy_gap,
                            "near_degenerate_pairs": [
                                list(pair)
                                for pair in tracking_diagnostics.near_degenerate_pairs
                            ],
                            "ambiguous_pairs": [
                                list(pair) for pair in tracking_diagnostics.ambiguous_pairs
                            ],
                        }
                        messages[bead] = self._diagnostic_message(tracking_diagnostics)
                    contexts[bead] = ExcitedStateFrameContext(
                        self.fingerprint,
                        coordinates_batch[bead],
                        result.payload,
                        permutation,
                        phases,
                        current_strengths,
                        current_dipoles,
                        tdc,
                        diagnostic_data,
                    )
                except BackendFailure as exc:
                    statuses[bead] = int(exc.status)
                    messages[bead] = str(exc)
                except StateTrackingError as exc:
                    statuses[bead] = int(ExcitedStateStatus.STATE_TRACKING_FAILED)
                    messages[bead] = str(exc)
                except Exception as exc:
                    statuses[bead] = int(ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR)
                    messages[bead] = (
                        f"{self.backend.name} raised {type(exc).__name__}: {exc}"
                    )
        finally:
            cleanup_error: OSError | None = None
            if call_path is not None:
                try:
                    self._scratch.finish_call(
                        call_path, success=bool(np.all(statuses == 0))
                    )
                except OSError as exc:
                    cleanup_error = exc
            if cleanup_error is not None:
                successful = statuses == 0
                statuses[successful] = int(ExcitedStateStatus.SCRATCH_ERROR)
                for bead in np.flatnonzero(successful):
                    messages[bead] = (
                        f"cannot clean excited-state scratch {call_path}: {cleanup_error}"
                    )
                    energies[bead] = np.nan
                    gradients[bead] = np.nan
                    coupling_values[bead] = None
                    overlap_values[bead] = None
                    contexts[bead] = None
            wall = time.perf_counter() - wall_start
            with self._lock:
                self._frames += nbeads
                failures = int(np.count_nonzero(statuses))
                self._failures += failures
                self._wall_seconds += wall
                self._backend_seconds += backend_seconds
                for status in statuses:
                    self._status_counts[ExcitedStateStatus(int(status)).name] += 1

        couplings = None
        if all(value is not None for value in coupling_values):
            couplings = np.stack(coupling_values)
        overlaps = None
        if all(value is not None for value in overlap_values):
            overlaps = np.stack(overlap_values)
        return AdiabaticResult(
            energies,
            gradients,
            couplings,
            overlaps,
            statuses,
            tuple(contexts),
            tuple(messages),
        )
