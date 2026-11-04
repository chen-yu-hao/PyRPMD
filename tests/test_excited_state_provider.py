from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.potentials.excited_state import (
    AvoidedCrossingBackend,
    BackendFailure,
    BackendResult,
    ExcitedStateBackend,
    ExcitedStateProvider,
    ExcitedStateStatus,
    StateTrackingError,
    ToyWavefunction,
    align_states,
    integrated_connection,
    path_derivative_couplings,
    time_derivative_coupling,
)


def _q(x, *, beads=1):
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = x if np.isscalar(x) else np.asarray(x)
    return q


def test_npi_and_hst_match_independent_two_state_rotation_reference():
    angle = 0.31
    overlap = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    reference = np.array([[0.0, -angle], [angle, 0.0]])
    np.testing.assert_allclose(integrated_connection(overlap, method="npi"), reference, atol=2e-15)
    np.testing.assert_allclose(
        integrated_connection(overlap, method="hst"),
        np.array([[0.0, -np.sin(angle)], [np.sin(angle), 0.0]]),
        atol=2e-15,
    )
    np.testing.assert_allclose(time_derivative_coupling(overlap, 0.5), reference / 0.5)


def test_path_coupling_has_the_requested_overlap_line_integral():
    angle = 0.08
    overlap = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    displacement = np.array([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
    coupling = path_derivative_couplings(overlap, displacement)
    contraction = np.einsum("ijac,ac->ij", coupling, displacement)
    np.testing.assert_allclose(contraction, integrated_connection(overlap), atol=2e-15)
    np.testing.assert_allclose(coupling, -coupling.swapaxes(0, 1), atol=2e-15)


def test_state_assignment_corrects_permutation_and_phase_and_reports_degeneracy():
    overlap = np.array([[0.01, -0.999], [0.998, 0.02]])
    energies = np.array([1.0000002, 1.0])
    gradients = np.arange(12.0).reshape(2, 2, 3)
    coupling = np.zeros((2, 2, 2, 3))
    coupling[0, 1, 0, 0] = 2.0
    coupling[1, 0] = -coupling[0, 1]
    tracked = align_states(
        overlap,
        energies,
        gradients,
        derivative_couplings=coupling,
        degeneracy_tolerance=1e-6,
        subspace_tolerance=0.05,
    )
    np.testing.assert_array_equal(tracked["permutation"], [1, 0])
    assert np.all(np.diag(tracked["overlap"]) > 0)
    np.testing.assert_array_equal(tracked["energies"], energies[[1, 0]])
    np.testing.assert_array_equal(tracked["gradients"], gradients[[1, 0]])
    assert tracked["diagnostics"].near_degenerate_pairs == ((0, 1),)
    np.testing.assert_allclose(
        tracked["derivative_couplings"],
        -tracked["derivative_couplings"].swapaxes(0, 1),
    )


def test_transition_dipoles_follow_relative_ground_to_state_phase():
    physical = np.array([[0.2, 0.0, 0.0], [0.7, -0.1, 0.3]])
    raw_phases = np.array([-1.0, 1.0])
    raw = physical * (np.conjugate(raw_phases[0]) * raw_phases)[:, None]
    tracked = align_states(
        np.diag(raw_phases),
        np.array([0.0, 0.5]),
        np.zeros((2, 1, 3)),
        transition_dipoles=raw,
        subspace_tolerance=1e-8,
    )
    np.testing.assert_allclose(tracked["transition_dipoles"], physical, atol=0.0, rtol=0.0)


def test_singular_retained_subspace_is_rejected_instead_of_projected():
    with pytest.raises(StateTrackingError, match="retained-subspace"):
        integrated_connection(np.diag([1.0, 0.1]), subspace_tolerance=0.2)


def test_toy_energies_gradients_and_nac_match_centered_finite_differences():
    backend = AvoidedCrossingBackend(1, slope=0.03, coupling=0.007, harmonic=0.002)
    q = np.array([[0.17, -0.04, 0.03]])
    center = backend.calculate(q, Path("."), None)
    epsilon = 1e-6
    for atom, axis in np.ndindex(1, 3):
        plus, minus = q.copy(), q.copy()
        plus[atom, axis] += epsilon
        minus[atom, axis] -= epsilon
        forward = backend.calculate(plus, Path("."), center.payload)
        backward = backend.calculate(minus, Path("."), center.payload)
        energy_derivative = (forward.energies - backward.energies) / (2 * epsilon)
        overlap_derivative = (forward.overlap - backward.overlap) / (2 * epsilon)
        np.testing.assert_allclose(
            energy_derivative, center.gradients[:, atom, axis], atol=2e-10, rtol=2e-9
        )
        np.testing.assert_allclose(
            overlap_derivative, center.derivative_couplings[:, :, atom, axis], atol=2e-9, rtol=2e-8
        )


class _PermutingToy(AvoidedCrossingBackend):
    name = "permuting-toy"

    def calculate(self, geometry, scratch, previous_payload):
        raw = super().calculate(geometry, scratch, None)
        x = float(geometry[0, 0])
        permutation = np.array([1, 0]) if x >= 0 else np.array([0, 1])
        phase = np.array([-1.0, 1.0]) if x >= 0 else np.ones(2)
        vectors = raw.payload.vectors[:, permutation] * phase[None, :]
        payload = ToyWavefunction(geometry, vectors)
        coupling = raw.derivative_couplings[permutation][:, permutation]
        coupling = coupling * phase[:, None, None, None] * phase[None, :, None, None]
        overlap = None if previous_payload is None else previous_payload.vectors.T @ vectors
        return BackendResult(
            raw.energies[permutation],
            raw.gradients[permutation],
            payload,
            coupling,
            overlap,
            raw.oscillator_strengths[permutation],
            None,
            raw.metadata,
        )

    def scientific_identity(self):
        return {**super().scientific_identity(), "permuting_test_gauge": True}


def test_provider_tracks_backend_state_swap_and_phase_flip(tmp_path):
    provider = ExcitedStateProvider(
        _PermutingToy(1), ["H"], 2, scratch_root=tmp_path, coupling_method="npi"
    )
    first_q = _q(-0.02)
    second_q = _q(0.02)
    first = provider.evaluate(first_q)
    second = provider.evaluate(
        second_q, previous_q=first_q, previous_context=first.context
    )
    reference = AvoidedCrossingBackend(1).calculate(second_q[0], Path("."), None)
    np.testing.assert_allclose(second.energies[0], reference.energies, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(second.gradients[0], reference.gradients, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(second.couplings[0], reference.derivative_couplings, atol=2e-15)
    assert np.all(np.diag(second.overlaps[0]) > 0)
    provider.close()


def test_near_degeneracy_is_a_success_diagnostic_not_a_false_convergence_claim(tmp_path):
    provider = ExcitedStateProvider(
        AvoidedCrossingBackend(1, coupling=1e-8),
        ["H"],
        2,
        degeneracy_tolerance=1e-6,
        scratch_root=tmp_path,
    )
    q0, q1 = _q(0.0), _q(1e-5)
    first = provider.evaluate(q0)
    second = provider.evaluate(q1, previous_q=q0, previous_context=first.context)
    assert second.status.tolist() == [0]
    assert "near-degenerate" in second.messages[0]
    assert second.context[0].diagnostics["tracking"]["near_degenerate_pairs"] == [[0, 1]]
    provider.close()


class _RecordingToy(AvoidedCrossingBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = []

    def calculate(self, geometry, scratch, previous_payload):
        self.paths.append(scratch)
        (scratch / "same-name.tmp").write_text(str(geometry[0, 0]))
        return super().calculate(geometry, scratch, previous_payload)


def test_each_bead_gets_a_unique_scratch_and_success_cleanup(tmp_path):
    backend = _RecordingToy(1)
    provider = ExcitedStateProvider(backend, ["H"], 2, scratch_root=tmp_path)
    provider.evaluate(_q([0.0, 0.1, 0.2], beads=3))
    assert len(set(map(str, backend.paths))) == 3
    assert all(path.name.startswith("bead-") for path in backend.paths)
    assert all(not path.exists() for path in backend.paths)
    provider.close()


class _FailingBackend(ExcitedStateBackend):
    name = "failing"
    program_version = "1"

    def calculate(self, geometry, scratch, previous_payload):
        raise BackendFailure(ExcitedStateStatus.SCF_NOT_CONVERGED, "deliberate SCF failure")

    def scientific_identity(self):
        return {"backend": self.name, "program_version": self.program_version}


def test_backend_failure_maps_to_status_without_half_valid_numbers(tmp_path):
    provider = ExcitedStateProvider(
        _FailingBackend(), ["H"], 2, require_overlap=False, require_couplings=False,
        scratch_root=tmp_path,
    )
    result = provider.evaluate(_q(0.0))
    assert result.status.tolist() == [ExcitedStateStatus.SCF_NOT_CONVERGED]
    assert np.isnan(result.energies).all()
    assert np.isnan(result.gradients).all()
    assert "deliberate SCF failure" in result.messages[0]
    stats = provider.statistics()
    assert stats.failure_count == 1
    assert stats.status_counts == {"SCF_NOT_CONVERGED": 1}
    provider.close()


def test_context_roundtrip_reproduces_next_frame_and_checks_identity(tmp_path):
    backend = AvoidedCrossingBackend(1)
    provider = ExcitedStateProvider(backend, ["H"], 2, scratch_root=tmp_path / "one")
    q0, q1 = _q(-0.1), _q(-0.08)
    first = provider.evaluate(q0)
    snapshot = provider.export_context(first.context)
    restored_provider = ExcitedStateProvider(
        AvoidedCrossingBackend(1), ["H"], 2, scratch_root=tmp_path / "two"
    )
    restored = restored_provider.import_context(snapshot)
    continuous = provider.evaluate(q1, previous_q=q0, previous_context=first.context)
    resumed = restored_provider.evaluate(q1, previous_q=q0, previous_context=restored)
    for field in ("energies", "gradients", "couplings", "overlaps"):
        np.testing.assert_array_equal(getattr(continuous, field), getattr(resumed, field))
    changed = ExcitedStateProvider(
        AvoidedCrossingBackend(1, slope=0.03), ["H"], 2, scratch_root=tmp_path / "three"
    )
    with pytest.raises(ValueError, match="identity"):
        changed.import_context(snapshot)
    provider.close()
    restored_provider.close()
    changed.close()


def test_run_identity_changes_for_scientific_options_not_scratch_location(tmp_path):
    first = ExcitedStateProvider(
        AvoidedCrossingBackend(1), ["H"], 2, scratch_root=tmp_path / "a"
    )
    second = ExcitedStateProvider(
        AvoidedCrossingBackend(1), ["H"], 2, scratch_root=tmp_path / "b"
    )
    changed = ExcitedStateProvider(
        AvoidedCrossingBackend(1), ["H"], 2, coupling_method="hst", scratch_root=tmp_path / "c"
    )
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert changed.to_dict() != first.to_dict()
    assert changed.fingerprint != first.fingerprint
    first.close(); second.close(); changed.close()


def test_scratch_creation_failure_maps_to_all_frame_statuses(tmp_path, monkeypatch):
    provider = ExcitedStateProvider(AvoidedCrossingBackend(1), ["H"], 2, scratch_root=tmp_path)

    def fail_create(call_id, nbeads):
        raise BackendFailure(ExcitedStateStatus.SCRATCH_ERROR, "deliberate scratch failure")

    monkeypatch.setattr(provider._scratch, "create_call", fail_create)
    result = provider.evaluate(_q([0.0, 0.1], beads=2))
    assert result.status.tolist() == [ExcitedStateStatus.SCRATCH_ERROR] * 2
    assert np.isnan(result.energies).all()
    assert result.context == (None, None)
    assert provider.statistics().failure_count == 2
    provider.close()


def test_cleanup_failure_invalidates_successful_frames(tmp_path, monkeypatch):
    provider = ExcitedStateProvider(AvoidedCrossingBackend(1), ["H"], 2, scratch_root=tmp_path)

    def fail_cleanup(call, *, success):
        raise OSError("deliberate cleanup failure")

    monkeypatch.setattr(provider._scratch, "finish_call", fail_cleanup)
    result = provider.evaluate(_q(0.0))
    assert result.status.tolist() == [ExcitedStateStatus.SCRATCH_ERROR]
    assert np.isnan(result.energies).all()
    assert result.context == (None,)
    provider.close()


def test_unexpected_backend_exception_is_explicitly_diagnosed(tmp_path):
    class Broken(ExcitedStateBackend):
        name = "broken"
        program_version = "1"

        def calculate(self, geometry, scratch, previous_payload):
            raise RuntimeError("bad solver state")

        def scientific_identity(self):
            return {"backend": self.name, "program_version": self.program_version}

    provider = ExcitedStateProvider(
        Broken(), ["H"], 2, require_overlap=False, require_couplings=False,
        scratch_root=tmp_path,
    )
    result = provider.evaluate(_q(0.0))
    assert result.status.tolist() == [ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR]
    assert "RuntimeError: bad solver state" in result.messages[0]
    provider.close()


def test_success_cleanup_policy_preserves_failed_scratch_after_close(tmp_path):
    class ScratchFailing(_FailingBackend):
        def calculate(self, geometry, scratch, previous_payload):
            (scratch / "failure.txt").write_text("diagnostic", encoding="utf-8")
            return super().calculate(geometry, scratch, previous_payload)

    provider = ExcitedStateProvider(
        ScratchFailing(),
        ["H"],
        2,
        require_overlap=False,
        require_couplings=False,
        scratch_root=tmp_path,
        scratch_cleanup="success",
    )
    result = provider.evaluate(_q(0.0))
    assert result.status.tolist() == [ExcitedStateStatus.SCF_NOT_CONVERGED]
    provider.close()
    retained = list(tmp_path.rglob("failure.txt"))
    assert len(retained) == 1
    assert retained[0].read_text(encoding="utf-8") == "diagnostic"
