"""Molecular, overlap and dynamics checks for the optional PySCF bridge.

The independent OpenMolcas values below are reproduced by PySCF's official
``pyscf/nac/test/test_sacasscf.py`` (LiH, 1.5 angstrom, STO-3G, CAS(2,2)).
Source: https://github.com/pyscf/pyscf/blob/master/pyscf/nac/test/test_sacasscf.py
Finite-difference tests use actual displaced-geometry wavefunctions, including
the doubly occupied Li core, rather than differentiating an analytical model.
"""

from dataclasses import replace
import copy
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("pyscf.nac.sacasscf")

from pyrpmd.adiabatic import AdiabaticPythonPotential, ElectronicFrame
from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.pyscf_multistate import (
    PySCFMultistatePotential, PySCFStatus, casscf_overlap,
)


def _bridge(**kwargs):
    settings = {"casscf_conv_tol": 1e-12, "casscf_conv_tol_grad": 1e-8,
                "response_atol": 1e-12, "response_rtol": 1e-10, "cache_size": 32}
    settings.update(kwargs)
    return PySCFMultistatePotential(["Li", "H"], **settings)


@pytest.fixture(scope="module")
def molecule():
    q = np.zeros((1, 2, 3))
    q[0, 1, 0] = 1.5*ANGSTROM_TO_AU
    bridge = _bridge()
    result = bridge.evaluate(q)
    assert not np.any(result.status), result.messages
    yield bridge, q, result
    bridge.close()


def test_lih_energies_and_nac_match_independent_openmolcas_reference(molecule):
    _, _, result = molecule
    np.testing.assert_allclose(result.energies[0], [-7.85629118, -7.72175252],
                               atol=1e-7, rtol=0.)
    # A common electronic phase changes the sign of both nuclear components.
    reference = np.array([.183701729060390, -.0691462064586138])
    actual = result.couplings[0, 0, 1, :, 0]
    phase = np.sign(actual @ reference)
    np.testing.assert_allclose(phase*actual, reference, atol=1e-6, rtol=0.)
    np.testing.assert_allclose(result.couplings, -result.couplings.swapaxes(1, 2),
                               atol=0., rtol=0.)
    np.testing.assert_allclose(result.gradients.sum(axis=2), 0., atol=1e-10, rtol=0.)
    assert result.overlaps is None


@pytest.mark.parametrize("atom", [0, 1])
def test_lih_full_nac_and_gradients_from_centered_wavefunction_differences(molecule, atom):
    bridge, q, center = molecule
    epsilon = 2e-4
    plus, minus = q.copy(), q.copy()
    plus[0, atom, 0] += epsilon
    minus[0, atom, 0] -= epsilon
    forward = bridge.evaluate(plus, previous_q=q, previous_context=center.context)
    backward = bridge.evaluate(minus, previous_q=q, previous_context=center.context)
    assert not np.any(forward.status), forward.messages
    assert not np.any(backward.status), backward.messages
    derivative = (forward.overlaps-backward.overlaps)/(2*epsilon)
    np.testing.assert_allclose(derivative[0], center.couplings[0, :, :, atom, 0],
                               atol=3e-6, rtol=0.)
    gradient = (forward.energies-backward.energies)/(2*epsilon)
    np.testing.assert_allclose(gradient[0], center.gradients[0, :, atom, 0],
                               atol=5e-7, rtol=0.)
    np.testing.assert_allclose(
        casscf_overlap(center.context[0], forward.context[0]), forward.overlaps[0],
        atol=2e-14, rtol=0.,
    )


def test_exact_same_geometry_overlap_contains_inactive_core(molecule):
    bridge, q, first = molecule
    result = bridge.evaluate(q, previous_q=q, previous_context=first.context)
    assert not np.any(result.status), result.messages
    np.testing.assert_allclose(result.overlaps[0], np.eye(2), atol=2e-13, rtol=0.)
    np.testing.assert_allclose(casscf_overlap(first.context[0], first.context[0]),
                               np.eye(2), atol=2e-13, rtol=0.)


def test_core_active_overlap_matches_full_occupied_determinants(molecule):
    from pyscf import fci, gto
    bridge, q, initial = molecule
    displaced = q.copy()
    displaced[:, 0, 0] += .2
    displaced[:, 1, 0] += .4
    displaced[:, 1, 1] += .15
    current = bridge.evaluate(displaced)
    assert not np.any(current.status), current.messages
    old, new = initial.context[0], current.context[0]
    left, right = old.wavefunction, new.wavefunction
    assert left.ncore == 1
    orbital = left.mo.T @ gto.intor_cross("int1e_ovlp", left.mol, right.mol) @ right.mo

    # Independent Slater-determinant sum over the complete occupied space.
    # This deliberately does not use a core determinant or Schur complement.
    overlaps = []
    for electrons in left.nelecas:
        strings = fci.cistring.gen_occslst(range(left.ncas), electrons)
        occupied = [np.r_[np.arange(left.ncore), left.ncore+string] for string in strings]
        overlaps.append(np.array([[np.linalg.det(orbital[np.ix_(i, j)])
                                   for j in occupied] for i in occupied]))
    reference = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for a, b in np.ndindex(left.ci.shape[1:]):
                for c, d in np.ndindex(right.ci.shape[1:]):
                    reference[i, j] += (left.ci[i, a, b]*right.ci[j, c, d]
                                        *overlaps[0][a, c]*overlaps[1][b, d])
    reference = reference[np.ix_(old.order, new.order)]*old.phases[:, None]*new.phases[None]
    np.testing.assert_allclose(casscf_overlap(old, new), reference, atol=3e-14, rtol=0.)


def test_three_state_full_coupling_matrix_without_inactive_core():
    bridge = PySCFMultistatePotential(
        ["H", "H"], nstates=3, casscf_conv_tol=1e-12,
        casscf_conv_tol_grad=1e-8, response_atol=1e-12, response_rtol=1e-10,
    )
    try:
        q = np.array([[[0., 0., 0.], [2.4, 0., 0.]]])
        center = bridge.evaluate(q)
        assert not np.any(center.status), center.messages
        assert center.context[0].wavefunction.ncore == 0
        epsilon = 2e-4
        plus, minus = q.copy(), q.copy()
        plus[0, 1, 0] += epsilon
        minus[0, 1, 0] -= epsilon
        forward = bridge.evaluate(plus, previous_q=q, previous_context=center.context)
        backward = bridge.evaluate(minus, previous_q=q, previous_context=center.context)
        assert not np.any(forward.status), forward.messages
        assert not np.any(backward.status), backward.messages
        np.testing.assert_allclose((forward.overlaps-backward.overlaps)[0]/(2*epsilon),
                                   center.couplings[0, :, :, 1, 0], atol=1e-6, rtol=0.)
        np.testing.assert_allclose((forward.energies-backward.energies)[0]/(2*epsilon),
                                   center.gradients[0, :, 1, 0], atol=5e-7, rtol=0.)
    finally:
        bridge.close()


def test_previous_gauge_changes_are_transported_consistently(molecule):
    bridge, q, old = molecule
    qnew = q.copy()
    qnew[0, 1, 0] += 1e-3
    normal = bridge.evaluate(qnew, previous_q=q, previous_context=old.context)
    signs = np.array([-1., 1.])
    flipped = replace(old.context[0], phases=np.asarray(old.context[0].phases)*signs)
    result = bridge.evaluate(qnew, previous_q=q, previous_context=(flipped,))
    assert not np.any(result.status), result.messages
    np.testing.assert_array_equal(result.energies, normal.energies)
    np.testing.assert_array_equal(result.gradients, normal.gradients)
    np.testing.assert_allclose(result.couplings, normal.couplings*
                               signs[None, :, None, None, None]*signs[None, None, :, None, None],
                               atol=2e-14, rtol=0.)
    np.testing.assert_allclose(result.overlaps, normal.overlaps*
                               signs[None, :, None]*signs[None, None, :],
                               atol=2e-14, rtol=0.)
    assert np.all(np.diagonal(result.overlaps, axis1=1, axis2=2) > 0.)


def test_overlap_order_tracks_an_explicit_permutation(molecule):
    _, q, _ = molecule
    bridge = _bridge(state_order="overlap")
    try:
        old = bridge.evaluate(q)
        permutation = np.array([1, 0])
        context = old.context[0]
        reordered = replace(context, order=np.asarray(context.order)[permutation],
                            phases=np.asarray(context.phases)[permutation])
        result = bridge.evaluate(q, previous_q=q, previous_context=(reordered,))
        assert not np.any(result.status), result.messages
        np.testing.assert_allclose(result.energies, old.energies[:, permutation], atol=0., rtol=0.)
        np.testing.assert_allclose(result.overlaps, np.eye(2)[None], atol=2e-13, rtol=0.)
        np.testing.assert_allclose(result.gradients, old.gradients[:, permutation], atol=0., rtol=0.)
    finally:
        bridge.close()


def test_cached_solutions_do_not_confuse_interleaved_previous_frames(molecule):
    bridge, q, center = molecule
    other_q = q.copy()
    other_q[0, 1, 0] += .02
    other = bridge.evaluate(other_q)
    new_q = q.copy()
    new_q[0, 1, 0] += .01
    first = bridge.evaluate(new_q, previous_q=q, previous_context=center.context)
    interleaved = bridge.evaluate(new_q, previous_q=other_q, previous_context=other.context)
    repeat = bridge.evaluate(new_q, previous_q=q, previous_context=center.context)
    assert not np.any(interleaved.status), interleaved.messages
    np.testing.assert_array_equal(first.energies, repeat.energies)
    np.testing.assert_array_equal(first.couplings, repeat.couplings)
    np.testing.assert_array_equal(first.overlaps, repeat.overlaps)
    assert np.max(abs(first.overlaps-interleaved.overlaps)) > 1e-4
    np.testing.assert_allclose(interleaved.overlaps[0],
                               casscf_overlap(other.context[0], interleaved.context[0]),
                               atol=2e-14, rtol=0.)


def test_adapter_returns_detached_immutable_arrays(molecule):
    bridge, q, _ = molecule
    wrapped = AdiabaticPythonPotential(bridge)
    frame = ElectronicFrame(q, wrapped.evaluate(q))
    saved = frame.result.energies.copy()
    raw = bridge.evaluate(q)
    if raw.energies.flags.writeable:
        raw.energies[:] = 999.
    np.testing.assert_array_equal(frame.result.energies, saved)
    with pytest.raises(ValueError):
        frame.result.energies[0, 0] = 0.


def test_context_cannot_be_modified_through_arrays_or_molecule(molecule):
    _, _, result = molecule
    context = result.context[0]
    for array in (context.q, context.order, context.phases, context.wavefunction.mo,
                  context.wavefunction.ci, context.wavefunction.couplings):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    old_coordinates = context.wavefunction.mol.atom_coords()
    molecule_copy = context.wavefunction.mol
    molecule_copy.set_geom_(old_coordinates+1., unit="Bohr")
    np.testing.assert_array_equal(context.wavefunction.mol.atom_coords(), old_coordinates)


def test_context_json_roundtrip_restores_exact_electronic_history(molecule):
    bridge, q, result = molecule
    snapshot = json.loads(json.dumps(bridge.export_context(result.context), allow_nan=False))
    restored_provider = _bridge()
    try:
        restored = restored_provider.import_context(snapshot)
        for original, copy_context in zip(result.context, restored):
            for name in ("q", "order", "phases"):
                np.testing.assert_array_equal(getattr(original, name), getattr(copy_context, name))
            for name in ("mo", "ci", "energies", "gradients", "couplings"):
                np.testing.assert_array_equal(getattr(original.wavefunction, name),
                                              getattr(copy_context.wavefunction, name))
        new_q = q.copy()
        new_q[0, 1, 0] += .003
        continuous = bridge.evaluate(new_q, previous_q=q, previous_context=result.context)
        resumed = restored_provider.evaluate(new_q, previous_q=q, previous_context=restored)
        assert not np.any(resumed.status), resumed.messages
        for name in ("energies", "gradients", "couplings", "overlaps"):
            np.testing.assert_array_equal(getattr(continuous, name), getattr(resumed, name))
    finally:
        restored_provider.close()


def test_context_snapshot_checksum_and_identity_are_checked(molecule):
    bridge, _, result = molecule
    snapshot = bridge.export_context(result.context)
    snapshot["frames"][0]["mo"][0][0] += .001
    with pytest.raises(ValueError, match="checksum"):
        bridge.import_context(snapshot)
    snapshot = bridge.export_context(result.context)
    snapshot["scientific_identity"]["basis"] = "6-31g"
    with pytest.raises(ValueError, match="identity"):
        bridge.import_context(snapshot)


@pytest.mark.parametrize("field,value,match", [
    ("order", [0, 0], "permutation"),
    ("order", [0., 1.], "integer"),
    ("phases", [0., 1.], "signs"),
    ("energies", [-1., -1.], "degenerate"),
    ("mo", [[1.]], "MO shape"),
    ("ci", np.ones((2, 2, 2)).tolist(), "orthonormal"),
    ("couplings", np.ones((2, 2, 2, 3)).tolist(), "antisymmetric"),
])
def test_context_snapshot_structural_validation_is_independent_of_checksum(molecule, field, value, match):
    bridge, _, result = molecule
    snapshot = bridge.export_context(result.context)
    snapshot["frames"][0][field] = value
    payload = {key: value for key, value in snapshot.items() if key != "sha256"}
    snapshot["sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    with pytest.raises(ValueError, match=match):
        bridge.import_context(snapshot)


def test_fingerprint_detaches_configuration_and_includes_scientific_options():
    first, second = _bridge(), _bridge(gap_tolerance=2e-7)
    try:
        identity = first.to_dict()
        assert json.dumps(identity, sort_keys=True, allow_nan=False)
        assert identity != second.to_dict()
        original = copy.deepcopy(identity)
        identity.clear()
        assert first.to_dict() == original
    finally:
        first.close()
        second.close()


def test_resolved_basis_identity_and_calculations_are_frozen(monkeypatch, molecule):
    from pyscf import gto
    _, q, _ = molecule
    original = _bridge()
    identity = original.to_dict()
    load_basis = gto.basis.load

    def changed_basis(*args, **kwargs):
        basis = copy.deepcopy(load_basis(*args, **kwargs))
        basis[0][1][0] *= 1.001
        return basis

    monkeypatch.setattr(gto.basis, "load", changed_basis)
    changed = _bridge()
    try:
        assert original.to_dict() == identity
        assert changed.to_dict()["resolved_basis"] != identity["resolved_basis"]
        assert original._fingerprint != changed._fingerprint
        # Mole construction uses each provider's frozen numerical coefficients,
        # even when the external basis loader changes after construction.
        assert original._molecule(q[0])._basis == identity["resolved_basis"]
        assert changed._molecule(q[0])._basis == changed.to_dict()["resolved_basis"]
    finally:
        original.close()
        changed.close()


@pytest.mark.parametrize("settings,match", [
    ({"spin": 2}, "singlet"),
    ({"ncas": True}, "integer"),
    ({"nstates": 1}, "at least"),
    ({"nstates": 4}, "singlet"),
    ({"nelecas": 3}, "even"),
    ({"nelecas": (2, 1)}, "singlet"),
    ({"nelecas": (1, False)}, "integer"),
    ({"charge": 1}, "electron count"),
    ({"charge": False}, "integer"),
    ({"basis": {}}, "basis"),
    ({"state_order": "automatic"}, "state_order"),
    ({"casscf_conv_tol": np.nan}, "positive"),
    ({"response_rtol": 0.}, "positive"),
    ({"gap_tolerance": 1j}, "positive"),
    ({"phase_overlap_tolerance": 1.}, "below one"),
    ({"max_fci_cycles": 2.5}, "integer"),
    ({"max_memory_mb": -1}, "positive"),
    ({"cache_size": -1}, "at least"),
])
def test_unsupported_and_invalid_settings_fail_before_a_calculation(settings, match):
    with pytest.raises(ValueError, match=match):
        _bridge(**settings)


def test_duplicate_batch_geometries_share_cached_solve(molecule):
    _, q, _ = molecule
    bridge = _bridge(cache_size=1)
    try:
        result = bridge.evaluate(np.repeat(q, 2, axis=0))
        assert not np.any(result.status), result.messages
        info = bridge.cache_info()
        assert info["solves"] == info["misses"] == 1
        assert info["hits"] == 1
        assert info["size"] == info["maxsize"] == 1
        bridge.clear_cache()
        assert bridge.cache_info()["size"] == 0
        repeated = bridge.evaluate(q, previous_q=q, previous_context=(result.context[0],))
        assert not np.any(repeated.status), repeated.messages
        np.testing.assert_allclose(repeated.overlaps[0], np.eye(2), atol=2e-13, rtol=0.)
    finally:
        bridge.close()


def test_failed_batch_frame_has_status_and_preserves_diagnostics(molecule):
    bridge, q, _ = molecule
    batch = np.concatenate((q, np.zeros_like(q)))
    result = bridge.evaluate(batch)
    assert result.status[0] == PySCFStatus.SUCCESS
    assert result.status[1] != PySCFStatus.SUCCESS
    assert result.context[0] is not None and result.context[1] is None
    assert result.messages[0] == "" and result.messages[1]
    with pytest.raises(PotentialError, match="failed"):
        AdiabaticPythonPotential(bridge).evaluate(batch)


def test_near_degenerate_guard_stops_before_dividing_by_gap(molecule):
    _, q, _ = molecule
    bridge = _bridge(gap_tolerance=1.)
    try:
        result = bridge.evaluate(q)
        assert result.status[0] == PySCFStatus.DEGENERATE_STATES
        assert "gap_tolerance" in result.messages[0]
    finally:
        bridge.close()


def test_nonconverged_scf_has_status_and_adapter_raises(molecule):
    _, q, _ = molecule
    bridge = _bridge(max_scf_cycles=1)
    try:
        failed = bridge.evaluate(q)
        assert failed.status[0] != PySCFStatus.SUCCESS
        assert "SCF" in failed.messages[0]
        with pytest.raises(PotentialError, match="SCF"):
            AdiabaticPythonPotential(bridge).evaluate(q)
    finally:
        bridge.close()


def test_response_nonconvergence_is_not_reported_as_success(molecule, monkeypatch):
    from pyscf.nac.sacasscf import NonAdiabaticCouplings
    _, q, _ = molecule
    original = NonAdiabaticCouplings.kernel

    def unconverged(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.converged = False
        return result

    monkeypatch.setattr(NonAdiabaticCouplings, "kernel", unconverged)
    bridge = _bridge()
    try:
        result = bridge.evaluate(q)
        assert np.all(result.status != PySCFStatus.SUCCESS)
        assert "converg" in result.messages[0].lower()
    finally:
        bridge.close()


@pytest.mark.parametrize("kind", ["nac_scalar", "nac_missing_atom_axis", "complex_gradient", "nonfinite_nac"])
def test_malformed_backend_derivatives_fail_without_broadcast_or_precision_loss(molecule, monkeypatch, kind):
    from pyscf.grad.sacasscf import Gradients
    from pyscf.nac.sacasscf import NonAdiabaticCouplings
    _, q, _ = molecule
    method = Gradients if kind == "complex_gradient" else NonAdiabaticCouplings
    original = method.kernel

    def malformed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kind == "nac_scalar":
            return 0.
        if kind == "nac_missing_atom_axis":
            return np.zeros(3)
        if kind == "nonfinite_nac":
            return np.full_like(result, np.nan)
        return np.asarray(result, dtype=complex)+1e-9j

    monkeypatch.setattr(method, "kernel", malformed)
    bridge = _bridge()
    try:
        result = bridge.evaluate(q)
        assert result.status[0] == PySCFStatus.ELECTRONIC_STRUCTURE_ERROR
        assert result.messages[0]
        assert bridge.cache_info()["size"] == 0
        assert np.isnan(result.energies).all()
    finally:
        bridge.close()


def test_previous_geometry_mismatch_fails_explicitly(molecule):
    bridge, q, result = molecule
    wrong = q.copy()
    wrong[0, 1, 0] += .01
    with pytest.raises((ValueError, PotentialError), match="previous|context|geometry"):
        bridge.evaluate(q, previous_q=wrong, previous_context=result.context)


def test_previous_context_from_another_model_cannot_be_reused(molecule):
    _, q, result = molecule
    other = _bridge(gap_tolerance=2e-7)
    try:
        with pytest.raises(ValueError, match="identity"):
            other.evaluate(q, previous_q=q, previous_context=result.context)
    finally:
        other.close()


def test_small_molecular_rpsh_energy_drift_and_step_refinement():
    from examples.nonadiabatic.pyscf_lih_rpsh import run
    coarse = run(steps=6, dt=2., initial_state=1)
    fine = run(steps=12, dt=1., initial_state=1)
    assert coarse["max_energy_drift_hartree"] < 1e-6
    assert fine["max_energy_drift_hartree"] < .4*coarse["max_energy_drift_hartree"] + 1e-10
    assert coarse["max_overlap_singular_value_deviation"] < 1e-6
    np.testing.assert_allclose(np.asarray(fine["population_history"]).sum(axis=1),
                               1., atol=2e-12, rtol=0.)
    assert len(fine["physics_fingerprint"]) == 64


def test_actual_rpsh_continues_bitwise_from_serialized_electronic_history(molecule):
    from pyrpmd.adiabatic import AdiabaticResult, AdiabaticRPSHDynamics, AdiabaticRPSHState
    from pyrpmd.constants import AMU_TO_AU, KB_AU
    from pyrpmd.dynamics import State

    _, q, _ = molecule
    original_bridge, restored_bridge = _bridge(), _bridge()
    settings = {"masses": np.array([7.0160034366, 1.00782503223])*AMU_TO_AU,
                "beta": 1./(KB_AU*300.), "nbeads": 1, "dt": 1., "nstates": 2,
                "seed": 13, "native": False}
    original = AdiabaticRPSHDynamics(AdiabaticPythonPotential(original_bridge), **settings)
    restored = AdiabaticRPSHDynamics(AdiabaticPythonPotential(restored_bridge), **settings)
    p = np.zeros_like(q)
    p[0, 0, 0], p[0, 1, 0] = -.5, .5
    try:
        state = original.initialize(State(q, p), active_surface=1,
                                    adiabatic_coefficients=np.array([.6, .8], dtype=complex))
        initial_rng = copy.deepcopy(original.rng.bit_generator.state)
        for _ in range(2):
            original.step(state)
        assert original.rng.bit_generator.state != initial_rng
        data = state.bead_frame.result
        snapshot = json.loads(json.dumps({
            "q": state.q.tolist(), "p": state.p.tolist(), "step": state.step,
            "time": state.time, "coefficients_real": state.coefficients.real.tolist(),
            "coefficients_imag": state.coefficients.imag.tolist(),
            "active_surface": state.active_surface,
            "energies": data.energies.tolist(), "gradients": data.gradients.tolist(),
            "couplings": data.couplings.tolist(), "overlaps": data.overlaps.tolist(),
            "context": original_bridge.export_context(data.context),
            "rng": original.rng.bit_generator.state,
        }, allow_nan=False))
        qcopy = np.asarray(snapshot["q"])
        restored_frame = ElectronicFrame(qcopy, AdiabaticResult(
            np.asarray(snapshot["energies"]), np.asarray(snapshot["gradients"]),
            np.asarray(snapshot["couplings"]), np.asarray(snapshot["overlaps"]),
            context=restored_bridge.import_context(snapshot["context"]),
        ))
        resumed = AdiabaticRPSHState(
            qcopy, np.asarray(snapshot["p"]), snapshot["step"], snapshot["time"],
            np.asarray(snapshot["coefficients_real"])+1j*np.asarray(snapshot["coefficients_imag"]),
            snapshot["active_surface"], restored_frame, restored_frame,
        )
        restored.rng.bit_generator.state = snapshot["rng"]
        for _ in range(2):
            original.step(state)
            restored.step(resumed)
            for field in ("q", "p", "coefficients"):
                np.testing.assert_array_equal(getattr(state, field), getattr(resumed, field))
            assert (state.step, state.time, state.active_surface) == (
                resumed.step, resumed.time, resumed.active_surface,
            )
            assert original.rng.bit_generator.state == restored.rng.bit_generator.state
            assert original.last_probabilities == restored.last_probabilities
            assert original.energy(state) == restored.energy(resumed)
    finally:
        original_bridge.close()
        restored_bridge.close()
