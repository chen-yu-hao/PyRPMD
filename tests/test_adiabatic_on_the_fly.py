"""Numerical, gauge and failure-atomicity checks for direct electronic data."""
import copy

import numpy as np
import pytest

from pyrpmd.adiabatic import (
    AdiabaticPythonPotential, AdiabaticResult, AdiabaticRPSHDynamics,
    ElectronicFrame, QDNRPMDDynamics, adiabatic_gradient_matrix,
    electronic_transport, validate_adiabatic_result,
)
from pyrpmd.constants import ANGSTROM_TO_AU, EV_TO_AU
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.potentials.base import PotentialError


class DiabaticModel:
    def evaluate(self, q):
        scalar = .04*np.sum(q*q, axis=(1, 2))
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0] = scalar+.12*q[:, 0, 0]
        matrix[:, 1, 1] = scalar-.12*q[:, 0, 0]
        matrix[:, 0, 1] = matrix[:, 1, 0] = .08
        gradient = np.zeros((len(q), 2, 2, q.shape[1], 3))
        gradient[:, 0, 0] = gradient[:, 1, 1] = .08*q
        gradient[:, 0, 0, 0, 0] += .12
        gradient[:, 1, 1, 0, 0] -= .12
        return DiabaticResult(matrix, gradient)


class QCCallback:
    """An exact mock QC provider with genuine geometry-to-geometry overlaps."""
    def __init__(self, *, flip=False, overlap=True, fail=False):
        self.flip, self.overlap, self.fail = flip, overlap, fail
        self.calls = []

    def evaluate(self, q, *, previous_q=None, previous_context=None):
        self.calls.append((q.copy(), None if previous_q is None else previous_q.copy()))
        if self.fail and previous_q is not None:
            raise RuntimeError("QC solver failed")
        r = DiabaticModel().evaluate(q)
        energy, u = np.linalg.eigh(r.matrix)
        if self.flip:
            # Geometry-dependent arbitrary eigenvector signs from a QC solver.
            u[:, :, 0] *= np.where(q[:, 0, 0] > 0, -1., 1.)[:, None]
        gradop = np.einsum("bsi,bstac,btj->bijac", u, r.gradients, u)
        gradient = np.einsum("biiac->biac", gradop)
        gap = energy[:, None, :]-energy[:, :, None]
        diagonal = np.arange(2)
        gap[:, diagonal, diagonal] = 1.
        nac = gradop/gap[..., None, None]
        nac[:, diagonal, diagonal] = 0.
        overlap = None
        if previous_context is not None and self.overlap:
            overlap = previous_context.swapaxes(1, 2)@u
        return AdiabaticResult(energy, gradient, nac, overlap, context=u)


def nuclear(beads=4):
    rng = np.random.default_rng(10)
    return State(rng.normal(scale=.2, size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))


def test_qd_nrpmd_matches_complete_diabatic_dynamics_in_changing_bases():
    provider = AdiabaticPythonPotential(QCCallback(flip=True))
    qd = QDNRPMDDynamics(provider, [2.], 8., 4, .025, 2, native=False)
    local = qd.initialize(nuclear(), 0, seed=11)
    u = local.electronic_frame.result.context
    z = np.einsum("bij,bj->bi", u, local.mapping_q+1j*local.mapping_p)
    fixed = NRPMDState(local.q, local.p, mapping_q=z.real, mapping_p=z.imag)
    diabatic = NRPMDDynamics(DiabaticModel(), [2.], 8., 4, .025, 2, native=False)
    assert qd.energy(local) == pytest.approx(diabatic.energy(fixed), abs=2e-14)
    for _ in range(60):
        qd.step(local)
        diabatic.step(fixed)
    u = local.electronic_frame.result.context
    z = np.einsum("bij,bj->bi", u, local.mapping_q+1j*local.mapping_p)
    np.testing.assert_allclose(local.q, fixed.q, atol=2e-13, rtol=0)
    np.testing.assert_allclose(local.p, fixed.p, atol=2e-13, rtol=0)
    np.testing.assert_allclose(z, fixed.mapping_q+1j*fixed.mapping_p, atol=2e-13, rtol=0)
    assert qd.energy(local) == pytest.approx(diabatic.energy(fixed), abs=2e-13)
    assert np.sum(qd.populations(local)) == pytest.approx(1., abs=2e-13)


def test_operator_gradients_recover_fixed_basis_positive_derivatives():
    state = nuclear(3)
    result = validate_adiabatic_result(QCCallback().evaluate(state.q), state.q.shape)
    u = result.context
    local = adiabatic_gradient_matrix(result)
    physical = np.einsum("bsi,bijac,btj->bstac", u, local, u)
    eps = 1e-6
    plus, minus = state.q.copy(), state.q.copy()
    plus[:, 0, 0] += eps
    minus[:, 0, 0] -= eps
    finite_difference = (DiabaticModel().evaluate(plus).matrix-DiabaticModel().evaluate(minus).matrix)/(2*eps)
    np.testing.assert_allclose(physical[..., 0, 0], finite_difference, atol=2e-11, rtol=0)


def test_qd_nrpmd_complex_eigenvector_gauge_preserves_physical_trajectory():
    def callback(q, *, previous_q=None, previous_context=None):
        r = QCCallback().evaluate(q)
        slopes = np.array([.3, -.2])
        phases = np.exp(1j*q[:, 0, 0, None]*slopes)
        u = r.context*phases[:, None, :]
        nac = r.couplings*phases.conj()[:, :, None, None, None]*phases[:, None, :, None, None]
        nac[:, np.arange(2), np.arange(2), 0, 0] = 1j*slopes
        overlap = None if previous_context is None else previous_context.conj().swapaxes(1, 2)@u
        return AdiabaticResult(r.energies, r.gradients, nac, overlap, context=u)
    qd = QDNRPMDDynamics(AdiabaticPythonPotential(callback), [2.], 8., 2, .025, 2, native=False)
    local = qd.initialize(nuclear(2), seed=4)
    u = local.electronic_frame.result.context
    z = np.einsum("bij,bj->bi", u, local.mapping_q+1j*local.mapping_p)
    fixed = NRPMDState(local.q, local.p, mapping_q=z.real, mapping_p=z.imag)
    diabatic = NRPMDDynamics(DiabaticModel(), [2.], 8., 2, .025, 2, native=False)
    for _ in range(20):
        qd.step(local)
        diabatic.step(fixed)
    np.testing.assert_allclose(local.q, fixed.q, atol=1e-13, rtol=0)
    np.testing.assert_allclose(local.p, fixed.p, atol=1e-13, rtol=0)
    u = local.electronic_frame.result.context
    z = np.einsum("bij,bj->bi", u, local.mapping_q+1j*local.mapping_p)
    np.testing.assert_allclose(z, fixed.mapping_q+1j*fixed.mapping_p, atol=1e-13, rtol=0)


def test_qc_adapter_converts_energy_gradient_nac_and_previous_geometry():
    received = []
    def callback(q, *, previous_q, previous_context):
        received.append((q.copy(), previous_q, previous_context))
        nac = np.zeros((len(q), 2, 2, 1, 3))
        nac[:, 0, 1, 0, 0], nac[:, 1, 0, 0, 0] = 2., -2.
        return AdiabaticResult(np.ones((len(q), 2)), np.ones((len(q), 2, 1, 3)), nac, context="qc-state")
    provider = AdiabaticPythonPotential(callback, units={"length": "angstrom", "energy": "eV"})
    q = np.ones((1, 1, 3))*ANGSTROM_TO_AU
    frame = ElectronicFrame(q, provider.evaluate(q))
    r = provider.evaluate(q*2, previous=frame)
    np.testing.assert_allclose(received[0][0], 1.)
    np.testing.assert_allclose(received[1][1], 1.)
    assert received[1][2] == "qc-state"
    np.testing.assert_allclose(r.energies, EV_TO_AU)
    np.testing.assert_allclose(r.gradients, EV_TO_AU/ANGSTROM_TO_AU)
    assert r.couplings[0, 0, 1, 0, 0] == pytest.approx(2/ANGSTROM_TO_AU)


@pytest.mark.parametrize("field,value,match", [
    ("energies", [[np.nan, 1.]], "energies"),
    ("gradients", np.zeros((1, 2, 3)), "gradients"),
    ("couplings", np.ones((1, 2, 2, 1, 3)), "anti-Hermitian"),
    ("status", [1], "failed"),
    ("status", [False], "integers"),
    ("overlaps", np.ones((2, 2)), "overlaps"),
])
def test_qc_contract_rejects_malformed_data(field, value, match):
    q = np.zeros((1, 1, 3))
    r = QCCallback().evaluate(q)
    data = vars(r).copy()
    data[field] = value
    with pytest.raises(PotentialError, match=match):
        validate_adiabatic_result(AdiabaticResult(**data), q.shape)


def test_result_detaches_provider_buffers_and_makes_frames_read_only():
    q = np.zeros((1, 1, 3))
    r = QCCallback().evaluate(q)
    frame = ElectronicFrame(q, r)
    r.energies[:] = 999
    q[:] = 4
    assert np.max(frame.result.energies) < 1
    assert np.max(frame.q) == 0
    with pytest.raises(ValueError):
        frame.result.energies[0, 0] = 0


def test_transport_requires_overlap_unless_continuous_gauge_is_explicit():
    provider = AdiabaticPythonPotential(QCCallback(overlap=False))
    qd = QDNRPMDDynamics(provider, [2.], 8., 1, .05, 2, native=False)
    s = qd.initialize(nuclear(1))
    before = s.copy()
    with pytest.raises(PotentialError, match="continuous-gauge"):
        qd.step(s)
    np.testing.assert_array_equal(s.q, before.q)
    np.testing.assert_array_equal(s.p, before.p)
    np.testing.assert_array_equal(s.mapping_q, before.mapping_q)
    assert s.step == 0
    provider.metadata["continuous_gauge"] = True
    other = QDNRPMDDynamics(provider, [2.], 8., 1, .05, 2, native=False)
    other.step(s)
    assert s.step == 1


def test_transport_rejects_lost_subspace_norm():
    q = np.zeros((1, 1, 3))
    old = ElectronicFrame(q, QCCallback().evaluate(q))
    data = vars(old.result).copy()
    data["overlaps"] = np.eye(2)[None]*.8
    new = ElectronicFrame(q, AdiabaticResult(**data))
    with pytest.raises(PotentialError, match="subspace"):
        electronic_transport(old, new)


def test_rpsh_is_invariant_to_qc_eigenvector_sign_flips():
    q = np.zeros((1, 1, 3))
    q[0, 0, 0] = -.01
    p = np.zeros_like(q)
    p[0, 0, 0] = 2.
    runs = []
    for flip in (False, True):
        d = AdiabaticRPSHDynamics(AdiabaticPythonPotential(QCCallback(flip=flip)), [2.], 8., 1,
                                  .05, 2, seed=15, native=False)
        s = d.initialize(State(q, p), 0, adiabatic_coefficients=np.array([.8, .6j]))
        d.step(s)
        physical_c = s.centroid_frame.result.context[0]@s.coefficients
        runs.append((s, physical_c, d.last_probabilities))
    np.testing.assert_allclose(runs[0][1], runs[1][1], atol=2e-14, rtol=0)
    np.testing.assert_allclose(runs[0][2], runs[1][2], atol=2e-14, rtol=0)
    np.testing.assert_allclose(runs[0][0].q, runs[1][0].q, atol=0, rtol=0)
    np.testing.assert_allclose(runs[0][0].p, runs[1][0].p, atol=2e-14, rtol=0)


def test_adiabatic_hop_conserves_ring_energy_and_internal_momentum():
    d = AdiabaticRPSHDynamics(AdiabaticPythonPotential(QCCallback()), [2.], 8., 4, .05, 2, native=False)
    state = nuclear(4)
    state.p[:, 0, 0] += 5.
    s = d.initialize(state)
    internal = s.p-s.p.mean(axis=0)
    before = d.energy(s)
    hop = d.attempt_hop(s, 1)
    assert hop.accepted
    assert hop.ring_energy_change == pytest.approx(0., abs=1e-13)
    assert d.energy(s) == pytest.approx(before, abs=1e-13)
    np.testing.assert_allclose(s.p-s.p.mean(axis=0), internal, atol=1e-14)


def test_adiabatic_frustrated_hop_reverses_only_nac_component():
    d = AdiabaticRPSHDynamics(AdiabaticPythonPotential(QCCallback()), [2.], 8., 1, .05, 2,
                              native=False, frustrated_hops="reverse")
    p = np.array([[[.01, .2, .3]]])
    s = d.initialize(State(np.zeros_like(p), p))
    before = d.energy(s)
    hop = d.attempt_hop(s, 1)
    assert hop.frustrated and not hop.accepted
    np.testing.assert_allclose(s.p, [[[-.01, .2, .3]]], atol=1e-15)
    assert d.energy(s) == pytest.approx(before)


@pytest.mark.parametrize("method", [QDNRPMDDynamics, AdiabaticRPSHDynamics])
def test_qc_failure_is_transactional(method):
    callback = QCCallback(fail=True)
    d = method(AdiabaticPythonPotential(callback), [2.], 8., 2, .05, 2, native=False)
    s = d.initialize(nuclear(2))
    before = s.copy()
    rng = copy.deepcopy(d.rng.bit_generator.state) if hasattr(d, "rng") else None
    with pytest.raises(RuntimeError, match="QC solver"):
        d.step(s)
    np.testing.assert_array_equal(s.q, before.q)
    np.testing.assert_array_equal(s.p, before.p)
    assert s.step == before.step and s.time == before.time
    if rng is not None:
        assert d.rng.bit_generator.state == rng


def test_stale_electronic_frame_rejected_before_propagation():
    d = QDNRPMDDynamics(AdiabaticPythonPotential(QCCallback()), [2.], 8., 1, .05, 2, native=False)
    s = d.initialize(nuclear(1))
    s.q[0, 0, 0] += .1
    with pytest.raises(ValueError, match="electronic frame"):
        d.step(s)
