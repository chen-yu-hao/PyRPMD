from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyrpmd.potentials.excited_state import (
    ExcitedStateProvider,
    ExcitedStateStatus,
    PySCFTDDFTBackend,
    auxiliary_wavefunction_overlap,
)
from pyrpmd.potentials.excited_state.pyscf_tddft import _align_displaced_overlap


@pytest.fixture(scope="module", autouse=True)
def _require_pyscf():
    pytest.importorskip("pyscf")



def _backend(**kwargs):
    settings = dict(
        symbols=["H", "H"], nstates=2, basis="sto-3g", xc=None, tda=True,
        scf_conv_tol=1e-12, td_conv_tol=1e-10, grid_level=0,
        transition_properties=True, initial_nac="unavailable",
    )
    settings.update(kwargs)
    return PySCFTDDFTBackend(**settings)


def test_pyscf_tda_energy_gradient_and_overlap_derivative_reference(tmp_path):
    backend = _backend()
    q = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    center = backend.calculate(q, tmp_path, None)
    assert center.energies.shape == (2,)
    assert center.gradients.shape == (2, 2, 3)
    assert center.oscillator_strengths.shape == (2,)
    assert center.transition_dipoles.shape == (2, 3)
    np.testing.assert_allclose(
        auxiliary_wavefunction_overlap(center.payload, center.payload),
        np.eye(2), atol=2e-8, rtol=0.0,
    )

    epsilon = 2e-4
    plus, minus = q.copy(), q.copy()
    plus[1, 2] += epsilon
    minus[1, 2] -= epsilon
    forward = backend.calculate(plus, tmp_path, center.payload)
    backward = backend.calculate(minus, tmp_path, center.payload)
    numerical_gradient = (forward.energies - backward.energies) / (2 * epsilon)
    np.testing.assert_allclose(
        numerical_gradient, center.gradients[:, 1, 2], atol=3e-5, rtol=0.0
    )
    forward_overlap = _align_displaced_overlap(forward.overlap, subspace_tolerance=0.1)
    backward_overlap = _align_displaced_overlap(backward.overlap, subspace_tolerance=0.1)
    derivative = (forward_overlap - backward_overlap) / (2 * epsilon)
    np.testing.assert_allclose(derivative, -derivative.T, atol=3e-5, rtol=0.0)

    finite_difference_nac = _backend(
        initial_nac="finite_difference", transition_properties=False
    ).initial_derivative_couplings(q, center.payload, tmp_path)
    np.testing.assert_allclose(
        finite_difference_nac[:, :, 1, 2], derivative, atol=3e-5, rtol=0.0
    )


def test_pyscf_provider_tracks_phase_continuously_along_lih_trajectory(tmp_path):
    backend = PySCFTDDFTBackend(
        symbols=["Li", "H"],
        nstates=2,
        basis="sto-3g",
        xc=None,
        tda=True,
        scf_conv_tol=1e-12,
        td_conv_tol=1e-10,
        grid_level=0,
        transition_properties=True,
        initial_nac="unavailable",
    )
    provider = ExcitedStateProvider(
        backend,
        ["Li", "H"],
        2,
        coupling_method="npi",
        require_couplings=False,
        scratch_root=tmp_path,
        subspace_tolerance=0.1,
    )
    geometries = []
    for distance in (3.0, 3.001, 3.002):
        q = np.zeros((1, 2, 3))
        q[0, 1, 0] = distance
        geometries.append(q)
    try:
        first = provider.evaluate(geometries[0])
        second = provider.evaluate(
            geometries[1], previous_q=geometries[0], previous_context=first.context
        )
        third = provider.evaluate(
            geometries[2], previous_q=geometries[1], previous_context=second.context
        )
        assert first.status.tolist() == [ExcitedStateStatus.SUCCESS]
        assert second.status.tolist() == [ExcitedStateStatus.SUCCESS]
        assert third.status.tolist() == [ExcitedStateStatus.SUCCESS]
        assert np.all(np.diag(second.overlaps[0]).real > 0.0)
        assert np.all(np.diag(third.overlaps[0]).real > 0.0)
        mu2 = second.context[0].transition_dipoles[1]
        mu3 = third.context[0].transition_dipoles[1]
        assert float(np.vdot(mu2, mu3).real) >= 0.0
        assert not second.context[0].diagnostics["tracking"]["ambiguous_pairs"]
        assert not third.context[0].diagnostics["tracking"]["ambiguous_pairs"]
    finally:
        provider.close()


def test_pyscf_tda_backend_runs_short_rpsh_trajectory(tmp_path):
    from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticRPSHDynamics
    from pyrpmd.dynamics import State

    backend = _backend(
        transition_properties=False,
        initial_nac="finite_difference",
        nac_finite_difference_step=2e-4,
    )
    provider = ExcitedStateProvider(
        backend,
        ["H", "H"],
        2,
        coupling_method="npi",
        scratch_root=tmp_path,
        subspace_tolerance=0.1,
    )
    potential = AdiabaticPythonPotential(provider)
    dynamics = AdiabaticRPSHDynamics(
        potential,
        masses=[1837.1526473652, 1837.1526473652],
        beta=1000.0,
        nbeads=1,
        dt=0.01,
        nstates=2,
        native=False,
        seed=23,
        electronic_substeps=8,
        overlap_tolerance=0.1,
    )
    q = np.array([[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]])
    p = np.array([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
    try:
        state = dynamics.initialize(State(q, p), active_surface=0)
        initial_energy = dynamics.energy(state)
        for _ in range(2):
            dynamics.step(state)
        drift = dynamics.energy(state) - initial_energy
        assert abs(drift) < 5e-5
        assert state.step == 2
        np.testing.assert_allclose(dynamics.populations(state).sum(), 1.0, atol=2e-12)
        assert provider.statistics().failure_count == 0
    finally:
        potential.close()
