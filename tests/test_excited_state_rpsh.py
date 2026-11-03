from __future__ import annotations

import numpy as np

from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticRPSHDynamics
from pyrpmd.dynamics import State
from pyrpmd.potentials.excited_state import AvoidedCrossingBackend, ExcitedStateProvider


def test_excited_state_provider_runs_short_rpsh_with_bounded_numerical_drift(tmp_path):
    provider = ExcitedStateProvider(
        AvoidedCrossingBackend(1, slope=0.02, coupling=0.01, harmonic=0.002),
        ["H"],
        2,
        coupling_method="analytic",
        scratch_root=tmp_path,
    )
    potential = AdiabaticPythonPotential(provider)
    dynamics = AdiabaticRPSHDynamics(
        potential,
        masses=[1837.1526473652],
        beta=1000.0,
        nbeads=1,
        dt=0.01,
        nstates=2,
        native=False,
        seed=17,
        electronic_substeps=8,
        overlap_tolerance=1e-10,
    )
    nuclear = State(
        np.array([[[-0.20, 0.03, 0.0]]]),
        np.array([[[1.0, 0.0, 0.0]]]),
    )
    try:
        state = dynamics.initialize(nuclear)
        initial = dynamics.energy(state)
        for _ in range(8):
            dynamics.step(state)
        final = dynamics.energy(state)
        assert abs(final - initial) < 2e-10
        assert state.step == 8
        np.testing.assert_allclose(dynamics.populations(state).sum(), 1.0, atol=2e-14)
        assert provider.statistics().failure_count == 0
    finally:
        potential.close()
