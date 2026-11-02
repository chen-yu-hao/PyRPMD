"""Independent Fourier tests of the heavy-nucleus electronic limit.

The finite, moving-nucleus comparisons are runnable examples. These fast
regressions use analytic electronic quadrature and a Gaussian spatial
quadrature instead of noisy Monte Carlo tolerances. They test the population
weights and diabatic propagation against an independent wavepacket solver.
"""
import numpy as np
import pytest

from pyrpmd.analysis.quantum_reference import SplitOperator1D
from pyrpmd.dynamics import State
from pyrpmd.mash import MultistateMASHDynamics
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling, TullyModelOne


@pytest.mark.parametrize("nstates", [2, 3])
def test_tully_and_three_state_populations_match_fourier_heavy_nucleus_limit(nstates):
    model = TullyModelOne() if nstates == 2 else LinearVibronicCoupling(
        [[-.01, .008, .003], [.008, .004, .007], [.003, .007, .014]],
        [np.diag([.01, -.006, .004])], coordinates=[0])
    mass, center, sigma, duration = 1e14, -.25, .002, 60.
    grid = np.linspace(center-.04, center+.04, 128, endpoint=False)
    qgrid = np.zeros((len(grid), 1, 3))
    qgrid[:, 0, 0] = grid
    psi = np.zeros((len(grid), nstates), dtype=complex)
    psi[:, 0] = np.exp(-(grid-center)**2/(4*sigma**2))
    psi /= np.sqrt((grid[1]-grid[0])*np.sum(abs(psi)**2))
    reference = SplitOperator1D(grid, model.evaluate(qgrid).matrix, mass, .5)
    exact = reference.propagate(psi, round(duration/.5), sample_stride=round(duration/.5))

    # Uniform conditional complex-sphere first moments: E[P_initial]=H_N/N.
    # N equally spaced phase vectors integrate every off-diagonal moment to
    # zero. This quadrature exactly integrates a fixed-nucleus quadratic
    # electronic observable; it is not a replacement sampling prescription
    # for coupled nuclear trajectories.
    harmonic = sum(1./k for k in range(1, nstates+1))
    moments = np.full(nstates, (1-harmonic/nstates)/(nstates-1))
    moments[0] = harmonic/nstates
    roots, spatial_weights = np.polynomial.hermite.hermgauss(5)
    spatial_weights /= np.sqrt(np.pi)
    estimate = np.zeros(nstates)
    driver = MultistateMASHDynamics(model, [mass], 100., 1, duration, nstates,
                                    rtol=1e-11, atol=1e-13, max_step=2., native=False)
    for root, weight in zip(roots, spatial_weights):
        q = np.zeros((1, 1, 3))
        q[0, 0, 0] = center+np.sqrt(2)*sigma*root
        for phase in range(nstates):
            c = np.sqrt(moments)*np.exp(2j*np.pi*phase*np.arange(nstates)/nstates)
            state = driver.initialize(State(q, np.zeros_like(q)), basis="diabatic", coefficients=c)
            driver.step(state)
            estimate += weight*driver.populations(state, basis="diabatic")/nstates
    np.testing.assert_allclose(estimate, exact.populations[-1], rtol=0., atol=2e-9)
    assert np.max(exact.edge_population) < 1e-20
    np.testing.assert_allclose(exact.norm, 1., rtol=0., atol=1e-12)
    assert exact.populations[-1, 0] < .95  # Nontrivial electronic transfer.
