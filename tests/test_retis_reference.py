"""Independent flux bookkeeping and canonical double-well RETIS references.

The continuous-time reference is derived from a phase-space flux integral,
not from a fitted rate or a second implementation of the RETIS estimator.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import PESResult
from pyrpmd.retis import ReplicaExchangeTIS, estimate_retis_flux
from pyrpmd.transition_interface import (
    TransitionInterfaceSampler,
    combine_tis_rate,
    estimate_effective_flux,
)


class QuarticDoubleWell:
    """Quartic reaction coordinate with separable harmonic spectators."""

    def evaluate(self, q):
        energy = np.sum((q[:, :, 0] ** 2 - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
        return PESResult(energy, force)


def phase_point(x, momentum, nbeads=1):
    q = np.zeros((nbeads, 1, 3))
    p = np.zeros_like(q)
    q[:, 0, 0] = x
    p[:, 0, 0] = momentum
    return State(q, p)


@pytest.mark.parametrize(
    "trace, minus_frames, plus_frames",
    [
        # Two return-to-A excursions.  Shared crossing edges occur in both
        # neighboring stored paths, but occur only once in this time trace.
        ([-0.2, -0.5, -0.1, 0.4, 1.0, 0.5,
          -0.2, -0.6, -0.3, 0.2, 1.5, 3.9, -0.1], [5, 5], [5, 5]),
        # The last positive path reaches B.  Its last interior frame already
        # counts the interval into B, so no extra interval may be added.
        ([-0.2, -0.5, -0.1, 0.4, 1.0, 0.5,
          -0.2, -0.6, -0.3, 0.2, 1.5, 3.9, 4.1], [5, 5], [5, 5]),
        # A B-history bridge between reactive excursions contributes no
        # A-history time, including its final outside-to-A interval.
        ([-0.2, -0.5, -0.1, 0.4, 1.0, 3.9, 4.1,
          5.0, 2.0, 0.5, -0.2, -0.6, -0.3, 0.2, 1.5, 3.9, 4.1],
         [5, 5], [5, 5]),
        # The first positive path is the direct two-frame [A, B] path.
        # Its outbound interval is assigned to the minus path's last A frame.
        ([-0.2, -0.5, -0.1, 4.1, 5.0, 0.5,
          -0.2, -0.6, -0.3, 0.2, 1.5, 3.9, 4.1], [5, 5], [2, 5]),
    ],
)
def test_minus_flux_equals_independent_stitched_history_trace(
    trace, minus_frames, plus_frames,
):
    metadata = dict(dt=0.2, basin_a=0.0, basin_b=4.0, first_interface=0.0)
    direct = estimate_effective_flux(trace, **metadata)
    retis = estimate_retis_flux(minus_frames, plus_frames, **metadata)
    assert direct["effective_crossings"] == 2
    assert retis["flux_au"] == pytest.approx(direct["flux_au"], rel=1e-14)
    assert retis["converged"] is False


@pytest.mark.parametrize("nbeads", [1, 4])
def test_zero_exchange_is_an_involution_with_endpoint_canonical_correction(nbeads):
    dynamics = Dynamics(QuarticDoubleWell(), [1], 2.0, nbeads, 0.06, native=False)
    sampler = ReplicaExchangeTIS(
        dynamics, Cartesian(0), basin_a=-1.0, basin_b=1.0,
        interfaces=(-1.0, -0.4), max_path_steps=500, seed=92, minus=True,
    )
    minus = sampler.generate_minus_path(phase_point(-1.15, -0.8, nbeads))
    plus = sampler.generate_path(phase_point(-0.4, 1.4, nbeads))

    # Endpoint propagation is deterministic.  Retry the same proposal only
    # to isolate its mathematical reversibility from Metropolis randomness.
    for _ in range(20):
        new_minus, new_plus, move = sampler.exchange_zero(minus, plus)
        if move.accepted:
            break
    assert move.accepted
    expected = -(dynamics.beta / nbeads) * (
        dynamics.energy(new_minus.state(0)) + dynamics.energy(new_plus.state(0))
        - dynamics.energy(minus.state(0)) - dynamics.energy(plus.state(0))
    )
    assert move.log_acceptance_ratio == pytest.approx(expected, abs=1e-12)
    for _ in range(20):
        recovered_minus, recovered_plus, reverse = sampler.exchange_zero(new_minus, new_plus)
        if reverse.accepted:
            break
    assert reverse.accepted
    assert reverse.log_acceptance_ratio == pytest.approx(-expected, abs=1e-12)
    for original, recovered in ((minus, recovered_minus), (plus, recovered_plus)):
        np.testing.assert_allclose(recovered.q, original.q, atol=3e-12, rtol=0)
        np.testing.assert_allclose(recovered.p, original.p, atol=3e-12, rtol=0)


def _independent_block_errors(crossing, cycle_times, block_size):
    """Block complete synchronized observations before applying a ratio."""
    nblocks = len(crossing) // block_size
    probabilities = np.asarray(crossing[:nblocks * block_size], float).reshape(
        nblocks, block_size, -1,
    ).mean(axis=1)
    duration = np.asarray(cycle_times[:nblocks * block_size]).reshape(
        nblocks, block_size,
    ).mean(axis=1)
    fluxes = 1 / duration
    rates = probabilities.prod(axis=1) / duration
    return np.std(fluxes, ddof=1) / np.sqrt(nblocks), np.std(rates, ddof=1) / np.sqrt(nblocks)


def test_quartic_retis_rate_matches_flux_integral_and_tis_with_external_flux():
    # With A/B at the two minima, every nonstationary left/right-well orbit
    # visits its basin.  Symmetry gives <h_A^history>=1/2.  The normalized
    # positive momentum flux at x=-1 is 1/sqrt(2*pi*beta*m).  Thus
    # f_A = 2/(sqrt(2*pi*beta*m)*Z_x) and P(B|lambda_0)=exp(-beta).
    # The two harmonic spectator integrals cancel exactly, independently
    # checking the separable multidimensional dynamics as well as the 1D rate.
    beta, dt = 1.0, 0.04
    partition, quadrature_error = quad(
        lambda x: np.exp(-beta * (x * x - 1) ** 2), -np.inf, np.inf,
        epsabs=1e-12, epsrel=1e-12,
    )
    assert quadrature_error < 1e-10
    exact_flux = 2 / (np.sqrt(2 * np.pi * beta) * partition)
    exact_probability = np.exp(-beta)
    exact_rate = exact_flux * exact_probability
    dynamics = Dynamics(QuarticDoubleWell(), [1], beta, 1, dt, native=False)
    settings = dict(
        basin_a=-1.0, basin_b=1.0, interfaces=(-1.0, -0.5), max_path_steps=600,
    )
    retis = ReplicaExchangeTIS(dynamics, Cartesian(0), **settings, seed=981, minus=True)
    seed = retis.generate_path(phase_point(-0.3, 1.4), 1)
    minus = retis.generate_minus_path(phase_point(-1.1, -0.7))
    result = retis.sample_interfaces([seed, seed], minus_path=minus, n_samples=2400, burn_in=300)
    summary = result.summary(block_size=120)
    crossing = np.column_stack([item.crossing for item in result.interfaces])
    cycle_times = (result.minus.path_frames + result.interfaces[0].path_frames - 4) * dt
    flux_se, rate_se = _independent_block_errors(crossing, cycle_times, 120)
    assert flux_se > 0 and rate_se > 0
    # Five block standard errors plus a separately stated 1% integration
    # allowance.  These finite-chain checks do not establish convergence.
    assert abs(summary["flux_au"] - exact_flux) <= 5 * flux_se + 0.01 * exact_flux
    assert abs(summary["rate_estimate_au"] - exact_rate) <= 5 * rate_se + 0.01 * exact_rate
    assert not summary["requires_independent_flux"]
    assert summary["converged"] is False
    assert all(item.length_rejections == 0 for item in result.interfaces)

    tis = TransitionInterfaceSampler(dynamics, Cartesian(0), **settings, seed=314)
    independent = tis.sample_interfaces([seed, seed], n_samples=1200, burn_in=200)
    external = dict(
        basin_a=-1.0, basin_b=1.0, first_interface=-1.0, dt_au=dt, flux_au=exact_flux,
    )
    external_rate = combine_tis_rate(independent, external)["rate_estimate_au"]
    tis_crossing = np.column_stack([item.crossing for item in independent])
    _, external_se = _independent_block_errors(
        tis_crossing, np.full(len(tis_crossing), 1 / exact_flux), 120,
    )
    assert abs(external_rate - exact_rate) <= 5 * external_se + 0.01 * exact_rate
    assert abs(summary["rate_estimate_au"] - external_rate) <= (
        5 * np.hypot(rate_se, external_se) + 0.02 * exact_rate
    )
