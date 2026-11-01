"""Independent polynomial and full-Hamiltonian checks for compact Coulomb RPC."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water.compact_split import NativeCompactRPMD, compact_system, splitting_function
from benchmarks.water.qtip4pf import initial_grid, make_system
from benchmarks.water.split_reference import ContractedSlowPotential, GroupEvaluator


def test_primary_polynomial_endpoints_and_short_potential_three_derivatives():
    # Independent polynomial coefficients from JCP 131, 094102, Eq. (35).
    from numpy.polynomial import Polynomial
    f = Polynomial([0, 7/4, 0, 0, 0, -21/4, 7, -5/2])
    long = Polynomial([7/4, 0, 0, 0, -21/4, 7, -5/2])
    x = np.linspace(0, 1, 21)
    np.testing.assert_allclose(splitting_function(x), f(x), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(splitting_function([1, 2, 100]), 1)
    # Short potential = 1/x - f(x)/x. At x=1 its first 3 derivatives vanish.
    for order, reciprocal_derivative in enumerate((1., -1., 2., -6.)):
        assert long.deriv(order)(1) == pytest.approx(reciprocal_derivative, abs=1e-14)
    assert all(long.deriv(order)(0) == 0 for order in (1, 2, 3))


def test_compact_split_sum_matches_original_full_model_with_nonzero_LJ():
    pytest.importorskip("openmm")
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=36)
    q[3:6, 0] -= .6  # Nonzero, nontrivial LJ inside the deliberately small cutoff.
    original = GroupEvaluator(make_system(molecules=8, box_angstrom=box, cutoff_angstrom=3., ewald_tolerance=1e-7))
    split_system, metadata = compact_system(molecules=8, box_angstrom=box, cutoff_angstrom=3.,
                                            splitting_angstrom=2.5, ewald_tolerance=1e-7)
    split = GroupEvaluator(split_system)
    try:
        frames = np.stack([q, q + np.random.default_rng(21).normal(scale=.001, size=q.shape)])
        energy, force = original.evaluate(frames)
        split_energy, split_force = split.evaluate(frames)
        np.testing.assert_allclose(split_energy, energy, atol=2e-14, rtol=0)
        np.testing.assert_allclose(split_force, force, atol=3e-14, rtol=0)
        lj_energy, lj_force = split.evaluate(frames, groups=1 << 3)
        assert np.max(abs(lj_energy)) > 1e-7 and np.max(abs(lj_force)) > 1e-7
        assert not metadata["identical_to_2008_qtip_RPC_claimed"]
    finally:
        original.close()
        split.close()


@pytest.mark.parametrize("copies", [1, 3, 8])
def test_compact_contracted_hamiltonian_gradient(copies):
    pytest.importorskip("openmm")
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=23)
    q[3:6, 0] -= .6
    ring = q[None] + np.random.default_rng(13).normal(scale=.02, size=(8, *q.shape))
    system, _ = compact_system(molecules=8, box_angstrom=box, cutoff_angstrom=3., splitting_angstrom=2.5)
    evaluator = GroupEvaluator(system)
    try:
        potential = ContractedSlowPotential(evaluator, 8, copies)
        energy, force = potential.evaluate(ring)
        index, delta = (2, 4, 1), 1e-5
        plus, minus = ring.copy(), ring.copy()
        plus[index] += delta
        minus[index] -= delta
        gradient = (potential.evaluate(plus)[0] - potential.evaluate(minus)[0]) / (2 * delta)
        assert force[index] == pytest.approx(-gradient, abs=2e-10, rel=2e-6)
        assert np.isfinite(energy)
    finally:
        evaluator.close()


@pytest.mark.parametrize("copies", [1, 3])
def test_native_compact_rpc_matches_independent_bab_with_virtual_sites(copies):
    pytest.importorskip("openmm")
    from benchmarks.water.native_optimizations import openmm_frequency_beta
    from benchmarks.water.native_reference import sample_momenta
    from benchmarks.water.split_reference import SplitMTS
    from pyrpmd.constants import FS_TO_AU
    from pyrpmd.dynamics import Dynamics, State
    grid, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=52)
    grid[3:6, 2] -= .6
    native = NativeCompactRPMD(molecules=8, box_angstrom=box, beads=8, long_copies=copies,
                               dt_fs=.1, splitting_angstrom=2.5, cutoff_angstrom=3.)
    evaluator = GroupEvaluator(native.system)
    independent = SplitMTS(ContractedSlowPotential(evaluator, 8, copies), native.masses, 298, 8, .1, .1)
    independent.fast = Dynamics(independent.fast.potential, native.masses, openmm_frequency_beta(298),
                                8, .1 * FS_TO_AU, native=True, threads=1, integrator="normal-mode-bab")
    rng = np.random.default_rng(20)
    q = np.repeat(grid[None], 8, axis=0) + rng.normal(0, .02, (8, 24, 3))
    p = np.stack([sample_momenta(rng, native.masses, 298 * 8, remove_com=False)[0] for _ in range(8)])
    state = State(q.copy(), p.copy())
    try:
        native.set_state(q, p)
        for _ in range(3):
            native.integrator.step(1)
            independent.step(state)
        nq, np_ = native.get_state()
        np.testing.assert_allclose(nq, state.q, atol=2e-9, rtol=0)
        np.testing.assert_allclose(np_, state.p, atol=5e-8, rtol=0)
    finally:
        native.close()
        evaluator.close()
