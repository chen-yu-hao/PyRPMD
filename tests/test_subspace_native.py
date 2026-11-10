"""Compare the native kernel against the existing SciPy/Python algorithm."""

import numpy as np
import pytest

from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_rpmd import _RingPolymer, ring_polymer_rate
from pyrpmd.analysis.subspace_native import NativeRingPolymer, make_ring

native = pytest.importorskip("pyrpmd._subspace")


def model(dimension=1):
    axes = (np.linspace(-5, 5, 41),) + (np.linspace(-2, 2, 17),) * (dimension - 1)
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = 0.5 / np.cosh(points[..., 0])**2
    for a in range(1, dimension):
        values += (0.3 + 0.03 * points[..., 0]) * points[..., a]**2
    return SubspaceModel(axes, values, np.ones(dimension), 0., reactant_reference=-3.)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_native_spline_matches_scipy_at_knots_boundaries_and_outside(dimension):
    m = model(dimension)
    ring = NativeRingPolymer(m, 2., 4, 2)
    rng = np.random.default_rng(9)
    points = rng.uniform(-6, 6, (200, dimension))
    points = np.concatenate([points, m.grid_points().reshape(-1, dimension)[::11],
                             [[a[0] for a in m.axes], [a[-1] for a in m.axes]]])
    energy, gradient = m.energy_and_gradient(points)
    e, g = np.empty(len(points)), np.empty_like(points)
    stats = native.evaluate(ring.plan, points, e, g, 2)
    np.testing.assert_allclose(e, energy, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(g, gradient, rtol=2e-12, atol=2e-14)
    assert stats["outside"] == np.count_nonzero(np.any(
        (points < [a[0] for a in m.axes]) | (points > [a[-1] for a in m.axes]), axis=1))


@pytest.mark.parametrize("beads", [1, 2, 5, 8, 32])
@pytest.mark.parametrize("constrained", [False, True])
def test_native_proposal_matches_python_free_ring_and_force_steps(beads, constrained):
    m = model(2)
    py = _RingPolymer(m, 3., beads)
    cpp = NativeRingPolymer(m, 3., beads, 3)
    rng = np.random.default_rng(8)
    q = rng.normal(scale=0.1, size=(7, beads, 2))
    p = py.sample_momenta(rng, len(q), constrained)
    if constrained:
        q[:, 0, 0] = 0.
    _, g = py.potential(q)
    f = -py.modes(g)
    if constrained:
        f[:, 0, 0] = 0.
    dt = 0.015
    prop = py.free_propagator(dt)
    trial, sides = py.propagate(q, p, f, dt, prop, constrained, 13, surface=0.)
    actual, actual_sides = cpp.propagate(q, p, f, dt, prop, constrained, 13, surface=0.)
    for a, b in zip(actual, trial):
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-12)
    np.testing.assert_array_equal(actual_sides, sides)
    assert cpp.native_steps == 13 * len(q)
    assert cpp.python_steps == cpp.python_pes_calls == 0
    assert cpp.threads_used == (3 if native.openmp else 1)


def test_seeded_rate_keeps_statistics_and_is_thread_count_independent():
    options = dict(windows=9, walkers=8, burn_in=5, sweeps=12, leapfrog_steps=4,
                   trajectories=32, md_dt=0.03, md_time=2., bootstrap=25, seed=12)
    py = ring_polymer_rate(model(), 2., 8, backend="python", **options)
    one = ring_polymer_rate(model(), 2., 8, backend="native", threads=1, **options)
    many = ring_polymer_rate(model(), 2., 8, backend="native", threads=4, **options)
    for field in ("rate", "rate_standard_error", "qtst_rate", "qtst_log_standard_error", "transmission",
                  "transmission_standard_error", "mean_force", "mean_force_standard_error", "free_energy", "kappa"):
        np.testing.assert_allclose(getattr(one, field), getattr(py, field), rtol=5e-11, atol=1e-12)
        np.testing.assert_array_equal(getattr(one, field), getattr(many, field))
    assert one.diagnostics["hmc_acceptance"] == py.diagnostics["hmc_acceptance"]
    assert many.diagnostics["native_steps"] > 0
    assert many.diagnostics["python_steps"] == many.diagnostics["python_pes_callbacks"] == 0
    assert many.diagnostics["physical_converged"] is False


def test_native_rejects_callbacks_instead_of_changing_the_potential():
    from dataclasses import replace

    m = replace(model(), function=lambda p: np.zeros(p.shape[:-1]))
    with pytest.raises(RuntimeError, match="callbacks"):
        make_ring(m, 2., 4, "native", 2)
    assert make_ring(m, 2., 4, "auto", 2).backend == "python"


def test_nonfinite_native_trial_preserves_the_accepted_state():
    ring = NativeRingPolymer(model(), 2., 4, 2)
    q = np.zeros((3, 4, 1))
    p = np.full_like(q, np.inf)
    f = np.zeros_like(q)
    with pytest.raises(FloatingPointError):
        ring.propagate(q, p, f, 0.1, ring.free_propagator(0.1), False, 2)
    np.testing.assert_array_equal(q, 0.)
    np.testing.assert_array_equal(f, 0.)
