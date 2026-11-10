"""Reference parity, safety and OpenMP determinism for the subspace kernels."""
from concurrent.futures import ThreadPoolExecutor
import sys

import numpy as np
import pytest

from pyrpmd.analysis.subspace_model import SubspaceModel, model_from_function
from pyrpmd.analysis.subspace_rpmd import _RingPolymer, ring_polymer_rate

native = pytest.importorskip("pyrpmd._subspace_native")


def grid_model(d=1, *, nonuniform=False):
    axes = tuple(np.linspace(-4., 4., 12 + i) for i in range(d))
    # SubspaceModel currently requires uniform axes; direct spline tests also
    # cover arbitrary knot spacing below.
    mesh = np.meshgrid(*axes, indexing="ij")
    values = .3 / np.cosh(mesh[0])**2
    for i in range(1, d):
        values = values + .2 * mesh[i]**2 + .01 * mesh[0] * mesh[i]
    return SubspaceModel(axes, values, np.arange(d) + 1., 0., "asymptotic", -3., {})


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_spline_values_gradients_and_boundaries(dimension, threads):
    if threads > 1 and not native.OPENMP_ENABLED:
        pytest.skip("OpenMP disabled")
    model = grid_model(dimension)
    ring = _RingPolymer(model, 3., 1, backend="native", threads=threads)
    rng = np.random.default_rng(19)
    points = np.concatenate([rng.uniform(-6, 6, (80, dimension)),
                             np.full((1, dimension), -4.), np.full((1, dimension), 4.)])
    energy = np.empty(len(points)); gradient = np.empty_like(points)
    count, outside = native.evaluate(ring.native.context, points, energy, gradient)
    expected, derivative = model.energy_and_gradient(points)
    np.testing.assert_allclose(energy, expected, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(gradient, derivative, atol=2e-14, rtol=2e-12)
    assert count == len(points)
    assert outside == np.any((points < -4) | (points > 4), axis=1).sum()


def test_nonuniform_spline_knots():
    from scipy.interpolate import make_interp_spline
    x = np.array([-4., -2., -1.9, -.1, .5, .6, 3., 5.])
    spl = make_interp_spline(x, np.sin(x), k=3)
    ctx = native.make_context((spl.t,), spl.c, np.ones(1), np.ones((1, 1)), np.zeros(1), 1)
    points = np.linspace(-4, 5, 105)[:, None].copy()
    energy = np.empty(len(points)); gradient = np.empty_like(points)
    native.evaluate(ctx, points, energy, gradient)
    np.testing.assert_allclose(energy, spl(points[:, 0]), atol=2e-14)
    np.testing.assert_allclose(gradient[:, 0], spl(points[:, 0], nu=1), atol=2e-14)


@pytest.mark.parametrize("beads", [1, 2, 3, 8, 32, 64])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("dimension", [1, 2])
def test_fused_steps_match_numpy(beads, constrained, dimension):
    ref = _RingPolymer(grid_model(dimension), 3., beads, backend="numpy")
    opt = _RingPolymer(grid_model(dimension), 3., beads, backend="native",
                       threads=2 if native.OPENMP_ENABLED else 1)
    rng = np.random.default_rng(4)
    q = rng.normal(0., .15, (7, beads, dimension))
    p = ref.sample_momenta(rng, len(q), constrained)
    e, g = ref.potential(q); f = -ref.modes(g)
    if constrained:
        f[:, 0, 0] = 0.
    original = [v.copy() for v in (q, p, f, e, g)]
    side_a = np.empty((18, len(q)), bool); side_b = np.empty_like(side_a)
    a = ref.advance((q, p, f, e, g), .002, ref.free_propagator(.002), constrained, 17, side=side_a)
    b = opt.advance((q, p, f, e, g), .002, opt.free_propagator(.002), constrained, 17, side=side_b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(side_a, side_b)
    for x, y in zip((q, p, f, e, g), original):
        np.testing.assert_array_equal(x, y)
    if constrained:
        np.testing.assert_array_equal(b[0][:, 0, 0], q[:, 0, 0])


def rate_options():
    return dict(windows=8, walkers=4, burn_in=5, sweeps=10, trajectories=16,
                md_time=1., md_dt=.02, bootstrap=12, seed=11)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_complete_rate_and_pair_bootstrap_parity(dimension):
    a = ring_polymer_rate(grid_model(dimension), 3., 4, backend="numpy", **rate_options())
    b = ring_polymer_rate(grid_model(dimension), 3., 4, backend="native", **rate_options())
    for name in ("rate", "rate_standard_error", "qtst_rate", "transmission", "transmission_standard_error"):
        assert getattr(a, name) == pytest.approx(getattr(b, name), rel=2e-10, abs=1e-12)
    for name in ("mean_force", "mean_force_standard_error", "free_energy", "kappa"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=2e-10, rtol=2e-10)
    assert a.diagnostics["hmc_acceptance"] == b.diagnostics["hmc_acceptance"]
    assert a.diagnostics["spline_outside_fraction"] == b.diagnostics["spline_outside_fraction"]


@pytest.mark.parametrize("threads,block", [(1, 1), (1, 11), (2, 7), (4, 32)])
def test_thread_and_block_independence(threads, block):
    if threads > 1 and not native.OPENMP_ENABLED:
        pytest.skip("OpenMP disabled")
    a = ring_polymer_rate(grid_model(), 3., 4, backend="native", threads=1, block_steps=5, **rate_options())
    b = ring_polymer_rate(grid_model(), 3., 4, backend="native", threads=threads, block_steps=block, **rate_options())
    assert a.rate == b.rate
    np.testing.assert_array_equal(a.kappa, b.kappa)
    np.testing.assert_array_equal(a.mean_force, b.mean_force)


def test_shared_immutable_context_is_reentrant():
    ring = _RingPolymer(grid_model(2), 3., 4, backend="native")
    q = np.random.default_rng(0).normal(size=(50, 2))
    def evaluate(_):
        e = np.empty(len(q)); g = np.empty_like(q)
        native.evaluate(ring.native.context, q, e, g)
        return e, g
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(evaluate, range(6)))
    for e, g in results:
        np.testing.assert_array_equal(e, results[0][0]); np.testing.assert_array_equal(g, results[0][1])


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_thread_and_block_controls(bad):
    with pytest.raises(ValueError):
        ring_polymer_rate(grid_model(), 3., 4, threads=bad, **rate_options())
    with pytest.raises(ValueError):
        ring_polymer_rate(grid_model(), 3., 4, block_steps=bad, **rate_options())


def test_python_callbacks_are_never_replaced_by_a_grid():
    model = model_from_function(lambda q: .1 * q[..., 0]**2, [np.linspace(-4, 4, 20)])
    ring = _RingPolymer(model, 3., 1, backend="auto")
    assert ring.native is None
    with pytest.raises(ValueError, match="static grid"):
        _RingPolymer(model, 3., 1, backend="native")
    with pytest.raises(ValueError):
        _RingPolymer(model, 3., 1, backend="auto", threads=2)


def test_numpy_fallback_without_extension(monkeypatch):
    import pyrpmd
    monkeypatch.delattr(pyrpmd, "_subspace_native", raising=False)
    monkeypatch.setitem(sys.modules, "pyrpmd._subspace_native", None)
    ring = _RingPolymer(grid_model(), 3., 1, backend="auto")
    assert ring.native is None
    with pytest.raises(ValueError, match="not installed"):
        _RingPolymer(grid_model(), 3., 1, backend="native")


def test_native_buffers_are_checked():
    ctx = _RingPolymer(grid_model(), 3., 1, backend="native").native.context
    q = np.zeros((5, 1)); e = np.empty(5); g = np.empty_like(q)
    for bad in (q.astype(np.float32), q.astype(">f8"), np.zeros((5, 2))[:, ::2], np.zeros((5, 2))):
        with pytest.raises((ValueError, BufferError)):
            native.evaluate(ctx, bad, e, g)
    with pytest.raises(ValueError, match="overlap"):
        native.evaluate(ctx, q, q[:, 0], g)
    e.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        native.evaluate(ctx, q, e, g)
    with pytest.raises(ValueError):
        native.evaluate(None, q, np.empty(5), g)


def test_failed_proposal_does_not_mutate_accepted_state():
    ring = _RingPolymer(grid_model(), 3., 1, backend="native")
    q = np.zeros((2, 1, 1)); p = np.full_like(q, 1e308)
    e, g = ring.potential(q); f = -ring.modes(g)
    before = [v.copy() for v in (q, p, f, e, g)]
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError):
        ring.advance((q, p, f, e, g), 1e10, ring.free_propagator(1e10), False, 3)
    for a, b in zip(before, (q, p, f, e, g)):
        np.testing.assert_array_equal(a, b)


def test_noncanonical_mode_matrix_does_not_use_canonical_fft():
    ref = _RingPolymer(grid_model(), 3., 32, backend="numpy")
    ref.c[:, -1] *= -1.
    model = grid_model()
    spline = model.interpolant._spline
    ctx = native.make_context(tuple(np.ascontiguousarray(t) for t in spline.t),
                              np.ascontiguousarray(spline.c), ref.mass, ref.c, ref.omega, 1)
    rng = np.random.default_rng(42)
    q = rng.normal(0., .1, (3, 32, 1)); p = ref.sample_momenta(rng, 3, False)
    e, g = ref.potential(q); f = -ref.modes(g)
    expected = ref.advance((q, p, f, e, g), .001, ref.free_propagator(.001), False, 4)
    result = [np.ascontiguousarray(a).copy() for a in (q, p, f, e, g)]
    native.advance(ctx, *result, .001, 4, 0, None, 0.)
    for a, b in zip(result, expected):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_unaligned_buffer_rejected():
    ctx = _RingPolymer(grid_model(), 3., 1, backend="native").native.context
    q = np.ndarray((5, 1), dtype=np.float64, buffer=bytearray(41), offset=1)
    q.fill(0.)
    with pytest.raises(ValueError):
        native.evaluate(ctx, q, np.empty(5), np.empty((5, 1)))
