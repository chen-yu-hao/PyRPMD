"""Blocked Schur/Sylvester regression, including scaling and CAP options."""
import numpy as np
import pytest

from pyrpmd.analysis.subspace_linalg import triangular_sylvester
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate


@pytest.mark.parametrize("n,block", [(1, 1), (7, 2), (67, 16), (257, 32)])
def test_blocked_solution_and_residual(n, block):
    rng = np.random.default_rng(14)
    t = np.triu(.03 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))))
    t[np.diag_indices(n)] = rng.normal(size=n) - 1j * rng.uniform(.1, 1., n)
    rhs = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    original_t, original_rhs = t.copy(), rhs.copy()
    a, sa, ia, _ = triangular_sylvester(t, rhs, solver="lapack")
    b, sb, ib, _ = triangular_sylvester(t, rhs, solver="blocked", block_size=block)
    np.testing.assert_allclose(a / sa, b / sb, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(t.conj().T @ b - b @ t - sb * rhs) / np.linalg.norm(rhs) < 5e-13
    assert ia == ib == 0
    np.testing.assert_array_equal(t, original_t); np.testing.assert_array_equal(rhs, original_rhs)


def test_singular_tile_falls_back_to_reference():
    t = np.diag(np.arange(9, dtype=complex)); rhs = np.ones((9, 9), complex)
    a, sa, ia, _ = triangular_sylvester(t, rhs, solver="lapack")
    b, sb, ib, meta = triangular_sylvester(t, rhs, solver="blocked", block_size=3)
    assert meta["solver"] == "lapack" and meta["fallback_reason"]
    np.testing.assert_array_equal(a, b)
    assert sa == sb and ia == ib


def test_lapack_scaling_is_not_discarded():
    t = np.diag(np.arange(8, dtype=float) - 1j * 1e-10)
    rhs = np.full((8, 8), 1e300 + 0j)
    a, sa, ia, _ = triangular_sylvester(t, rhs, solver="lapack")
    b, sb, ib, _ = triangular_sylvester(t, rhs, solver="blocked", block_size=2)
    assert 0 < sb <= 1 and np.isfinite(b).all()
    # Compare in scaled units, avoiding division into overflow.
    np.testing.assert_allclose(b * sa, a * sb, rtol=2e-12, atol=0)
    assert ia == ib


def model():
    x = np.linspace(-4., 4., 65)
    return SubspaceModel((x,), .2 / np.cosh(x)**2, [1.], 0., "asymptotic", -3., {})


@pytest.mark.parametrize("normalization", ["default", "slice"])
def test_dense_rate_preserves_threshold_truncation_and_normalization(normalization):
    m = model()
    options = dict(energy_floor=0., truncation_energy=2., normalization=normalization)
    a = exact_quantum_rate(m, 3., sylvester_solver="lapack", **options)
    b = exact_quantum_rate(m, 3., sylvester_solver="blocked", sylvester_block_size=16, **options)
    assert a.rate == pytest.approx(b.rate, rel=1e-12)
    assert a.reactant_partition == b.reactant_partition


def test_grid_absorber_matches_callable():
    m = model()
    def cap(p): return .1 * np.maximum(abs(p[..., 0]) - 2.5, 0.)**2
    a = exact_quantum_rate(m, 3., absorber=cap)
    b = exact_quantum_rate(m, 3., absorber=cap(m.grid_points()).reshape(m.shape))
    assert a.rate == b.rate


def test_dense_memory_guard_precedes_matrix_allocation(monkeypatch):
    from pyrpmd.analysis import subspace_quantum as module
    def forbidden(*args, **kwargs): raise AssertionError("allocated dense Hamiltonian")
    monkeypatch.setattr(module._GridOperators, "dense_hamiltonian", forbidden)
    with pytest.raises(MemoryError, match="estimate"):
        exact_quantum_rate(model(), 3., max_memory_mb=.001)


@pytest.mark.parametrize("options", [{"solver": "unknown"}, {"block_size": True}, {"block_size": 0}])
def test_invalid_solver_controls(options):
    with pytest.raises(ValueError):
        triangular_sylvester(np.eye(3, dtype=complex), np.eye(3, dtype=complex), **options)
