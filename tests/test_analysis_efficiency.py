"""Allocation shortcuts must retain the same sampling and MBAR problem."""

import numpy as np
import pytest

from pyrpmd.analysis import fes


def _legacy_block_resample(x, block, rng):
    """Independent reference for the existing circular-block draw order."""
    n = len(x)
    block = max(1, min(int(block), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    indices = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
    return x[indices]


@pytest.mark.parametrize("n,block", [(1, 1), (3, 0), (7, 2), (10, 3), (10, 99),
                                    (101, 7), (1024, 5)])
@pytest.mark.parametrize("bit_generator", [np.random.PCG64, np.random.Philox])
def test_block_resampling_preserves_indices_and_rng_state(n, block, bit_generator):
    # Non-contiguous rows also exercise the array-valued callers, while each
    # row identifies exactly which time point was selected.
    samples = np.arange(n * 6).reshape(n, 2, 3)[::-1]
    reference_rng = np.random.Generator(bit_generator(1951))
    actual_rng = np.random.Generator(bit_generator(1951))
    for _ in range(3):
        expected = _legacy_block_resample(samples, block, reference_rng)
        actual = fes._block_resample(samples, block, actual_rng)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_rng.random(20), reference_rng.random(20))


@pytest.mark.parametrize("method", ["auto", "legacy"])
@pytest.mark.parametrize("combine_identical_states", [True, False])
def test_optional_state_weights_preserve_public_default_and_all_other_mbar_outputs(
    method, combine_identical_states,
):
    counts = np.array([50, 70, 40])
    rng = np.random.default_rng(381)
    x = np.concatenate([rng.normal(center, 0.8, n)
                        for center, n in zip([-0.3, 0.5, -0.3], counts)])
    u = fes.umbrella_reduced_potentials(x[:, None], [[-0.3], [0.5], [-0.3]],
                                       [[1.2]], 1.4)
    common = np.linspace(-50, 30, len(x))
    options = dict(target_u=common + 0.2 * x**2, tolerance=1e-11,
                   method=method, combine_identical_states=combine_identical_states)
    full = fes.solve_mbar(u + common, counts, **options)
    compact = fes.solve_mbar(u + common, counts, return_state_weights=False, **options)
    assert full.state_weights.shape == (sum(counts), len(counts))
    np.testing.assert_allclose(full.state_weights.sum(axis=0), 1.0, atol=1e-10)
    assert compact.state_weights is None
    for name in ("free_energies", "weights", "overlap", "components"):
        np.testing.assert_array_equal(getattr(compact, name), getattr(full, name))
    assert compact.converged is full.converged is True
    assert compact.residual == full.residual
    for name, value in full.diagnostics.items():
        np.testing.assert_equal(compact.diagnostics[name], value)


def test_fes_reuses_bias_columns_with_the_same_periodic_multistream_bootstrap(monkeypatch):
    rng = np.random.default_rng(1825)
    centers = np.array([[-0.6, -0.2], [0.6, 0.3], [-0.6, -0.2]])
    stiffness = np.array([[[0.6, 0.1], [0.1, 1.2]],
                          [[0.4, -0.08], [-0.08, 0.9]],
                          [[0.6, 0.1], [0.1, 1.2]]])
    counts = np.array([300, 250, 200])
    samples = [rng.normal(center, [1.4, 0.65], size=(n, 2))
               for center, n in zip(centers, counts)]
    for i, sample in enumerate(samples):
        sample[:, 0] += (i - 1) * 4 * np.pi
    original_samples = [sample.copy() for sample in samples]
    edges = [np.linspace(-np.pi, np.pi, 8), np.array([-1.5, -0.3, 0.4, 1.5])]
    periods, beta, draws, seed, min_count = (2 * np.pi, None), 1.3, 6, 915, 3
    wrapped = [sample.copy() for sample in samples]
    for sample in wrapped:
        sample[:, 0] = (sample[:, 0] + np.pi) % (2 * np.pi) - np.pi
    x = np.concatenate(wrapped)
    build_bias, solve, histogram = fes.umbrella_reduced_potentials, fes.solve_mbar, fes._histogram
    u = build_bias(x, centers, stiffness, beta, periods)
    fit = solve(u, counts)
    _, _, _, point_free, point_mask = histogram(x, fit.weights, edges, beta, min_count)
    reference_bin = np.unravel_index(np.nanargmin(point_free), point_free.shape)
    gs = np.array([fes.statistical_inefficiency(np.column_stack(
        (np.sin(sample[:, 0]), np.cos(sample[:, 0]), sample[:, 1]))) for sample in wrapped])
    plan = fes._bootstrap_block_plan(counts, gs)
    assert plan["status"] != "unavailable"
    reference_rng = np.random.default_rng(seed)
    expected_coordinates, expected_matrices, reference_curves = [x], [u], []
    for _ in range(draws):
        xb = np.concatenate([_legacy_block_resample(sample, block, reference_rng)
                             for sample, block in zip(wrapped, plan["block_lengths"])])
        ub = build_bias(xb, centers, stiffness, beta, periods)
        mb = solve(ub, counts, initial_free_energies=fit.free_energies)
        assert mb.converged and len(np.unique(mb.components)) == 1
        _, _, _, fb, _ = histogram(xb, mb.weights, edges, beta, min_count)
        if np.isfinite(fb[reference_bin]):
            reference_curves.append(fb - fb[reference_bin])
        expected_coordinates.append(xb)
        expected_matrices.append(ub)
    bias_calls, solve_calls, histogram_calls = [], [], []

    def counted_bias(*args, **kwargs):
        bias_calls.append(True)
        return build_bias(*args, **kwargs)

    def checked_solve(reduced, sizes, **kwargs):
        np.testing.assert_array_equal(reduced, expected_matrices[len(solve_calls)])
        np.testing.assert_array_equal(sizes, counts)
        assert kwargs.get("return_state_weights") is False
        solve_calls.append(True)
        return solve(reduced, sizes, **kwargs)

    def checked_histogram(coordinates, *args, **kwargs):
        np.testing.assert_array_equal(coordinates, expected_coordinates[len(histogram_calls)])
        histogram_calls.append(True)
        return histogram(coordinates, *args, **kwargs)

    monkeypatch.setattr(fes, "umbrella_reduced_potentials", counted_bias)
    monkeypatch.setattr(fes, "solve_mbar", checked_solve)
    monkeypatch.setattr(fes, "_histogram", checked_histogram)
    result = fes.reconstruct_fes(samples, centers, stiffness, beta, bins=edges,
                                 periods=periods, bootstrap=draws, seed=seed,
                                 min_count=min_count)
    assert len(bias_calls) == 1
    assert len(solve_calls) == len(histogram_calls) == draws + 1
    np.testing.assert_array_equal(result["free_energy"], point_free)
    np.testing.assert_array_equal(result["mask"], point_mask)
    assert result["bootstrap_replicates"] == len(reference_curves)
    expected_se = np.full_like(point_free, np.nan)
    expected_valid = np.zeros_like(point_free, dtype=int)
    curves = np.asarray(reference_curves)
    for index in np.ndindex(point_free.shape):
        values = curves[(slice(None),) + index]
        values = values[np.isfinite(values)]
        if not point_mask[index]:
            expected_valid[index] = len(values)
            if len(values) > 1:
                expected_se[index] = values.std(ddof=1)
    np.testing.assert_allclose(result["standard_error"], expected_se, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(result["bootstrap_valid_per_bin"], expected_valid)
    for original, sample in zip(original_samples, samples):
        np.testing.assert_array_equal(sample, original)


@pytest.mark.parametrize("estimator", ["rate", "surface"])
def test_rate_pmf_internal_solves_do_not_retain_expanded_state_weights(monkeypatch, estimator):
    from pyrpmd.analysis import surface

    module = fes if estimator == "rate" else surface
    solve = module.solve_mbar
    results = []

    def checked_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        assert result.state_weights is None
        results.append(result)
        return result

    sample = np.tile(np.arange(0.1, 1.0, 0.2), 40)
    np.random.default_rng(51).shuffle(sample)
    monkeypatch.setattr(module, "solve_mbar", checked_solve)
    options = dict(bins=np.linspace(0, 1, 6), bootstrap=3, seed=72, min_count=1)
    if estimator == "rate":
        result = fes.mbar_rate_pmf([sample], [0.5], 1e-12, 1.0,
                                   reference=0.1, xi_star=0.7, **options)
    else:
        result = surface.surface_rate_pmf([sample[:, None]], [[0.5]], 1e-12, 1.0,
                                          basin=(0.1, 0.5), dividing_surface=0.7,
                                          mass_metric=1.0, **options)
    assert result["ready"]
    assert len(results) == 4
