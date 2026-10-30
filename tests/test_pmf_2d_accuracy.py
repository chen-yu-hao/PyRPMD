"""Independent PMF oracles and regressions for support/periodic semantics.

The Gaussian fixture samples each biased Boltzmann distribution analytically;
it does not use the dynamics, MBAR solver, or reconstruction to make its input.
The reference density is integrated over each finite, nonuniform rectangle,
so discretization at a bin center is not confused with statistical error.
"""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from pyrpmd.analysis.fes import reconstruct_fes, solve_mbar, umbrella_reduced_potentials


def correlated_harmonic_fixture():
    physical = np.array([[1.8, 0.7], [0.7, 1.1]])
    mean, beta = np.array([0.25, -0.3]), 1.3
    centers = np.array([(x, y) for x in [-1.3, -0.2, 0.9, 1.7]
                        for y in [-1.6, -0.1, 1.2]])
    stiffness = np.array([[[2.5 + 0.1 * i, 0.35], [0.35, 3.2 + 0.08 * i]]
                          for i in range(len(centers))])
    counts = np.array([4000 + 173 * i for i in range(len(centers))])
    rng = np.random.default_rng(21223)
    samples, exact_f = [], []
    for c, k, n in zip(centers, stiffness, counts):
        hessian, linear = physical + k, physical @ mean + k @ c
        biased_mean = np.linalg.solve(hessian, linear)
        samples.append(rng.multivariate_normal(
            biased_mean, np.linalg.inv(beta * hessian), n))
        exact_f.append(0.5 * np.linalg.slogdet(hessian)[1] + 0.5 * beta * (
            mean @ physical @ mean + c @ k @ c - linear @ biased_mean))
    edges = [
        np.array([-2.2, -1.7, -1.3, -0.95, -0.6, -0.25, 0.1, 0.5, 0.9, 1.4, 2.0, 2.6]),
        np.array([-2.8, -2.1, -1.65, -1.25, -0.85, -0.45, -0.1, 0.3, 0.75, 1.3, 2.0, 2.8]),
    ]
    return dict(physical=physical, mean=mean, beta=beta, centers=centers,
                stiffness=stiffness, counts=counts, samples=samples, edges=edges,
                exact_free_energies=np.asarray(exact_f) - exact_f[0])


def gaussian_bin_density(physical, mean, beta, edges):
    """18-point tensor quadrature, independent of histogram/MBAR routines."""
    z, w = leggauss(18)
    result = np.empty(tuple(len(e) - 1 for e in edges))
    prefactor = np.sqrt(np.linalg.det(beta * physical)) / (2 * np.pi)
    for i, j in np.ndindex(result.shape):
        x = (edges[0][i + 1] + edges[0][i]) / 2 + z * np.diff(edges[0])[i] / 2
        y = (edges[1][j + 1] + edges[1][j]) / 2 + z * np.diff(edges[1])[j] / 2
        delta = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1) - mean
        density = prefactor * np.exp(-0.5 * beta * np.einsum(
            "...i,ij,...j->...", delta, physical, delta))
        result[i, j] = np.einsum("i,j,ij", w, w, density) / 4
    return result


@pytest.fixture(scope="module")
def harmonic_data():
    return correlated_harmonic_fixture()


def test_coupled_gaussian_free_energies_moments_and_nonuniform_density(harmonic_data):
    a = harmonic_data
    x = np.concatenate(a["samples"])
    u = umbrella_reduced_potentials(x, a["centers"], a["stiffness"], a["beta"])
    fit = solve_mbar(u, a["counts"], tolerance=1e-11)
    assert fit.converged and len(set(fit.components)) == 1
    np.testing.assert_allclose(fit.free_energies, a["exact_free_energies"], atol=0.075, rtol=0)
    np.testing.assert_allclose(fit.weights @ x, a["mean"], atol=0.025, rtol=0)
    covariance = (x - a["mean"]).T @ (fit.weights[:, None] * (x - a["mean"]))
    np.testing.assert_allclose(covariance, np.linalg.inv(a["beta"] * a["physical"]),
                               atol=0.035, rtol=0)
    result = reconstruct_fes(a["samples"], a["centers"], a["stiffness"], a["beta"],
                             bins=a["edges"], bootstrap=3, min_count=60)
    exact = -np.log(gaussian_bin_density(a["physical"], a["mean"], a["beta"], a["edges"])) / a["beta"]
    good = ~result["mask"]
    assert np.count_nonzero(good) >= 80
    assert np.std((result["free_energy"] - exact)[good]) < 0.075
    np.testing.assert_allclose(result["overlap"].sum(axis=1), 1.0, atol=2e-8)
    assert result["bootstrap_attempted"] == result["bootstrap_replicates"] == 3
    assert np.all(result["bootstrap_valid_per_bin"][good] >= 2)

    # The 1D PMF is -log(integral(exp(-beta F(x,y)),dy))/beta. A
    # minimum-energy cut has the wrong curvature for this coupled Gaussian.
    density_x = result["probability_mass"].sum(axis=1) / np.diff(a["edges"][0])
    sigma_x = np.sqrt(np.linalg.inv(a["beta"] * a["physical"])[0, 0])
    exact_x = np.diff(ndtr((a["edges"][0] - a["mean"][0]) / sigma_x)) / np.diff(a["edges"][0])
    enough = result["raw_count"].sum(axis=1) > 500
    marginal_difference = -np.log(density_x[enough] / exact_x[enough]) / a["beta"]
    assert np.std(marginal_difference) < 0.06


def test_nonsquare_grid_preserves_x_y_axis_order_against_independent_quadrature(harmonic_data):
    a = harmonic_data
    edges = [a["edges"][0], np.array([
        -2.8, -2.2, -1.8, -1.4, -1.1, -0.8, -0.5,
        -0.2, 0.1, 0.5, 0.9, 1.4, 2.0, 2.8,
    ])]
    result = reconstruct_fes(a["samples"], a["centers"], a["stiffness"], a["beta"],
                             bins=edges, bootstrap=0, min_count=60)
    exact_density = gaussian_bin_density(a["physical"], a["mean"], a["beta"], edges)
    assert result["free_energy"].shape == result["density"].shape == (11, 13)
    np.testing.assert_array_equal(result["edges"][0], edges[0])
    np.testing.assert_array_equal(result["edges"][1], edges[1])
    good = ~result["mask"]
    assert np.count_nonzero(good) >= 90
    residual = (result["free_energy"] + np.log(exact_density) / a["beta"])[good]
    assert np.std(residual) < 0.08


def test_target_weights_match_independent_pymbar_and_energy_gauges(harmonic_data):
    pymbar = pytest.importorskip("pymbar", reason="independent optional reference solver")
    a = harmonic_data
    x = np.concatenate(a["samples"])
    u = umbrella_reduced_potentials(x, a["centers"], a["stiffness"], a["beta"])
    target = 0.15 * (x[:, 0] - 0.4) ** 2 + 0.05 * x[:, 1]
    fit = solve_mbar(u, a["counts"], target_u=target, tolerance=1e-11)
    reference = pymbar.MBAR(np.vstack([u, target]), np.r_[a["counts"], 0],
                            relative_tolerance=1e-12, solver_protocol="robust")
    np.testing.assert_allclose(fit.free_energies, reference.f_k[:-1], atol=2e-9, rtol=0)
    np.testing.assert_allclose(fit.weights, reference.W_nk[:, -1], rtol=2e-9, atol=1e-12)
    common = 1000 * np.sin(x[:, 0]) - 4000
    state_shifts = np.linspace(-100, 300, len(a["counts"]))
    shifted = solve_mbar(u + common + state_shifts[:, None], a["counts"],
                         target_u=target + common, tolerance=1e-10)
    assert shifted.converged
    np.testing.assert_allclose(shifted.weights, fit.weights, rtol=2e-8, atol=1e-12)
    expected = fit.free_energies + state_shifts - state_shifts[0]
    np.testing.assert_allclose(shifted.free_energies, expected, atol=2e-8, rtol=0)


def test_periodic_histogram_invariant_under_framewise_winding():
    period = 2 * np.pi
    theta = -np.pi + period * (np.arange(800) + 0.5) / 800
    # Shuffling prevents the intentionally regular coverage being mistaken
    # for a slowly relaxing time series; the weak bias is effectively flat.
    rng = np.random.default_rng(719)
    rng.shuffle(theta)
    shifted = theta + period * rng.integers(-8, 9, len(theta))
    original, original_shifted = theta.copy(), shifted.copy()
    kwargs = dict(centers=[[0]], stiffness=[[1e-12]], beta=1.0,
                  bins=20, periods=[period], bootstrap=12, seed=61)
    result = reconstruct_fes([theta[:, None]], **kwargs)
    winding = reconstruct_fes([shifted[:, None]], **kwargs)
    np.testing.assert_allclose(result["density"], 1 / period, atol=1e-11)
    for name in ("density", "free_energy", "standard_error", "statistical_inefficiency"):
        np.testing.assert_allclose(winding[name], result[name], atol=2e-12, rtol=0)
    np.testing.assert_array_equal(winding["raw_count"], result["raw_count"])
    np.testing.assert_array_equal(winding["mask"], result["mask"])
    # Analysis must not modify the caller's time streams.
    np.testing.assert_array_equal(theta, original)
    np.testing.assert_array_equal(shifted, original_shifted)
    explicit = reconstruct_fes([shifted[:, None]], ranges=[[0, period]], **kwargs)
    np.testing.assert_allclose(explicit["density"], 1 / period, atol=1e-11)


@pytest.mark.parametrize("options", [
    {"bins": [[0, 1, np.nan]]}, {"bins": [[0, 1, np.inf]]},
    {"ranges": [[0, np.inf]]}, {"ranges": [[0, 1], [0, 1]]},
    {"periods": [0]}, {"periods": [np.nan]}, {"periods": [1, 2]},
    {"periods": [1], "ranges": [[0, 2]]},
    {"centers": [[np.nan]]}, {"stiffness": [[np.inf]]},
])
def test_invalid_fes_geometry_is_rejected(options):
    kwargs = dict(centers=[[0.5]], stiffness=[[1]], beta=1.0, bootstrap=0)
    kwargs.update(options)
    with pytest.raises(ValueError):
        reconstruct_fes([np.linspace(0, 1, 40)[:, None]], **kwargs)


def test_bootstrap_never_gives_an_error_for_originally_masked_bin():
    rng = np.random.default_rng(3)
    x = np.r_[rng.uniform(0, 1, 500), rng.uniform(1, 2, 4), rng.uniform(2, 3, 500)]
    rng.shuffle(x)
    result = reconstruct_fes([x[:, None]], [[1.5]], [[1e-12]], 1,
                             bins=[[0, 1, 2, 3]], bootstrap=30, min_count=5)
    assert result["mask"][1] and np.isnan(result["free_energy"][1])
    assert np.isnan(result["standard_error"][1])
    assert result["bootstrap_valid_per_bin"][1] == 0
    assert result["bootstrap_attempted"] == 30
    assert len(result["bootstrap_diagnostics"]) == 30


def test_sparse_reference_records_failed_draws_instead_of_silently_dropping_them():
    # Narrow first bin has the highest density and becomes the fixed gauge,
    # but its one observed frame is absent from many resamples.
    x = np.r_[0.00005, np.linspace(0.02, 0.99, 1000)]
    np.random.default_rng(17).shuffle(x)
    result = reconstruct_fes([x[:, None]], [[0.5]], [[1e-12]], 1,
                             bins=[[0, 0.0001, 0.5, 1]], bootstrap=40, seed=29)
    failed = [record for record in result["bootstrap_diagnostics"] if not record["valid"]]
    assert result["bootstrap_attempted"] == 40
    assert 0 < len(failed) < 40
    assert all(record["reason"] == "reference_bin_masked" for record in failed)
    assert len(failed) + result["bootstrap_replicates"] == 40


def test_unmixed_stream_does_not_report_zero_bootstrap_uncertainty():
    result = reconstruct_fes([np.full((100, 1), 0.5)], [[0.5]], [[1]], 1,
                             bins=[[0, 0.3, 0.7, 1]], bootstrap=8)
    assert result["bootstrap_block_plan"]["status"] == "unavailable"
    assert result["bootstrap_block_plan"]["block_lengths"].tolist() == [50]
    assert result["bootstrap_block_plan"]["block_length_capped"].tolist() == [True]
    assert np.isnan(result["standard_error"]).all()
    assert result["effective_samples_per_stream"].tolist() == [1.0]


def test_connected_windows_do_not_interpolate_a_hole_in_2d():
    rng = np.random.default_rng(310)
    x = rng.uniform(-2, 2, (10000, 2))
    x = x[np.linalg.norm(x, axis=1) > 0.8]
    result = reconstruct_fes([x], [[0, 0]], np.eye(2) * 1e-12, 1,
                             bins=[8, 8], ranges=[[-2, 2], [-2, 2]], bootstrap=0)
    assert result["connected"]
    assert result["mask"][3:5, 3:5].all()
    assert np.isnan(result["free_energy"][3:5, 3:5]).all()


@pytest.fixture(scope="module")
def sampling_harmonic_library(tmp_path_factory):
    from pyrpmd.dynamics import native_capabilities

    compiler = shutil.which("cc")
    if compiler is None or not native_capabilities()["available"]:
        pytest.skip("native extension and C compiler required for sampling oracle")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path_factory.mktemp("pmf-sampling") / "harmonic.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-std=c11",
                    "-I", str(root / "native/include"),
                    str(root / "native/examples/harmonic_pes.c"), "-o", str(library)],
                   check=True, capture_output=True)
    return library


@pytest.mark.parametrize("beads", [1, 4, 16])
def test_actual_2d_umbrella_stream_has_centroid_boltzmann_distribution(
    sampling_harmonic_library, beads, tmp_path,
):
    from pyrpmd.system import System
    from pyrpmd.workflows.fes import run_fes

    beta, dt = 1.4, 0.06
    stiffness = np.array([[1.3, 0.35], [0.35, 2.0]])
    center = np.array([0.6, -0.9])
    system = System(("H",), np.array([2.3]), np.zeros((1, 3)))
    config = {
        "task": "fes", "beta": beta, "beads": beads, "dt_au": dt,
        "native": "required", "seed": 291,
        "potential": {"provider": "cabi", "library": str(sampling_harmonic_library)},
        "dynamics": {"loop_backend": "native", "block_steps": 256},
        "sampling": {"equilibration_steps": 1200, "production_steps": 60000,
                     "sample_interval": 30, "andersen_interval": 30, "replicas": 1},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                for axis in ("x", "y")],
        "umbrella": {"centers": [center.tolist()], "stiffness": stiffness.tolist()},
        "analysis": {"bins": [10, 12], "bootstrap": 0},
        "output": {"checkpoint_steps": 61200},
    }
    result = run_fes(config, system, tmp_path, return_data=True)
    samples = result["streams"][0]["samples"]
    hessian = np.eye(2) + stiffness
    expected_mean = np.linalg.solve(hessian, stiffness @ center)
    expected_covariance = np.linalg.inv(beta * hessian)
    assert len(samples) == 2000
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=0.045, rtol=0)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), expected_covariance,
                               atol=0.035, rtol=0)
    assert result["summary"]["converged"] is False
