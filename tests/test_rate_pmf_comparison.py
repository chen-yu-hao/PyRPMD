"""Same-stream estimator diagnostics cannot promote or overwrite a rate."""
import json

import h5py
import numpy as np
import pytest

from benchmarks.h3.rate_pmf_comparison import compare_run, compare_samples


def sample_problem():
    rng = np.random.default_rng(78162)
    centers = np.array([-.75, 0., .8])
    stiffness = np.array([3., 6., 4.])
    beta = 1.2
    grid = np.linspace(-1.5, 1.5, 31)
    target = np.exp(-grid**4)
    biased = target[None] * np.exp(-.5 * beta * stiffness[:, None] * (grid[None] - centers[:, None])**2)
    biased /= biased.sum(axis=1)[:, None]
    hist = [rng.multinomial(n, p) for n, p in zip((2400, 3200, 2800), biased)]
    samples = [np.repeat(grid, h) for h in hist]
    return samples, centers, stiffness, beta


def test_wham_matches_unbinned_mbar_on_bin_centers_and_reuses_one_fit(monkeypatch):
    import benchmarks.h3.rate_pmf_comparison as tool

    original = tool.solve_mbar
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(tool, "solve_mbar", counted)
    samples, centers, stiffness, beta = sample_problem()
    result = compare_samples(samples, centers, stiffness, beta, ranges=[-1.55, 1.55],
                             xi_star=.5, bins=[31, 62], ui_points=[101, 201])
    assert len(calls) == 1
    assert result["role"] == "diagnostic_only" and result["rate"] is None
    assert "ready" not in result
    assert result["mbar"][0]["supported"] and result["wham"][0]["supported"]
    assert result["wham"][0]["wham_minus_mbar_hartree"] == pytest.approx(0., abs=2e-9)
    np.testing.assert_allclose(np.asarray(result["wham"][0]["free_energy_hartree"], float),
                               np.asarray(result["mbar"][0]["free_energy_hartree"], float), atol=2e-9)
    assert result["mbar"][0]["ess"]["global_kish_ess"] > 1
    assert result["wham"][0]["ess"]["global_kish_ess"] == pytest.approx(
        result["mbar"][0]["ess"]["global_kish_ess"], rel=2e-9)
    assert [r["points"] for r in result["ui"]] == [101, 201]
    assert result["ui"][0]["ess"]["global_kish_ess"] is None
    json.dumps(result, allow_nan=False)


def test_histogram_fit_retains_tails_and_does_not_clip_unsupported_endpoints():
    samples, centers, stiffness, beta = sample_problem()
    result = compare_samples(samples, centers, stiffness, beta, ranges=[-.5, .6],
                             xi_star=.6, bins=[11, 22], ui_points=[11, 21])
    for wham in result["wham"]:
        assert wham["fit_samples"] == sum(map(len, samples))
        assert wham["fit_range"][0] <= min(a.min() for a in samples)
        assert wham["fit_range"][1] >= max(a.max() for a in samples)
    for estimator in ("mbar", "wham", "ui"):
        for curve in result[estimator]:
            assert not curve["supported"]
            assert curve["diagnostic_barrier_hartree"] is None
            support = curve["empirical_support"] if estimator == "ui" else curve["support"]
            assert support["dividing_surface"]["status"] == "outside_histogram_centers"


def test_gaussian_ui_recovers_an_independent_harmonic_barrier():
    from scipy.special import ndtri

    beta, physical_k, umbrella_k = 2.0, .7, 4.0
    centers = np.array([-1., 0., 1.])
    normal = ndtri((np.arange(2000) + .5) / 2000)
    samples = [umbrella_k * center / (physical_k + umbrella_k)
               + normal / np.sqrt(beta * (physical_k + umbrella_k)) for center in centers]
    result = compare_samples(samples, centers, umbrella_k, beta, ranges=[-2., 2.],
                             xi_star=.5, bins=[20], ui_points=[101, 201])
    for curve in result["ui"]:
        assert curve["supported"]
        assert curve["diagnostic_barrier_hartree"] == pytest.approx(.5 * physical_k * .5**2, abs=3e-4)


def saved_run(tmp_path, ready=False):
    samples, centers, stiffness, beta = sample_problem()
    root = tmp_path / "run"
    (root / "umbrella").mkdir(parents=True)
    (root / "input").mkdir()
    (root / "input/resolved.json").write_text('{"preserve":true}\n')
    (root / "rate_pmf.json").write_text(json.dumps({"ready": ready, "xi_star": .5,
                                                   "reference": 0.0, "k_qtst_au": None}))
    with h5py.File(root / "umbrella/fes.h5", "w") as saved:
        saved.attrs.update(measure="rate_flux", beta=beta)
        saved["edges_0"] = np.linspace(-1.55, 1.55, 32)
        for index, (sample, center, kappa) in enumerate(zip(samples, centers, stiffness)):
            group = saved.create_group(f"streams/{index}")
            group.attrs.update(window=index, replica=0)
            group["samples"] = sample[:, None]
            group["center"] = [center]
            group["stiffness"] = [[kappa]]
    return root


@pytest.mark.parametrize("ready", [False, True])
def test_readonly_run_keeps_original_samples_and_rate_decision(tmp_path, ready):
    root = saved_run(tmp_path, ready)
    originals = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = compare_run(root, bins=[31], ui_points=[101])
    assert result["source_ready_unchanged"] is ready
    assert result["source_pmf"]["ready"] is ready
    assert result["rate"] is None
    assert all(p.read_bytes() == contents for p, contents in originals.items())
    assert json.loads((root / "rate_pmf_comparison.json").read_text()) == result


@pytest.mark.parametrize("target", ["rate_pmf.json", "umbrella/fes.h5", "input/resolved.json"])
def test_comparison_cannot_overwrite_any_input_artifact(tmp_path, target):
    root = saved_run(tmp_path)
    before = (root / target).read_bytes()
    with pytest.raises(FileExistsError, match="new file"):
        compare_run(root, output=root / target)
    assert (root / target).read_bytes() == before


def test_undefined_surface_requires_explicit_diagnostic_choice(tmp_path):
    root = saved_run(tmp_path)
    pmf = root / "rate_pmf.json"
    pmf.write_text('{"ready":false,"xi_star":null}')
    with pytest.raises(ValueError, match="explicit diagnostic"):
        compare_run(root)
    result = compare_run(root, xi_star=.5, bins=[31], ui_points=[101])
    assert result["source_pmf"]["xi_star"] is None
    assert result["xi_star"] == .5 and result["rate"] is None


def test_non_rate_flux_measure_is_not_silently_reinterpreted(tmp_path):
    root = saved_run(tmp_path)
    with h5py.File(root / "umbrella/fes.h5", "r+") as saved:
        saved.attrs["measure"] = "cv_density"
    with pytest.raises(ValueError, match="rate_flux"):
        compare_run(root)
