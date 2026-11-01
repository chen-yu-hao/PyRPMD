"""Independent physics and reproducibility checks for the paper benchmark."""

from dataclasses import asdict, replace
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pyrpmd.nonadiabatic import NRPMDState
from pyrpmd.thermal_nrpmd import ThermalEnsemble, ThermalNRPMDDynamics, ThermalNRPMDSampler


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1]/"examples/nonadiabatic/published_nrpmd_benchmark.py"
    spec = importlib.util.spec_from_file_location("published_nrpmd_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_full_matrix_gradient_and_lowest_adiabatic_partition(benchmark):
    model = benchmark.PaperTwoStateModel(alpha=.4, kappa=.8, delta=1.2)
    q = np.array([[[-1.3, .7, -.2]], [[.6, -.5, .3]]])
    result, split = model.evaluate(q), model.lowest_adiabatic_split(q)
    np.testing.assert_allclose(split.scalar[:, None, None]*np.eye(2)+split.mapping.matrix, result.matrix)
    np.testing.assert_allclose(np.linalg.eigvalsh(split.mapping.matrix)[:, 0], 0., atol=1e-14)
    for axis in range(3):
        plus, minus = q.copy(), q.copy()
        plus[:, 0, axis] += 1e-6
        minus[:, 0, axis] -= 1e-6
        finite_difference = (model.evaluate(plus).matrix-model.evaluate(minus).matrix)/2e-6
        np.testing.assert_allclose(finite_difference, result.gradients[:, :, :, 0, axis], atol=2e-9)
        low_difference = (model.lowest_adiabatic_split(plus).scalar-model.lowest_adiabatic_split(minus).scalar)/2e-6
        np.testing.assert_allclose(low_difference, split.scalar_gradient[:, 0, axis], atol=2e-9)


def test_paper_split_hamiltonian_trace_correction_and_force(benchmark):
    model = benchmark.PaperTwoStateModel()
    rng = np.random.default_rng(109)
    q = rng.normal(scale=.4, size=(3, 1, 3))
    state = NRPMDState(q, rng.normal(size=q.shape),
                      mapping_q=rng.normal(size=(3, 2)), mapping_p=rng.normal(size=(3, 2)))
    dt = 1e-7
    paper = benchmark.Paper2013Dynamics(model, [1.], 1., 3, dt, 2, native=False)
    trace = ThermalNRPMDDynamics(model, [1.], 1., 3, dt, 2, native=False)
    radius = np.hypot(q[:, 0, 0], 1.)
    expected_difference = np.sum(radius*(.5*np.sum(state.mapping_q**2+state.mapping_p**2, axis=1)-2))
    assert paper.energy(state)-trace.energy(state) == pytest.approx(expected_difference, abs=2e-14)
    force = np.empty_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = state.copy(), state.copy()
        plus.q[index] += 1e-6
        minus.q[index] -= 1e-6
        force[index] = -(paper.energy(plus)-paper.energy(minus))/2e-6
    moved = state.copy()
    paper.step(moved)
    np.testing.assert_allclose((moved.p-state.p)/dt, force, atol=5e-6, rtol=2e-6)
    mapping = model.lowest_adiabatic_split(state.q).mapping.matrix
    np.testing.assert_allclose((moved.mapping_q-state.mapping_q)/dt,
                               np.einsum("bij,bj->bi", mapping, state.mapping_p), atol=4e-6)
    np.testing.assert_allclose((moved.mapping_p-state.mapping_p)/dt,
                               -np.einsum("bij,bj->bi", mapping, state.mapping_q), atol=4e-6)


def test_dvr_kubo_matches_separable_harmonic_and_analytic_two_level(benchmark):
    times = np.linspace(0., 5., 13)
    beta, delta = 1.3, .8
    reference = benchmark.exact_kubo(times, kappa=0., delta=delta, beta=beta, points=81, extent=8.)
    np.testing.assert_allclose(reference["Cxx"], np.cos(times)/beta, atol=2e-10)
    expected = .25*(1.+np.tanh(beta*delta)/(beta*delta)*np.cos(2*delta*times))
    np.testing.assert_allclose(reference["C11"], expected, atol=2e-10)
    assert reference["population_1"] == pytest.approx(.5, abs=1e-13)
    degenerate = benchmark.exact_kubo(times, kappa=0., delta=0., points=65, extent=8.)
    np.testing.assert_allclose(degenerate["C11"], .5, atol=2e-10)


def test_paper_dvr_domain_resolution_and_time_reversal(benchmark):
    times = np.array([0., .4, 1., 2., 5.])
    reference = benchmark.exact_kubo(times, points=81, extent=8.)
    refined = benchmark.exact_kubo(-times, points=101, extent=9.)
    for name in ("Cxx", "C11"):
        np.testing.assert_allclose(reference[name], refined[name], atol=2e-9)
    # These relations test distinct operators: x centroid Kubo variance is
    # not the ordinary bead x^2 thermal variance, nor is P1^2(0) simply 1/2.
    assert reference["Cxx"][0] > 1.5
    assert .4 < reference["C11"][0] < .5


def test_correlations_use_seo_at_initial_time_and_retain_same_parents(benchmark):
    model = benchmark.PaperTwoStateModel()
    sampler = ThermalNRPMDSampler(model, [1.], 1., np.zeros((2, 1, 3)), 2, seed=718)
    parents = sampler.sample(4, burn_in=8)
    saved = [state.copy() for state in parents.states]
    coarse = benchmark.propagate_correlations(parents, dt=.02, duration=.2, output_interval=.1)
    fine = benchmark.propagate_correlations(parents, dt=.01, duration=.2, output_interval=.1)
    expected_x = np.array([state.q[:, 0, 0].mean()**2 for state in parents.states])
    expected_electronic = parents.electronic_populations[:, 0]*np.array([state.populations()[0] for state in parents.states])
    np.testing.assert_array_equal(coarse["numerators"][:, 0, 0], expected_x)
    np.testing.assert_array_equal(coarse["numerators"][:, 0, 1], expected_electronic)
    np.testing.assert_array_equal(coarse["numerators"][:, 0], fine["numerators"][:, 0])
    np.testing.assert_array_equal(coarse["time"], fine["time"])
    for state, previous in zip(parents.states, saved):
        for name in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_array_equal(getattr(state, name), getattr(previous, name))
    assert fine["maximum_energy_drift_per_parent"].max() < .35*coarse["maximum_energy_drift_per_parent"].max()


def test_source_chain_bootstrap_joint_signed_ratio_and_insufficient_blocks(benchmark):
    rng = np.random.default_rng(843)
    signs = np.where(rng.uniform(size=(3, 512)) < .2, -1., 1.)
    values = rng.normal(size=(3, 512, 4, 2))
    result = benchmark.correlation_bootstrap(values, signs, replicates=64, seed=12)
    expected = np.mean(values*signs[:, :, None, None], axis=(0, 1))/signs.mean()
    np.testing.assert_allclose(result["estimate"], expected)
    assert result["confidence_intervals_available"]
    assert result["valid_replicates"] == 64
    repeated = benchmark.correlation_bootstrap(values, signs, replicates=64, seed=12)
    assert repeated == result
    short = benchmark.correlation_bootstrap(values[:, :8], signs[:, :8], replicates=8, seed=12)
    assert not short["confidence_intervals_available"]
    assert short["standard_error"] is None
    assert "fewer_than_eight_complete_blocks_in_a_source_chain" in short["unavailable_reasons"]
    unresolved = benchmark.correlation_bootstrap(values[:, :8], -np.ones((3, 8)), replicates=32)
    assert unresolved["estimate"] is None
    assert unresolved["valid_replicates"] == 0
    assert unresolved["percentile_interval"] is None


def test_bootstrap_keeps_independent_chain_disagreement(benchmark):
    # Even with no within-chain variation, disagreement between independent
    # chains must survive the hierarchical resampling.
    values = np.broadcast_to(np.arange(3.)[:, None, None, None], (3, 128, 2, 1)).copy()
    result = benchmark.correlation_bootstrap(values, np.ones((3, 128)), replicates=128, seed=0)
    assert result["confidence_intervals_available"]
    assert np.asarray(result["standard_error"]).min() > .2


@pytest.mark.parametrize("change", [
    {"beads": ()}, {"beads": (1, 1)}, {"beads": (1.5,)}, {"chains": True},
    {"samples": 0}, {"dt": float("nan")}, {"duration": -.1}, {"stride": 0},
    {"dt": .03}, {"potential_split": "unknown"}, {"min_average_sign": 1.1},
])
def test_benchmark_config_rejects_invalid_values(benchmark, change):
    with pytest.raises(ValueError):
        benchmark.BenchmarkConfig(**change)


def test_chain_resume_and_raw_ancestry_are_exact_and_content_checked(benchmark, tmp_path):
    config = benchmark.BenchmarkConfig(beads=(1,), chains=2, samples=4, burn_in=2,
        stride=1, dt=.02, duration=.1, output_interval=.1, bootstrap_replicates=8, refinement_parents=2)
    report = benchmark._run_chain(asdict(config), 1, 0, tmp_path)
    resumed = benchmark._run_chain(asdict(config), 1, 0, tmp_path, resume=True)
    assert resumed == json.loads(json.dumps(benchmark._jsonable(report)))
    with np.load(tmp_path/report["arrays"], allow_pickle=False) as arrays:
        assert arrays["initial_q"].shape == (4, 1, 1, 3)
        assert arrays["initial_mapping_q"].shape == (4, 1, 2)
        np.testing.assert_array_equal(arrays["refinement_source_indices"], [0, 1])
        np.testing.assert_array_equal(arrays["numerators"][:2, 0], arrays["refined_numerators"][:, 0])
    with pytest.raises(ValueError, match="identical"):
        benchmark._run_chain(asdict(replace(config, seed=2)), 1, 0, tmp_path, resume=True)
    with (tmp_path/report["arrays"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="modified"):
        benchmark._run_chain(asdict(config), 1, 0, tmp_path, resume=True)


def test_digitized_data_comparison_keeps_provenance_and_separate_errors(benchmark, tmp_path):
    source = tmp_path/"figure.csv"
    source.write_text("observable,series,time,value,absolute_uncertainty\nCxx,dvr,.3,1.8,.01\nCxx,P1,0,1.9,.02\n")
    reference = benchmark.exact_kubo([0., 1.], points=65, extent=8.)
    result = {"beads": 1, "time": [0., 1.], "statistics": {
        "estimate": [[1.92, .3], [1., .2]], "confidence_intervals_available": True,
        "standard_error": [[.05, .02], [.05, .02]]}}
    compared = benchmark.compare_published_data(source, reference, [result])
    assert len(compared["points"]) == 2
    expected = benchmark.exact_kubo([.3], points=65, extent=8.)["Cxx"][0]
    assert compared["points"][0]["computed_value"] == pytest.approx(expected, abs=1e-13)
    assert abs(expected-np.interp(.3, reference["time"], reference["Cxx"])) > 1e-3
    assert compared["points"][0]["computed_curve_interpolation"] == "none_spectral_sum_at_published_time"
    assert compared["points"][1]["computed_bootstrap_standard_error"] == .05
    assert compared["points"][1]["digitization_absolute_uncertainty"] == .02
    assert len(compared["source_csv_sha256"]) == 64


def test_published_vector_dataset_has_verifiable_provenance_and_original_times(benchmark):
    root = Path(__file__).resolve().parents[1]
    path = root/"examples/nonadiabatic/data/richardson_thoss_2013_fig1c.csv"
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["csv_sha256"]
    extractor = root/"examples/nonadiabatic/extract_published_nrpmd_figure.py"
    assert hashlib.sha256(extractor.read_bytes()).hexdigest() == provenance["extraction_script_sha256"]
    assert provenance["publication"]["doi"] == benchmark.PAPER["doi"]
    assert provenance["numerical_reference_used_for_calibration"] is False
    assert provenance["original_pdf_or_image_redistributed"] is False
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == provenance["rows"] == 166
    for observable in ("Cxx", "C11"):
        for series in ("dvr", "P1", "P4", "P8"):
            group = [row for row in rows if row["observable"] == observable and row["series"] == series]
            assert len(group) == (20 if series == "dvr" else 21)
            times = np.array([float(row["time"]) for row in group])
            assert np.all(np.diff(times) > 0)
            assert abs(times[0]) < 1e-4 and abs(times[-1]-5) < 1e-4
            assert all(float(row["absolute_uncertainty"]) > 0 for row in group)
    # An external numerical check: the independent spectral reference agrees
    # with the actual paper's black DVR markers, calibrated from axis ticks.
    reference = benchmark.exact_kubo([0., 5.], points=81, extent=8.)
    compared = benchmark.compare_published_data(path, reference, [])
    assert len(compared["points"]) == 40
    assert all(item["maximum_absolute_difference"] < 2e-4 for item in compared["summary"])
