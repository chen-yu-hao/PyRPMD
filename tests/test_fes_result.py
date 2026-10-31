"""Common saved surfaces preserve numerics, provenance and uncertainty meaning."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.fes_result import (
    FESResult, estimate_fes_uncertainty, write_fes_result,
)
from pyrpmd.analysis.reporting import collect_run
from pyrpmd.analysis.report_compare import collect_comparisons
from pyrpmd.analysis.projection import project_run
from pyrpmd.checkpoint import atomic_savez, load_checkpoint


def surface(method="umbrella", dimension=1):
    shape = (2,) * dimension
    free = np.arange(2 ** dimension, dtype=float).reshape(shape)
    mask = np.zeros(shape, bool)
    mask.flat[-1] = True
    data = {"free_energy": free, "edges": [np.arange(3.)] * dimension,
            "mask": mask, "raw_count": np.ones(shape, int),
            "standard_error": np.full(shape, .25),
            "replica_spread": np.full(shape, .75), "measure": "cv_density"}
    return FESResult.from_data(data, method=method, cv_units=["bohr"] * dimension,
                              metadata={"beta": 2., "estimator": "test", "converged": False})


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_round_trip_has_uniform_layout_and_preserves_arrays(tmp_path, method, dimension):
    original = surface(method, dimension)
    summary = write_fes_result(tmp_path, original)
    assert {path.name for path in tmp_path.iterdir()} == {"fes.npz", "fes.csv", "fes.json", "summary.json"}
    assert summary["task"] == "fes" and summary["method"] == method
    result = FESResult.read(tmp_path)
    assert result.method == method
    assert result.measure == "cv_density" and result.metadata["beta"] == 2.
    np.testing.assert_array_equal(result.free_energy, original.free_energy)
    np.testing.assert_array_equal(result.mask, original.mask)
    np.testing.assert_array_equal(result.grid, original.grid)
    np.testing.assert_array_equal(result.uncertainty, original.uncertainty)
    assert result.uncertainty.flat[0] == (.25 if method == "umbrella" else .75)
    assert np.isnan(result.uncertainty.flat[-1])
    fields = (tmp_path / "fes.csv").read_text().splitlines()[0].split(",")
    assert fields[dimension:] == ["free_energy_hartree", "uncertainty_hartree", "masked",
                                   "uncertainty_masked", "raw_count"]
    with load_checkpoint(tmp_path / "fes.npz") as archive:
        assert archive.envelope["kind"] == "fes_result"
        assert str(archive["uncertainty_kind"]) == result.uncertainty_kind
    assert json.loads((tmp_path / "fes.json").read_text()) == summary


@pytest.mark.parametrize("task", ["fes", "metadynamics", "opes"])
def test_historical_archives_use_one_adapter_without_relabelling_spread(tmp_path, task):
    atomic_savez(tmp_path / "fes.npz", free_energy=[0., 1.], edges_0=[0., 1., 2.],
                 standard_error=[.1, .2], replica_spread=[.3, .4], mask=[False, True])
    (tmp_path / "summary.json").write_text(json.dumps({"task": task, "units": ["bohr"], "measure": "cv_density"}))
    result = FESResult.read(tmp_path)
    assert result.method == ("umbrella" if task == "fes" else task)
    assert result.uncertainty[0] == (.1 if task == "fes" else .3)
    assert result.summary()["uncertainty_kind"] == ("block_bootstrap" if task == "fes" else "replica_spread")
    assert np.isnan(result.uncertainty[1])


@pytest.mark.parametrize("key,value,match", [
    ("measure", "rate_flux", "measure contradicts"),
    ("units", ["radian"], "CV units contradict"),
    ("method", "opes", "method contradicts"),
    ("beta", 3., "beta contradicts"),
])
def test_saved_metadata_cannot_be_overridden_by_a_conflicting_summary(tmp_path, key, value, match):
    summary = write_fes_result(tmp_path, surface())
    with pytest.raises(ValueError, match=match):
        FESResult.read(tmp_path, summary={**summary, key: value})


@pytest.mark.parametrize("legacy_task", ["metadynamics", "opes"])
def test_legacy_task_cannot_hide_a_contradictory_canonical_method(tmp_path, legacy_task):
    summary = write_fes_result(tmp_path, surface("umbrella"))
    with pytest.raises(ValueError, match="method contradicts"):
        FESResult.read(tmp_path, summary={**summary, "task": legacy_task})
    arrays = {"free_energy": np.array([0., 1.]), "edges_0": np.arange(3.)}
    with pytest.raises(ValueError, match="method contradicts"):
        FESResult.from_arrays(arrays, metadata={"task": legacy_task, "method": "umbrella"})


def test_in_memory_adapter_rejects_conflicting_method_or_measure_provenance():
    data = {"free_energy": np.array([0., 1.]), "edges": [np.arange(3.)], "measure": "cv_density"}
    with pytest.raises(ValueError, match="method contradicts"):
        FESResult.from_data(data, method="umbrella", cv_units=["bohr"], metadata={"method": "opes"})
    with pytest.raises(ValueError, match="measure contradicts"):
        FESResult.from_data(data, method="umbrella", cv_units=["bohr"], metadata={"measure": "rate_flux"})


def test_legacy_sampling_measure_is_adapted_but_conflicts_are_rejected():
    arrays = {"free_energy": np.array([0., 1.]), "edges_0": np.arange(3.),
              "sampling_measure": np.asarray("rate_flux")}
    assert FESResult.from_arrays(arrays).measure == "rate_flux"
    with pytest.raises(ValueError, match="measure contradicts"):
        FESResult.from_arrays(arrays, metadata={"measure": "cv_density"})


def test_scalar_flux_axis_survives_common_serialization(tmp_path):
    result = surface()
    result.metadata.update(measure="rate_flux", flux_axis=0)
    write_fes_result(tmp_path, result)
    with load_checkpoint(tmp_path / "fes.npz") as archive:
        assert archive["flux_axis"].item() == 0
    assert FESResult.read(tmp_path).metadata["flux_axis"] == 0


def test_replica_spread_cannot_be_relabelled_as_bootstrap_standard_error():
    arrays = surface("metadynamics").arrays()
    arrays["uncertainty_kind"] = "block_bootstrap"
    arrays.pop("fes_metadata")
    with pytest.raises(ValueError, match="uncertainty kind contradicts"):
        FESResult.from_arrays(arrays)


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_unsupported_or_noninteger_schema_versions_are_rejected(version):
    arrays = surface().arrays()
    arrays["fes_schema_version"] = version
    with pytest.raises(ValueError, match="Unsupported FESResult schema"):
        FESResult.from_arrays(arrays)


def test_report_refuses_method_mismatch_with_saved_axis_selection(tmp_path):
    write_fes_result(tmp_path, surface("umbrella"))
    (tmp_path / "input").mkdir()
    (tmp_path / "input/resolved.json").write_text(json.dumps({
        "task": "fes", "fes": {"method": "opes"}, "opes": {"cvs": ["y"]},
        "cvs": {"x": {"kind": "cartesian", "axis": 0, "atom": 0},
                "y": {"kind": "cartesian", "axis": 1, "atom": 0}}}))
    run = collect_run(tmp_path)
    assert run["_comparison_identity_available"] is False
    assert not any(plot["id"].endswith("-fes") for plot in run["_plots"])
    assert any("method contradicts" in item["reason"] for item in run["skipped"])


def test_replica_spread_keeps_sample_sd_and_shared_walkers_have_no_independent_error():
    histories = np.array([[0., 1., 2.], [0., 3., 7.], [0., 5., 8.]])
    mask = np.array([False, False, True])
    spread = estimate_fes_uncertainty(histories, kind="replica_spread", mask=mask)
    np.testing.assert_array_equal(spread[:2], histories.std(axis=0, ddof=1)[:2])
    assert np.isnan(spread[-1])
    assert np.isnan(estimate_fes_uncertainty(histories[:1], kind="replica_spread", mask=mask)).all()
    assert np.isnan(estimate_fes_uncertainty(histories, kind="replica_spread", mask=mask,
                                           independent=False)).all()


def test_block_bootstrap_errors_count_valid_draws_per_bin():
    draws = np.array([[0., 1., 3., np.nan], [0., 2., np.nan, np.nan],
                      [0., 3., np.nan, np.nan]])
    error = estimate_fes_uncertainty(draws, kind="block_bootstrap", mask=np.zeros(4, bool))
    np.testing.assert_array_equal(error[:2], [0., 1.])
    assert np.isnan(error[2:]).all()


def test_report_consumes_three_dimensional_result_as_labelled_slices(tmp_path):
    write_fes_result(tmp_path, surface("umbrella", 3))
    run = collect_run(tmp_path, energy_unit="hartree", length_unit="bohr")
    slices = [plot for plot in run["_plots"] if "-fes-slice-" in plot["id"]]
    assert len(slices) == 3
    assert all(plot["z"].shape == (2, 2) and "not a marginal" in plot["caption"] for plot in slices)
    assert run["metrics"]["fes_supported_bins"] == 7


def test_projection_uses_self_contained_fes_metadata_without_resolved_input(tmp_path):
    source, target = tmp_path / "source", tmp_path / "projected"
    result = surface("metadynamics", 2)
    result.mask[:] = False
    write_fes_result(source, result)
    summary = project_run(source, axes=[0], output=target)
    assert summary["beta"] == 2. and summary["supported_bins"] == 2
    assert (target / "pmf.npz").is_file()


def test_shared_walker_report_reads_one_global_history_without_fabricated_error(tmp_path):
    result = surface("metadynamics")
    result.uncertainty[:] = np.nan
    result.metadata.update(multiple_walkers=True, walkers=2,
                           uncertainty_note="Shared-bias walkers are correlated; independent error is unavailable.")
    write_fes_result(tmp_path, result)
    (tmp_path / "HILLS.csv").write_text(
        "step,time_au,height_hartree,bias_before_hartree,walker\n"
        "1,1,0.3,0,0\n1,1,0.4,0,1\n2,2,0.2,0.3,0\n2,2,0.1,0.4,1\n")
    atomic_savez(tmp_path / "checkpoints/metadynamics_walkers.npz",
                 samples=[[[.1], [.2]], [[.3], [.4]]], sample_steps=[1, 2])
    run = collect_run(tmp_path, energy_unit="hartree")
    heights = next(plot for plot in run["_plots"] if plot["id"].endswith("sampling-hill-height"))
    assert [series["label"] for series in heights["series"]] == ["Walker 0", "Walker 1"]
    np.testing.assert_array_equal(heights["series"][0]["y"], [.3, .2])
    traces = next(plot for plot in run["_plots"] if plot["id"].endswith("sampling-trace-1"))
    assert len(traces["series"]) == 2 and "correlated" in traces["caption"]
    assert np.isnan(next(plot for plot in run["_plots"] if plot["id"].endswith("-fes"))["series"][0]["yerr"]).all()


def test_report_and_comparison_use_same_contract_for_all_sampling_methods(tmp_path):
    runs = []
    for index, method in enumerate(("umbrella", "metadynamics", "opes")):
        directory = tmp_path / method
        (directory / "input").mkdir(parents=True)
        config = {"task": "fes", "fes": {"method": method}, method: {"cvs": ["x"]},
                  "temperature_K": 300, "beads": 1, "potential": {"provider": "harmonic"},
                  "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}]}
        (directory / "input/resolved.json").write_text(json.dumps(config))
        (directory / "input/system.json").write_text(json.dumps({"symbols": ["H"], "masses": [1837.]}))
        write_fes_result(directory, surface(method))
        run = collect_run(directory, prefix=f"run-{index}", energy_unit="hartree")
        plot = next(plot for plot in run["_plots"] if plot["id"].endswith("-fes"))
        assert plot["series"][0]["yerr"][0] == (.25 if method == "umbrella" else .75)
        assert ("1 SE" if method == "umbrella" else "1 SD") in plot["caption"]
        runs.append(run)
    compared = collect_comparisons(runs)
    overlay, = [plot for plot in compared["plots"] if plot["id"].startswith("comparison-pmf")]
    assert len(overlay["series"]) == 3
    assert all(method in series["label"] for method, series in zip(
        ("umbrella", "metadynamics", "opes"), overlay["series"]))
    assert "not a standard error" in overlay["caption"]
