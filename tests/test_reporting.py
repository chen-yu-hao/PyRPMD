"""Saved-data reports must preserve scientific meaning and source files."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.reporting import collect_run, generate_report
from pyrpmd.analysis.report_compare import collect_comparisons, fit_arrhenius, read_reference
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import AU_ENERGY_KCAL_MOL, AU_TIME_FS, BOHR_ANGSTROM
from pyrpmd.cli import main


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _rate(tmp_path, *, name="rate", temperature=300, rate=2e-12, estimate=4e-12,
          mass=1837.0, beads=16, task="rate"):
    path = tmp_path / name
    system = {"symbols": ["H"] * 3, "masses": [mass, 1837., 1837.],
              "geometry": [[0., 0., -1.4], [0., 0., 0.], [0., 0., 1.4]]}
    config = {"task": task, "temperature_K": temperature, "beads": beads,
              "dt_au": 0.1 / AU_TIME_FS, "potential": {"provider": "harmonic", "params": {"k": 0.002}},
              "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                           "breaking_bonds": [[1, 2]], "rinf_au": 12.},
              "dynamics": {"integrator": "cayley-bcb"}}
    _json(path / "input/resolved.json", config)
    _json(path / "input/system.json", system)
    suffix = "cm3_s" if task == "rate" else "s"
    summary = {"task": task, "status": "pilot", "converged": False,
               "rate_" + suffix: rate, "rate_estimate_" + suffix: estimate,
               "rate_standard_error_" + suffix: None, "kappa": 0.4,
               "plateau_detected": rate is not None, "plateau_start_au": 2.0,
               "pmf": {"xi_star": 0.5, "k_qtst_" + suffix: 1e-11, "measure": "rate_flux"}, "warnings": []}
    _json(path / "results/summary.json", summary)
    atomic_savez(path / "rate_pmf.npz", xi=[0., 0.5, 1.], free_energy=[0., 0.02, 0.01],
                 standard_error=[0.001, np.nan, 0.002], mask=[False, False, True])
    atomic_savez(path / "recrossing.npz", time_au=[1., 2., 3.], kappa=[1., 0.6, -0.1],
                 standard_error=[0.1, np.nan, 0.2])
    return path


def _fes(tmp_path):
    path = tmp_path / "fes"
    config = {"task": "fes", "temperature_K": 300, "beads": 4,
              "cvs": [{"name": "r12", "kind": "distance", "atoms": [0, 1]},
                      {"name": "angle", "kind": "angle", "atoms": [0, 1, 2]}],
              "umbrella": {"cvs": ["r12", "angle"]}, "potential": {"provider": "harmonic"}}
    _json(path / "input/resolved.json", config)
    _json(path / "input/system.json", {"symbols": ["H", "H", "H"], "masses": [1837.] * 3})
    _json(path / "summary.json", {"task": "fes", "units": ["bohr", "radian"],
          "status": "sampled", "measure": "cv_density", "converged": False, "samples": 10})
    atomic_savez(path / "fes.npz", free_energy=[[0., 1., 2.], [3., 4., 5.]],
                 mask=[[False, True, False], [False, False, False]], raw_count=[[1, 0, 2], [2, 2, 3]],
                 standard_error=np.full((2, 3), np.nan), edges_0=[1., 2., 4.], edges_1=[0., 1., 2., 3.],
                 overlap=np.eye(2), effective_samples_per_stream=[3., 2.])
    return path


def test_units_masks_and_signed_transmission(tmp_path):
    path = _rate(tmp_path, rate=None)
    run = collect_run(path)
    pmf = next(p for p in run["_plots"] if p["id"].endswith("-pmf"))
    curve = pmf["series"][0]
    assert curve["y"][1] == pytest.approx(0.02 * AU_ENERGY_KCAL_MOL)
    assert np.isnan(curve["y"][2]) and np.isnan(curve["yerr"][1])
    kappa = next(p for p in run["_plots"] if p["id"].endswith("-recrossing"))["series"][0]
    np.testing.assert_allclose(kappa["x"], np.arange(1, 4) * AU_TIME_FS)
    assert kappa["y"][-1] == -0.1
    assert run["metrics"]["rate"] is None
    assert any(f["code"] == "rate_withheld" for f in run["findings"])
    comparisons = collect_comparisons([run])
    assert not any(p["id"].startswith("comparison-rates") for p in comparisons["plots"])
    candidates = collect_comparisons([run], include_estimates=True)
    assert "withheld diagnostic" in candidates["plots"][-1]["series"][0]["label"]


def test_2d_cv_units_are_derived_from_saved_definitions(tmp_path):
    run = collect_run(_fes(tmp_path))
    plot = next(p for p in run["_plots"] if p["id"].endswith("-fes"))
    np.testing.assert_allclose(plot["xedges"], np.array([1., 2., 4.]) * BOHR_ANGSTROM)
    np.testing.assert_allclose(plot["yedges"], [0., 1., 2., 3.])
    assert "H1-H2" in plot["xlabel"] and "radian" in plot["ylabel"]
    assert plot["z"].shape == (2, 3) and plot["mask"][0, 1]
    assert not any(p["id"].endswith("-error") for p in run["_plots"])


def test_arrhenius_fit_recovers_known_activation_energy():
    from scipy.constants import R
    temperature = np.array([250., 350., 500., 800.])
    rate = 7e-10 * np.exp(-12500 / (R * temperature))
    fit = fit_arrhenius(temperature, rate)
    assert fit["activation_energy_kJ_mol"] == pytest.approx(12.5, rel=1e-11)
    assert fit["A"] == pytest.approx(7e-10, rel=1e-11)
    assert fit["r_squared_log"] == pytest.approx(1)
    with pytest.raises(ValueError, match="Repeated"):
        fit_arrhenius([300, 300, 400], [1., 2., 3.])


def test_kie_matches_temperatures_and_requires_isotope_change(tmp_path):
    h = collect_run(_rate(tmp_path, name="H", rate=6., mass=1837.), label="H")
    d = collect_run(_rate(tmp_path, name="D", rate=2., mass=3674.), label="D")
    h["_rate"]["standard_error"] = 0.6
    d["_rate"]["standard_error"] = 0.4
    result = collect_comparisons([h, d], series=["H", "D"], kie=["H", "D"])
    ratio = result["metrics"]["kinetic_isotope_effect"]["data"][0]
    assert ratio["ratio"] == 3
    assert ratio["standard_error"] == pytest.approx(3 * np.sqrt(0.01 + 0.04))
    d["metadata"]["temperature_K"] = 400
    result = collect_comparisons([h, d], series=["H", "D"], kie=["H", "D"])
    assert result["metrics"]["kinetic_isotope_effect"]["data"] == []
    d["metadata"]["temperature_K"] = 300
    d["_system"]["masses"] = h["_system"]["masses"]
    with pytest.raises(ValueError, match="isotope mass change"):
        collect_comparisons([h, d], series=["H", "D"], kie=["H", "D"])


def test_reference_units_and_source_are_required(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("series,temperature_K,rate,rate_unit,source\nexperiment,300,1e-12,cm3 molecule^-1 s^-1,doi:example\n")
    rows = read_reference(path)
    assert rows[0]["standard_error"] is None
    path.write_text("series,temperature_K,rate,rate_unit,source\nexperiment,300,1e-12,unknown,\n")
    with pytest.raises(ValueError, match="canonical rate_unit"):
        read_reference(path)


def test_broken_optional_arrays_are_reported_and_not_silently_accepted(tmp_path):
    path = _rate(tmp_path)
    (path / "rate_pmf.npz").write_bytes(b"not a valid archive")
    run = collect_run(path)
    assert any(s["figure"] == "rate PMF" for s in run["skipped"])
    assert any(p["id"].endswith("-recrossing") for p in run["_plots"])


def test_report_is_offline_strict_json_and_does_not_change_sources(tmp_path):
    pytest.importorskip("matplotlib")
    source = _rate(tmp_path)
    paths = [p for p in source.rglob("*") if p.is_file()]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = generate_report(source, formats=["png"], dpi=100)
    out = Path(report["output"])
    assert (out / "index.html").exists() and (out / "summary.csv").exists()
    saved = json.loads((out / "report.json").read_text(), parse_constant=lambda v: pytest.fail(v))
    assert saved["runs"][0]["converged"] is False
    assert len(saved["figures"]) >= 2
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert all((out / f).exists() for f in saved["generated_files"])
    with pytest.raises(ValueError, match="not empty"):
        generate_report(source, formats=["png"], dpi=100)
    generate_report(source, formats=["png"], dpi=100, overwrite=True)
    (out / "my-notes.txt").write_text("Keep this")
    with pytest.raises(ValueError, match="outside its generated inventory"):
        generate_report(source, formats=["png"], dpi=100, overwrite=True)
    assert (out / "my-notes.txt").read_text() == "Keep this"


def test_cli_report_and_analyze_shortcut(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    source = _rate(tmp_path)
    out = tmp_path / "report-cli"
    assert main(["report", str(source), "--output", str(out), "--formats", "svg", "--style", "nature"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["figures"] >= 2 and Path(data["html"]).exists()
    assert json.loads((out / "report.json").read_text())["settings"]["style"] == "nature"
    out2 = tmp_path / "report-analyze"
    assert main(["analyze", str(source), "--report", "--report-output", str(out2), "--report-style", "nature"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "analysis_report" in data and data["rate_cm3_s"] == 2e-12
    assert json.loads((out2 / "report.json").read_text())["settings"]["style"] == "nature"


def test_run_shortcut_with_actual_harmonic_workflow(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    source = tmp_path / "actual"
    example = Path(__file__).resolve().parents[1] / "examples/manual/quickstart.inp"
    assert main(["run", str(example), "--output", str(source), "--report", "--report-style", "nature"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task"] == "fes" and result["analysis_report"]["figures"] >= 3
    assert json.loads((source / "analysis-report/report.json").read_text())["settings"]["style"] == "nature"


@pytest.mark.parametrize("command,completed", [("run", "Simulation"), ("analyze", "Analysis")])
def test_missing_plot_dependency_after_science_reports_saved_result(tmp_path, capsys, monkeypatch, command, completed):
    import pyrpmd.cli as cli
    import pyrpmd.runner as runner
    import pyrpmd.analysis.reporting as reporting

    path = _rate(tmp_path)
    summary = path / "results/summary.json"
    before = summary.read_bytes()
    calls = []

    def finish(*args, **kwargs):
        calls.append(1)
        return {"task": "rate", "output": str(path), "rate_cm3_s": 2e-12}

    def missing(*args, **kwargs):
        raise ImportError("Figure generation requires Matplotlib")

    monkeypatch.setattr(runner, "run" if command == "run" else "analyze_run", finish)
    monkeypatch.setattr(cli, "_job", lambda args: object())
    monkeypatch.setattr(reporting, "generate_report", missing)
    assert main([command, str(path), "--report"]) == 2
    error = capsys.readouterr().err
    assert f"{completed} completed" in error
    assert str(path) in error and "pyrpmd report" in error and "Matplotlib" in error
    assert calls == [1] and summary.read_bytes() == before


def test_existing_report_output_failure_preserves_analysis_and_user_files(tmp_path, capsys):
    path = _rate(tmp_path)
    target = tmp_path / "occupied-report"
    target.mkdir()
    (target / "notes.txt").write_text("Keep these notes")
    source = path / "results/summary.json"
    before = source.read_bytes()
    assert main(["analyze", str(path), "--report", "--report-output", str(target)]) == 2
    error = capsys.readouterr().err
    assert "Analysis completed" in error and "not empty" in error and "pyrpmd report" in error
    assert source.read_bytes() == before
    assert (target / "notes.txt").read_text() == "Keep these notes"
