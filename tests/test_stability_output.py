"""Public outputs use the registry, and a research run warns once on root."""
import json
import warnings

from pyrpmd.cli import main
from pyrpmd.input import JobSpec
from pyrpmd.runner import prepare, run_resolved
from pyrpmd.scaffold import init_project


def test_info_validate_and_prepare_expose_the_same_stability_registry(tmp_path, capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "path_sampling" in info["workflows"]
    starter = init_project(tmp_path / "project")
    assert main(["validate", starter["input"]]) == 0
    checked = json.loads(capsys.readouterr().out)
    prepared = prepare(starter["input"])
    assert checked["stability"] == prepared["stability"]
    assert checked["stability"]["level"] == info["stability"]["tasks"]["fes"]["level"]


def test_research_warning_is_once_per_run_and_summary_is_classified(tmp_path, monkeypatch):
    starter = init_project(tmp_path / "project")
    config, system = JobSpec.from_file(starter["input"]).resolve()
    config["task"] = "path_sampling"
    config.pop("fes", None)
    from pyrpmd.workflows import path_sampling
    monkeypatch.setattr(path_sampling, "resolve_path_sampling", lambda *args, **kwargs: None)
    monkeypatch.setattr(path_sampling, "run_path_sampling", lambda *args, **kwargs: {"task": "path_sampling"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_resolved(config, system)
    assert sum("Research-level workflow" in str(item.message) for item in caught) == 1
    assert result["stability"]["level"] == "research"
    assert json.loads((tmp_path / "project/run/results/summary.json").read_text())["stability"] == result["stability"]
