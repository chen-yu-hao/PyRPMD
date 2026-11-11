"""Every main workflow can start from installed resources outside the checkout."""

import pytest

from pyrpmd.cli import parser
from pyrpmd.input import JobSpec
from pyrpmd.runner import run, prepare
from pyrpmd.scaffold import init_project


def test_help_groups_everyday_commands_before_tools():
    help_text = parser().format_help()
    assert help_text.index("Everyday:") < help_text.index("Workflow-specific:") < help_text.index("Developer tools:")
    assert help_text.index("init ") < help_text.index("import-rpmdrate ")
    assert "Quickstart:" in help_text
    assert parser().parse_args(["init", "new", "--task", "rate"]).task == "rate"


@pytest.mark.parametrize("task", ["fes", "pimd", "drpmd", "rate"])
def test_starter_is_self_contained_resolves_and_runs(tmp_path, monkeypatch, task):
    destination = tmp_path / task
    starter = init_project(destination, task=task)
    monkeypatch.chdir(tmp_path)
    resolved, _ = JobSpec.from_file(starter["input"]).resolve()
    assert resolved["task"] == task
    assert resolved["output"]["directory"] == str(destination / "run")
    prepared = prepare(starter["input"], numerical=True)
    assert prepared["ready"]
    # Keep the shipped budgets: this checks the actual first-run contract.
    result = run(starter["input"])
    assert result["task"] == task
    assert (destination / "run/results/summary.json").is_file()
    before = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="not empty"):
        init_project(destination, task=task)
    assert {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()} == before
