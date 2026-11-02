"""Public extension budgets are explicit, exclusive and forwarded without conversion."""

import json

import pytest

from pyrpmd.cli import main
from pyrpmd.input import InputError


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["--max-time", "20 ps"], {"max_time": "20 ps", "production_steps": None}),
        (["--production-steps", "200000"], {"max_time": None, "production_steps": 200000}),
    ],
)
def test_extend_forwards_one_total_budget(monkeypatch, capsys, arguments, expected):
    calls = []

    def extend(directory, **budget):
        calls.append((directory, budget))
        return {"output": directory, "status": "complete"}

    monkeypatch.setattr("pyrpmd.runner.extend_run", extend)
    assert main(["extend", "run with spaces", *arguments]) == 0
    assert calls == [("run with spaces", expected)]
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"output": "run with spaces", "status": "complete"}
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--max-time", "20 ps", "--production-steps", "200000"],
        ["--production-steps", "200000.5"],
    ],
)
def test_invalid_budget_arguments_fail_before_execution(monkeypatch, capsys, arguments):
    def unexpected_execution(*args, **kwargs):
        pytest.fail("invalid budget arguments reached the workflow")

    monkeypatch.setattr("pyrpmd.runner.extend_run", unexpected_execution)
    with pytest.raises(SystemExit) as caught:
        main(["extend", "saved-run", *arguments])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--production-steps" in captured.err


def test_extension_scope_error_is_actionable(monkeypatch, capsys):
    def unsupported_extension(*args, **kwargs):
        raise InputError("Rate production extension requires fixed-window umbrella sampling")

    monkeypatch.setattr("pyrpmd.runner.extend_run", unsupported_extension)
    assert main(["extend", "adaptive-run", "--production-steps", "200000"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fixed-window umbrella sampling" in captured.err
    assert "Traceback" not in captured.err
