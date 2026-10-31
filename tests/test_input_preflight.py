"""Malformed input must fail before providers or run directories are created."""

import copy

import numpy as np
import pytest

from pyrpmd.cli import main
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import run
from pyrpmd.units import parse_quantity, time_steps


def _job():
    return {
        "task": "drpmd", "temperature": "300 K", "beads": 1,
        "system": {"symbols": ["H", "H"],
                   "geometry": [[0, 0, -2], [0, 0, 2]],
                   "fragments": [[0], [1]]},
        "potential": {"provider": "free"},
        "dynamics": {"timestep": "0.1 fs"},
        "sampling": {"equilibration_steps": 0, "production_steps": 2,
                     "interval_steps": 1},
        "collision": {"bmax": "1 bohr", "trajectories": 1, "max_steps": 2},
        "outcomes": {"bond_cutoffs": {"H-H": "0.5 bohr"}},
    }


@pytest.mark.parametrize("section,field", [
    ("potential", "params"), ("potential", "initializer_params"),
    ("sampling", "thermostat_params"), ("outcomes", "params"),
    ("outcomes", "bond_cutoffs"),
])
@pytest.mark.parametrize("value", [None, [], "invalid", False, 3])
def test_nested_tables_fail_before_run_initialization(tmp_path, section, field, value):
    data = _job()
    data[section][field] = value
    if section == "potential" and field == "initializer_params" and value is None:
        resolved, _ = JobSpec.from_mapping(data).resolve()
        assert resolved["potential"][field] is None
        return
    output = tmp_path / "run"
    with pytest.raises(InputError, match=rf"{section}\.{field} must be a table"):
        run(data, output=output)
    assert not output.exists()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", ["off", "auto", 0, 1, 1.0, [], {}])
def test_native_policy_fails_before_run_initialization(tmp_path, nested, value):
    data = _job()
    (data["dynamics"] if nested else data)["native"] = value
    with pytest.raises(InputError, match="native must be"):
        run(data, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("value", [True, False, None, "required", np.bool_(False)])
def test_supported_native_policy_preserves_input(value):
    data = _job()
    data["dynamics"]["native"] = value
    before = copy.deepcopy(data)
    resolved, _ = JobSpec.from_mapping(data).resolve()
    actual = resolved["dynamics"]["native"]
    assert actual == value
    if isinstance(value, (bool, np.bool_)):
        assert type(actual) is bool
    assert data == before


def test_cli_reports_nested_input_error_without_traceback_or_output(tmp_path, capsys):
    source = tmp_path / "bad.py"
    data = _job()
    data["sampling"]["thermostat_params"] = "invalid"
    source.write_text(f"def build():\n    return {data!r}\n", encoding="utf-8")
    output = tmp_path / "run"
    assert main(["run", str(source), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "sampling.thermostat_params must be a table" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


@pytest.mark.parametrize("dt", [0, -1, np.inf, -np.inf, np.nan, True,
                               np.bool_(True), 1 + 0j, "1", np.array([1.0]), 10**400])
def test_time_steps_rejects_invalid_timestep_even_for_zero_duration(dt):
    with pytest.raises(ValueError, match="Time step must be"):
        time_steps(0, dt, allow_zero=True)


def test_time_steps_rejects_unrepresentable_ratio():
    with pytest.raises(ValueError, match="divided by the time step must be finite"):
        time_steps(1e308, 1e-308)


@pytest.mark.parametrize("value", [10**400, (10**400, "fs"), "1 fs^100000/fs^99999"])
def test_quantity_conversion_overflow_is_an_actionable_value_error(value):
    with pytest.raises(ValueError, match="Physical quantities must be finite"):
        time_steps(value, 1)


@pytest.mark.parametrize("allow_zero", [False, True])
def test_negative_duration_cannot_round_into_a_zero_step_request(allow_zero):
    with pytest.raises(ValueError, match="Negative duration"):
        time_steps(-1e-10, 1, allow_zero=allow_zero)


def test_time_steps_keeps_zero_integer_and_quantity_contracts():
    assert time_steps(0, np.float64(.1), allow_zero=True) == 0
    assert time_steps("1 fs", .1 * parse_quantity("1 fs", "time")) == 10
    assert time_steps((3, "au"), 1) == 3
    with pytest.raises(ValueError, match="integer multiple"):
        time_steps(1, .3)
    with pytest.raises(ValueError, match="at least one step"):
        time_steps(0, 1)
    with pytest.raises(ValueError, match="Negative duration"):
        time_steps(-1, 1, allow_zero=True)
