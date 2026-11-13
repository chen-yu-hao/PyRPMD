import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.workflows.nonadiabatic_rate import (
    load_nonadiabatic_rate_input, resolve_nonadiabatic_rate_config, run_nonadiabatic_rate,
)


EXAMPLE = Path(__file__).resolve().parents[1]/"examples"/"nonadiabatic"/"thermal_rate.inp"


def small_config():
    result = load_nonadiabatic_rate_input(EXAMPLE)
    result["sampling"].update(samples=8, burn_in=2, stride=1)
    result["dynamics"].update(steps=4, stride=1)
    return result


def test_thermal_rate_input_resolves_file_relative_factory_without_running_it():
    config = load_nonadiabatic_rate_input(EXAMPLE)
    assert config["task"] == "nonadiabatic_rate"
    assert config["potential"]["factory"] == str(EXAMPLE.parent/"thermal_rate_model.py")+":build"
    assert np.asarray(config["q"]).shape == (2, 1, 3)


def test_thermal_rate_runs_and_retains_raw_signed_normalization(tmp_path):
    config = small_config()
    result = run_nonadiabatic_rate(config, tmp_path/"run")
    assert result["method"] == "seo_thermal_nrpmd"
    assert result["formal_rate_s_inv"] is None
    assert result["candidate_rate"] is None
    with load_checkpoint(tmp_path/"run"/"population-rate.npz",
                          expected_kind="PyRPMD.thermal-population-rate.v1", require_integrity=True) as saved:
        raw = {key: saved[key].copy() for key in ("numerator", "reactant_weight", "signs")}
    assert raw["numerator"].shape == (8, 5)
    assert raw["reactant_weight"].shape == (8,)
    assert np.all(np.isin(raw["signs"], [-1., 1.]))
    assert result["signed_reactant_weight"] == pytest.approx(raw["reactant_weight"].mean())
    summary = json.loads((tmp_path/"run"/"results"/"summary.json").read_text())
    assert summary["physics_fingerprint"] == result["physics_fingerprint"]
    manifest = json.loads((tmp_path/"run"/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["factory_sha256"]) == 64
    assert config == small_config(), "runner must not mutate the caller's input"


def test_thermal_rate_reproducible_and_refuses_overwriting_existing_run(tmp_path):
    config = small_config()
    first = run_nonadiabatic_rate(config, tmp_path/"one")
    second = run_nonadiabatic_rate(config, tmp_path/"two")
    assert first == second
    with pytest.raises(ValueError, match="empty directory"):
        run_nonadiabatic_rate(config, tmp_path/"one")
    with load_checkpoint(tmp_path/"one"/"population-rate.npz") as a:
        with load_checkpoint(tmp_path/"two"/"population-rate.npz") as b:
            for key in ("time", "numerator", "reactant_weight", "signs", "population_correlation"):
                np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("change,match", [
    ({"unknown": 1}, "unknown"),
    ({"seed": True}, "seed"),
    ({"beta": -1}, "beta"),
    ({"nstates": 1}, "two states"),
    ({"beads": 3}, "expected.*3, 1, 3"),
    ({"rate": {"reactant_state": 0, "product_state": 0}}, "differ"),
    ({"rate": {"plateau_start": 0}}, "both"),
    ({"rate": {"plateau_start": 0, "plateau_stop": 1000}}, "interval"),
    ({"rate": {"min_average_sign": 2}}, "exceed one"),
    ({"dynamics": {"dt": 0.01, "native": "auto"}}, "boolean"),
    ({"sampling": {"samples": True}}, "positive integer"),
    ({"potential": {"provider": "adiabatic_python", "factory": "test:build"}}, "real diabatic"),
])
def test_thermal_rate_rejects_bad_configuration_before_sampling(change, match):
    config = small_config()
    config.update(copy.deepcopy(change))
    with pytest.raises(ValueError, match=match):
        resolve_nonadiabatic_rate_config(config)


def test_explicit_short_interval_is_not_automatically_a_physical_rate(tmp_path):
    config = small_config()
    config["rate"].update(plateau_start=0., plateau_stop=.08, block_size=2)
    result = run_nonadiabatic_rate(config, tmp_path/"pilot")
    assert result["formal_rate_s_inv"] is None
    assert result["plateau_detected"] is False
    if result["rate_curve_available"]:
        assert result["complete_blocks"] == 4
        assert result["candidate_rate"] is not None
    else:
        assert result["candidate_rate"] is None
        assert result["unavailable_reasons"]


def test_cli_runs_thermal_rate_and_reports_invalid_input(tmp_path, capsys):
    from pyrpmd.cli import main

    path = tmp_path/"rate.json"
    path.write_text(json.dumps(small_config()))
    assert main(["nonadiabatic-rate", str(path), "--output", str(tmp_path/"cli-run")]) == 0
    out = capsys.readouterr()
    assert not out.err
    summary = json.loads(out.out)
    assert summary["task"] == "nonadiabatic_rate"
    assert summary["formal_rate_s_inv"] is None
    assert (tmp_path/"cli-run"/"population-rate.npz").is_file()
    invalid = small_config()
    invalid["sampling"]["samples"] = False
    path.write_text(json.dumps(invalid))
    assert main(["nonadiabatic-rate", str(path), "--output", str(tmp_path/"invalid-run")]) == 2
    failure = capsys.readouterr()
    assert "sampling.samples" in failure.err
    assert not (tmp_path/"invalid-run").exists()
