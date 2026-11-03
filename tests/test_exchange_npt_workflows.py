"""Periodic image sums and all-bead exchange NPT at the input/restart boundary."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare
from pyrpmd.workflows.fes import _fingerprint
from pyrpmd.workflows.path_integral import KIND, run_path_integral

from test_exchange_workflows import exchange_input
from test_pimd_method_workflows import arrays


def npt_input(method="bosonic", mode="full_winding"):
    raw = exchange_input(method, periodic=True)
    raw["path_integral"]["periodic_mode"] = mode
    raw["barostat"] = {"method": "bzp", "pressure": "0.04 au",
                       "time_constant": "1.2 au", "friction": "0.8 au"}
    return raw


@pytest.mark.parametrize("method", ["bosonic", "fermionic"])
def test_full_winding_npt_preflight_uses_pressure_and_all_bead_measure(method, tmp_path):
    raw = npt_input(method)
    raw["output"]["directory"] = str(tmp_path / "unused")
    raw["path_integral"].update(winding_tolerance=1e-11, winding_max_terms=10000)
    result = prepare(raw, numerical=True)
    assert result["ready"]
    assert result["path_integral_preparation"]["barostat_coordinate_scaling"] == "all_beads"
    assert "pressure_numerator_au" in result["pes_check"]["initial_exchange_estimators"]
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("method", ["bosonic", "fermionic"])
@pytest.mark.parametrize("mode", ["minimum_image", "full_winding"])
def test_exchange_npt_atomic_resume_keeps_cells_piston_signed_observables_and_rng(
        tmp_path, monkeypatch, method, mode):
    from pyrpmd.exchange_barostat import ExchangeBZPBarostat

    config, system = JobSpec.from_mapping(npt_input(method, mode)).resolve()
    whole = run_path_integral(config, system, tmp_path / "whole")
    original = ExchangeBZPBarostat.step

    def interrupt(self, state, dynamics, **kwargs):
        result = original(self, state, dynamics, **kwargs)
        if state.step == 8:
            raise RuntimeError("exchange NPT interruption")
        return result

    monkeypatch.setattr(ExchangeBZPBarostat, "step", interrupt)
    with pytest.raises(RuntimeError, match="NPT interruption"):
        run_path_integral(config, system, tmp_path / "split")
    checkpoint = tmp_path / "split" / "checkpoints" / "path_integral_0000.npz"
    partial = arrays(checkpoint)
    assert partial["elapsed"] == 7
    assert np.shape(partial["cells"]) == (2, 3, 3)
    monkeypatch.setattr(ExchangeBZPBarostat, "step", original)
    resumed = run_path_integral(config, system, tmp_path / "split")
    a, b = arrays(whole["replicas"][0]["checkpoint"]), arrays(checkpoint)
    for key in ("q", "p", "step", "time", "rng", "positions", "potential", "sample_steps",
                "exchange_estimators", "cells", "barostat", "fingerprint"):
        np.testing.assert_array_equal(a[key], b[key])
    expected_summary = copy.deepcopy(whole)
    expected_summary["replicas"][0]["checkpoint"] = resumed["replicas"][0]["checkpoint"]
    assert resumed == expected_summary
    assert resumed["barostat_coordinate_scaling"] == "all_beads"
    assert resumed["ensemble"] == "npt"
    rows = json.loads(str(b["exchange_estimators"]))
    volumes = np.abs(np.linalg.det(b["cells"]))
    for row, volume in zip(rows, volumes):
        assert row["volume_numerator_bohr3"] == pytest.approx(volume * row["sign"])
        assert row["enthalpy_numerator_hartree"] == pytest.approx(
            row["internal_energy_numerator_hartree"] + .04 * volume * row["sign"])
    monkeypatch.setattr(ExchangeBZPBarostat, "step", lambda *args, **kwargs: pytest.fail("complete run propagated"))
    assert run_path_integral(config, system, tmp_path / "split") == resumed


def test_npt_empty_sample_prefix_resumes_without_reconstructing_at_wrong_cell(tmp_path, monkeypatch):
    from pyrpmd.exchange_barostat import ExchangeBZPBarostat

    config, system = JobSpec.from_mapping(npt_input()).resolve()
    original = ExchangeBZPBarostat.step

    def fail(self, state, dynamics, **kwargs):
        original(self, state, dynamics, **kwargs)
        raise RuntimeError("before first sample")

    monkeypatch.setattr(ExchangeBZPBarostat, "step", fail)
    with pytest.raises(RuntimeError, match="first sample"):
        run_path_integral(config, system, tmp_path)
    monkeypatch.setattr(ExchangeBZPBarostat, "step", original)
    assert run_path_integral(config, system, tmp_path)["samples"] == 6


@pytest.mark.parametrize("damage", ["coupling", "missing_cell", "missing_history",
    "missing_pressure", "missing_volume", "volume", "enthalpy"])
def test_npt_checkpoint_rejects_incompatible_measure_and_observable_history(tmp_path, damage):
    config, system = JobSpec.from_mapping(npt_input()).resolve()
    result = run_path_integral(config, system, tmp_path)
    path = result["replicas"][0]["checkpoint"]
    data = {key: value for key, value in arrays(path).items() if not key.startswith("_")}
    rows = json.loads(str(data["exchange_estimators"]))
    if damage == "coupling":
        piston = json.loads(str(data["barostat"]))
        piston["coupling"] = "centroid"
        data["barostat"] = json.dumps(piston)
    elif damage == "missing_cell":
        del data["cells"]
    elif damage == "missing_history":
        del data["exchange_estimators"]
    else:
        if damage == "missing_pressure":
            for row in rows:
                row.pop("pressure_numerator_au")
        elif damage == "missing_volume":
            for row in rows:
                row.pop("volume_numerator_bohr3")
        elif damage == "volume":
            rows[0]["volume_numerator_bohr3"] *= 1.01
        elif damage == "enthalpy":
            rows[0]["enthalpy_numerator_hartree"] += .01
        data["exchange_estimators"] = json.dumps(rows)
    atomic_savez(path, kind=KIND, **data)
    with pytest.raises(ValueError):
        run_path_integral(config, system, tmp_path)


def test_unresolved_fermionic_npt_volume_and_enthalpy_are_withheld(tmp_path):
    raw = npt_input("fermionic")
    raw["path_integral"]["minimum_average_sign"] = 1.
    config, system = JobSpec.from_mapping(raw).resolve()
    result = run_path_integral(config, system, tmp_path)
    replica = result["replicas"][0]
    assert replica["thermodynamics"]["average_sign"] < 1.
    assert replica["mean_volume_bohr3"] is None
    assert replica["thermodynamics"]["enthalpy_hartree"] is None
    assert replica["bosonic_sampling_mean_volume_bohr3"] > 0
    assert result["physical_converged"] is False


@pytest.mark.parametrize("field,value", [("winding_tolerance", 1e-10), ("winding_max_terms", 20000)])
def test_winding_controls_enter_scientific_identity_and_block_changed_resume(tmp_path, field, value):
    raw = npt_input()
    config, system = JobSpec.from_mapping(raw).resolve()
    original = _fingerprint(config, system, {})
    run_path_integral(config, system, tmp_path)
    changed_raw = copy.deepcopy(raw)
    changed_raw["path_integral"][field] = value
    changed, other_system = JobSpec.from_mapping(changed_raw).resolve()
    assert _fingerprint(changed, other_system, {}) != original
    with pytest.raises(ValueError, match="physics differs"):
        run_path_integral(changed, other_system, tmp_path)


@pytest.mark.parametrize("field,value", [("winding_tolerance", 0.), ("winding_tolerance", True),
    ("winding_tolerance", float("nan")), ("winding_max_terms", 0), ("winding_max_terms", 2.5)])
def test_invalid_winding_controls_fail_during_preflight(field, value):
    raw = npt_input()
    raw["path_integral"][field] = value
    with pytest.raises(InputError):
        JobSpec.from_mapping(raw).resolve()


def test_inactive_winding_and_thermostat_controls_are_rejected():
    raw = npt_input(mode="minimum_image")
    raw["path_integral"]["winding_tolerance"] = 1e-12
    with pytest.raises(InputError, match="full_winding"):
        JobSpec.from_mapping(raw).resolve()
    raw = npt_input()
    raw["sampling"]["gamma_centroid"] = 1.
    with pytest.raises(InputError, match="gamma_centroid"):
        JobSpec.from_mapping(raw).resolve()


def test_required_native_npt_fails_before_provider_or_output(tmp_path):
    raw = npt_input()
    raw["native"] = "required"
    raw["output"]["directory"] = str(tmp_path / "unused")
    with pytest.raises(InputError, match="Python propagation loop"):
        prepare(raw, numerical=True)
    assert not (tmp_path / "unused").exists()
