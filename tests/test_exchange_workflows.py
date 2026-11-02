"""Exchange inputs, scientific identities, honest summaries and exact restart."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare
from pyrpmd.workflows.fes import _fingerprint
from pyrpmd.workflows.path_integral import KIND, run_path_integral

from test_pimd_method_workflows import arrays, raw_input


def exchange_input(method="fermionic", *, periodic=False, thermostat="pile-l"):
    raw = raw_input("bosonic", thermostat)
    raw["path_integral"] = {"method": method, "groups": [[0, 1]], "block_size": 1}
    if periodic:
        raw["system"].update(cell=[[3., 0., 0.], [.4, 3.2, 0.], [.1, .2, 3.4]],
                             pbc=[True, True, True])
        raw["potential"] = {"provider": "free"}
        raw["path_integral"]["periodic_mode"] = "minimum_image"
        raw["dynamics"]["integrator"] = "velocity-verlet"
        raw["sampling"]["thermostat"] = "andersen"
    return raw


@pytest.mark.parametrize("periodic", [False, True])
def test_fermionic_preflight_uses_bosonic_action_and_signed_estimators(tmp_path, periodic):
    raw = exchange_input(periodic=periodic)
    raw["output"]["directory"] = str(tmp_path / "unused")
    result = prepare(raw, numerical=True)
    assert result["ready"]
    assert result["path_integral_preparation"]["method"] == "fermionic"
    assert result["path_integral_preparation"]["observable"] == "fermionic_sign_reweighted_thermodynamics"
    estimators = result["pes_check"]["initial_exchange_estimators"]
    assert -1 <= estimators["sign"] <= 1
    assert np.isfinite(estimators["internal_energy_numerator_hartree"])
    assert ("pressure_numerator_au" in estimators) == periodic
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("method,periodic,thermostat", [
    ("fermionic", False, "pile-l"), ("fermionic", False, "andersen"),
    ("fermionic", True, "andersen"), ("bosonic", True, "andersen"),
])
def test_exchange_interrupted_resume_preserves_signed_history_and_rng(
        tmp_path, monkeypatch, method, periodic, thermostat):
    config, system = JobSpec.from_mapping(
        exchange_input(method, periodic=periodic, thermostat=thermostat)).resolve()
    whole = run_path_integral(config, system, tmp_path / "whole")
    name = "sample_step" if thermostat == "pile-l" else "step"
    original = getattr(Dynamics, name)

    def interrupt(self, state, **kwargs):
        result = original(self, state, **kwargs)
        if state.step == 8:
            raise RuntimeError("exchange checkpoint interruption")
        return result

    monkeypatch.setattr(Dynamics, name, interrupt)
    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        run_path_integral(config, system, tmp_path / "split")
    checkpoint = tmp_path / "split" / "checkpoints" / "path_integral_0000.npz"
    partial = arrays(checkpoint)
    assert partial["elapsed"] == 7
    assert len(json.loads(str(partial["exchange_estimators"]))) == 2
    monkeypatch.setattr(Dynamics, name, original)
    resumed = run_path_integral(config, system, tmp_path / "split")
    a, b = arrays(whole["replicas"][0]["checkpoint"]), arrays(checkpoint)
    for key in ("q", "p", "step", "time", "rng", "positions", "potential", "sample_steps",
                "exchange_estimators", "fingerprint"):
        np.testing.assert_array_equal(a[key], b[key])
    assert resumed["replicas"][0]["thermodynamics"] == whole["replicas"][0]["thermodynamics"]
    assert resumed["physical_converged"] is False
    monkeypatch.setattr(Dynamics, name, lambda *args, **kwargs: pytest.fail("complete run propagated"))
    assert run_path_integral(config, system, tmp_path / "split") == resumed


def test_old_bosonic_identity_and_checkpoint_remain_usable(tmp_path, monkeypatch):
    config, system = JobSpec.from_mapping(raw_input("bosonic")).resolve()
    assert _fingerprint(config, system, {"path_integral_replica": 0}) == (
        "bacded7edc0914128355f395be7ec69d26391bec317fc9ab9825eb059f98772d")
    result = run_path_integral(config, system, tmp_path)
    path = result["replicas"][0]["checkpoint"]
    before = arrays(path)
    legacy = {key: value for key, value in before.items()
              if not key.startswith("_") and key != "exchange_estimators"}
    atomic_savez(path, kind=KIND, **legacy)
    monkeypatch.setattr(Dynamics, "sample_step", lambda *args, **kwargs: pytest.fail("reanalysis propagated"))
    resumed = run_path_integral(config, system, tmp_path)
    after = arrays(path)
    for key in legacy:
        np.testing.assert_array_equal(before[key], after[key])
    assert before["exchange_estimators"] == after["exchange_estimators"]
    assert resumed == result


@pytest.mark.parametrize("field,value", [
    ("minimum_average_sign", .2), ("sign_sigma", 4.), ("block_size", 2),
    ("groups", [[0], [1]]), ("method", "bosonic"),
])
def test_new_exchange_controls_enter_run_identity(field, value):
    raw = exchange_input()
    config, system = JobSpec.from_mapping(raw).resolve()
    original = _fingerprint(config, system, {"path_integral_replica": 0})
    raw["path_integral"][field] = value
    changed, other_system = JobSpec.from_mapping(raw).resolve()
    assert _fingerprint(changed, other_system, {"path_integral_replica": 0}) != original
    # Resolving uses defaults without changing an existing input's identity.
    assert "minimum_average_sign" not in config["path_integral"]
    assert "sign_sigma" not in config["path_integral"]


def test_periodic_mode_enters_fingerprint_and_changed_threshold_cannot_resume(tmp_path):
    raw = exchange_input(periodic=True)
    config, system = JobSpec.from_mapping(raw).resolve()
    removed = copy.deepcopy(config)
    removed["path_integral"].pop("periodic_mode")
    assert _fingerprint(config, system, {}) != _fingerprint(removed, system, {})
    run_path_integral(config, system, tmp_path)
    raw["path_integral"]["minimum_average_sign"] = .1
    changed, system = JobSpec.from_mapping(raw).resolve()
    with pytest.raises(ValueError, match="physics differs"):
        run_path_integral(changed, system, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("minimum_average_sign", 0.), ("minimum_average_sign", 1.1),
    ("minimum_average_sign", True), ("minimum_average_sign", np.nan),
    ("sign_sigma", 0.), ("sign_sigma", "3"), ("sign_sigma", np.inf),
    ("block_size", 0), ("block_size", 2.5), ("block_size", True),
    ("block_size", 7), ("periodic_mode", "winding"),
])
def test_invalid_exchange_controls_fail_before_running(field, value):
    raw = exchange_input()
    raw["path_integral"][field] = value
    with pytest.raises(InputError):
        JobSpec.from_mapping(raw).resolve()


def test_periodic_exchange_requires_explicit_mic_and_cartesian_propagation():
    raw = exchange_input(periodic=True)
    raw["path_integral"].pop("periodic_mode")
    with pytest.raises(InputError, match="minimum_image"):
        JobSpec.from_mapping(raw).resolve()
    raw = exchange_input(periodic=True)
    raw["dynamics"]["integrator"] = "normal-mode-bab"
    with pytest.raises(InputError, match="velocity-verlet"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["bosonic", "fermionic"])
def test_exchange_bzp_is_explicitly_rejected(method):
    raw = exchange_input(method, periodic=True)
    raw["barostat"] = {"method": "bzp", "pressure": "1 bar", "time_constant": "10 fs"}
    with pytest.raises(InputError, match="exchange.*BZP.*centroid"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("damage", ["missing", "length", "sign", "nan", "unknown", "schema",
                                    "context", "missing_virial", "energy_sum"])
def test_checkpoint_validates_exchange_history(tmp_path, damage):
    config, system = JobSpec.from_mapping(exchange_input()).resolve()
    result = run_path_integral(config, system, tmp_path)
    path = result["replicas"][0]["checkpoint"]
    data = {key: value for key, value in arrays(path).items() if not key.startswith("_")}
    rows = json.loads(str(data["exchange_estimators"]))
    if damage == "missing":
        del data["exchange_estimators"]
    else:
        if damage == "length":
            rows.pop()
        elif damage == "sign":
            rows[0]["sign"] = 1.2
        elif damage == "nan":
            rows[0]["internal_energy_numerator_hartree"] = float("nan")
        elif damage == "unknown":
            rows[0]["unexpected"] = 1.
        elif damage == "schema":
            rows[0].pop("kinetic_energy_numerator_hartree")
        elif damage == "context":
            for row in rows:
                row["pressure_numerator_au"] = 1.
        elif damage == "missing_virial":
            for row in rows:
                row.pop("virial_internal_energy_numerator_hartree")
        elif damage == "energy_sum":
            rows[0]["internal_energy_numerator_hartree"] += 1.
        data["exchange_estimators"] = json.dumps(rows)
    atomic_savez(path, kind=KIND, **data)
    with pytest.raises(ValueError, match="exchange.*history"):
        run_path_integral(config, system, tmp_path)


def test_unresolved_sign_never_reports_bosonic_potential_as_fermionic_energy(tmp_path):
    raw = exchange_input()
    raw["path_integral"]["minimum_average_sign"] = 1.
    config, system = JobSpec.from_mapping(raw).resolve()
    result = run_path_integral(config, system, tmp_path)
    replica = result["replicas"][0]
    report = replica["thermodynamics"]
    assert not report["sign_resolved"]
    assert report["average_sign"] < 1.
    assert report["average_sign_standard_error"] is not None
    assert report["internal_energy_hartree"] is None
    assert replica["potential_energy_hartree"] is None
    assert replica["bosonic_sampling_potential_energy_hartree"] > 0
    assert result["warnings"]
    assert result["physical_converged"] is False
    json.dumps(result, allow_nan=False)


def test_changing_virial_support_rolls_back_before_saving_mixed_history(tmp_path, monkeypatch):
    from pyrpmd.pimd_methods import BosonicExchangePotential
    config, system = JobSpec.from_mapping(exchange_input(periodic=True)).resolve()
    whole = run_path_integral(config, system, tmp_path / "whole")
    original = BosonicExchangePotential.estimators
    calls = 0

    def changing(self, positions, **kwargs):
        nonlocal calls
        row = original(self, positions, **kwargs)
        calls += 1
        if calls == 2:
            row.pop("pressure_numerator_au")
        return row

    monkeypatch.setattr(BosonicExchangePotential, "estimators", changing)
    with pytest.raises(ValueError, match="columns changed"):
        run_path_integral(config, system, tmp_path / "split")
    path = tmp_path / "split" / "checkpoints" / "path_integral_0000.npz"
    partial = arrays(path)
    assert partial["elapsed"] == 6
    assert len(json.loads(str(partial["exchange_estimators"]))) == 1
    monkeypatch.setattr(BosonicExchangePotential, "estimators", original)
    resumed = run_path_integral(config, system, tmp_path / "split")
    a, b = arrays(whole["replicas"][0]["checkpoint"]), arrays(path)
    for key in ("q", "p", "rng", "positions", "exchange_estimators"):
        np.testing.assert_array_equal(a[key], b[key])
    assert resumed["replicas"][0]["thermodynamics"] == whole["replicas"][0]["thermodynamics"]
