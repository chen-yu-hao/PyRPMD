"""Scientific reporting must distinguish failed experiments from pilot rates."""

import json
from math import sqrt
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.transition_state_search import rates


def _arguments(monkeypatch, directory, cases):
    evidence = directory / "evidence.json"
    monkeypatch.setattr(sys, "argv", [
        "rates.py", "--cases", *cases,
        "--output", str(directory / "runs"),
        "--evidence", str(evidence),
        "--library", str(directory / "unused-mocked-pes.so"),
        "--workers", "1",
    ])
    return evidence


def _pilot(case, seed, estimate):
    return {
        "case": case,
        "settings": {"seed": seed},
        "physical_validation": False,
        "results": {
            "status": "unconverged",
            "rate_estimate_cm3_s": estimate,
            "plateau_detected": False,
        },
        "comparison": {
            "status": "pilot_comparison",
            "physical_validation_passed": None,
            "reason": "Finite sampling; physical convergence is not established.",
        },
    }


def test_duplicate_cli_cases_are_rejected_before_sampling(monkeypatch, tmp_path, capsys):
    evidence = _arguments(monkeypatch, tmp_path, ["classical-a", "classical-a"])
    run_case = Mock(return_value=_pilot("classical-a", 11, 1e-18))
    monkeypatch.setattr(rates, "run_case", run_case)

    with pytest.raises(SystemExit) as stopped:
        rates.main()

    assert stopped.value.code != 0
    assert "duplicate" in capsys.readouterr().err.lower()
    run_case.assert_not_called()
    assert not evidence.exists()


@pytest.mark.parametrize("second_label", ["classical-a", "classical-b"])
def test_repeated_seed_cannot_count_as_an_independent_replicate(second_label):
    records = [_pilot("classical-a", 11, 1e-18), _pilot(second_label, 11, 3e-18)]

    with pytest.raises(ValueError, match="independent classical seed records must be unique"):
        rates.report(records)


def test_between_seed_uncertainty_excludes_same_seed_timestep_sensitivity_case():
    records = [
        _pilot("classical-a", 11, 1e-18),
        _pilot("classical-b", 12, 2e-18),
        _pilot("classical-c", 13, 4e-18),
        # The half-dt run deliberately reuses A's seed. It measures a different
        # numerical protocol and must not become a fourth independent replicate.
        _pilot("classical-halfdt", 11, 1e-10),
    ]

    report = rates.report(records)
    statistics = report["independent_classical_seed_statistics"]

    assert len(statistics["independent_seeds"]) == 3
    assert set(statistics["independent_seeds"]) == {11, 12, 13}
    assert statistics["mean_rate_cm3_molecule_s"] == pytest.approx(7e-18 / 3, rel=1e-12, abs=0)
    # For the three observations 1, 2, 4, sample variance is 7/3 and SEM sqrt(7)/3.
    assert statistics["between_seed_standard_error_cm3_molecule_s"] == pytest.approx(
        sqrt(7) * 1e-18 / 3, rel=1e-12, abs=0,
    )
    assert report["physical_validation"] is False
    assert len(report["cases"]) == 4  # Preserve the sensitivity result in the evidence.


def test_sampler_failure_is_saved_before_cli_exits_nonzero(monkeypatch, tmp_path):
    evidence = _arguments(monkeypatch, tmp_path, ["classical-a"])
    # Exercise the real run_case exception handling, report and CLI. Only the
    # external PES loader and expensive sampler are replaced; no BKMP2 build or
    # trajectories are needed for this reporting regression.
    monkeypatch.setattr(rates, "import_object", lambda _: (
        lambda **kwargs: SimpleNamespace(metadata={"name": "mock PES"}, close=lambda: None)
    ))
    sampler = Mock(side_effect=RuntimeError("synthetic sampling failure"))
    monkeypatch.setattr(rates, "run_resolved", sampler)

    with pytest.raises(SystemExit) as stopped:
        rates.main()

    assert stopped.value.code == 1
    sampler.assert_called_once()
    report = json.loads(evidence.read_text())
    saved_case = json.loads((tmp_path / "runs" / "classical-a" / "case.json").read_text())
    assert report["cases"] == [saved_case]
    assert saved_case["error"]["type"] == "RuntimeError"
    assert saved_case["error"]["message"] == "synthetic sampling failure"
    assert report["physical_validation"] is False


def test_unconverged_pilot_is_preserved_without_becoming_an_operational_failure(monkeypatch, tmp_path):
    evidence = _arguments(monkeypatch, tmp_path, ["classical-a"])
    pilot = _pilot("classical-a", 11, 1.4e-18)
    monkeypatch.setattr(rates, "run_case", Mock(return_value=pilot))

    assert rates.main() in (None, 0)

    report = json.loads(evidence.read_text())
    assert report["cases"] == [pilot]
    assert report["physical_validation"] is False
    assert report["cases"][0]["results"]["status"] == "unconverged"
    assert report["cases"][0]["comparison"]["physical_validation_passed"] is None
    assert "error" not in report["cases"][0]


def test_failed_case_does_not_discard_later_case_evidence(monkeypatch, tmp_path):
    evidence = _arguments(monkeypatch, tmp_path, ["classical-a", "classical-b"])
    failure = {
        "case": "classical-a", "physical_validation": False,
        "error": {"type": "RuntimeError", "message": "sampling stopped"},
    }
    pilot = _pilot("classical-b", 12, 1.5e-18)
    run_case = Mock(side_effect=[failure, pilot])
    monkeypatch.setattr(rates, "run_case", run_case)

    with pytest.raises(SystemExit) as stopped:
        rates.main()

    assert stopped.value.code == 1
    assert run_case.call_count == 2
    report = json.loads(evidence.read_text())
    assert report["cases"] == [failure, pilot]
    assert report["physical_validation"] is False
