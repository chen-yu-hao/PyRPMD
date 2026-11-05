"""Level-resolved electronic partition functions and their rate-input wiring."""

import copy
import math

import pytest

from pyrpmd.constants import KB_AU, WAVENUMBER_TO_AU
from pyrpmd.electronic_partition import (
    ELECTRONIC_PRESETS, ElectronicModel, electronic_levels,
    electronic_partition_function, electronic_weight,
)
from pyrpmd.input import InputError, JobSpec
from pyrpmd.studies import plan_rate_study

from test_explicit_rate_preparation import _job


def _cl_h2(temperature):
    x = math.exp(-882.3515 * WAVENUMBER_TO_AU / (KB_AU * temperature))
    return 2 / (4 + 2 * x)


def test_partition_function_matches_closed_forms():
    t = 298.15
    beta = 1 / (KB_AU * t)
    o = 5 + 3 * math.exp(-beta * 158.265 * WAVENUMBER_TO_AU) + math.exp(-beta * 226.977 * WAVENUMBER_TO_AU)
    assert electronic_partition_function("O(3P)", t) == pytest.approx(o, rel=1e-14)
    assert electronic_weight("doublet", ["Cl(2P)", "singlet"], t) == pytest.approx(_cl_h2(t), rel=1e-14)
    # High- and low-temperature limits recover total and ground degeneracies.
    assert electronic_partition_function("Cl(2P)", 1e9) == pytest.approx(6, rel=1e-5)
    assert electronic_partition_function("Cl(2P)", 1.0) == pytest.approx(4, rel=1e-12)


def test_explicit_levels_units_and_presets_are_equivalent():
    explicit = [{"energy": 0, "degeneracy": 4}, {"energy": "882.3515 cm-1", "degeneracy": 2}]
    pairs = [[0.0, 4], [882.3515 * WAVENUMBER_TO_AU, 2]]
    for spec in (explicit, pairs, {"levels": explicit}, {"preset": "cl(2p)"}):
        levels = electronic_levels(spec).levels
        assert [g for _, g in levels] == [4, 2]
        assert [e for e, _ in levels] == pytest.approx([e for e, _ in ELECTRONIC_PRESETS["Cl(2P)"]], rel=1e-14)
    assert electronic_partition_function("H(2S)", "500 K") == 2


def test_common_energy_shift_cancels_only_within_one_species():
    shifted = [[-1.0, 4], [-1.0 + 882.3515 * WAVENUMBER_TO_AU, 2]]
    model = ElectronicModel.from_mapping({"transition_state": [[-1.0, 2]], "reactants": [shifted, "singlet"]})
    assert model.weight(400) == pytest.approx(_cl_h2(400), rel=1e-10)
    # A shift of one species alone is a physical energy offset, not a convention.
    lowered = ElectronicModel.from_mapping({"transition_state": "doublet", "reactants": [shifted, "singlet"]})
    with pytest.raises(ValueError, match="not finite"):
        lowered.weight(400)


@pytest.mark.parametrize("spec, message", [
    ("Xx(2P)", "unknown electronic preset"),
    ([], "nonempty"),
    ([[0.0, 0]], "positive integer"),
    ([[0.0, 1.5]], "positive integer"),
    ([[0.0, True]], "positive integer"),
    ([{"energy": 0}], "exactly energy and degeneracy"),
    ({"preset": "singlet", "levels": [[0, 1]]}, "exactly one"),
    ([["1 angstrom", 1]], "wrong dimension"),
])
def test_invalid_levels_are_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        electronic_levels(spec)


def test_model_checks_reactant_count_and_temperature():
    with pytest.raises(ValueError, match="2 entries"):
        ElectronicModel.from_mapping({"transition_state": "doublet", "reactants": ["Cl(2P)"]}, reactant_count=2)
    model = ElectronicModel.from_mapping({"transition_state": "doublet", "reactants": ["Cl(2P)", "singlet"]})
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="temperature|finite"):
            model.weight(bad)


def _electronic_job(temperature="400 K"):
    job = _job()
    job["temperature"] = temperature
    job["reaction"]["electronic"] = {"transition_state": "doublet", "reactants": ["Cl(2P)", "singlet"]}
    return job


def test_rate_input_evaluates_weight_at_job_temperature():
    job = _electronic_job()
    original = copy.deepcopy(job)
    config, _ = JobSpec.from_mapping(job).resolve()
    assert job == original
    reaction = config["reaction"]
    assert reaction["electronic_weight"] == pytest.approx(_cl_h2(400), rel=1e-12)
    model = reaction["electronic_weight_model"]
    assert model["temperature_K"] == pytest.approx(400)
    assert [r.get("preset") for r in model["reactants"]] == ["Cl(2P)", "singlet"]


@pytest.mark.parametrize("change, message", [
    (lambda r: r.update(electronic_weight=0.5), "not both"),
    (lambda r: r["electronic"].update(reactants=["Cl(2P)"]), "2 entries"),
    (lambda r: r["electronic"].pop("transition_state"), "exactly transition_state and reactants"),
    (lambda r: r["electronic"].update(reactants=["Cl(2P)", "nope"]), "unknown electronic preset"),
])
def test_rate_input_rejects_ambiguous_electronic_tables(change, message):
    job = _electronic_job()
    change(job["reaction"])
    with pytest.raises(InputError, match=message):
        JobSpec.from_mapping(job).resolve()


def test_temperature_study_reevaluates_electronic_weight_per_case():
    plan = plan_rate_study({"job": _electronic_job("300 K"), "temperatures": [200, 300, 1000]})
    seen = set()
    for case in plan["cases"]:
        t = case.parameters["temperature_K"]
        reaction = case.config["reaction"]
        assert reaction["electronic_weight"] == pytest.approx(_cl_h2(t), rel=1e-12)
        assert reaction["electronic_weight_model"]["temperature_K"] == pytest.approx(t)
        seen.add(round(reaction["electronic_weight"], 12))
    assert len(seen) == 3
