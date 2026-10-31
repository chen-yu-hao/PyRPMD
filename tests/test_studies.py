"""Analytical kinetics, controlled studies and reproducible workflow execution."""

from copy import deepcopy
import json

import numpy as np
import pytest

from pyrpmd.constants import AMU_TO_AU, KB_AU
from pyrpmd.studies import (analyze_rate_study, arrhenius_fit, kinetic_isotope_effect,
                           plan_rate_study, run_rate_study)


def site_job():
    return {
        "task": "site_rate", "temperature": "300 K", "beads": 1, "seed": 61,
        "system": {"symbols": ["H"], "masses": [2.0], "geometry": [[0., .1, 0.]]},
        "potential": {"provider": "harmonic", "params": {"k": .002}},
        "dynamics": {"timestep": .02, "loop_backend": "python"},
        "sampling": {"equilibration_steps": 2, "production_steps": 40,
                     "interval_steps": 1, "andersen_interval": 5, "replicas": 1},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
        "umbrella": {"cvs": ["x"], "centers": [-1., -.5, 0., .5, 1.], "stiffness": .01},
        "analysis": {"bins": 8, "ranges": [[-1.5, 1.5]], "bootstrap": 0},
        "site_rate": {"cv": "x", "basin": [-.6, .6], "dividing_surface": .6, "direction": 1},
        "recrossing": {"parent_replicas": 2, "parent_samples": 2, "equilibration_steps": 2,
                       "parent_interval_steps": 2, "child_pairs": 1, "child_steps": 8, "bootstrap": 4},
        "output": {"checkpoint_steps": 4},
    }


def test_arrhenius_recovers_activation_energy_and_prefactor_in_either_rate_unit():
    temperatures = np.array([200., 300., 500., 800.])
    ea, prefactor = .003, 2.7e11
    rates = prefactor * np.exp(-ea / (KB_AU * temperatures))
    fit = arrhenius_fit(temperatures, rates, standard_errors=.03 * rates)
    assert fit["activation_energy_hartree"] == pytest.approx(ea, rel=1e-13)
    assert fit["A"] == pytest.approx(prefactor, rel=1e-13)
    scaled = arrhenius_fit(temperatures, rates * 1e-20, standard_errors=.03 * rates * 1e-20,
                           rate_unit="cm^3 molecule^-1 s^-1")
    assert scaled["activation_energy_hartree"] == pytest.approx(ea, rel=1e-13)
    assert scaled["activation_energy_standard_error_hartree"] == pytest.approx(fit["activation_energy_standard_error_hartree"])
    assert not fit["physical_converged"]


def test_arrhenius_does_not_hide_curvature_or_invent_two_point_error():
    t = np.array([200., 250., 300., 400.])
    rates = np.exp(20 - 500 / t + 200000 / t**2)
    result = arrhenius_fit(t, rates)
    assert np.max(np.abs(result["residuals_ln_k"])) > .1
    assert arrhenius_fit(t[:2], rates[:2])["ln_A_standard_error"] is None
    with pytest.raises(ValueError, match="distinct"):
        arrhenius_fit([300, 300], [1, 2])
    with pytest.raises(ValueError, match="positive"):
        arrhenius_fit([300, 400], [0, 2])


def test_kie_direction_and_independent_uncertainty():
    kie = kinetic_isotope_effect(4., 2., numerator_error=.4, denominator_error=.1)
    assert kie["ratio"] == pytest.approx(2)
    assert kie["standard_error"] == pytest.approx(2 * np.hypot(.1, .05))
    assert kinetic_isotope_effect(2., 4.)["ratio"] == .5
    assert kie["confidence_interval_95"][0] < 2 < kie["confidence_interval_95"][1]
    with pytest.raises(ValueError, match="both"):
        kinetic_isotope_effect(1, 2, numerator_error=.1)


def test_scan_keeps_physical_durations_changes_masses_and_uses_independent_seeds():
    job = site_job()
    before = deepcopy(job)
    study = {"job": job, "temperatures": [300, "400 K"], "replicates": 2,
             "isotopologues": {"H": [], "D": [{"atom": 0, "mass": "2.01410177812 amu"}]},
             "convergence": {"beads": [1, 2, 4], "timestep": [.02, .01]}}
    plan = plan_rate_study(study)
    assert job == before
    assert len(plan["cases"]) == 2 * 2 * 2 * 4
    seeds = [c.config["seed"] for c in plan["cases"]]
    assert len(set(seeds)) == len(seeds)
    for case in plan["cases"]:
        c = case.config
        assert c["sampling"]["production_steps"] * c["dt_au"] == pytest.approx(.8)
        assert c["sampling"]["andersen_interval"] * c["dt_au"] == pytest.approx(.1)
        assert c["recrossing"]["child_steps"] * c["dt_au"] == pytest.approx(.16)
        assert c["output"]["checkpoint_steps"] * c["dt_au"] == pytest.approx(.08)
        assert c["beta"] * KB_AU * case.parameters["temperature_K"] == pytest.approx(1)
        if case.parameters["isotopologue"] == "D":
            assert case.system.masses[0] == pytest.approx(2.01410177812 * AMU_TO_AU)
    repeat = plan_rate_study(study)
    assert [c.to_dict() for c in plan["cases"]] == [c.to_dict() for c in repeat["cases"]]


def test_finer_timestep_must_preserve_all_durations_exactly():
    with pytest.raises(ValueError, match="integer multiple"):
        plan_rate_study({"job": site_job(), "convergence": {"timestep": [.02, .015]}})
    with pytest.raises(ValueError, match="gas-phase"):
        plan_rate_study({"job": site_job(), "convergence": {"separation": [10, 20]}})
    with pytest.raises(ValueError, match="drpmd"):
        plan_rate_study({"job": site_job(), "convergence": {"bmax": [2, 4]}})


def test_convergence_requires_equivalence_not_noisy_agreement():
    plan = plan_rate_study({"job": site_job(), "convergence": {"beads": [1, 2, 4]}, "relative_tolerance": .05})
    def records(error):
        return [{"case_id": c.case_id, "rate": 100 + 1 / c.parameters["beads"], "standard_error": error} for c in plan["cases"]]
    noisy = analyze_rate_study(plan, records(10))
    assert not noisy["numerically_converged_on_tested_grid"]
    precise = analyze_rate_study(plan, records(.1))
    assert precise["numerically_converged_on_tested_grid"]
    assert not precise["physical_converged"]
    unsupported = records(.1)
    unsupported[0]["rate"] = None
    assert not analyze_rate_study(plan, unsupported)["numerically_converged_on_tested_grid"]
    missing_error = records(None)
    result = analyze_rate_study(plan, missing_error)
    assert all(c["status"] == "inconclusive" for c in result["convergence_checks"])


def test_temperature_fits_and_kies_use_only_finest_supported_cases():
    plan = plan_rate_study({"job": site_job(), "temperatures": [300., 400., 600.],
                           "isotopologues": {"H": [], "D": [{"atom": 0, "mass": 4.}]},
                           "convergence": {"beads": [1, 4]}})
    records = []
    for c in plan["cases"]:
        k = 1e10 * np.exp(-.002 / (KB_AU * c.parameters["temperature_K"]))
        if c.parameters["isotopologue"] == "D":
            k /= 3
        if c.parameters["beads"] == 1:
            k *= 100
        records.append({"case_id": c.case_id, "rate": k, "standard_error": .01 * k})
    result = analyze_rate_study(plan, records)
    assert result["arrhenius"]["H"]["A"] == pytest.approx(1e10)
    assert all(kie["ratio"] == pytest.approx(3.) for kie in result["kinetic_isotope_effects"])


def test_real_workflow_scan_resume_and_changed_identity_guard(tmp_path):
    study = {"job": site_job(), "convergence": {"beads": [1, 2]}}
    output = tmp_path / "study"
    first = run_rate_study(study, output=output)
    second = run_rate_study(study, output=output, resume=True)
    assert first == second
    assert len(first["cases"]) == 2
    assert not first["physical_converged"]
    assert (output / "rates.csv").is_file()
    saved = json.loads((output / "study.json").read_text())
    assert len(saved["cases"]) == 2
    changed = deepcopy(study)
    changed["temperatures"] = [500]
    with pytest.raises(ValueError, match="identical"):
        run_rate_study(changed, output=output, resume=True)


def test_cli_plan_does_not_run_or_write_simulations(tmp_path, capsys):
    from pyrpmd.cli import main
    source = tmp_path / "study.json"
    source.write_text(json.dumps({"job": site_job(), "temperatures": [300, 600]}))
    assert main(["study", str(source), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["cases"]) == 2
    assert not list(tmp_path.glob("case-*"))
