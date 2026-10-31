"""Independent scientific regression checks for coordinated rate studies."""

from copy import deepcopy

import numpy as np
import pytest

from pyrpmd.studies import analyze_rate_study, plan_rate_study


def collision_job():
    return {
        "task": "drpmd", "temperature": 300.0, "beads": 2,
        "system": {"symbols": ["H", "H", "H"], "masses": [1., 1., 1.],
                   "geometry": [[-1., 0., 0.], [1., 0., 0.], [0., 0., 10.]],
                   "fragments": [[0, 1], [2]]},
        "potential": {"provider": "free"},
        "dynamics": {"timestep": .02},
        "sampling": {"equilibration_steps": 4, "production_steps": 40, "interval_steps": 4,
                     "andersen_interval": 2},
        "collision": {"bmax": 1.0, "trajectories": 2, "max_steps": 200},
        "outcomes": {"bond_cutoffs": {"H-H": 2.5}, "exit_distance": 12.0,
                     "check_interval": 2, "persistence_steps": 4},
        "output": {"checkpoint_steps": 20, "trajectory_stride": 4},
    }


def test_separation_and_isotope_scans_reposition_thermal_parents():
    job = collision_job()
    original = deepcopy(job)
    plan = plan_rate_study({
        "job": job, "isotopologues": {"light": [], "heavy": [{"atom": 0, "mass": 2.}]},
        "convergence": {"separation": [10., 14.], "timestep": [.02, .01]},
    })
    for case in plan["cases"]:
        system, config = case.system, case.config
        centers = [np.average(system.geometry[list(fragment)], axis=0,
                              weights=system.masses[list(fragment)]) for fragment in system.fragments]
        assert np.linalg.norm(centers[1] - centers[0]) == pytest.approx(case.parameters["separation"], abs=1e-12)
        # Uniform translations must preserve the intrafragment coordinates.
        np.testing.assert_allclose(system.geometry[1] - system.geometry[0], [2., 0., 0.], atol=1e-12)
        assert config["outcomes"]["exit_distance_au"] == 12.
        assert config["collision"]["max_steps"] * config["dt_au"] == pytest.approx(4.)
        assert config["outcomes"]["persistence_steps"] * config["dt_au"] == pytest.approx(.08)
        assert config["outcomes"]["check_interval"] * config["dt_au"] == pytest.approx(.04)
    assert job == original


def test_analysis_rejects_explicitly_incompatible_record_units():
    plan = plan_rate_study({"job": collision_job()})
    records = [{"case_id": c.case_id, "rate": 1., "standard_error": .1,
                "rate_unit": "s^-1"} for c in plan["cases"]]
    with pytest.raises(ValueError, match="unit"):
        analyze_rate_study(plan, records)


def test_unresolved_kie_interval_preserves_the_remaining_study_results():
    plan = plan_rate_study({"job": collision_job(), "isotopologues": {
        "light": [], "heavy": [{"atom": 0, "mass": 2.}],
    }})
    records = [{"case_id": c.case_id,
                "rate": 1. if c.parameters["isotopologue"] == "light" else 1e-300,
                "standard_error": 1., "rate_unit": "cm^3 molecule^-1 s^-1"}
               for c in plan["cases"]]
    report = analyze_rate_study(plan, records)
    assert len(report["rows"]) == 2
    assert len(report["kinetic_isotope_effects"]) == 1
    assert report["kinetic_isotope_effects"][0]["ratio"] is None
    assert report["kinetic_isotope_effects"][0].get("reason")
