"""TRPMD child pairing, checkpoint identity, and heat-aware rate diagnostics."""

import copy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows import rate


def child_task(output, *, method="trpmd", friction=.5):
    system = System(
        ("H", "H", "H"), np.array([2., 2., 2.]),
        np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]),
        ((0,), (1, 2)),
    )
    config = {
        "task": "rate", "beta": 10., "beads": 4, "dt_au": 1e-4,
        "seed": 15, "native": False, "potential": {"provider": "free"},
        "sampling": {},
        "reaction": {
            "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]], "rinf": 10.,
        },
    }
    if method == "trpmd":
        config["recrossing"] = {"dynamics": "trpmd", "friction_scale": friction}
    elif method != "default":
        config["recrossing"] = {"dynamics": method}
    return {
        "config": config, "system": system, "chain": 1, "parent": 2, "pair": 3,
        "q": np.repeat(system.geometry[None], 4, axis=0), "xi_star": 1.,
        "child_steps": 12, "output": str(output),
    }


def test_trpmd_child_pair_has_opposite_initial_momenta_independent_noise_and_unit_initial_flux(tmp_path, monkeypatch):
    task = child_task(tmp_path)
    starts = []
    original = Dynamics.trpmd_step

    def record(self, state, *args, **kwargs):
        if state.step == 0:
            starts.append((state.p.copy(), copy.deepcopy(self.rng.bit_generator.state)))
        return original(self, state, *args, **kwargs)

    monkeypatch.setattr(Dynamics, "trpmd_step", record)
    result = rate._child_task(task)
    assert len(starts) == 2
    np.testing.assert_array_equal(starts[0][0], -starts[1][0])
    assert starts[0][1] != starts[1][1]
    assert result["denominator"] > 0
    np.testing.assert_allclose(result["numerator"] / result["denominator"], 1.)
    assert result["propagation"]["python_steps"] == 24
    # Completed-pair replay must not advance even one stochastic step.
    starts.clear()
    replay = rate._child_task(task)
    assert starts == []
    np.testing.assert_array_equal(replay["numerator"], result["numerator"])
    assert replay["denominator"] == result["denominator"]


def test_zero_friction_children_are_identical_to_unchanged_default_rpmd(tmp_path):
    rpmd = rate._child_task(child_task(tmp_path / "rpmd", method="default"))
    trpmd = rate._child_task(child_task(tmp_path / "trpmd", friction=0))
    for key in ("numerator", "denominator", "energy_drift",
                "endpoint_energy_error_hartree_per_bead"):
        np.testing.assert_array_equal(trpmd[key], rpmd[key])


def test_changed_friction_and_rpmd_reject_existing_trpmd_checkpoint(tmp_path):
    task = child_task(tmp_path)
    rate._child_task(task)
    changed = copy.deepcopy(task)
    changed["config"]["recrossing"]["friction_scale"] = 1.
    with pytest.raises(ValueError, match="checkpoint physics differs"):
        rate._child_task(changed)
    rpmd = child_task(tmp_path, method="default")
    with pytest.raises(ValueError, match="checkpoint physics differs"):
        rate._child_task(rpmd)


def test_default_rpmd_checkpoint_replays_with_execution_only_tuning(tmp_path):
    task = child_task(tmp_path, method="default")
    result = rate._child_task(task)
    task["config"]["parallel"] = {"reuse_potentials": False}
    task["config"]["recrossing"] = {"child_batch_pairs": 3}
    replay = rate._child_task(task)
    np.testing.assert_array_equal(replay["numerator"], result["numerator"])
    assert replay["denominator"] == result["denominator"]


@pytest.mark.parametrize("task_name,unit", [("rate", "cm3_s"), ("surface_rate", "s_inv")])
@pytest.mark.parametrize("energy_change", [0., 10.])
def test_trpmd_aggregate_reports_heat_and_keeps_formal_rpmd_rate_empty(
    tmp_path, task_name, unit, energy_change,
):
    options = {"child_steps": 40, "plateau_fraction": .5,
               "plateau_tolerance": .1, "bootstrap": 40}
    children = [{
        "chain": chain, "parent": parent, "pair": 0,
        "numerator": np.ones(40), "denominator": 1.,
        "energy_drift": 2., "endpoint_energy_error_hartree_per_bead": energy_change,
    } for chain in (0, 1) for parent in range(20)]
    config = {"task": task_name, "seed": 18, "beta": 2., "dt_au": .05,
              "recrossing": {"dynamics": "trpmd", "friction_scale": .5}}
    pmf = {"ready": True, "warnings": [], "k_qtst_" + unit: 12.,
           "free_energy_barrier_hartree": .01,
           "pmf_bootstrap_barriers": [.005, .015, .02, .025],
           "pmf_bootstrap_qtst": [11.9, 12., 12.1, 12.2],
           "bootstrap": {"requested": 4, "attempted": 4, "valid": 4,
                         "block_plan": {"status": "finite_block_estimate"}}}
    result = rate._aggregate(children, pmf, config, str(tmp_path), options)
    assert result["status"] == "trpmd_sensitivity_estimate"
    assert result["rate_" + unit] is None
    assert result["trpmd_rate_estimate_" + unit] == 12.
    assert result["centroid_friction"] == 0.
    assert result["energy_conservation_applicable"] is False
    assert result["maximum_endpoint_energy_change_kbt"] == 2 * energy_change
    assert result["converged"] is False
    assert not any("endpoint_energy_error" in key for key in result)
    assert "maximum_scaled_energy_drift" not in result
    warnings = " ".join(result["warnings"])
    assert "internal modes exchange heat" in warnings
    assert "error exceeds 0.1 kBT" not in warnings
    assert "legacy scaled endpoint drift" not in warnings
    if task_name == "surface_rate":
        assert not result["rate_quality"]["passed"]
        assert result["rate_quality"]["reasons"] == ["trpmd_sensitivity_is_not_an_rpmd_rate"]

    # Removing the TRPMD setting leaves the original NVE diagnostics/policy.
    config.pop("recrossing")
    rpmd = rate._aggregate(children, pmf, config, str(tmp_path), options)
    assert "trpmd_rate_estimate_" + unit not in rpmd
    assert rpmd["maximum_endpoint_energy_error_kbt"] == 2 * energy_change
    if energy_change:
        assert "error exceeds 0.1 kBT" in " ".join(rpmd["warnings"])
        assert rpmd["rate_" + unit] is None
    else:
        assert rpmd["rate_" + unit] == 12.


@pytest.mark.parametrize("raw,dynamics", [
    ({"dynamics": "cmd"}, {}),
    ({"dynamics": "trpmd", "friction_scale": -1}, {}),
    ({"dynamics": "trpmd", "friction_scale": True}, {}),
    ({"dynamics": "trpmd", "friction_scale": float("nan")}, {}),
    ({"dynamics": "trpmd", "child_batch_pairs": 2}, {}),
    ({"dynamics": "trpmd"}, {"loop_backend": "native"}),
    ({"dynamics": "rpmd", "friction_scale": 0}, {}),
])
def test_invalid_recrossing_modes_are_rejected_early(raw, dynamics):
    with pytest.raises(ValueError):
        rate._recrossing_dynamics({"recrossing": raw, "dynamics": dynamics})
