"""End-to-end composition of independent PMF and recrossing stages.

The teaching exchange PES and deliberately short trajectories test routing,
clock separation, restart and support gates. They do not establish a physical
reaction rate or convergence of an enhanced-sampling calculation.
"""

import importlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.fes_result import FESResult, write_fes_result
from pyrpmd.analysis.frozen_bias import frozen_bias_fes
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import JobSpec
from pyrpmd.runner import prepare, run_resolved
from pyrpmd.units import parse_quantity
from pyrpmd.workflows.rate_curve import estimate_fes_curve


def _spec(method):
    source = Path(__file__).resolve().parents[1] / "examples/rate/pilot.inp"
    spec = JobSpec.from_file(source)
    raw = spec.data
    raw["native"] = False
    raw["parallel"] = {"backend": "serial"}
    raw["dynamics"].update(integrator="cayley-bcb", loop_backend="python", block_steps=7)
    raw["sampling"]["thermostat"] = "andersen"
    umbrellas = raw.pop("umbrella")
    raw.pop("analysis", None)
    raw["recrossing"] = {
        "parent_replicas": 1, "parent_samples": 2, "child_pairs": 1,
        "equilibration_steps": 4, "parent_interval_steps": 2,
        "child_steps": 8, "bootstrap": 4,
    }
    pmf = {"method": method,
           "dynamics": {"integrator": "normal-mode-bab", "timestep": "0.05 fs", "loop_backend": "python", "block_steps": 11},
           "analysis": {"bins": 8, "ranges": [[-0.3, 1.3]], "bootstrap": 0, "min_count": 1}}
    if method == "umbrella":
        pmf.update(estimator="mbar", umbrella=umbrellas,
                   sampling={"thermostat": "pile-l", "gamma_centroid": "0.02 fs^-1",
                             "equilibration": "5 fs", "production": "20 fs", "interval": "0.2 fs"})
    else:
        pmf.update(estimator="reweighted_histogram",
                   sampling={"thermostat": "andersen", "equilibration_steps": 2,
                             "production_steps": 32, "sample_interval": 2,
                             "andersen_interval": 4, "replicas": 1})
        pmf[method] = {"sigma": 0.1, "pace": 4, "freeze_after_steps": 12,
                       "freeze_equilibration_steps": 4,
                       **({"mode": "standard", "height": "0.0001 hartree"}
                          if method == "metadynamics" else {"barrier": "0.001 hartree"})}
    raw["rate"] = {"dividing_surface": 1.0, "pmf": pmf}
    return spec


def _arrays(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def test_independent_umbrella_operator_and_clock_reach_real_recrossing_and_resume(tmp_path, monkeypatch):
    spec = _spec("umbrella")
    config, system = spec.resolve()
    constructions = []
    original = Dynamics.__init__

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        caller = inspect.currentframe().f_back.f_code.co_name
        constructions.append((caller, self.dt, self.integrator_name))

    monkeypatch.setattr(Dynamics, "__init__", observe)
    output = tmp_path/"run"
    result = run_resolved(config, system, output=output)
    assert result["pmf"]["ready"]
    assert result["child_trajectories"] == 4
    assert result["pmf"]["sampler"] == "umbrella"
    stage_dt, root_dt = parse_quantity("0.05 fs", "time"), parse_quantity("0.1 fs", "time")
    windows = [item for item in constructions if item[0] == "_sample_window"]
    assert len(windows) == 22
    assert all(dt == stage_dt and integrator == "normal-mode-bab" for _, dt, integrator in windows)
    for caller in ("_parent_task", "_child_task"):
        instances = [item for item in constructions if item[0] == caller]
        assert instances
        assert all(dt == root_dt and integrator == "cayley-bcb" for _, dt, integrator in instances)
    stage = json.loads((output/"input/pmf-resolved.json").read_text())
    assert stage["sampling"]["thermostat"] == "pile-l"
    assert stage["sampling"]["integrator"] == "baoab"
    with load_checkpoint(output/"umbrella/checkpoints/window_00000_replica_0000.npz") as saved:
        assert int(saved["step"]) == 500
        assert float(saved["time"]) == pytest.approx(parse_quantity("25 fs", "time"))
    with load_checkpoint(output/"checkpoints/parent_0000.npz") as saved:
        assert int(saved["step"]) == 8
        assert float(saved["time"]) == pytest.approx(8*root_dt)
        assert saved["snapshots"].shape[0] == 2
    recrossing = _arrays(output/"recrossing.npz")
    np.testing.assert_allclose(recrossing["time_au"], np.arange(1, 9)*root_dt)

    before = {name: _arrays(output/name) for name in ("rate_pmf.npz", "recrossing.npz")}
    repeated = run_resolved(config, system, output=output, resume=True)
    assert repeated["kappa"] == result["kappa"]
    assert repeated["pmf"] == result["pmf"]
    for name, arrays in before.items():
        for key, values in _arrays(output/name).items():
            np.testing.assert_array_equal(values, arrays[key])

    execution = _spec("umbrella")
    execution.data["dynamics"].update(loop_backend="auto", block_steps=3)
    execution.data["rate"]["pmf"]["dynamics"].update(loop_backend="auto", block_steps=5)
    tuned_config, tuned_system = execution.resolve()
    tuned = run_resolved(tuned_config, tuned_system, output=output, resume=True)
    assert tuned["kappa"] == result["kappa"]
    np.testing.assert_array_equal(_arrays(output/"rate_pmf.npz")["free_energy"], before["rate_pmf.npz"]["free_energy"])

    changed = _spec("umbrella")
    changed.data["rate"]["pmf"]["sampling"]["gamma_centroid"] = "0.03 fs^-1"
    changed_config, changed_system = changed.resolve()
    with pytest.raises((ValueError, RuntimeError), match="changed scientific"):
        run_resolved(changed_config, changed_system, output=output, resume=True)


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_actual_enhanced_rate_sampling_with_insufficient_support_stops_before_recrossing(method, tmp_path, monkeypatch):
    import pyrpmd.workflows.rate as rate

    config, system = _spec(method).resolve()
    monkeypatch.setattr(rate, "_parent_task", lambda task: pytest.fail("unsupported PMF must not launch parents"))
    monkeypatch.setattr(rate, "_child_task", lambda task: pytest.fail("unsupported PMF must not launch children"))
    result = run_resolved(config, system, output=tmp_path/"run")
    assert result["status"] == "unsupported_pmf"
    assert result["pmf"]["sampler"] == method
    assert not result["pmf"]["ready"]
    assert result["rate_estimate_cm3_s"] is None
    output = tmp_path/"run"
    assert not list((output/"checkpoints").glob("parent_*.npz"))
    surface = FESResult.read(output/"pmf")
    assert surface.method == method and surface.measure == "rate_flux"
    assert surface.metadata["reweighted"] is True
    assert surface.mask.any()
    with load_checkpoint(output/"pmf/checkpoints"/(method+"_0000.npz")) as saved:
        assert int(saved["step"]) == 34
        assert float(saved["time"]) == pytest.approx(34*parse_quantity("0.05 fs", "time"))
    resumed = run_resolved(config, system, output=output, resume=True)
    assert resumed["pmf"] == result["pmf"]


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_top_level_frozen_pmf_table_preserves_preparation_and_actual_sampling(method, tmp_path):
    nested = _spec(method)
    top = deepcopy(nested)
    top.data[method] = top.data["rate"]["pmf"].pop(method)
    expected_plan = prepare(nested, numerical=True)
    actual_plan = prepare(top, numerical=True)
    assert actual_plan["ready"] and expected_plan["ready"]
    assert actual_plan["pmf_plan"] == expected_plan["pmf_plan"]
    nested_config, nested_system = nested.resolve()
    top_config, top_system = top.resolve()
    expected = run_resolved(nested_config, nested_system, output=tmp_path / "nested")
    actual = run_resolved(top_config, top_system, output=tmp_path / "top")
    assert actual["pmf"] == expected["pmf"]
    checkpoint = Path("pmf/checkpoints") / (method + "_0000.npz")
    expected_arrays = _arrays(tmp_path / "nested" / checkpoint)
    actual_arrays = _arrays(tmp_path / "top" / checkpoint)
    assert actual_arrays.keys() == expected_arrays.keys()
    for key in expected_arrays:
        np.testing.assert_array_equal(actual_arrays[key], expected_arrays[key], err_msg=key)


def _complete_frozen_sampler(method):
    """A supported stationary FES fixture; parent/child dynamics stay real.

    Uniform biased xi draws under V=-a*xi give a target PMF a*xi. This tests
    the estimator/recrossing contract independently of global exploration by
    the deliberately short real samplers in the preceding test.
    """
    def worker(config, system, output, executor=None, return_data=False):
        assert config["analysis"]["measure"] == "rate_flux"
        assert config["analysis"]["method"] == "reweighted_histogram"
        edges = np.linspace(-0.3, 1.3, 9)
        values = np.tile((edges[1:]+edges[:-1])/2, 20)[:, None]
        state = State(np.repeat(system.geometry[None], config["beads"], axis=0),
                      np.zeros((config["beads"], system.natoms, 3)))
        stream = {"samples": values, "sample_steps": np.arange(100, 100+len(values)),
                  "sample_bias": -0.003*values[:, 0], "state": state, "complete": True}
        data = frozen_bias_fes([stream], config["beta"], bins=[8], ranges=[[-0.3, 1.3]],
                               periods=[None], widths=[0.1], equilibration_steps=2,
                               freeze_after_steps=12, freeze_equilibration_steps=4)
        metadata = {"method": method, "estimator": "reweighted_histogram",
                    "measure": "rate_flux", "reweighted": True, "beta": config["beta"],
                    "warnings": [], "converged": False}
        surface = FESResult.from_data(data, method=method, cv_units=("dimensionless",), metadata=metadata)
        summary = write_fes_result(output, surface, metadata)
        return {"summary": summary, "data": data, "streams": [stream]}
    return worker


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_supported_frozen_fes_continues_real_constrained_parents_and_paired_children(method, tmp_path, monkeypatch):
    workflow = importlib.import_module("pyrpmd.workflows."+method)
    monkeypatch.setattr(workflow, "run_"+method, _complete_frozen_sampler(method))
    config, system = _spec(method).resolve()
    output = tmp_path/"run"
    result = run_resolved(config, system, output=output)
    assert result["pmf"]["ready"]
    assert result["pmf"]["method"] == "reweighted_histogram"
    assert result["pmf"]["free_energy_barrier_hartree"] == pytest.approx(0.003)
    assert result["child_trajectories"] == 4
    assert np.isfinite(result["kappa"])
    assert result["rate_estimate_cm3_s"] == pytest.approx(result["pmf"]["k_qtst_cm3_s"]*result["kappa"])
    assert result["pmf"]["barrier_standard_error_hartree"] is None
    assert result["rate_standard_error_cm3_s"] is None
    assert not result["rate_uncertainty"]["available"]
    with load_checkpoint(output/"checkpoints/parent_0000.npz") as parent:
        assert parent["snapshots"].shape[0] == 2
        assert int(parent["step"]) == 8
        assert float(parent["time"]) == pytest.approx(8*config["dt_au"])
    assert len(list((output/"checkpoints").glob("child_*.npz"))) == 2
    repeated = run_resolved(config, system, output=output, resume=True)
    assert repeated["kappa"] == result["kappa"]
    np.testing.assert_allclose(_arrays(output/"recrossing.npz")["time_au"], np.arange(1, 9)*config["dt_au"])


def test_automatic_peak_search_rejects_a_hole_beyond_the_selected_peak():
    # Centers are [-0.1, 0, ..., 1.1]; the peak is at 0.4, a hole at 0.9.
    edges = np.arange(-0.15, 1.16, 0.1)
    grid = (edges[1:]+edges[:-1])/2
    values = 1.-(grid-0.4)**2
    mask = np.isclose(grid, 0.9)
    surface = FESResult(edges=(edges,), free_energy=values, uncertainty=np.full_like(values, np.nan),
                        mask=mask, cv_units=("dimensionless",), method="metadynamics",
                        uncertainty_kind="replica_spread", metadata={"measure": "rate_flux", "reweighted": True})
    options = {"search_range": [0., 1.], "xi_star": None}
    automatic = estimate_fes_curve(surface, options, uncertainty_reason="test", strict_search=True)
    assert automatic["xi_star"] == pytest.approx(0.4)
    assert not automatic["ready"]
    assert not automatic["support"]["complete_search_range"]
    explicit = estimate_fes_curve(surface, {**options, "xi_star": 0.4},
                                  uncertainty_reason="test", strict_search=True)
    assert explicit["ready"]
    assert explicit["free_energy_barrier_hartree"] == pytest.approx(0.16)
    crossing_hole = estimate_fes_curve(surface, {**options, "xi_star": 1.0},
                                       uncertainty_reason="test", strict_search=True)
    assert not crossing_hole["ready"]
