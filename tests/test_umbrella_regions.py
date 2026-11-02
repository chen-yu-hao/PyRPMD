"""Explicit window plans, RPMDrate coefficients and heterogeneous bias physics."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri

from pyrpmd.analysis.fes import reconstruct_fes, umbrella_integration
from pyrpmd.bias import UmbrellaBias
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import ANGSTROM_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows import fes
from pyrpmd.workflows.rate_pmf import configure_rate_umbrella
from pyrpmd.workflows.umbrella_plan import plan_preview, resolve_plan


def system():
    return System(("H",), np.array([2.3]), np.zeros((1, 3)))


def config():
    return {
        "schema_version": 2, "task": "fes", "beta": 1.7,
        "temperature_K": 1 / (1.7 * KB_AU), "beads": 3, "dt_au": .03,
        "seed": 903, "native": False,
        "potential": {"provider": "harmonic", "k": .4},
        "sampling": {"equilibration_steps": 6, "production_steps": 40,
                     "sample_interval": 2, "andersen_interval": 3, "replicas": 1},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"coordinate": "x", "force_constant": "1 hartree/bohr^2",
                     "windows": [{"id": "left", "center": -.2},
                                 {"id": "right", "center": .2,
                                  "force_constant": "3 hartree/bohr^2"}]},
        "analysis": {"bins": 5, "bootstrap": 0},
        "output": {"checkpoint_steps": 7},
    }


def rate_config():
    cfg = config()
    cfg.update(task="rate", temperature_K=1000, beta=1 / (1000 * KB_AU))
    cfg.pop("cvs")
    cfg["reaction"] = {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                       "breaking_bonds": [[1, 2]], "rinf": "8 bohr"}
    cfg["umbrella"] = {
        "force_constant": .00016,
        "regions": [
            {"id": "entrance", "start": 0, "stop": .8, "spacing": .05,
             "include_endpoints": "left"},
            {"id": "barrier", "start": .8, "stop": 1.05, "spacing": .01,
             "include_endpoints": "both", "force_constant": .00032},
        ],
    }
    atoms = System(("H", "H", "H"), np.ones(3),
                   np.array([[-2., 0, 0], [0., 0, 0], [2., 0, 0]]))
    return cfg, atoms


def test_regions_expand_density_and_force_together_without_pes(monkeypatch):
    cfg, atoms = rate_config()
    original = copy.deepcopy(cfg)
    monkeypatch.setattr(Dynamics, "__init__", lambda *a, **kw: pytest.fail("preflight used dynamics"))
    resolved = configure_rate_umbrella(cfg, atoms, cfg["reaction"])
    cv, _, windows = resolve_plan(resolved, atoms)
    assert cfg == original
    assert len(windows) == 42
    np.testing.assert_allclose([w.center[0] for w in windows[:16]], np.arange(16) * .05)
    np.testing.assert_allclose([w.center[0] for w in windows[16:]], .8 + np.arange(26) * .01)
    assert sum(w.center[0] == .8 for w in windows) == 1
    np.testing.assert_allclose([w.stiffness[0, 0] for w in windows], [.16] * 16 + [.32] * 26)
    assert len({w.identifier for w in windows}) == 42
    preview = plan_preview(resolved, atoms)
    assert preview["cv_units"] == ["dimensionless"]
    assert preview["windows"][16]["source"] == "barrier"
    assert preview["windows"][16]["stiffness_au"] == [[.32]]
    # The resolved physical K is the one in the actual ring-polymer bias.
    q = np.repeat(atoms.geometry[None], 3, axis=0)
    window = windows[16]
    bias = UmbrellaBias(cv, window.center, window.stiffness)
    delta = cv.value(q)[0] - window.center[0]
    assert bias.energy_force(q)[0] == pytest.approx(3 * .5 * .32 * delta**2)


@pytest.mark.parametrize("temperature", [300, 1000])
@pytest.mark.parametrize("force,normalized", [
    (.1, True), ("0.1 1/T", True), ("0.1 hartree/K", True),
    ("0.1 hartree", False),
])
def test_new_coefficients_multiply_temperature_but_physical_units_do_not(temperature, force, normalized):
    cfg, atoms = rate_config()
    cfg.update(temperature_K=temperature, beta=1 / (temperature * KB_AU))
    cfg["umbrella"] = {"centers": [0, 1], "force_constant": force}
    resolved = configure_rate_umbrella(cfg, atoms, cfg["reaction"])
    _, _, windows = resolve_plan(resolved, atoms)
    assert windows[0].stiffness[0, 0] == pytest.approx(.1 * temperature if normalized else .1)


def test_legacy_numeric_kappa_and_default_strength_are_preserved():
    cfg, atoms = rate_config()
    cfg.pop("schema_version")
    cfg["umbrella"] = {"centers": [0, 1], "kappa": .1}
    resolved = configure_rate_umbrella(cfg, atoms, cfg["reaction"])
    _, _, centers, stiffness = fes.resolve_umbrellas(resolved, atoms)
    np.testing.assert_array_equal(centers, [[0], [1]])
    np.testing.assert_array_equal(stiffness, [[.1]])
    assert all(w.identifier is None for w in resolve_plan(resolved, atoms)[2])
    cfg["umbrella"].pop("kappa")
    resolved = configure_rate_umbrella(cfg, atoms, cfg["reaction"])
    assert fes.resolve_umbrellas(resolved, atoms)[3][0, 0] == pytest.approx(50 / cfg["beta"])


@pytest.mark.parametrize("change,match", [
    ({"centers": [0, 1]}, "exactly one"),
    ({"stiffness": 1}, "conflicting force"),
    ({"regions": [{"start": 0, "stop": 1, "spacing": .3}]}, "integer multiple"),
    ({"regions": [{"start": 0, "stop": 1, "spacing": 0}]}, "positive spacing"),
    ({"regions": [{"start": 0, "stop": 1, "spacing": .5, "spacingg": .5}]}, "Unknown"),
    ({"regions": [{"start": 0, "stop": .8, "spacing": .4},
                  {"start": .6, "stop": 1, "spacing": .2}]}, "overlap"),
    ({"regions": [{"start": 0, "stop": .8, "spacing": .4},
                  {"start": .8, "stop": 1, "spacing": .1}]}, "duplicate umbrella center"),
    ({"force_constant": [.1, .2]}, "force_constant is scalar"),
])
def test_ambiguous_or_invalid_plans_fail_during_pure_preflight(change, match):
    cfg, atoms = rate_config()
    cfg["umbrella"].update(change)
    with pytest.raises(ValueError, match=match):
        configure_rate_umbrella(cfg, atoms, cfg["reaction"])


def test_unit_converted_region_endpoints_detect_duplicate_boundary():
    cfg = config()
    cfg["cvs"][0]["unit"] = "angstrom"
    cfg["umbrella"] = {"stiffness": 1, "regions": [
        {"start": 0, "stop": .8, "spacing": .05},
        {"start": ".8 angstrom", "stop": "1 angstrom", "spacing": ".1 angstrom"},
    ]}
    with pytest.raises(ValueError, match="duplicate umbrella center"):
        resolve_plan(cfg, system())
    cfg["umbrella"]["regions"][0]["include_endpoints"] = "left"
    windows = resolve_plan(cfg, system())[2]
    assert len(windows) == 19
    assert windows[16].center[0] == pytest.approx(.8 * ANGSTROM_TO_AU)


@pytest.mark.parametrize("backend", ["adaptive_umbrella", "replica_exchange"])
def test_heterogeneous_backends_reject_before_any_output(backend, tmp_path):
    cfg = config()
    cfg[backend] = {"enabled": True}
    with pytest.raises(ValueError, match="does not yet support heterogeneous"):
        fes.run_fes(cfg, system(), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_uniform_regions_remain_usable_for_adaptive_windows(tmp_path):
    cfg = config()
    cfg["umbrella"] = {"force_constant": "1 hartree/bohr^2", "regions": [
        {"start": -.2, "stop": .2, "spacing": .4},
    ]}
    cfg["adaptive_umbrella"] = {
        "enabled": True, "pilot_steps": 60, "pilot_equilibration_steps": 10,
        "max_rounds": 2, "max_windows": 8,
        "target_overlap": 1e-8, "min_effective_samples": 1,
    }
    sampled = fes.run_fes(cfg, system(), tmp_path, return_data=True)
    assert sampled["summary"]["adaptive_umbrella"]["target_met"]
    resumed = fes.run_fes(cfg, system(), tmp_path, return_data=True)
    for before, after in zip(sampled["streams"], resumed["streams"]):
        np.testing.assert_array_equal(before["samples"], after["samples"])


def test_per_window_recording_budget_is_validated_before_dynamics(monkeypatch):
    cfg = config()
    cfg["umbrella"]["windows"][1]["sampling"] = {
        "production": {"steps": 5}, "sample_every": {"steps": 3},
    }
    monkeypatch.setattr(Dynamics, "__init__", lambda *a, **kw: pytest.fail("preflight used dynamics"))
    with pytest.raises(ValueError, match="at least two saved samples"):
        fes.validate_fes_request(cfg, system())


def test_window_durations_use_actual_inherited_sampling_timestep():
    cfg = config()
    cfg["sampling"]["dt_au"] = .06
    cfg["umbrella"]["windows"][1]["sampling"] = {
        "production": "1.2 autime", "sample_every": ".18 autime",
    }
    windows = resolve_plan(cfg, system())[2]
    assert windows[1].sampling["production_steps"] == 20
    assert windows[1].sampling["interval_steps"] == 3


def test_full_matrix_preserves_cv_axes_while_windows_select_states():
    cfg = config()
    cfg["cvs"].append({"name": "y", "kind": "cartesian", "atom": 0, "axis": "y"})
    cfg["umbrella"] = {"grid": [[-.2, .2], [-.2, .2]],
                       "force_matrix": [["1 hartree/bohr^2", ".2 hartree/bohr^2"],
                                        [".2 hartree/bohr^2", "2 hartree/bohr^2"]]}
    _, _, centers, stiffness = fes.resolve_umbrellas(cfg, system())
    assert centers.shape == (4, 2)
    np.testing.assert_array_equal(stiffness, [[1, .2], [.2, 2]])
    cfg["umbrella"]["force_matrix"][1][0] = ".1 hartree/bohr^2"
    with pytest.raises(ValueError, match="symmetric"):
        resolve_plan(cfg, system())


def test_ui_and_mbar_recover_harmonic_pmf_with_different_stiffnesses():
    centers = np.linspace(-1.2, 1.2, 5)
    stiffness = np.array([.5, 2., 8., 1.2, 3.])
    physical, beta = .7, 1.3
    means = stiffness * centers / (physical + stiffness)
    deviations = 1 / np.sqrt(beta * (physical + stiffness))
    # Exact finite-sample Gaussian moments isolate bias subtraction from noise.
    z = ndtri((np.arange(8000) + .5) / 8000)
    z = (z - z.mean()) / z.std()
    exact = [mean + deviation * z for mean, deviation in zip(means, deviations)]
    grid = np.linspace(-1.5, 1.5, 101)
    curve, derivative = umbrella_integration(exact, centers, stiffness, beta, grid)
    np.testing.assert_allclose(derivative, physical * grid, atol=2e-14)
    np.testing.assert_allclose(curve, .5 * physical * grid**2, atol=2e-14)
    rng = np.random.default_rng(591)
    samples = [rng.normal(mean, deviation, (12000, 1))
               for mean, deviation in zip(means, deviations)]
    reconstructed = reconstruct_fes(samples, centers[:, None], stiffness[:, None, None],
                                    beta, bins=24, ranges=[[-1.5, 1.5]], bootstrap=0)
    axis = .5 * (reconstructed["edges"][0][:-1] + reconstructed["edges"][0][1:])
    residual = reconstructed["free_energy"] - .5 * physical * axis**2
    assert reconstructed["connected"]
    assert not reconstructed["mask"].any()
    assert residual.std() < .04


@pytest.mark.parametrize("native", [False, True])
def test_actual_heterogeneous_trajectories_metadata_reanalysis_and_exact_resume(tmp_path, monkeypatch, native):
    cfg = config()
    cfg["native"] = native
    cfg["umbrella"]["windows"][1]["sampling"] = {
        "production": {"steps": 24}, "sample_every": {"steps": 3},
        "chains_per_window": 2,
    }
    observed, original = [], UmbrellaBias.energy_force

    def record(bias, q):
        observed.append(float(bias.stiffness[0, 0]))
        return original(bias, q)

    with monkeypatch.context() as patch:
        patch.setattr(UmbrellaBias, "energy_force", record)
        sampled = fes.run_fes(cfg, system(), tmp_path, return_data=True)
    assert set(observed) == {1., 3.}
    assert [len(s["samples"]) for s in sampled["streams"]] == [20, 8, 8]
    assert [s["stiffness"][0, 0] for s in sampled["streams"]] == [1., 3., 3.]
    checkpoints = {}
    for stream in sampled["streams"]:
        path = Path(stream["checkpoint"])
        checkpoints[path] = path.read_bytes()
        with load_checkpoint(path, require_integrity=True) as saved:
            assert str(saved["window_id"]) == stream["window_id"]
            np.testing.assert_array_equal(saved["stiffness"], stream["stiffness"])
    saved_plan = json.loads((tmp_path / "umbrella-plan.json").read_text())
    assert [w["stiffness_au"] for w in saved_plan["windows"]] == [[[1.]], [[3.]]]
    assert saved_plan["windows"][1]["raw_samples_per_chain"] == 8
    again = fes.reanalyze_fes(cfg, system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(again["data"]["free_energy"], sampled["data"]["free_energy"])
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "sample_step", lambda *a, **k: pytest.fail("completed resume propagated"))
        resumed = fes.run_fes(cfg, system(), tmp_path, return_data=True)
    for before, after in zip(sampled["streams"], resumed["streams"]):
        np.testing.assert_array_equal(before["samples"], after["samples"])
    assert all(path.read_bytes() == raw for path, raw in checkpoints.items())
    cfg["umbrella"]["windows"][1]["force_constant"] = "4 hartree/bohr^2"
    with pytest.raises(RuntimeError, match="checkpoint physics differs"):
        fes.reanalyze_fes(cfg, system(), tmp_path)
    plan_bytes = (tmp_path / "umbrella-plan.json").read_bytes()
    with pytest.raises(RuntimeError, match="umbrella plan differs"):
        fes.run_fes(cfg, system(), tmp_path)
    assert (tmp_path / "umbrella-plan.json").read_bytes() == plan_bytes
    assert all(path.read_bytes() == raw for path, raw in checkpoints.items())


def test_adding_a_window_does_not_renumber_existing_random_streams(tmp_path):
    cfg = config()
    first = fes.run_fes(cfg, system(), tmp_path / "first", return_data=True)
    cfg["umbrella"]["windows"].insert(0, {"center": -.4})
    second = fes.run_fes(cfg, system(), tmp_path / "second", return_data=True)
    by_id = {stream["window_id"]: stream for stream in second["streams"]}
    for stream in first["streams"]:
        np.testing.assert_array_equal(stream["samples"], by_id[stream["window_id"]]["samples"])


def test_heterogeneous_windows_and_lengths_are_identical_with_trajectory_batching(tmp_path):
    cfg = config()
    cfg["umbrella"]["windows"][1]["sampling"] = {
        "production": {"steps": 24}, "sample_every": {"steps": 3},
    }
    scalar = fes.run_fes(cfg, system(), tmp_path / "scalar", return_data=True)
    cfg["parallel"] = {"trajectory_batch_size": 2}
    batched = fes.run_fes(cfg, system(), tmp_path / "batched", return_data=True)
    for before, after in zip(scalar["streams"], batched["streams"]):
        for key in ("samples", "energies", "last_q", "last_p"):
            np.testing.assert_array_equal(before[key], after[key])
