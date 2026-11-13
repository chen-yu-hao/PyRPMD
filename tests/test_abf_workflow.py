"""Public dispatch, science identities and exact interrupted ABF replay."""

import copy
import json
import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare, run, analyze_run, resume_run
from pyrpmd.workflows.abf import run_abf
from pyrpmd.workflows.common import _fingerprint
from pyrpmd.workflows.sampling import FESSampler


def job(method="abf", thermostat="andersen"):
    table = {"ranges_au": [[-1., 1.]], "bins": 9, "pace": 3, "full_samples": 3, "wall_stiffness_au": 1.}
    if method == "eabf":
        table.update(coupling_au=2., extended_mass_au=3., friction_au=.1)
    return {"task": "fes", "temperature": 300, "beads": 4, "seed": 19, "native": False,
            "system": {"symbols": ["H"], "masses": [1.], "geometry": [[.1, 0, 0]]},
            "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .1}},
            "dynamics": {"timestep": .02}, "output": {"checkpoint_steps": 5},
            "fes": {"method": method, "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
                    method: table, "sampling": {"thermostat": thermostat, "equilibration_steps": 2,
                                                  "production_steps": 23, "interval_steps": 2,
                                                  "andersen_interval": 3},
                    "analysis": {"min_count": 1}}}


@pytest.mark.parametrize("method", ["abf", "eabf"])
def test_public_run_reanalyze_prepare_and_resume(tmp_path, method):
    config = job(method)
    assert prepare(config)[method + "_preparation"]["bins"] == 9
    output = tmp_path/method
    summary = run(config, output=output)
    assert summary["method"] == method
    assert summary["samples"] == 23
    assert (output/"fes.npz").is_file()
    assert analyze_run(output)["method"] == method
    assert resume_run(output)["method"] == method


@pytest.mark.parametrize("method", ["abf", "eabf"])
@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_interruption_restores_every_state_and_rng_bit(tmp_path, monkeypatch, method, thermostat):
    config, system = JobSpec.from_mapping(job(method, thermostat)).resolve()
    full, split = tmp_path/"full", tmp_path/"split"
    run_abf(config, system, full)
    original = FESSampler.step
    def fail(self, dynamics, state, bias):
        if state.step == 7:
            # Mutate through a complete stochastic step before simulating an
            # interrupt. Workflow must restore MD, baths, RNG and auxiliaries.
            original(self, dynamics, state, bias)
            raise KeyboardInterrupt("injected interruption")
        return original(self, dynamics, state, bias)
    with monkeypatch.context() as patch:
        patch.setattr(FESSampler, "step", fail)
        with pytest.raises(KeyboardInterrupt):
            run_abf(config, system, split)
    run_abf(config, system, split)
    with load_checkpoint(full/"checkpoints"/f"{method}_0000.npz") as first, load_checkpoint(
            split/"checkpoints"/f"{method}_0000.npz") as second:
        for key in ("q", "p", "step", "time", "operators", "bias_state", "samples", "sample_steps", "fingerprint"):
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)


@pytest.mark.parametrize("method", ["abf", "eabf"])
def test_scientific_parameters_change_identity_and_incompatible_checkpoint_rejected(tmp_path, method):
    raw = job(method)
    config, system = JobSpec.from_mapping(raw).resolve()
    first = _fingerprint(config, system, {})
    changed = copy.deepcopy(raw)
    changed["fes"][method]["pace"] = 4
    other, _ = JobSpec.from_mapping(changed).resolve()
    assert first != _fingerprint(other, system, {})
    run_abf(config, system, tmp_path)
    with pytest.raises(ValueError, match="physics"):
        run_abf(other, system, tmp_path)


def test_checkpoint_clock_is_validated_before_propagation(tmp_path):
    config, system = JobSpec.from_mapping(job()).resolve()
    run_abf(config, system, tmp_path)
    path = tmp_path/"checkpoints"/"abf_0000.npz"
    with load_checkpoint(path) as saved:
        arrays = {k: saved[k].copy() for k in saved.files if k != "_pyrpmd_checkpoint"}
    bias = json.loads(str(arrays["bias_state"]))
    bias["refreshes"] += 1
    arrays["bias_state"] = np.asarray(json.dumps(bias))
    atomic_savez(path, kind="abf_stream", **arrays)
    with pytest.raises(ValueError, match="clock"):
        run_abf(config, system, tmp_path)


@pytest.mark.parametrize("mutation", [
    lambda j: j["fes"]["abf"].update(derivative="omit"),
    lambda j: j["fes"]["abf"].update(full_samples=True),
    lambda j: j["fes"]["analysis"].update(measure="rate_flux"),
    lambda j: j["fes"]["analysis"].update(bins=3),
    lambda j: j["fes"]["sampling"].update(andersen_schedule="geometric"),
    lambda j: j["fes"]["sampling"].update(thermostat_params={"gamma": .1}),
    lambda j: j["fes"]["sampling"].update(parents=11),
    lambda j: j["fes"]["analysis"].update(max_grid_points=1),
    lambda j: j["output"].update(trajectory_stride=13),
    lambda j: j["dynamics"].update(loop_backend="native"),
    lambda j: j["fes"]["cvs"][0].update(estimator="bead_average"),
])
def test_unsupported_options_fail_explicitly(mutation):
    raw = job()
    mutation(raw)
    with pytest.raises((InputError, ValueError)):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["abf", "eabf"])
def test_redundant_cvs_fail_before_pes_or_output(tmp_path, monkeypatch, method):
    raw = job(method)
    config, system = JobSpec.from_mapping(raw).resolve()
    raw["fes"]["cvs"].append({"name": "also_x", "kind": "cartesian", "atom": 0, "axis": "x"})
    raw["fes"][method]["ranges_au"] = [[-1., 1.], [-1., 1.]]
    config["cvs"] = raw["fes"]["cvs"]
    config[method]["ranges_au"] = raw["fes"][method]["ranges_au"]

    def unexpected_provider(*args, **kwargs):
        pytest.fail("rank-deficient CVs must be rejected before PES construction")

    monkeypatch.setattr("pyrpmd.workflows.abf.make_task_potential", unexpected_provider)
    with pytest.raises(InputError, match="full rank"):
        prepare(raw)
    with pytest.raises(ValueError, match="full rank"):
        run_abf(config, system, tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()
