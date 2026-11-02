"""History-dependent biases reuse the common canonical bath/checkpoint state."""

import copy
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from pyrpmd.bias import MetadynamicsBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import JobSpec
from pyrpmd.opes import OPESBias
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.workflows import metadynamics, opes
from pyrpmd.workflows.sampling import fes_sampler

from test_metadynamics import config as meta_config, system
from test_opes import config as opes_config


BATHS = {
    "nhc": {"tau": .4, "chain_length": 3},
    "svr": {"tau": .4},
    "gle": {"gle": {"A": [[.7, .1], [-.1, 1.2]], "C": np.eye(2).tolist()}},
}


def _task(workflow, output, mode):
    cfg = meta_config() if workflow == "metadynamics" else opes_config()
    cfg["sampling"].update(thermostat=mode, thermostat_params=copy.deepcopy(BATHS[mode]))
    return {"config": cfg, "system": system(), "output": str(output), "replica": 0}


@pytest.mark.parametrize("workflow", ["metadynamics", "opes"])
@pytest.mark.parametrize("mode", list(BATHS))
@pytest.mark.parametrize("failure", ["force", "history"])
def test_composed_baths_restart_exactly_after_force_or_history_failure(
        tmp_path, monkeypatch, workflow, mode, failure):
    module = metadynamics if workflow == "metadynamics" else opes
    full = module._sample_replica(_task(workflow, tmp_path / "full", mode))
    task = _task(workflow, tmp_path / "restart", mode)
    with monkeypatch.context() as patch:
        if failure == "history":
            cls, name = (MetadynamicsBias, "deposit") if workflow == "metadynamics" else (OPESBias, "update")
            original = getattr(cls, name)

            def fail_after_update(self, values, **kwargs):
                result = original(self, values, **kwargs)
                if kwargs["step"] == 12:
                    raise RuntimeError("injected history failure")
                return result

            patch.setattr(cls, name, fail_after_update)
        else:
            original_sample, original_force = Dynamics.sample_step, HarmonicPotential.evaluate
            request = {"armed": False, "calls": 0}

            def arm_second_force(self, state, *args, **kwargs):
                request["armed"], request["calls"] = state.step == 11, 0
                return original_sample(self, state, *args, **kwargs)

            def fail_second_force(self, q):
                if request["armed"]:
                    request["calls"] += 1
                    if request["calls"] == 2:
                        raise RuntimeError("injected force failure")
                return original_force(self, q)

            patch.setattr(Dynamics, "sample_step", arm_second_force)
            patch.setattr(HarmonicPotential, "evaluate", fail_second_force)
        with pytest.raises(RuntimeError, match="injected"):
            module._sample_replica(task)
    checkpoint = tmp_path / "restart/checkpoints" / f"{workflow}_0000.npz"
    with load_checkpoint(checkpoint) as data:
        assert int(data["step"]) == int(data["elapsed"]) == 11
        operators = json.loads(str(data["operator_state"]))
        assert operators["sampling"]
    resumed = module._sample_replica(task)
    for name in ("q", "p"):
        np.testing.assert_array_equal(getattr(full["state"], name), getattr(resumed["state"], name))
    assert (full["state"].step, full["state"].time) == (resumed["state"].step, resumed["state"].time)
    for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(full[key], resumed[key])
    for key in ("rng", "operator_state", "propagation",
                "hills" if workflow == "metadynamics" else "bias_state"):
        assert full[key] == resumed[key]


@pytest.mark.parametrize("workflow", ["metadynamics", "opes"])
@pytest.mark.parametrize("missing", ["operator_state", "auxiliary"])
def test_new_sampling_checkpoint_cannot_silently_restart_its_bath(tmp_path, workflow, missing):
    module = metadynamics if workflow == "metadynamics" else opes
    task = _task(workflow, tmp_path, "nhc")
    module._sample_replica(task)
    checkpoint = tmp_path / "checkpoints" / f"{workflow}_0000.npz"
    with load_checkpoint(checkpoint) as data:
        payload = {key: data[key].copy() for key in data.files if key != "_pyrpmd_checkpoint"}
    if missing == "operator_state":
        payload.pop("operator_state")
    else:
        state = json.loads(str(payload["operator_state"]))
        state["sampling"] = {}
        payload["operator_state"] = json.dumps(state)
    atomic_savez(checkpoint, kind="arrays", **payload)
    with pytest.raises(ValueError, match="missing.*state"):
        module._sample_replica(task)


@pytest.mark.parametrize("workflow,mode", [
    ("metadynamics", "andersen"), ("metadynamics", "pile-l"),
    ("metadynamics", "nose-hoover-chain"), ("opes", "andersen"),
])
def test_legacy_bias_workflows_keep_their_checkpoint_payload(tmp_path, workflow, mode):
    cfg = meta_config() if workflow == "metadynamics" else opes_config()
    cfg["sampling"]["thermostat"] = mode
    if workflow == "metadynamics":
        cfg["sampling"]["thermostat_params"] = {"tdamp": .4, "reset_after_equilibration": True}
    module = metadynamics if workflow == "metadynamics" else opes
    result = module._sample_replica({"config": cfg, "system": system(), "output": str(tmp_path), "replica": 0})
    assert "operator_state" not in result
    with load_checkpoint(tmp_path / "checkpoints" / f"{workflow}_0000.npz") as data:
        assert "operator_state" not in data
        if workflow == "metadynamics":
            assert (json.loads(str(data["thermostat_state"])) is not None) == (mode == "nose-hoover-chain")


@pytest.mark.parametrize("workflow", ["metadynamics", "opes"])
@pytest.mark.parametrize("mode", list(BATHS) + ["pile-g", "ffl", "pile-l"])
def test_frontend_resolves_shared_sampling_for_enhanced_biases(workflow, mode):
    filename = "metadynamics/standard.inp" if workflow == "metadynamics" else "opes/harmonic_2d.inp"
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "examples" / filename).read_text())
    parameters = copy.deepcopy(BATHS.get(mode, {"tau": .4} if mode == "pile-g" else
                                        {"gamma": .2} if mode == "ffl" else {"pile_scale": 1.2}))
    cfg["sampling"].update(thermostat=mode, thermostat_params=parameters)
    before = copy.deepcopy(cfg)
    resolved, _ = JobSpec.from_mapping(cfg).resolve()
    assert cfg == before
    assert resolved["sampling"]["integrator"] == ("obcbo" if mode == "nhc" else "bcocb")
    assert fes_sampler(resolved).thermostat_spec["mode"] == mode


def test_opes_can_use_the_existing_numpy_pile_stream(tmp_path):
    cfg = opes_config()
    cfg["sampling"]["thermostat"] = "pile-l"
    result = opes._sample_replica({"config": cfg, "system": system(), "output": str(tmp_path), "replica": 0})
    assert result["state"].step == 27
    assert "operator_state" not in result
