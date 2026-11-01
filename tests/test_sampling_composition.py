"""Canonical sampling components survive workflow observation and exchange failures."""
import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.units import parse_quantity
from pyrpmd.workflows.fes import run_fes
from pyrpmd.workflows.sampling import fes_sampler, thermostat_spec
import pyrpmd.workflows.replica_exchange as rex


def _config(mode):
    parameters = {
        "nhc": {"tau": .7, "chain_length": 3, "substeps": 2},
        "svr": {"tau": .7},
        "pile-g": {"tau": .7, "pile_scale": 1.0},
        "gle": {"gle": {"A": [[.7, .1], [-.1, .9]], "C": np.eye(2).tolist()}},
        "nm-gle": {"gle": {"A": [[.7, .1], [-.1, .9]], "C": np.eye(2).tolist()}},
    }[mode]
    return {
        "task": "fes", "beta": 1.7, "beads": 4, "dt_au": .02,
        "seed": 903, "native": False, "dynamics": {"integrator": "cayley-bcb"},
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .4}},
        "sampling": {"thermostat": mode, "thermostat_params": parameters,
                     "equilibration_steps": 4, "production_steps": 20,
                     "sample_interval": 2, "andersen_interval": 7},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"centers": [[-.2], [.2]], "stiffness": .8},
        "analysis": {"bins": 3, "bootstrap": 0},
        "output": {"checkpoint_steps": 7},
    }


def _system():
    return System(("H",), np.array([2.3]), np.zeros((1, 3)))


def _payload(path):
    with load_checkpoint(path, require_integrity=True) as saved:
        return {key: saved[key] for key in saved.files}


def _assert_checkpoints_equal(reference, resumed):
    paths = sorted((reference / "checkpoints").glob("*.npz"))
    assert paths
    for path in paths:
        a, b = _payload(path), _payload(resumed / "checkpoints" / path.name)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("mode", ["nhc", "svr", "pile-g", "gle", "nm-gle"])
def test_umbrella_observation_failure_restores_auxiliaries_and_exact_restart(tmp_path, monkeypatch, mode):
    pytest.importorskip("pyrpmd._thermal")
    cfg = _config(mode)
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    run_fes(cfg, _system(), reference)
    energy = Dynamics.energy
    failed = False

    def fail_after_accepted_operator_step(dynamics, state, bias=None):
        nonlocal failed
        if state.step == 10 and not failed:
            failed = True
            raise RuntimeError("injected observation failure after thermostat")
        return energy(dynamics, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "energy", fail_after_accepted_operator_step)
        with pytest.raises(RuntimeError, match="after thermostat"):
            run_fes(cfg, _system(), resumed)
    partial = _payload(resumed / "checkpoints/window_00000_replica_0000.npz")
    assert int(partial["elapsed"]) == 9
    assert isinstance(json.loads(str(partial["operator_state"])), dict)
    run_fes(cfg, _system(), resumed)
    _assert_checkpoints_equal(reference, resumed)


@pytest.mark.parametrize("mode", ["nhc", "svr", "gle"])
def test_replica_exchange_failure_restores_every_window_operator(tmp_path, monkeypatch, mode):
    pytest.importorskip("pyrpmd._thermal")
    cfg = _config(mode)
    cfg["replica_exchange"] = {"enabled": True, "interval_steps": 5}
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    run_fes(cfg, _system(), reference)
    exchange = rex._exchange_sweep
    calls = 0

    def fail_after_exchange(*args, **kwargs):
        nonlocal calls
        result = exchange(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected exchange failure after swap")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(rex, "_exchange_sweep", fail_after_exchange)
        with pytest.raises(RuntimeError, match="after swap"):
            run_fes(cfg, _system(), resumed)
    partial = _payload(resumed / "checkpoints/exchange_0000.npz")
    assert int(partial["elapsed"]) == 7
    assert len(json.loads(str(partial["operator_states"]))) == 2
    run_fes(cfg, _system(), resumed)
    _assert_checkpoints_equal(reference, resumed)


@pytest.mark.parametrize("mode", ["none", "trpmd", "piglet"])
def test_noncanonical_sampling_cannot_be_silently_used_as_mbar_measure(tmp_path, mode):
    cfg = _config("svr")
    cfg["sampling"]["thermostat"] = mode
    with pytest.raises(ValueError, match="canonical"):
        run_fes(cfg, _system(), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_noncanonical_gle_covariance_is_rejected_before_output(tmp_path):
    cfg = _config("gle")
    cfg["sampling"]["thermostat_params"]["gle"]["C"] = [[2.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="canonical GLE"):
        run_fes(cfg, _system(), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_unit_adapter_preserves_input_and_resolves_nhc_aliases():
    sampling = {"thermostat": "nose-hoover-chain",
                "thermostat_params": {"tdamp": "20 fs", "chain_length": 4, "substeps": 2}}
    before = copy.deepcopy(sampling)
    spec = thermostat_spec(sampling)
    assert sampling == before
    assert spec["mode"] == "nhc"
    assert spec["tau"] == parse_quantity("20 fs", "time")
    assert spec["chain_length"] == 4
    cfg = _config("nhc")
    cfg["sampling"] = {**cfg["sampling"], **sampling}
    assert fes_sampler(cfg).integrator == "obcbo"


def test_new_bath_checkpoint_cannot_omit_auxiliary_state(tmp_path):
    pytest.importorskip("pyrpmd._thermal")
    cfg = _config("nhc")
    run_fes(cfg, _system(), tmp_path)
    path = tmp_path / "checkpoints/window_00000_replica_0000.npz"
    data = _payload(path)
    data.pop("operator_state")
    data.pop("_pyrpmd_checkpoint", None)
    atomic_savez(path, kind="arrays", **data)
    with pytest.raises(ValueError, match="missing thermostat operator"):
        run_fes(cfg, _system(), tmp_path)


@pytest.mark.parametrize("corruption", ["rng", "sampling"])
def test_rejected_operator_restore_leaves_live_bath_unchanged(corruption):
    pytest.importorskip("pyrpmd._thermal")
    dynamics = Dynamics(HarmonicPotential(.4), [2.0], 1.7, 4, .02,
                        native=False, thermostat={"mode": "svr", "tau": .7})
    state = State(np.zeros((4, 1, 3)), dynamics.sample_momenta())
    dynamics.step(state)
    old = dynamics.operator_state_dict()
    dynamics.step(state)
    before = dynamics.operator_state_dict()
    broken = copy.deepcopy(old)
    if corruption == "rng":
        broken["rng"] = {}
    else:
        broken["sampling"] = {"invalid-json": {}}
    with pytest.raises((ValueError, TypeError, KeyError)):
        dynamics.load_operator_state_dict(broken)
    assert dynamics.operator_state_dict() == before


def test_rejected_first_explicit_sample_does_not_select_a_different_retry_stream():
    kwargs = dict(masses=[2.0], beta=1.7, nbeads=4, dt=.02, native=False, seed=93)
    dynamics = Dynamics(HarmonicPotential(.4), **kwargs)
    reference = Dynamics(HarmonicPotential(.4), **kwargs)
    state = State(np.zeros((4, 1, 3)), np.full((4, 1, 3), .2))
    expected = state.copy()
    before = dynamics.operator_state_dict()
    state.step = -1
    with pytest.raises(ValueError, match="state clock"):
        dynamics.sample_step(state, thermostat={"mode": "pile-l"})
    assert dynamics.operator_state_dict() == before
    state.step = 0
    dynamics.sample_step(state)
    reference.sample_step(expected)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
