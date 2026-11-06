"""NP7 serial MetaD block scheduling and R10 restart contract tests."""

from pathlib import Path

import numpy as np

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System
from pyrpmd.workflows import biased_sampling
from pyrpmd.workflows.metadynamics import _sample_replica


def _system():
    return System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))


def _config(*, thermostat="andersen", grid=False):
    table = {
        "mode": "well_tempered", "sigma": [0.3], "height": 0.05,
        "pace": 3, "bias_factor": 5.0,
    }
    if grid:
        table["grid"] = {"ranges": [[-2.0, 2.0]], "bins": 40}
    return {
        "task": "metadynamics", "beta": 2.0, "beads": 3, "dt_au": 0.02,
        "seed": 321, "native": True,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 3, "production_steps": 24,
                     "sample_interval": 2, "andersen_interval": 4,
                     "replicas": 1, "thermostat": thermostat},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "metadynamics": table,
        "analysis": {"bins": [6], "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }


def _run(config, output):
    return _sample_replica({"config": config, "system": _system(),
                            "output": str(output), "replica": 0})


def _assert_same(a, b):
    np.testing.assert_allclose(a["state"].q, b["state"].q, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(a["state"].p, b["state"].p, rtol=1e-11, atol=1e-12)
    for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-11, atol=1e-12)
    assert a["hills"] == b["hills"]
    assert a["rng"] == b["rng"]


def test_andersen_history_blocks_match_scalar_reference(tmp_path, monkeypatch):
    config = _config()
    import pyrpmd._native as native
    calls = []
    original_advance = native.advance

    def counted_advance(*args):
        calls.append(1)
        return original_advance(*args)

    monkeypatch.setattr(native, "advance", counted_advance)
    blocked = _run(config, tmp_path / "blocked")
    original = biased_sampling._history_block
    monkeypatch.setattr(biased_sampling, "_history_block", lambda *args, **kwargs: False)
    scalar = _run(config, tmp_path / "scalar")
    monkeypatch.setattr(biased_sampling, "_history_block", original)
    _assert_same(blocked, scalar)
    assert blocked["propagation"]["native_steps"] == 27
    assert blocked["propagation"]["python_steps"] == 0
    assert blocked["propagation"]["fallback_reasons"] == {}
    assert len(calls) <= 9 + 5 + 2
    with load_checkpoint(Path(tmp_path / "blocked/checkpoints/metadynamics_0000.npz")) as saved:
        assert int(saved["native_path_contract"]) == 1


def test_grid_and_pile_l_history_blocks_match_scalar_reference(tmp_path, monkeypatch):
    for thermostat, grid in (("andersen", True), ("pile-l", False)):
        config = _config(thermostat=thermostat, grid=grid)
        blocked = _run(config, tmp_path / f"blocked-{thermostat}-{grid}")
        original = biased_sampling._history_block
        monkeypatch.setattr(biased_sampling, "_history_block", lambda *args, **kwargs: False)
        scalar = _run(config, tmp_path / f"scalar-{thermostat}-{grid}")
        monkeypatch.setattr(biased_sampling, "_history_block", original)
        _assert_same(blocked, scalar)
        assert blocked["propagation"]["native_steps"] == 27
        assert blocked["propagation"]["python_steps"] == 0


def test_missing_contract_marker_keeps_legacy_restart_path(tmp_path):
    config = _config()
    _run(config, tmp_path)
    path = tmp_path / "checkpoints/metadynamics_0000.npz"
    with load_checkpoint(path) as saved:
        payload = {name: saved[name] for name in saved.files if name != "native_path_contract"}
    from pyrpmd.checkpoint import atomic_savez
    atomic_savez(path, **payload)
    from pyrpmd.workflows.metadynamics import _read_stream, resolve_metadynamics
    cv, _, options = resolve_metadynamics(config, _system())
    restored = _read_stream(config, _system(), tmp_path, 0, cv, options)
    assert restored["native_path_contract"] == 0
    assert restored["collision_clock"] is None


def test_no_native_extension_keeps_scalar_history_semantics(tmp_path):
    config = _config()
    config["native"] = False
    result = _run(config, tmp_path)
    assert result["propagation"]["native_steps"] == 0
    assert result["propagation"]["python_steps"] == 27
    assert result["propagation"]["fallback_reasons"] == {
        "history-dependent metadynamics bias": 27
    }
