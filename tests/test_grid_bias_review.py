"""Independent regressions for interpolation and cache ownership boundaries."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.bias import GridMetadynamicsBias, MetadynamicsBias
from pyrpmd.cv import CVStack, Cartesian
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.system import System


def grid(*, mode="standard", hills=None, grid_state=None, bins=8):
    return GridMetadynamicsBias(
        CVStack([Cartesian(0, 0)]), [.4], .1, 2., mode=mode,
        bias_factor=5. if mode == "well_tempered" else None,
        grid_ranges=[[-2., 2.]], grid_bins=bins, hills=hills,
        grid_state=grid_state,
    )


@pytest.mark.parametrize("mode", ["standard", "well_tempered"])
def test_accepted_coarse_grid_can_deposit_in_its_signed_hermite_undershoot(mode):
    bias = grid(mode=mode)
    bias.deposit([.3], step=1, time=.1)
    first = bias.grid_state()["values"].copy()
    center = [-1.3256]
    before = float(bias.physical_energy(center))
    # Cubic Hermite interpolation of positive nodal Gaussians can undershoot.
    # It is the applied energy, so changing/clipping only this value would
    # disagree with the forces and with the WT deposition exponent.
    assert before < -1e-6
    deposited = bias.deposit(center, step=2, time=.2)
    assert deposited["bias_before"] == before
    expected_height = .1 if mode == "standard" else .1 * np.exp(-2 * before / 4.)
    assert deposited["height"] == pytest.approx(expected_height, rel=1e-15)
    replay = grid(mode=mode, hills=bias.hills)
    np.testing.assert_array_equal(replay.grid_state()["values"], bias.grid_state()["values"])
    restored = grid(mode=mode, hills=bias.hills, grid_state=bias.grid_state())
    np.testing.assert_array_equal(restored.grid_state()["values"], bias.grid_state()["values"])
    bias.rollback_hills(1)
    np.testing.assert_array_equal(bias.grid_state()["values"], first)


def test_direct_gaussian_history_still_rejects_negative_bias_before():
    cv = CVStack([Cartesian(0, 0)])
    with pytest.raises(ValueError, match="invalid.*bias"):
        MetadynamicsBias(cv, [.4], .1, 2., mode="standard", hills=[{
            "center": [0.], "height": .1, "time": .1, "step": 1, "bias_before": -.01,
        }])


@pytest.mark.parametrize("tamper", ["zero", "single_value", "derivative"])
def test_cache_history_metadata_does_not_authorize_changed_coefficients(tamper):
    bias = grid(bins=50)
    bias.deposit([.3], step=1, time=.1)
    cache = copy.deepcopy(bias.grid_state())
    cache["values"].flags.writeable = True
    if tamper == "zero":
        cache["values"][:] = 0.
    elif tamper == "single_value":
        cache["values"][0, 20] += .01
    else:
        cache["values"][1, 20] += .01
    with pytest.raises(ValueError, match="cache"):
        grid(bins=50, hills=bias.hills, grid_state=cache)


@pytest.mark.parametrize("backend", ["direct", "grid"])
def test_returned_deposition_record_does_not_alias_the_authoritative_history(backend):
    bias = grid(bins=50) if backend == "grid" else MetadynamicsBias(
        CVStack([Cartesian(0, 0)]), [.4], .1, 2., mode="standard")
    report = bias.deposit([.3], step=1, time=.1)
    history = copy.deepcopy(bias.hills)
    before = bias.physical_energy_gradient([[.3], [1.5]])
    report["center"][0] = 1.5
    report["height"] = 0.
    assert bias.hills == history
    after = bias.physical_energy_gradient([[.3], [1.5]])
    for actual, expected in zip(after, before):
        np.testing.assert_array_equal(actual, expected)
    if backend == "grid":
        replay = grid(bins=50, hills=bias.hills)
        np.testing.assert_array_equal(replay.grid_state()["values"], bias.grid_state()["values"])


@pytest.mark.parametrize("backend", ["direct", "grid"])
def test_hill_only_restart_rejects_times_that_disagree_with_the_deposition_steps(tmp_path, backend):
    from pyrpmd.workflows.metadynamics import _sample_replica, _read_stream, resolve_metadynamics

    config = {
        "task": "metadynamics", "beta": 2., "beads": 3, "dt_au": .02,
        "seed": 3, "native": False,
        "potential": {"provider": "harmonic", "params": {"k": .1}},
        "cvs": [{"kind": "cartesian", "name": "x", "atom": 0, "axis": 0}],
        "sampling": {"production_steps": 6, "sample_interval": 1, "replicas": 1},
        "metadynamics": {"mode": "standard", "sigma": [.3], "height": .01, "pace": 2},
        "analysis": {"bootstrap": 0},
    }
    if backend == "grid":
        config["metadynamics"]["grid"] = {"ranges": [[-2., 2.]], "bins": 70}
    system = System(("H",), [2.], [[0., 0., 0.]])
    _sample_replica({"config": config, "system": system, "output": str(tmp_path), "replica": 0})
    path = tmp_path / "checkpoints/metadynamics_0000.npz"
    with load_checkpoint(path) as saved:
        # Missing caches are supported: replay must validate the authoritative
        # records too, independently of a cache's history hash.
        data = {name: saved[name] for name in saved.files if name not in ("grid_values", "grid_metadata")}
    hills = json.loads(str(data["hills"]))
    hills[0]["time"] = 0.
    data["hills"] = json.dumps(hills)
    atomic_savez(path, **data)
    cv, _, options = resolve_metadynamics(config, system)
    with pytest.raises(ValueError, match="hill|time|deposition clock"):
        _read_stream(config, system, tmp_path, 0, cv, options)
