"""RPMDrate entrance initialization must not translate/permutate fixed atoms."""

import hashlib

import numpy as np
import pytest

from pyrpmd.dynamics import native_capabilities
from pyrpmd.system import System
from pyrpmd.workflows import fes


def _task(tmp_path, fixed, target, replica, dimension=2):
    geometry = np.array([[-1.75, 0., 0.], [0., 0., 0.], [1.75, 0., 0.]])
    system = System(("H", "H", "H"), np.full(3, 1837.), geometry,
                    ((0,), (1, 2)), fixed_atoms=fixed)
    specs = [
        {"name": "xi", "kind": "rpmdrate_xi", "estimator": "centroid",
         "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
         "breaking_bonds": [[1, 2]], "rinf": 10.,
         "equivalent_paths": [[0, 2, 1]]},
        {"name": "r12", "kind": "distance", "atoms": [1, 2]},
    ][:dimension]
    config = {
        "beta": 200., "beads": 4, "dt_au": .01, "seed": 41, "native": False,
        "potential": {"provider": "builtin", "name": "free"},
        "sampling": {"equilibration_steps": 0, "production_steps": 4,
                     "sample_interval": 1, "andersen_interval": 10},
        "analysis": {"measure": "cv_density"}, "cvs": specs,
        "output": {"checkpoint_steps": 2},
    }
    return {"config": config, "system": system, "specs": specs,
            "center": [target, 1.75][:dimension],
            "stiffness": np.eye(dimension)*.1,
            "window": 0, "replica": replica, "output": str(tmp_path)}


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fixed", [(0,), (1,), (2,), (1, 2)])
@pytest.mark.parametrize("replica", [0, 1])
@pytest.mark.parametrize("target", [0.0, 1.0])
def test_two_cv_fixed_atoms_survive_initialization_sampling_and_resume(
    tmp_path, fixed, replica, target, native,
):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    task = _task(tmp_path, fixed, target, replica)
    task["config"]["native"] = native
    original = task["system"].geometry.copy()
    result = fes._sample_window(task)
    checkpoint = tmp_path / "checkpoints" / f"window_00000_replica_{replica:04d}.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        expected = np.repeat(original[None, list(fixed), :], 4, axis=0)
        np.testing.assert_array_equal(data["q"][:, fixed, :], expected)
        assert not data["p"][:, fixed, :].any()
        assert data["elapsed"] == 4
    np.testing.assert_array_equal(task["system"].geometry, original)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    resumed = fes._sample_window(task)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    np.testing.assert_array_equal(resumed["samples"], result["samples"])
    np.testing.assert_array_equal(resumed["last_q"], result["last_q"])


def test_mobile_two_cv_system_retains_rigid_entrance_initialization(tmp_path, monkeypatch):
    task = _task(tmp_path, (), 0., 1)
    original = fes.separate_reactants_to_xi
    calls = []

    def spy(*args):
        calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(fes, "separate_reactants_to_xi", spy)
    fes._sample_window(task)
    assert calls == [0.]


def test_fixed_unreachable_center_fails_without_moving_reference(tmp_path):
    task = _task(tmp_path, (0, 1, 2), 0., 0)
    before = task["system"].geometry.copy()
    with pytest.raises(ValueError, match="fixed atoms held in place"):
        fes._sample_window(task)
    np.testing.assert_array_equal(task["system"].geometry, before)
