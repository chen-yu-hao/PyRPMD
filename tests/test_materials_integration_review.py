"""Cross-PR invariants for periodic sampling, grid bias and bath restart."""

import numpy as np
import pytest

from pyrpmd.bias import GridMetadynamicsBias
from pyrpmd.workflows.metadynamics import _sample_replica
from test_surface_extensions import meta_config, slab


@pytest.mark.parametrize("thermostat", ["nose-hoover-chain", "pile-l", "andersen"])
def test_grid_wall_deposition_failure_restores_thermostat_and_exact_prefix(
    tmp_path, monkeypatch, thermostat,
):
    config, system = meta_config(thermostat, beads=4), slab()
    config["cvs"] = config["cvs"][:2]
    config["metadynamics"]["sigma"] = [.2, .2]
    config["metadynamics"]["grid"] = {
        "ranges": [[-3., 3.], [-3., 3.]], "bins": [150, 150],
    }
    config["analysis"]["bins"] = [4, 5]

    def task(directory):
        return {"config": config, "system": system, "output": str(directory), "replica": 0}

    reference = _sample_replica(task(tmp_path / "reference"))
    original = GridMetadynamicsBias.deposit

    def fail_after_grid_update(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs["step"] == 10:
            raise RuntimeError("interrupted after grid and hill update")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(GridMetadynamicsBias, "deposit", fail_after_grid_update)
        with pytest.raises(RuntimeError, match="after grid and hill update"):
            _sample_replica(task(tmp_path / "resumed"))
    resumed = _sample_replica(task(tmp_path / "resumed"))

    for key in ("hills", "rng", "thermostat_state", "propagation"):
        assert resumed[key] == reference[key]
    for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(resumed[key], reference[key])
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(resumed["state"], key), getattr(reference["state"], key))
    np.testing.assert_array_equal(resumed["grid_state"]["values"], reference["grid_state"]["values"])
    assert resumed["grid_state"]["metadata"] == reference["grid_state"]["metadata"]
