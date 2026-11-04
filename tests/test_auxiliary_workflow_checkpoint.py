import numpy as np

from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.potentials.base import Potential, coordinates
from pyrpmd.workflows import nonadiabatic as workflow


class _ZeroModel(Potential):
    nstates = 2
    natoms = 1
    metadata = {"provider": "t26-workflow-zero", "batch_independent": True}

    def evaluate(self, q):
        q = coordinates(q)
        return DiabaticResult(
            np.zeros((len(q), 2, 2)), np.zeros((len(q), 2, 2, 1, 3))
        )

    def to_dict(self):
        return {"model": "t26-workflow-zero"}


def _state():
    q = np.zeros((2, 1, 3))
    return NRPMDState(
        q, np.zeros_like(q), mapping_q=np.tile([np.sqrt(3.0), 1.0], (2, 1)),
        mapping_p=np.zeros((2, 2)),
    )


def test_workflow_writes_v2_sidecar_and_reads_legacy_primary(tmp_path):
    config = {
        "method": "nrpmd", "beads": 2, "dt_au": 0.01,
        "ensemble": {"steps": 0, "sample_stride": 1},
    }
    identity = "t26-test-fingerprint"
    dynamics = NRPMDDynamics(_ZeroModel(), [2.0], 10.0, 2, 0.01, 2, native=False)
    state = _state()
    history = {key: [] for key in ("steps", *workflow._HISTORY)}
    workflow._append(history, dynamics, state)
    rng = np.random.default_rng(74)
    path = tmp_path / "nonadiabatic_000000.npz"

    workflow._save(path, config, identity, 0, state, rng, history)
    sidecar = workflow._operator_path(path)
    assert sidecar.is_file()
    loaded, loaded_history = workflow._load(
        path, config, identity, 0, dynamics, np.random.default_rng(74)
    )
    np.testing.assert_array_equal(loaded.mapping_q, state.mapping_q)
    np.testing.assert_array_equal(loaded.mapping_p, state.mapping_p)
    np.testing.assert_array_equal(loaded_history["steps"], history["steps"])

    sidecar.unlink()
    legacy, _ = workflow._load(
        path, config, identity, 0, dynamics, np.random.default_rng(74)
    )
    np.testing.assert_array_equal(legacy.mapping_q, state.mapping_q)
    np.testing.assert_array_equal(legacy.mapping_p, state.mapping_p)
