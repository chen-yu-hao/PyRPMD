"""Checkpoint identities captured before the dRPMD module split."""

import numpy as np
import pytest

from pyrpmd.system import System
from pyrpmd.workflows.drpmd import _physics_hash


def test_nonadiabatic_dispatch_preserves_the_workflow_hook(monkeypatch, tmp_path):
    from pyrpmd.workflows import drpmd

    calls = []
    expected = {"status": "unresolved"}

    def propagate(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(drpmd, "_nrpmd_trajectory", propagate)
    config = {"nonadiabatic": {"method": "nrpmd"}}
    args = (None, None, None, config, tmp_path, 7, 2, {}, "identity", False)
    assert drpmd._trajectory(*args) is expected
    assert calls == [args]
    with pytest.raises(ValueError, match="extension is not implemented"):
        drpmd._trajectory(*args, extend_unresolved=True)
    assert calls == [args]


@pytest.mark.parametrize(
    "as_mapping, expected",
    [
        (False, "80112e33fee370926fb701a06810224f216359e6f8454b0cefffa0775b83d522"),
        (True, "fb2126738879ce7d177af028366c9cbf9bb40550bb8e4bfd97e43320577c7890"),
    ],
)
def test_legacy_restart_identity(as_mapping, expected):
    # Values were recorded with main at c4fa743f, before extracting the
    # workflow and consolidating JSON serialization. System.to_dict() carries
    # additional derived identity fields; its legacy digest is intentionally
    # different from the dataclass digest used by dRPMD's Python entry point.
    molecular = System(
        ("H", "H"),
        np.array([1.0, 1.0]),
        np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]),
        ((0,), (1,)),
    )
    config = {
        "beads": 1,
        "beta": 1.0,
        "dt_au": 0.1,
        "seed": 17,
        "native": False,
        "system": {"separation_au": 4.0},
        "potential": {"provider": "free"},
        "sampling": {"parents": 2},
        "collision": {"trajectories": 5, "bmax_au": 0.5, "max_steps": 200},
        "parallel": {"backend": "serial"},
        "output": {"directory": "ignored"},
    }
    system = molecular.to_dict() if as_mapping else molecular
    assert _physics_hash(config, system, 2) == expected
    if not as_mapping:
        # dRPMD historically serialized dataclass fields before considering
        # custom export methods. A frontend's custom to_dict must not silently
        # change the identity of an existing dRPMD checkpoint.
        molecular.to_dict = lambda: {"custom_export": "not the checkpoint schema"}
        assert _physics_hash(config, molecular, 2) == expected
