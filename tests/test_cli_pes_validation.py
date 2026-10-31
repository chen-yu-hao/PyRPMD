"""The PES check command must retain the original geometry's diagnostics."""

import json

import numpy as np
import pytest

from pyrpmd.cli import main
from pyrpmd.potentials import PESResult
from pyrpmd.system import read_xyz


class ReusedResultPotential:
    def __init__(self, *, incorrect_forces=False):
        self.result = PESResult(np.empty(1), np.empty((1, 1, 3)))
        self.incorrect_forces = incorrect_forces
        self.closed = False

    def evaluate(self, q):
        self.result.energy[:] = 0.5 * np.sum(q * q, axis=(1, 2))
        self.result.forces[:] = 0.0 if self.incorrect_forces else -q
        return self.result

    def close(self):
        self.closed = True


@pytest.mark.parametrize("incorrect_forces", [False, True])
def test_pes_check_with_reused_results(
    tmp_path, monkeypatch, capsys, incorrect_forces,
):
    geometry = tmp_path / "atom.xyz"
    geometry.write_text("1\ncoordinates in angstrom\nH 0.1 -0.2 0.3\n")
    q = read_xyz(geometry).geometry
    provider = ReusedResultPotential(incorrect_forces=incorrect_forces)
    monkeypatch.setattr("pyrpmd.potentials.make_potential", lambda *args: provider)

    code = main([
        "pes", "check", "unused.so", "--geometry", str(geometry),
        "--tolerance", "1e-8",
    ])

    captured = capsys.readouterr()
    assert provider.closed
    if incorrect_forces:
        assert code == 2
        assert "finite-difference check failed" in captured.err
        assert not captured.out
    else:
        assert code == 0, captured.err
        result = json.loads(captured.out)
        assert result["passed"] is True
        assert result["max_force_error_au"] < 1e-10
        np.testing.assert_allclose(result["energy_hartree"], [0.5 * np.sum(q * q)])
