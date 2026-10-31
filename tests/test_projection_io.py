"""Saved FES projection preserves the normalized retained-axis order."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.projection import project_run
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.io import write_json


@pytest.mark.parametrize("axes", [0, np.int64(0), (0,), [2, 0]])
def test_saved_projection_accepts_scalar_and_reordered_axes(tmp_path, axes):
    source = tmp_path / "source"
    edges = [np.array([0., 1., 3.]), np.array([0., 2., 3., 6.]),
             np.array([0., 1., 2., 4., 8.])]
    free = np.arange(24, dtype=float).reshape(2, 3, 4) / 20
    beta = 2.
    write_json(source / "input" / "resolved.json", {"beta": beta})
    write_json(source / "summary.json", {"measure": "cv_density"})
    atomic_savez(source / "fes.npz", free_energy=free, mask=np.zeros(free.shape, bool),
                 **{f"edges_{i}": edge for i, edge in enumerate(edges)})
    output = tmp_path / "projection"
    result = project_run(source, axes=axes, output=output)
    retained = [int(axes)] if isinstance(axes, (int, np.integer)) else list(axes)
    assert result["axes"] == retained
    assert json.loads((output / "summary.json").read_text())["axes"] == retained

    # Direct Boltzmann quadrature independently checks the saved array order.
    weighted = np.exp(-beta * free)
    for axis in range(free.ndim):
        if axis not in retained:
            shape = [1] * free.ndim
            shape[axis] = len(edges[axis]) - 1
            weighted *= np.diff(edges[axis]).reshape(shape)
    eliminated = tuple(axis for axis in range(free.ndim) if axis not in retained)
    expected = -np.log(weighted.sum(axis=eliminated)) / beta
    expected -= expected.min()
    expected = expected.transpose([sorted(retained).index(axis) for axis in retained])
    with load_checkpoint(output / "pmf.npz", require_integrity=True) as saved:
        np.testing.assert_allclose(saved["free_energy"], expected, atol=1e-14)
        np.testing.assert_array_equal(saved["mask"], np.zeros(expected.shape, bool))
        for position, source_axis in enumerate(retained):
            np.testing.assert_array_equal(saved[f"edges_{position}"], edges[source_axis])
    table = np.genfromtxt(output / "pmf.csv", delimiter=",", names=True)
    assert table.dtype.names[:len(retained)] == tuple(f"cv{axis}_au" for axis in retained)
    np.testing.assert_allclose(table["free_energy_hartree"], expected.ravel())
    assert set(p.name for p in output.iterdir()) == {"pmf.npz", "pmf.csv", "summary.json"}
