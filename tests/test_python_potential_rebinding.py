"""Rejected periodic rebindings must not commit adapter geometry or protocol."""

import numpy as np
import pytest

from pyrpmd.potentials import PotentialError, PythonPotential


@pytest.mark.parametrize("metadata", [{}, {"supports_pbc": True}])
def test_incompatible_periodic_contract_keeps_nonperiodic_binding(metadata):
    class Molecular:
        def evaluate(self, q):
            return .5 * np.sum(q * q, axis=(1, 2)), -q

    obj = Molecular()
    obj.metadata = metadata
    pes = PythonPotential(obj)
    q = np.ones((2, 1, 3))
    initial = pes.evaluate(q)
    with pytest.raises(PotentialError, match="Periodic Python PES"):
        pes.bind_system({"cell": np.eye(3) * 6., "pbc": [True, True, False]})
    assert pes.cell is None
    np.testing.assert_array_equal(pes.pbc, [False, False, False])
    np.testing.assert_array_equal(pes.evaluate(q).energy, initial.energy)


def test_callback_validation_failure_does_not_commit_keyword_binding():
    class CellModel:
        metadata = {"periodic": True}

        def __init__(self):
            self.accepted = None

        def bind_system(self, system):
            # A well-behaved callback validates before changing its own state.
            # The wrapper cannot roll back side effects of arbitrary user code.
            if not np.all(system["pbc"]):
                raise ValueError("model requires full PBC")
            self.accepted = np.array(system["cell"], copy=True)

        def evaluate(self, q, *, cell, pbc):
            return np.full(len(q), np.linalg.det(cell)), np.zeros_like(q)

    model = CellModel()
    cell = np.eye(3) * 6.
    pes = PythonPotential(model, cell=cell)
    pes.bind_system({"cell": cell, "pbc": [True] * 3})
    q = np.ones((2, 1, 3))
    initial = pes.evaluate(q)
    with pytest.raises(ValueError, match="full PBC"):
        pes.bind_system({"cell": cell * 2., "pbc": [False] * 3})
    np.testing.assert_array_equal(pes.cell, cell)
    np.testing.assert_array_equal(pes.pbc, [True] * 3)
    np.testing.assert_array_equal(model.accepted, cell)
    np.testing.assert_array_equal(pes.evaluate(q).energy, initial.energy)

    pes.bind_system({"cell": cell * 2., "pbc": [True] * 3})
    np.testing.assert_array_equal(pes.cell, cell * 2.)
    np.testing.assert_allclose(pes.evaluate(q).energy, initial.energy * 8.)
