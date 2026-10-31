"""Actual CPU Torch checks for cell/PBC and strain-derivative contracts."""

import numpy as np
import pytest
from unittest.mock import patch

from pyrpmd.potentials import PotentialError, check_potential
from pyrpmd.potentials.optional import TorchPotential
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


def _torch():
    return pytest.importorskip("torch")


def test_torch_periodic_autograd_force_and_full_strain_virial():
    torch = _torch()

    class Model(torch.nn.Module):
        def forward(self, q, cell, pbc):
            assert pbc.dtype == torch.bool and pbc.shape == (len(q), 3)
            fractional = torch.matmul(q, torch.linalg.inv(cell))
            d = fractional[:, 1] - fractional[:, 0]
            d = d - torch.round(d)*pbc
            dr = torch.matmul(d[:, None], cell).squeeze(1)
            return .4*torch.sum(dr**2, dim=1) + .07*torch.sum(cell**2, dim=(1, 2))

    cell = np.array([[4., .4, .2], [.1, 5., .3], [.1, .2, 6.]])
    x = np.array([[[.2, .3, .1], [1.2, .9, .2]], [[.3, .1, .2], [1.7, .2, .5]]])
    flags = [True, False, True]
    pes = TorchPotential(Model(), device="cpu", periodic=True, cell=cell, pbc=flags,
                         units={"positions": "angstrom", "energy": "eV"})
    result = pes.evaluate(x)
    assert check_potential(pes, x)["max_absolute_force_error"] < 1e-10
    shifted = x.copy()
    shifted[:, 1] += cell[0]
    np.testing.assert_allclose(pes.evaluate(shifted).energy, result.energy, atol=1e-13)
    np.testing.assert_allclose(pes.evaluate(shifted).forces, result.forces, atol=1e-13)
    eps = 1e-5
    for a in range(3):
        for b in range(3):
            strain = np.zeros((3, 3)); strain[a, b] = eps
            energies = []
            for sign in (1, -1):
                deform = np.eye(3) + sign*strain
                pes.bind_system({"cell": cell @ deform.T, "pbc": flags})
                energies.append(pes.evaluate(x @ deform.T).energy)
            np.testing.assert_allclose(result.virial[:, a, b], -(energies[0]-energies[1])/(2*eps), atol=1e-10)
    pes.close()
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(x)


def test_torch_asymmetric_strain_orientation():
    torch = _torch()

    class Model(torch.nn.Module):
        def forward(self, q, cell, pbc):
            return cell[:, 0, 0]**2 + 3*cell[:, 0, 1] + q[:, 0, 0]**2

    cell = np.array([[3., .4, .2], [.1, 5., .3], [.1, .2, 6.]])
    x = np.array([[[1., 2., 3.]]])
    pes = TorchPotential(Model(), device="cpu", periodic=True, cell=cell)
    result = pes.evaluate(x)
    expected = -(np.outer([6., 3., 0.], cell[0]) + np.outer([2., 0., 0.], x[0, 0]))
    np.testing.assert_allclose(result.virial[0], expected)
    assert result.virial[0, 0, 1] != result.virial[0, 1, 0]


def test_torch_provided_stress_units_detached_energy():
    torch = _torch()

    class Model(torch.nn.Module):
        def forward(self, q, cell, pbc):
            return {"energy": torch.ones(len(q), dtype=torch.float64),
                    "forces": torch.ones_like(q),
                    "stress": torch.ones((len(q), 3, 3), dtype=torch.float64)}

    cell = np.eye(3)*5
    with TorchPotential(Model(), device="cpu", periodic=True, cell=cell,
                        units={"positions": "angstrom", "energy": "eV"}) as pes:
        result = pes.evaluate(np.ones((2, 1, 3)))
        np.testing.assert_allclose(result.forces, EV_HARTREE*BOHR_ANGSTROM)
        np.testing.assert_allclose(result.virial, -(5*BOHR_ANGSTROM)**3*EV_HARTREE)


def test_torch_periodic_and_virial_guards():
    torch = _torch()

    class Detached(torch.nn.Module):
        def forward(self, q, cell, pbc):
            return {"energy": torch.ones(len(q), dtype=torch.float64), "forces": torch.zeros_like(q)}

    with pytest.raises(PotentialError, match="periodic=True"):
        TorchPotential(Detached(), device="cpu", cell=np.eye(3))
    with pytest.raises(PotentialError, match="nonsingular cell"):
        TorchPotential(Detached(), device="cpu", periodic=True)
    for mode, message in (("autograd", "detached energy"), ("provided", "requires model")):
        p = TorchPotential(Detached(), device="cpu", periodic=True, cell=np.eye(3), virial=mode)
        with pytest.raises(PotentialError, match=message):
            p.evaluate(np.zeros((1, 1, 3)))
    p = TorchPotential(Detached(), device="cpu", periodic=True, cell=np.eye(3))
    assert p.evaluate(np.zeros((1, 1, 3))).virial is None


def test_torch_disconnected_graph_cannot_silently_produce_zero_forces():
    torch = _torch()

    class Reattached(torch.nn.Module):
        def forward(self, q):
            return torch.sum(q*q, dim=(1, 2)).detach().requires_grad_(True)

    with TorchPotential(Reattached(), device="cpu") as pes:
        with pytest.raises(PotentialError, match="disconnected"):
            pes.evaluate(np.ones((2, 1, 3)))


def test_torch_delegate_must_match_exact_device_and_closes_on_rejection():
    torch = _torch()

    class Delegate:
        metadata = {"execution_device": "cuda:1"}

        def __init__(self):
            self.closed = False

        def evaluate(self, x):
            return np.zeros(len(x)), np.zeros_like(x)

        def close(self):
            self.closed = True

    delegate = Delegate()
    # No CUDA allocation is performed: the imported factory returns a Python
    # delegate. Simulate the availability probe to exercise device validation.
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "device_count", return_value=2), \
            patch("pyrpmd.potentials.optional.import_object", return_value=lambda **kw: delegate):
        with pytest.raises(PotentialError, match="requested cuda:0.*cuda:1"):
            TorchPotential(factory="unused:factory", device="cuda:0")
    assert delegate.closed


def test_torch_delegate_closes_on_periodic_contract_failure():
    _torch()

    class Delegate:
        metadata = {"execution_device": "cpu"}
        closed = False

        def evaluate(self, x):
            return np.zeros(len(x)), np.zeros_like(x)

        def close(self):
            self.closed = True

    delegate = Delegate()
    with patch("pyrpmd.potentials.optional.import_object", return_value=lambda **kw: delegate):
        with pytest.raises(PotentialError, match="Periodic Python PES"):
            TorchPotential(factory="unused:factory", device="cpu", periodic=True, cell=np.eye(3))
    assert delegate.closed
