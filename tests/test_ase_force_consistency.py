"""An ASE PES must use the energy differentiated by its forces and stress."""

import numpy as np
import pytest

from pyrpmd.potentials import check_potential
from pyrpmd.potentials.optional import ASEPotential
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


def _calculator(*, free_energy=True):
    pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes

    class ElectronicFreeEnergy(Calculator):
        # Mimic a smeared electronic calculation: forces/stress differentiate
        # the free energy, while the extrapolated energy has a varying offset.
        implemented_properties = ["energy", "forces", "stress"] + (
            ["free_energy"] if free_energy else []
        )

        def __init__(self):
            super().__init__()
            self.calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            d = atoms.get_distance(0, 1, mic=True, vector=True)
            volume = atoms.get_volume()
            energy = .4 * np.dot(d, d) + .02 * volume
            self.results = {
                "energy": energy + (.13 * np.dot(d, d) + .03 * volume
                                    if free_energy else 0.),
                "forces": np.stack([.8 * d, -.8 * d]),
                "stress": .8 * np.outer(d, d) / volume + .02 * np.eye(3),
            }
            if free_energy:
                self.results["free_energy"] = energy

    return ElectronicFreeEnergy()


@pytest.mark.parametrize("pbc", [[True, True, True], [True, True, False]])
def test_electronic_free_energy_matches_force_and_strain_derivatives(pbc):
    cell = np.array([[8., 0., 0.], [1.3, 9., 0.], [.4, .6, 15.]])
    q = np.array([[[.3, .2, 1.], [1.7, .5, 2.]],
                  [[.1, .4, .7], [2.1, .9, 1.5]]])
    calc = _calculator()
    with ASEPotential(["H", "H"], calc, cell=cell, pbc=pbc) as pes:
        result = pes.evaluate(q)
        assert calc.calls == len(q)  # Energy, forces and stress share one solve.
        assert abs(result.energy[-1] / EV_HARTREE - calc.results["energy"]) > .1
        assert result.energy[-1] == pytest.approx(calc.results["free_energy"] * EV_HARTREE)
        assert check_potential(pes, q)["max_absolute_force_error"] < 2e-10

        # An independent homogeneous-strain check includes the explicit cell
        # contribution; position-only r.F would miss the volume-dependent term.
        eps = 1e-5
        for a in range(3):
            for b in range(3):
                strain = np.zeros((3, 3))
                strain[a, b] = eps
                energies = []
                for sign in (1, -1):
                    transform = np.eye(3) + sign * strain
                    pes.bind_system({"cell": cell @ transform.T, "pbc": pbc})
                    energies.append(pes.evaluate(q @ transform.T).energy)
                numerical = -(energies[0] - energies[1]) / (2 * eps)
                np.testing.assert_allclose(result.virial[:, a, b], numerical,
                                           atol=2e-10, rtol=1e-8)


def test_calculator_without_free_energy_keeps_ordinary_energy():
    calc = _calculator(free_energy=False)
    q = np.array([[[.3, .2, 1.], [1.7, .5, 2.]]])
    with ASEPotential(["H", "H"], calc, cell=np.eye(3) * 8.) as pes:
        result = pes.evaluate(q)
        assert result.energy[0] == pytest.approx(calc.results["energy"] * EV_HARTREE)
        assert calc.calls == 1
        assert check_potential(pes, q)["max_absolute_force_error"] < 2e-10


@pytest.mark.parametrize("failure", ["missing", "not_implemented", "failed_scf"])
def test_declared_free_energy_failure_does_not_fall_back(failure):
    pytest.importorskip("ase")
    from ase.calculators.calculator import (
        CalculationFailed, Calculator, PropertyNotImplementedError, all_changes,
    )

    class FailedFreeEnergy(Calculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            # A stale/extrapolated energy must not hide a broken free-energy
            # request, even if that energy happens to be finite.
            self.results = {"energy": 1., "forces": np.zeros((1, 3))}
            if properties == ["free_energy"]:
                if failure == "not_implemented":
                    raise PropertyNotImplementedError("electronic solver unavailable")
                if failure == "failed_scf":
                    raise CalculationFailed("SCF did not converge")

    expected = CalculationFailed if failure == "failed_scf" else PropertyNotImplementedError
    with ASEPotential(["H"], FailedFreeEnergy()) as pes:
        with pytest.raises(expected):
            pes.evaluate(np.zeros((1, 1, 3)))


def test_real_emt_energy_forces_and_stress_remain_consistent():
    pytest.importorskip("ase")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
    atoms.positions[0] += [.03, -.02, .01]
    atoms.calc = EMT()
    q = atoms.positions[None] / BOHR_ANGSTROM
    with ASEPotential(atoms.get_chemical_symbols(), EMT(),
                      cell=np.asarray(atoms.cell) / BOHR_ANGSTROM) as pes:
        result = pes.evaluate(q)
        np.testing.assert_allclose(result.energy, atoms.get_potential_energy() * EV_HARTREE)
        np.testing.assert_allclose(result.forces[0], atoms.get_forces() * EV_HARTREE * BOHR_ANGSTROM)
        np.testing.assert_allclose(result.virial[0],
                                   -atoms.get_stress(voigt=False) * atoms.get_volume() * EV_HARTREE)
        assert check_potential(pes, q)["max_absolute_force_error"] < 2e-9
