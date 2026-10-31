"""Numerical contracts for fixed periodic cells and material PES adapters."""

import importlib.util
import itertools
import json
import socket
import struct
import threading
import unittest

import numpy as np

from pyrpmd.geometry import minimum_image, wrap_positions
from pyrpmd.system import System, system_from_config
from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.potentials import (
    PESResult, PotentialError, PythonPotential, PeriodicCosinePotential,
    check_potential, make_potential, validate_result,
    CABIPotential,
)
from pyrpmd.potentials.optional import ASEPotential
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE
from pyrpmd.potentials.socket import IPIPotential


class PeriodicGeometryTests(unittest.TestCase):
    def test_system_roundtrip_and_legacy_identity(self):
        molecular = System(("H",), [1837.0], [[0, 0, 0]])
        self.assertNotIn("cell", molecular.to_dict())
        self.assertNotIn("pbc", molecular.to_dict())
        self.assertEqual(molecular.to_dict(), System.from_dict(molecular.to_dict()).to_dict())
        spec = {
            "symbols": ["H"], "geometry": [[1, 0, 0]], "geometry_unit": "angstrom",
            "cell": [[6, 0, 0], [1, 7, 0], [0, 0, 12]], "pbc": [True, True, False],
        }
        system = system_from_config(spec)
        np.testing.assert_allclose(system.cell, np.array(spec["cell"]) * ANGSTROM_TO_AU)
        np.testing.assert_allclose(system.geometry, [[ANGSTROM_TO_AU, 0, 0]])
        restored = System.from_dict(json.loads(json.dumps(system.to_dict())))
        self.assertEqual(system.to_dict(), restored.to_dict())
        spec["cell_unit"] = "bohr"
        np.testing.assert_array_equal(system_from_config(spec).cell, spec["cell"])
        for bad in (
            {"pbc": [1, 0, 1]}, {"pbc": [True, True, True]},
            {"cell": np.zeros((3, 3))}, {"cell": np.full((3, 3), np.nan)},
            {"cell": np.eye(3, dtype=complex)},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                System(("H",), [1837], [[0, 0, 0]], **bad)

    def test_triclinic_minimum_image_against_exhaustive_search(self):
        cell = np.array([[4., 0., 0.], [3.7, 1.1, 0.], [.8, -.3, 5.]])
        vectors = np.random.default_rng(12).uniform(-3, 3, (25, 3))
        for flags in ([True, True, True], [True, False, True], [False, True, False]):
            translations = np.array(list(itertools.product(*[range(-8, 9) if f else (0,) for f in flags]))) @ cell
            actual = minimum_image(vectors, cell, flags)
            distances = np.sum((vectors[:, None, :] - translations[None, :, :]) ** 2, axis=2)
            np.testing.assert_allclose(np.sum(actual * actual, axis=1), np.min(distances, axis=1), atol=2e-13)
            shift = np.array([301, -207, 49]) * flags
            np.testing.assert_allclose(minimum_image(vectors + shift @ cell, cell, flags), actual, atol=3e-13)
        # This cell gives a genuine counterexample to naive fractional rounding.
        fractional = vectors @ np.linalg.inv(cell)
        naive = vectors - np.rint(fractional) @ cell
        exact = minimum_image(vectors, cell)
        self.assertGreater(float(np.max(np.linalg.norm(naive, axis=1) - np.linalg.norm(exact, axis=1))), .1)

    def test_wrap_and_unwrapped_ring_centroid_contract(self):
        cell = np.array([[4., 0., 0.], [2., 3., 0.], [0., 0., 20.]])
        fractional = np.array([[2.3, -1.2, 1.7], [-.4, 4.2, -.5]])
        actual = wrap_positions(fractional @ cell, cell, [True, True, False]) @ np.linalg.inv(cell)
        np.testing.assert_allclose(actual, [[.3, .8, 1.7], [.6, .2, -.5]])
        # A ring crossing a cell boundary must keep its continuous bead images.
        from pyrpmd.dynamics import State
        q = np.array([[[3.9, 0., 0.]], [[4.1, 0., 0.]]])
        state = State(q, np.zeros_like(q))
        np.testing.assert_allclose(state.q.mean(axis=0), [[4., 0., 0.]])
        np.testing.assert_array_equal(state.q, q)


class PeriodicPotentialTests(unittest.TestCase):
    def test_cosine_force_and_lattice_translation(self):
        cell = np.array([[5., 0., 0.], [1.5, 4., 0.], [1., .2, 12.]])
        potential = PeriodicCosinePotential(cell, [True, True, False], height=[.01, .03, 0], repeats=[2, 1, 1], confinement=.02)
        q = np.random.default_rng(13).normal(size=(4, 2, 3))
        self.assertLess(check_potential(potential, q)["max_absolute_force_error"], 1e-9)
        shifted = q.copy()
        shifted[:, 0] += 2 * cell[0] - cell[1]
        np.testing.assert_allclose(potential.evaluate(q).energy, potential.evaluate(shifted).energy, atol=2e-16)
        np.testing.assert_allclose(potential.evaluate(q).forces, potential.evaluate(shifted).forces, atol=2e-16)
        system = System(("H", "H"), [1837, 1837], q[0], cell=cell, pbc=[True, True, False])
        made = make_potential({"provider": "builtin", "name": "periodic_cosine", "params": {"height": .02}}, system)
        self.assertIsInstance(made, PeriodicCosinePotential)
        for spec in ({"provider": "harmonic"}, {"provider": "cabi", "abi_version": 1, "library": "missing.so"}, {"provider": "torch", "model": "missing.pt"}, {"provider": "f2py", "module": "missing"}):
            with self.subTest(spec=spec), self.assertRaisesRegex(PotentialError, "periodic|cell/PBC"):
                make_potential(spec, system)
        with self.assertRaisesRegex(PotentialError, "nonperiodic"):
            CABIPotential("missing.so", system=system, abi_version=1)

    def test_python_periodic_contract_and_units(self):
        cell = np.diag([8., 9., 10.])
        flags = [True, True, True]
        with self.assertRaisesRegex(PotentialError, "declare"):
            PythonPotential(lambda q: (np.zeros(len(q)), np.zeros_like(q)), cell=cell, pbc=flags)

        class Provider:
            metadata = {"periodic": True}

            def evaluate(self, q, *, cell, pbc):
                np.testing.assert_allclose(cell, np.diag([8., 9., 10.]) * BOHR_ANGSTROM)
                np.testing.assert_array_equal(pbc, flags)
                phase = q * (2 * np.pi / np.diag(cell))
                e = np.sum(1 - np.cos(phase), axis=(1, 2))
                f = -np.sin(phase) * (2 * np.pi / np.diag(cell))
                return PESResult(e, f, virial=np.broadcast_to(np.eye(3), (len(q), 3, 3)))

        pes = PythonPotential(Provider(), units={"positions": "angstrom", "energy": "eV"}, cell=cell, pbc=flags)
        q = np.random.default_rng(8).normal(size=(2, 1, 3))
        self.assertLess(check_potential(pes, q)["max_absolute_force_error"], 1e-10)
        np.testing.assert_allclose(pes.evaluate(q).virial, np.broadcast_to(np.eye(3) * EV_HARTREE, (2, 3, 3)))
        for virial in (np.ones((1, 3)), np.full((1, 3, 3), np.nan)):
            with self.assertRaises(PotentialError):
                validate_result(PESResult([0.], np.zeros((1, 1, 3)), virial=virial), (1, 1, 3))

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE runtime required")
    def test_ase_hydrogen_isotopes_share_electronic_energy_and_force(self):
        from ase.calculators.lj import LennardJones

        cell = np.diag([15., 16., 17.])
        q = np.array([[[.1, .2, .3], [2.3, .4, .2]]])
        results = []
        for isotope in ("H", "D", "T"):
            with ASEPotential([isotope, isotope], LennardJones(), cell=cell) as pes:
                np.testing.assert_array_equal(pes.atoms.numbers, [1, 1])
                results.append(pes.evaluate(q))
        for result in results[1:]:
            np.testing.assert_array_equal(result.energy, results[0].energy)
            np.testing.assert_array_equal(result.forces, results[0].forces)
            np.testing.assert_array_equal(result.virial, results[0].virial)

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE runtime required")
    def test_real_ase_periodic_energy_force_virial(self):
        from ase import Atoms
        from ase.calculators.lj import LennardJones
        cell = np.array([[16., 0., 0.], [2., 17., 0.], [0., 0., 18.]])
        q = np.array([[[.2, .3, .1], [2.4, .5, .3]]])
        atoms = Atoms("HH", positions=q[0] * BOHR_ANGSTROM, cell=cell * BOHR_ANGSTROM, pbc=True)
        atoms.calc = LennardJones()
        with ASEPotential(["D", "H"], LennardJones(), cell=cell) as pes:
            result = pes.evaluate(q)
            np.testing.assert_allclose(result.energy, [atoms.get_potential_energy() * EV_HARTREE])
            np.testing.assert_allclose(result.forces[0], atoms.get_forces() * EV_HARTREE * BOHR_ANGSTROM)
            np.testing.assert_allclose(result.virial[0], -atoms.get_stress(voigt=False) * atoms.get_volume() * EV_HARTREE)
            shifted = q.copy()
            shifted[:, 0] += cell[1]
            np.testing.assert_allclose(pes.evaluate(shifted).energy, result.energy, atol=1e-14)
            self.assertLess(check_potential(pes, q)["max_absolute_force_error"], 1e-8)

    def test_ipi_real_cell_inverse_and_virial_wire(self):
        cell = np.array([[9., 0., 0.], [2., 10., 0.], [.3, .4, 12.]])
        with self.assertRaisesRegex(ValueError, "PBC flags"):
            IPIPotential(port=0, cell=cell, pbc=[True, True, False])
        flags = [True, True, False]
        pes = IPIPotential(port=0, timeout=4, cell=cell, pbc=flags, init={"pbc": flags})
        errors = []
        seen = []
        virial = np.arange(9, dtype=float).reshape(3, 3)

        def driver():
            try:
                with socket.create_connection(pes.address) as connection:
                    def recv(n):
                        data = b""
                        while len(data) < n:
                            block = connection.recv(n - len(data))
                            if not block:
                                raise EOFError()
                            data += block
                        return data

                    status = b"NEEDINIT"
                    while True:
                        header = recv(12).strip()
                        if header == b"STATUS":
                            connection.sendall(status.ljust(12))
                        elif header == b"INIT":
                            _, length = struct.unpack("=ii", recv(8))
                            self.assertEqual(json.loads(recv(length))["pbc"], flags)
                            status = b"READY"
                        elif header == b"POSDATA":
                            h = np.frombuffer(recv(72), dtype="=f8").reshape(3, 3)
                            ih = np.frombuffer(recv(72), dtype="=f8").reshape(3, 3)
                            np.testing.assert_array_equal(h, cell.T)
                            np.testing.assert_allclose(ih @ h, np.eye(3), atol=1e-15)
                            count = struct.unpack("=i", recv(4))[0]
                            seen.append(np.frombuffer(recv(count * 24), dtype="=f8").reshape(count, 3).copy())
                            status = b"HAVEDATA"
                        elif header == b"GETFORCE":
                            connection.sendall(b"FORCEREADY".ljust(12) + struct.pack("=di", 0., count)
                                               + np.zeros((count, 3), dtype="=f8").tobytes()
                                               + virial.T.astype("=f8").tobytes() + struct.pack("=i", 0))
                            status = b"READY"
                        elif header == b"EXIT":
                            return
                        else:
                            raise ValueError(header)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=driver, daemon=True)
        thread.start()
        try:
            q = np.array([[[19., -2., 1.]], [[21., 3., 1.]]])
            result = pes.evaluate(q)
            np.testing.assert_array_equal(seen, q)
            np.testing.assert_array_equal(result.virial, np.broadcast_to(virial, (2, 3, 3)))
        finally:
            pes.close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        if errors:
            raise errors[0]


if __name__ == "__main__":
    unittest.main()
