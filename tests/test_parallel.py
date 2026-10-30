import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.parallel import execution_plan, run_tasks, ProcessBeadPotential
from pyrpmd.parallel.execution import _context, make_task_potential
from pyrpmd.potentials import HarmonicPotential


def square(x):
    return x * x


class ParallelTests(unittest.TestCase):
    def test_spawn_tasks_order(self):
        self.assertEqual(
            run_tasks(
                square, range(7), {"backend": "process", "trajectory_workers": 3}
            ),
            [x * x for x in range(7)],
        )

    def test_persistent_bead_chunks_match_serial(self):
        q = np.random.default_rng(7).normal(size=(5, 2, 3))
        expected = HarmonicPotential(k=0.3).evaluate(q)
        p = ProcessBeadPotential({"provider": "harmonic", "k": 0.3}, None, 3)
        try:
            for x in (q, q + 1):
                result = p.evaluate(x)
                reference = HarmonicPotential(k=0.3).evaluate(x)
                np.testing.assert_allclose(result.energy, reference.energy)
                np.testing.assert_allclose(result.forces, reference.forces)
        finally:
            p.close()
        with self.assertRaises(RuntimeError):
            p.evaluate(q)

    def test_gpu_one_and_multiple_trajectories_per_card(self):
        for per_card in (1, 3):
            plan = execution_plan(
                {"mode": "gpu", "devices": [0, 2], "trajectories_per_gpu": per_card}
            )
            self.assertEqual(plan.trajectory_workers, 2 * per_card)
            devices = []

            class Provider:
                def __init__(self, device):
                    self.metadata = {"execution_device": device}

            def factory(spec, system):
                devices.append(spec["device"])
                return Provider(spec["device"])

            with patch("pyrpmd.potentials.make_potential", side_effect=factory):
                for slot in range(2 * per_card):
                    with _context(plan, slot):
                        make_task_potential(
                            {"provider": "torch", "model": "test"}, None
                        )
            self.assertEqual(devices, ["cuda:0"] * per_card + ["cuda:2"] * per_card)

    def test_periodic_ase_virial_survives_bead_processes(self):
        try:
            from ase.calculators.lj import LennardJones
        except ImportError:
            self.skipTest("optional ASE runtime not installed")
        from pyrpmd.potentials.optional import ASEPotential
        from pyrpmd.system import System

        system = System(("H", "D"), np.array([1837.0, 3671.0]),
                        np.array([[0.0, 0.0, 0.0], [2.2, 0.3, 0.0]]),
                        cell=np.diag([10.0, 11.0, 12.0]), pbc=[True, True, False])
        serial = ASEPotential(system.symbols, LennardJones(rc=2.5),
                              cell=system.cell, pbc=system.pbc)
        parallel = ProcessBeadPotential(
            {"provider": "ase", "factory": "ase.calculators.lj:LennardJones",
             "params": {"rc": 2.5}}, system, 3)
        q = np.repeat(system.geometry[None], 5, axis=0)
        q[:, 1, 0] += np.arange(5) * 0.02
        q[3, 0] += system.cell[0]
        try:
            reference, result = serial.evaluate(q), parallel.evaluate(q)
            for field in ("energy", "forces", "virial"):
                np.testing.assert_allclose(getattr(result, field), getattr(reference, field),
                                           rtol=1e-13, atol=1e-13)
        finally:
            serial.close()
            parallel.close()

    def test_inconsistent_chunk_virials_rejected(self):
        from pyrpmd.parallel.execution import _combine
        from pyrpmd.potentials import PESResult, PotentialError

        a = PESResult(np.zeros(1), np.zeros((1, 1, 3)), virial=np.zeros((1, 3, 3)))
        b = PESResult(np.zeros(1), np.zeros((1, 1, 3)))
        with self.assertRaisesRegex(PotentialError, "virial availability"):
            _combine([a, b])

    def test_gpu_does_not_accept_cpu_provider(self):
        plan = execution_plan({"mode": "gpu", "devices": [0]})
        with _context(plan, 0):
            with self.assertRaises(ValueError):
                make_task_potential({"provider": "free"}, None)

    def test_unsupported_resource_topology_rejected(self):
        for config in (
            {"pes_ranks": 2},
            {"coordinator_ranks": 1},
            {"mode": "gpu", "bead_workers": 2},
            {"devices": [0, 0], "mode": "gpu"},
        ):
            with self.assertRaises(ValueError):
                execution_plan(config)


if __name__ == "__main__":
    unittest.main()
