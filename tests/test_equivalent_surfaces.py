"""Reference min(max()) surface and zero-independent energy diagnostics."""

import tempfile
import unittest
import numpy as np
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.bias import FluxMetricBias
from pyrpmd.system import System
from pyrpmd.potentials import HarmonicPotential, PESResult
from pyrpmd.workflows.rate import _child_task
from pyrpmd.workflows.rate import _aggregate
from pyrpmd.workflows.fes import separate_reactants_to_xi


class ShiftedHarmonic(HarmonicPotential):
    def __init__(self, offset=0):
        super().__init__(k=0.03)
        self.offset = offset

    def evaluate(self, q):
        r = super().evaluate(q)
        return PESResult(r.energy + self.offset, r.forces)


class EquivalentSurfaces(unittest.TestCase):
    def coordinate(self, masses=None, permutations=None):
        return RPMDrateCoordinate(
            [[0], [1, 2]],
            np.ones(3) * 1800 if masses is None else masses,
            np.array([[-1.75, 0, 0], [0, 0, 0], [1.75, 0, 0]]),
            [[0, 1]],
            [[1, 2]],
            10,
            equivalent_paths=[[0, 2, 1]] if permutations is None else permutations,
        )

    def test_h3_both_products_and_entrance(self):
        cv = self.coordinate()
        entrance = np.array([[-10.0, 0, 0], [-0.7, 0, 0], [0.7, 0, 0]])
        product01 = np.array([[-0.7, 0, 0], [0.7, 0, 0], [10.0, 0, 0]])
        product02 = product01[[0, 2, 1]]
        self.assertLess(cv.surfaces(entrance)[3], 0)
        self.assertGreater(cv.surfaces(product01)[3], 0)
        self.assertGreater(cv.surfaces(product02)[3], 0)
        for q in (entrance, product01, product02):
            expected = np.linalg.norm(q[1] - q[2]) - min(
                np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[0] - q[2])
            )
            self.assertAlmostEqual(cv.surfaces(q)[3], expected, places=13)
            self.assertAlmostEqual(
                cv.surfaces(q)[3], cv.surfaces(q[[0, 2, 1]])[3], places=13
            )

    def test_equivalent_branch_gradient_hessian_and_metric(self):
        cv = self.coordinate()
        q = np.array([[-1.5, 0.2, 0.1], [0.1, 0, 0.1], [1.9, 0.3, -0.2]])[[0, 2, 1]]
        value, g, h = cv.centroid_value_gradient_hessian(q)
        delta = 1e-5
        gn = np.empty_like(g)
        hn = np.empty_like(h)
        for i in np.ndindex(q.shape):
            plus = q.copy()
            minus = q.copy()
            plus[i] += delta
            minus[i] -= delta
            vp, gp = cv.centroid_value_gradient(plus)
            vm, gm = cv.centroid_value_gradient(minus)
            gn[i] = (vp - vm) / (2 * delta)
            hn[:, :, i[0], i[1]] = (gp - gm) / (2 * delta)
        np.testing.assert_allclose(g, gn, atol=1e-9)
        np.testing.assert_allclose(h, hn, atol=1e-8)
        bias = FluxMetricBias(cv, np.ones(3) * 1800, 200)
        ring = np.repeat(q[None], 4, axis=0)
        f = bias.energy_force(ring)[1]
        i = (2, 1, 0)
        plus = ring.copy()
        minus = ring.copy()
        plus[i] += delta
        minus[i] -= delta
        expected = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / (
            2 * delta
        )
        self.assertAlmostEqual(f[i], expected, places=9)

    def test_permutations_are_mass_and_partition_preserving_bijections(self):
        for permutation in ([0, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1], [0, 2.0, 1]):
            with self.subTest(permutation=permutation):
                with self.assertRaises(ValueError):
                    self.coordinate(permutations=[permutation])
        with self.assertRaises(ValueError):
            self.coordinate(masses=np.array([1800, 1800, 3600]))

    def test_multiple_bonds_match_minimum_of_column_maxima(self):
        reference = np.array(
            [[-2.0, 0.1, 0], [0.0, 0, 0], [1.8, 0.2, 0], [3.1, 0.4, 0.1]]
        )
        permutations = [[0, 3, 2, 1]]
        cv = RPMDrateCoordinate(
            [[0], [1, 2, 3]],
            np.ones(4),
            reference,
            [[0, 1], [1, 2]],
            [[1, 2], [2, 3]],
            12,
            equivalent_paths=permutations,
        )
        rng = np.random.default_rng(145)
        found_difference = False
        for _ in range(40):
            q = reference + rng.normal(scale=0.5, size=(4, 3))
            values = []
            for forming, breaking in cv.paths:
                values.append(
                    [
                        ref
                        - np.linalg.norm(q[f[0]] - q[f[1]])
                        + np.linalg.norm(q[b[0]] - q[b[1]])
                        for ref, f, b in zip(cv.reference_difference, forming, breaking)
                    ]
                )
            values = np.asarray(values)
            exact = float(np.min(np.max(values, axis=0)))
            self.assertAlmostEqual(cv.surfaces(q)[3], exact, places=12)
            found_difference |= not np.isclose(exact, np.max(np.min(values, axis=1)))
        self.assertTrue(
            found_difference, "test must distinguish min(max()) from max(min())"
        )

    def test_child_energy_error_does_not_depend_on_potential_zero(self):
        system = System(
            ("H", "H", "H"),
            np.ones(3) * 2,
            np.array([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]]),
            ((0,), (1, 2)),
        )
        common = {
            "beta": 5.0,
            "beads": 4,
            "dt_au": 0.05,
            "seed": 13,
            "native": False,
            "sampling": {},
            "reaction": {
                "fragments": [[0], [1, 2]],
                "forming_bonds": [[0, 1]],
                "breaking_bonds": [[1, 2]],
                "rinf": 10,
                "equivalent_paths": [[0, 2, 1]],
            },
        }
        results = []
        for offset in (0.0, -100.0):
            config = {
                **common,
                "potential": {
                    "provider": "python",
                    "factory": __name__ + ":ShiftedHarmonic",
                    "params": {"offset": offset},
                },
            }
            with tempfile.TemporaryDirectory() as output:
                results.append(
                    _child_task(
                        {
                            "config": config,
                            "system": system,
                            "chain": 0,
                            "parent": 0,
                            "pair": 0,
                            "q": np.repeat(system.geometry[None], 4, axis=0),
                            "xi_star": 1.0,
                            "child_steps": 100,
                            "output": output,
                        }
                    )
                )
        for key in (
            "endpoint_energy_error_hartree_per_bead",
            "endpoint_energy_error_kbt",
        ):
            self.assertGreater(results[0][key], 0)
            self.assertAlmostEqual(results[0][key], results[1][key], places=11)

    def test_entrance_initialization_keeps_internal_bonds(self):
        cv = self.coordinate()
        geometry = cv.reference
        for target in (-0.05, 0.0, 0.2, 0.5, 0.8):
            q = separate_reactants_to_xi(geometry, cv, target, cv.masses)
            self.assertAlmostEqual(np.linalg.norm(q[1] - q[2]), 1.75, places=12)
            self.assertAlmostEqual(cv.centroid_value_gradient(q)[0], target, places=10)
            self.assertGreater(
                min(np.linalg.norm(q[i] - q[j]) for i, j in ((0, 1), (0, 2), (1, 2))),
                1.0,
            )
            np.testing.assert_allclose(
                np.average(q, axis=0, weights=cv.masses),
                np.average(geometry, axis=0, weights=cv.masses),
                atol=1e-12,
            )

    def test_publication_gate_is_energy_zero_independent(self):
        pmf = {"k_qtst_cm3_s": 1.0, "free_energy_barrier_hartree": 0.01, "warnings": []}
        config = {"seed": 1, "beta": 5.0, "dt_au": 0.1}
        options = {
            "child_steps": 40,
            "plateau_fraction": 0.5,
            "plateau_tolerance": 0.1,
            "bootstrap": 4,
        }
        results = []
        for legacy in (0.9, 0.0001):
            children = [
                {
                    "chain": 0,
                    "parent": i,
                    "pair": 0,
                    "numerator": np.ones(40) * 0.8,
                    "denominator": 1.0,
                    "energy_drift": legacy,
                    "endpoint_energy_error_hartree_per_bead": 0.01,
                }
                for i in range(2)
            ]
            with tempfile.TemporaryDirectory() as output:
                results.append(_aggregate(children, pmf, config, output, options))
        self.assertEqual(results[0]["rate_cm3_s"], results[1]["rate_cm3_s"])
        self.assertIsNotNone(results[0]["rate_cm3_s"])
        children[0].pop("endpoint_energy_error_hartree_per_bead")
        with tempfile.TemporaryDirectory() as output:
            self.assertIsNone(
                _aggregate(children, pmf, config, output, options)["rate_cm3_s"]
            )


if __name__ == "__main__":
    unittest.main()
