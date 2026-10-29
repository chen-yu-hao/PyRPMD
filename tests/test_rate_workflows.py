import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np

from pyrpmd.cv import Distance, RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, resolve_umbrellas, reanalyze_fes, _fingerprint
from pyrpmd.workflows.rate import (
    project_positions,
    project_momenta,
    constrained_step,
    _child_task,
    run_rate,
    _recrossing_options,
)
from pyrpmd.workflows.prepare import numerical_hessian, prepare_rate


class SphereSurface(Distance):
    def centroid_value_gradient(self, q):
        r, g = super().centroid_value_gradient(q)
        return r - 1.4, g


class RateSamplerTests(unittest.TestCase):
    def test_recrossing_integer_counts_are_not_truncated(self):
        for value in (2.9, True, -1):
            config = {
                "dt_au": 1.0,
                "sampling": {},
                "recrossing": {"child_steps": value},
            }
            with self.assertRaises(ValueError):
                _recrossing_options(config)

    def test_nonlinear_cayley_rattle_constraint_and_reversibility(self):
        masses = np.array([2.0, 3.0])
        surface = SphereSurface([0, 1])
        rng = np.random.default_rng(1)
        q = np.repeat(
            np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])[None], 4, axis=0
        ) + rng.normal(scale=0.01, size=(4, 2, 3))
        dyn = Dynamics(FreePotential(), masses, 3.0, 4, 0.002, seed=11, native=False)
        state = State(project_positions(q, surface, masses), dyn.sample_momenta())
        project_momenta(state.p, state.q, surface, masses)
        original = state.copy()
        for _ in range(120):
            constrained_step(dyn, state, surface)
            value, g = surface.centroid_value_gradient(state.q.mean(0))
            self.assertLess(abs(value), 2e-10)
            self.assertLess(abs(np.sum(g * state.p.mean(0) / masses[:, None])), 1e-11)
        state.p *= -1
        for _ in range(120):
            constrained_step(dyn, state, surface)
        np.testing.assert_allclose(state.q, original.q, atol=2e-8)
        np.testing.assert_allclose(state.p, -original.p, atol=2e-8)

    def test_child_pair_unconstrained_initial_flux(self):
        system = System(
            ("H", "H", "H"),
            np.array([2.0, 2.0, 2.0]),
            np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            ((0,), (1, 2)),
        )
        config = {
            "beta": 10.0,
            "beads": 4,
            "dt_au": 1e-4,
            "seed": 1,
            "native": False,
            "potential": {"provider": "free"},
            "sampling": {},
            "reaction": {
                "fragments": [[0], [1, 2]],
                "forming_bonds": [[0, 1]],
                "breaking_bonds": [[1, 2]],
                "rinf": 10.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            task = {
                "config": config,
                "system": system,
                "chain": 0,
                "parent": 0,
                "pair": 0,
                "q": np.repeat(system.geometry[None], 4, axis=0),
                "xi_star": 1.0,
                "child_steps": 10,
                "output": directory,
            }
            result = _child_task(task)
            again = _child_task(task)
            self.assertGreater(result["denominator"], 0)
            np.testing.assert_allclose(result["numerator"] / result["denominator"], 1.0)
            np.testing.assert_array_equal(result["numerator"], again["numerator"])

    def test_numerical_hessian_and_ambiguous_preparation(self):
        q = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        potential = HarmonicPotential(k=0.02, center=q)
        h = numerical_hessian(potential, q)
        np.testing.assert_allclose(h, np.eye(9) * 0.02, atol=1e-10)
        system = System(("H", "H", "H"), np.ones(3) * 1800, q)
        result = prepare_rate({"reaction": {}}, system, potential, numerical=True)
        self.assertFalse(result["ready"])
        self.assertIn("negative", result["diagnostics"][1])

    def test_teaching_pes_ts_only_inference_and_gradient(self):
        from pyrpmd.runner import prepare
        from pyrpmd.potentials.python import import_object

        source = Path(__file__).resolve().parents[1] / "examples" / "rate"
        result = prepare(source / "pilot.inp", numerical=True)
        self.assertTrue(result["ready"])
        reaction = result["reaction_preparation"]["reaction"]
        self.assertEqual(reaction["forming_bonds"], [[0, 1]])
        potential = import_object(str(source / "toy_exchange.py") + ":build")()
        x = np.array([[[-2.0, 0.2, 0.1], [0.1, 0.0, 0.0], [1.7, 0.1, -0.1]]])
        force = potential.evaluate(x).forces
        numerical = np.empty_like(force)
        h = 1e-6
        for idx in np.ndindex(x.shape):
            plus = x.copy()
            minus = x.copy()
            plus[idx] += h
            minus[idx] -= h
            numerical[idx] = -(
                potential.evaluate(plus).energy[0] - potential.evaluate(minus).energy[0]
            ) / (2 * h)
        np.testing.assert_allclose(force, numerical, atol=1e-8, rtol=1e-6)
        isotope_system = System.from_dict(result["system"])
        isotope_system.masses[0] *= 2
        ambiguous = prepare_rate(
            {"reaction": {}}, isotope_system, potential, numerical=True
        )
        self.assertFalse(ambiguous["ready"])
        self.assertEqual(len(ambiguous["candidates"]), 2)
        selected = prepare_rate(
            {"reaction": {"entry_side": 1}}, isotope_system, potential, numerical=True
        )
        self.assertTrue(selected["ready"])

    def test_full_rate_pipeline_counts_real_children(self):
        # A free model exercises plumbing/flux normalization only; it is not
        # a molecular rate benchmark. The teaching example tests a real TS.
        system = System(
            ("H", "H", "H"),
            np.array([2.0, 2.0, 2.0]),
            np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            ((0,), (1, 2)),
        )
        config = {
            "task": "rate",
            "beta": 5.0,
            "temperature_K": 300.0,
            "beads": 2,
            "dt_au": 0.003,
            "seed": 17,
            "native": False,
            "potential": {"provider": "free"},
            "sampling": {
                "equilibration_steps": 50,
                "production_steps": 200,
                "sample_interval": 2,
                "andersen_interval": 20,
            },
            "reaction": {
                "fragments": [[0], [1, 2]],
                "forming_bonds": [[0, 1]],
                "breaking_bonds": [[1, 2]],
                "rinf": 4.0,
            },
            "umbrella": {"centers": np.linspace(0, 1, 11).tolist(), "kappa": 10.0},
            "analysis": {"bins": 30, "bootstrap": 2},
            "parallel": {"backend": "serial"},
            "output": {"checkpoint_steps": 200},
            "recrossing": {
                "dividing_surface": 1.0,
                "parent_replicas": 1,
                "parent_samples": 2,
                "equilibration_steps": 20,
                "parent_interval_steps": 5,
                "child_pairs": 1,
                "child_steps": 20,
                "bootstrap": 10,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_rate(config, system, directory)
            self.assertEqual(result["child_trajectories"], 4)
            self.assertTrue(np.isfinite(result["kappa"]))
            self.assertFalse(result["converged"])
            self.assertTrue((Path(directory) / "recrossing.csv").is_file())


class FESWorkflowTests(unittest.TestCase):
    def config(self):
        return {
            "task": "fes",
            "beta": 2.0,
            "temperature_K": 150.0,
            "beads": 4,
            "dt_au": 0.03,
            "seed": 2,
            "native": False,
            "potential": {"provider": "harmonic", "params": {"k": 0.2}},
            "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
            "umbrella": {"centers": [-0.7, 0.0, 0.7], "stiffness": 2.0},
            "sampling": {
                "equilibration_steps": 100,
                "production_steps": 300,
                "sample_interval": 3,
                "andersen_interval": 20,
            },
            "analysis": {"bins": 12, "bootstrap": 3},
            "parallel": {"backend": "serial"},
            "output": {"checkpoint_steps": 80},
        }

    def test_actual_sampling_resume_and_physics_guard(self):
        system = System(("H",), np.array([2.0]), np.zeros((1, 3)))
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            first = run_fes(config, system, directory, return_data=True)
            second = run_fes(config, system, directory, return_data=True)
            self.assertEqual(first["summary"]["samples"], 300)
            for a, b in zip(first["streams"], second["streams"]):
                np.testing.assert_array_equal(a["samples"], b["samples"])
                np.testing.assert_array_equal(a["last_q"], b["last_q"])
            self.assertTrue((Path(directory) / "fes.csv").is_file())
            reanalysis = copy.deepcopy(config)
            reanalysis["analysis"]["bins"] = 7
            reanalysis["resume"] = True
            rebuilt = reanalyze_fes(reanalysis, system, directory, return_data=True)
            self.assertEqual(rebuilt["data"]["free_energy"].shape, (7,))
            altered = copy.deepcopy(config)
            altered["beta"] *= 1.1
            with self.assertRaises(Exception):
                run_fes(altered, system, directory)

    def test_mixed_units_and_2d_grid(self):
        system = System(
            ("H", "H"), np.ones(2), np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        )
        config = {
            "cvs": [
                {"name": "r", "kind": "distance", "atoms": [0, 1], "unit": "angstrom"},
                {"name": "x", "kind": "cartesian", "atom": 0, "axis": "y"},
            ],
            "umbrella": {
                "grid": [
                    {"start": 1.0, "stop": 2.0, "count": 3},
                    {"start": -1.0, "stop": 1.0, "count": 2},
                ],
                "stiffness": ["10 kcal/mol/angstrom^2", "2 hartree/bohr^2"],
            },
        }
        cv, specs, centers, k = resolve_umbrellas(config, system)
        self.assertEqual(centers.shape, (6, 2))
        self.assertGreater(centers[0, 0], 1.8)
        self.assertEqual(k[1, 1], 2.0)
        config["umbrella"]["grid"][0]["count"] = 2.9
        with self.assertRaises(ValueError):
            resolve_umbrellas(config, system)

    def test_measure_is_physics_but_resume_and_bins_are_not(self):
        system = System(("H",), np.array([2.0]), np.zeros((1, 3)))
        config = self.config()
        first = _fingerprint(config, system, {})
        config["resume"] = True
        config["analysis"]["bins"] = 19
        self.assertEqual(first, _fingerprint(config, system, {}))
        config["analysis"]["measure"] = "rate_flux"
        self.assertNotEqual(first, _fingerprint(config, system, {}))


if __name__ == "__main__":
    unittest.main()
