"""Independent physics checks for the opt-in full-potential diatom QCT orbit."""
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad, solve_ivp

from benchmarks.h3.diatomic_action import (
    ActionConvergenceError, ActionDomainError, DiatomicAction, H2_REDUCED_MASS,
)


MU = H2_REDUCED_MASS
OMEGA = 0.020064050017587955
R0 = 1.4014719755049045
LIBRARY = Path(__file__).resolve().parents[1] / "benchmarks/h3/build/libbkmp2.so"


def harmonic(**kwargs):
    return DiatomicAction(lambda r: 0.5*MU*OMEGA**2*(r-R0)**2, MU,
                          gradient=lambda r: MU*OMEGA**2*(r-R0),
                          equilibrium_bracket=(1.2, 1.6), **kwargs)


def morse(**kwargs):
    depth = 0.1745
    alpha = OMEGA*np.sqrt(MU/(2*depth))
    def potential(r):
        return depth*(1-np.exp(-alpha*(r-R0)))**2
    def gradient(r):
        x = np.exp(-alpha*(r-R0))
        return 2*depth*alpha*x*(1-x)
    return DiatomicAction(potential, MU, gradient=gradient,
                          equilibrium_bracket=(1.2, 1.6), **kwargs), depth


class AnalyticDiatomicActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = harmonic()
        cls.orbits = [cls.model.quantize(v) for v in range(3)]

    def test_harmonic_action_energy_turning_points_and_period(self):
        for v, orbit in enumerate(self.orbits):
            with self.subTest(v=v):
                energy = (v+0.5)*OMEGA
                amplitude = np.sqrt(2*energy/(MU*OMEGA**2))
                self.assertAlmostEqual(orbit.action, v+0.5, places=10)
                self.assertAlmostEqual(orbit.energy_hartree, energy, places=12)
                np.testing.assert_allclose(orbit.turning_points_bohr,
                                           [R0-amplitude, R0+amplitude], rtol=0, atol=1e-11)
                self.assertAlmostEqual(orbit.period_au/(2*np.pi/OMEGA), 1, places=10)
        self.assertEqual(self.model.action(0), 0)

    def test_harmonic_phase_equals_exact_time_evolution(self):
        phases = np.r_[np.linspace(0, 1, 257), 1e-12, 0.5-1e-12, 0.5+1e-12, 1-1e-12]
        for orbit in self.orbits:
            amplitude = np.sqrt(2*orbit.energy_hartree/(MU*OMEGA**2))
            r, p = orbit.sample_phase(phases)
            np.testing.assert_allclose(r, R0-amplitude*np.cos(2*np.pi*phases), rtol=0, atol=2e-11)
            # p is evaluated from E-V; relative error is ill-conditioned at a
            # turning point, where only an absolute momentum check is useful.
            np.testing.assert_allclose(p, MU*OMEGA*amplitude*np.sin(2*np.pi*phases), rtol=0, atol=2e-7)

    def test_time_uniform_density_has_harmonic_moments(self):
        orbit = self.orbits[1]
        phases = (np.arange(8192)+0.5)/8192
        r, p = orbit.sample_phase(phases)
        amplitude = np.sqrt(2*orbit.energy_hartree/(MU*OMEGA**2))
        self.assertAlmostEqual(float(np.mean(r)), R0, places=11)
        self.assertAlmostEqual(float(np.mean((r-R0)**2)), amplitude**2/2, places=11)
        self.assertAlmostEqual(float(np.mean((r-R0)**4)), 3*amplitude**4/8, places=11)
        self.assertAlmostEqual(float(np.mean(p)), 0, places=10)
        self.assertAlmostEqual(float(np.mean(p*p/(2*MU))), orbit.energy_hartree/2, places=11)

    def test_rng_consumes_one_uniform_and_is_deterministic(self):
        rng, oracle = np.random.default_rng(817), np.random.default_rng(817)
        expected = self.orbits[0].sample_phase(oracle.random())
        np.testing.assert_array_equal(self.orbits[0].sample(rng), expected)
        self.assertEqual(rng.bit_generator.state, oracle.bit_generator.state)
        expected = self.orbits[0].sample_phase(oracle.random(17))
        actual = self.orbits[0].sample(rng, size=17)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(rng.bit_generator.state, oracle.bit_generator.state)

    def test_cartesian_adapter_real_masses_com_and_j0(self):
        # Unequal masses exercise the mass factors, not only symmetric H2.
        masses = np.array([3*MU, 1.5*MU])
        orbit = self.orbits[1]
        phases = np.array([0.0, 0.173, 0.5, 0.77, 1.0])
        q, p = orbit.cartesian_at_phase(phases, masses=masses, axis=(1, -2, 3))
        np.testing.assert_allclose(np.sum(q*masses[None, :, None], axis=1), 0, atol=5e-13)
        np.testing.assert_array_equal(p.sum(axis=1), 0)
        np.testing.assert_allclose(np.cross(q, p).sum(axis=1), 0, atol=1e-14)
        r = np.linalg.norm(q[:, 1]-q[:, 0], axis=1)
        energy = np.sum(p*p/(2*masses[None, :, None]), axis=(1, 2)) + self.model.values(r)
        np.testing.assert_allclose(energy, orbit.energy_hartree, rtol=0, atol=1e-12)
        rng, oracle = np.random.default_rng(99), np.random.default_rng(99)
        expected = orbit.cartesian_at_phase(oracle.random(), masses=masses)
        actual = orbit.sample_cartesian(rng, masses=masses)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(rng.bit_generator.state, oracle.bit_generator.state)

    def test_morse_nonharmonic_analytic_quantization_and_period(self):
        model, depth = morse()
        for v in (0, 1, 2):
            orbit = model.quantize(v)
            action = v+0.5
            exact_energy = OMEGA*action-OMEGA**2*action**2/(4*depth)
            exact_period = 2*np.pi/(OMEGA*np.sqrt(1-exact_energy/depth))
            self.assertAlmostEqual(orbit.energy_hartree, exact_energy, places=11)
            self.assertAlmostEqual(orbit.period_au/exact_period, 1, places=9)

    def test_phase_mass_axis_and_quantum_number_guards(self):
        orbit = self.orbits[0]
        for phase in (-0.1, 1.1, np.nan, np.inf, [0.0, np.nan], 1+2j, True):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                orbit.sample_phase(phase)
        for v in (-1, 0.5, True):
            with self.subTest(v=v), self.assertRaises(ValueError):
                self.model.quantize(v)
        for masses in ((1, 2), (0, 1), (np.nan, 1)):
            with self.subTest(masses=masses), self.assertRaises(ValueError):
                orbit.cartesian_at_phase(0.2, masses=masses)
        for axis in ((0, 0, 0), (0, 1), (0, np.inf, 1)):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                orbit.cartesian_at_phase(0.2, axis=axis)
        with self.assertRaises(TypeError):
            orbit.sample(817)

    def test_bounded_domain_failures_do_not_extrapolate_potential(self):
        for r in (0.199, 12.001, np.nan):
            with self.subTest(r=r), self.assertRaises(ActionDomainError):
                self.model.values(r)
        with self.assertRaises(ActionDomainError):
            self.model.turning_points(self.model.energy_limit_hartree)
        with self.assertRaises(ActionDomainError):
            self.model.quantize(100)
        with self.assertRaises(ActionDomainError):
            DiatomicAction(lambda r: (r-2)**2, 1, gradient=lambda r: 2*(r-2), domain=(-1, 4))
        with self.assertRaises(ActionDomainError):
            DiatomicAction(lambda r: (r-1)**2*(r-3)**2, 1,
                            gradient=lambda r: 4*(r-1)*(r-2)*(r-3),
                            domain=(0.2, 4), equilibrium_bracket=(0.8, 1.2))
        with self.assertRaises(ActionDomainError):
            DiatomicAction.from_bkmp2(domain=(0.1, 12))

    def test_refinement_limits_fail_explicitly(self):
        model, _ = morse(time_cells=16, max_time_cells=32, time_tolerance=1e-13)
        with self.assertRaisesRegex(ActionConvergenceError, "time CDF"):
            model.quantize(1)
        model, _ = morse(quadrature_order=8, max_quadrature_order=16,
                          quadrature_rtol=1e-15, quadrature_atol=1e-16)
        with self.assertRaisesRegex(ActionConvergenceError, "quadrature"):
            model.action(0.06)


@unittest.skipUnless(LIBRARY.is_file(), "Build verified benchmarks/h3/build/libbkmp2.so for the real-PES checks")
class BKMP2DiatomicActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DiatomicAction.from_bkmp2()
        cls.orbits = [cls.model.quantize(v) for v in range(3)]

    def test_action_against_direct_untransformed_quadrature(self):
        for v, orbit in enumerate(self.orbits):
            with self.subTest(v=v):
                left, right = orbit.turning_points_bohr
                result, error = quad(
                    lambda r: np.sqrt(max(0, 2*MU*(orbit.energy_hartree-self.model.values(r))))/np.pi,
                    left, right, epsabs=1e-11, epsrel=1e-11, limit=150)
                self.assertLess(error, 2e-10)
                self.assertAlmostEqual(result, v+0.5, places=9)
                np.testing.assert_allclose(self.model.values(np.asarray([left, right])),
                                           orbit.energy_hartree, rtol=0, atol=2e-13)

    def test_time_phase_matches_independent_hamilton_equation_integration(self):
        for orbit in self.orbits[:2]:
            with self.subTest(v=orbit.v):
                phase = np.linspace(0, 1, 129)
                def derivative(t, y):
                    return [y[1]/MU, -self.model.gradients(y[0])]
                solution = solve_ivp(derivative, [0, orbit.period_au],
                                     [orbit.turning_points_bohr[0], 0.0],
                                     method="DOP853", rtol=2e-12, atol=2e-12,
                                     t_eval=phase*orbit.period_au)
                self.assertTrue(solution.success)
                r, p = orbit.sample_phase(phase)
                np.testing.assert_allclose(r, solution.y[0], rtol=0, atol=3e-8)
                np.testing.assert_allclose(p, solution.y[1], rtol=0, atol=1e-6)

    def test_actual_pes_energy_shell_and_metadata_scope(self):
        for orbit in self.orbits:
            with self.subTest(v=orbit.v):
                r, p = orbit.sample(np.random.default_rng(995+orbit.v), size=1024)
                energy = p*p/(2*MU)+self.model.values(r)
                np.testing.assert_allclose(energy, orbit.energy_hartree, rtol=0, atol=3e-13)
                self.assertAlmostEqual(orbit.energy_absolute_hartree,
                                       orbit.energy_hartree+self.model.minimum_hartree, places=14)
                self.assertIn("not an exact quantum eigenvalue", orbit.metadata["scope"])
                self.assertEqual(orbit.metadata["reference"]["provenance"]["spectator_pair_distances_bohr"], [99., 99.])

    def test_bkmp2_domain_and_quadrature_refinement(self):
        tighter = DiatomicAction.from_bkmp2(domain=(0.25, 10), quadrature_order=128,
                                           time_cells=256, time_tolerance=2e-9)
        for baseline in self.orbits[:2]:
            refined = tighter.quantize(baseline.v)
            self.assertAlmostEqual(refined.energy_hartree, baseline.energy_hartree, places=11)
            self.assertAlmostEqual(refined.period_au/baseline.period_au, 1, places=9)
            phases = np.linspace(0, 1, 513)
            for first, second in zip(refined.sample_phase(phases), baseline.sample_phase(phases)):
                np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
