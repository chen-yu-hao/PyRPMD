import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pyrpmd.dynamics import State, Dynamics, CayleyIntegrator, native_capabilities
from pyrpmd.potentials import (
    FreePotential,
    HarmonicPotential,
    PESResult,
    PotentialError,
)


class CayleyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)

    def reference(self, q, p, m, beta, dt):
        P = len(q)
        L = (
            2 * np.eye(P)
            - np.roll(np.eye(P), 1, axis=0)
            - np.roll(np.eye(P), -1, axis=0)
        )
        M = np.eye(P) + 0.25 * dt**2 * (P / beta) ** 2 * L
        mid = np.linalg.solve(
            M, (q + 0.5 * dt * p / m[None, :, None]).reshape(P, -1)
        ).reshape(q.shape)
        return 2 * mid - q, p - dt * m[None, :, None] * (P / beta) ** 2 * np.einsum(
            "jk,kac->jac", L, mid
        )

    def test_numpy_and_native_against_independent_dense_map(self):
        choices = [False] + ([True] if native_capabilities()["available"] else [])
        for P in [1, 2, 3, 5, 8, 32]:
            for dt in [0.0, 0.04, -0.04]:
                m = np.array([1.0, 3.0])
                q = self.rng.normal(size=(P, 2, 3))
                p = self.rng.normal(size=q.shape)
                qr, pr = self.reference(q, p, m, 3.0, dt)
                for native in choices:
                    with self.subTest(P=P, dt=dt, native=native):
                        a = CayleyIntegrator(m, 3.0, P, dt, native=native)
                        qn = q.copy()
                        pn = p.copy()
                        a.free_step(qn, pn)
                        np.testing.assert_allclose(qn, qr, atol=3e-13)
                        np.testing.assert_allclose(pn, pr, atol=3e-13)

    def test_square_root_half_steps(self):
        for P in [1, 2, 3, 8]:
            q = self.rng.normal(size=(P, 2, 3))
            p = self.rng.normal(size=q.shape)
            a = CayleyIntegrator([2, 3], 2.0, P, 0.11, native=False)
            q1 = q.copy()
            p1 = p.copy()
            q2 = q.copy()
            p2 = p.copy()
            a.free_step(q1, p1)
            a.half_step(q2, p2)
            a.half_step(q2, p2)
            np.testing.assert_allclose(q1, q2, atol=3e-14)
            np.testing.assert_allclose(p1, p2, atol=3e-14)

    def test_time_reversal_and_energy(self):
        dyn = Dynamics(HarmonicPotential(0.4), [2, 3], 4.0, 8, 0.005)
        state = State(self.rng.normal(size=(8, 2, 3)), self.rng.normal(size=(8, 2, 3)))
        original = state.copy()
        E0 = dyn.energy(state)
        for _ in range(100):
            dyn.step(state)
        self.assertLess(abs(dyn.energy(state) - E0) / abs(E0), 3e-5)
        state.p *= -1
        for _ in range(100):
            dyn.step(state)
        np.testing.assert_allclose(state.q, original.q, atol=2e-12)
        np.testing.assert_allclose(state.p, -original.p, atol=2e-12)

    def test_p1_velocity_verlet(self):
        d = Dynamics(HarmonicPotential(2.0), [3.0], 7.0, 1, 0.2, native=False)
        s = State(np.array([[[1.0, 2.0, 3.0]]]), np.array([[[0.2, 0.3, 0.4]]]))
        p_half = s.p - 0.1 * 2 * s.q
        q = s.q + 0.2 * p_half / 3
        p = p_half - 0.1 * 2 * q
        d.step(s)
        np.testing.assert_allclose(s.q, q)
        np.testing.assert_allclose(s.p, p)

    @unittest.skipUnless(native_capabilities()["openmp"], "OpenMP extension not built")
    def test_parallel_contexts_and_threads(self):
        q = self.rng.normal(size=(16, 5, 3))
        p = self.rng.normal(size=q.shape)

        def run(beta, threads):
            a = CayleyIntegrator(
                np.arange(1, 6), beta, 16, 0.04, native="required", threads=threads
            )
            x = q.copy()
            v = p.copy()
            a.free_step(x, v)
            return x, v

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, beta, 2) for beta in (2.0, 5.0)]
            got = [f.result() for f in futures]
        for beta, result in zip((2.0, 5.0), got):
            expected = run(beta, 1)
            for a, b in zip(expected, result):
                np.testing.assert_array_equal(a, b)

    def test_alias_and_bad_parameters(self):
        with self.assertRaises(ValueError):
            CayleyIntegrator([1], 1.0, 0, 0.1)
        a = CayleyIntegrator([1], 1.0, 2, 0.1, native=False)
        q = np.ones((2, 1, 3))
        with self.assertRaises(ValueError):
            a.free_step(q, q)

    def test_failed_step_is_transactional(self):
        class Fail:
            n = 0

            def evaluate(self, q):
                self.n += 1
                if self.n == 2:
                    raise PotentialError("outside domain")
                return PESResult(np.zeros(len(q)), np.ones_like(q))

        d = Dynamics(Fail(), [1], 2.0, 2, 0.1)
        s = State(np.zeros((2, 1, 3)), np.ones((2, 1, 3)))
        old = s.copy()
        with self.assertRaises(PotentialError):
            d.step(s)
        np.testing.assert_array_equal(s.q, old.q)
        np.testing.assert_array_equal(s.p, old.p)
        self.assertEqual(s.step, 0)


class ThermalTests(unittest.TestCase):
    def test_momentum_and_centroid_variances(self):
        d = Dynamics(FreePotential(), [2.0], 3.0, 8, 0.1, seed=432, native=False)
        ps = np.array([d.sample_momenta() for _ in range(20000)])
        self.assertAlmostEqual(float(ps.var()), 2 * 8 / 3, delta=0.05)
        centroid_velocity = ps.mean(axis=1)[:, 0, :] / 2
        self.assertAlmostEqual(float(centroid_velocity.var()), 1 / (3 * 2), delta=0.004)

    def test_fragment_centroids_only_and_segmented_parent(self):
        q = np.random.default_rng(4).normal(size=(8, 3, 3))
        p = np.zeros_like(q)
        d1 = Dynamics(HarmonicPotential(0.01), [1, 2, 3], 3.0, 8, 0.01, seed=5)
        d2 = Dynamics(HarmonicPotential(0.01), [1, 2, 3], 3.0, 8, 0.01, seed=5)
        a = State(q, p)
        b = a.copy()
        frags = [[0], [1, 2]]
        d1.thermalize(a, 40, interval=7, fragments=frags)
        d2.thermalize(b, 13, interval=7, fragments=frags)
        d2.thermalize(b, 27, interval=7, fragments=frags, resample=False)
        np.testing.assert_allclose(a.q, b.q, atol=3e-13)
        np.testing.assert_allclose(a.p, b.p, atol=3e-13)
        np.testing.assert_allclose(
            a.q[:, 0].mean(axis=0), q[:, 0].mean(axis=0), atol=1e-14
        )
        self.assertGreater(float(np.std(a.q[:, 0], axis=0).max()), 0.1)
        for f in frags:
            np.testing.assert_allclose(a.p[:, f].sum(axis=(0, 1)), 0.0, atol=2e-14)

    def test_fragment_constraints_at_every_force_evaluation(self):
        q = np.random.default_rng(9).normal(size=(4, 3, 3))
        m = np.array([1.0, 2.0, 3.0])
        ref = np.sum(q * m[None, :, None], axis=(0, 1)) / (4 * m.sum())

        class ConstrainedHarmonic(HarmonicPotential):
            def evaluate(self, x):
                com = np.sum(x * m[None, :, None], axis=(0, 1)) / (4 * m.sum())
                np.testing.assert_allclose(com, ref, atol=2e-14)
                return super().evaluate(x)

        dyn = Dynamics(ConstrainedHarmonic(2.0), m, 3.0, 4, 0.2, seed=71)
        state = State(q, np.zeros_like(q))
        dyn.thermalize(state, 30, interval=5, fragments=[[0, 1, 2]])

    def test_bcocb_harmonic_configuration(self):
        P = 4
        beta = 2.0
        k = 0.7
        m = 1.3
        d = Dynamics(HarmonicPotential(k), [m], beta, P, 0.15, seed=22, native=False)
        s = State(np.zeros((P, 1, 3)), np.zeros((P, 1, 3)))
        qs = []
        for i in range(15000):
            d.sample_step(s, gamma_centroid=1.0)
            if i >= 1000 and i % 5 == 0:
                qs.append(s.q.copy())
        modes = 2 * P / beta * np.sin(np.pi * np.arange(P) / P)
        expected = np.sum(1 / (beta * (k + m * modes * modes)))
        self.assertAlmostEqual(
            float(np.mean(np.asarray(qs) ** 2)), expected, delta=0.12 * expected
        )


if __name__ == "__main__":
    unittest.main()
