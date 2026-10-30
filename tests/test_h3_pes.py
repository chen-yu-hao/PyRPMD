"""Optional real chemical PES checks; the H3 CI job builds the PES explicitly."""

from pathlib import Path
import importlib.util
import unittest

import numpy as np
from pyrpmd.potentials import PotentialError, make_potential, validate_result
from pyrpmd.potentials.python import import_object

ROOT = Path(__file__).resolve().parents[1]
FACTORY = str(ROOT / "benchmarks/h3/pes.py")
LIBRARY = ROOT / "benchmarks/h3/build/libbkmp2.so"


@unittest.skipUnless(LIBRARY.is_file(), "optional real BKMP2 PES has not been built")
class RealBKMP2Tests(unittest.TestCase):
    def test_module_aliases_share_native_and_python_library_lock(self):
        from benchmarks.h3.pes import BKMP2Potential

        spec = importlib.util.spec_from_file_location("pyrpmd_bkmp2_lock_alias", FACTORY)
        alias = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alias)
        original = BKMP2Potential(LIBRARY)
        separately_imported = alias.BKMP2Potential(LIBRARY)
        self.assertIs(original._lock, separately_imported._lock)
        self.assertIs(original.native_binding().lock, separately_imported.native_binding().lock)

    def test_author_oracles_geometry_and_analytic_forces(self):
        report = import_object(FACTORY + ":verify")(LIBRARY)
        self.assertTrue(report["passed"])
        self.assertEqual(report["ts_negative_hessian_modes"], 1)
        self.assertLess(abs(report["barrier_kcal_mol"] - 9.6133013), 1e-5)

    def test_mixed_batch_failure_is_reported_without_fortran_stop(self):
        potential = make_potential(
            {
                "provider": "python",
                "factory": FACTORY + ":build",
                "params": {"library": str(LIBRARY)},
            }
        )
        q = np.zeros((2, 3, 3))
        q[0, :, 0] = [-1.757, 0.0, 1.757]
        q[1, :, 0] = [-0.1, 0.0, 0.1]
        result = potential.evaluate(q)
        np.testing.assert_array_equal(result.status, [0, 1])
        self.assertTrue(np.isfinite(result.energy[0]))
        self.assertTrue(np.isfinite(result.forces[0]).all())
        with self.assertRaises(PotentialError):
            validate_result(result, q.shape)


if __name__ == "__main__":
    unittest.main()
