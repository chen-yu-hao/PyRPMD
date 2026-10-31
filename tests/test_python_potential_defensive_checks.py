"""Python PES validation should not repeat work for unchanged atomic units."""

import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.potentials import PESResult, PotentialError
from pyrpmd.potentials import python as python_potential
from pyrpmd.potentials.python import PythonPotential


class PythonPotentialValidationTests(unittest.TestCase):
    def test_atomic_units_validate_once_and_preserve_owned_output_buffers(self):
        energy = np.zeros(2)
        forces = np.zeros((2, 1, 3))
        virial = np.zeros((2, 3, 3))

        def provider(q):
            energy[:] = np.sum(q * q, axis=(1, 2))
            forces[:] = -2 * q
            virial[:] = energy[:, None, None]
            return PESResult(energy, forces, virial=virial)

        q = np.ones((2, 1, 3))
        for units in (None, {"positions": "bohr", "energy": "hartree"}):
            with self.subTest(units=units):
                potential = PythonPotential(provider, units=units)
                with patch.object(
                    python_potential, "validate_result",
                    wraps=python_potential.validate_result,
                ) as validate:
                    result = potential.evaluate(q)
                self.assertEqual(validate.call_count, 1)
                for name, source in (("energy", energy), ("forces", forces), ("virial", virial)):
                    value = getattr(result, name)
                    self.assertFalse(np.shares_memory(value, source))
                    self.assertTrue(value.flags.c_contiguous)
                    self.assertTrue(value.flags.writeable)
                potential.evaluate(3 * q)
                np.testing.assert_array_equal(result.energy, np.full(2, 3.0))
                np.testing.assert_array_equal(result.forces, -2 * q)
                np.testing.assert_array_equal(result.virial, np.full((2, 3, 3), 3.0))
                np.testing.assert_array_equal(energy, np.full(2, 27.0))

    def test_atomic_units_still_reject_malformed_and_nonfinite_provider_results(self):
        q = np.zeros((2, 1, 3))
        results = (
            (np.zeros(1), np.zeros_like(q)),
            (np.zeros(2), np.zeros((2, 2, 3))),
            (np.zeros(2), np.full_like(q, np.inf)),
            PESResult(np.zeros(2), np.zeros_like(q), virial=np.zeros((2, 3))),
        )
        for result in results:
            with self.subTest(result_type=type(result).__name__):
                potential = PythonPotential(lambda unused, result=result: result)
                with self.assertRaises(PotentialError):
                    potential.evaluate(q)

    def test_atomic_units_preserve_failed_frame_status_and_messages(self):
        failed = PESResult(
            [np.nan], np.full((1, 1, 3), np.nan), [7], ["outside model domain"],
            np.full((1, 3, 3), np.nan),
        )
        potential = PythonPotential(lambda q: failed)
        result = potential.evaluate(np.zeros((1, 1, 3)))
        np.testing.assert_array_equal(result.status, [7])
        self.assertEqual(result.messages, ("outside model domain",))
        self.assertTrue(np.isnan(result.energy).all())
        self.assertTrue(np.isnan(result.forces).all())
        self.assertTrue(np.isnan(result.virial).all())

    def test_nonidentity_conversion_keeps_output_overflow_validation(self):
        maximum = np.finfo(float).max
        q = np.zeros((1, 1, 3))
        # Exercise a conversion which increases magnitudes. Current named
        # units scale down, but converted values still need their own check.
        with patch.object(python_potential, "unit_factors", return_value=(1.0, 2.0)):
            potential = PythonPotential(lambda q: (np.full(len(q), maximum), np.zeros_like(q)))
        with patch.object(
            python_potential, "validate_result", wraps=python_potential.validate_result,
        ) as validate, np.errstate(over="ignore"):
            with self.assertRaisesRegex(PotentialError, "nonfinite"):
                potential.evaluate(q)
        self.assertEqual(validate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
