import unittest

import numpy as np

from pyrpmd.constants import BOHR_ANGSTROM
from pyrpmd.covalent import COVALENT_RADII_ANGSTROM, covalent_radii
from pyrpmd.system import _SYMBOLS


class CovalentRadiiTests(unittest.TestCase):
    def test_cordero_table_covers_hydrogen_through_curium(self):
        self.assertEqual(set(COVALENT_RADII_ANGSTROM), set(_SYMBOLS[1:97]))
        self.assertTrue(np.all(covalent_radii(_SYMBOLS[1:97]) > 0))
        # Transition metals and heavy elements absent from the old table.
        np.testing.assert_allclose(
            covalent_radii(["Ti", "Fe", "Ni", "Cu", "Xe", "Au", "U", "Cm"]),
            [1.60, 1.32, 1.24, 1.32, 1.40, 1.36, 1.96, 1.69],
        )

    def test_hydrogen_isotopes_share_radius_and_hydrogen_override(self):
        np.testing.assert_array_equal(covalent_radii(["H", "D", "T"]), [0.31] * 3)
        np.testing.assert_array_equal(
            covalent_radii(["H", "D", "T"], {"H": 0.4, "T": 0.35}),
            [0.4, 0.4, 0.35],
        )

    def test_heavy_elements_need_explicit_radii_without_global_fallback(self):
        for symbol in _SYMBOLS[97:]:
            with self.subTest(symbol=symbol):
                with self.assertRaisesRegex(ValueError, "preparation.covalent_radii"):
                    covalent_radii([symbol])
        np.testing.assert_allclose(
            covalent_radii(["Bk", "Og"], {"Bk": 1.7, "Og": 1.9}), [1.7, 1.9]
        )

    def test_numeric_and_unit_overrides_are_consistent_and_do_not_mutate(self):
        overrides = {"Fe": "1.52 angstrom", "Og": [2.0, "bohr"]}
        actual = covalent_radii(["Fe", "Og", "H"], overrides)
        np.testing.assert_allclose(actual, [1.52, 2 * BOHR_ANGSTROM, 0.31])
        self.assertEqual(overrides, {"Fe": "1.52 angstrom", "Og": [2.0, "bohr"]})
        actual[0] = 99.0
        self.assertEqual(covalent_radii(["Fe"])[0], 1.32)

    def test_invalid_overrides_are_rejected_even_if_unused(self):
        invalid = (
            True, np.bool_(True), 0, -1, np.nan, np.inf, 1j,
            "1.2", "1 hartree", None, [],
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive finite length"):
                    covalent_radii(["H"], {"Fe": value})
        for overrides in ([], 3, "Fe=1.32"):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    covalent_radii(["H"], overrides)
        with self.assertRaisesRegex(ValueError, "unknown element"):
            covalent_radii(["H"], {"Ff": 1.0})

    def test_unknown_and_malformed_symbols_are_rejected(self):
        for symbols in (["X"], ["fe"], ["Hydrogen"], ["Unobtainium"]):
            with self.subTest(symbols=symbols):
                with self.assertRaisesRegex(ValueError, "unknown element"):
                    covalent_radii(symbols)
        for symbols in ("Fe", None, [1], [["H"]]):
            with self.subTest(symbols=symbols):
                with self.assertRaisesRegex(ValueError, "symbols"):
                    covalent_radii(symbols)


if __name__ == "__main__":
    unittest.main()
