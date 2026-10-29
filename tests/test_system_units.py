import tempfile
import unittest
from pathlib import Path
import numpy as np
from pyrpmd.system import System, system_from_config
from pyrpmd.units import parse_quantity, unit_factor
from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU


class SystemTests(unittest.TestCase):
    def test_inline_geometry_units_are_explicit_and_dimension_checked(self):
        spec = {"symbols": ["H"], "masses": [1.0], "geometry": [[1.0, 2.0, 3.0]]}
        np.testing.assert_array_equal(
            system_from_config(spec).geometry, spec["geometry"]
        )
        np.testing.assert_array_equal(
            system_from_config(dict(spec, geometry_unit="bohr")).geometry,
            spec["geometry"],
        )
        np.testing.assert_allclose(
            system_from_config(dict(spec, geometry_unit="angstrom")).geometry,
            np.asarray(spec["geometry"]) * ANGSTROM_TO_AU,
        )
        for unit in ("fs", "not_a_unit", ""):
            with self.subTest(unit=unit):
                with self.assertRaises(ValueError):
                    system_from_config(dict(spec, geometry_unit=unit))

    def test_quantum_numbers_and_indices_are_integers(self):
        base = {"symbols": ["H"], "masses": [1.0], "geometry": [[0.0, 0.0, 0.0]]}
        for field, value in (
            ("charge", 0.5),
            ("charge", True),
            ("multiplicity", 1.5),
            ("multiplicity", True),
            ("multiplicity", 0),
            ("fragments", [[0.0]]),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    System(**dict(base, **{field: value}))
        for value in (0.5, True, -1):
            with self.subTest(isotope=value):
                with self.assertRaises(ValueError):
                    system_from_config(
                        dict(base, isotopes=[{"atom": value, "mass": "2 amu"}])
                    )
        system = System(**dict(base, charge=np.int64(-1), multiplicity=np.int32(2)))
        self.assertEqual(system.charge, -1)

    def test_isotope_mass_precedes_reactant_com_placement(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.xyz"
            b = Path(td) / "b.xyz"
            a.write_text("2\nH2\nH 0 0 0\nH 0 0 2\n")
            b.write_text("2\nH2\nH 0 0 1\nH 0 0 3\n")
            system = system_from_config(
                {
                    "reactants": [str(a), str(b)],
                    "separation": "10 angstrom",
                    "isotopes": [
                        {"atom": 0, "mass": "3 amu"},
                        {"atom": 3, "mass": "2 amu"},
                    ],
                }
            )
            centers = [
                np.average(
                    system.geometry[list(f)], axis=0, weights=system.masses[list(f)]
                )
                for f in system.fragments
            ]
            np.testing.assert_allclose(centers[0], 0.0, atol=2e-15)
            np.testing.assert_allclose(
                centers[1], [0.0, 0.0, 10 * ANGSTROM_TO_AU], atol=2e-15
            )
            np.testing.assert_allclose(
                system.masses[[0, 3]], [3 * AMU_TO_AU, 2 * AMU_TO_AU]
            )
            self.assertAlmostEqual(
                np.linalg.norm(system.geometry[1] - system.geometry[0]),
                2 * ANGSTROM_TO_AU,
            )

    def test_explicit_mass_for_unknown_default_and_array_isotope(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "xe.xyz"
            path.write_text("1\nXe\nXe 1 2 3\n")
            system = system_from_config(
                {"geometry": str(path), "isotopes": [{"atom": 0, "mass": "132 amu"}]}
            )
            np.testing.assert_allclose(system.masses, [132 * AMU_TO_AU])
        system = system_from_config(
            {
                "symbols": ["H", "H"],
                "geometry": [[0, 0, 0], [1, 0, 0]],
                "masses_amu": [1.0, 1.0],
                "isotopes": [{"atom": 1, "mass": "2 amu"}],
            }
        )
        np.testing.assert_allclose(system.masses, [AMU_TO_AU, 2 * AMU_TO_AU])
        duplicate = {
            "symbols": ["H"],
            "geometry": [[0, 0, 0]],
            "masses": [1.0],
            "masses_amu": [1.0],
        }
        with self.assertRaises(ValueError):
            system_from_config(duplicate)


class UnitsTests(unittest.TestCase):
    def test_compound_units_and_dimension_rejection(self):
        self.assertAlmostEqual(
            parse_quantity("2 hartree/angstrom^2", "spring"), 2 / ANGSTROM_TO_AU**2
        )
        self.assertAlmostEqual(
            unit_factor("1/fs", "frequency"), 1 / parse_quantity("1 fs", "time")
        )
        for value in (True, (True, "bohr"), ("2", "bohr"), "1 fs"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_quantity(value, "length")


if __name__ == "__main__":
    unittest.main()
