"""Compare PyRPMD's complete H--Cm table with ASE's Cordero dataset.

ASE is a validation-only dependency. This script does not change core imports.
Run with ``PYTHONPATH=src python benchmarks/transition_state_search/covalent_radii.py
--output covalent-radii.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import ase
import ase.data
import numpy as np

from pyrpmd.constants import BOHR_ANGSTROM
from pyrpmd.covalent import COVALENT_RADII_ANGSTROM, covalent_radii


def run_comparison():
    symbols = ase.data.chemical_symbols[1:97]
    actual = covalent_radii(symbols)
    reference = ase.data.covalent_radii[1:97]
    differences = [
        {"symbol": symbol, "pyrpmd_angstrom": float(ours), "ase_angstrom": float(theirs),
         "difference_angstrom": float(ours - theirs)}
        for symbol, ours, theirs in zip(symbols, actual, reference)
        if ours != theirs
    ]
    isotope_cases = []
    for overrides, expected in (
        (None, [0.31, 0.31, 0.31]),
        ({"H": 0.4}, [0.4, 0.4, 0.4]),
        ({"H": 0.4, "D": 0.35}, [0.4, 0.35, 0.4]),
        ({"H": 0.4, "T": 0.35}, [0.4, 0.4, 0.35]),
    ):
        measured = covalent_radii(["H", "D", "T"], overrides)
        isotope_cases.append({
            "symbols": ["H", "D", "T"], "overrides": overrides,
            "actual_angstrom": measured.tolist(), "expected_angstrom": expected,
            "passed": bool(np.array_equal(measured, expected)),
        })
    heavy_cases = []
    for symbol in ase.data.chemical_symbols[97:119]:
        rejection = None
        try:
            covalent_radii([symbol])
        except ValueError as error:
            rejection = str(error)
        supplied = float(covalent_radii([symbol], {symbol: 1.7})[0])
        heavy_cases.append({
            "symbol": symbol, "without_override_error": rejection,
            "with_explicit_1_7_angstrom_override": supplied,
            "passed": bool(rejection and "preparation.covalent_radii" in rejection and supplied == 1.7),
        })
    unknown_cases = []
    for symbol in ("X", "Ff", "Unobtainium"):
        error_message = None
        try:
            covalent_radii([symbol], {symbol: 1.7})
        except ValueError as error:
            error_message = str(error)
        unknown_cases.append({"symbol": symbol, "error": error_message,
                              "passed": bool(error_message and "unknown element" in error_message)})
    og_bohr_override = float(covalent_radii(["Og"], {"Og": "2 bohr"})[0])
    checks = {
        "exactly_H_through_Cm": set(COVALENT_RADII_ANGSTROM) == set(symbols),
        "all_96_match_ase_exactly": not differences,
        "all_isotope_alias_and_override_cases": all(case["passed"] for case in isotope_cases),
        "all_22_heavy_elements_require_and_accept_explicit_overrides": all(case["passed"] for case in heavy_cases),
        "unknown_symbols_stay_rejected_even_with_overrides": all(case["passed"] for case in unknown_cases),
        "explicit_bohr_override_converted_to_angstrom": bool(np.isclose(og_bohr_override, 2 * BOHR_ANGSTROM, rtol=0, atol=1e-14)),
    }
    source_path = Path(ase.data.__file__)
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema": "pyrpmd.covalent_radii.validation.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "paper": "Cordero et al., Covalent radii revisited, Dalton Trans. (2008), 2832-2838, Table 2",
            "doi": "10.1039/B801115J", "url": "https://doi.org/10.1039/B801115J",
            "ase_version": ase.__version__, "ase_source_path": str(source_path),
            "ase_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "scope": "Full software-table cross-check against ASE's Cordero dataset; no independent experimental fit or digitization of the paper",
            "conventions": "PyRPMD documents C(sp3) and Mn/Fe/Co low-spin choices; all match ASE numerically",
            "missing_value_note": "ASE's entries after Cm are placeholders, excluded from the 96-value comparison; PyRPMD requires explicit values",
        },
        "code_sha256": {
            name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
            for name in ("benchmarks/transition_state_search/covalent_radii.py", "src/pyrpmd/covalent.py")
        },
        "table_comparison": {
            "count": len(symbols), "symbols": symbols,
            "matched": len(symbols) - len(differences), "differences": differences,
            "max_absolute_difference_angstrom": float(np.max(np.abs(actual - reference))),
        },
        "isotope_cases": isotope_cases, "heavy_element_cases": heavy_cases,
        "unknown_symbol_cases": unknown_cases,
        "Og_2_bohr_override_angstrom": og_bohr_override,
        "checks": checks, "core_checks_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_comparison()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"matched": result["table_comparison"]["matched"],
                      "core_checks_passed": result["core_checks_passed"]}))
    raise SystemExit(0 if result["core_checks_passed"] else 1)


if __name__ == "__main__":
    main()
