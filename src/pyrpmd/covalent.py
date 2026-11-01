"""Covalent radii for heuristic molecular connectivity, in angstrom.

The H--Cm values are from Cordero et al., *Dalton Trans.* (2008),
2832--2838, Table 2, https://doi.org/10.1039/B801115J. Carbon uses the
sp3 value; Mn, Fe and Co use the low-spin values. These are the same
choices documented by ASE's ``ase.data.covalent_radii`` table.

The source does not provide radii beyond Cm. Those elements require
explicit overrides rather than an arbitrary default. Connectivity remains
a distance-based heuristic, and overrides also allow different chemical
environments or spin states to be represented.
"""

from __future__ import annotations

from collections.abc import Mapping
import numbers
from types import MappingProxyType

import numpy as np

from .constants import BOHR_ANGSTROM
from .units import parse_quantity


COVALENT_RADII_ANGSTROM = MappingProxyType({
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.40,
    "At": 1.50,
    "Rn": 1.50,
    "Fr": 2.60,
    "Ra": 2.21,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2.00,
    "U": 1.96,
    "Np": 1.90,
    "Pu": 1.87,
    "Am": 1.80,
    "Cm": 1.69,
})

_ISOTOPES = {"D": "H", "T": "H"}
_ELEMENTS = set(COVALENT_RADII_ANGSTROM) | set(
    "Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
) | set(_ISOTOPES)


def covalent_radii(symbols, overrides=None):
    """Return one covalent radius in angstrom for each element symbol.

    ``overrides`` maps symbols to positive lengths. Bare numbers are in
    angstrom, matching the table; strings such as ``"1.7 angstrom"`` or
    ``"3.2 bohr"`` carry explicit units. All supplied overrides are checked,
    including those for elements absent from ``symbols``. D and T inherit
    H's radius (including an H override) unless overridden individually.

    Radii for elements after Cm must be supplied explicitly. Unknown symbols
    and missing data are errors; neither receives a guessed radius.
    """
    if isinstance(symbols, (str, bytes)):
        raise ValueError("symbols must be a sequence of element symbols")
    try:
        symbols = tuple(symbols)
    except TypeError as error:
        raise ValueError("symbols must be a sequence of element symbols") from error
    if any(not isinstance(symbol, str) for symbol in symbols):
        raise ValueError("symbols must contain element-symbol strings")
    unknown = set(symbols) - _ELEMENTS
    if unknown:
        raise ValueError(
            f"unknown element symbols for covalent radii: {sorted(unknown)}"
        )
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError(
            "covalent_radii overrides must be a mapping of symbols to lengths"
        )

    radii = dict(COVALENT_RADII_ANGSTROM)
    checked = {}
    for symbol, value in overrides.items():
        if not isinstance(symbol, str) or symbol not in _ELEMENTS:
            raise ValueError(f"unknown element in covalent_radii overrides: {symbol!r}")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"covalent radius for {symbol} must be a positive finite length"
            )
        try:
            radius = (
                float(value)
                if isinstance(value, numbers.Real)
                else parse_quantity(value, "length") * BOHR_ANGSTROM
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"covalent radius for {symbol} must be a positive finite length"
            ) from error
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"covalent radius for {symbol} must be a positive finite length"
            )
        checked[symbol] = radius
    radii.update(checked)
    for isotope, element in _ISOTOPES.items():
        radii[isotope] = checked.get(isotope, radii[element])

    missing = set(symbols) - radii.keys()
    if missing:
        raise ValueError(
            f"automatic connectivity has no radii for {sorted(missing)}; "
            "supply reaction.preparation.covalent_radii overrides "
            "(numbers in angstrom or explicit length units), "
            "or supply reaction explicitly"
        )
    return np.array([radii[symbol] for symbol in symbols], dtype=float)
