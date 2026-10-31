"""Small explicit unit grammar; no eval and no implicit physical conversions."""

from __future__ import annotations

import math
import numbers
import re
from .constants import (
    ANGSTROM_TO_AU,
    FS_TO_AU,
    AMU_TO_AU,
    KCAL_MOL_TO_AU,
    EV_TO_AU,
    WAVENUMBER_TO_AU,
    HARTREE_J,
    ATOMIC_TIME_S,
    ELECTRON_MASS_KG,
    BOHR_M,
)

# Dimension order: length, energy, time, mass, temperature, angle.
ZERO = (0, 0, 0, 0, 0, 0)
DIMS = {
    "dimensionless": ZERO,
    "length": (1, 0, 0, 0, 0, 0),
    "energy": (0, 1, 0, 0, 0, 0),
    "time": (0, 0, 1, 0, 0, 0),
    "mass": (0, 0, 0, 1, 0, 0),
    "temperature": (0, 0, 0, 0, 1, 0),
    "angle": (0, 0, 0, 0, 0, 1),
    "force": (-1, 1, 0, 0, 0, 0),
    "spring": (-2, 1, 0, 0, 0, 0),
    "frequency": (0, 0, -1, 0, 0, 0),
    "pressure": (-3, 1, 0, 0, 0, 0),
}
_U = {
    "1": (1.0, ZERO),
    "bohr": (1.0, DIMS["length"]),
    "a0": (1.0, DIMS["length"]),
    "angstrom": (ANGSTROM_TO_AU, DIMS["length"]),
    "nm": (10 * ANGSTROM_TO_AU, DIMS["length"]),
    "m": (1e10 * ANGSTROM_TO_AU, DIMS["length"]),
    "hartree": (1.0, DIMS["energy"]),
    "ha": (1.0, DIMS["energy"]),
    "ev": (EV_TO_AU, DIMS["energy"]),
    "kcalmol": (KCAL_MOL_TO_AU, DIMS["energy"]),
    "kjmol": (KCAL_MOL_TO_AU / 4.184, DIMS["energy"]),
    "j": (1 / HARTREE_J, DIMS["energy"]),
    "wavenumber": (WAVENUMBER_TO_AU, DIMS["energy"]),
    "autime": (1.0, DIMS["time"]),
    "fs": (FS_TO_AU, DIMS["time"]),
    "ps": (FS_TO_AU * 1000, DIMS["time"]),
    "ns": (FS_TO_AU * 1e6, DIMS["time"]),
    "s": (1 / ATOMIC_TIME_S, DIMS["time"]),
    "amu": (AMU_TO_AU, DIMS["mass"]),
    "u": (AMU_TO_AU, DIMS["mass"]),
    "me": (1.0, DIMS["mass"]),
    "kg": (1 / ELECTRON_MASS_KG, DIMS["mass"]),
    "k": (1.0, DIMS["temperature"]),
    "rad": (1.0, DIMS["angle"]),
    "radian": (1.0, DIMS["angle"]),
    "deg": (math.pi / 180.0, DIMS["angle"]),
    "pa": (BOHR_M**3 / HARTREE_J, DIMS["pressure"]),
    "kpa": (1e3 * BOHR_M**3 / HARTREE_J, DIMS["pressure"]),
    "mpa": (1e6 * BOHR_M**3 / HARTREE_J, DIMS["pressure"]),
    "gpa": (1e9 * BOHR_M**3 / HARTREE_J, DIMS["pressure"]),
    "bar": (1e5 * BOHR_M**3 / HARTREE_J, DIMS["pressure"]),
    "atm": (101325 * BOHR_M**3 / HARTREE_J, DIMS["pressure"]),
}
_NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")
_TOKEN = re.compile(r"([a-z][a-z0-9]*|1)(?:\^([+-]?\d+))?$")


def unit_factor(unit: str, dimension=None) -> float:
    """Convert one declared unit to atomic units, checking dimensions if given."""
    unit = unit.strip().lower().replace("å", "angstrom").replace("Å", "angstrom")
    unit = (
        unit.replace("**", "^")
        .replace("kcal/mol", "kcalmol")
        .replace("kj/mol", "kjmol")
    )
    unit = unit.replace("cm^-1", "wavenumber").replace("cm-1", "wavenumber")
    unit = unit.replace("a.u.", "au").replace(" ", "")
    if unit in ("au", "atomic", "atomic_units"):
        return 1.0
    if not unit:
        if dimension not in (None, "dimensionless"):
            raise ValueError(f"A unit is required for {dimension}")
        return 1.0
    parts = re.split(r"([*/])", unit)
    factor, dim, sign = 1.0, list(ZERO), 1
    for part in parts:
        if part == "*":
            sign = 1
            continue
        if part == "/":
            sign = -1
            continue
        m = _TOKEN.fullmatch(part)
        if not m or m[1] not in _U:
            raise ValueError(f"Unknown or invalid unit {unit!r}")
        exponent = int(m[2] or "1") * sign
        f, d = _U[m[1]]
        factor *= f**exponent
        dim = [a + exponent * b for a, b in zip(dim, d)]
        sign = 1
    if dimension is not None:
        expected = DIMS.get(
            dimension, dimension if isinstance(dimension, tuple) else None
        )
        if expected is None:
            raise ValueError(f"Unknown dimension {dimension!r}")
        if tuple(dim) != expected:
            raise ValueError(f"Unit {unit!r} has wrong dimension for {dimension}")
    return factor


def parse_quantity(value, dimension=None) -> float:
    """Parse '0.1 fs'. Numeric values are explicitly canonical atomic values.

    For temperature the canonical unit is kelvin. InputLoader can require a
    string for user-facing quantities; numerical APIs may pass atomic values.
    """
    if isinstance(value, bool):
        raise ValueError("Boolean is not a physical quantity")
    if isinstance(value, numbers.Real):
        result = float(value)
    elif isinstance(value, str):
        m = _NUMBER.fullmatch(value)
        if not m:
            raise ValueError(f"Invalid quantity {value!r}")
        result = float(m[1]) * unit_factor(m[2], dimension)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if isinstance(value[0], bool) or not isinstance(value[0], numbers.Real):
            raise ValueError("Quantity value must be a real number")
        result = float(value[0]) * unit_factor(str(value[1]), dimension)
    else:
        raise ValueError(f"Expected number or quantity string, got {value!r}")
    if not math.isfinite(result):
        raise ValueError("Physical quantities must be finite")
    return result


def time_steps(value, dt, *, allow_zero=False) -> int:
    t = parse_quantity(value, "time")
    if dt <= 0:
        raise ValueError("Time step must be positive")
    n = int(round(t / dt))
    if not math.isclose(t / dt, n, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(
            f"Duration {value!r} is not an integer multiple of the time step"
        )
    if n < (0 if allow_zero else 1):
        raise ValueError(
            "Duration must span at least one step"
            if not allow_zero
            else "Negative duration"
        )
    return n
