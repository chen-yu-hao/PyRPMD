"""Validated molecular identity and XYZ parsing; coordinates/masses in atomic units."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .constants import AMU_TO_AU, ANGSTROM_TO_AU
from .units import parse_quantity
from .geometry import validate_cell_pbc

_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_Z = {s: i for i, s in enumerate(_SYMBOLS)}
# Dominant stable-isotope masses, u. Explicit isotope/mass input overrides these.
_MASSES = {
    "H": 1.00782503223,
    "D": 2.01410177812,
    "T": 3.01604928199,
    "He": 4.00260325413,
    "Li": 7.0160034366,
    "Be": 9.012183065,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Ne": 19.9924401762,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Ar": 39.9623831237,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Ni": 57.93534241,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    "Br": 78.9183376,
    "I": 126.9044719,
}


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass
class System:
    symbols: tuple[str, ...]
    masses: np.ndarray
    geometry: np.ndarray
    fragments: tuple[tuple[int, ...], ...] = ()
    charge: int = 0
    multiplicity: int = 1
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    fixed_atoms: tuple[int, ...] = ()

    def __post_init__(self):
        self.symbols = tuple(self.symbols)
        if np.iscomplexobj(self.masses) or np.iscomplexobj(self.geometry):
            raise ValueError("System masses and coordinates must be real")
        self.masses = np.array(self.masses, dtype=float, order="C", copy=True)
        self.geometry = np.array(self.geometry, dtype=float, order="C", copy=True)
        self.fragments = tuple(
            tuple(_integer(i, "Fragment atom index", 0) for i in f)
            for f in self.fragments
        )
        self.charge = _integer(self.charge, "Charge")
        self.multiplicity = _integer(self.multiplicity, "Multiplicity", 1)
        self.cell, self.pbc = validate_cell_pbc(self.cell, self.pbc)
        self.pbc = tuple(bool(flag) for flag in self.pbc)
        self.fixed_atoms = tuple(_integer(i, "Fixed atom index", 0) for i in self.fixed_atoms)
        n = len(self.symbols)
        if n < 1 or self.masses.shape != (n,) or self.geometry.shape != (n, 3):
            raise ValueError("System needs N symbols, masses[N], geometry[N,3]")
        if (
            not np.all(np.isfinite(self.geometry))
            or not np.all(np.isfinite(self.masses))
            or np.any(self.masses <= 0)
        ):
            raise ValueError("System coordinates must be finite and masses positive")
        if any(s not in _Z and s not in ("D", "T") for s in self.symbols):
            raise ValueError("Unknown element symbol")
        if len(set(self.fixed_atoms)) != len(self.fixed_atoms) or any(
            i >= n for i in self.fixed_atoms
        ):
            raise ValueError("fixed_atoms must contain distinct in-range 0-based indices")
        if self.fragments:
            all_ids = [i for f in self.fragments for i in f]
            if any(not f for f in self.fragments) or sorted(all_ids) != list(range(n)):
                raise ValueError(
                    "Fragments must form a disjoint, complete 0-based atom partition"
                )

    @property
    def natoms(self):
        return len(self.symbols)

    @property
    def atomic_numbers(self):
        return np.array(
            [1 if s in ("D", "T") else _Z[s] for s in self.symbols], dtype=np.int32
        )

    @property
    def atom_ids(self):
        return tuple(f"{s}{i}" for i, s in enumerate(self.symbols))

    @property
    def mobile_mask(self):
        mask = np.ones(self.natoms, dtype=bool)
        mask[list(self.fixed_atoms)] = False
        return mask

    def to_dict(self):
        result = {
            "symbols": list(self.symbols),
            "masses": self.masses.tolist(),
            "geometry": self.geometry.tolist(),
            "fragments": [list(f) for f in self.fragments],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }
        # Preserve the exact pre-materials molecular checkpoint identity.
        if self.cell is not None:
            result["cell"] = self.cell.tolist()
            if any(self.pbc) or not getattr(self, "_omit_nonperiodic_pbc", False):
                result["pbc"] = list(self.pbc)
        if self.fixed_atoms:
            result["fixed_atoms"] = list(self.fixed_atoms)
        return result

    @classmethod
    def from_dict(cls, d):
        result = cls(
            tuple(d["symbols"]),
            np.array(d["masses"]),
            np.array(d["geometry"]),
            tuple(tuple(f) for f in d.get("fragments", [])),
            d.get("charge", 0),
            d.get("multiplicity", 1),
            d.get("cell"),
            d.get("pbc", (False, False, False)),
            d.get("fixed_atoms", ()),
        )
        # Both precursor branches used the same manifest version, but the
        # surface branch omitted explicit false flags. Keep that byte-level
        # representation for saved fingerprints without changing boundaries.
        if result.cell is not None and "pbc" not in d:
            result._omit_nonperiodic_pbc = True
        return result


def _xyz_data(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        n = int(lines[0].strip())
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid XYZ header: {path}") from e
    if n < 1 or len(lines) < n + 2:
        raise ValueError(f"Incomplete XYZ: {path}")
    symbols, xyz = [], []
    for i, line in enumerate(lines[2 : n + 2]):
        cells = line.split()
        if len(cells) < 4:
            raise ValueError(
                f"XYZ atom line {i+3} must contain symbol and three coordinates"
            )
        s = cells[0]
        if s not in _Z and s not in ("D", "T"):
            raise ValueError(f"Unknown XYZ element {s!r}")
        symbols.append(s)
        xyz.append([float(x) for x in cells[1:4]])
    return tuple(symbols), np.asarray(xyz, dtype=float)


def _masses_for_symbols(symbols, masses=None, isotopes=()):
    if masses is None:
        overrides = [None] * len(symbols)
    else:
        if len(masses) != len(symbols):
            raise ValueError("Provide exactly one mass for each atom")
        overrides = list(masses)
    seen = set()
    for isotope in isotopes:
        i = _integer(isotope["atom"], "Isotope atom index", 0)
        if i >= len(symbols):
            raise ValueError("Isotope atom index out of range")
        if i in seen:
            raise ValueError("Repeated isotope override for an atom")
        seen.add(i)
        overrides[i] = isotope["mass"]
    result = []
    for symbol, value in zip(symbols, overrides):
        if value is None:
            if symbol not in _MASSES:
                raise ValueError(
                    f"Provide explicit isotope masses for {symbol}; no implicit approximate mass"
                )
            value = _MASSES[symbol] * AMU_TO_AU
        result.append(parse_quantity(value, "mass"))
    return np.asarray(result, dtype=float)


def read_xyz(path, *, unit="angstrom", masses=None):
    symbols, xyz = _xyz_data(path)
    factor = parse_quantity(f"1 {unit}", "length")
    return System(symbols, _masses_for_symbols(symbols, masses), xyz * factor)


def system_from_config(spec):
    if isinstance(spec, System):
        return spec
    masses = spec.get("masses")
    if "masses_amu" in spec:
        if masses is not None:
            raise ValueError("Specify masses or masses_amu, not both")
        values = np.asarray(spec["masses_amu"])
        if values.ndim != 1 or values.dtype.kind not in "iuf":
            raise ValueError("masses_amu must be a list of real numbers")
        masses = values * AMU_TO_AU
    isotopes = spec.get("isotopes", ())
    charge = spec.get("charge", 0)
    multiplicity = spec.get("multiplicity", 1)
    inline = "symbols" in spec and "geometry" in spec and not isinstance(spec["geometry"], str)
    cell = spec.get("cell")
    if cell is not None:
        raw_cell = np.asarray(cell)
        if raw_cell.dtype.kind not in "iuf":
            raise ValueError("cell must contain real numerical values")
        cell_unit = spec.get("cell_unit", spec.get("geometry_unit", "bohr" if inline else "angstrom"))
        cell = np.asarray(cell, dtype=float) * parse_quantity(f"1 {cell_unit}", "length")
    elif "cell_unit" in spec:
        raise ValueError("cell_unit requires a cell")
    cell, pbc = validate_cell_pbc(cell, spec.get("pbc"))
    boundaries = {"cell": cell, "pbc": pbc, "fixed_atoms": spec.get("fixed_atoms", ())}
    if (
        "symbols" in spec
        and "geometry" in spec
        and not isinstance(spec["geometry"], str)
    ):
        symbols = tuple(spec["symbols"])
        # Numerical Python geometry is canonical AU unless explicitly declared.
        factor = parse_quantity(f"1 {spec.get('geometry_unit','bohr')}", "length")
        if np.iscomplexobj(spec["geometry"]):
            raise ValueError("System coordinates must be real")
        geometry = np.asarray(spec["geometry"], dtype=float) * factor
        return System(
            symbols,
            _masses_for_symbols(symbols, masses, isotopes),
            geometry,
            spec.get("fragments", ()),
            charge,
            multiplicity,
            **boundaries,
        )
    factor = parse_quantity(f"1 {spec.get('geometry_unit','angstrom')}", "length")
    if "reactants" in spec:
        if len(spec["reactants"]) != 2:
            raise ValueError("dRPMD needs exactly two reactant structures")
        (sa, qa), (sb, qb) = [_xyz_data(p) for p in spec["reactants"]]
        symbols = sa + sb
        mass = _masses_for_symbols(symbols, masses, isotopes)
        r = spec.get("separation_au")
        if r is None:
            r = parse_quantity(spec["separation"], "length")
        if not np.isfinite(r) or r <= 0:
            raise ValueError("Reactant separation must be positive and finite")
        # Isotopes affect fragment COMs, so resolve masses before placement.
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("System masses must be positive and finite")
        qa = qa * factor
        qb = qb * factor
        qa -= np.average(qa, axis=0, weights=mass[: len(sa)])
        qb -= np.average(qb, axis=0, weights=mass[len(sa) :])
        qb[:, 2] += r
        fragments = (tuple(range(len(sa))), tuple(range(len(sa), len(symbols))))
        if (
            "fragments" in spec
            and tuple(
                tuple(_integer(i, "Fragment atom index", 0) for i in f)
                for f in spec["fragments"]
            )
            != fragments
        ):
            raise ValueError(
                "Reactant files define their atom partition; explicit fragments disagree"
            )
        return System(
            symbols, mass, np.vstack([qa, qb]), fragments, charge, multiplicity,
            **boundaries,
        )
    else:
        p = spec.get("transition_state", spec.get("geometry"))
        if not p:
            raise ValueError(
                "Specify system.geometry, transition_state or two reactants"
            )
        symbols, xyz = _xyz_data(p)
        return System(
            symbols,
            _masses_for_symbols(symbols, masses, isotopes),
            xyz * factor,
            spec.get("fragments", ()),
            charge,
            multiplicity,
            **boundaries,
        )
