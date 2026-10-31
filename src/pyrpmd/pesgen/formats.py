"""Explicit PySCF-to-Caracal interchange formats; no fictitious QM outputs."""
from pathlib import Path
import re
import numpy as np
from ..constants import BOHR_ANGSTROM
from .reference import ReferencePoint, _array


def read_xyz(path):
    """Strict, ordered multi-frame XYZ in angstrom, returned in bohr."""
    lines = Path(path).read_text().splitlines()
    frames, symbols, i = [], None, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"invalid XYZ atom count at line {i+1}") from exc
        if n < 1 or i+2+n > len(lines):
            raise ValueError("truncated/empty XYZ frame")
        names, q = [], []
        for line in lines[i+2:i+2+n]:
            row = line.split()
            if len(row) != 4 or not re.fullmatch(r"[A-Z][a-z]?", row[0]):
                raise ValueError("XYZ requires element and three coordinates only")
            names.append(row[0]); q.append([float(x) for x in row[1:]])
        if symbols is not None and names != symbols:
            raise ValueError("all path frames must use identical atom ordering")
        symbols = names
        frames.append(_array(q, (n, 3), "XYZ coordinates") / BOHR_ANGSTROM)
        i += n+2
    if not frames:
        raise ValueError("no XYZ frames")
    return tuple(symbols), np.stack(frames)


def write_xyz(path, symbols, frames):
    q = np.asarray(frames, float)
    if q.ndim == 2:
        q = q[None]
    if q.shape[1:] != (len(symbols), 3) or not np.isfinite(q).all():
        raise ValueError("invalid XYZ coordinates")
    lines = []
    for frame in q:
        lines += [str(len(symbols)), "PyRPMD PES reference; coordinates in angstrom"]
        lines += [f"{s} " + " ".join(f"{x:.16e}" for x in v*BOHR_ANGSTROM)
                  for s, v in zip(symbols, frame)]
    Path(path).write_text("\n".join(lines) + "\n")


def export_qmdff_reference(point: ReferencePoint, prefix):
    """Write the numeric subset consumed by Caracal's ORCA-format readers.

    Files explicitly state that PySCF, not ORCA, generated the data. Hirshfeld
    charges are NOT CM5 charges: upstream applies its own CM5 correction once.
    """
    prefix = Path(prefix)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", prefix.name):
        raise ValueError("reference prefix must be a short simple identifier")
    if any(x is None for x in (point.hessian, point.hirshfeld, point.mayer)):
        raise ValueError("QMDFF requires Hessian, Hirshfeld charges and Mayer BO")
    n = len(point.symbols)
    if n < 2:
        raise ValueError("Caracal QMDFF generation requires at least two atoms")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    header = "# PySCF-generated Caracal interchange, NOT an ORCA calculation.\n"
    out = header + f"FINAL SINGLE POINT ENERGY {point.energy:.17e}\n"
    out += "HIRSHFELD ANALYSIS\n# neutral spherical HF reference densities\n# atomic units\n# atom element charge\n# zero-based atom indices\n"
    out += "\n".join(f"{i} {s} {v:.17e}" for i, (s, v) in enumerate(zip(point.symbols, point.hirshfeld)))
    out += "\n\nMayer bond orders larger than -infinity\n"
    # One pair per line avoids the 128-character upstream line-length limit.
    out += "\n".join(f"B({i},{j}) : {point.mayer[i,j]:.16E}"
                     for i in range(n) for j in range(i+1, n)) + "\n\n"
    prefix.with_suffix(".out").write_text(out)
    # The mass field is unused by rdo; use PyRPMD's canonical atomic masses.
    from ..system import _masses_for_symbols
    masses = _masses_for_symbols(point.symbols)
    from ..constants import AMU_TO_AU
    masses = masses / AMU_TO_AU
    hess = header + "$atoms\n" + str(n) + "\n"
    hess += "\n".join(f"{s} {m:.12e} " + " ".join(f"{x:.17e}" for x in q)
                      for s, m, q in zip(point.symbols, masses, point.positions))
    hess += f"\n$hessian\n{3*n}\n"
    for col in range(0, 3*n, 5):
        hess += " ".join(str(j) for j in range(col, min(col+5, 3*n))) + "\n"
        hess += "\n".join(str(i) + " " + " ".join(f"{x:.17e}" for x in point.hessian[i, col:col+5])
                          for i in range(3*n)) + "\n"
    prefix.with_suffix(".hess").write_text(hess + "$end\n")
    point.save(prefix.with_suffix(".json"))


def write_grad_hess(points, path):
    points = list(points)
    if not points or any(p.hessian is None for p in points):
        raise ValueError("TREQ derivative points require full Hessians")
    symbols = points[0].symbols
    if any(p.symbols != symbols for p in points):
        raise ValueError("inconsistent derivative-point atom ordering")
    def vector(a):
        a = np.asarray(a).ravel(order="C")
        return "\n".join(" ".join(f"{v:.16e}" for v in a[i:i+5]) for i in range(0, len(a), 5))
    text = f"# PySCF reference; bohr/hartree, positive gradients\nNPOINTS {len(points)}\nNATOM {len(symbols)}\n"
    for i, point in enumerate(points, 1):
        text += f"\n*POINT {i}\nENERGY {point.energy:.16e}\n"
        for name, array in (("GEOMETRY", point.positions), ("GRADIENT", point.gradient), ("HESSIAN", point.hessian)):
            text += f"\n{name}\n{vector(array)}\nEND\n"
    Path(path).write_text(text)


def read_eshifts(path, expected):
    rows = [line.split()[1:] for line in Path(path).read_text().splitlines()
            if line.strip().lower().startswith("eshift ")]
    if len(rows) != 1 or len(rows[0]) != expected:
        raise ValueError("Caracal did not produce the expected energy shifts")
    shifts = np.array([float(x.replace("D", "E").replace("d", "e")) for x in rows[0]])
    if not np.isfinite(shifts).all():
        raise ValueError("nonfinite QMDFF energy shift")
    return shifts.tolist()


def validate_coordinates(path, natoms):
    """Validate a nonredundant Caracal coordinate definition, not its conditioning."""
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
    # Bonds: i j; angles: i j k; torsions/OOP: 0/1 i j k l.
    rows = []
    for line in lines:
        fields = list(map(int, line.split()))
        if len(fields) not in (2, 3, 5) or (len(fields) == 5 and fields[0] not in (0, 1)):
            raise ValueError("expected bond, angle or 0/1-prefixed torsion/OOP row")
        atoms = fields[1:] if len(fields) == 5 else fields
        if len(set(atoms)) != len(atoms) or not all(1 <= j <= natoms for j in atoms):
            raise ValueError("invalid one-based internal-coordinate atom indices")
        rows.append(fields)
    if natoms < 3 or len(rows) != 3*natoms-6:
        raise ValueError("TREQ currently requires exactly 3N-6 nonredundant internals and N>=3")
    return rows
