"""Molecular PySCF reference data, not an on-the-fly dynamics provider.

Cartesian arrays use atom-major x/y/z order and atomic units throughout.
Hirshfeld populations use neutral, spherical HF atoms in the molecular basis;
this explicit convention is not asserted identical to ORCA's atomic tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import minimize


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace one document only after a complete, durable write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_hash(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


@dataclass
class ReferencePoint:
    symbols: tuple[str, ...]
    positions: np.ndarray
    energy: float
    gradient: np.ndarray
    hessian: np.ndarray | None = None
    hirshfeld: np.ndarray | None = None
    mayer: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.symbols = tuple(self.symbols)
        n = len(self.symbols)
        if n < 1 or any(not isinstance(x, str) or not x.isalpha() for x in self.symbols):
            raise ValueError("symbols must be a nonempty sequence of element names")
        self.positions = _array(self.positions, (n, 3), "positions")
        self.gradient = _array(self.gradient, (n, 3), "gradient")
        self.energy = float(self.energy)
        if not np.isfinite(self.energy):
            raise ValueError("reference energy is nonfinite")
        if self.hessian is not None:
            self.hessian = _array(self.hessian, (3*n, 3*n), "hessian")
            if not np.allclose(self.hessian, self.hessian.T, atol=1e-7, rtol=0):
                raise ValueError("reference Hessian is not symmetric")
        if self.hirshfeld is not None:
            self.hirshfeld = _array(self.hirshfeld, (n,), "Hirshfeld charges")
        if self.mayer is not None:
            self.mayer = _array(self.mayer, (n, n), "Mayer bond orders")
            if not np.allclose(self.mayer, self.mayer.T, atol=1e-10, rtol=0):
                raise ValueError("Mayer matrix is not symmetric")
        self.provenance = json.loads(json.dumps(self.provenance, allow_nan=False))

    def payload(self):
        return {"schema": "pyrpmd.reference/1", "symbols": list(self.symbols),
                "positions": self.positions.tolist(), "energy": self.energy,
                "gradient": self.gradient.tolist(),
                "hessian": None if self.hessian is None else self.hessian.tolist(),
                "hirshfeld": None if self.hirshfeld is None else self.hirshfeld.tolist(),
                "mayer": None if self.mayer is None else self.mayer.tolist(),
                "units": {"positions": "bohr", "energy": "hartree",
                          "gradient": "hartree/bohr", "hessian": "hartree/bohr^2"},
                "provenance": self.provenance}

    def save(self, path):
        value = self.payload()
        atomic_json(Path(path), {"sha256": canonical_hash(value), "data": value})

    @classmethod
    def load(cls, path):
        outer = json.loads(Path(path).read_text())
        value = outer["data"]
        if outer.get("sha256") != canonical_hash(value):
            raise ValueError("reference checksum mismatch")
        if value.get("schema") != "pyrpmd.reference/1" or value.get("units") != {
            "positions": "bohr", "energy": "hartree", "gradient": "hartree/bohr",
            "hessian": "hartree/bohr^2"
        }:
            raise ValueError("reference schema/units mismatch")
        return cls(**{k: value[k] for k in ("symbols", "positions", "energy", "gradient",
                                         "hessian", "hirshfeld", "mayer", "provenance")})


@dataclass(frozen=True)
class PySCFSettings:
    method: str = "RHF"
    basis: str = "def2-svp"
    xc: str | None = None
    charge: int = 0
    spin: int = 0  # N_alpha - N_beta, not multiplicity
    conv_tol: float = 1e-10
    max_cycle: int = 150
    grid_level: int = 4
    population_grid_level: int = 4
    charge_sum_tolerance: float = 5e-4
    hessian: str = "analytic"
    hessian_step: float = 1e-3
    memory_mb: int = 2000

    def __post_init__(self):
        if self.method not in ("RHF", "UHF", "RKS", "UKS"):
            raise ValueError("method must be RHF, UHF, RKS or UKS")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("an explicit named all-electron basis is required")
        for key in ("charge", "spin", "max_cycle", "grid_level", "population_grid_level", "memory_mb"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.spin < 0 or self.max_cycle < 1 or self.memory_mb < 1:
            raise ValueError("spin >= 0, max_cycle >= 1 and memory_mb >= 1 are required")
        if self.method.startswith("R") and self.spin:
            raise ValueError("restricted reference methods require spin=0")
        if not 0 <= self.grid_level <= 9 or not 0 <= self.population_grid_level <= 9:
            raise ValueError("quadrature grid levels must be in [0, 9]")
        if (self.method.endswith("KS") and (not isinstance(self.xc, str) or not self.xc.strip())):
            raise ValueError("RKS/UKS requires an explicit xc functional")
        if self.method.endswith("HF") and self.xc is not None:
            raise ValueError("HF does not accept an xc functional")
        for key in ("conv_tol", "charge_sum_tolerance", "hessian_step"):
            x = getattr(self, key)
            if isinstance(x, bool) or not np.isfinite(x) or x <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.hessian not in ("analytic", "finite_difference"):
            raise ValueError("hessian must be analytic or finite_difference")


def mayer_bond_orders(density, overlap, slices):
    """Mayer BO: PS*SP, with the spin-density term for unrestricted references."""
    s = np.asarray(overlap, float)
    dm = np.asarray(density, float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError("invalid AO overlap")
    if dm.shape == s.shape:
        dms = (dm,)
        factor = 1.0
    elif dm.shape == (2, *s.shape):
        dms = tuple(dm)
        factor = 2.0
    else:
        raise ValueError("density shape disagrees with AO overlap")
    pair = np.zeros_like(s)
    for d in dms:
        ds = d @ s
        pair += factor * ds * ds.T
    n = len(slices)
    bo = np.zeros((n, n))
    for a, (p0, p1) in enumerate(slices):
        for b, (q0, q1) in enumerate(slices[:a]):
            bo[a, b] = bo[b, a] = np.sum(pair[p0:p1, q0:q1])
    if not np.isfinite(bo).all():
        raise ValueError("Mayer analysis returned nonfinite values")
    return bo


class PySCFReference:
    """Cold-start molecular HF/DFT with derivative and population provenance.

    PySCF imports are lazy. A fresh SCF object is constructed for every geometry:
    neighboring points do not silently inherit an electronic state by warm start.
    """
    def __init__(self, settings: PySCFSettings | dict | None = None):
        self.settings = settings if isinstance(settings, PySCFSettings) else PySCFSettings(**(settings or {}))
        import pyscf
        self.version = pyscf.__version__
        self._atom_cache = {}

    def _solve(self, symbols, q):
        from pyscf import gto, scf, dft
        s = self.settings
        q = _array(q, (len(symbols), 3), "positions")
        mol = gto.M(atom=list(zip(symbols, q.tolist())), unit="Bohr", basis=s.basis,
                    charge=s.charge, spin=s.spin, verbose=0, max_memory=s.memory_mb)
        if mol.has_ecp() or mol.cart or np.any(mol.atom_charges() <= 0):
            raise ValueError("reference populations require real all-electron spherical-basis atoms")
        cls = getattr(dft if s.method.endswith("KS") else scf, s.method)
        mf = cls(mol)
        mf.conv_tol, mf.max_cycle = s.conv_tol, s.max_cycle
        if s.method.endswith("KS"):
            mf.xc = s.xc
            mf.grids.level = s.grid_level
        energy = mf.kernel()
        if not mf.converged or not np.isfinite(energy):
            raise RuntimeError("PySCF SCF did not converge; reference point was not committed")
        gradient = np.asarray(mf.nuc_grad_method().kernel(), float)
        if not np.isfinite(gradient).all():
            raise RuntimeError("PySCF gradient is nonfinite")
        return mf, float(energy), gradient

    def _populations(self, mf):
        from pyscf import gto, dft
        from pyscf.scf import atom_hf
        s, mol = self.settings, mf.mol
        dm = np.asarray(mf.make_rdm1())
        total = dm.sum(axis=0) if dm.ndim == 3 else dm
        slices = [(int(x[2]), int(x[3])) for x in mol.aoslice_by_atom()]
        bo = mayer_bond_orders(dm, mf.get_ovlp(), slices)
        atom_dm = []
        for i, (p0, p1) in enumerate(slices):
            symbol = mol.atom_symbol(i)
            key = (symbol, canonical_hash(mol._basis[symbol]))
            if key not in self._atom_cache:
                atom = gto.M(atom=[(symbol, (0., 0., 0.))], basis={symbol: mol._basis[symbol]},
                             spin=int(mol.atom_charge(i)) % 2, verbose=0, max_memory=s.memory_mb)
                amf = atom_hf.AtomHF1e(atom) if atom.nelectron == 1 else atom_hf.AtomSphAverageRHF(atom)
                amf.conv_tol, amf.max_cycle = s.conv_tol, s.max_cycle
                amf.kernel()
                if not amf.converged:
                    raise RuntimeError(f"neutral spherical atom {symbol} did not converge")
                adm = np.asarray(amf.make_rdm1())
                self._atom_cache[key] = (adm.sum(axis=0) if adm.ndim == 3 else adm).copy()
            adm = self._atom_cache[key]
            if adm.shape != (p1-p0, p1-p0):
                raise ValueError("atomic/molecular AO ordering mismatch")
            atom_dm.append(adm)
        grids = dft.gen_grid.Grids(mol)
        grids.level = s.population_grid_level
        grids.build()
        electrons = np.zeros(mol.natm)
        integrated = 0.0
        for start in range(0, len(grids.weights), 8192):
            xyz, weight = grids.coords[start:start+8192], grids.weights[start:start+8192]
            ao = dft.numint.eval_ao(mol, xyz)
            rho = np.einsum("pi,ij,pj->p", ao, total, ao, optimize=True)
            promol = np.stack([np.einsum("pi,ij,pj->p", ao[:, p0:p1], adm, ao[:, p0:p1],
                                        optimize=True)
                               for (p0, p1), adm in zip(slices, atom_dm)])
            promol = np.maximum(promol, 0.)
            denominator = promol.sum(axis=0)
            fractions = np.divide(promol, denominator, out=np.zeros_like(promol), where=denominator > 1e-300)
            electrons += fractions @ (weight * rho)
            integrated += float(np.dot(weight, rho))
        charges = mol.atom_charges() - electrons
        error = float(charges.sum() - s.charge)
        if abs(error) > s.charge_sum_tolerance or abs(integrated - mol.nelectron) > s.charge_sum_tolerance:
            raise RuntimeError("Hirshfeld integration failed charge conservation; increase population_grid_level")
        return charges, bo, {"hirshfeld_reference": "neutral-spherical-HF-same-basis",
                            "charge_sum_error": error, "integrated_electrons": integrated,
                            "population_grid_points": len(grids.weights)}

    def calculate(self, symbols, positions, *, hessian=True, populations=False):
        symbols = tuple(symbols)
        q = _array(positions, (len(symbols), 3), "positions")
        mf, energy, gradient = self._solve(symbols, q)
        s = self.settings
        info = {"engine": "PySCF", "version": self.version, "settings": asdict(s),
                "scf_converged": True, "scf_stability_checked": False,
                "electronic_state_continuity_checked": False,
                "geometry_sha256": canonical_hash({"symbols": symbols, "q_bohr": q.tolist()})}
        if s.method.startswith("U"):
            info["spin_square"], info["effective_multiplicity"] = map(float, mf.spin_square())
        h = None
        if hessian:
            if s.hessian == "analytic":
                raw = np.asarray(mf.Hessian().kernel())  # [atom, atom, xyz, xyz]
                h = raw.transpose(0, 2, 1, 3).reshape(q.size, q.size)
            else:
                h = np.empty((q.size, q.size))
                for k in range(q.size):
                    plus, minus = q.copy(), q.copy()
                    plus.flat[k] += s.hessian_step
                    minus.flat[k] -= s.hessian_step
                    gp = self._solve(symbols, plus)[2]
                    gm = self._solve(symbols, minus)[2]
                    h[:, k] = ((gp - gm) / (2*s.hessian_step)).ravel()
            if not np.isfinite(h).all():
                raise RuntimeError("PySCF Hessian is nonfinite")
            info["hessian_antisymmetry_max"] = float(np.max(np.abs(h-h.T)))
            if info["hessian_antisymmetry_max"] > 1e-5:
                raise RuntimeError("Hessian antisymmetry exceeds 1e-5 hartree/bohr^2")
            h = (h + h.T) * 0.5
        charges = bo = None
        if populations:
            charges, bo, popinfo = self._populations(mf)
            info.update(popinfo)
        return ReferencePoint(symbols, q, energy, gradient, h, charges, bo, info)

    def optimize(self, symbols, positions, *, gradient_tolerance=1e-5, max_steps=200):
        """Minimum optimization, never a transition-state search or an IRC claim."""
        if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0 or type(max_steps) is not int or max_steps < 1:
            raise ValueError("invalid minimum-optimization controls")
        q = _array(positions, (len(symbols), 3), "positions")
        def objective(flat):
            _, e, g = self._solve(symbols, flat.reshape(q.shape))
            return e, g.ravel()
        result = minimize(objective, q.ravel(), jac=True, method="BFGS",
                          options={"gtol": gradient_tolerance, "maxiter": max_steps})
        if not np.isfinite(result.x).all() or np.max(np.abs(result.jac)) > gradient_tolerance:
            raise RuntimeError(f"minimum optimization did not meet gradient tolerance: {result.message}")
        return result.x.reshape(q.shape)
