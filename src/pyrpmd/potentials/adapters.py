"""Lazy, reusable adapters for supported ML and electronic-structure engines.

All public coordinates/cells are bohr; energies and configurational virials are
hartree, and forces are -dE/dq. Optional engines are imported only on request.
"""

from __future__ import annotations

import importlib
import numpy as np

from ..constants import KB_AU
from ..geometry import validate_cell_pbc
from ..system import _Z
from .base import (
    Potential, PESResult, PotentialError, BackendUnavailableError,
    coordinates, validate_result,
)
from .optional import ASEPotential
from .python import BOHR_ANGSTROM, EV_HARTREE


def _load(module, name, extra):
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise BackendUnavailableError(
            f"{extra} provider needs its supported optional API; "
            f"install 'pyrpmd[{extra}]' (missing {module}.{name})"
        ) from exc


def _symbols(symbols):
    if isinstance(symbols, str) or symbols is None:
        raise ValueError("potential symbols must be an explicit sequence of elements")
    result = tuple("H" if s in ("D", "T") else s for s in symbols)
    if not result or any(s not in _Z or _Z[s] == 0 for s in result):
        raise ValueError("potential symbols must contain real chemical elements")
    return result


def _identity(symbols, charge, multiplicity):
    symbols = _symbols(symbols)
    if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, (int, np.integer)):
        raise ValueError("charge must be an integer")
    if (isinstance(multiplicity, (bool, np.bool_))
            or not isinstance(multiplicity, (int, np.integer)) or multiplicity < 1):
        raise ValueError("multiplicity must be a positive integer")
    return symbols, int(charge), int(multiplicity)


def _positive(value, name, *, integer=False):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or value <= 0
            or (integer and not isinstance(value, (int, np.integer)))):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    return int(value) if integer else float(value)


def _get(system, key, default=None):
    return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)


class _BoundPotential(Potential):
    """Fixed composition with cell rebinding, shared by direct engine adapters."""

    def _bind(self, symbols, cell, pbc, charge=0, multiplicity=1):
        self.symbols, self.charge, self.multiplicity = _identity(symbols, charge, multiplicity)
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        self.closed = False
        self._check_boundary(self.pbc)

    def _check_boundary(self, pbc):
        pass

    def bind_system(self, system):
        symbols, charge, mult = _identity(
            _get(system, "symbols", self.symbols),
            _get(system, "charge", self.charge),
            _get(system, "multiplicity", self.multiplicity),
        )
        if (symbols, charge, mult) != (self.symbols, self.charge, self.multiplicity):
            raise PotentialError("system composition, charge or multiplicity differs from provider")
        cell, pbc = validate_cell_pbc(_get(system, "cell"), _get(system, "pbc"))
        self._check_boundary(pbc)
        if not np.array_equal(pbc, self.pbc):
            self._reset_engine()
        self.cell, self.pbc = cell, pbc

    def _reset_engine(self):
        pass

    def _coordinates(self, x):
        if self.closed:
            raise PotentialError(f"{self.metadata['provider']} provider is closed")
        x = coordinates(x)
        if x.shape[1] != len(self.symbols):
            raise PotentialError("potential atom count mismatch")
        return x

    def close(self):
        self._reset_engine()
        self.closed = True


class ModelASEPotential(ASEPotential):
    """An engine-owned ASE calculator reused across all frames and calls."""

    def __init__(self, symbols, calculator, *, provider, device="cpu", cell=None,
                 pbc=None, charge=0, multiplicity=1):
        symbols, self.charge, self.multiplicity = _identity(symbols, charge, multiplicity)
        super().__init__(symbols, calculator, cell=cell, pbc=pbc)
        self.atoms.info.update(charge=self.charge, spin=self.multiplicity)
        self.metadata = dict(ASEPotential.metadata, provider=provider,
                             execution_device=str(device), native_batch=False)
        self.closed = False

    def bind_system(self, system):
        if (_get(system, "charge", self.charge), _get(system, "multiplicity", self.multiplicity)) != (
            self.charge, self.multiplicity
        ):
            raise PotentialError("system charge or multiplicity differs from provider")
        super().bind_system(system)

    def evaluate(self, x):
        if self.closed:
            raise PotentialError(f"{self.metadata['provider']} provider is closed")
        try:
            return super().evaluate(x)
        except PotentialError:
            raise
        except Exception as exc:
            raise PotentialError(f"{self.metadata['provider']} evaluation failed: {exc}") from exc

    def close(self):
        if not self.closed:
            super().close()
            self.closed = True


class DeePMDPotential(_BoundPotential):
    """Native DeepPot batching; backend-selected device, full or no PBC."""

    metadata = {"provider": "deepmd", "thread_safety": "context_serial",
                "execution_device": "backend", "native_batch": True,
                "periodic": True, "supports_pbc": True}

    def __init__(self, symbols, model, *, cell=None, pbc=None, type_map=None,
                 units=None, model_kwargs=None, eval_kwargs=None):
        from .python import unit_factors
        self._bind(symbols, cell, pbc)
        if model is None:
            raise ValueError("DeePMD requires a model checkpoint")
        self.length_factor, self.energy_factor = unit_factors(
            {"positions": "angstrom", "energy": "eV"} if units is None else units
        )
        DeepPot = _load("deepmd.infer", "DeepPot", "deepmd")
        self.engine = DeepPot(str(model), **dict(model_kwargs or {}))
        try:
            species = self.engine.get_type_map() if type_map is None else list(type_map)
            if (not species or len(set(species)) != len(species)
                    or any(s not in species for s in self.symbols)):
                raise ValueError("DeePMD type_map must uniquely map every system element")
            self.atom_types = np.array([species.index(s) for s in self.symbols], dtype=np.int32)
            self.eval_kwargs = dict(eval_kwargs or {})
            if set(self.eval_kwargs) & {"coords", "cells", "atom_types", "atomic", "mixed_type"}:
                raise ValueError("DeePMD eval_kwargs cannot override coordinates, cells, types or output layout")
        except BaseException:
            self.close()
            raise

    def _check_boundary(self, pbc):
        if pbc.any() and not pbc.all():
            raise PotentialError("DeePMD DeepPot supports full or no PBC; partial PBC is unsupported")

    def bind_system(self, system):
        # Standard DeepPot checkpoints have fixed electronic composition; charge
        # and spin-dependent models must supply their trained fparam/aparam.
        system = {"symbols": _get(system, "symbols", self.symbols),
                  "cell": _get(system, "cell"), "pbc": _get(system, "pbc")}
        super().bind_system(system)

    def evaluate(self, x):
        x = self._coordinates(x)
        cells = None if not self.pbc.any() else np.broadcast_to(
            self.cell.reshape(1, 9) / self.length_factor, (len(x), 9)
        ).copy()
        try:
            result = self.engine.eval(x / self.length_factor, cells, self.atom_types,
                                      atomic=False, **self.eval_kwargs)
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise PotentialError("DeePMD model must return energy, force and virial")
            e, f, w = result
            e = np.asarray(e)
            w = np.asarray(w)
            if e.shape not in ((len(x),), (len(x), 1)):
                raise PotentialError("DeePMD energy shape mismatch")
            if w.shape not in ((len(x), 9), (len(x), 3, 3)):
                raise PotentialError("DeePMD virial shape mismatch")
            return validate_result(PESResult(
                e.reshape(-1) * self.energy_factor,
                np.asarray(f) * self.energy_factor / self.length_factor,
                virial=w.reshape(-1, 3, 3) * self.energy_factor,
            ), x.shape)
        except PotentialError:
            raise
        except Exception as exc:
            raise PotentialError(f"DeePMD evaluation failed: {exc}") from exc

    def _reset_engine(self):
        # PBC rebinding only changes eval input; the trained model stays loaded.
        pass

    def close(self):
        if not self.closed:
            close = getattr(self.engine, "close", None)
            if callable(close):
                close()
            self.engine = None
            self.closed = True


class TBLitePotential(_BoundPotential):
    """On-the-fly xTB with a reusable tblite calculator, in native atomic units."""

    metadata = {"provider": "tblite", "thread_safety": "context_serial",
                "execution_device": "cpu", "native_batch": False,
                "periodic": True, "supports_pbc": True}

    def __init__(self, symbols, *, method="GFN2-xTB", charge=0, multiplicity=1,
                 cell=None, pbc=None, accuracy=1.0, max_iterations=250,
                 electronic_temperature=300.0, settings=None, warm_start=False):
        self._bind(symbols, cell, pbc, charge, multiplicity)
        self.Calculator = _load("tblite.interface", "Calculator", "tblite")
        if method not in ("GFN2-xTB", "GFN1-xTB", "IPEA1-xTB"):
            raise ValueError("tblite method must be GFN2-xTB, GFN1-xTB or IPEA1-xTB")
        self.method = method
        self.settings = {"accuracy": _positive(accuracy, "accuracy"),
                         "max-iter": _positive(max_iterations, "max_iterations", integer=True),
                         "verbosity": 0}
        if (isinstance(electronic_temperature, (bool, np.bool_))
                or not np.isfinite(electronic_temperature) or electronic_temperature < 0):
            raise ValueError("electronic_temperature must be nonnegative kelvin")
        self.settings["temperature"] = float(electronic_temperature) * KB_AU
        self.settings.update(dict(settings or {}))
        if not isinstance(warm_start, (bool, np.bool_)):
            raise ValueError("warm_start must be a boolean")
        self.warm_start = bool(warm_start)
        self.metadata = dict(type(self).metadata, requires_bead_affinity=self.warm_start)
        self.engine = None
        self.results = {}

    def _reset_engine(self):
        self.engine = None
        self.results = {}

    def evaluate(self, x):
        x = self._coordinates(x)
        es, fs, ws = [], [], []
        for i, frame in enumerate(x):
            try:
                if self.engine is None:
                    self.engine = self.Calculator(
                        self.method, np.array([_Z[s] for s in self.symbols], dtype=np.int32),
                        frame, charge=self.charge, uhf=self.multiplicity - 1,
                        lattice=self.cell, periodic=self.pbc,
                    )
                    for name, value in self.settings.items():
                        self.engine.set(name, value)
                else:
                    self.engine.update(positions=frame, lattice=self.cell)
                result = self.engine.singlepoint(self.results.get(i) if self.warm_start else None)
                es.append(float(result.get("energy")))
                fs.append(-np.asarray(result.get("gradient")).copy())
                # tblite calls the tensile energy-strain derivative "virial";
                # its ASE adapter uses stress=virial/V. PyRPMD W=-V*stress.
                ws.append(-np.asarray(result.get("virial")).copy())
                if self.warm_start:
                    self.results[i] = result
            except Exception as exc:
                self._reset_engine()
                raise PotentialError(f"tblite frame {i} failed: {exc}") from exc
        self.results = {i: r for i, r in self.results.items() if i < len(x)}
        return validate_result(PESResult(es, fs, virial=ws), x.shape)


class PySCFPotential(_BoundPotential):
    """Molecular Born-Oppenheimer HF/DFT energies and analytic nuclear forces."""

    metadata = {"provider": "pyscf", "thread_safety": "context_serial",
                "execution_device": "cpu", "native_batch": False,
                "periodic": False, "supports_pbc": False}

    def __init__(self, symbols, *, basis="def2-svp", method="HF", xc=None,
                 charge=0, multiplicity=1, cell=None, pbc=None, ecp=None,
                 conv_tol=1e-10, conv_tol_grad=1e-6, max_cycle=100,
                 grid_level=3, grid_response=True, density_fit=False, auxbasis=None, warm_start=False,
                 max_memory=2000):
        self._bind(symbols, cell, pbc, charge, multiplicity)
        try:
            self.gto = importlib.import_module("pyscf.gto")
            self.scf = importlib.import_module("pyscf.scf")
            self.dft = importlib.import_module("pyscf.dft")
        except ImportError as exc:
            raise BackendUnavailableError("PySCF provider requires pip install 'pyrpmd[pyscf]'") from exc
        method = str(method).upper()
        if method in ("HF", "DFT"):
            method = ("U" if self.multiplicity > 1 else "R") + ("HF" if method == "HF" else "KS")
        if method not in ("RHF", "UHF", "RKS", "UKS"):
            raise ValueError("PySCF method must be HF, DFT, RHF, UHF, RKS or UKS")
        if method.startswith("R") and self.multiplicity != 1:
            raise ValueError("restricted PySCF methods require singlet multiplicity")
        if method.endswith("KS") and (not isinstance(xc, str) or not xc.strip()):
            raise ValueError("PySCF DFT requires an explicit xc functional")
        if method.endswith("HF") and xc is not None:
            raise ValueError("xc is only valid for PySCF DFT methods")
        for value, name in ((density_fit, "density_fit"), (warm_start, "warm_start"),
                            (grid_response, "grid_response")):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
        if isinstance(grid_level, (bool, np.bool_)) or not isinstance(grid_level, (int, np.integer)) or not 0 <= grid_level <= 9:
            raise ValueError("grid_level must be an integer from 0 to 9")
        self.method, self.basis, self.ecp, self.xc = method, basis, ecp, xc
        self.conv_tol = _positive(conv_tol, "conv_tol")
        self.conv_tol_grad = _positive(conv_tol_grad, "conv_tol_grad")
        self.max_cycle = _positive(max_cycle, "max_cycle", integer=True)
        self.max_memory = _positive(max_memory, "max_memory")
        self.grid_level, self.density_fit, self.auxbasis = int(grid_level), bool(density_fit), auxbasis
        self.grid_response = bool(grid_response)
        self.warm_start = bool(warm_start)
        self.metadata = dict(type(self).metadata, requires_bead_affinity=self.warm_start,
                             method=self.method, basis=str(self.basis))
        self.mol = self.engine = None
        self.densities = {}

    def _check_boundary(self, pbc):
        if pbc.any():
            raise PotentialError("PySCF adapter supports molecules only; periodic PySCF is not implemented")

    def _reset_engine(self):
        self.mol = self.engine = None
        self.densities = {}

    def _prepare(self, frame):
        if self.mol is None:
            options = dict(atom=list(zip(self.symbols, frame)), unit="Bohr", basis=self.basis,
                           charge=self.charge, spin=self.multiplicity - 1, verbose=0)
            if self.ecp is not None:
                options["ecp"] = self.ecp
            self.mol = self.gto.M(**options)
            factory = getattr(self.dft if self.method.endswith("KS") else self.scf, self.method)
            self.engine = factory(self.mol)
            if self.density_fit:
                self.engine = self.engine.density_fit(auxbasis=self.auxbasis)
            self.engine.conv_tol = self.conv_tol
            self.engine.conv_tol_grad = self.conv_tol_grad
            self.engine.max_cycle = self.max_cycle
            self.engine.max_memory = self.max_memory
            self.engine.verbose = 0
            self.engine.chkfile = None
            if self.method.endswith("KS"):
                self.engine.xc = self.xc
                self.engine.grids.level = self.grid_level
        else:
            self.mol = self.mol.set_geom_(frame, unit="Bohr", inplace=False)
            self.engine.reset(self.mol)

    def evaluate(self, x):
        x = self._coordinates(x)
        es, fs = [], []
        for i, frame in enumerate(x):
            try:
                self._prepare(frame)
                density = self.densities.get(i) if self.warm_start else None
                if density is None:
                    # SCF.reset retains mo_coeff/mo_occ. Passing dm0=None would
                    # therefore reuse the preceding frame's orbitals inside
                    # PySCF, even with warm_start=False (or for a new slot).
                    density = self.engine.get_init_guess(
                        mol=self.mol, key=self.engine.init_guess,
                    )
                e = self.engine.kernel(dm0=density)
                if not self.engine.converged:
                    raise PotentialError(f"PySCF SCF did not converge in frame {i}")
                grad = self.engine.nuc_grad_method()
                if self.method.endswith("KS"):
                    grad.grid_response = self.grid_response
                gradient = grad.kernel()
                es.append(float(e))
                fs.append(-np.asarray(gradient).copy())
                if self.warm_start:
                    density = self.engine.make_rdm1()
                    self.densities[i] = np.asarray(density).copy()
            except Exception as exc:
                self._reset_engine()
                if isinstance(exc, PotentialError):
                    raise
                raise PotentialError(f"PySCF frame {i} failed: {exc}") from exc
        self.densities = {i: d for i, d in self.densities.items() if i < len(x)}
        force = np.asarray(fs)
        # Isolated molecule: uniform Cartesian strain has no cell contribution.
        virial = np.einsum("bna,bnc->bac", force, x)
        return validate_result(PESResult(es, force, virial=virial), x.shape)


def make_named_potential(kind, *, symbols, model=None, device="cpu", params=None,
                         cell=None, pbc=None, charge=0, multiplicity=1):
    """Build a named provider from a serializable, public input specification."""
    options = dict(params or {})
    protected = {"symbols", "cell", "pbc", "charge", "multiplicity", "device", "model"}
    if protected & set(options):
        raise ValueError("named adapters obtain symbols/cell/charge/spin from System, model/device from potential")
    common = dict(symbols=symbols, cell=cell, pbc=pbc, charge=charge, multiplicity=multiplicity)
    if kind in ("tblite", "xtb", "pyscf"):
        if model is not None:
            raise ValueError(f"{kind} uses potential.params.method, not a model checkpoint")
        if device != "cpu":
            raise ValueError(f"{kind} adapter supports device='cpu' only")
        cls = PySCFPotential if kind == "pyscf" else TBLitePotential
        return cls(**common, **options)
    if model is None:
        raise ValueError(f"{kind} requires an explicit model checkpoint or model name")
    if kind == "deepmd":
        if device not in ("cpu", "auto", None):
            raise ValueError("DeepPot selects its device through its backend; use serial/bead mode and configure the backend, not a PyRPMD GPU device")
        return DeePMDPotential(symbols, model, cell=cell, pbc=pbc, **options)
    if kind == "mace":
        calculator_type = _load("mace.calculators", "MACECalculator", "mace")
        if "model_paths" in options or "models" in options:
            raise ValueError("MACE model is supplied through potential.model")
        options.setdefault("default_dtype", "float64")
        calculator = calculator_type(model_paths=model, device=device, **options)
    elif kind == "nequip":
        calculator_type = _load("nequip.integrations.ase", "NequIPCalculator", "nequip")
        if "compile_path" in options:
            raise ValueError("NequIP compiled model is supplied through potential.model")
        options.setdefault("chemical_species_to_atom_type_map", True)
        loader = getattr(calculator_type, "from_compiled_model", None)
        if loader is None:
            raise BackendUnavailableError("NequIP requires the compiled-model API (nequip>=0.7)")
        calculator = loader(compile_path=str(model), device=device, **options)
    elif kind == "fairchem":
        calculator_type = _load("fairchem.core", "FAIRChemCalculator", "fairchem")
        if not isinstance(options.get("task_name"), str) or not options["task_name"]:
            raise ValueError("fairchem requires potential.params.task_name for the trained chemistry domain")
        if options["task_name"] != "omol" and (charge != 0 or multiplicity != 1):
            raise ValueError(
                "fairchem tasks other than omol do not support selecting charge or spin; "
                "use System charge=0 and default multiplicity=1 (model inputs are charge=0, spin=0)"
            )
        loader = getattr(calculator_type, "from_model_checkpoint", None)
        if loader is None:
            raise BackendUnavailableError("fairchem requires FAIRChemCalculator.from_model_checkpoint; update fairchem-core")
        calculator = loader(str(model), device=device, **options)
    else:
        raise ValueError(f"unknown named provider {kind!r}")
    try:
        provider = ModelASEPotential(calculator=calculator, provider=kind, device=device, **common)
        if kind == "fairchem" and options["task_name"] != "omol":
            # UMA's nonmolecular tasks use a zero spin embedding, not the
            # molecular singlet multiplicity of one. System identity remains
            # unchanged for later bind_system validation.
            provider.atoms.info.update(charge=0, spin=0)
        return provider
    except BaseException:
        close = getattr(calculator, "close", None)
        if callable(close):
            close()
        raise
