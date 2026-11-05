"""Persistent NVT/NPT operator executor with explicit force-provider boundaries.

All numeric inputs are atomic units. Native PES bindings execute complete
blocks without Python callbacks. The explicit ``allow_python=True`` bridge
uses the same operators with copied force buffers and reports every Python
force callback. See docs/native-ensembles.md for equations and limits.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import atomic_savez_uncompressed, load_checkpoint
from .validation import finite_real, integer_count

_THERMOSTATS = {name: i for i, name in enumerate((
    "none", "andersen", "langevin", "pile-l", "pile-g", "svr", "gle",
    "nm-gle", "nm-gle-g", "piglet", "trpmd", "ffl", "nhc",
))}
_BAROSTATS = {"none": 0, "bzp": 1, "mttk": 2, "rgb": 3, "scr": 4}
_COLUMNS = ("step", "time", "potential_energy", "kinetic_primitive",
            "kinetic_centroid_virial", "ring_polymer_hamiltonian", "volume",
            "pressure", "cell_00", "cell_01", "cell_02", "cell_10", "cell_11",
            "cell_12", "cell_20", "cell_21", "cell_22", "fictitious_kinetic_energy")
_ARRAYS = ("q", "p", "aux", "energy", "force", "virial", "cell", "cell_p")
_FORMAT = "pyrpmd.native-ensemble.v1"
_LIBRARY_LOCKS: dict[str, threading.RLock] = {}
_LIBRARY_LOCKS_GUARD = threading.Lock()
_PYTHON_POTENTIAL_LOCKS: dict[int, threading.RLock] = {}


_integer = partial(
    integer_count,
    maximum=2 ** 64 - 1,
    type_message="{name} must be an integer",
    bounds_message="{name} is out of range",
)


_scalar = partial(
    finite_real,
    type_message="{name} must be a finite real number in atomic units",
    finite_message="invalid {name}",
    bounds_message="invalid {name}",
)


def _array(value: Any, shape: tuple[int, ...] | None = None, *, allow_bool: bool = False) -> np.ndarray:
    if np.asarray(value).dtype.kind not in ("fiub" if allow_bool else "fiu"):
        raise ValueError("arrays must contain real numerical values")
    a = np.array(value, dtype=np.float64, order="C", copy=True)
    if shape is not None and a.shape != shape:
        raise ValueError(f"expected shape {shape}, received {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("arrays must be finite")
    return a


def _unknown(data: dict, allowed: set[str], name: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(extra))}")


def _matrix_root(a: np.ndarray, label: str) -> np.ndarray:
    if not np.allclose(a, a.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{label} must be symmetric")
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    tolerance = 1e-11 * max(1.0, float(np.max(np.abs(a))))
    if values.min() < -tolerance:
        raise ValueError(f"{label} is not positive semidefinite")
    return np.ascontiguousarray((vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T)


def _validate_piglet_npt(spec: dict, beads: int):
    """The centroid bath must preserve the canonical piston-coupled momentum.

    PIGLET internal modes deliberately violate the classical fluctuation--
    dissipation relation. The zero-frequency centroid, however, must have a
    unit physical variance and no physical/auxiliary covariance. Otherwise
    the thermostat does not preserve the extended centroid/piston measure.
    This checks that necessary condition, not the accuracy of a fitted set.
    """
    a = _array(spec.get("A"))
    if a.ndim != 3 or a.shape[0] != beads or a.shape[1] != a.shape[2]:
        raise ValueError("PIGLET requires explicitly calibrated per-mode matrices")
    cov = _array(spec.get("C", np.eye(a.shape[-1])))
    if cov.ndim == 2:
        cov = np.repeat(cov[None], beads, axis=0)
    if cov.shape != a.shape:
        raise ValueError("GLE covariance dimensions differ from A")
    if (not math.isclose(float(cov[0, 0, 0]), 1.0, rel_tol=1e-10, abs_tol=1e-12)
            or not np.allclose(cov[0, 0, 1:], 0, atol=1e-12, rtol=0)
            or not np.allclose(cov[0, 1:, 0], 0, atol=1e-12, rtol=0)):
        raise ValueError("PIGLET NPT requires a canonical centroid with zero physical/auxiliary covariance")


def _gle_matrices(spec: dict, mode: str, beads: int, beta: float,
                  dt: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """One-time matrix preparation; no Python linear algebra during propagation."""
    from scipy.linalg import expm

    _unknown(spec, {"A", "C", "beta", "beads", "provenance", "parameter_metadata"}, "GLE")
    a = _array(spec["A"])
    if a.ndim not in (2, 3) or a.shape[-1] != a.shape[-2] or not 1 <= a.shape[-1] <= 64:
        raise ValueError("GLE A must have shape [s,s] or [P,s,s], with 1 <= s <= 64")
    n = a.shape[-1]
    nm = mode != "gle"
    blocks = beads if nm else 1
    if a.ndim == 2:
        if mode == "piglet":
            raise ValueError("PIGLET requires explicitly calibrated per-mode matrices")
        a = np.repeat(a[None], blocks, axis=0)
    if a.shape != (blocks, n, n):
        raise ValueError("GLE drift matrix bead count mismatch")
    cov = _array(spec.get("C", np.eye(n)))
    if cov.ndim == 2:
        cov = np.repeat(cov[None], blocks, axis=0)
    if cov.shape != a.shape:
        raise ValueError("GLE covariance dimensions differ from A")
    if mode == "piglet":
        if not isinstance(spec.get("provenance"), str) or not spec["provenance"].strip():
            raise ValueError("PIGLET requires parameter provenance; no invented presets are supplied")
        if _integer(spec.get("beads"), "GLE beads", 1) != beads or not math.isclose(
            _scalar(spec.get("beta"), "GLE beta", positive=True), beta, rel_tol=1e-12,
        ):
            raise ValueError("PIGLET parameters do not match beta and bead count")
        if not math.isclose(float(cov[0, 0, 0]), 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("PIGLET centroid momentum must have the classical temperature")
    ts, ss, roots = [], [], []
    for drift, covariance in zip(a, cov):
        if np.min(np.real(np.linalg.eigvals(drift))) <= 0:
            raise ValueError("GLE drift must have eigenvalues with positive real parts")
        if mode != "piglet" and (
            not math.isclose(float(covariance[0, 0]), 1.0, rel_tol=1e-10, abs_tol=1e-12)
            or not np.allclose(covariance[0, 1:], 0, atol=1e-12, rtol=0)
            or not np.allclose(covariance[1:, 0], 0, atol=1e-12, rtol=0)
        ):
            raise ValueError("canonical GLE needs unit physical variance and zero physical/auxiliary covariance")
        roots.append(_matrix_root(covariance, "GLE covariance"))
        _matrix_root(drift @ covariance + covariance @ drift.T, "continuous GLE diffusion")
        transition = expm(-dt * drift)
        noise = covariance - transition @ covariance @ transition.T
        ts.append(transition)
        ss.append(_matrix_root((noise + noise.T) / 2, "discrete GLE diffusion"))
    return n, _array(ts).ravel(), _array(ss).ravel(), _array(roots).ravel()


class NativePeriodicPES:
    """Trusted compiled callback described by ``include/pyrpmd/thermal_pes.h``.

    ``identity`` must identify all immutable context/model parameters when
    ``context`` is nonzero. The owner is pinned and checked for ``closed`` at
    every block boundary. Evaluators must be stateless with respect to outputs.
    """
    def __init__(self, library: str | Path, symbol: str, natoms: int, *,
                 context: int = 0, owner: object | None = None,
                 identity: str | None = None):
        self.path = Path(library).resolve(strict=True)
        self.natoms = _integer(natoms, "natoms", 1)
        self.context = _integer(context, "context")
        if self.context and (owner is None or not isinstance(identity, str) or not identity.strip()):
            raise ValueError("a nonzero context requires a lifetime owner and immutable model identity")
        self.owner = owner
        self.lib = ctypes.CDLL(str(self.path))
        self.function = getattr(self.lib, symbol)
        self.address = ctypes.cast(self.function, ctypes.c_void_p).value
        self.identity = {
            "kind": "thermal-callback-v1", "symbol": symbol, "natoms": self.natoms,
            "library_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "model": identity,
        }
        with _LIBRARY_LOCKS_GUARD:
            self.lock = _LIBRARY_LOCKS.setdefault(str(self.path), threading.RLock())

    @contextmanager
    def acquire(self):
        with self.lock:
            if getattr(self.owner, "closed", False):
                raise RuntimeError("native PES context is closed")
            # The compiled callback receives its cell explicitly, while
            # evaluate() uses this wrapper's bound reference geometry. Keep
            # their Python-visible binding transactional at block boundaries.
            previous = {key: getattr(self, key) for key in ("cell", "pbc", "metadata")
                        if hasattr(self, key)}
            try:
                yield ("thermal-v1", self.address, self.context, self.natoms)
            except BaseException:
                for key in ("cell", "pbc", "metadata"):
                    if key in previous:
                        setattr(self, key, previous[key])
                    elif hasattr(self, key):
                        delattr(self, key)
                raise


    def bind_system(self, system):
        from .geometry import validate_cell_pbc
        get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
        with self.lock:
            self.cell, flags = validate_cell_pbc(get("cell"), get("pbc"))
            self.pbc = np.ascontiguousarray(flags, dtype=np.int32)
        self.metadata = {"supports_pbc": True, "provider": "native_periodic",
                         "thread_safety": "serialized", "execution_device": "cpu"}

    def evaluate(self, q):
        """Preflight/reference evaluation; not called from the native step loop."""
        from .potentials.base import (
            PESResult,
            PotentialError,
            coordinates,
            validate_result,
        )
        x = coordinates(q)
        if x.shape[1] != self.natoms:
            raise PotentialError("native PES atom count mismatch")
        if not hasattr(self, "pbc"):
            raise PotentialError("bind_system must precede Python PES evaluation")
        e, f, v = np.full(len(x), np.nan), np.full_like(x, np.nan), np.full((len(x), 3, 3), np.nan)
        message = ctypes.create_string_buffer(512)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        with self.acquire():
            cell = np.eye(3) if self.cell is None else self.cell
            signature = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.c_void_p, ctypes.c_int64)
            fn = signature(self.address)
            status = fn(self.context, len(x), self.natoms, ptr(x), ptr(cell),
                        self.pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr(e), ptr(f), ptr(v), message, 512)
        if status:
            raise PotentialError(f"native thermal PES failed: {message.value.decode(errors='replace')}")
        return validate_result(PESResult(e, f, virial=v), x.shape)


class _PythonPotentialBinding:
    """Explicit copied-buffer interpreter bridge, separate from native PES ABIs.

    The outer lease holds the provider cell stable across an entire accepted
    block. On failure it restores the previous cell alongside C++ rollback.
    Provider evaluations themselves retain the ordinary stateless PES contract.
    """
    def __init__(self, potential, beads, natoms, cell, *, variable_cell):
        if not callable(getattr(potential, "evaluate", None)):
            raise ValueError("Python ensemble potential requires evaluate(q)")
        if variable_cell and not callable(getattr(potential, "set_cell", None)):
            raise ValueError("Python NPT requires explicit potential.set_cell(cell) support")
        self.potential, self.beads, self.natoms = potential, beads, natoms
        self.cell = None if cell is None else np.array(cell, dtype=np.float64, copy=True)
        self.variable_cell = variable_cell
        self.calls = 0
        with _LIBRARY_LOCKS_GUARD:
            self.lock = _PYTHON_POTENTIAL_LOCKS.setdefault(id(potential), threading.RLock())

    def set_cell(self, cell):
        if self.cell is None:
            return
        value = np.array(cell, dtype=np.float64, copy=True).reshape(3, 3)
        setter = getattr(self.potential, "set_cell", None)
        if callable(setter):
            setter(value.copy())
        elif not np.array_equal(value, self.cell):
            raise ValueError("Python potential cannot change its periodic cell")
        self.cell = value

    @contextmanager
    def acquire(self):
        with self.lock:
            before = None if self.cell is None else self.cell.copy()
            try:
                yield None
            except BaseException:
                if before is not None:
                    self.set_cell(before)
                raise

    def evaluate(self, coordinates, cell):
        from .potentials.base import PotentialError, evaluate_checked
        q = np.frombuffer(coordinates, dtype=np.float64).reshape(self.beads, self.natoms, 3).copy()
        if self.cell is not None:
            self.set_cell(np.frombuffer(cell, dtype=np.float64).reshape(3, 3))
        self.calls += 1
        result = evaluate_checked(self.potential, q)
        virial = result.virial
        if virial is None:
            if self.cell is not None:
                raise PotentialError("periodic ensemble propagation requires per-bead physical configurational virials")
            virial = np.zeros((self.beads, 3, 3), dtype=np.float64)
        return (np.ascontiguousarray(result.energy, dtype=np.float64).tobytes(),
                np.ascontiguousarray(result.forces, dtype=np.float64).tobytes(),
                np.ascontiguousarray(virial, dtype=np.float64).tobytes())

class NativeEnsemble:
    """A persistent C++ trajectory with immutable physics configuration.

    q: [P,N,3] bohr; masses: [N] electron masses; beta: hartree^-1;
    dt: atomic time; p: optional [P,N,3] atomic momenta. Omitted p is drawn in
    C++, including the joint GLE auxiliary distribution. Thermostat/barostat
    parameters use inverse atomic time, atomic time and hartree/bohr^3.
    ``allow_python=True`` explicitly selects a copied-buffer Python PES bridge
    for ordinary evaluator objects. Native bindings remain the default. The
    bridge reports its force callbacks and requires real virials for a cell.
    """
    def __init__(self, q, masses, beta: float, dt: float, *, p=None,
                 potential=None, potential_identity: str | None = None,
                 thermostat: str | dict = "pile-l", barostat: str | dict = "none",
                 integrator: str = "cayley-bcb", cell=None, pbc=None,
                 fixed_atoms=(), seed: int = 0, fuse_modes: bool = True,
                 allow_python: bool = False, sampling_integrator: str | None = None):
        try:
            from . import _thermal
        except ImportError as exc:
            raise RuntimeError("native ensemble extension is unavailable; rebuild PyRPMD with C++ enabled") from exc
        self._native = _thermal
        if getattr(_thermal, "THERMAL_KERNEL_API", 0) < 2:
            raise RuntimeError("rebuild PyRPMD: the native extension lacks the versioned thermal-kernel API")
        coordinates = _array(q)
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or 0 in coordinates.shape:
            raise ValueError("q must have shape [P,N,3] with positive P,N")
        self.beads, self.natoms, _ = coordinates.shape
        mass = _array(masses, (self.natoms,))
        if np.any(mass <= 0):
            raise ValueError("masses must be positive")
        beta = _scalar(beta, "beta", positive=True)
        dt = _scalar(dt, "dt", positive=True)
        self.dt = dt
        from .engine import PropagationPlan
        self.plan = PropagationPlan(integrator, thermostat, barostat,
                                    sampling_integrator=sampling_integrator)
        if (getattr(potential, "metadata", {}) or {}).get("action") == "suzuki_chin":
            raise ValueError("NativeEnsemble observable rows use primitive estimators; use SuzukiChinPotential with Dynamics and an external BZPBarostat")
        integrator = self.plan.ring_integrator
        if integrator not in ("cayley-bcb", "normal-mode-bab"):
            raise ValueError("native ensembles support Cayley BCOCB or exact-normal-mode BAOAB, not Verlet")
        fixed = np.zeros(self.natoms)
        for value in fixed_atoms:
            index = _integer(value, "fixed atom")
            if index >= self.natoms or fixed[index]:
                raise ValueError("fixed atom is out of range or repeated")
            fixed[index] = 1
        mom = np.zeros_like(coordinates) if p is None else _array(p, coordinates.shape)
        h = np.eye(3) if cell is None else _array(cell, (3, 3))
        flags = [cell is not None] * 3 if pbc is None else list(pbc)
        if len(flags) != 3 or any(not isinstance(x, (bool, np.bool_)) for x in flags):
            raise ValueError("pbc must contain three booleans")
        if cell is None and any(flags):
            raise ValueError("PBC requires a cell")
        # The public adapters and the persistent executor share one canonical
        # configuration schema. Native matrix/shape/ensemble checks remain here.
        t = self.plan.thermostat_spec
        b = self.plan.barostat_spec
        if not isinstance(b, dict):
            raise ValueError("NativeEnsemble requires a barostat specification")
        mode, bmode = t["mode"], b["mode"]
        if mode not in _THERMOSTATS or bmode not in _BAROSTATS:
            raise ValueError("requested operator is unavailable in the native executor")
        middle_npt = sampling_integrator is not None and bmode != "none"
        if middle_npt:
            if bmode != "bzp" or mode != "pile-l":
                raise ValueError("explicit middle NPT requires isotropic BZP and PILE-L")
            if getattr(_thermal, "MIDDLE_BZP_API", 0) < 1:
                raise RuntimeError("rebuild the native extension for middle BZP sampling")
        cell_mask = _array(b.get("cell_mask", [[1, 1, 1], [0, 1, 1], [0, 0, 1]]), (3, 3), allow_bool=True)
        external = _array(b.get("stress", np.zeros((3, 3))), (3, 3))
        if not np.allclose(external, external.T, atol=1e-12, rtol=1e-12):
            raise ValueError("external stress must be symmetric")
        cfg = {
            "atoms": self.natoms, "beads": self.beads, "beta": beta, "dt": dt,
            "masses": mass, "fixed": fixed,
            "thermostat": _THERMOSTATS[mode], "barostat": _BAROSTATS[bmode],
            "cayley": integrator == "cayley-bcb", "initialize_momenta": p is None,
            "has_cell": cell is not None, "pbc": _array(flags, allow_bool=True),
            "gamma": _scalar(t.get("gamma", 1.0), "gamma", nonnegative=True),
            "gamma0": _scalar(t.get("gamma_centroid", 1.0), "gamma_centroid", nonnegative=True),
            "pile_scale": _scalar(t.get("pile_scale", 1.0), "pile_scale", nonnegative=True),
            "tau": _scalar(t.get("tau", 1.0), "thermostat tau", positive=True),
            "interval": _integer(t.get("interval", 10), "Andersen interval", 1),
            "seed": _integer(seed, "seed"), "gle_size": 0,
            "gle_T": np.empty(0), "gle_S": np.empty(0), "gle_Croot": np.empty(0),
            "pressure": _scalar(b.get("pressure", 0), "pressure"),
            "baro_tau": _scalar(b.get("tau", 1.0), "barostat tau", positive=True),
            "baro_gamma": _scalar(b.get("gamma", 1.0), "barostat gamma", nonnegative=True),
            "compressibility": _scalar(b.get("compressibility", 0.0), "compressibility", nonnegative=True),
            "cell_mask": cell_mask.ravel(),
            "reference_cell": _array(b.get("reference_cell", h), (3, 3)).ravel(),
            "external_stress": external.ravel(),
            "potential": 0, "address": 0, "context": 0, "parameters": np.ones(3),
        }
        if t.get("rng") == "philox-v1":
            cfg["rng"] = "philox-v1"
        if middle_npt:
            cfg["middle_npt"] = True  # Scientific/checkpoint-significant, absent on legacy maps.
        if mode == "nhc":
            if "tau" in t and "tdamp" in t:
                raise ValueError("specify only one of NHC tau and tdamp")
            cfg["tau"] = _scalar(t.get("tdamp", t.get("tau", 1.0)), "NHC tdamp", positive=True)
            cfg["nhc_length"] = _integer(t.get("chain_length", 3), "NHC chain_length", 1)
            cfg["nhc_substeps"] = _integer(t.get("substeps", 1), "NHC substeps", 1)
            if cfg["nhc_length"] > 64 or cfg["nhc_substeps"] > 100000:
                raise ValueError("NHC chain_length must be <=64 and substeps <=100000")
        if mode in ("gle", "nm-gle", "nm-gle-g", "piglet"):
            if "gle" not in t:
                raise ValueError("GLE requires an explicit drift matrix")
            if mode == "piglet" and bmode != "none":
                if bmode != "bzp":
                    raise ValueError("PIGLET NPT supports only the isotropic BZP barostat")
                _validate_piglet_npt(t["gle"], self.beads)
            cfg["gle_size"], cfg["gle_T"], cfg["gle_S"], cfg["gle_Croot"] = _gle_matrices(
                dict(t["gle"]), mode, self.beads, beta, dt if bmode == "none" else dt / 2,
            )
        elif "gle" in t:
            raise ValueError("GLE matrices supplied to a non-GLE thermostat")
        if mode == "trpmd" and "gamma_centroid" in t and cfg["gamma0"] != 0:
            raise ValueError("TRPMD must leave the centroid unthermostatted")
        if mode == "trpmd":
            cfg["gamma0"] = 0.0
        self.thermostat, self.barostat = mode, bmode
        self._report_gle_parameters = "parameter_metadata" in t.get("gle", {})
        self.splitting = ("BCOCB" if cfg["cayley"] else "BAOAB") if bmode == "none" else "OBABO"
        if middle_npt:
            self.splitting = "BZP-BCOCB" if cfg["cayley"] else "BZP-BAOAB"
        if mode == "nhc" and bmode == "none":
            self.splitting = "OBCBO" if cfg["cayley"] else "OBABO"
        self._binding = None
        self._python_binding = None
        self._potential = potential
        if not isinstance(allow_python, bool):
            raise ValueError("allow_python must be boolean")
        if potential is None or isinstance(potential, dict):
            spec = {"kind": "free"} if potential is None else dict(potential)
            kind = spec.get("kind", "free")
            _unknown(spec, {"kind"} | {"free": set(), "harmonic": {"k"}, "periodic_cosine": {"heights"}}.get(kind, set()), "potential")
            cfg["potential"] = {"free": 0, "harmonic": 1, "periodic_cosine": 2}.get(kind, -1)
            if cfg["potential"] < 0:
                raise ValueError("unknown native built-in potential")
            if kind == "harmonic":
                cfg["parameters"] = _array([_scalar(spec.get("k", 1), "k", nonnegative=True)])
            elif kind == "periodic_cosine":
                cfg["parameters"] = _array(spec["heights"], (3,))
                if any(not flag and height != 0 for flag, height in zip(flags, cfg["parameters"])):
                    raise ValueError("cosine height must be zero along a nonperiodic direction")
            self._potential_identity = {"kind": kind, "parameters": cfg["parameters"].tolist()}
        else:
            if isinstance(potential, NativePeriodicPES):
                self._binding = potential
                self._potential_identity = potential.identity
                cfg["potential"] = 3
            elif allow_python:
                if not isinstance(potential_identity, str) or not potential_identity.strip():
                    raise ValueError("Python provider requires potential_identity for safe restart")
                self._binding = self._python_binding = _PythonPotentialBinding(
                    potential, self.beads, self.natoms, None if cell is None else h,
                    variable_cell=bmode != "none",
                )
                self._potential_identity = {"kind": "python-pes", "model": potential_identity}
                cfg["potential"] = 6
                cfg["python_callback"] = self._python_binding.evaluate
            else:
                from .potentials.native import descriptor_binding
                from .potentials._builtin_native import resolve_native_binding
                binding = (resolve_native_binding(potential, self.natoms, cell=cell)
                           if callable(getattr(potential, "native_binding", None)) else potential)
                self._binding = descriptor_binding(binding)
                if not isinstance(potential_identity, str) or not potential_identity.strip():
                    raise ValueError("native provider requires potential_identity for safe restart")
                self._potential_identity = {"kind": self._binding.kind, "model": potential_identity}
                cfg["potential"] = {"cabi-v1": 4, "fortran-batch-v1": 5}[self._binding.kind]
        if not isinstance(fuse_modes, bool):
            raise ValueError("fuse_modes must be boolean")
        cfg["fuse_modes"] = fuse_modes
        self._cfg = cfg
        with self._lease() as descriptor:
            if descriptor is not None:
                _, address, context, natoms = descriptor
                if natoms != self.natoms:
                    raise ValueError("native PES atom count mismatch")
                cfg["address"], cfg["context"] = int(address), int(context)
            self._handle = _thermal.create(cfg, coordinates, mom, h.ravel(), self._binding)
            if isinstance(self._binding, NativePeriodicPES):
                self._sync_bound_cell(h)
        # Addresses and initialization draws are not physics. All model, units,
        # fixed positions, integrator and stochastic matrices are fingerprinted.
        physics = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                   for key, value in cfg.items()
                   if key not in {"address", "context", "initialize_momenta", "seed", "python_callback"}}
        rng_abi = (getattr(_thermal, "philox_abi", "philox-v1")
                   if cfg.get("rng") == "philox-v1"
                   else getattr(_thermal, "rng_abi", "std-cxx17-local-build-v1"))
        physics.update(potential_identity=self._potential_identity, initial_cell=h.tolist(),
                       fixed_positions=coordinates[0, fixed.astype(bool)].tolist(),
                       rng_abi=rng_abi, format=_FORMAT)
        if self._report_gle_parameters:
            # New parameter records carry their calibration and source into
            # restart identity. Legacy explicit matrices keep their old hash.
            physics["gle_parameters"] = t["gle"]
        self.fingerprint = hashlib.sha256(json.dumps(physics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self._call_lock = threading.RLock()

    def _lease(self):
        return nullcontext(None) if self._binding is None else self._binding.acquire()

    def _sync_bound_cell(self, cell):
        if self._python_binding is not None:
            self._python_binding.set_cell(cell)
        elif isinstance(self._binding, NativePeriodicPES):
            self._binding.bind_system({
                "cell": np.asarray(cell).reshape(3, 3) if self._cfg["has_cell"] else None,
                "pbc": [bool(flag) for flag in self._cfg["pbc"]],
            })

    def run(self, steps: int, *, stride: int = 1, record_centroids: bool = False,
            trajectory: str | Path | None = None, record_moments: bool = False, sample_start: int = 0) -> dict:
        """Advance one complete C++ block; observation times use the absolute clock.

        A native/PES failure restores the complete pre-block state and RNG.
        Python evaluators require explicit construction with allow_python=True;
        native evaluator failures never trigger a backend fallback.
        No final frame is inserted off the stride clock, so concatenated blocks
        have exactly the same samples as a single call.
        """
        return self._run(steps, stride=stride, record_centroids=record_centroids,
                         trajectory=trajectory, record_moments=record_moments,
                         sample_start=sample_start)

    def run_prefix(self, steps: int, *, stride: int = 1, record_centroids: bool = True,
                   record_moments: bool = True, sample_start: int = 0,
                   observe=None, record_energy: bool | None = None,
                   check_energy: bool = False, energy_parts: bool = False,
                   ring_radii: bool = False, _outer_transaction: bool = False,
                   _atomic: bool = False) -> dict:
        """Advance a C++ block and retain every fully accepted step on failure.

        Force and observation validation share a per-step transaction including
        RNG, auxiliaries, cached forces and cell. The result adds ``completed``
        and ``error`` (None on success). Failed-step observations are discarded.
        Prefix records stay in memory; streaming ``run`` remains block-atomic.

        Supplying ``observe`` or energy/diagnostic options selects the common
        propagation observation contract. Its binary mask has length ``steps``;
        only selected rows are recorded, with centroid momenta returned directly
        in ``centroid_momenta``. ``record_energy`` defaults to True in this mode.
        Unrequested thermal estimators and ring radii are neither computed nor
        validated. ``check_energy`` validates the total Hamiltonian initially
        and after every step, including steps without an observation. Optional
        ``energy_parts`` contains physical potential/P, centroid kinetic,
        noncentroid kinetic/P and spring/P. This mode requires the default
        stride/start and enabled centroid/momentum records.
        """
        common = None
        for name, flag in (("check_energy", check_energy), ("energy_parts", energy_parts),
                           ("ring_radii", ring_radii)):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if record_energy is not None and not isinstance(record_energy, (bool, np.bool_)):
            raise ValueError("record_energy must be boolean or None")
        if observe is not None or record_energy is not None or check_energy or energy_parts or ring_radii:
            steps = _integer(steps, "steps")
            values = np.ones(steps, dtype=np.uint8) if observe is None else np.asarray(observe)
            if values.shape != (steps,) or values.dtype.kind not in "bui" or np.any((values != 0) & (values != 1)):
                raise ValueError("observe must be a boolean/binary mask with length steps")
            if stride != 1 or sample_start != 0 or not record_centroids or not record_moments:
                raise ValueError("common observations require stride=1, sample_start=0 and centroid/momentum records")
            common = {"mask": np.asarray(values, dtype=np.uint8).tobytes(),
                      "record_energy": True if record_energy is None else bool(record_energy),
                      "check_energy": bool(check_energy), "energy_parts": bool(energy_parts),
                      "ring_radii": bool(ring_radii)}
        return self._run(steps, stride=stride, record_centroids=record_centroids,
                         record_moments=record_moments, sample_start=sample_start,
                         keep_prefix=True, common_options=common,
                         outer_transaction=_outer_transaction, atomic=_atomic)

    def _run(self, steps, *, stride=1, record_centroids=False, trajectory=None,
             record_moments=False, sample_start=0, keep_prefix=False, common_options=None,
             outer_transaction=False, atomic=False):
        steps = _integer(steps, "steps")
        stride = _integer(stride, "stride", 1)
        sample_start = _integer(sample_start, "sample_start")
        if not isinstance(record_centroids, bool):
            raise ValueError("record_centroids must be boolean")
        if not isinstance(record_moments, bool) or (record_moments and not record_centroids):
            raise ValueError("record_moments requires record_centroids=True")
        width = (7 if record_moments else 3) * self.natoms
        temporary = None
        published = False
        published_stat = None
        data = None
        path = None if trajectory is None else Path(trajectory)
        # Hold both the engine and provider locks through publication/rollback.
        # Otherwise another thread could advance this engine before an I/O
        # failure restores its state. The extra snapshot is O(P*N), not O(steps).
        with self._call_lock, self._lease():
            callback_start = 0 if self._python_binding is None else self._python_binding.calls
            before = (self._native.state(self._handle)
                      if not outer_transaction and (keep_prefix or isinstance(self._binding, NativePeriodicPES)) else None)
            try:
                if path is not None:
                    if path.exists():
                        raise FileExistsError(f"trajectory already exists: {path}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
                    os.close(fd)
                    before = self._native.state(self._handle)
                advance = self._native.run_prefix if keep_prefix and not atomic else self._native.run
                result = advance(self._handle, steps, stride, record_centroids,
                                 temporary or "", record_moments, sample_start,
                                 *(() if common_options is None else (common_options,)))
                if keep_prefix and atomic:
                    result.update(completed=steps, error="")
                if keep_prefix and result["error"] and self._python_binding is not None:
                    accepted = self._native.state(self._handle)
                    self._python_binding.set_cell(np.frombuffer(accepted["cell"], dtype=np.float64).reshape(3, 3))
                frames = result["frames"]
                if path is None:
                    samples = np.frombuffer(result["samples"], dtype=np.float64).reshape(frames, len(_COLUMNS))
                    records = (np.frombuffer(result["centroids"], dtype=np.float64).reshape(frames, width)
                               if record_centroids else None)
                else:
                    # Flush before publishing; link is exclusive (no overwrite).
                    # Windows' _commit requires a writable descriptor. The
                    # native writer has closed; do not truncate its payload.
                    with open(temporary, "rb+") as stream:
                        os.fsync(stream.fileno())
                    published_stat = os.stat(temporary)
                    os.link(temporary, path)
                    published = True
                    # Windows cannot unlink either hard link while the file
                    # is memory-mapped. Retain its identity for rollback, then
                    # remove our temporary name before opening the mapping.
                    os.unlink(temporary)
                    temporary = None
                    data = np.load(path, mmap_mode="r" if frames else None, allow_pickle=False)
                    samples, records = data[:, :len(_COLUMNS)], data[:, len(_COLUMNS):]
                centroids = (records[:, :3*self.natoms].reshape(frames, self.natoms, 3)
                             if record_centroids else None)
                velocities = (records[:, 3*self.natoms:6*self.natoms].reshape(frames, self.natoms, 3)
                              if record_moments and common_options is None else None)
                radii = (records[:, 6*self.natoms:]
                         if record_moments and (common_options is None or common_options["ring_radii"]) else None)
                common_records = {}
                if common_options is not None:
                    common_records["centroid_momenta"] = records[:, 3*self.natoms:6*self.natoms].reshape(frames, self.natoms, 3)
                    common_records["energy_parts"] = (np.frombuffer(result["energy_parts"], dtype=np.float64).reshape(frames, 4)
                                                       if common_options["energy_parts"] else None)
                if isinstance(self._binding, NativePeriodicPES):
                    current = self._native.state(self._handle)
                    self._sync_bound_cell(np.frombuffer(current["cell"], dtype=np.float64).reshape(3, 3))
                if temporary is not None:
                    os.unlink(temporary)
                    temporary = None
            except BaseException:
                # No arrays have been handed to the caller yet. Release this
                # mapping before rollback attempts to remove the owned file.
                if isinstance(data, np.memmap) and data._mmap is not None:
                    data._mmap.close()
                if before is not None:
                    self._native.restore(self._handle, before)
                # Only remove the hard link that WE created. Never delete an
                # existing or concurrently replaced user's trajectory.
                if (published and published_stat is not None and path.exists()
                        and os.path.samestat(published_stat, path.lstat())):
                    path.unlink()
                raise
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
        return {"samples": samples, "columns": _COLUMNS, "centroids": centroids,
                "velocities": velocities, "radii_gyration_squared": radii,
                "step": result["step"], "force_evaluations": result["force_evaluations"],
                "backend": "native-cxx17" if self._python_binding is None else "native-operators-python-pes",
                "python_step_callbacks": 0 if self._python_binding is None else self._python_binding.calls-callback_start,
                "thermostat": self.thermostat, "barostat": self.barostat,
                "splitting": self.splitting, "trajectory": None if path is None else str(path),
                **({"gle_parameters": self.plan.thermostat_spec["gle"]}
                   if self._report_gle_parameters else {}),
                **common_records,
                **({"completed": result["completed"], "error": result["error"] or None} if keep_prefix else {})}

    def set_state(self, q, p, step: int = 0) -> None:
        """Import physical coordinates, momenta and clock without resetting baths.

        All cached forces, energies and virials are reevaluated at the imported
        coordinates. A rejected state or failed PES leaves the entire engine
        unchanged. This is an explicit state edit, not a checkpoint restore.
        """
        shape = (self.beads, self.natoms, 3)
        coordinates, momenta = _array(q, shape), _array(p, shape)
        clock = _integer(step, "step")
        with self._call_lock, self._lease():
            before = self._native.state(self._handle) if isinstance(self._binding, NativePeriodicPES) else None
            try:
                self._native.set_state(self._handle, coordinates, momenta, clock)
                if before is not None:
                    self._sync_bound_cell(np.frombuffer(before["cell"], dtype=np.float64).reshape(3, 3))
            except BaseException:
                if before is not None:
                    self._native.restore(self._handle, before)
                raise

    def copy_thermostat_from(self, other: "NativeEnsemble", *, step: int | None = None) -> None:
        """Transfer compatible bath variables and RNG across execution backends.

        Coordinates, momenta, force caches and piston state remain unchanged.
        The native layer verifies thermostat parameters and masses. GLE also
        verifies its prepared transition matrices; changing the bath duration
        requires a new declared propagation scheme rather than a silent copy.
        """
        if not isinstance(other, NativeEnsemble) or other._native is not self._native:
            raise TypeError("thermostat source must use the same native extension")
        clock = None if step is None else _integer(step, "step")
        first, second = sorted((self, other), key=id)
        with first._call_lock, second._call_lock:
            self._native.copy_thermostat(self._handle, other._handle, clock)

    def apply_thermostat(self, q, p, dt: float | None = None, *, step: int | None = None) -> np.ndarray:
        """Apply this engine's bath to external momenta, without force calls.

        ``p`` is updated in place and returned; external and internal ``q``
        are never changed. Use a separate free-potential helper for a bath in
        another propagator rather than alternating this operation with run(). RNG,
        auxiliary variables and the bath-application counter persist in the
        ordinary checkpoint. Optional ``step`` supplies the physical collision
        clock (for Andersen) when called as an externally scheduled operator.
        The operation is transactional, including a failed
        copy back to the external array. For GLE, the duration must match the
        transition prepared at construction. This operator accepts no barostat.
        """
        if self.barostat != "none":
            raise ValueError("a standalone thermostat cannot apply a barostat")
        duration = self.dt if dt is None else _scalar(dt, "thermostat duration", positive=True)
        clock = None if step is None else _integer(step, "step")
        coordinates = _array(q, (self.beads, self.natoms, 3))
        if (not isinstance(p, np.ndarray) or p.dtype != np.dtype(np.float64)
                or p.shape != coordinates.shape or not p.flags.writeable):
            raise ValueError("thermostat momenta must be a writable float64 array with shape [P,N,3]")
        if np.shares_memory(np.asarray(q), p):
            raise ValueError("thermostat coordinates and momenta must not overlap")
        momenta = _array(p, coordinates.shape)
        with self._call_lock:
            before = self._native.state(self._handle)
            try:
                raw = self._native.apply_thermostat(self._handle, coordinates, momenta, duration, clock)
                updated = np.frombuffer(raw, dtype=np.float64).reshape(p.shape)
                p[:] = updated
            except BaseException:
                self._native.restore(self._handle, before)
                raise
        return p

    def state(self) -> dict:
        """Return a detached, checksummed restart record, not mutable live arrays."""
        with self._call_lock:
            raw = self._native.state(self._handle)
        shapes = {"q": (self.beads, self.natoms, 3), "p": (self.beads, self.natoms, 3),
                  "aux": (self.beads, self.natoms, 3, max(0, self._cfg["gle_size"] - 1)),
                  "energy": (self.beads,), "force": (self.beads, self.natoms, 3),
                  "virial": (self.beads, 3, 3), "cell": (3, 3), "cell_p": (3, 3)}
        if self.thermostat == "nhc":
            shapes["aux"] = (2, self._cfg["nhc_length"])
        for key in _ARRAYS:
            raw[key] = np.frombuffer(raw[key], dtype=np.float64).reshape(shapes[key]).copy()
        raw.update(format=_FORMAT, fingerprint=self.fingerprint)
        raw["digest"] = _state_digest(raw)
        return raw

    def _rollback_token(self):
        """Capture an owned binary state without serializing a checkpoint."""
        with self._call_lock:
            return self._native.snapshot(self._handle)

    def _initial_energy(self, q, p, masses, beta):
        """Original NumPy energy reduction, with reusable transaction scratch."""
        with self._call_lock:
            if not hasattr(self, "_energy_work"):
                self._energy_work = (np.empty_like(q), np.empty_like(q))
            kinetic, spring = self._energy_work
            raw = self._native.initial_energy_terms(self._handle, q, p, masses, kinetic, spring)
            potential = np.frombuffer(raw, dtype=np.float64)
            return float(np.sum(potential) + np.sum(kinetic)
                         + .5*(self.beads/beta)**2*np.sum(spring))

    def _physical_state(self) -> dict:
        """Refresh reusable arrays for the internal propagation cache."""
        with self._call_lock:
            if not hasattr(self, "_physical_buffers"):
                shape = (self.beads, self.natoms, 3)
                self._physical_buffers = (np.empty(shape), np.empty(shape))
            q, p = self._physical_buffers
            raw = self._native.physical_state(self._handle, q, p)
        result = {"q": q, "p": p, "step": raw["step"]}
        if self._cfg["has_cell"]:
            result["cell"] = np.frombuffer(raw["cell"], dtype=np.float64).reshape(3, 3)
        return result

    def _restore_rollback_token(self, token) -> None:
        """Restore a token produced by :meth:`_rollback_token`."""
        with self._call_lock, self._lease():
            self._native.restore_snapshot(self._handle, token)
            if self._cfg["has_cell"]:
                raw = self._native.physical_state(self._handle)
                self._sync_bound_cell(np.frombuffer(raw["cell"], dtype=np.float64).reshape(3, 3))

    def restore(self, state: dict) -> None:
        """Restore only an intact checkpoint with exactly matching physics."""
        expected = set(_ARRAYS) | {"step", "force_evaluations", "rng", "format", "fingerprint", "digest"}
        if set(state) != expected or state.get("format") != _FORMAT:
            raise ValueError("invalid native checkpoint schema")
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("checkpoint physics/backend differs")
        if state["digest"] != _state_digest(state):
            raise ValueError("checkpoint checksum differs; use a new engine to change q/p")
        raw = dict(state)
        for key in _ARRAYS:
            raw[key] = _array(state[key]).tobytes()
        with self._call_lock, self._lease():
            before = (self._native.state(self._handle)
                      if self._python_binding is not None or isinstance(self._binding, NativePeriodicPES) else None)
            try:
                self._native.restore(self._handle, raw)
                self._sync_bound_cell(state["cell"])
            except BaseException:
                if before is not None:
                    self._native.restore(self._handle, before)
                raise

    def save(self, path: str | Path) -> None:
        """Save the common atomic NPZ container with the native v1 payload.

        The numerical state, RNG, native checksum and fingerprint are unchanged.
        Array data remain uncompressed, as in pre-container native checkpoints.
        """
        state = self.state()
        metadata = {key: value for key, value in state.items() if key not in _ARRAYS}
        atomic_savez_uncompressed(
            path, kind=_FORMAT, metadata=json.dumps(metadata, sort_keys=True),
            **{key: state[key] for key in _ARRAYS},
        )

    def load(self, path: str | Path) -> None:
        """Load current containers or an unchanged pre-container native NPZ."""
        with load_checkpoint(path, expected_kind=_FORMAT) as saved:
            if set(saved.files) != set(_ARRAYS) | {"metadata"}:
                raise ValueError("invalid native checkpoint files")
            state = json.loads(str(saved["metadata"]))
            state.update({key: saved[key].copy() for key in _ARRAYS})
        self.restore(state)


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key in _ARRAYS:
        a = np.ascontiguousarray(state[key], dtype=np.float64)
        digest.update(key.encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    metadata = {key: value for key, value in state.items() if key not in _ARRAYS and key != "digest"}
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


def ensemble_from_config(config: dict, *, base_directory: str | Path = ".",
                         potential_override=None, potential_identity: str | None = None) -> NativeEnsemble:
    """Build the explicit native driver from a unit-aware JSON/TOML mapping.

    This schema does not masquerade as a legacy ``task=pimd`` input. Unknown
    fields and Python-only potentials are errors rather than silent fallbacks.
    """
    from .constants import KB_AU
    from .units import parse_quantity, unit_factor

    spec = dict(config)
    _unknown(spec, {"schema_version", "positions", "q", "masses", "beads", "temperature",
                    "beta", "timestep", "integrator", "sampling_integrator", "cell", "geometry_unit", "cell_unit",
                    "pbc", "fixed_atoms", "seed", "p", "potential", "thermostat", "barostat", "run"}, "native input")
    if spec.get("schema_version", "native-ensembles/1") != "native-ensembles/1":
        raise ValueError("expected schema_version='native-ensembles/1'")
    if ("temperature" in spec) == ("beta" in spec):
        raise ValueError("supply exactly one of temperature or beta")
    if "temperature" in spec:
        temperature = parse_quantity(spec["temperature"], "temperature")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        beta = 1/(KB_AU*temperature)
    else:
        beta = parse_quantity(spec["beta"], (0, -1, 0, 0, 0, 0))
    beads = _integer(spec.get("beads", 1), "beads", 1)
    masses = [parse_quantity(value, "mass") for value in spec["masses"]]
    length = unit_factor(spec.get("geometry_unit", "bohr"), "length")
    if ("positions" in spec) == ("q" in spec):
        raise ValueError("supply exactly one of positions or q")
    q = (_array(spec["q"], (beads, len(masses), 3)) if "q" in spec else
         np.repeat(_array(spec["positions"], (len(masses), 3))[None], beads, axis=0))*length
    cell = None
    if "cell" in spec:
        cell = _array(spec["cell"], (3, 3))*unit_factor(spec.get("cell_unit", spec.get("geometry_unit", "bohr")), "length")
    raw_t = spec.get("thermostat", "pile-l")
    t = {"mode": raw_t} if isinstance(raw_t, str) else copy_mapping(raw_t)
    for key in ("gamma", "gamma_centroid"):
        if key in t:
            t[key] = parse_quantity(t[key], "frequency")
    for key in ("tau", "tdamp"):
        if key in t:
            t[key] = parse_quantity(t[key], "time")
    if "gle" in t:
        g = t["gle"]
        if isinstance(g, dict) and ("generator" in g or "file" in g):
            from .gle_parameters import resolve_gle_parameters
            g = resolve_gle_parameters(g, mode=t.get("mode", "pile-l"),
                                       base_directory=base_directory)
            t["gle"] = g
        factor = unit_factor(g.pop("A_unit", "autime^-1"), "frequency")
        g["A"] = _array(g["A"])*factor
        if "beta" in g:
            g["beta"] = parse_quantity(g["beta"], (0, -1, 0, 0, 0, 0))
    raw_b = spec.get("barostat", "none")
    b = {"mode": raw_b} if isinstance(raw_b, str) else copy_mapping(raw_b)
    for key, dimension in (("pressure", "pressure"), ("tau", "time"), ("gamma", "frequency"), ("compressibility", "inverse_pressure")):
        if key in b:
            b[key] = parse_quantity(b[key], dimension)
    if "stress" in b:
        b["stress"] = _array([[parse_quantity(value, "pressure") for value in row] for row in b["stress"]], (3, 3))
    if "reference_cell" in b:
        b["reference_cell"] = _array(b["reference_cell"], (3, 3))*unit_factor(
            b.pop("reference_cell_unit", spec.get("cell_unit", spec.get("geometry_unit", "bohr"))), "length")
    potential = copy_mapping(spec.get("potential", {"kind": "free"}))
    if potential.get("kind") == "native_periodic":
        _unknown(potential, {"kind", "library", "symbol"}, "native periodic potential")
        potential = NativePeriodicPES(Path(base_directory)/potential["library"], potential["symbol"], len(masses))
    else:
        if "k" in potential:
            potential["k"] = parse_quantity(potential["k"], "spring")
        if "heights" in potential:
            potential["heights"] = [parse_quantity(value, "energy") for value in potential["heights"]]
    if potential_override is not None:
        potential = potential_override
    return NativeEnsemble(q, masses, beta, parse_quantity(spec["timestep"], "time"),
                          p=spec.get("p"), potential=potential, potential_identity=potential_identity,
                          thermostat=t, barostat=b,
                          integrator=spec.get("integrator", "cayley-bcb"), cell=cell,
                          sampling_integrator=spec.get("sampling_integrator"),
                          pbc=spec.get("pbc"), fixed_atoms=spec.get("fixed_atoms", ()), seed=spec.get("seed", 0))


def copy_mapping(value: dict) -> dict:
    import copy
    if not isinstance(value, dict):
        raise ValueError("expected a configuration table")
    return copy.deepcopy(value)


def main(argv=None) -> int:
    """``pyrpmd-native input.toml``: one native call for the requested trajectory."""
    import argparse
    import tomllib

    from .units import time_steps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as source:
            config = json.load(source) if args.input.suffix.lower() == ".json" else tomllib.load(source)
        run = dict(config.get("run", {}))
        _unknown(run, {"steps", "duration", "stride", "interval", "record_centroids", "output"}, "run")
        engine = ensemble_from_config(config, base_directory=args.input.resolve().parent)
        if args.resume is not None:
            engine.load(args.resume)
        if ("steps" in run) == ("duration" in run):
            raise ValueError("run needs exactly one of steps or duration")
        steps = _integer(run["steps"], "steps") if "steps" in run else time_steps(run["duration"], engine.dt, allow_zero=True)
        if "stride" in run and "interval" in run:
            raise ValueError("run accepts stride or interval, not both")
        stride = _integer(run.get("stride", 1), "stride", 1) if "interval" not in run else time_steps(run["interval"], engine.dt)
        output = args.output or Path(run.get("output", "runs/native-ensemble"))
        output.mkdir(parents=True, exist_ok=True)
        start = engine.state()["step"]
        trajectory = output/f"trajectory-{start:012d}-{start+steps:012d}.npy"
        result = engine.run(steps, stride=stride, record_centroids=run.get("record_centroids", False), trajectory=trajectory)
        checkpoint = output/"checkpoint.npz"
        engine.save(checkpoint)
        summary = {key: value for key, value in result.items() if key not in {"samples", "centroids", "velocities", "radii_gyration_squared"}}
        summary.update(frames=len(result["samples"]), fingerprint=engine.fingerprint,
                       checkpoint=str(checkpoint), physical_converged=False,
                       units={"length": "bohr", "time": "atomic", "energy": "hartree", "pressure": "hartree/bohr^3"})
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, NotImplementedError) as exc:
        parser.exit(2, f"pyrpmd-native: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
