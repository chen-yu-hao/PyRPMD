"""Persistent C++ NVT/NPT trajectories; Python runs only at block boundaries.

This is an explicit native API, not a silent replacement for existing RPMD
reaction workflows. All numeric inputs are atomic units. Only immutable native
PES evaluators are accepted; ASE/Torch/Python callables cannot be made native by
wrapping their step loop. See docs/native-ensembles.md for equations and limits.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import ctypes
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import numpy as np

_THERMOSTATS = {name: i for i, name in enumerate((
    "none", "andersen", "langevin", "pile-l", "pile-g", "svr", "gle",
    "nm-gle", "nm-gle-g", "piglet", "trpmd", "ffl",
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


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or value > 2**64 - 1:
        raise ValueError(f"{name} is out of range")
    return value


def _scalar(value: Any, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number in atomic units")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
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


def _gle_matrices(spec: dict, mode: str, beads: int, beta: float,
                  dt: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """One-time matrix preparation; no Python linear algebra during propagation."""
    from scipy.linalg import expm

    _unknown(spec, {"A", "C", "beta", "beads", "provenance"}, "GLE")
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
            yield ("thermal-v1", self.address, self.context, self.natoms)


class NativeEnsemble:
    """A persistent C++ trajectory with immutable physics configuration.

    q: [P,N,3] bohr; masses: [N] electron masses; beta: hartree^-1;
    dt: atomic time; p: optional [P,N,3] atomic momenta. Omitted p is drawn in
    C++, including the joint GLE auxiliary distribution. Thermostat/barostat
    parameters use inverse atomic time, atomic time and hartree/bohr^3.
    """
    def __init__(self, q, masses, beta: float, dt: float, *, p=None,
                 potential=None, potential_identity: str | None = None,
                 thermostat: str | dict = "pile-l", barostat: str | dict = "none",
                 integrator: str = "cayley-bcb", cell=None, pbc=None,
                 fixed_atoms=(), seed: int = 0):
        try:
            from . import _thermal
        except ImportError as exc:
            raise RuntimeError("native ensemble extension is unavailable; rebuild PyRPMD with C++ enabled") from exc
        self._native = _thermal
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
        t = {"mode": thermostat} if isinstance(thermostat, str) else dict(thermostat)
        _unknown(t, {"mode", "gamma", "gamma_centroid", "pile_scale", "tau", "interval", "gle"}, "thermostat")
        mode = t.get("mode", "pile-l")
        if mode not in _THERMOSTATS:
            raise ValueError(f"unsupported thermostat {mode!r}")
        thermostat_fields = {
            "none": set(), "andersen": {"interval"}, "langevin": {"gamma"},
            "pile-l": {"gamma_centroid", "pile_scale"}, "pile-g": {"tau", "pile_scale"},
            "svr": {"tau"}, "gle": {"gle"}, "nm-gle": {"gle"},
            "nm-gle-g": {"gle", "tau"}, "piglet": {"gle"},
            "trpmd": {"pile_scale", "gamma_centroid"}, "ffl": {"gamma"},
        }
        _unknown(t, {"mode"} | thermostat_fields[mode], f"{mode} thermostat")
        b = {"mode": barostat} if isinstance(barostat, str) else dict(barostat)
        _unknown(b, {"mode", "pressure", "tau", "gamma", "compressibility", "cell_mask", "reference_cell", "stress"}, "barostat")
        bmode = b.get("mode", "none")
        aliases = {"isotropic": "bzp", "flexible": "mttk", "mtk": "mttk", "anisotropic": "rgb", "stochastic-rescaling": "scr"}
        bmode = aliases.get(bmode, bmode)
        if bmode in ("sc-isotropic", "suzuki-chin"):
            raise NotImplementedError("Suzuki-Chin NPT requires the high-order action and its force/virial derivatives; primitive PIMD is not a substitute")
        if bmode not in _BAROSTATS:
            raise ValueError(f"unsupported barostat {bmode!r}")
        barostat_fields = {
            "none": set(), "bzp": {"pressure", "tau", "gamma"},
            "mttk": {"pressure", "tau", "gamma", "cell_mask"},
            "rgb": {"stress", "reference_cell", "tau", "gamma", "cell_mask"},
            "scr": {"pressure", "tau", "compressibility"},
        }
        _unknown(b, {"mode"} | barostat_fields[bmode], f"{bmode} barostat")
        if bmode != "none" and "pressure" not in b and bmode != "rgb":
            raise ValueError("NPT requires an explicit external pressure")
        if bmode == "rgb" and ("stress" not in b or "reference_cell" not in b):
            raise ValueError("RGB requires explicit stress and reference_cell")
        cell_mask = _array(b.get("cell_mask", [[1, 1, 1], [0, 1, 1], [0, 0, 1]]), (3, 3))
        external = _array(b.get("stress", np.zeros((3, 3))), (3, 3))
        if not np.allclose(external, external.T, atol=1e-12, rtol=1e-12):
            raise ValueError("external stress must be symmetric")
        cfg = {
            "atoms": self.natoms, "beads": self.beads, "beta": beta, "dt": dt,
            "masses": mass, "fixed": fixed,
            "thermostat": _THERMOSTATS[mode], "barostat": _BAROSTATS[bmode],
            "cayley": integrator == "cayley-bcb", "initialize_momenta": p is None,
            "has_cell": cell is not None, "pbc": _array(flags),
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
        if mode in ("gle", "nm-gle", "nm-gle-g", "piglet"):
            if "gle" not in t:
                raise ValueError("GLE requires an explicit drift matrix")
            if mode == "piglet" and bmode != "none":
                raise ValueError("PIGLET NPT is not validated by this implementation")
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
        self.splitting = ("BCOCB" if cfg["cayley"] else "BAOAB") if bmode == "none" else "OBABO"
        self._binding = None
        self._potential = potential
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
            else:
                from .potentials.native import descriptor_binding
                binding = potential.native_binding() if callable(getattr(potential, "native_binding", None)) else potential
                self._binding = descriptor_binding(binding)
                if not isinstance(potential_identity, str) or not potential_identity.strip():
                    raise ValueError("native provider requires potential_identity for safe restart")
                self._potential_identity = {"kind": self._binding.kind, "model": potential_identity}
                cfg["potential"] = {"cabi-v1": 4, "fortran-batch-v1": 5}[self._binding.kind]
        self._cfg = cfg
        with self._lease() as descriptor:
            if descriptor is not None:
                _, address, context, natoms = descriptor
                if natoms != self.natoms:
                    raise ValueError("native PES atom count mismatch")
                cfg["address"], cfg["context"] = int(address), int(context)
            self._handle = _thermal.create(cfg, coordinates, mom, h.ravel(), self._binding)
        # Addresses and initialization draws are not physics. All model, units,
        # fixed positions, integrator and stochastic matrices are fingerprinted.
        physics = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                   for key, value in cfg.items()
                   if key not in {"address", "context", "initialize_momenta", "seed"}}
        physics.update(potential_identity=self._potential_identity, initial_cell=h.tolist(),
                       fixed_positions=coordinates[0, fixed.astype(bool)].tolist(),
                       rng_abi=getattr(_thermal, "rng_abi", "std-cxx17-local-build-v1"), format=_FORMAT)
        self.fingerprint = hashlib.sha256(json.dumps(physics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self._call_lock = threading.RLock()

    def _lease(self):
        return nullcontext(None) if self._binding is None else self._binding.acquire()

    def run(self, steps: int, *, stride: int = 1, record_centroids: bool = False,
            trajectory: str | Path | None = None) -> dict:
        """Advance one complete C++ block; observation times use the absolute clock.

        A native/PES failure restores the complete pre-block state and RNG.
        Unsupported Python potentials are rejected; no fallback is attempted.
        No final frame is inserted off the stride clock, so concatenated blocks
        have exactly the same samples as a single call.
        """
        steps = _integer(steps, "steps")
        stride = _integer(stride, "stride", 1)
        if not isinstance(record_centroids, bool):
            raise ValueError("record_centroids must be boolean")
        temporary = None
        path = None if trajectory is None else Path(trajectory)
        if path is not None:
            if path.exists():
                raise FileExistsError(f"trajectory already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            os.close(fd)
        try:
            with self._call_lock, self._lease():
                result = self._native.run(self._handle, steps, stride, record_centroids, temporary or "")
            frames = result["frames"]
            if path is None:
                samples = np.frombuffer(result["samples"], dtype=np.float64).reshape(frames, len(_COLUMNS))
                centroids = (np.frombuffer(result["centroids"], dtype=np.float64).reshape(frames, self.natoms, 3)
                             if record_centroids else None)
            else:
                # The C++ loop writes NPY records directly, with O(P*N) state
                # memory. This is a memory map, not a second trajectory copy.
                # Exclusive publication avoids clobbering a file created by
                # another process during a long trajectory.
                os.link(temporary, path)
                os.unlink(temporary)
                temporary = None
                data = np.load(path, mmap_mode="r" if frames else None, allow_pickle=False)
                samples = data[:, :len(_COLUMNS)]
                centroids = data[:, len(_COLUMNS):].reshape(frames, self.natoms, 3) if record_centroids else None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return {"samples": samples, "columns": _COLUMNS, "centroids": centroids,
                "step": result["step"], "force_evaluations": result["force_evaluations"],
                "backend": "native-cxx17", "python_step_callbacks": 0,
                "thermostat": self.thermostat, "barostat": self.barostat,
                "splitting": self.splitting, "trajectory": None if path is None else str(path)}

    def state(self) -> dict:
        """Return a detached, checksummed restart record, not mutable live arrays."""
        with self._call_lock:
            raw = self._native.state(self._handle)
        shapes = {"q": (self.beads, self.natoms, 3), "p": (self.beads, self.natoms, 3),
                  "aux": (self.beads, self.natoms, 3, max(0, self._cfg["gle_size"] - 1)),
                  "energy": (self.beads,), "force": (self.beads, self.natoms, 3),
                  "virial": (self.beads, 3, 3), "cell": (3, 3), "cell_p": (3, 3)}
        for key in _ARRAYS:
            raw[key] = np.frombuffer(raw[key], dtype=np.float64).reshape(shapes[key]).copy()
        raw.update(format=_FORMAT, fingerprint=self.fingerprint)
        raw["digest"] = _state_digest(raw)
        return raw

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
            self._native.restore(self._handle, raw)

    def save(self, path: str | Path) -> None:
        """Atomic NPZ checkpoint; contains no pickle or executable objects."""
        state = self.state()
        metadata = {key: value for key, value in state.items() if key not in _ARRAYS}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                np.savez(output, metadata=json.dumps(metadata, sort_keys=True),
                         **{key: state[key] for key in _ARRAYS})
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as saved:
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


def ensemble_from_config(config: dict, *, base_directory: str | Path = ".") -> NativeEnsemble:
    """Build the explicit native driver from a unit-aware JSON/TOML mapping.

    This schema does not masquerade as a legacy ``task=pimd`` input. Unknown
    fields and Python-only potentials are errors rather than silent fallbacks.
    """
    from .constants import KB_AU
    from .units import parse_quantity, unit_factor

    spec = dict(config)
    _unknown(spec, {"schema_version", "positions", "q", "masses", "beads", "temperature",
                    "beta", "timestep", "integrator", "cell", "geometry_unit", "cell_unit",
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
    if "tau" in t:
        t["tau"] = parse_quantity(t["tau"], "time")
    if "gle" in t:
        g = t["gle"]
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
    return NativeEnsemble(q, masses, beta, parse_quantity(spec["timestep"], "time"),
                          p=spec.get("p"), potential=potential, thermostat=t, barostat=b,
                          integrator=spec.get("integrator", "cayley-bcb"), cell=cell,
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
        summary = {key: value for key, value in result.items() if key not in {"samples", "centroids"}}
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
