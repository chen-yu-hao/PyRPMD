"""Experimental isolated, collinear j=0 diatomic switching diagnostic.

The physical-potential kicks and exact free-ring propagation are composed as
BAB, entirely in a standalone C++ loop.  The production RPMD kernel is not
modified.  A quintic schedule changes a harmonic well into the asymptotic
BKMP2 diatomic potential, followed optionally by NVE evolution.  There is no
thermostat, momentum resampling, or post-switch energy retuning.

The phase arrays are bond lengths and canonical *relative* momenta, in atomic
units.  Atomic masses (one common mass or a pair) are supplied to the class;
the native dynamics use their reduced mass.  beta is the ring-ensemble
parameter in inverse hartree, not a collision temperature.  This construction
does not establish an excited quantum pure state or state populations.
"""
from __future__ import annotations

from contextlib import nullcontext
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

try:
    import fcntl
except ImportError:  # Import remains possible on platforms without this POSIX experiment.
    fcntl = None

import numpy as np

from pyrpmd.constants import FS_TO_AU

HERE = Path(__file__).resolve().parent
CPP_PATH = HERE / "diatomic_switch.cpp"
FFT_PATH = HERE.parent.parent / "native" / "ring_fft.h"
_FLAGS = ["-std=c++17", "-O3", "-fPIC", "-shared"]
_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_TRACE_FIELDS = (
    "time_fs", "lambda", "kinetic_per_bead_hartree",
    "harmonic_potential_per_bead_hartree", "full_potential_per_bead_hartree",
    "spring_per_bead_hartree", "ring_hamiltonian_per_bead_hartree",
    "switching_work_per_bead_hartree", "energy_balance_residual_per_bead_hartree",
    "primitive_internal_per_bead_hartree", "centroid_bond_bohr",
    "centroid_radial_momentum_au", "bond_spread_bohr", "minimum_bond_bohr",
    "maximum_bond_bohr", "centroid_harmonic_energy_hartree",
    "lambda_rate_per_atomic_time", "hold_energy_drift_per_bead_hartree",
)
_HOLD_FIELDS = (
    "primitive_internal_per_bead_hartree", "kinetic_per_bead_hartree",
    "full_potential_per_bead_hartree", "spring_per_bead_hartree",
    "centroid_harmonic_energy_hartree", "bond_spread_bohr",
)


class _Options(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_int64) for name in
        ("beads", "switch_steps", "hold_steps", "trace_stride")]
        + [(name, ctypes.c_double) for name in
        ("dt", "beta", "mass", "omega", "equilibrium", "energy_zero", "asymptote",
         "target_omega", "target_equilibrium", "target_offset")]
        + [("target_kind", ctypes.c_int32)])


class _Output(ctypes.Structure):
    _fields_ = [("completed_steps", ctypes.c_int64), ("trace_count", ctypes.c_int64),
        ("status", ctypes.c_int32)] + [(name, ctypes.c_double) for name in
        ("initial_harmonic_energy", "final_energy", "work", "max_abs_residual",
         "max_abs_hold_drift", "hold_initial_energy", "min_bond", "max_bond", "hold_duration")]
    _fields_ += [("hold_integrals", ctypes.c_double * len(_HOLD_FIELDS))]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _real_array(value, name):
    """Reject complex, boolean and non-numeric data before float conversion."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _finite(value, name, *, positive=False, nonnegative=False):
    raw = _real_array(value, name)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    number = float(raw)
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def _steps(duration, dt, name, *, positive=False):
    value = _finite(duration, name, positive=positive, nonnegative=True)
    ratio = value / dt
    if not math.isfinite(ratio) or ratio > 2**52:
        raise ValueError(f"{name} contains too many steps")
    count = round(ratio)
    if (positive and count < 1) or abs(ratio-count) > max(1e-9, 8*np.finfo(float).eps*abs(ratio)):
        raise ValueError(f"{name} must be an integer multiple of dt_fs")
    return count


def _build_library():
    """Build a content-addressed, standalone library, including compiler identity."""
    if fcntl is None:
        raise RuntimeError("The experimental standalone diatomic switch currently requires a POSIX build environment")
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError("A C++17 compiler is required for the experimental diatomic switch")
    command[0] = str(Path(shutil.which(command[0])).resolve())
    version = subprocess.run(command+["--version"], check=True, text=True,
                             capture_output=True).stdout.strip()
    sources = {str(CPP_PATH): _sha(CPP_PATH), str(FFT_PATH): _sha(FFT_PATH)}
    specification = {"schema": "pyrpmd.experimental.diatomic-switch-build.v1",
        "sources": sources, "compiler": command, "compiler_version": version,
        "compiler_flags": _FLAGS, "system": platform.system(), "machine": platform.machine()}
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    directory = HERE / "build"
    directory.mkdir(exist_ok=True)
    library = directory / f"libdiatomic_switch_{key[:20]}.so"
    manifest = library.with_suffix(".build.json")
    with (directory / ".diatomic-switch-build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists() or not manifest.exists():
            with tempfile.TemporaryDirectory(prefix="diatomic-switch-", dir=directory) as temporary:
                destination = Path(temporary) / "kernel.so"
                compiled = subprocess.run(command+_FLAGS+[str(CPP_PATH), "-o", str(destination)],
                                          text=True, capture_output=True)
                if compiled.returncode:
                    raise RuntimeError(f"Diatomic switch compilation failed:\n{compiled.stderr}")
                if any(_sha(path) != digest for path, digest in sources.items()):
                    raise RuntimeError("Diatomic switch source changed during compilation")
                metadata = {**specification, "build_key": key,
                    "library_path": str(library), "library_sha256": _sha(destination)}
                temp_manifest = Path(temporary) / "build.json"
                temp_manifest.write_text(json.dumps(metadata, indent=2)+"\n")
                os.replace(destination, library)
                os.replace(temp_manifest, manifest)
        metadata = json.loads(manifest.read_text())
        if metadata.get("build_key") != key or metadata.get("library_sha256") != _sha(library):
            raise RuntimeError("Diatomic switch binary/build provenance checksum mismatch")
    return library, metadata


@dataclass
class DiatomicSwitchResult:
    r: np.ndarray
    p: np.ndarray
    trace: list[dict]
    summary: dict


class DiatomicSwitchError(RuntimeError):
    """Explicit failed preparation, retaining the last accepted phase point."""
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class DiatomicSwitcher:
    """Prepare an isolated collinear bond; does not evolve an incoming atom.

    ``potential`` must be the benchmark's unmodified BKMP2Potential.  A
    ``None`` potential plus ``target_omega`` supplies an analytic harmonic
    target for validation only.  Both wells share ``energy_zero_hartree``;
    for BKMP2 it is determined from the full PES at the supplied equilibrium.
    ``target_offset`` is an optional known constant for harmonic oracle tests.

    Construction compiles/loads the native kernel and exposes ``metadata``
    and ``library_path`` so callers can freeze all inputs before workers run.
    """
    def __init__(self, potential, *, mass, omega, equilibrium, asymptote_bohr=99.,
                 target_omega=None, target_equilibrium=None, target_offset=0.,
                 energy_zero_hartree=0.):
        masses = _real_array(mass, "mass")
        if masses.ndim == 0:
            masses = np.repeat(masses, 2)
        if masses.shape != (2,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("mass must be one positive atomic mass or a positive pair in electron masses")
        self.masses = masses.copy()
        self.reduced_mass = float(masses[0]*masses[1]/masses.sum())
        if not math.isfinite(self.reduced_mass) or self.reduced_mass <= 0:
            raise ValueError("Invalid reduced mass")
        self.omega = _finite(omega, "omega", positive=True)
        self.equilibrium = _finite(equilibrium, "equilibrium", positive=True)
        self.asymptote_bohr = _finite(asymptote_bohr, "asymptote_bohr", positive=True)
        self.potential = potential
        self._radial_address = None
        self._lock = nullcontext()
        asymptote_checks = None
        if potential is None:
            if target_omega is None:
                raise ValueError("An analytic harmonic target requires target_omega")
            self.target_kind = 1
            self.target_omega = _finite(target_omega, "target_omega", positive=True)
            self.target_equilibrium = _finite(
                self.equilibrium if target_equilibrium is None else target_equilibrium,
                "target_equilibrium", positive=True)
            self.target_offset = _finite(target_offset, "target_offset")
            self.energy_zero_hartree = _finite(energy_zero_hartree, "energy_zero_hartree")
            target_metadata = {"kind": "analytic_harmonic_validation_target",
                "omega_au": self.target_omega, "equilibrium_bohr": self.target_equilibrium,
                "offset_hartree": self.target_offset}
        else:
            from benchmarks.h3.pes import BKMP2Potential, _BKMP2_RADIAL
            if type(potential) is not BKMP2Potential:
                raise TypeError("Native diatomic switch requires the unmodified BKMP2Potential")
            if ("radial" in potential.__dict__
                    or BKMP2Potential.radial is not _BKMP2_RADIAL):
                raise TypeError("Native diatomic switch cannot bypass an overridden radial method")
            potential.native_binding()  # Do not bypass overridden provider semantics.
            if target_omega is not None or target_equilibrium is not None or target_offset != 0. or energy_zero_hartree != 0.:
                raise ValueError("Harmonic-target options cannot override a BKMP2 target")
            self.target_kind = 0
            self.target_omega = self.omega
            self.target_equilibrium = self.equilibrium
            self.target_offset = 0.
            self._radial_address = ctypes.cast(potential._radial, ctypes.c_void_p).value
            self._lock = potential._lock
            self.energy_zero_hartree = potential.radial(
                [self.asymptote_bohr, self.equilibrium, self.asymptote_bohr])[0]
            farther = max(150., 1.5*self.asymptote_bohr)
            asymptote_checks = []
            for bond in (.8, 1., self.equilibrium, 2., 3.):
                near_e, near_g = potential.radial([self.asymptote_bohr, bond, self.asymptote_bohr])
                far_e, far_g = potential.radial([farther, bond, farther])
                asymptote_checks.append({"bond_bohr": bond,
                    "near_asymptote_bohr": self.asymptote_bohr, "far_asymptote_bohr": farther,
                    "energy_difference_hartree": near_e-far_e,
                    "bond_gradient_difference_hartree_per_bohr": float(near_g[1]-far_g[1])})
            target_metadata = {"kind": "BKMP2_asymptotic_diatom",
                "distances": "[asymptote_bohr, bond_bohr, asymptote_bohr]",
                "bond_gradient_component": 1, **potential.metadata}
        self.library_path, build_metadata = _build_library()
        self._library = ctypes.CDLL(str(self.library_path))
        self._run = self._library.pyrpmd_diatomic_switch_v1
        self._run.argtypes = [ctypes.POINTER(_Options), ctypes.c_void_p,
            _ARRAY, _ARRAY, _ARRAY, _ARRAY, _ARRAY, ctypes.c_int64,
            ctypes.POINTER(_Output), ctypes.c_char_p, ctypes.c_size_t]
        self._run.restype = ctypes.c_int
        self.metadata = {**build_metadata, "cpp_sha256": _sha(CPP_PATH),
            "python_sha256": _sha(__file__), "ring_fft_header_path": str(FFT_PATH),
            "ring_fft_header_sha256": _sha(FFT_PATH), "target": target_metadata,
            "asymptote_checks": asymptote_checks,
            "energy_zero_hartree": self.energy_zero_hartree,
            "energy_zero_definition": "BKMP2 at supplied equilibrium, shared by harmonic well" if potential else "specified common harmonic offset",
            "integrator": "physical half kick / exact free-ring step / physical half kick",
            "schedule": "quintic 10s^3-15s^4+6s^5 with endpoint-time forces",
            "work_quadrature": "trapezoidal integral of lambda_dot * mean(Vfull-Vharm)",
            "scope": "Experimental isolated collinear j=0 preparation; no quantum pure-state or state-population claim"}

    def run(self, r, p, *, beta, dt_fs=.02, switch_time_fs=50., hold_time_fs=0.,
            trace_stride_fs=.2, raise_on_error=True):
        """Return copied endpoint arrays, trace, and explicitly separated work/drift.

        Durations and trace stride must be integer multiples of ``dt_fs``.
        A zero switching time gives a sudden jump at t=0, including its work.
        At every endpoint the full PES is checked, including at lambda=0.
        A failed endpoint is rejected and the last accepted state is retained.

        ``primitive_internal_per_bead_hartree`` always uses the full PES:
        K + Vfull - energy_zero - S.  During switching it is a full-reference
        diagnostic, not H_lambda/P.  Hold averages use a trapezoidal time
        integral at *every* accepted native step, independently of trace stride.
        Ensemble uncertainty must be computed across trajectory means, not
        by treating correlated time points as independent samples.
        """
        r = np.array(_real_array(r, "r"), dtype=np.float64, order="C", copy=True)
        p = np.array(_real_array(p, "p"), dtype=np.float64, order="C", copy=True)
        if r.ndim != 1 or not 1 <= len(r) <= 1000000 or p.shape != r.shape:
            raise ValueError("r and p must be matching nonempty one-dimensional phase arrays")
        if not np.isfinite(r).all() or not np.isfinite(p).all():
            raise ValueError("Input phase arrays must be finite")
        beta = _finite(beta, "beta", positive=True)
        dt = _finite(dt_fs, "dt_fs", positive=True)
        switching = _steps(switch_time_fs, dt, "switch_time_fs")
        holding = _steps(hold_time_fs, dt, "hold_time_fs")
        stride = _steps(trace_stride_fs, dt, "trace_stride_fs", positive=True)
        total = switching+holding
        # Extra capacity holds the last accepted endpoint on a failure.
        capacity = total//stride+4
        if capacity > 10000000:
            raise ValueError("Requested trace is too large; increase trace_stride_fs")
        options = _Options(len(r), switching, holding, stride, dt*FS_TO_AU,
            beta, self.reduced_mass, self.omega, self.equilibrium,
            self.energy_zero_hartree, self.asymptote_bohr, self.target_omega,
            self.target_equilibrium, self.target_offset, self.target_kind)
        r_out, p_out = r.copy(), p.copy()
        raw_trace = np.empty((capacity, len(_TRACE_FIELDS)), dtype=np.float64)
        output = _Output()
        error = ctypes.create_string_buffer(1024)
        with self._lock:
            status = self._run(ctypes.byref(options), self._radial_address,
                r, p, r_out, p_out, raw_trace, capacity, ctypes.byref(output), error, len(error))
        trace = []
        for raw in raw_trace[:output.trace_count]:
            row = {name: (float(value) if math.isfinite(value) else None)
                   for name, value in zip(_TRACE_FIELDS, raw)}
            row["time_fs"] /= FS_TO_AU
            trace.append(row)
        clean = lambda value: float(value) if math.isfinite(value) else None
        hold_average = {name: clean(value/output.hold_duration) if output.hold_duration > 0 else None
                        for name, value in zip(_HOLD_FIELDS, output.hold_integrals)}
        summary = {"schema": "pyrpmd.experimental.diatomic-switch.v1",
            "status": "completed" if status == 0 else "failed", "native_status": status,
            "error": error.value.decode("utf-8", errors="replace") if status else None,
            "beads": len(r), "atomic_masses_au": self.masses.tolist(),
            "reduced_mass_au": self.reduced_mass, "beta_hartree_inverse": beta,
            "dt_fs": dt, "switch_time_fs": switching*dt, "hold_time_fs": holding*dt,
            "trace_stride_fs": stride*dt, "completed_steps": output.completed_steps,
            "completed_time_fs": output.completed_steps*dt,
            "omega_au": self.omega, "equilibrium_bohr": self.equilibrium,
            "energy_zero_hartree": self.energy_zero_hartree, "Vmin": self.energy_zero_hartree,
            "initial_harmonic_energy_per_bead_hartree": clean(output.initial_harmonic_energy),
            "final_ring_hamiltonian_per_bead_hartree": clean(output.final_energy),
            "switching_work_per_bead_hartree": clean(output.work) if trace else None,
            "maximum_absolute_energy_balance_residual_per_bead_hartree": clean(output.max_abs_residual) if trace else None,
            "final_energy_balance_residual_per_bead_hartree": trace[-1]["energy_balance_residual_per_bead_hartree"] if trace else None,
            "hold_initial_energy_per_bead_hartree": clean(output.hold_initial_energy),
            "maximum_absolute_hold_energy_drift_per_bead_hartree": clean(output.max_abs_hold_drift) if output.completed_steps >= switching and trace else None,
            "minimum_bond_bohr": clean(output.min_bond), "maximum_bond_bohr": clean(output.max_bond),
            "hold_time_average": hold_average,
            "accepted_hold_duration_fs": output.hold_duration/FS_TO_AU,
            "hold_averaging": "Trapezoidal time average of every accepted native step; compute ensemble SEM over independent trajectory means",
            "primitive_diagnostic": "K + Vfull - Vmin - S per bead, using full target reference even during the ramp; not the driven Hamiltonian",
            "sudden_switch": switching == 0,
            "work_includes_initial_jump": switching == 0,
            "initial_jump_work_per_bead_hartree": (trace[0]["switching_work_per_bead_hartree"] if trace else None) if switching == 0 else 0.,
            "cpp_sha256": self.metadata["cpp_sha256"], "python_sha256": self.metadata["python_sha256"],
            "library_sha256": self.metadata["library_sha256"],
            "target": self.metadata["target"],
            "scope": self.metadata["scope"]}
        result = DiatomicSwitchResult(r_out, p_out, trace, summary)
        if status and raise_on_error:
            raise DiatomicSwitchError(summary["error"], result)
        return result
