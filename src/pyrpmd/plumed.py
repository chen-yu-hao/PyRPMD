"""Optional PLUMED centroid-bias bridge with an explicit update clock.

PLUMED evaluates a physical bias W(q_centroid).  The sum-beads Hamiltonian
receives P*W, so *each* bead receives PLUMED's centroid force.  A PLUMED
``update`` occurs once after an accepted complete MD step, never on a force
or diagnostic evaluation.  No PLUMED import is required to import PyRPMD.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import threading

import numpy as np

from .constants import AMU_TO_AU, ATOMIC_TIME_S, BOHR_M, KCAL_MOL_TO_AU
from .dynamics import Dynamics, State
from .geometry import validate_cell_pbc


@dataclass(frozen=True)
class PlumedBiasResult:
    """Owned bias outputs, in atomic units.

    ``energy`` and ``forces`` belong to H_P. ``physical_energy`` belongs to
    W. ``plumed_virial`` is the *unscaled*, raw PLUMED virial convention;
    it is a diagnostic, not a PyRPMD pressure estimator.
    """

    energy: float
    forces: np.ndarray
    physical_energy: float
    centroid_forces: np.ndarray
    plumed_virial: np.ndarray
    step: int


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    # setStep is PLUMED's portable C-int command (also on Windows).
    if value > np.iinfo(np.intc).max:
        raise ValueError(f"{name} exceeds the PLUMED setStep integer range")
    return int(value)


class PlumedBias:
    """One optional PLUMED kernel for one fixed-cell centroid trajectory.

    Parameters are PyRPMD atomic units; ``input_lines`` remain in PLUMED's
    own input units (nm, kJ/mol, ps by default, or an explicit UNITS line).
    See ``docs/plumed.md`` for unit declarations and a complete example.

    Use ``step(dynamics, state)`` to propagate, and ``evaluate(q)`` for
    read-only diagnostics.  Passing this object directly as Dynamics' bias
    is rejected because that protocol does not supply a step/update clock.
    All-coordinate nonperiodic and fully periodic cells are supported;
    mixed-axis PBC, energy-based CVs, NPT, and automatic checkpointing of an
    opaque PLUMED kernel are not supported.
    """

    history_dependent = True

    def __init__(
        self, input_lines, masses, beta, nbeads, dt, *, cell=None, pbc=None,
        charges=None, kernel_path=None, log_file=None, restart=False,
        start_step=0, _kernel=None,
    ):
        self.beta = _positive(beta, "beta")
        self.dt = _positive(dt, "dt")
        self.nbeads = _integer(nbeads, "nbeads", 1)
        self._completed_step = _integer(start_step, "start_step", 0)
        raw_masses = np.asarray(masses)
        if (raw_masses.ndim != 1 or raw_masses.size == 0
                or raw_masses.dtype.kind not in "iuf"):
            raise ValueError("masses must be a real nonempty vector")
        self._masses = np.array(raw_masses, dtype=np.float64, order="C", copy=True)
        if not np.isfinite(self._masses).all() or np.any(self._masses <= 0):
            raise ValueError("masses must be positive and finite")
        self.shape = (self.nbeads, self._masses.size, 3)
        resolved_cell, flags = validate_cell_pbc(cell, pbc)
        if flags.any() and not flags.all():
            raise ValueError("PLUMED bridge supports all-axis PBC or no PBC; mixed-axis PBC is unsupported")
        self.pbc = tuple(bool(flag) for flag in flags)
        # A zero box is PLUMED's nonperiodic representation, not an artificial
        # large periodic box that could silently affect distance CVs.
        self._box = (np.zeros((3, 3), dtype=np.float64) if not flags.any()
                     else resolved_cell)
        if charges is None:
            self._charges = None
        else:
            raw = np.asarray(charges)
            if raw.shape != (self._masses.size,) or raw.dtype.kind not in "iuf":
                raise ValueError("charges must be a real vector of length N")
            self._charges = np.array(raw, dtype=np.float64, order="C", copy=True)
            if not np.isfinite(self._charges).all():
                raise ValueError("charges must be finite")
        if not isinstance(restart, (bool, np.bool_)):
            raise ValueError("restart must be boolean")
        if isinstance(input_lines, str):
            script = input_lines
        else:
            try:
                lines = list(input_lines)
            except TypeError as exc:
                raise ValueError("input_lines must be PLUMED text or a sequence of lines") from exc
            if not all(isinstance(line, str) for line in lines):
                raise ValueError("all PLUMED input lines must be strings")
            script = "\n".join(lines)
        if not script.strip() or "\x00" in script:
            raise ValueError("PLUMED input must be nonempty text without NUL characters")
        self.input_lines = script
        self._closed = False
        self._failed = False
        self._busy = False
        self._lock = threading.RLock()
        self._stop_flag = np.zeros(1, dtype=np.intc)
        self._buffers = None
        self._bound_q = None
        self._kernel = None
        if _kernel is not None and kernel_path is not None:
            raise ValueError("choose a kernel_path or an injected kernel, not both")
        try:
            if _kernel is None:
                try:
                    module = importlib.import_module("plumed")
                except ImportError as exc:
                    raise ImportError(
                        "Optional PLUMED bridge requires the PLUMED Python wrapper and "
                        "a compatible libplumedKernel; see docs/plumed.md"
                    ) from exc
                self._kernel = (module.Plumed() if kernel_path is None
                                else module.Plumed(kernel=str(kernel_path)))
            else:
                self._kernel = _kernel
            version = np.zeros(1, dtype=np.intc)
            self._kernel.cmd("getApiVersion", version)
            if int(version[0]) < 8:
                raise RuntimeError("PLUMED API >= 8 is required for the multiline, split-update bridge")
            self.api_version = int(version[0])
            self._kernel.cmd("setRealPrecision", 8)
            for key, factor in (
                ("setMDEnergyUnits", 4.184 / KCAL_MOL_TO_AU),
                ("setMDLengthUnits", BOHR_M / 1e-9),
                ("setMDTimeUnits", ATOMIC_TIME_S / 1e-12),
                ("setMDMassUnits", 1.0 / AMU_TO_AU),
                ("setMDChargeUnits", 1.0),
            ):
                self._kernel.cmd(key, float(factor))
            self._kernel.cmd("setNatoms", int(self._masses.size))
            self._kernel.cmd("setMDEngine", "PyRPMD-centroid")
            self._kernel.cmd("setTimestep", self.dt)
            self._kernel.cmd("setKbT", 1.0 / self.beta)
            self._kernel.cmd("setRestart", int(restart))
            if log_file is not None:
                self._kernel.cmd("setLogFile", str(log_file))
            self._kernel.cmd("init")
            self._kernel.cmd("readInputLines", script)
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        """Read PLUMED input text; referenced paths retain the process cwd."""
        return cls(Path(path).read_text(encoding="utf-8"), *args, **kwargs)

    @property
    def completed_step(self):
        return self._completed_step

    @property
    def stop_requested(self):
        """Whether a PLUMED action requested stopping after a committed step."""
        return bool(self._stop_flag[0])

    def _check_open(self):
        if self._closed:
            raise RuntimeError("PLUMED bridge is closed")
        if self._failed:
            raise RuntimeError(
                "PLUMED bridge failed and cannot be reused; restore a consistent "
                "external MD/PLUMED checkpoint into a new kernel"
            )

    def _coordinates(self, q):
        q = np.asarray(q)
        if q.shape != self.shape or q.dtype.kind not in "iuf" or not np.isfinite(q).all():
            raise ValueError(f"PLUMED coordinates must be finite real {self.shape}")
        return q

    def _calculate(self, q, step):
        """Calculate without update; retained arrays outlive all kernel calls."""
        q = self._coordinates(q)
        positions = np.ascontiguousarray(np.mean(q, axis=0), dtype=np.float64)
        force = np.zeros_like(positions)
        virial = np.zeros((3, 3), dtype=np.float64)
        bias = np.zeros(1, dtype=np.float64)
        needed = np.zeros(1, dtype=np.intc)
        # Kernel commands retain pointers, including across update().
        self._buffers = (positions, force, virial, bias, needed)
        try:
            self._kernel.cmd("setStep", int(step))
            self._kernel.cmd("setPositions", positions)
            self._kernel.cmd("setMasses", self._masses)
            if self._charges is not None:
                self._kernel.cmd("setCharges", self._charges)
            self._kernel.cmd("setBox", self._box)
            self._kernel.cmd("setForces", force)
            self._kernel.cmd("setVirial", virial)
            self._kernel.cmd("setStopFlag", self._stop_flag)
            self._kernel.cmd("prepareCalc")
            self._kernel.cmd("isEnergyNeeded", needed)
            if needed[0]:
                raise ValueError(
                    "PLUMED ENERGY-dependent actions are unsupported by the centroid "
                    "bridge; their ring-polymer force chain is not a centroid force"
                )
            self._kernel.cmd("performCalcNoUpdate")
            self._kernel.cmd("getBias", bias)
            if not (np.isfinite(bias).all() and np.isfinite(force).all()
                    and np.isfinite(virial).all()):
                raise FloatingPointError("PLUMED returned nonfinite bias energy, force, or virial")
            energy = self.nbeads * float(bias[0])
            if not math.isfinite(energy):
                raise FloatingPointError("PLUMED ring-polymer bias energy overflow")
        except BaseException:
            # PLUMED does not expose a generic state snapshot/rollback API.
            # A failing arbitrary action may already have changed its state.
            self._failed = True
            raise
        return PlumedBiasResult(
            energy, np.broadcast_to(force, self.shape).copy(), float(bias[0]),
            force.copy(), virial.copy(), int(step),
        )

    def evaluate(self, q):
        """Evaluate the current bias at q without depositing/updating history."""
        with self._lock:
            self._check_open()
            if self._busy:
                raise RuntimeError("cannot run PLUMED diagnostics during a propagation step")
            return self._calculate(q, self._completed_step)

    def energy_force(self, q):
        raise RuntimeError(
            "Use PlumedBias.step(dynamics, state) to advance PLUMED exactly once "
            "per step, or PlumedBias.evaluate(q) for read-only diagnostics"
        )

    def step(self, dynamics, state, *, thermostat=None, gamma_centroid=1.0,
             andersen_interval=10):
        """Advance the existing Dynamics map, then update PLUMED once.

        ``thermostat`` is None, ``'andersen'``, or ``'pile-l'``. The PILE-L
        path reuses Dynamics.sample_step; its integrator restrictions apply.
        A failed force/PES call leaves the supplied MD state and RNG intact.
        A failed PLUMED command also poisons this bridge, because its opaque
        internal state/output files cannot be rolled back generically.
        """
        with self._lock:
            self._check_open()
            if self._busy:
                raise RuntimeError("a PLUMED step is already in progress")
            if self.stop_requested:
                raise RuntimeError("PLUMED requested a stop; retain the last completed state")
            if type(dynamics) is not Dynamics or not isinstance(state, State):
                raise TypeError("PLUMED step requires the standard Dynamics and State classes")
            if (dynamics.nbeads != self.nbeads or dynamics.beta != self.beta
                    or dynamics.dt != self.dt or not np.array_equal(dynamics.masses, self._masses)):
                raise ValueError("PLUMED and Dynamics must use identical P, beta, dt, and masses")
            if dynamics.loop_backend == "native":
                raise ValueError("PLUMED requires the Python step orchestration; use auto or python")
            if thermostat not in (None, "andersen", "pile-l"):
                raise ValueError("thermostat must be None, 'andersen', or 'pile-l'")
            if thermostat == "andersen":
                andersen_interval = _integer(andersen_interval, "andersen_interval", 1)
            if state.step != self._completed_step:
                raise ValueError("MD state clock does not match PLUMED completed_step")
            _integer(state.step + 1, "next step", 0)
            if not math.isclose(state.time, state.step * self.dt, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("PLUMED time requires state.time = state.step * dt")
            # Validate the caller's destination buffers before the irreversible
            # update, not only the independent writable trial copy.
            dynamics._check_parameters()
            dynamics.integrator._check(state.q, state.p)
            dynamics._check_fixed(state.q, state.p)
            self._coordinates(state.q)
            if self._bound_q is not None and not np.array_equal(state.q, self._bound_q):
                raise ValueError("PLUMED is bound to one trajectory; coordinates changed outside its step")
            trial = state.copy()
            old_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            adapter = _StepBias(self, state.step)
            self._busy = True
            try:
                if thermostat == "pile-l":
                    dynamics.sample_step(trial, adapter, gamma_centroid=gamma_centroid)
                else:
                    if thermostat == "andersen" and trial.step % andersen_interval == 0:
                        trial.p[:] = dynamics.sample_momenta()
                    dynamics.step(trial, adapter)
                if adapter.calls != 2 or trial.step != self._completed_step + 1:
                    raise RuntimeError("PLUMED requires the standard two-force Dynamics step")
                if not (np.isfinite(trial.q).all() and np.isfinite(trial.p).all()
                        and math.isfinite(trial.time)):
                    raise FloatingPointError("nonfinite PLUMED integration state")
                try:
                    self._kernel.cmd("update")
                except BaseException:
                    self._failed = True
                    raise
            except BaseException:
                dynamics.rng.bit_generator.state = old_rng
                raise
            finally:
                self._busy = False
            try:
                state.q[:] = trial.q
                state.p[:] = trial.p
                state.step = trial.step
                state.time = trial.time
                self._completed_step = trial.step
                self._bound_q = trial.q.copy()
                dynamics._record_python_steps(1)
            except BaseException:
                # Covers unexpected caller mutation/concurrent buffer changes
                # after validation. History has advanced and cannot be retried.
                self._failed = True
                raise
            return state

    def close(self):
        """Flush/finalize the kernel exactly once; safe after failed commands."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._kernel is not None:
                self._kernel.finalize()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class _StepBias:
    """Apply a fixed history at the initial and endpoint force clocks."""

    def __init__(self, bridge, start_step):
        self.bridge = bridge
        self.start_step = start_step
        self.calls = 0

    def energy_force(self, q):
        if self.calls >= 2:
            raise RuntimeError("PLUMED step received more than two force evaluations")
        result = self.bridge._calculate(q, self.start_step + self.calls)
        self.calls += 1
        return result.energy, result.forces
