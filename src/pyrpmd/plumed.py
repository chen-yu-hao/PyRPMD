"""Optional PLUMED centroid-bias bridge with an explicit update clock.

PLUMED evaluates a physical bias W(q_centroid).  The sum-beads Hamiltonian
receives P*W, so *each* bead receives PLUMED's centroid force.  A PLUMED
``update`` occurs once after an accepted complete MD step, never on a force
or diagnostic evaluation.  No PLUMED import is required to import PyRPMD.
"""

from __future__ import annotations

import copy
import importlib
import math
import re
import shlex
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .constants import AMU_TO_AU, ATOMIC_TIME_S, BOHR_M, KCAL_MOL_TO_AU
from .dynamics import Dynamics, State
from .geometry import validate_cell_pbc
from .validation import checked_integer, finite_positive


@dataclass(frozen=True)
class PlumedFileRestartPlan:
    """Audited, file-only PLUMED state and its deterministic input program.

    This deliberately describes a small subset, not a serializer for arbitrary
    PLUMED actions. The workflow stores the listed files and the MD state in one
    atomic checkpoint; PLUMED reconstructs the bias through its own RESTART.
    """

    input_lines: str
    files: tuple[str, ...]


def file_restart_plan(input_lines):
    """Validate the fixed-hill restart profile before constructing a kernel.

    Ordinary/WT METAD with fixed widths is reconstructed from HILLS. Stateful
    options (grids, adaptive widths/pace, walkers, accumulated work, etc.) are
    rejected. Only a small set of coordinate-only actions is admitted, and all
    output paths must stay below the workflow's PLUMED directory. Round-trip
    output precision and per-step flushing are explicit parts of this profile.
    """
    if not isinstance(input_lines, str) or not input_lines.strip() or "\x00" in input_lines:
        raise ValueError("PLUMED restart input must be nonempty text")
    # Expand PLUMED's standard ACTION ... / ... ACTION multiline notation.
    records, pending, pending_action = [], None, None
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED restart input") from error
        if not tokens:
            continue
        if pending is not None:
            if tokens[0] == "...":
                if len(tokens) > 2 or (len(tokens) == 2 and tokens[1] != pending_action):
                    raise ValueError("mismatched PLUMED multiline action terminator")
                records.append(pending)
                pending = pending_action = None
            else:
                if "..." in tokens:
                    raise ValueError("nested PLUMED multiline actions are unsupported")
                pending.extend(tokens)
        elif tokens[-1] == "...":
            pending = tokens[:-1]
            action_tokens = pending[1:] if pending and pending[0].endswith(":") else pending
            if not action_tokens:
                raise ValueError("missing PLUMED multiline action")
            pending_action = action_tokens[0]
        elif "..." in tokens:
            raise ValueError("invalid PLUMED multiline action")
        else:
            records.append(tokens)
    if pending is not None:
        raise ValueError("unterminated PLUMED multiline action")
    stateless = {
        "UNITS", "DISTANCE", "ANGLE", "TORSION", "POSITION", "COM", "CENTER",
        "COORDINATION", "GROUP", "FIXEDATOM", "COMBINE", "CUSTOM", "MATHEVAL",
        "CONSTANT", "RESTRAINT", "UPPER_WALLS", "LOWER_WALLS",
    }
    metad_keywords = {"ARG", "SIGMA", "HEIGHT", "PACE", "FILE", "BIASFACTOR", "TEMP", "FMT", "RESTART", "LABEL"}
    print_keywords = {"ARG", "FILE", "STRIDE", "FMT", "RESTART", "UPDATE_FROM", "UPDATE_UNTIL", "LABEL"}
    files, normalized = [], []
    for record in records:
        tokens = list(record)
        head = 1 if tokens[0].endswith(":") else 0
        if len(tokens) <= head:
            raise ValueError("missing PLUMED action after label")
        action = tokens[head]
        if action not in stateless | {"METAD", "PRINT", "FLUSH"}:
            raise ValueError(f"PLUMED action {action!r} has no verified file restart contract")
        if any(any(character.isspace() for character in token) or "{" in token or "}" in token or "#" in token
               for token in tokens):
            raise ValueError("PLUMED file restart requires simple whitespace-free keyword values")
        options = {}
        for token in tokens[head + 1:]:
            key, separator, value = token.partition("=")
            if not re.fullmatch(r"[A-Z][A-Z_0-9]*", key) or key in options:
                raise ValueError("invalid or repeated PLUMED restart keyword")
            options[key] = value if separator else None
            if key == "RESTART" and value != "AUTO":
                raise ValueError("PLUMED restart overrides are controlled by the workflow")
            if ("FILE" in key and not (action in {"METAD", "PRINT"} and key == "FILE")) or key == "PDB":
                raise ValueError("external PLUMED input/state files are unsupported by file restart")
            if value and ("@" in value or "\\" in value):
                raise ValueError("external/escaped PLUMED references are unsupported by file restart")
        if action == "METAD" and set(options) - metad_keywords:
            raise ValueError("PLUMED file restart supports only fixed-width ordinary/WT METAD without grids or walkers")
        if action == "METAD":
            try:
                if any(not options.get(key) for key in ("ARG", "SIGMA", "HEIGHT", "PACE")):
                    raise ValueError("missing fixed-hill parameter")
                sigma = [float(value) for value in options["SIGMA"].split(",")]
                height = float(options["HEIGHT"])
                if (not all(math.isfinite(value) and value > 0 for value in sigma)
                        or not math.isfinite(height) or height < 0
                        or not options["PACE"].isdigit() or int(options["PACE"]) < 1):
                    raise ValueError("invalid fixed-hill parameter")
                for key in ("BIASFACTOR", "TEMP"):
                    if key in options:
                        value = float(options[key])
                        if not math.isfinite(value) or value <= (1 if key == "BIASFACTOR" else 0):
                            raise ValueError("invalid tempering parameter")
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError("PLUMED file restart requires finite fixed SIGMA/HEIGHT, positive integer PACE, BIASFACTOR>1 and TEMP>0") from error
        if action == "PRINT" and set(options) - print_keywords:
            raise ValueError("unsupported PRINT option in PLUMED file restart")
        if action == "FLUSH":
            if set(options) - {"STRIDE", "LABEL"}:
                raise ValueError("unsupported FLUSH option in PLUMED file restart")
            # One final FLUSH is sufficient and is always executed after outputs.
            continue
        if action in {"METAD", "PRINT"}:
            name = options.get("FILE", "HILLS" if action == "METAD" else "COLVAR")
            path = Path(name) if isinstance(name, str) else None
            if (path is None or not name or path.is_absolute() or ".." in path.parts
                    or ":" in name or not path.parts or str(path) != name
                    or any(ord(char) < 32 for char in name)):
                raise ValueError("PLUMED restart FILE must be a normalized relative path below RUN/plumed")
            if name in files:
                raise ValueError("PLUMED restart output FILE paths must be distinct")
            files.append(name)
            if "FILE" not in options:
                tokens.append("FILE=" + name)
            if "FMT" in options and options["FMT"] not in {"%.17g", "%.17e"}:
                raise ValueError("PLUMED file restart requires round-trip FMT=%.17g or FMT=%.17e")
            if "FMT" not in options:
                tokens.append("FMT=%.17g")
        normalized.append(" ".join(tokens))
    if not normalized:
        raise ValueError("PLUMED restart input contains no actions")
    normalized.append("FLUSH STRIDE=1")
    return PlumedFileRestartPlan("\n".join(normalized) + "\n", tuple(files))


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


_positive = partial(
    finite_positive,
    shape="scalar",
    message="{name} must be positive and finite",
)


_integer = partial(
    checked_integer,
    maximum=np.iinfo(np.intc).max,
    message="{name} must be an integer >= {minimum}",
    maximum_message="{name} exceeds the PLUMED setStep integer range",
)


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
        self._restart = bool(restart)
        self._restart_primed = False
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

    def _prime_file_restart(self, q):
        """Initialize the restored update clock for the audited file profile.

        PLUMED 2.9 METAD skips the first update of every newly constructed
        kernel, including a restarted one. Replaying the saved endpoint primes
        that flag without propagating MD or adding a hill. The workflow removes
        the replayed PRINT row after checking the files kept their saved prefix.
        Newer duplicate-hill-aware kernels also leave this endpoint unchanged.
        """
        with self._lock:
            self._check_open()
            if not self._restart or self._completed_step == 0 or self._restart_primed or self._busy:
                raise RuntimeError("PLUMED file restart clock can only be primed once at a restored positive step")
            file_restart_plan(self.input_lines)
            before = self._calculate(q, self._completed_step)
            try:
                self._kernel.cmd("update")
                after = self._calculate(q, self._completed_step)
                if (before.physical_energy != after.physical_energy
                        or not np.array_equal(before.centroid_forces, after.centroid_forces)
                        or not np.array_equal(before.plumed_virial, after.plumed_virial)):
                    raise RuntimeError("PLUMED restart clock replay changed its bias; this kernel/action is not restart-compatible")
                self._restart_primed = True
                self._bound_q = np.array(q, dtype=np.float64, copy=True)
            except BaseException:
                self._failed = True
                raise

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
