"""Extended PLUMED coordinate mapping and native multiple-walker helpers.

This module is deliberately separate from :mod:`pyrpmd.plumed`: existing
single-trajectory centroid inputs continue to use that implementation and its
v1 checkpoint contract.  The extended workflow opts into this module only
when replicas, bead-resolved coordinates, or native PLUMED walkers are
requested.
"""

from __future__ import annotations

import importlib
import math
import re
import shlex
from pathlib import Path

import numpy as np

from .plumed import PlumedBias, PlumedBiasResult


_WALKER_ACTIONS = {
    "files": frozenset({"METAD", "PBMETAD"}),
    "mpi": frozenset({"METAD", "PBMETAD", "OPES_METAD", "OPES_EXPANDED"}),
}


def _checked_mode(mode: str) -> str:
    if mode not in ("centroid", "beads"):
        raise ValueError("plumed.mode must be 'centroid' or 'beads'")
    return mode


def _checked_walkers(walkers: str) -> str:
    if walkers not in ("independent", "files", "mpi"):
        raise ValueError("plumed.walkers must be 'independent', 'files' or 'mpi'")
    return walkers


def plumed_atom_map(nbeads: int, natoms: int, mode: str = "centroid") -> list[dict]:
    """Return the explicit one-based PLUMED-to-ring-polymer atom map."""
    mode = _checked_mode(mode)
    if (isinstance(nbeads, bool) or not isinstance(nbeads, (int, np.integer))
            or int(nbeads) < 1):
        raise ValueError("nbeads must be an integer >= 1")
    if (isinstance(natoms, bool) or not isinstance(natoms, (int, np.integer))
            or int(natoms) < 1):
        raise ValueError("natoms must be an integer >= 1")
    nbeads, natoms = int(nbeads), int(natoms)
    if mode == "centroid":
        return [
            {"plumed_atom": atom + 1, "source": "centroid", "atom": atom}
            for atom in range(natoms)
        ]
    return [
        {
            "plumed_atom": bead * natoms + atom + 1,
            "source": "bead",
            "bead": bead,
            "atom": atom,
        }
        for bead in range(nbeads)
        for atom in range(natoms)
    ]


def _line_action(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    head = 1 if tokens[0].endswith(":") else 0
    return tokens[head] if head < len(tokens) else None


def configure_multiple_walkers(
    input_lines: str,
    *,
    walkers: str,
    count: int,
    walker_id: int,
    read_stride: int = 1,
    directory: str = "../shared",
) -> str:
    """Inject workflow-owned PLUMED walker keywords into supported actions.

    User-provided ``WALKERS_*`` options are rejected so the configured replica
    count, ID and communicator cannot disagree with PLUMED's scientific state.
    File walkers are supported for METAD/PBMETAD.  MPI walkers additionally
    cover OPES_METAD and OPES_EXPANDED, whose PLUMED contracts use
    ``WALKERS_MPI``.
    """
    walkers = _checked_walkers(walkers)
    if not isinstance(input_lines, str) or not input_lines.strip() or "\x00" in input_lines:
        raise ValueError("PLUMED input must be nonempty text")
    if walkers == "independent":
        return input_lines
    for value, name, minimum in (
        (count, "walker count", 2),
        (walker_id, "walker id", 0),
        (read_stride, "walker read stride", 1),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or int(value) < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    count, walker_id, read_stride = int(count), int(walker_id), int(read_stride)
    if walker_id >= count:
        raise ValueError("walker id must be smaller than walker count")
    if walkers == "files":
        if (not isinstance(directory, str) or not directory or any(c.isspace() for c in directory)
                or "\x00" in directory):
            raise ValueError("walker directory must be a nonempty whitespace-free path")
    if re.search(r"\bWALKERS_[A-Z0-9_]*\b", input_lines):
        raise ValueError("PLUMED WALKERS_* keywords are owned by the PyRPMD workflow")

    keywords = (
        f"WALKERS_N={count} WALKERS_ID={walker_id} "
        f"WALKERS_DIR={directory} WALKERS_RSTRIDE={read_stride}"
        if walkers == "files" else "WALKERS_MPI"
    )
    targets = _WALKER_ACTIONS[walkers]
    output: list[str] = []
    injected = 0
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        action = _line_action(tokens)
        if action not in targets:
            output.append(line)
            continue
        injected += 1
        if tokens and tokens[-1] == "...":
            output.append(line)
            output.append("  " + keywords)
        else:
            body, marker, comment = line.partition("#")
            body = body.rstrip() + " " + keywords
            output.append(body + ((" #" + comment) if marker else ""))
    if injected == 0:
        supported = ", ".join(sorted(targets))
        raise ValueError(
            f"plumed.walkers={walkers!r} requires a supported bias action ({supported})"
        )
    return "\n".join(output) + ("\n" if input_lines.endswith("\n") else "")


def ensure_checkpoint_flush(input_lines: str) -> str:
    """Append a per-step FLUSH action unless the script already has one."""
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        if _line_action(tokens) == "FLUSH":
            return input_lines
    return input_lines.rstrip() + "\nFLUSH STRIDE=1\n"


def validate_managed_output_paths(input_lines: str) -> None:
    """Reject output/state paths that can escape an extended run directory.

    PLUMED has many actions, so this check is intentionally lexical and
    conservative: write-like ``*FILE`` keywords must be normalized relative
    paths.  Explicit read-file keywords (``*RFILE``) are not treated as output
    state; users remain responsible for keeping those immutable.
    """
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        for token in tokens:
            key, separator, value = token.partition("=")
            if not separator or not value or "FILE" not in key or key.endswith("RFILE"):
                continue
            if key == "WALKERS_DIR":
                continue
            path = Path(value)
            if (path.is_absolute() or ".." in path.parts or str(path) != value
                    or any(ord(char) < 32 for char in value)):
                raise ValueError(
                    f"PLUMED state path {key}={value!r} must stay below the replica directory"
                )


def _configured_kernel(kernel_path, injected_kernel, replica_comm, multi_sim_comm):
    if injected_kernel is not None and kernel_path is not None:
        raise ValueError("choose a kernel_path or an injected kernel, not both")
    if injected_kernel is None and replica_comm is None and multi_sim_comm is None:
        return None
    if injected_kernel is None:
        try:
            module = importlib.import_module("plumed")
        except ImportError as exc:
            raise ImportError(
                "Optional PLUMED bridge requires the PLUMED Python wrapper and "
                "a compatible libplumedKernel; see docs/plumed.md"
            ) from exc
        kernel = module.Plumed() if kernel_path is None else module.Plumed(kernel=str(kernel_path))
    else:
        kernel = injected_kernel
    try:
        if replica_comm is not None:
            kernel.cmd("setMPIComm", replica_comm)
        if multi_sim_comm is not None:
            kernel.cmd("setMPImultiSimComm", multi_sim_comm)
    except BaseException:
        try:
            kernel.finalize()
        finally:
            raise
    return kernel


class ExtendedPlumedBias(PlumedBias):
    """PLUMED bias with explicit centroid or bead-resolved coordinate mapping.

    ``mode='centroid'`` has the original Hamiltonian contribution
    ``P W(q_centroid)``.  ``mode='beads'`` presents ``P*N`` atoms to PLUMED in
    bead-major order and adds the returned bias once to the ring-polymer
    Hamiltonian; PLUMED's force on each synthetic atom is applied directly to
    the corresponding bead coordinate.
    """

    def __init__(
        self,
        input_lines,
        masses,
        beta,
        nbeads,
        dt,
        *,
        mode="centroid",
        cell=None,
        pbc=None,
        charges=None,
        kernel_path=None,
        log_file=None,
        restart=False,
        start_step=0,
        replica_comm=None,
        multi_sim_comm=None,
        _kernel=None,
    ):
        mode = _checked_mode(mode)
        raw_masses = np.asarray(masses)
        if raw_masses.ndim != 1:
            raise ValueError("masses must be a real nonempty vector")
        original_masses = np.array(raw_masses, dtype=np.float64, order="C", copy=True)
        original_charges = None if charges is None else np.array(
            np.asarray(charges), dtype=np.float64, order="C", copy=True
        )
        p = int(nbeads) if not isinstance(nbeads, bool) else 0
        if p < 1:
            raise ValueError("nbeads must be an integer >= 1")
        configured = _configured_kernel(kernel_path, _kernel, replica_comm, multi_sim_comm)
        if mode == "beads":
            plumed_masses = np.tile(original_masses, p)
            plumed_charges = None if original_charges is None else np.tile(original_charges, p)
            super().__init__(
                input_lines,
                plumed_masses,
                beta,
                1,
                dt,
                cell=cell,
                pbc=pbc,
                charges=plumed_charges,
                kernel_path=None if configured is not None else kernel_path,
                log_file=log_file,
                restart=restart,
                start_step=start_step,
                _kernel=configured,
            )
            # Restore the physical Dynamics contract after the parent has
            # initialized PLUMED with P*N synthetic atoms.
            self.nbeads = p
            self._masses = original_masses
            self._charges = original_charges
            self.shape = (p, original_masses.size, 3)
            self._plumed_masses = plumed_masses
            self._plumed_charges = plumed_charges
        else:
            super().__init__(
                input_lines,
                original_masses,
                beta,
                p,
                dt,
                cell=cell,
                pbc=pbc,
                charges=original_charges,
                kernel_path=None if configured is not None else kernel_path,
                log_file=log_file,
                restart=restart,
                start_step=start_step,
                _kernel=configured,
            )
            self._plumed_masses = self._masses
            self._plumed_charges = self._charges
        self.mode = mode

    @property
    def atom_map(self):
        return plumed_atom_map(self.nbeads, self._masses.size, self.mode)

    def _calculate(self, q, step):
        if self.mode == "centroid":
            return super()._calculate(q, step)
        q = self._coordinates(q)
        positions = np.ascontiguousarray(q.reshape(-1, 3), dtype=np.float64)
        force = np.zeros_like(positions)
        virial = np.zeros((3, 3), dtype=np.float64)
        bias = np.zeros(1, dtype=np.float64)
        needed = np.zeros(1, dtype=np.intc)
        self._buffers = (positions, force, virial, bias, needed)
        try:
            self._kernel.cmd("setStep", int(step))
            self._kernel.cmd("setPositions", positions)
            self._kernel.cmd("setMasses", self._plumed_masses)
            if self._plumed_charges is not None:
                self._kernel.cmd("setCharges", self._plumed_charges)
            self._kernel.cmd("setBox", self._box)
            self._kernel.cmd("setForces", force)
            self._kernel.cmd("setVirial", virial)
            self._kernel.cmd("setStopFlag", self._stop_flag)
            self._kernel.cmd("prepareCalc")
            self._kernel.cmd("isEnergyNeeded", needed)
            if needed[0]:
                raise ValueError(
                    "PLUMED ENERGY-dependent actions are unsupported by the bead bridge; "
                    "their force chain is not available"
                )
            self._kernel.cmd("performCalcNoUpdate")
            self._kernel.cmd("getBias", bias)
            if not (np.isfinite(bias).all() and np.isfinite(force).all()
                    and np.isfinite(virial).all()):
                raise FloatingPointError("PLUMED returned nonfinite bias energy, force, or virial")
            energy = float(bias[0])
            if not math.isfinite(energy):
                raise FloatingPointError("PLUMED bead-resolved bias energy overflow")
        except BaseException:
            self._failed = True
            raise
        bead_force = force.reshape(self.shape)
        return PlumedBiasResult(
            energy,
            bead_force.copy(),
            energy,
            np.sum(bead_force, axis=0),
            virial.copy(),
            int(step),
        )

    def _prime_file_restart(self, q):
        """Prime a restored extended kernel without constraining its actions."""
        with self._lock:
            self._check_open()
            if (not self._restart or self._completed_step == 0
                    or self._restart_primed or self._busy):
                raise RuntimeError(
                    "PLUMED file restart clock can only be primed once at a restored positive step"
                )
            before = self._calculate(q, self._completed_step)
            try:
                self._kernel.cmd("update")
                after = self._calculate(q, self._completed_step)
                if (before.physical_energy != after.physical_energy
                        or not np.array_equal(before.forces, after.forces)
                        or not np.array_equal(before.plumed_virial, after.plumed_virial)):
                    raise RuntimeError(
                        "PLUMED restart clock replay changed its bias; this kernel/action "
                        "is not restart-compatible"
                    )
                self._restart_primed = True
                self._bound_q = np.array(q, dtype=np.float64, copy=True)
            except BaseException:
                self._failed = True
                raise


__all__ = [
    "ExtendedPlumedBias",
    "configure_multiple_walkers",
    "ensure_checkpoint_flush",
    "plumed_atom_map",
    "validate_managed_output_paths",
]
