"""Extended PLUMED coordinate mapping and native multiple-walker helpers.

The legacy :mod:`pyrpmd.plumed` bridge remains the implementation for existing
single-trajectory centroid inputs.  This module is used only by the opt-in
replica, native-walker, and bead-resolved workflow.
"""

from __future__ import annotations

import importlib
import math
import shlex
from pathlib import PurePosixPath

import numpy as np

from .plumed import PlumedBias, PlumedBiasResult


_FILE_WALKER_ACTIONS = frozenset({"METAD", "PBMETAD"})
_MPI_WALKER_ACTIONS = frozenset({
    "METAD",
    "PBMETAD",
    "OPES_METAD",
    "OPES_METAD_EXPLORE",
    "OPES_EXPANDED",
})
_VES_WALKER_ACTIONS = frozenset({
    "OPT_AVERAGED_SGD",
    "OPT_ROBBINS_MONRO_SGD",
    "OPT_ADAM",
})


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
    if (
        isinstance(nbeads, (bool, np.bool_))
        or not isinstance(nbeads, (int, np.integer))
        or int(nbeads) < 1
    ):
        raise ValueError("nbeads must be an integer >= 1")
    if (
        isinstance(natoms, (bool, np.bool_))
        or not isinstance(natoms, (int, np.integer))
        or int(natoms) < 1
    ):
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


def _append_keyword(line: str, keyword: str, tokens: list[str]) -> list[str]:
    if tokens and tokens[-1] == "...":
        return [line, "  " + keyword]
    body, marker, comment = line.partition("#")
    body = body.rstrip() + " " + keyword
    return [body + ((" #" + comment) if marker else "")]


def configure_multiple_walkers(
    input_lines: str,
    *,
    walkers: str,
    count: int,
    walker_id: int,
    read_stride: int = 1,
    directory: str = "../shared",
) -> str:
    """Inject workflow-owned PLUMED walker options into supported actions.

    File sharing is admitted for METAD/PBMETAD.  MPI multi-sim walkers are
    admitted for METAD/PBMETAD, OPES variants, and the supported VES
    optimizers.  PyRPMD owns every walker ID and communicator option; users may
    not provide a second, conflicting walker topology in the PLUMED program.
    """
    walkers = _checked_walkers(walkers)
    if (
        not isinstance(input_lines, str)
        or not input_lines.strip()
        or "\x00" in input_lines
    ):
        raise ValueError("PLUMED input must be nonempty text")
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        for token in tokens:
            key = token.partition("=")[0]
            if key.startswith("WALKERS_") or key == "MULTIPLE_WALKERS":
                raise ValueError(
                    "PLUMED WALKERS_* and MULTIPLE_WALKERS options are owned "
                    "by the PyRPMD workflow"
                )
    if walkers == "independent":
        return input_lines
    for value, name, minimum in (
        (count, "walker count", 2),
        (walker_id, "walker id", 0),
        (read_stride, "walker read stride", 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    count, walker_id, read_stride = int(count), int(walker_id), int(read_stride)
    if walker_id >= count:
        raise ValueError("walker id must be smaller than walker count")
    if walkers == "files" and (
        not isinstance(directory, str)
        or not directory
        or any(character.isspace() for character in directory)
        or "\x00" in directory
    ):
        raise ValueError("walker directory must be a nonempty whitespace-free path")

    file_keyword = (
        f"WALKERS_N={count} WALKERS_ID={walker_id} "
        f"WALKERS_DIR={directory} WALKERS_RSTRIDE={read_stride}"
    )
    output: list[str] = []
    injected = 0
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        action = _line_action(tokens)
        keyword = None
        if walkers == "files" and action in _FILE_WALKER_ACTIONS:
            keyword = file_keyword
        elif walkers == "mpi" and action in _MPI_WALKER_ACTIONS:
            keyword = "WALKERS_MPI"
        elif walkers == "mpi" and action in _VES_WALKER_ACTIONS:
            keyword = "MULTIPLE_WALKERS"
        if keyword is None:
            output.append(line)
        else:
            injected += 1
            output.extend(_append_keyword(line, keyword, tokens))
    if injected == 0:
        supported = (
            sorted(_FILE_WALKER_ACTIONS)
            if walkers == "files"
            else sorted(_MPI_WALKER_ACTIONS | _VES_WALKER_ACTIONS)
        )
        raise ValueError(
            f"plumed.walkers={walkers!r} requires a supported action "
            f"({', '.join(supported)})"
        )
    return "\n".join(output) + ("\n" if input_lines.endswith("\n") else "")


def ensure_checkpoint_flush(input_lines: str) -> str:
    """Append a per-step FLUSH action unless the program already has one."""
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        if _line_action(tokens) == "FLUSH":
            return input_lines
    return input_lines.rstrip() + "\nFLUSH STRIDE=1\n"


def validate_managed_output_paths(input_lines: str) -> None:
    """Reject write-like PLUMED paths that escape a managed replica directory."""
    for line in input_lines.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError("invalid quoted PLUMED input") from error
        for token in tokens:
            key, separator, value = token.partition("=")
            if not separator or not value or "FILE" not in key:
                continue
            if key == "WALKERS_DIR":
                # This exact path is injected by the workflow and resolves from
                # replica-NNNN to the managed sibling directory.
                if value != "../shared":
                    raise ValueError("PLUMED WALKERS_DIR is owned by the workflow")
                continue
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != value
                or ":" in value
                or "\\" in value
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(
                    f"PLUMED state path {key}={value!r} must be a normalized "
                    "relative path below the replica directory"
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
        kernel = (
            module.Plumed()
            if kernel_path is None
            else module.Plumed(kernel=str(kernel_path))
        )
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
    """PLUMED bias with explicit centroid or bead-resolved atom mapping.

    ``mode='centroid'`` retains the legacy sum-beads contribution
    ``P W(q_centroid)``.  ``mode='beads'`` presents ``P*N`` synthetic atoms to
    PLUMED in bead-major order and adds the returned bias once; each synthetic
    atom force is applied directly to the corresponding bead coordinate.
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
        original_masses = np.array(
            raw_masses, dtype=np.float64, order="C", copy=True
        )
        original_charges = (
            None
            if charges is None
            else np.array(
                np.asarray(charges), dtype=np.float64, order="C", copy=True
            )
        )
        if (
            isinstance(nbeads, (bool, np.bool_))
            or not isinstance(nbeads, (int, np.integer))
            or int(nbeads) < 1
        ):
            raise ValueError("nbeads must be an integer >= 1")
        p = int(nbeads)
        configured = _configured_kernel(
            kernel_path, _kernel, replica_comm, multi_sim_comm
        )
        if mode == "beads":
            plumed_masses = np.tile(original_masses, p)
            plumed_charges = (
                None if original_charges is None else np.tile(original_charges, p)
            )
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
            # Restore the physical Dynamics shape after PLUMED has been
            # initialized with P*N synthetic atoms.
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
                    "PLUMED ENERGY-dependent actions are unsupported by the "
                    "bead bridge because their PES force chain is unavailable"
                )
            self._kernel.cmd("performCalcNoUpdate")
            self._kernel.cmd("getBias", bias)
            if not (
                np.isfinite(bias).all()
                and np.isfinite(force).all()
                and np.isfinite(virial).all()
            ):
                raise FloatingPointError(
                    "PLUMED returned nonfinite bias energy, force, or virial"
                )
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
        """Prime a restored extended kernel and verify replay invariance."""
        with self._lock:
            self._check_open()
            if (
                not self._restart
                or self._completed_step == 0
                or self._restart_primed
                or self._busy
            ):
                raise RuntimeError(
                    "PLUMED file restart clock can only be primed once at a "
                    "restored positive step"
                )
            before = self._calculate(q, self._completed_step)
            try:
                self._kernel.cmd("update")
                after = self._calculate(q, self._completed_step)
                if (
                    before.physical_energy != after.physical_energy
                    or not np.array_equal(before.forces, after.forces)
                    or not np.array_equal(
                        before.plumed_virial, after.plumed_virial
                    )
                ):
                    raise RuntimeError(
                        "PLUMED restart replay changed its bias; this action "
                        "state is not restart-compatible"
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
