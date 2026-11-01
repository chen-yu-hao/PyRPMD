"""LAMMPS shared-library force adapter with explicit atom and unit contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import BOHR_ANGSTROM, EV_TO_AU, KCAL_MOL_TO_AU
from ..geometry import validate_cell_pbc
from ..io import file_sha256
from .base import (
    BackendUnavailableError, PESResult, Potential, PotentialError,
    coordinates, validate_result,
)
from .python import import_object


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


def _chemical_symbols(symbols):
    if symbols is None or isinstance(symbols, str) or not len(symbols):
        raise ValueError("LAMMPS requires an ordered sequence of System symbols")
    return tuple("H" if s in ("D", "T") else str(s) for s in symbols)


class LAMMPSPotential(Potential):
    """Reuse one configured, single-rank ``lammps.lammps`` instance.

    An input script (``model``) or factory defines atom IDs, types, charges,
    topology and potential. Atom IDs must be consecutive 1..N in PyRPMD order.
    ``type_map`` identifies each LAMMPS type, starting with type 1. Supported
    backend unit styles are ``real`` and ``metal``; they are read from the engine.

    Only zero-step force evaluations are issued. Engine-side fixes are refused,
    including SHAKE, thermostats, time-dependent driving and force modifiers.
    """

    def __init__(self, symbols, model=None, *, factory=None, factory_kwargs=None,
                 type_map=None, cell=None, pbc=None, device="cpu", cmdargs=None,
                 library_name="", asset_files=()):
        self.closed = False
        self.engine = None
        self.symbols = _chemical_symbols(symbols)
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        if (model is None) == (factory is None):
            raise ValueError("LAMMPS requires exactly one of model or factory")
        if factory is None and factory_kwargs:
            raise ValueError("LAMMPS factory_kwargs require a factory")
        if type_map is None or isinstance(type_map, str):
            raise ValueError("LAMMPS requires params.type_map, ordered from atom type 1")
        self.type_map = _chemical_symbols(type_map)
        if (not isinstance(asset_files, (list, tuple))
                or not all(isinstance(path, (str, Path)) for path in asset_files)):
            raise ValueError("LAMMPS asset_files must list every external data, potential and include file")
        self.asset_sha256 = {str(Path(path).resolve()): file_sha256(path) for path in asset_files}
        if cmdargs is not None and (not isinstance(cmdargs, (list, tuple))
                or not all(isinstance(value, str) for value in cmdargs)):
            raise ValueError("LAMMPS cmdargs must be a list of strings")
        if str(device) != "cpu":
            raise ValueError("LAMMPS acceleration is configured by its own packages; PyRPMD GPU device routing is unsupported")
        try:
            from lammps import lammps
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError("LAMMPS provider requires matching lammps Python/shared-library installations") from exc
        # PyRPMD MPI ranks must never implicitly share COMM_WORLD for bead PESs.
        # With a one-rank world, None works for both MPI and MPI-stub builds.
        comm = None
        try:
            from mpi4py import MPI
        except ImportError:
            pass
        else:
            if MPI.COMM_WORLD.Get_size() > 1:
                comm = MPI.COMM_SELF
        try:
            if factory:
                if cmdargs or library_name:
                    raise ValueError("LAMMPS factory owns cmdargs and library_name")
                self.engine = import_object(factory)(comm=comm, **dict(factory_kwargs or {}))
            else:
                options = ["-log", "none", "-screen", "none", "-nocite"] + list(cmdargs or [])
                try:
                    self.engine = lammps(name=library_name, cmdargs=options, comm=comm)
                except OSError as exc:
                    raise BackendUnavailableError(
                        f"LAMMPS shared library could not load; check its MPI/runtime dependencies: {exc}"
                    ) from exc
                path = Path(model).resolve()
                # Avoid a process-wide chdir. Scripts can explicitly reference
                # assets with ${pyrpmd_model_dir}/filename.
                directory = str(path.parent)
                if any(char in directory for char in ('"', "\n", "\r", "$")):
                    raise ValueError("LAMMPS model directory contains unsupported script characters")
                self.engine.command(f'variable pyrpmd_model_dir string "{directory}"')
                self.engine.file(str(path))
            self._check_engine()
            units = _text(self.engine.extract_global("units"))
            if units not in ("real", "metal"):
                raise PotentialError("LAMMPS provider supports only real or metal units")
            self.unit_style = units
            self.energy_factor = KCAL_MOL_TO_AU if units == "real" else EV_TO_AU
            # Native pressure conversion preserves LAMMPS' own constants and
            # includes reciprocal-space virials, unlike sum(r*f) under PBC.
            self._nktv2p = float(self.engine.extract_global("nktv2p"))
            if not np.isfinite(self._nktv2p) or self._nktv2p <= 0:
                raise PotentialError("LAMMPS did not supply a valid pressure conversion")
            self._validate_box(self.cell, self.pbc)
            self._validate_atoms()
            reserved = {"pyrpmd_pe", "pyrpmd_pressure"}
            if reserved.intersection(self.engine.available_ids("compute")):
                raise PotentialError("LAMMPS compute IDs pyrpmd_pe/pyrpmd_pressure are reserved")
            self.engine.command("compute pyrpmd_pe all pe")
            self.engine.command("compute pyrpmd_pressure all pressure NULL virial")
            self.engine.command("thermo_style custom step c_pyrpmd_pe " + " ".join(
                f"c_pyrpmd_pressure[{i}]" for i in range(1, 7)))
            self.engine.command("thermo_modify norm no lost error")
            self.engine.command("neigh_modify every 1 delay 0 check yes")
            self.metadata = {
                "provider": "lammps", "version": int(self.engine.version()),
                "thread_safety": "context_serial", "execution_device": "backend",
                "periodic": bool(self.pbc.any()), "supports_pbc": True,
                "native_batch": False, "trajectory_batch_safe": True,
                "virial": True, "unit_style": units, "type_map": list(self.type_map),
                "atom_order": "consecutive LAMMPS IDs 1..N",
                "model_sha256": file_sha256(model) if model is not None else None,
                "factory": factory,
                "asset_sha256": dict(self.asset_sha256),
                "pair_style": _text(self.engine.extract_global("pair_style")),
                "kspace_style": _text(self.engine.extract_global("kspace_style")),
            }
        except BaseException:
            self.close()
            raise

    def _check_engine(self):
        if self.engine is None or not callable(getattr(self.engine, "extract_setting", None)):
            raise TypeError("LAMMPS factory must return a configured lammps.lammps instance")
        if self.engine.extract_setting("world_size") != 1:
            raise PotentialError("Each LAMMPS provider must use one rank (MPI.COMM_SELF)")
        if self.engine.extract_setting("dimension") != 3:
            raise PotentialError("LAMMPS provider requires dimension 3")
        if self.engine.available_ids("fix"):
            raise PotentialError("LAMMPS force providers cannot contain fixes; PyRPMD owns integration and constraints")
        if self.engine.available_ids("dump"):
            raise PotentialError("LAMMPS force providers cannot contain dumps; export PyRPMD trajectories instead")
        # In particular DPD contributes random/velocity-dependent forces even
        # during run 0, with no fix required. Hybrid pair styles can conceal
        # such a substyle, so this initial provider deliberately accepts a
        # bounded set of coordinate-only pair families, including their
        # accelerator suffixes. Extend only with a numerical force contract.
        pair = _text(self.engine.extract_global("pair_style"))
        families = ("lj/", "lj96/", "lj128/", "buck", "born", "coul/", "morse",
                    "soft", "table", "eam", "meam", "tersoff", "sw", "edip",
                    "airebo", "rebo", "bop", "agni", "snap", "pace", "mliap",
                    "deepmd", "quip", "kim", "zero/")
        if pair not in ("none", "zero") and not str(pair).startswith(families):
            raise PotentialError(f"LAMMPS pair_style {pair!r} is outside the supported coordinate-only force families (hybrid and driven/velocity-dependent styles are unsupported)")
        if self.engine.extract_global("respa_levels"):
            raise PotentialError("LAMMPS force providers require run_style verlet, not engine-side r-RESPA")
        # Rotating particles and mutable per-particle electronic/shape degrees
        # of freedom cannot be represented by PyRPMD Cartesian q alone.
        style = _text(self.engine.extract_global("atom_style"))
        if style not in ("atomic", "charge", "bond", "angle", "molecular", "full"):
            raise PotentialError(f"LAMMPS atom_style {style!r} has no supported Cartesian-only contract")

    def _box_data(self):
        lo, hi, xy, yz, xz, pbc, _ = self.engine.extract_box()
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        lengths = hi-lo
        matrix = np.array([[lengths[0], 0, 0], [xy, lengths[1], 0], [xz, yz, lengths[2]]])
        return lo, matrix, np.asarray(pbc, bool)

    def _validate_box(self, cell, pbc):
        lo, matrix, engine_pbc = self._box_data()
        if not np.array_equal(engine_pbc, pbc):
            raise PotentialError("LAMMPS box periodicity does not match System PBC")
        if not np.isfinite(matrix).all() or np.any(np.diag(matrix) <= 0):
            raise PotentialError("LAMMPS requires a finite, nondegenerate simulation box")
        if cell is not None and not np.allclose(matrix, cell * BOHR_ANGSTROM, rtol=1e-10, atol=1e-10):
            raise PotentialError("LAMMPS restricted-triclinic row cell does not match System.cell")
        self._origin, self._box_matrix = lo, matrix

    def _validate_atoms(self):
        n = len(self.symbols)
        if self.engine.get_natoms() != n or self.engine.extract_setting("nlocal") != n:
            raise PotentialError("LAMMPS atom count/ownership does not match System")
        ids = np.asarray(self.engine.numpy.extract_atom("id")[:n], dtype=np.int64)
        if not np.array_equal(np.sort(ids), np.arange(1, n+1)):
            raise PotentialError("LAMMPS atom IDs must be consecutive 1..N in System order")
        types = np.asarray(self.engine.numpy.extract_atom("type")[:n], dtype=np.int64)
        if len(self.type_map) != self.engine.extract_setting("ntypes") or np.any(types < 1) or np.any(types > len(self.type_map)):
            raise PotentialError("LAMMPS type_map must identify every atom type")
        observed = tuple(self.type_map[int(t)-1] for t in types[np.argsort(ids)])
        if observed != self.symbols:
            raise PotentialError("LAMMPS type_map/atom IDs do not match System symbols/order")
        return ids

    def bind_system(self, system):
        if self.closed:
            raise PotentialError("LAMMPS provider is closed")
        def get(key, default=None):
            return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)
        if _chemical_symbols(get("symbols")) != self.symbols:
            raise PotentialError("LAMMPS system atom identities/order differ from the provider")
        cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        self._validate_box(cell, pbc)
        self.cell, self.pbc = cell, pbc

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("LAMMPS provider is closed")
        x = coordinates(x)
        if x.shape[1] != len(self.symbols):
            raise PotentialError("LAMMPS atom count mismatch")
        energies, forces, virials = [], [], []
        n = len(self.symbols)
        for frame in x:
            self._check_engine()
            ids = self._validate_atoms()
            position = frame * BOHR_ANGSTROM
            # Wrap a temporary copy only; PyRPMD's diffusion paths stay unwrapped.
            frac = (position-self._origin) @ np.linalg.inv(self._box_matrix)
            if np.any((frac[:, ~self.pbc] < 0) | (frac[:, ~self.pbc] >= 1)):
                raise PotentialError("LAMMPS coordinates lie outside a nonperiodic box boundary")
            frac[:, self.pbc] -= np.floor(frac[:, self.pbc])
            position = frac @ self._box_matrix + self._origin
            self.engine.numpy.extract_atom("x")[:n] = position[ids-1]
            # No kinetic contribution is requested; zero stale velocities as
            # well because velocity-dependent potentials are unsupported.
            self.engine.numpy.extract_atom("v")[:n] = 0.0
            try:
                self.engine.command("run 0 pre yes post no")
            except Exception as exc:
                # LAMMPS does not guarantee engine state after a command
                # failure. Do not reuse a partially changed neighbor/domain
                # state if a caller catches the failed force evaluation.
                self.close()
                raise PotentialError(f"LAMMPS force evaluation failed: {exc}") from exc
            origin, matrix, pbc = self._box_data()
            if (not np.array_equal(pbc, self.pbc)
                    or not np.allclose(matrix, self._box_matrix, rtol=1e-12, atol=1e-12)
                    or not np.allclose(origin, self._origin, rtol=1e-12, atol=1e-12)):
                self.close()
                raise PotentialError("LAMMPS box changed during force evaluation; shrink-wrap boundaries are unsupported")
            ids = self._validate_atoms()  # Neighbor rebuilds may reorder atoms.
            energy = float(self.engine.extract_compute("pyrpmd_pe", 0, 0))
            raw = np.asarray(self.engine.numpy.extract_atom("f")[:n]).copy()
            ordered = np.empty_like(raw)
            ordered[ids-1] = raw
            tensor = np.asarray(self.engine.numpy.extract_compute("pyrpmd_pressure", 0, 1), float)
            xx, yy, zz, xy, xz, yz = tensor
            matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            volume = abs(np.linalg.det(self._box_matrix))
            energies.append(energy * self.energy_factor)
            forces.append(ordered * self.energy_factor * BOHR_ANGSTROM)
            virials.append(matrix * volume / self._nktv2p * self.energy_factor)
        return validate_result(PESResult(energies, forces, virial=virials), x.shape)

    def set_cell(self, cell):
        if self.closed:
            raise PotentialError("LAMMPS provider is closed")
        if not self.pbc.all():
            raise PotentialError("LAMMPS variable-cell updates require full periodicity")
        cell, _ = validate_cell_pbc(cell, self.pbc)
        if not np.allclose(np.triu(cell, 1), 0, rtol=0, atol=1e-12) or np.any(np.diag(cell) <= 0):
            raise PotentialError("LAMMPS requires a restricted-triclinic lower-triangular row cell")
        matrix = cell * BOHR_ANGSTROM
        if self.engine.extract_setting("triclinic") == 0 and np.any(np.tril(matrix, -1)):
            raise PotentialError("LAMMPS cannot add tilt to an orthogonal engine box; initialize a triclinic box")
        # reset_box() is restricted to empty simulations in supported LAMMPS
        # releases. change_box changes populated domains; omit remap because
        # the caller, not LAMMPS, scales the physical Cartesian coordinates.
        parts = ["change_box all"]
        for axis, lower, length in zip("xyz", self._origin, np.diag(matrix)):
            parts.append(f"{axis} final {lower:.17g} {lower+length:.17g}")
        if self.engine.extract_setting("triclinic"):
            parts.extend([f"xy final {matrix[1, 0]:.17g}",
                          f"xz final {matrix[2, 0]:.17g}",
                          f"yz final {matrix[2, 1]:.17g}"])
        self.engine.command(" ".join(parts) + " units box")
        self._validate_box(cell, self.pbc)
        self.cell = cell

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            close = getattr(engine, "close", None)
            if callable(close):
                close()  # Never finalize MPI or global LAMMPS/Kokkos state.
        self.closed = True
