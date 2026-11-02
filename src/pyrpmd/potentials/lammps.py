"""LAMMPS shared-library force adapter with explicit atom and unit contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
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
    """Evaluate a force model with private, persistent LAMMPS instances.

    An input script (``model``) or factory defines atom IDs, types, charges,
    topology and potential. Atom IDs must be consecutive 1..N in PyRPMD order.
    ``type_map`` identifies each LAMMPS type, starting with type 1. Supported
    backend unit styles are ``real`` and ``metal``; they are read from the engine.

    ``batch_mode="serial"`` preserves the original single-instance evaluator.
    ``batch_mode="persistent"`` retains an instance for each batch slot and
    reuses its neighbor lists until an atom moves half ``neighbor_skin`` (bohr).
    A no-integration force sweep refreshes forces and ghost coordinates on
    reuse; ``run 0 pre no`` alone would incorrectly return stale forces.

    An explicit private MPI ``communicator`` enables spatial decomposition
    within each instance. Every member must call the adapter collectively with
    identical input. ``COMM_WORLD`` itself is never accepted. Engine-side fixes
    are refused, including SHAKE, thermostats and force modifiers.
    """

    def __init__(self, symbols, model=None, *, factory=None, factory_kwargs=None,
                 type_map=None, cell=None, pbc=None, device="cpu", cmdargs=None,
                 library_name="", asset_files=(), batch_mode="serial",
                 neighbor_skin=None, communicator=None):
        self.closed = False
        self.engine = None
        self._replicas = []
        self._communicator = None
        self._owns_communicator = False
        self._comm_size = 1
        self._MPI = None
        self._neighbor_positions = None
        self._neighbor_ids = None
        self._rebuild_count = self._reuse_count = 0
        if batch_mode not in ("serial", "persistent"):
            raise ValueError("LAMMPS batch_mode must be serial or persistent")
        if neighbor_skin is not None and batch_mode != "persistent":
            raise ValueError("LAMMPS neighbor_skin requires batch_mode='persistent'")
        if neighbor_skin is None:
            neighbor_skin = 2.0 / BOHR_ANGSTROM
        if (isinstance(neighbor_skin, (bool, np.bool_)) or not np.isscalar(neighbor_skin)
                or np.asarray(neighbor_skin).dtype.kind not in "fiu"
                or not np.isfinite(neighbor_skin) or neighbor_skin <= 0):
            raise ValueError("LAMMPS neighbor_skin must be finite and positive, in bohr")
        self.batch_mode = batch_mode
        self.neighbor_skin = float(neighbor_skin)
        self.symbols = _chemical_symbols(symbols)
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        frozen_factory_kwargs = deepcopy(factory_kwargs)
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
        try:
            from mpi4py import MPI
        except ImportError:
            if communicator is not None:
                raise BackendUnavailableError("LAMMPS communicator requires mpi4py")
        else:
            self._MPI = MPI
            if communicator is not None:
                if not isinstance(communicator, MPI.Comm):
                    raise TypeError("LAMMPS communicator must be a private mpi4py communicator")
                if communicator == MPI.COMM_NULL or communicator.Is_inter():
                    raise ValueError("LAMMPS communicator must be a non-null intracommunicator")
                if MPI.Comm.Compare(communicator, MPI.COMM_WORLD) == MPI.IDENT:
                    raise ValueError("LAMMPS cannot share MPI.COMM_WORLD; pass a private Split or Dup communicator")
                self._communicator = communicator.Dup()
                self._owns_communicator = True
                self._comm_size = self._communicator.Get_size()
            elif MPI.COMM_WORLD.Get_size() > 1:
                self._communicator = MPI.COMM_SELF
        try:
            if factory:
                if cmdargs or library_name:
                    raise ValueError("LAMMPS factory owns cmdargs and library_name")
                self.engine = import_object(factory)(comm=self._communicator,
                                                      **deepcopy(frozen_factory_kwargs or {}))
            else:
                options = ["-log", "none", "-screen", "none", "-nocite"] + list(cmdargs or [])
                try:
                    self.engine = lammps(name=library_name, cmdargs=options, comm=self._communicator)
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
            if self.batch_mode == "persistent":
                # The public library interface does not expose neighbor->skin.
                # Establish an explicit, known skin instead of guessing it from
                # the model script or relying on a private C++ object layout.
                self.engine.command(f"neighbor {self.neighbor_skin * BOHR_ANGSTROM:.17g} bin")
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
            if self.batch_mode != "serial":
                self.metadata.update(batch_mode=self.batch_mode,
                                     neighbor_skin_bohr=self.neighbor_skin,
                                     neighbor_style="bin")
            if communicator is not None:
                self.metadata["ranks_per_instance"] = self._comm_size
            self._replica_kwargs = dict(
                symbols=self.symbols, model=str(Path(model).resolve()) if model is not None else None,
                factory=factory, factory_kwargs=frozen_factory_kwargs,
                type_map=self.type_map, cell=None if self.cell is None else self.cell.copy(),
                pbc=self.pbc.copy(), device=device, cmdargs=deepcopy(cmdargs),
                library_name=library_name, asset_files=tuple(self.asset_sha256),
                batch_mode=batch_mode, neighbor_skin=self.neighbor_skin if batch_mode == "persistent" else None,
                communicator=self._communicator if self._owns_communicator else None,
            )
        except BaseException:
            self.close()
            raise

    def _check_engine(self):
        error = None
        try:
            self._check_engine_local()
        except Exception as exc:
            error = exc
        self._collective_error(error, "engine validation")

    def _collective_error(self, error, operation):
        """Fail on all ranks before anyone enters the next backend collective."""
        if self._comm_size > 1:
            failures = self._communicator.allgather(None if error is None else str(error))
            if any(message is not None for message in failures):
                raise PotentialError(f"LAMMPS collective {operation} failed: " + next(
                    message for message in failures if message is not None))
        elif error is not None:
            raise error

    def _collective_identical(self, signature, description):
        if self._comm_size > 1:
            signatures = self._communicator.allgather(signature)
            if any(value != signature for value in signatures):
                raise PotentialError(f"LAMMPS communicator members must supply identical {description}")

    def _check_engine_local(self):
        if self.engine is None or not callable(getattr(self.engine, "extract_setting", None)):
            raise TypeError("LAMMPS factory must return a configured lammps.lammps instance")
        if self.engine.extract_setting("world_size") != self._comm_size:
            raise PotentialError("LAMMPS engine rank count does not match its private communicator")
        if self._owns_communicator:
            actual = self.engine.get_mpi_comm()
            if (actual is None
                    or self._MPI.Comm.Compare(actual, self._MPI.COMM_WORLD) == self._MPI.IDENT
                    or self._MPI.Comm.Compare(actual, self._communicator) not in (self._MPI.IDENT, self._MPI.CONGRUENT)):
                raise PotentialError("LAMMPS factory must use the supplied private communicator, never COMM_WORLD")
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
        if self.batch_mode == "persistent":
            for category in ("bond", "angle", "dihedral", "improper"):
                style = str(_text(self.engine.extract_global(f"{category}_style")))
                if style.startswith(("lepton", "hybrid")):
                    raise PotentialError("LAMMPS persistent mode does not support variable-driven or hybrid bonded styles")
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
        nlocal = self.engine.extract_setting("nlocal")
        if self.engine.get_natoms() != n or (self._comm_size == 1 and nlocal != n):
            raise PotentialError("LAMMPS atom count/ownership does not match System")
        ids = np.asarray(self.engine.numpy.extract_atom("id")[:nlocal], dtype=np.int64)
        valid = len(ids) == nlocal and np.all((ids >= 1) & (ids <= n))
        if self._comm_size > 1:
            valid = self._communicator.allreduce(bool(valid), op=self._MPI.LAND)
        if not valid:
            raise PotentialError("LAMMPS atom IDs must be consecutive 1..N in System order")
        types = np.asarray(self.engine.numpy.extract_atom("type")[:nlocal], dtype=np.int64)
        valid = (len(self.type_map) == self.engine.extract_setting("ntypes")
                 and np.all((types >= 1) & (types <= len(self.type_map))))
        if self._comm_size > 1:
            valid = self._communicator.allreduce(bool(valid), op=self._MPI.LAND)
        if not valid:
            raise PotentialError("LAMMPS type_map must identify every atom type")
        if self._comm_size == 1:
            if not np.array_equal(np.sort(ids), np.arange(1, n+1)):
                raise PotentialError("LAMMPS atom IDs must be consecutive 1..N in System order")
            ordered_types = types[np.argsort(ids)]
        else:
            atoms = np.zeros((n, 2), dtype=np.int64)
            np.add.at(atoms[:, 0], ids-1, 1)
            np.add.at(atoms[:, 1], ids-1, types)
            atoms = self._sum_array(atoms)
            if not np.all(atoms[:, 0] == 1):
                raise PotentialError("LAMMPS atom IDs must be consecutive 1..N in System order")
            ordered_types = atoms[:, 1]
        observed = tuple(self.type_map[int(t)-1] for t in ordered_types)
        if observed != self.symbols:
            raise PotentialError("LAMMPS type_map/atom IDs do not match System symbols/order")
        return ids

    def _sum_array(self, values):
        if self._comm_size == 1:
            return values
        result = np.empty_like(values)
        self._communicator.Allreduce(values, result, op=self._MPI.SUM)
        return result

    @property
    def engines(self):
        """Currently allocated instances, ordered by persistent batch slot."""
        return tuple(p.engine for p in [self, *self._replicas] if p.engine is not None)

    @property
    def batch_stats(self):
        """Force-sweep counts; these diagnostics do not change run identity."""
        providers = [self, *self._replicas]
        return {"instances": len(self.engines),
                "neighbor_rebuilds": sum(p._rebuild_count for p in providers),
                "neighbor_reuses": sum(p._reuse_count for p in providers)}

    def bind_system(self, system):
        if self.closed:
            raise PotentialError("LAMMPS provider is closed")
        def get(key, default=None):
            return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)
        error = None
        try:
            if _chemical_symbols(get("symbols")) != self.symbols:
                raise PotentialError("LAMMPS system atom identities/order differ from the provider")
            cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
            for provider in [self, *self._replicas]:
                provider._validate_box(cell, pbc)
        except Exception as exc:
            error = exc
        self._collective_error(error, "system binding")
        self._collective_identical((None if cell is None else cell.tobytes(), pbc.tobytes()),
                                   "System cell/PBC")
        for provider in [self, *self._replicas]:
            provider.cell, provider.pbc = cell, pbc

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("LAMMPS provider is closed")
        error = None
        try:
            x = coordinates(x)
            if x.shape[1] != len(self.symbols):
                raise PotentialError("LAMMPS atom count mismatch")
            # Validate the whole batch before modifying any instance. Wrapping
            # stays in a temporary copy, preserving unwrapped diffusion paths.
            frac = (x * BOHR_ANGSTROM-self._origin) @ np.linalg.inv(self._box_matrix)
            if np.any((frac[:, :, ~self.pbc] < 0) | (frac[:, :, ~self.pbc] >= 1)):
                raise PotentialError("LAMMPS coordinates lie outside a nonperiodic box boundary")
            frac[:, :, self.pbc] -= np.floor(frac[:, :, self.pbc])
            positions = frac @ self._box_matrix + self._origin
        except Exception as exc:
            error = exc
        self._collective_error(error, "input validation")
        if self._comm_size > 1:
            self._collective_identical((x.shape, hashlib.sha256(x.tobytes()).digest()),
                                       "coordinates")
        energies, forces, virials = [], [], []
        try:
            if self.batch_mode == "persistent":
                self._ensure_replicas(len(x))
                providers = [self, *self._replicas]
            else:
                providers = [self] * len(x)
            for provider, position in zip(providers, positions):
                energy, force, virial = provider._evaluate_frame(position)
                energies.append(energy)
                forces.append(force)
                virials.append(virial)
            return validate_result(PESResult(energies, forces, virial=virials), x.shape)
        except BaseException:
            # Never expose a partially evaluated batch or reuse an engine
            # whose neighbors/domain may have been corrupted by a failed call.
            self.close()
            raise

    def _ensure_replicas(self, count):
        if count <= len(self._replicas)+1:
            return
        model = self._replica_kwargs["model"]
        assets = dict(self.asset_sha256)
        if model is not None:
            assets[model] = self.metadata["model_sha256"]
        error = None
        try:
            if any(file_sha256(path) != digest for path, digest in assets.items()):
                raise PotentialError("LAMMPS model/assets changed while creating persistent batch instances")
        except Exception as exc:
            error = exc
        self._collective_error(error, "persistent model/asset validation")
        while len(self._replicas)+1 < count:
            provider = LAMMPSPotential(**self._replica_kwargs)
            try:
                self._collective_error(
                    PotentialError("LAMMPS factory returned inconsistent persistent batch instances")
                    if provider.metadata != self.metadata else None, "persistent instance validation")
                if self.cell is not None and not np.array_equal(provider.cell, self.cell):
                    provider.set_cell(self.cell)
                self._replicas.append(provider)
            except BaseException:
                provider.close()
                raise

    def _evaluate_frame(self, position):
        try:
            self._check_engine()
            ids = self._validate_atoms()
            origin, box, pbc = self._box_data()
            if (not np.array_equal(pbc, self.pbc)
                    or not np.allclose(box, self._box_matrix, rtol=1e-12, atol=1e-12)
                    or not np.allclose(origin, self._origin, rtol=1e-12, atol=1e-12)):
                raise PotentialError("LAMMPS box changed outside set_cell")
            nlocal = len(ids)
            self.engine.numpy.extract_atom("x")[:nlocal] = position[ids-1]
            # No kinetic contribution is requested; zero stale velocities as
            # well because velocity-dependent potentials are unsupported.
            self.engine.numpy.extract_atom("v")[:nlocal] = 0.0
            reuse = (self.batch_mode == "persistent" and self._neighbor_positions is not None
                     and np.array_equal(ids, self._neighbor_ids)
                     and np.max(np.linalg.norm(position-self._neighbor_positions, axis=1))
                     < 0.5 * self.neighbor_skin * BOHR_ANGSTROM)
            if self._comm_size > 1:
                # Ownership/order can change on only one rank. Every member
                # must enter the same LAMMPS run/setup collectives.
                reuse = self._communicator.allreduce(bool(reuse), op=self._MPI.LAND)
            # One Verlet force sweep without any fixes refreshes ghost atoms,
            # forces, energy and reciprocal-space virials, but cannot integrate
            # positions or velocities. Its backend clock is not PyRPMD time.
            self.engine.command("run 1 pre no post no" if reuse else "run 0 pre yes post no")
            if reuse:
                self._reuse_count += 1
            else:
                self._neighbor_positions = position.copy()
                self._rebuild_count += 1
            origin, matrix, pbc = self._box_data()
            if (not np.array_equal(pbc, self.pbc)
                    or not np.allclose(matrix, self._box_matrix, rtol=1e-12, atol=1e-12)
                    or not np.allclose(origin, self._origin, rtol=1e-12, atol=1e-12)):
                raise PotentialError("LAMMPS box changed during force evaluation; shrink-wrap boundaries are unsupported")
            ids = self._validate_atoms()  # Neighbor rebuilds may reorder atoms.
            if not reuse:
                self._neighbor_ids = ids.copy()
            energy = float(self.engine.extract_compute("pyrpmd_pe", 0, 0))
            raw = np.asarray(self.engine.numpy.extract_atom("f")[:len(ids)])
            ordered = np.zeros((len(self.symbols), 3))
            ordered[ids-1] = raw
            ordered = self._sum_array(ordered)
            tensor = np.asarray(self.engine.numpy.extract_compute("pyrpmd_pressure", 0, 1), float)
            xx, yy, zz, xy, xz, yz = tensor
            matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            volume = abs(np.linalg.det(self._box_matrix))
            return (energy * self.energy_factor,
                    ordered * self.energy_factor * BOHR_ANGSTROM,
                    matrix * volume / self._nktv2p * self.energy_factor)
        except Exception as exc:
            if isinstance(exc, PotentialError):
                raise
            raise PotentialError(f"LAMMPS force evaluation failed: {exc}") from exc

    def set_cell(self, cell):
        if self.closed:
            raise PotentialError("LAMMPS provider is closed")
        error = None
        try:
            if not self.pbc.all():
                raise PotentialError("LAMMPS variable-cell updates require full periodicity")
            cell, _ = validate_cell_pbc(cell, self.pbc)
            if not np.allclose(np.triu(cell, 1), 0, rtol=0, atol=1e-12) or np.any(np.diag(cell) <= 0):
                raise PotentialError("LAMMPS requires a restricted-triclinic lower-triangular row cell")
            matrix = cell * BOHR_ANGSTROM
            providers = [self, *self._replicas]
            for provider in providers:
                if provider.engine.extract_setting("triclinic") == 0 and np.any(np.tril(matrix, -1)):
                    raise PotentialError("LAMMPS cannot add tilt to an orthogonal engine box; initialize a triclinic box")
        except Exception as exc:
            error = exc
        self._collective_error(error, "cell validation")
        self._collective_identical(cell.tobytes(), "cell updates")
        try:
            for provider in providers:
                provider._set_cell_one(cell, matrix)
                provider._neighbor_positions = None
        except BaseException:
            # A failed domain update cannot be rolled back reliably by LAMMPS.
            # Closing every replica prevents reuse of a partially updated cell.
            self.close()
            raise

    def _set_cell_one(self, cell, matrix):
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
        self.closed = True
        first_error = None
        replicas, self._replicas = self._replicas, []
        for provider in replicas:
            try:
                provider.close()
            except Exception as exc:
                first_error = first_error or exc
        if self.engine is not None:
            engine, self.engine = self.engine, None
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()  # Never finalize MPI or global LAMMPS/Kokkos state.
                except Exception as exc:
                    first_error = first_error or exc
        if self._owns_communicator:
            self._owns_communicator = False
            communicator, self._communicator = self._communicator, None
            try:
                communicator.Free()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
