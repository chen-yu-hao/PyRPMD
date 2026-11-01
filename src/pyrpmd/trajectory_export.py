"""Atomic, streaming exports in extxyz, H5MD and ASE trajectory formats.

Internal arrays use atomic units. Exported coordinates remain unwrapped; no
bead wrapping, atom replication or guessed topology is performed. Materials
archives contain centroids, not the individual bead configurations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numbers
import os
from pathlib import Path
import tempfile

import numpy as np

from . import __version__
from .checkpoint import load_checkpoint
from .constants import AMU_TO_AU, BOHR_ANGSTROM, FS_TO_AU, PS_TO_AU
from .geometry import validate_cell_pbc
from .system import System
from .trajectory import TrajectoryError, iter_chunks


@dataclass
class TrajectoryFrame:
    """One configuration, in bohr/atomic time; velocities are bohr/atomic time."""

    positions: np.ndarray
    step: int
    time: float
    velocities: np.ndarray | None = None
    cell: np.ndarray | None = None


def _frame(frame, system):
    if not isinstance(frame, TrajectoryFrame):
        raise ValueError("trajectory frames must be TrajectoryFrame instances")
    def array(value, shape, name):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError(f"{name} must be finite real {shape} values")
        return np.array(raw, dtype=float, copy=True)
    q = array(frame.positions, (system.natoms, 3), "positions")
    v = None if frame.velocities is None else array(frame.velocities, q.shape, "velocities")
    if (isinstance(frame.step, (bool, np.bool_)) or not isinstance(frame.step, numbers.Integral)
            or not 0 <= frame.step <= np.iinfo(np.int64).max):
        raise ValueError("trajectory step must be a nonnegative int64")
    if (isinstance(frame.time, (bool, np.bool_)) or not isinstance(frame.time, numbers.Real)
            or not np.isfinite(frame.time) or frame.time < 0):
        raise ValueError("trajectory time must be finite and nonnegative")
    cell, _ = validate_cell_pbc(system.cell if frame.cell is None else frame.cell, system.pbc)
    return TrajectoryFrame(q, int(frame.step), float(frame.time), v, cell)


def frames_from_states(states, system, *, bead=None):
    """Yield centroid frames, or one explicit 0-based bead, from live States.

    A frame owns its arrays, so a generator may advance and reuse one State.
    Momentum-to-velocity conversion uses the actual isotope masses.
    """
    if bead is not None and (isinstance(bead, (bool, np.bool_))
                             or not isinstance(bead, numbers.Integral) or bead < 0):
        raise ValueError("bead must be a nonnegative integer or None for centroids")
    for state in states:
        if (state.q.shape != state.p.shape or state.q.ndim != 3
                or state.q.shape[1:] != (system.natoms, 3)):
            raise ValueError("state atom count does not match export system")
        if bead is not None and bead >= len(state.q):
            raise ValueError("selected bead exceeds the state bead count")
        q = state.q.mean(axis=0) if bead is None else state.q[bead]
        p = state.p.mean(axis=0) if bead is None else state.p[bead]
        yield _frame(TrajectoryFrame(q, state.step, state.time,
                                    p / system.masses[:, None], system.cell), system)


class _ExtXYZ:
    def __init__(self, path, system, representation):
        self.stream = open(path, "w", encoding="utf-8", newline="\n")
        self.system = system
        self.representation = representation

    def write(self, frame):
        properties = "species:S:1:pos:R:3:masses:R:1:pyrpmd_symbol:S:1"
        if frame.velocities is not None:
            properties += ":velocities:R:3"
        cell = np.zeros((3, 3)) if frame.cell is None else frame.cell * BOHR_ANGSTROM
        lattice = " ".join(f"{v:.17g}" for v in cell.ravel())
        pbc = " ".join("T" if b else "F" for b in self.system.pbc)
        self.stream.write(f'{self.system.natoms}\nLattice="{lattice}" Properties={properties} '
                          f'pbc="{pbc}" Step={frame.step} Time={frame.time / FS_TO_AU:.17g} '
                          f'time_unit=fs position_unit=angstrom velocity_unit="angstrom/fs" '
                          f'pyrpmd_representation={self.representation}\n')
        positions = frame.positions * BOHR_ANGSTROM
        velocities = None if frame.velocities is None else frame.velocities * BOHR_ANGSTROM * FS_TO_AU
        for index, symbol in enumerate(self.system.symbols):
            # Isotopes remain chemical hydrogen for standard readers; masses
            # and the original label preserve the actual nuclear identity.
            element = "H" if symbol in ("D", "T") else symbol
            row = [element, *(f"{v:.17g}" for v in positions[index]),
                   f"{self.system.masses[index] / AMU_TO_AU:.17g}", symbol]
            if velocities is not None:
                row.extend(f"{v:.17g}" for v in velocities[index])
            self.stream.write(" ".join(row) + "\n")

    def close(self):
        self.stream.close()


class _ASETrajectory:
    def __init__(self, path, system, representation):
        try:
            from ase.io.trajectory import Trajectory
        except ImportError as error:
            raise ImportError("ASE .traj export requires PyRPMD[ase]") from error
        self.stream = Trajectory(str(path), mode="w")
        self.system, self.representation = system, representation

    def write(self, frame):
        from ase import Atoms, units
        atoms = Atoms(numbers=self.system.atomic_numbers,
                      positions=frame.positions * BOHR_ANGSTROM,
                      masses=self.system.masses / AMU_TO_AU,
                      cell=None if frame.cell is None else frame.cell * BOHR_ANGSTROM,
                      pbc=self.system.pbc)
        if frame.velocities is not None:
            atoms.set_velocities(frame.velocities * BOHR_ANGSTROM * FS_TO_AU / units.fs)
        atoms.info.update(step=frame.step, time_fs=frame.time / FS_TO_AU,
                          pyrpmd_symbols=list(self.system.symbols),
                          pyrpmd_representation=self.representation,
                          coordinate_convention="unwrapped")
        self.stream.write(atoms)

    def close(self):
        self.stream.close()


class _H5MD:
    def __init__(self, path, system, representation):
        import h5py
        self.stream = h5py.File(path, "w")
        self.system, self.count = system, 0
        h5md = self.stream.create_group("h5md")
        h5md.attrs["version"] = [1, 0]
        h5md.create_group("creator").attrs.update(name="PyRPMD", version=__version__)
        h5md.create_group("author").attrs["name"] = "unspecified"
        h5md.create_group("modules/units").attrs["version"] = [1, 0]
        self.particles = self.stream.create_group("particles/all")
        self.particles.attrs.update(pyrpmd_representation=representation,
                                    coordinate_convention="unwrapped")
        self.particles.create_dataset("species", data=system.atomic_numbers)
        self.particles.create_dataset("mass", data=system.masses / AMU_TO_AU).attrs["unit"] = "u"
        self.particles.create_dataset("pyrpmd_symbols", data=np.array(system.symbols, dtype="S3"))
        self.box = self.particles.create_group("box")
        self.box.attrs["dimension"] = 3
        self.box.attrs["boundary"] = np.array(["periodic" if b else "none" for b in system.pbc], dtype="S8")
        position = self.particles.create_group("position")
        self.steps = position.create_dataset("step", (0,), maxshape=(None,), dtype="i8", chunks=True)
        self.times = position.create_dataset("time", (0,), maxshape=(None,), dtype="f8", chunks=True)
        self.times.attrs["unit"] = "ps"
        self.positions = self._values(position, (system.natoms, 3), "Angstrom")
        self.velocities = None
        self.cells = None

    @staticmethod
    def _values(group, shape, unit):
        values = group.create_dataset("value", (0, *shape), maxshape=(None, *shape), dtype="f8", chunks=(1, *shape))
        values.attrs["unit"] = unit
        return values

    def _element(self, group, name, shape, unit):
        element = group.create_group(name)
        element["step"] = self.steps
        element["time"] = self.times
        return self._values(element, shape, unit)

    def write(self, frame):
        if self.count == 0:
            if frame.velocities is not None:
                self.velocities = self._element(self.particles, "velocity", (self.system.natoms, 3), "Angstrom ps-1")
            if frame.cell is not None:
                # A synchronized per-frame box also works with readers that
                # do not yet support H5MD's time-independent edges dataset.
                self.cells = self._element(self.box, "edges", (3, 3), "Angstrom")
        datasets = [(self.steps, frame.step), (self.times, frame.time / PS_TO_AU),
                    (self.positions, frame.positions * BOHR_ANGSTROM)]
        if self.velocities is not None:
            datasets.append((self.velocities, frame.velocities * BOHR_ANGSTROM * PS_TO_AU))
        if self.cells is not None:
            datasets.append((self.cells, frame.cell * BOHR_ANGSTROM))
        for dataset, value in datasets:
            dataset.resize(self.count + 1, axis=0)
            dataset[self.count] = value
        self.count += 1

    def close(self):
        self.stream.close()


def write_trajectory(path, frames, system, *, format=None, representation="configuration", overwrite=False):
    """Stream frames to a standard file, then atomically publish it.

    A failed generator, invalid later frame or failed writer leaves an existing
    destination unchanged. No output is published for an empty sequence. Units
    are angstrom/fs for extxyz and ASE metadata, angstrom/ps for H5MD. ASE's
    momenta use ASE's own consistent units. No PES or topology is invented.
    """
    if not isinstance(system, System):
        raise ValueError("trajectory export requires a validated System")
    if representation not in ("configuration", "centroid", "bead"):
        raise ValueError("representation must be configuration, centroid or bead")
    path = Path(path)
    selected = format or {".xyz":"extxyz", ".extxyz":"extxyz", ".h5":"h5md", ".h5md":"h5md", ".traj":"traj"}.get(path.suffix.lower())
    writers = {"extxyz":_ExtXYZ, "h5md":_H5MD, "traj":_ASETrajectory}
    if selected not in writers:
        raise ValueError("trajectory format must be extxyz, h5md or traj (or use a recognized suffix)")
    if path.exists() and not overwrite:
        raise FileExistsError(f"trajectory destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    writer = None
    count, previous, schema = 0, None, None
    try:
        writer = writers[selected](temporary, system, representation)
        for raw in frames:
            frame = _frame(raw, system)
            current_schema = (frame.velocities is not None, frame.cell is not None)
            if schema is not None and schema != current_schema:
                raise ValueError("velocity and cell availability must be consistent across frames")
            if previous is not None and (frame.step <= previous[0] or frame.time <= previous[1]):
                raise ValueError("trajectory steps and times must be strictly increasing")
            schema, previous = current_schema, (frame.step, frame.time)
            writer.write(frame)
            count += 1
        if not count:
            raise ValueError("cannot export an empty trajectory")
        writer.close(); writer = None
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication, including a racing exporter.
            os.link(temporary, path)
        return {"path": str(path), "format": selected, "frames": count,
                "atoms": system.natoms, "representation": representation,
                "coordinate_convention": "unwrapped"}
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _materials_source(source, replica):
    source = Path(source)
    if isinstance(replica, bool) or not isinstance(replica, numbers.Integral) or replica < 0:
        raise ValueError("replica must be a nonnegative integer")
    if source.is_dir():
        manifest = source / "manifest.json"
        if (manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("format")
                == "PyRPMD.materials-trajectory-stream.v1"):
            return manifest
        manifest = source / "samples" / f"materials_{replica:04d}" / "manifest.json"
        archive = source / "samples" / f"materials_{replica:04d}.npz"
        if manifest.exists() and archive.exists():
            raise ValueError("both stream and memory exports exist; select the intended manifest.json or .npz explicitly")
        if manifest.exists():
            return manifest
        if archive.exists():
            return archive
        raise ValueError("no exported materials trajectory found; analyze the completed/checkpointed run first")
    return source


def _metadata(value):
    if not isinstance(value, dict):
        raise TrajectoryError("materials trajectory metadata must be a mapping")
    expected = {"positions":"unwrapped_ring_polymer_centroids", "position_unit":"bohr",
                "velocity_unit":"bohr/atomic_time", "time_unit":"atomic_time"}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise TrajectoryError("unsupported materials coordinate convention or units")
    dt = value.get("production_timestep_au")
    neq = value.get("equilibration_steps")
    if (isinstance(dt, bool) or not isinstance(dt, numbers.Real) or not np.isfinite(dt) or dt <= 0
            or isinstance(neq, bool) or not isinstance(neq, numbers.Integral) or neq < 0):
        raise TrajectoryError("invalid materials production clock metadata")
    return System.from_dict(value["system"]), float(dt), int(neq)


def export_materials_trajectory(source, destination, *, replica=0, format=None, overwrite=False):
    """Export one materials centroid replica from a run, manifest or legacy NPZ.

    Reads the selected manifest prefix, with every chunk/hash checked. It never
    discovers frames by scanning orphan chunk files. A manifest is an analysis
    export; run ``pyrpmd analyze RUN --recompute`` to refresh it after resuming a simulation.
    """
    source = _materials_source(source, replica)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("trajectory export cannot replace its source")
    if source.suffix.lower() == ".json":
        saved = json.loads(source.read_text(encoding="utf-8"))
        if saved.get("format") != "PyRPMD.materials-trajectory-stream.v1":
            raise TrajectoryError("not a materials trajectory manifest")
        system, dt, neq = _metadata(saved["metadata"])
        schema = saved["trajectory"]["schema"]
        fields = ["sample_steps", "positions", "velocities"] + (["cells"] if "cells" in schema else [])
        chunks = iter_chunks(source.parent, saved["trajectory"], fields=fields)
        expected_frames = saved["trajectory"]["frames"]
        def frames():
            for chunk in chunks:
                for index, step in enumerate(chunk["sample_steps"]):
                    yield TrajectoryFrame(chunk["positions"][index], step, (int(step) - neq) * dt,
                                          chunk["velocities"][index],
                                          chunk["cells"][index] if "cells" in chunk else system.cell)
        # Never replace an integrity-checked chunk, even with --overwrite.
        if destination.resolve().parent == source.resolve().parent and destination.suffix == ".npz":
            raise ValueError("trajectory export cannot replace a source chunk")
        result = write_trajectory(destination, frames(), system, format=format,
                                  representation="centroid", overwrite=overwrite)
        if result["frames"] != expected_frames:  # iter_chunks also enforces this
            raise TrajectoryError("exported frame count disagrees with manifest")
    elif source.suffix.lower() == ".npz":
        with load_checkpoint(source, expected_kind="PyRPMD.materials-trajectory.v1", require_integrity=True) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            system, dt, neq = _metadata(metadata)
            steps, positions, velocities, times = (saved[key] for key in ("sample_steps", "positions", "velocities", "times"))
            if (steps.ndim != 1 or steps.dtype.kind not in "iu" or positions.shape != (len(steps), system.natoms, 3)
                    or velocities.shape != positions.shape or times.shape != steps.shape
                    or not np.allclose(times, (steps - neq) * dt, rtol=1e-12, atol=1e-12)):
                raise TrajectoryError("legacy materials trajectory arrays/clock disagree")
            cells = saved["cells"] if "cells" in saved.files else None
            if cells is not None and cells.shape != (len(steps), 3, 3):
                raise TrajectoryError("invalid legacy cell history")
            frames = (TrajectoryFrame(q, step, time, v, system.cell if cells is None else cells[index])
                      for index, (q, step, time, v) in enumerate(zip(positions, steps, times, velocities)))
            result = write_trajectory(destination, frames, system, format=format,
                                      representation="centroid", overwrite=overwrite)
    else:
        raise ValueError("materials export source must be a run directory, manifest.json or saved trajectory .npz")
    return {**result, "source": str(source)}
