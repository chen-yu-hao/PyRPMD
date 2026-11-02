"""Optional OpenMM force provider; PyRPMD owns all nuclear propagation.

Only non-virtual particles occur in PyRPMD arrays, in increasing OpenMM
particle-index order. OpenMM redistributes virtual-site forces to their parents.
No OpenMM integration step, thermostat, or constraint projection is performed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np

from ..constants import BOHR_M, KCAL_MOL_TO_AU
from ..geometry import validate_cell_pbc
from .base import (
    BackendUnavailableError, PESResult, Potential, PotentialError,
    coordinates, validate_result,
)
from .python import import_object

BOHR_NM = BOHR_M * 1e9
KJ_MOL_TO_AU = KCAL_MOL_TO_AU / 4.184


def _symbols(symbols):
    if symbols is None or isinstance(symbols, str) or not len(symbols):
        raise ValueError("OpenMM requires an ordered sequence of System symbols")
    return tuple("H" if s in ("D", "T") else str(s) for s in symbols)


def _box(cell, pbc):
    cell, pbc = validate_cell_pbc(cell, pbc)
    if pbc.any() and not pbc.all():
        raise PotentialError("OpenMM supports all or no periodic axes; partial PBC is unsupported")
    if pbc.any():
        # Do not silently rotate coordinates or change the physical lattice.
        scale = max(float(np.max(np.abs(cell))), 1.0)
        if not np.allclose(np.triu(cell, 1), 0, atol=1e-12*scale, rtol=0):
            raise PotentialError("OpenMM requires a reduced lower-triangular cell with row vectors")
        a, b, c = np.diag(cell)
        if (min(a, b, c) <= 0 or a < 2*abs(cell[1, 0])-1e-12*scale
                or a < 2*abs(cell[2, 0])-1e-12*scale
                or b < 2*abs(cell[2, 1])-1e-12*scale):
            raise PotentialError("OpenMM requires a reduced periodic cell (positive diagonals and bounded tilts)")
    return cell, pbc


class OpenMMPotential(Potential):
    """Evaluate a serialized or factory-built :class:`openmm.System`.

    ``model`` is a System object or the path to its XML serialization. A factory
    is called with ``factory_kwargs`` and must return a System, not a Simulation
    or Context. Its original object is copied through XML serialization so later
    external mutations cannot change this provider. Backend units are fixed.

    ``batch_mode="contexts"`` gives each batch row a persistent Context and
    evaluates independent rows concurrently, using at most ``max_workers`` host
    workers. Contexts grow to the largest batch seen and remain resident until
    :meth:`close`. Select CPU ``properties={"Threads": "1"}`` when assigning
    cores to these workers. GPU copies use the explicitly selected device and
    precision; their Contexts require additional device memory. OpenMM's public
    API still transfers each copy separately. No integrator is ever stepped.

    The default ``batch_mode="serial"`` preserves the original one-Context
    execution and metadata, including existing run identities/checkpoints.
    """

    def __init__(self, model=None, symbols=None, *, factory=None,
                 factory_kwargs=None, cell=None, pbc=None, device="cpu",
                 platform=None, properties=None, batch_mode="serial",
                 max_workers=1):
        self.closed = False
        self.context = self.integrator = None
        self._contexts, self._integrators = [], []
        self._executor = None
        if batch_mode not in ("serial", "contexts"):
            raise ValueError("OpenMM batch_mode must be serial or contexts")
        if (isinstance(max_workers, bool) or not isinstance(max_workers, Integral)
                or max_workers < 1):
            raise ValueError("OpenMM max_workers must be a positive integer")
        if batch_mode == "serial" and max_workers != 1:
            raise ValueError("OpenMM max_workers > 1 requires batch_mode='contexts'")
        self.batch_mode, self.max_workers = batch_mode, int(max_workers)
        self.symbols = _symbols(symbols)
        self.cell, self.pbc = _box(cell, pbc)
        if (model is None) == (factory is None):
            raise ValueError("OpenMM requires exactly one of model or factory")
        if factory is None and factory_kwargs:
            raise ValueError("OpenMM factory_kwargs require a factory")
        try:
            import openmm
            from openmm import unit
        except ImportError as exc:
            raise BackendUnavailableError("OpenMM provider requires pip install 'pyrpmd[openmm]'") from exc
        self.openmm, self.unit = openmm, unit
        candidate = import_object(factory)(**dict(factory_kwargs or {})) if factory else model
        if isinstance(candidate, (str, Path)):
            xml = Path(candidate).read_text(encoding="utf-8")
            candidate = openmm.XmlSerializer.deserialize(xml)
        if not isinstance(candidate, openmm.System):
            raise TypeError("OpenMM model/factory must supply an openmm.System")
        xml = openmm.XmlSerializer.serialize(candidate)
        self.system = openmm.XmlSerializer.deserialize(xml)
        self.model_sha256 = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        self.particle_indices = np.array([
            i for i in range(self.system.getNumParticles()) if not self.system.isVirtualSite(i)
        ], dtype=np.int64)
        if len(self.particle_indices) != len(self.symbols):
            raise PotentialError("OpenMM non-virtual particle count does not match System atom count")
        self._virtual_sites = len(self.particle_indices) != self.system.getNumParticles()
        if bool(self.system.usesPeriodicBoundaryConditions()) != bool(self.pbc.any()):
            raise PotentialError("OpenMM model periodicity must match System PBC")
        for force in self.system.getForces():
            name = force.__class__.__name__
            if "Barostat" in name or name in ("AndersenThermostat", "CMMotionRemover"):
                raise PotentialError(f"OpenMM {name} is a propagation operator; remove it from the force-only System")
        inverse = {int(p): i for i, p in enumerate(self.particle_indices)}
        self.required_bond_constraints = []
        for i in range(self.system.getNumConstraints()):
            a, b, distance = self.system.getConstraintParameters(i)
            if a not in inverse or b not in inverse:
                raise PotentialError("OpenMM constraints involving virtual sites are unsupported")
            self.required_bond_constraints.append((inverse[a], inverse[b],
                float(distance.value_in_unit(unit.nanometer))/BOHR_NM))
        self._constraints_bound = not self.required_bond_constraints
        requested = str(device).lower()
        if requested == "cuda":
            requested = "cuda:0"
        if requested == "cpu":
            selected = platform or "CPU"
            if selected not in ("CPU", "Reference"):
                raise ValueError("OpenMM device='cpu' requires CPU or Reference platform")
        elif requested.startswith("cuda:") and requested[5:].isdigit():
            selected = platform or "CUDA"
            if selected != "CUDA":
                raise ValueError("OpenMM CUDA device requires the CUDA platform")
        elif requested.startswith("opencl:") and requested[7:].isdigit():
            selected = platform or "OpenCL"
            if selected != "OpenCL":
                raise ValueError("OpenMM OpenCL device requires the OpenCL platform")
        else:
            raise ValueError("OpenMM device must be cpu, cuda[:INDEX], or opencl:INDEX")
        opts = dict(properties or {})
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in opts.items()):
            raise ValueError("OpenMM platform properties must map strings to strings")
        if requested != "cpu":
            device_index = requested.split(":", 1)[1]
            for key in ("DeviceIndex", selected + "DeviceIndex"):
                if key in opts and opts[key] != device_index:
                    raise ValueError("OpenMM property DeviceIndex conflicts with potential.device")
            opts.setdefault("DeviceIndex", device_index)
            legacy_precision = selected + "Precision"
            if legacy_precision in opts and "Precision" in opts and opts[legacy_precision] != opts["Precision"]:
                raise ValueError("OpenMM precision properties conflict")
            if legacy_precision not in opts:
                opts.setdefault("Precision", "double")
        try:
            backend = openmm.Platform.getPlatformByName(selected)
        except Exception as exc:
            raise BackendUnavailableError(f"OpenMM {selected} platform is unavailable: {exc}") from exc
        self._backend, self._properties = backend, opts.copy()
        try:
            if self.pbc.all():
                # Use PyRPMD's cell already during Context construction; the
                # XML's default box must not choose a different PME grid or
                # reject an otherwise valid cutoff before setPeriodicBoxVectors.
                self.system.setDefaultPeriodicBoxVectors(*[
                    openmm.Vec3(*row) for row in self.cell * BOHR_NM])
            self.context, self.integrator = self._new_context()
            self._contexts.append(self.context)
            self._integrators.append(self.integrator)
            if batch_mode == "contexts" and self.max_workers > 1:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="pyrpmd-openmm")
        except Exception as exc:
            self.close()
            raise PotentialError(f"Cannot initialize OpenMM {selected} context: {exc}") from exc
        self.metadata = {
            "provider": "openmm", "version": openmm.__version__,
            "model_sha256": self.model_sha256,
            "thread_safety": "context_serial", "execution_device": requested,
            "platform": selected, "platform_properties": opts,
            "periodic": bool(self.pbc.any()), "supports_pbc": True,
            "native_batch": False, "trajectory_batch_safe": True,
            "virial": False, "physical_particle_indices": self.particle_indices.tolist(),
            "required_bond_constraints": list(self.required_bond_constraints),
        }
        if batch_mode == "contexts":
            # Keep legacy metadata byte-for-byte compatible. The new explicit
            # execution policy is fingerprinted only when it is requested.
            self.metadata.update(batch_mode=batch_mode, max_workers=self.max_workers,
                                 batch_residency="persistent_context_per_row")

    def _new_context(self):
        integrator = self.openmm.VerletIntegrator(1.0 * self.unit.femtosecond)
        context = self.openmm.Context(self.system, integrator, self._backend,
                                      self._properties)
        if self.pbc.all():
            context.setPeriodicBoxVectors(*[
                self.openmm.Vec3(*row) for row in self.cell * BOHR_NM])
        return context, integrator

    def _ensure_contexts(self, count):
        """Publish new copies only after the complete allocation succeeds."""
        pending_contexts, pending_integrators = [], []
        try:
            for _ in range(len(self._contexts), count):
                context, integrator = self._new_context()
                pending_contexts.append(context)
                pending_integrators.append(integrator)
        except Exception as exc:
            # Release Contexts before their bound Integrators. Existing copies
            # are untouched, so a smaller subsequent request can still work.
            pending_contexts.clear()
            pending_integrators.clear()
            raise PotentialError(
                f"Cannot allocate OpenMM batch of {count} persistent Contexts: {exc}"
            ) from exc
        self._contexts.extend(pending_contexts)
        self._integrators.extend(pending_integrators)

    def _set_box(self, cell):
        vectors = [self.openmm.Vec3(*row) for row in cell * BOHR_NM]
        previous = [self.openmm.Vec3(*row) for row in self.cell * BOHR_NM]
        attempted = []
        try:
            for context in self._contexts:
                # Include the failing Context: a backend setter may mutate
                # its state before raising.
                attempted.append(context)
                context.setPeriodicBoxVectors(*vectors)
            self.system.setDefaultPeriodicBoxVectors(*vectors)
        except Exception as exc:
            try:
                for context in reversed(attempted):
                    context.setPeriodicBoxVectors(*previous)
                self.system.setDefaultPeriodicBoxVectors(*previous)
            except Exception as rollback_exc:
                self.close()
                raise PotentialError(
                    "OpenMM cell update and rollback failed; provider is closed"
                ) from rollback_exc
            raise PotentialError(f"OpenMM cell update failed; previous cell restored: {exc}") from exc

    def bind_system(self, system):
        if self.closed:
            raise PotentialError("OpenMM provider is closed")
        def get(key, default=None):
            return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)
        symbols = _symbols(get("symbols"))
        if symbols != self.symbols:
            raise PotentialError("OpenMM system atom identities/order differ from the provider")
        cell, pbc = _box(get("cell"), get("pbc"))
        if not np.array_equal(pbc, self.pbc):
            raise PotentialError("OpenMM model periodicity cannot change on rebind")
        if self.required_bond_constraints:
            # A force engine cannot enforce nuclear constraints. The canonical
            # System must carry the same constraints for PyRPMD's propagator.
            declared = get("bond_constraints", ())
            actual = {}
            for item in declared:
                if isinstance(item, dict):
                    pair = item.get("atoms", item.get("pair"))
                    distance = item.get("distance", item.get("length"))
                    if pair is None or distance is None:
                        raise PotentialError("Malformed System bond constraint for OpenMM")
                    a, b = pair
                else:
                    a, b, distance = item
                actual[tuple(sorted((int(a), int(b))))] = float(distance)
            for a, b, distance in self.required_bond_constraints:
                candidate = actual.get(tuple(sorted((a, b))))
                if candidate is None or not np.isclose(candidate, distance, rtol=1e-10, atol=1e-12):
                    raise PotentialError("OpenMM model constraints must also be declared as matching PyRPMD System bond_constraints")
        if pbc.all():
            self._set_box(cell)
        self.cell, self.pbc = cell, pbc
        self._constraints_bound = True

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("OpenMM provider is closed")
        if not self._constraints_bound:
            raise PotentialError("Bind OpenMM's constraints to matching PyRPMD System bond_constraints before evaluating")
        x = coordinates(x)
        if x.shape[1] != len(self.symbols):
            raise PotentialError("OpenMM atom count mismatch")
        if self.batch_mode == "contexts":
            return self._evaluate_contexts(x)
        energies, forces = [], []
        positions = np.zeros((self.system.getNumParticles(), 3), dtype=float)
        for frame in x:
            positions[self.particle_indices] = frame * BOHR_NM
            self.context.setPositions(positions * self.unit.nanometer)
            if self._virtual_sites:
                self.context.computeVirtualSites()
            state = self.context.getState(getEnergy=True, getForces=True)
            energies.append(state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) * KJ_MOL_TO_AU)
            raw = state.getForces(asNumpy=True).value_in_unit(self.unit.kilojoule_per_mole/self.unit.nanometer)
            forces.append(np.asarray(raw)[self.particle_indices] * KJ_MOL_TO_AU * BOHR_NM)
        return validate_result(PESResult(energies, forces), x.shape)

    def _evaluate_contexts(self, x):
        self._ensure_contexts(len(x))
        # Convert the complete host batch once. Only virtual sites are zeroed;
        # OpenMM recomputes them in each independent Context below.
        positions = np.zeros((len(x), self.system.getNumParticles(), 3), dtype=float)
        positions[:, self.particle_indices] = x * BOHR_NM
        energies = np.empty(len(x))
        forces = np.empty_like(x)

        def evaluate_row(index):
            context = self._contexts[index]
            try:
                context.setPositions(positions[index] * self.unit.nanometer)
                if self._virtual_sites:
                    context.computeVirtualSites()
                state = context.getState(getEnergy=True, getForces=True)
                energies[index] = (state.getPotentialEnergy().value_in_unit(
                    self.unit.kilojoule_per_mole) * KJ_MOL_TO_AU)
                raw = state.getForces(asNumpy=True).value_in_unit(
                    self.unit.kilojoule_per_mole/self.unit.nanometer)
                forces[index] = np.asarray(raw)[self.particle_indices] * KJ_MOL_TO_AU * BOHR_NM
            except Exception as exc:
                raise PotentialError(f"OpenMM batch row {index} failed: {exc}") from exc

        if self._executor is None or len(x) == 1:
            for index in range(len(x)):
                evaluate_row(index)
        else:
            futures = []
            try:
                for index in range(len(x)):
                    futures.append(self._executor.submit(evaluate_row, index))
            finally:
                # A failed frame must not leave other workers modifying their
                # Contexts after this method returns or a caller closes/retries.
                wait(futures)
            for future in futures:
                future.result()
        return validate_result(PESResult(energies, forces), x.shape)

    def set_cell(self, cell):
        if self.closed:
            raise PotentialError("OpenMM provider is closed")
        if not self.pbc.all():
            raise PotentialError("OpenMM cell updates require full periodicity")
        cell, _ = _box(cell, self.pbc)
        self._set_box(cell)
        self.cell = cell

    def close(self):
        # Delete Context before its bound Integrator, without finalizing any
        # global platform state shared with another process-local provider.
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.context = None
        self._contexts.clear()
        self.integrator = None
        self._integrators.clear()
        self.closed = True
