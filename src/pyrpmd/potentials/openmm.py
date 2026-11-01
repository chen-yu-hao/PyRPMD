"""Optional OpenMM force provider; PyRPMD owns all nuclear propagation.

Only non-virtual particles occur in PyRPMD arrays, in increasing OpenMM
particle-index order. OpenMM redistributes virtual-site forces to their parents.
No OpenMM integration step, thermostat, or constraint projection is performed.
"""

from __future__ import annotations

import hashlib
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
    """

    def __init__(self, model=None, symbols=None, *, factory=None,
                 factory_kwargs=None, cell=None, pbc=None, device="cpu",
                 platform=None, properties=None):
        self.closed = False
        self.context = self.integrator = None
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
        try:
            if self.pbc.all():
                # Use PyRPMD's cell already during Context construction; the
                # XML's default box must not choose a different PME grid or
                # reject an otherwise valid cutoff before setPeriodicBoxVectors.
                self.system.setDefaultPeriodicBoxVectors(*[
                    openmm.Vec3(*row) for row in self.cell * BOHR_NM])
            self.integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
            self.context = openmm.Context(self.system, self.integrator, backend, opts)
            if self.pbc.all():
                self._set_box(self.cell)
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

    def _set_box(self, cell):
        vectors = [self.openmm.Vec3(*row) for row in cell * BOHR_NM]
        self.context.setPeriodicBoxVectors(*vectors)

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
        self.context = None
        self.integrator = None
        self.closed = True
