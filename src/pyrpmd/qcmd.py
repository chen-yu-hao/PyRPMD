"""Fast quasi-centroid models for equivalent, nonperiodic bent triatomics.

This implements the *separable intramolecular* approximation of Fletcher et
al., JCP 155, 231101 (2021), doi:10.1063/5.0076704. Unconstrained PIMD supplies
the distributions of bead-averaged internal coordinates. Iterative Boltzmann
inversion (IBI) fits a correction to the original PES; classical dynamics on
that corrected PES uses one bead and the physical masses. This is neither an
adiabatic QCMD integrator nor a general molecular or intermolecular PMF.

All arrays use atomic units (angles in radians). The two terminal bonds and
all listed molecules share one correction, so their chemical equivalence is
an explicit modelling assumption. Isotopic or mass inequivalence is rejected.
"""

from __future__ import annotations

import copy
import json
import re

import numpy as np
from scipy.interpolate import CubicSpline

from .checkpoint import atomic_savez, load_checkpoint
from .cmd import _integer, _real, _scalar
from .io import fingerprint_payload, jsonable, referenced_files
from .potentials.base import Potential, PESResult, coordinates, validate_result
from .qcmd_geometry import TriatomicCoordinates


MODEL_KIND = "PyRPMD.fqcmd-model.v1"
INTERPOLATION = "clamped-cubic; bin-centres plus constant endpoint values; endpoint slopes=0"
SCOPE = "equivalent-nonperiodic-bent-triatomics; intramolecular corrections only"


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(jsonable(value), sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error


def qcmd_base_identity(potential_spec, system):
    """Fingerprint the base PES specification/content and molecular composition.

    Initial coordinates and trajectory seeds are deliberately excluded: the
    same fitted model can start different replicas of the same system. Local
    explicitly referenced PES sources are content-hashed, as for other PyRPMD
    workflows. Pin transitive Python imports and remote services separately.
    """
    if not isinstance(potential_spec, dict) or not potential_spec:
        raise ValueError("potential_spec must be a nonempty potential specification")
    if np.any(system.pbc) or system.cell is not None:
        raise ValueError("f-QCMD currently requires a nonperiodic system without a cell")
    return fingerprint_payload({
        "schema": "PyRPMD.fqcmd-base.v1", "potential": potential_spec,
        "source_files": referenced_files({"potential": potential_spec}),
        "symbols": system.symbols, "masses": system.masses,
        "charge": system.charge, "multiplicity": system.multiplicity,
    })


def _validate_base_potential(base):
    """Do not hide physical provider requirements behind a corrected wrapper."""
    if not callable(getattr(base, "evaluate", None)):
        raise TypeError("base must implement the physical evaluate contract")
    metadata = getattr(base, "metadata", {}) or {}
    required = getattr(base, "required_bond_constraints", None)
    if required is None:
        required = metadata.get("required_bond_constraints", ())
    if required is not None and len(required):
        raise ValueError("f-QCMD does not support a base PES with required bond constraints")
    # In PyRPMD ``metadata['periodic']`` can declare support for the optional
    # evaluate(..., cell, pbc) contract (e.g. FreePotential), not an active cell.
    if (np.any(getattr(base, "pbc", False)) or getattr(base, "cell", None) is not None
            or metadata.get("boundary_conditions") == "periodic"):
        raise ValueError("f-QCMD requires a nonperiodic base PES without a cell")
    if any(key in metadata for key in ("centroid_pmf_beta_au", "quasicentroid_pmf_beta_au")):
        raise ValueError("f-QCMD corrections require the original physical PES, not an already fitted PMF")


class QCMDModel:
    """Temperature-specific, conservative, tabulated f-QCMD corrections.

    ``grids=(bond_edges, angle_edges)`` defines fixed uniform histogram bins.
    ``corrections`` holds the correction at each bin centre, in hartree. The
    energy spline includes both grid endpoints, with the nearest centre's
    value and zero endpoint derivative. This explicit boundary rule covers
    the half-bins and never extrapolates outside the grid. Out-of-domain
    geometries raise, including during training; widen grids and refit.

    ``base_identity`` is a SHA-256 from :func:`qcmd_base_identity` for workflow
    use. The model is a correction, *not* a total potential. ``diagnostics``
    describe sampled distributions, not physical or dynamical convergence.
    Arrays and metadata are copied; scalar attributes are immutable.
    """

    def __init__(self, molecules, masses, beta, grids, corrections, *,
                 base_identity, training_beads, training=None, target_counts=None,
                 diagnostics=None):
        object.__setattr__(self, "_sealed", False)
        self.masses = _real(masses, "masses")
        if self.masses.ndim != 1 or not len(self.masses) or np.any(self.masses <= 0):
            raise ValueError("masses must be a nonempty positive vector")
        self.coordinates = TriatomicCoordinates(molecules, len(self.masses))
        self.molecules = np.asarray(self.coordinates.molecules, dtype=np.int64).copy()
        if not np.array_equal(np.sort(self.molecules.ravel()), np.arange(len(self.masses))):
            raise ValueError("f-QCMD models require disjoint triatomics covering every atom")
        molecular_masses = self.masses[self.molecules]
        if (not np.array_equal(molecular_masses, np.broadcast_to(molecular_masses[0], molecular_masses.shape))
                or molecular_masses[0, 1] != molecular_masses[0, 2]):
            raise ValueError("pooled f-QCMD requires equivalent molecular and terminal masses; isotope mixtures are unsupported")
        self.beta = _scalar(beta, "beta")
        self.training_beads = _integer(training_beads, "training_beads")
        if not isinstance(base_identity, str) or re.fullmatch(r"[0-9a-f]{64}", base_identity) is None:
            raise ValueError("base_identity must be a lowercase SHA-256 fingerprint")
        self.base_identity = base_identity
        if not isinstance(grids, (tuple, list)) or len(grids) != 2:
            raise ValueError("grids must contain bond and angle histogram edges")
        if not isinstance(corrections, (tuple, list)) or len(corrections) != 2:
            raise ValueError("corrections must contain bond and angle values")
        self.grids = tuple(_real(edge, "grid edges") for edge in grids)
        self.corrections = tuple(_real(c, "correction") for c in corrections)
        splines = []
        centres = []
        for name, edge, correction in zip(("bond", "angle"), self.grids, self.corrections):
            if (edge.ndim != 1 or len(edge) < 5 or np.any(np.diff(edge) <= 0)
                    or not np.allclose(np.diff(edge), edge[1] - edge[0], rtol=1e-10, atol=0)):
                raise ValueError(f"{name} grid needs at least four uniform, increasing bins")
            if edge[0] <= 0 or (name == "angle" and edge[-1] >= np.pi):
                raise ValueError("bond grids must be positive and angle grids strictly inside (0,pi)")
            centre = (edge[1:] + edge[:-1]) / 2
            centres.append(centre)
            if correction.shape != centre.shape:
                raise ValueError(f"{name} correction must have one value per histogram bin")
            knots = np.r_[edge[0], centre, edge[-1]]
            values = np.r_[correction[0], correction, correction[-1]]
            splines.append(CubicSpline(knots, values, bc_type="clamped", extrapolate=False))
        self.centres = tuple(centres)
        self._splines = tuple(splines)
        if target_counts is None:
            target_counts = tuple(np.zeros_like(c) for c in self.corrections)
        if not isinstance(target_counts, (tuple, list)) or len(target_counts) != 2:
            raise ValueError("target_counts must contain bond and angle counts")
        self.target_counts = tuple(_real(c, "target counts") for c in target_counts)
        if any(c.shape != v.shape or np.any(c < 0) for c, v in zip(self.target_counts, self.corrections)):
            raise ValueError("target counts must be nonnegative and match their grids")
        self._training = _json_copy({} if training is None else training, "training metadata")
        self._diagnostics = _json_copy([] if diagnostics is None else diagnostics, "diagnostics")
        if not isinstance(self._training, dict) or not isinstance(self._diagnostics, list):
            raise ValueError("training must be a mapping and diagnostics a list")
        for array in (self.masses, self.molecules, *self.grids, *self.centres,
                      *self.corrections, *self.target_counts):
            array.flags.writeable = False
        self.model_identity = fingerprint_payload(self.to_dict())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("QCMDModel is immutable; create a new fitted model")
        object.__setattr__(self, name, value)

    @property
    def training(self):
        return copy.deepcopy(self._training)

    @property
    def diagnostics(self):
        return copy.deepcopy(self._diagnostics)

    def to_dict(self):
        """Complete scientific model identity, excluding storage location."""
        return {
            "format": MODEL_KIND, "method": "f-qcmd", "scope": SCOPE,
            "pooling": "all equivalent molecules and both terminal bonds",
            "interpolation": INTERPOLATION, "domain_policy": "error",
            "molecules": self.molecules.tolist(), "masses": self.masses.tolist(),
            "beta": self.beta, "training_beads": self.training_beads,
            "base_identity": self.base_identity,
            "grids": [a.tolist() for a in self.grids],
            "corrections": [a.tolist() for a in self.corrections],
            "target_counts": [a.tolist() for a in self.target_counts],
            "training": self.training, "diagnostics": self.diagnostics,
            "converged": False, "physical_converged": False,
        }

    def save(self, path):
        """Atomically save an integrity-checked, non-pickle model container."""
        atomic_savez(path, kind=MODEL_KIND, metadata=json.dumps(self.to_dict(), allow_nan=False),
                     model_identity=self.model_identity)

    @classmethod
    def load(cls, path):
        with load_checkpoint(path, expected_kind=MODEL_KIND, require_integrity=True) as data:
            raw = json.loads(str(data["metadata"]))
            identity = str(data["model_identity"])
        if not isinstance(raw, dict) or set(raw) != {
            "format", "method", "scope", "pooling", "interpolation", "domain_policy",
            "molecules", "masses", "beta", "training_beads", "base_identity", "grids",
            "corrections", "target_counts", "training", "diagnostics", "converged", "physical_converged",
        }:
            raise ValueError("unsupported f-QCMD model fields")
        model = cls(**{key: raw[key] for key in (
            "molecules", "masses", "beta", "grids", "corrections", "base_identity",
            "training_beads", "training", "target_counts", "diagnostics")})
        if model.to_dict() != raw or model.model_identity != identity:
            raise ValueError("f-QCMD model identity or method convention changed")
        return model

    def validate_system(self, system, beta, *, base_identity=None):
        """Reject periodic, constrained, composition, isotope and PES mismatch."""
        if system.cell is not None or np.any(system.pbc):
            raise ValueError("f-QCMD currently requires a nonperiodic system without a cell")
        if system.fixed_atoms or system.bond_constraints:
            raise ValueError("f-QCMD requires all atoms mobile without bond constraints")
        if (system.natoms != len(self.masses) or not np.array_equal(system.masses, self.masses)):
            raise ValueError("f-QCMD fitted masses/composition differ from this system")
        symbols = np.asarray(system.symbols)[self.molecules]
        if (not np.array_equal(symbols, np.broadcast_to(symbols[0], symbols.shape))
                or symbols[0, 1] != symbols[0, 2]):
            raise ValueError("f-QCMD pooling requires chemically equivalent triatomics and terminal atoms")
        if not np.isclose(_scalar(beta, "beta"), self.beta, rtol=1e-12, atol=0):
            raise ValueError("f-QCMD model temperature differs from the job temperature")
        if base_identity is not None and base_identity != self.base_identity:
            raise ValueError("f-QCMD base potential identity or source content changed")
        self.correction(self.coordinates.values(system.geometry[None]))

    def correction(self, internal):
        """Return summed correction energy[B] and dU/d(r1,r2,theta)[B,M,3]."""
        internal = np.asarray(internal)
        if (internal.dtype.kind not in "iuf" or internal.ndim != 3
                or internal.shape[1:] != (len(self.molecules), 3)
                or internal.shape[0] < 1 or not np.isfinite(internal).all()):
            raise ValueError("internal coordinates must be finite [batch,molecule,3]")
        bonds, angles = internal[..., :2], internal[..., 2]
        for name, values, grid in zip(("bond", "angle"), (bonds, angles), self.grids):
            if np.any(values < grid[0]) or np.any(values > grid[-1]):
                raise ValueError(f"f-QCMD {name} lies outside fitted grid [{grid[0]}, {grid[-1]}]; widen the grid and refit")
        er, ea = self._splines[0](bonds), self._splines[1](angles)
        derivative = np.empty(internal.shape, dtype=np.float64)
        derivative[..., :2] = self._splines[0](bonds, 1)
        derivative[..., 2] = self._splines[1](angles, 1)
        energy = er.sum(axis=(1, 2)) + ea.sum(axis=1)
        if not np.isfinite(energy).all() or not np.isfinite(derivative).all():
            raise FloatingPointError("nonfinite f-QCMD correction")
        return energy, derivative


class QuasiCentroidPotential(Potential):
    """Base physical PES plus the model's conservative intramolecular correction.

    ``evaluate`` accepts arbitrary batch size, with unscaled physical forces.
    The wrapper borrows the base potential: its owner remains responsible for
    closing it. Production uses one bead; a batch here need not be a ring.
    """

    def __init__(self, base, model):
        if not isinstance(model, QCMDModel):
            raise TypeError("model must be a QCMDModel")
        _validate_base_potential(base)
        self.base = base
        self.model = model
        self.metadata = {
            "provider": "fqcmd_corrected_potential", "requires_atomic_units": True,
            "thread_safety": getattr(base, "metadata", {}).get("thread_safety", "context_serial"),
            "execution_device": getattr(base, "metadata", {}).get("execution_device", "cpu"),
            "quasicentroid_pmf_beta_au": model.beta, "model_identity": model.model_identity,
            "model_scope": SCOPE, "physical_converged": False,
        }

    def evaluate(self, q):
        q = coordinates(q)
        if q.shape[1] != len(self.model.masses):
            raise ValueError("f-QCMD model atom count differs from PES coordinates")
        internal = self.model.coordinates.values(q)
        energy, derivatives = self.model.correction(internal)
        forces = self.model.coordinates.forces(q, derivatives)
        base = validate_result(self.base.evaluate(q), q.shape)
        # Molecular correction forces sum to zero; use local origins to avoid
        # cancellation error after free translation of isolated molecules.
        virial = None
        if base.virial is not None:
            molq = q[:, self.model.molecules]
            molf = forces[:, self.model.molecules]
            extra = np.einsum("bmai,bmaj->bij", molq - molq[:, :, :1], molf)
            virial = base.virial + extra
        return PESResult(base.energy + energy, base.forces + forces,
                         base.status.copy(), base.messages, virial)


def fit_fqcmd(*args, **kwargs):
    """Fit a model using the restartable PIMD/IBI driver in :mod:`qcmd_fit`."""
    from .qcmd_fit import fit_fqcmd as fit
    return fit(*args, **kwargs)
