"""Experimental molecular quasi-centroids and fast QCMD by distribution matching.

The first implementation is for nonperiodic, fixed-topology bent triatomics.
Fletcher et al., JCP 155, 231101 (2021), DOI 10.1063/5.0076704, use ordinary
*unconstrained* PIMD distributions of bead-averaged internal coordinates, IBI
corrections to the original potential, and classical production dynamics. The
conditional PIMD sampler exported below is a separate research tool, not a
real-time adiabatic QCMD integrator. All quantities are in atomic units, with
angles in radians. No existing CMD implementation is changed by this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
from scipy.interpolate import CubicSpline

from ._qcmd_geometry import TriatomicQuasiCentroid, QuasiCentroidConstraint
from ._qcmd_sampling import ConstrainedQuasiCentroidSampler
from .checkpoint import atomic_savez, load_checkpoint
from .cmd import _integer, _real, _scalar, ibi_update
from .io import fingerprint_payload, jsonable, referenced_files
from .potentials.base import Potential, PESResult, coordinates, validate_result


__all__ = [
    "TriatomicQuasiCentroid", "QuasiCentroidConstraint",
    "ConstrainedQuasiCentroidSampler", "FQCMDModel", "FQCMDPotential",
    "coordinate_histograms", "update_fqcmd_model", "fqcmd_base_identity",
]

_MODEL_KIND = "PyRPMD.fqcmd-model.v1"
_INTERPOLATION = "not-a-knot-cubic;bounded-by-bin-edges"


def fqcmd_base_identity(potential_spec, system):
    """Fingerprint the original PES, its local files, and chemical identity.

    Starting coordinates are intentionally omitted: different thermal initial
    conditions can use the same fit. Masses, charge, spin, atom order, boundary
    conditions, provider options, and referenced source/model *contents* are
    retained. Indirect imports and remote services still need environment
    pinning, as with ordinary PyRPMD potential identities.
    """
    if not isinstance(potential_spec, Mapping):
        raise ValueError("potential_spec must be a provider specification")
    identity = {key: value for key, value in system.to_dict().items()
                if key != "geometry"}
    return fingerprint_payload({"schema": "fqcmd-base-v1", "potential": dict(potential_spec),
                                "files": referenced_files(dict(potential_spec)),
                                "system": identity})


def _groups_and_edges(geometry, groups, edges):
    if not isinstance(geometry, TriatomicQuasiCentroid):
        raise ValueError("geometry must be TriatomicQuasiCentroid")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(groups, dtype=object).flat):
        raise ValueError("coordinate group indices must be integers, not booleans")
    group = np.asarray(groups)
    shape = (len(geometry.molecules), 3)
    if group.dtype.kind not in "iu" or group.shape != shape or np.any(group < 0):
        raise ValueError(f"groups must be nonnegative integer coordinate types with shape {shape}")
    unique = np.unique(group)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("coordinate group indices must be contiguous starting at zero")
    if not isinstance(edges, (list, tuple)) or len(edges) != len(unique):
        raise ValueError("edges must provide one bin-edge vector per coordinate group")
    kinds = []
    checked = []
    for i, raw in enumerate(edges):
        columns = np.nonzero(group == i)[1]
        if np.any(columns == 2) and np.any(columns != 2):
            raise ValueError("a coordinate group cannot mix bond lengths and angles")
        kind = "angle" if columns[0] == 2 else "bond"
        grid = _real(raw, f"edges[{i}]")
        if grid.ndim != 1 or len(grid) < 5 or np.any(np.diff(grid) <= 0):
            raise ValueError("each grid needs at least five strictly increasing finite bin edges")
        if grid[0] <= 0 or (kind == "angle" and grid[-1] >= np.pi):
            raise ValueError("bond edges must be positive; angle edges must lie strictly inside (0,pi)")
        grid.flags.writeable = False
        checked.append(grid)
        kinds.append(kind)
    group = np.array(group, dtype=np.int64, copy=True)
    group.flags.writeable = False
    return group, tuple(checked), tuple(kinds)


def coordinate_histograms(positions, geometry, groups, edges, *, bead_average):
    """Accumulate explicit molecular-coordinate counts without dropping tails.

    For the quantum target use ``bead_average=True`` and [frame,bead,atom,3]
    positions. For each classical IBI iteration use ``False`` and
    [frame,atom,3]. Coordinates sharing a group are pooled; e.g. [[0,0,1]]
    enforces equivalent OH bonds in water. Every value must be within the
    specified edges. Counts use the *raw coordinate measure* dR or dTheta.
    The radial/angle Jacobians cancel in an IBI ratio on the same grid; these
    counts must not be used as a one-shot Cartesian Boltzmann potential.
    """
    if not isinstance(bead_average, (bool, np.bool_)):
        raise ValueError("bead_average must be boolean")
    group, grids, _ = _groups_and_edges(geometry, groups, edges)
    q = _real(positions, "positions")
    if q.ndim != (4 if bead_average else 3) or min(q.shape, default=0) < 1:
        raise ValueError("positions need [frame,bead,atom,3] for PIMD or [frame,atom,3] for classical data")
    values = geometry.internal_coordinates(q)
    if bead_average:
        values = values.mean(axis=1)
    result = []
    for i, grid in enumerate(grids):
        selected = values[:, group == i].reshape(-1)
        if np.any(selected < grid[0]) or np.any(selected > grid[-1]):
            raise ValueError(f"coordinate group {i} leaves the histogram domain; extend the explicit grid")
        result.append(np.histogram(selected, bins=grid)[0])
    return tuple(result)


class FQCMDModel:
    """Portable, temperature- and isotope-specific intramolecular IBI tables.

    ``corrections[i]`` is Delta V in hartree at the bin centres of ``edges[i]``.
    A not-a-knot cubic interpolates the *energy*, and its analytic derivative
    supplies conservative forces. The supported closed domain is the bin-edge
    interval (the two half-bins use the same end polynomial); evaluation beyond
    it fails. There is no silent constant/linear extrapolation or force clipping.

    The original PES retains all couplings. Only its nuclear quantum correction
    is separable. Matching marginals is not a proof of a correct joint PMF or
    converged spectroscopy. ``provenance`` should record PIMD temperature,
    bead count, time step, sampling length/seeds, and dataset content digests.
    Each IBI update automatically adds its count digests and fitting settings.
    """

    def __init__(self, geometry, groups, edges, corrections, *, beta, base_identity,
                 provenance=None):
        self.groups, self.edges, self.kinds = _groups_and_edges(geometry, groups, edges)
        # Own the molecular definition as well as every table and provenance
        # value. Reusing a geometry object elsewhere cannot retarget this fit.
        self.geometry = TriatomicQuasiCentroid(geometry.molecules, geometry.masses,
                                               singularity_tolerance=geometry.singularity_tolerance)
        self.beta = _scalar(beta, "beta")
        if (not isinstance(base_identity, str) or len(base_identity) != 64
                or any(c not in "0123456789abcdef" for c in base_identity)):
            raise ValueError("base_identity must be the SHA256 from fqcmd_base_identity")
        self.base_identity = base_identity
        if not isinstance(corrections, (tuple, list)) or len(corrections) != len(self.edges):
            raise ValueError("corrections need one energy vector per coordinate group")
        values, centers, splines = [], [], []
        for i, (edge, raw) in enumerate(zip(self.edges, corrections)):
            value = _real(raw, f"corrections[{i}]")
            if value.shape != (len(edge) - 1,):
                raise ValueError("each correction must contain one value per bin centre")
            center = .5 * (edge[1:] + edge[:-1])
            spline = CubicSpline(center, value, bc_type="not-a-knot", extrapolate=True)
            if not np.isfinite(spline.c).all():
                raise ValueError("correction interpolation overflowed; rescale the grid or energy")
            spline.c.flags.writeable = spline.x.flags.writeable = False
            value.flags.writeable = center.flags.writeable = False
            values.append(value)
            centers.append(center)
            splines.append(spline)
        self.corrections = tuple(values)
        self.centers = tuple(centers)
        self._splines = tuple(splines)
        if provenance is None:
            provenance = {}
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a JSON mapping")
        # Round trip owns the complete tree, rejecting nonportable/NaN values.
        self._provenance_json = json.dumps(jsonable(dict(provenance), callables=False),
                                           allow_nan=False, sort_keys=True)
        self._identity = fingerprint_payload(self.to_dict())
        self._sealed = True

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("FQCMDModel parameters are immutable; create a new model")
        object.__setattr__(self, name, value)

    @property
    def provenance(self):
        return json.loads(self._provenance_json)

    @property
    def fingerprint(self):
        return self._identity

    def to_dict(self):
        return {"format": _MODEL_KIND, "interpolation": _INTERPOLATION,
                "measure": "raw-internal-coordinate-density",
                "molecules": np.asarray(self.geometry.molecules).tolist(),
                "masses": self.geometry.masses.tolist(),
                "singularity_tolerance": self.geometry.singularity_tolerance,
                "groups": self.groups.tolist(), "edges": [x.tolist() for x in self.edges],
                "corrections": [x.tolist() for x in self.corrections],
                "beta_au": self.beta, "base_identity": self.base_identity,
                "provenance": self.provenance}

    def validate_system(self, system, beta, *, base_identity):
        """Check production compatibility before a trajectory starts."""
        if np.any(system.pbc) or system.cell is not None:
            raise ValueError("f-QCMD currently supports only nonperiodic gas-phase triatomics")
        if system.fixed_atoms or system.bond_constraints:
            raise ValueError("f-QCMD requires unconstrained mobile molecular atoms")
        if (len(system.masses) != len(self.geometry.masses)
                or not np.array_equal(system.masses, self.geometry.masses)):
            raise ValueError("f-QCMD model masses/atom order differ from the training system")
        if not np.isclose(_scalar(beta, "beta"), self.beta, rtol=1e-12, atol=0):
            raise ValueError("f-QCMD model beta differs from the production temperature")
        if base_identity != self.base_identity:
            raise ValueError("f-QCMD original potential or chemical identity differs from training")
        self.geometry.internal_coordinates(system.geometry)

    def save(self, path):
        """Atomically save the full scientific model with verified array hashes."""
        if fingerprint_payload(self.to_dict()) != self._identity:
            raise ValueError("f-QCMD model parameters changed; create a new model")
        metadata = self.to_dict()
        for field in ("molecules", "masses", "groups", "edges", "corrections"):
            del metadata[field]
        arrays = {"molecules": self.geometry.molecules, "masses": self.geometry.masses,
                  "groups": self.groups}
        for i, (edge, correction) in enumerate(zip(self.edges, self.corrections)):
            arrays[f"edges_{i}"] = edge
            arrays[f"correction_{i}"] = correction
        atomic_savez(path, kind=_MODEL_KIND, metadata=json.dumps(metadata, allow_nan=False),
                     fingerprint=self._identity, **arrays)

    @classmethod
    def load(cls, path):
        with load_checkpoint(path, expected_kind=_MODEL_KIND, require_integrity=True) as saved:
            metadata = json.loads(str(saved["metadata"]))
            expected = {"format", "interpolation", "measure", "beta_au", "base_identity", "provenance",
                        "singularity_tolerance"}
            if (not isinstance(metadata, dict) or set(metadata) != expected
                    or metadata["interpolation"] != _INTERPOLATION
                    or metadata["measure"] != "raw-internal-coordinate-density"):
                raise ValueError("unsupported f-QCMD model schema or interpolation")
            geometry = TriatomicQuasiCentroid(saved["molecules"], saved["masses"],
                                               singularity_tolerance=metadata["singularity_tolerance"])
            groups = np.asarray(saved["groups"])
            if groups.dtype.kind not in "iu" or groups.size == 0:
                raise ValueError("invalid saved f-QCMD groups")
            # Bound the inventory by its actual array count before using max.
            n = int(groups.max()) + 1
            keys = {"metadata", "fingerprint", "molecules", "masses", "groups"}
            if n > len(saved.files):
                raise ValueError("invalid saved f-QCMD group inventory")
            keys |= {f"{name}_{i}" for name in ("edges", "correction") for i in range(n)}
            if set(saved.files) != keys:
                raise ValueError("invalid saved f-QCMD table inventory")
            model = cls(geometry, groups, [saved[f"edges_{i}"] for i in range(n)],
                        [saved[f"correction_{i}"] for i in range(n)], beta=metadata["beta_au"],
                        base_identity=metadata["base_identity"], provenance=metadata["provenance"])
            if model.fingerprint != str(saved["fingerprint"]):
                raise ValueError("f-QCMD model scientific fingerprint changed")
        return model


class FQCMDPotential(Potential):
    """Original physical PES plus bounded, invariant f-QCMD corrections.

    The wrapper borrows the original PES: its owner must close it. It does not
    run another path integral, rescale bead forces, or modify the original PES.
    Production must use one bead and the physical masses at the training beta.
    """

    def __init__(self, base_potential, model):
        if not isinstance(model, FQCMDModel) or not callable(getattr(base_potential, "evaluate", None)):
            raise ValueError("FQCMDPotential requires a physical PES and an FQCMDModel")
        base_metadata = getattr(base_potential, "metadata", {}) or {}
        if any(key in base_metadata for key in ("centroid_pmf_beta_au", "quasicentroid_pmf_beta_au")):
            raise ValueError("f-QCMD corrections require the original physical PES, not an already fitted PMF")
        if base_metadata.get("execution_device", "cpu") != "cpu":
            raise ValueError("the f-QCMD reference correction wrapper requires an explicit CPU PES")
        # metadata['periodic'] advertises a capability (e.g. FreePotential),
        # not active boundary conditions. Inspect configured PBC instead.
        if (base_metadata.get("boundary_conditions") == "periodic"
                or np.asarray(getattr(base_potential, "pbc", False), dtype=bool).any()):
            raise ValueError("f-QCMD corrections currently require a nonperiodic PES")
        self.base_potential, self._model = base_potential, model
        self.metadata = {"provider": "fqcmd_intramolecular_correction",
                         "requires_atomic_units": True, "execution_device": "cpu",
                         "thread_safety": base_metadata.get("thread_safety", "context_serial"),
                         "quasicentroid_pmf_beta_au": model.beta,
                         "fqcmd_model_sha256": model.fingerprint,
                         "approximation": "separable-intramolecular-IBI",
                         "physical_converged": False}
        self._group_indices = [np.nonzero(model.groups == i) for i in range(len(model.edges))]

    @property
    def model(self):
        """The immutable fit used to construct this wrapper."""
        return self._model

    def evaluate(self, x):
        q = coordinates(x)
        model = self.model
        values = model.geometry.internal_coordinates(q)
        gradients = model.geometry.internal_gradients(q)
        energy = np.zeros(len(q))
        derivative = np.zeros_like(values)
        for i, (index, edge, spline) in enumerate(zip(self._group_indices, model.edges, model._splines)):
            selected = values[:, index[0], index[1]]
            if np.any(selected < edge[0]) or np.any(selected > edge[-1]):
                raise ValueError(f"f-QCMD coordinate group {i} leaves the fitted domain [{edge[0]}, {edge[-1]}]")
            energy += np.sum(spline(selected), axis=1)
            derivative[:, index[0], index[1]] = spline(selected, 1)
        local_force = -np.einsum("bmc,bmcax->bmax", derivative, gradients)
        extra = np.zeros_like(q)
        # Triatomics are disjoint: direct scatter has no duplicate-index loss.
        extra[:, np.asarray(model.geometry.molecules)] = local_force
        base = validate_result(self.base_potential.evaluate(q), q.shape)
        virial = (None if base.virial is None else
                  base.virial + np.einsum("bni,bnj->bij", q, extra))
        return validate_result(PESResult(base.energy + energy, base.forces + extra,
                                         base.status.copy(), tuple(base.messages), virial), q.shape)


def _counts(value, edges, name):
    if not isinstance(value, (tuple, list)) or len(value) != len(edges):
        raise ValueError(f"{name} must contain one count vector per group")
    result = []
    for i, (raw, grid) in enumerate(zip(value, edges)):
        count = _real(raw, f"{name}[{i}]")
        if count.shape != (len(grid) - 1,) or np.any(count < 0) or not 0 < count.sum() < np.inf:
            raise ValueError(f"{name} needs nonnegative counts with positive finite total")
        result.append(count)
    return tuple(result)


def update_fqcmd_model(model, target_counts, sampled_counts, *, damping=1.,
                      regularization=0., polynomial_degree=None):
    """Perform one deterministic, auditable f-QCMD IBI update.

    Counts must refer to the model's groups and edges. Each distribution is
    independently normalized, including bin widths. The update is
    Delta V <- Delta V + damping/beta * log(p_sample/p_target).
    ``regularization`` is an explicit common density floor relative to the
    largest density (see :func:`pyrpmd.cmd.ibi_update`). Empty bins fail when
    it is zero. There is no automatic tail regularization.

    Optionally fit the *increment* to a polynomial of degree 0..8. Its weighted
    least squares uses overlap counts n_t*n_s/(n_t+n_s); bins with zero
    overlap do not invent observations. Evaluating this polynomial across the
    specified grid is an additional, explicitly recorded basis approximation.
    The returned diagnostic compares the *supplied pre-update samples*, not
    a hypothetical equilibrium sample under the newly fitted model.
    """
    if not isinstance(model, FQCMDModel):
        raise ValueError("model must be FQCMDModel")
    damping = _scalar(damping, "damping")
    regularization = _scalar(regularization, "regularization", positive=False)
    if damping > 1:
        raise ValueError("damping must lie in (0,1]")
    if polynomial_degree is not None:
        polynomial_degree = _integer(polynomial_degree, "polynomial_degree", 0)
        if polynomial_degree > 8:
            raise ValueError("polynomial_degree must not exceed eight")
    targets = _counts(target_counts, model.edges, "target_counts")
    samples = _counts(sampled_counts, model.edges, "sampled_counts")
    updates, diagnostics = [], []
    for edge, center, old, target, sample in zip(model.edges, model.centers, model.corrections,
                                               targets, samples):
        pt, ps = target / target.sum(), sample / sample.sum()
        dt, ds = pt / np.diff(edge), ps / np.diff(edge)
        increment = ibi_update(np.zeros_like(old), dt, ds, beta=model.beta,
                               damping=damping, regularization=regularization)
        fit_rmse = None
        occupied = (target > 0) & (sample > 0)
        if polynomial_degree is not None:
            if np.count_nonzero(occupied) < polynomial_degree + 1:
                raise ValueError("insufficient common occupied bins to identify the polynomial IBI increment")
            small = np.minimum(target[occupied], sample[occupied])
            large = np.maximum(target[occupied], sample[occupied])
            weight = small / (1 + small / large)
            fit, info = np.polynomial.Polynomial.fit(
                center[occupied], increment[occupied], polynomial_degree,
                w=np.sqrt(weight / weight.max()), domain=[center[0], center[-1]], full=True)
            if info[1] != polynomial_degree + 1:
                raise ValueError("polynomial IBI increment is rank deficient")
            predicted = fit(center)
            fit_rmse = float(np.sqrt(np.average((predicted[occupied] - increment[occupied]) ** 2,
                                                weights=weight)))
            increment = predicted
        updated = old + increment
        # A deterministic energy gauge changes no force or normalized measure.
        updated -= updated[len(updated) // 2]
        if not np.isfinite(updated).all():
            raise ValueError("IBI correction overflowed")
        updates.append(updated)
        diagnostics.append({"l1_probability_error": float(np.sum(np.abs(pt - ps))),
                            "hellinger_distance": float(np.linalg.norm(np.sqrt(pt) - np.sqrt(ps)) / np.sqrt(2)),
                            "target_total": float(target.sum()), "sampled_total": float(sample.sum()),
                            "common_occupied_bins": int(occupied.sum()),
                            "increment_fit_rmse_hartree": fit_rmse})
    settings = {"damping": damping, "regularization": regularization,
                "polynomial_degree": polynomial_degree,
                "target_counts_sha256": fingerprint_payload(targets),
                "sampled_counts_sha256": fingerprint_payload(samples),
                "parent_model_sha256": model.fingerprint}
    provenance = model.provenance
    history = provenance.setdefault("ibi_history", [])
    if not isinstance(history, list):
        raise ValueError("provenance.ibi_history must be a list")
    history.append(settings)
    updated = FQCMDModel(model.geometry, model.groups, model.edges, updates, beta=model.beta,
                         base_identity=model.base_identity, provenance=provenance)
    return updated, {"iteration": len(history), "sampled_model_sha256": model.fingerprint,
                     "updated_model_sha256": updated.fingerprint,
                     "comparison": "pre-update classical vs target marginals", "groups": diagnostics,
                     "converged": False, "physical_converged": False}
