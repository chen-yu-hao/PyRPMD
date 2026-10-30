"""Independent classical triatomic PMF by integration over the third distance.

Atom order is Br,H,Cl, with x=r(Br,H), y=r(H,Cl), z=r(Br,Cl). After removing
translation and integrating overall rotation, the configurational measure is
proportional to x*y*z dx dy dz. Consequently the *classical* two-distance
density is x*y*integral(z*exp[-beta*(V+Wwall)], z=abs(x-y)..x+y).

This reference integrates the hidden coordinate rather than minimizing it.
It also integrates over each finite x/y bin, so a bin-center PES slice is
never used as a PMF oracle. It is valid for P=1, a rotationally invariant
distance-dependent PES and the stated common wall, not a quantum centroid
distribution. Missing translation/rotation constants cancel from PMF
differences. Normalized probabilities are conditional on the supplied finite
x/y rectangle; the unnormalized logarithmic density is retained as well.
When the provider declares a minimum pair distance, all three distances are
restricted to that explicit model domain; the excluded core is not evaluated.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import numbers
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import ANGSTROM_TO_AU, KB_AU, KCAL_MOL_TO_AU
from pyrpmd.potentials.base import validate_result


def distance_geometry(x, y, z):
    """Map valid distances in bohr to Cartesian Br,H,Cl coordinates.

    Inputs broadcast; the returned shape is broadcast_shape+(3,3). H is at
    the origin, Br lies on +x, and Cl lies in the xy plane. Triangle edges
    are allowed, but coincident nuclei and inconsistent triangles are not.
    """
    x, y, z = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    )
    if any(not np.isfinite(a).all() or np.any(a <= 0) for a in (x, y, z)):
        raise ValueError("all three distances must be finite and positive")
    tolerance = 32 * np.finfo(float).eps * np.maximum.reduce((x, y, z))
    if np.any(z < np.abs(x - y) - tolerance) or np.any(z > x + y + tolerance):
        raise ValueError("distances violate the triangle inequalities")
    cosine = (x * x + y * y - z * z) / (2 * x * y)
    cosine = np.clip(cosine, -1.0, 1.0)
    q = np.zeros(x.shape + (3, 3))
    q[..., 0, 0] = x
    q[..., 2, 0] = y * cosine
    q[..., 2, 1] = y * np.sqrt(np.maximum(0.0, (1 - cosine) * (1 + cosine)))
    return q


def common_wall_energy(x, y, z, *, wall_angstrom=6.0,
                       wall_k_kcal_mol_angstrom2=20.0):
    """Independent all-three-pair soft-wall energy, returned in hartree.

    The wall is .5*k*max(r-R,0)^2 per pair, evaluated on all three physical
    distances. R=0 or k=0 disables the wall, matching BrHClPotential.
    """
    radius, stiffness = float(wall_angstrom), float(wall_k_kcal_mol_angstrom2)
    if not np.isfinite([radius, stiffness]).all() or min(radius, stiffness) < 0:
        raise ValueError("wall radius and stiffness must be finite and nonnegative")
    x, y, z = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    )
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (x, y, z)):
        raise ValueError("wall distances must be finite and nonnegative")
    if radius == 0 or stiffness == 0:
        return np.zeros(x.shape)
    radius *= ANGSTROM_TO_AU
    stiffness *= KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
    return 0.5 * stiffness * sum(np.maximum(a - radius, 0.0)**2 for a in (x, y, z))


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@lru_cache(maxsize=32)
def _rule(order):
    nodes, weights = leggauss(order)
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def _bin_log_density(evaluator, x_bounds, y_bounds, beta, xy_order, z_order,
                     wall_angstrom, wall_k, max_batch, minimum_distance):
    nodes, weights = _rule(xy_order)
    znodes, zweights = _rule(z_order)
    widths = np.array([np.diff(x_bounds)[0], np.diff(y_bounds)[0]])
    lower = np.maximum([x_bounds[0], y_bounds[0]], minimum_distance)
    upper = np.array([x_bounds[1], y_bounds[1]])
    if np.any(upper <= lower):
        return -np.inf, 0
    x = (upper[0] + lower[0]) / 2 + (upper[0] - lower[0]) * nodes / 2
    y = (upper[1] + lower[1]) / 2 + (upper[1] - lower[1]) * nodes / 2
    x, y = (a.ravel() for a in np.meshgrid(x, y, indexing="ij"))
    # Density remains averaged over the original full bin, including any
    # zero-density part outside the explicitly declared provider domain.
    outer_log_weights = (np.log(weights[:, None] * weights[None, :]) - np.log(4)
                         + np.log((upper - lower) / widths).sum()).ravel()
    low, high = np.maximum(np.abs(x - y), minimum_distance), x + y
    midpoint, halfwidth = (low + high) / 2, (high - low) / 2
    count = len(x) * z_order
    total = -np.inf
    for start in range(0, count, max_batch):
        indices = np.arange(start, min(start + max_batch, count))
        outer, inner = indices // z_order, indices % z_order
        xb, yb = x[outer], y[outer]
        zb = midpoint[outer] + halfwidth[outer] * znodes[inner]
        q = np.ascontiguousarray(distance_geometry(xb, yb, zb))
        result = validate_result(evaluator(q), q.shape)
        energy = result.energy + common_wall_energy(
            xb, yb, zb, wall_angstrom=wall_angstrom,
            wall_k_kcal_mol_angstrom2=wall_k,
        )
        log_integrand = (outer_log_weights[outer] + np.log(zweights[inner])
                         + np.log(halfwidth[outer]) + np.log(xb) + np.log(yb)
                         + np.log(zb) - beta * energy)
        if not np.isfinite(log_integrand).all():
            raise FloatingPointError("nonfinite classical quadrature integrand")
        total = np.logaddexp(total, logsumexp(log_integrand))
    return float(total), count


def classical_pmf(potential, x_edges, y_edges, beta, *, wall_angstrom=6.0,
                  wall_k_kcal_mol_angstrom2=20.0, xy_order=4, z_order=32,
                  max_xy_order=16, max_z_order=256, tolerance_kbt=0.01,
                  max_batch=4096, minimum_pair_distance_bohr=None):
    """Integrate a finite-bin classical PMF with adaptive order doubling.

    Edges are in bohr, beta in inverse hartree. The provider returns the usual
    PyRPMD PESResult. ``bare_evaluate`` is preferred when the provider exposes
    it; otherwise ``evaluate`` must represent the unconfined physical PES.
    The common wall is added here independently, exactly once.

    ``minimum_pair_distance_bohr`` defaults to the provider's same-named
    metadata field, or zero when none is declared. It restricts x, y and z
    to the model's valid domain, leaving zero probability outside it. This
    is an explicit domain restriction, never a blanket suppression of PES
    errors. Bin densities are still divided by the original full bin area.

    Both Gauss-Legendre orders are doubled up to their caps. A bin is frozen
    once successive log-density estimates differ by <= tolerance_kbt. This
    is a numerical convergence diagnostic, not a rigorous error bound. The
    per-bin order and error arrays permit checking only the sampled comparison
    region without treating poorly resolved bins as accurate. At least two
    different order pairs are needed to mark a bin as converged.

    Memory used for a provider batch is bounded by max_batch; no full
    [bins,x_order,y_order,z_order,atoms,xyz] tensor is materialized.
    """
    edges = [np.asarray(axis, float) for axis in (x_edges, y_edges)]
    if any(axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
           or np.any(axis < 0) or np.any(np.diff(axis) <= 0) for axis in edges):
        raise ValueError("x/y edges must be finite, nonnegative, and strictly increasing")
    if isinstance(beta, (bool, np.bool_)) or np.ndim(beta) != 0 or not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(tolerance_kbt) or tolerance_kbt <= 0:
        raise ValueError("tolerance_kbt must be finite and positive")
    xy_order, z_order = _integer(xy_order, "xy_order", 2), _integer(z_order, "z_order", 2)
    max_xy_order = _integer(max_xy_order, "max_xy_order", xy_order)
    max_z_order = _integer(max_z_order, "max_z_order", z_order)
    max_batch = _integer(max_batch, "max_batch", 1)
    if minimum_pair_distance_bohr is None:
        minimum_pair_distance_bohr = getattr(potential, "metadata", {}).get("minimum_pair_distance_bohr", 0.)
        domain_source = "provider_metadata" if minimum_pair_distance_bohr else "unrestricted_positive_distances"
    else:
        domain_source = "explicit_argument"
    if (isinstance(minimum_pair_distance_bohr, (bool, np.bool_))
            or np.ndim(minimum_pair_distance_bohr) != 0
            or not np.isfinite(minimum_pair_distance_bohr) or minimum_pair_distance_bohr < 0):
        raise ValueError("minimum_pair_distance_bohr must be finite and nonnegative")
    minimum_pair_distance_bohr = float(minimum_pair_distance_bohr)
    common_wall_energy(0.0, 0.0, 0.0, wall_angstrom=wall_angstrom,
                       wall_k_kcal_mol_angstrom2=wall_k_kcal_mol_angstrom2)
    evaluator = getattr(potential, "bare_evaluate", None)
    uses_bare = callable(evaluator)
    if not uses_bare:
        evaluator = getattr(potential, "evaluate", None)
    if not callable(evaluator):
        raise TypeError("potential must expose bare_evaluate or evaluate")
    shape = (len(edges[0]) - 1, len(edges[1]) - 1)
    logarithms = np.full(shape, np.nan)
    errors = np.full(shape, np.nan)
    converged = np.zeros(shape, dtype=bool)
    used_xy = np.zeros(shape, dtype=int)
    used_z = np.zeros(shape, dtype=int)
    history, evaluations = [], 0
    while True:
        active = np.argwhere(~converged)
        for i, j in active:
            previous = logarithms[i, j]
            value, count = _bin_log_density(
                evaluator, edges[0][i:i+2], edges[1][j:j+2], float(beta), xy_order, z_order,
                wall_angstrom, wall_k_kcal_mol_angstrom2, max_batch, minimum_pair_distance_bohr,
            )
            evaluations += count
            logarithms[i, j], used_xy[i, j], used_z[i, j] = value, xy_order, z_order
            if count == 0:
                errors[i, j], converged[i, j] = 0., True
                used_xy[i, j], used_z[i, j] = 0, 0
            elif np.isfinite(previous):
                errors[i, j] = abs(value - previous)
                converged[i, j] = errors[i, j] <= tolerance_kbt
        finite_errors = errors[np.isfinite(errors)]
        history.append({
            "xy_order": xy_order, "z_order": z_order, "evaluated_bins": len(active),
            "converged_bins": int(converged.sum()), "total_bins": int(converged.size),
            "maximum_estimated_error_kbt": float(finite_errors.max()) if finite_errors.size else None,
            "cumulative_pes_evaluations": evaluations,
        })
        next_xy, next_z = min(2 * xy_order, max_xy_order), min(2 * z_order, max_z_order)
        if converged.all() or (next_xy, next_z) == (xy_order, z_order):
            break
        xy_order, z_order = next_xy, next_z
    volumes = np.diff(edges[0])[:, None] * np.diff(edges[1])[None, :]
    log_partition = float(logsumexp(logarithms + np.log(volumes)))
    if not np.isfinite(log_partition):
        raise ValueError("analysis rectangle has no probability in the declared PES domain")
    density = np.exp(logarithms - log_partition)
    probability = np.exp(logarithms + np.log(volumes) - log_partition)
    free_energy = (logarithms.max() - logarithms) / beta
    return {
        "schema": "pyrpmd.brhcl.classical-reference.v1", "beads": 1,
        "beta": float(beta), "x_edges": edges[0].copy(), "y_edges": edges[1].copy(),
        "free_energy": free_energy, "density": density, "probability_mass": probability,
        "log_density_unnormalized": logarithms,
        "log_partition_in_rectangle": log_partition,
        "quadrature_error_kbt": errors, "quadrature_converged": converged,
        "xy_order_per_bin": used_xy, "z_order_per_bin": used_z,
        "converged": bool(converged.all()),
        "status": "converged_by_order_doubling" if converged.all() else "quadrature_order_limit",
        "tolerance_kbt": float(tolerance_kbt), "history": history,
        "pes_evaluations": evaluations, "max_batch": max_batch,
        "potential_entry": "bare_evaluate" if uses_bare else "evaluate",
        "domain": {"minimum_pair_distance_bohr": minimum_pair_distance_bohr,
                   "source": domain_source,
                   "restriction": "all three pair distances must be >= the minimum",
                   "excluded_core_probability": "zero by declared model domain, not evaluated or estimated",
                   "finite_bin_normalization": "original full x/y bin area"},
        "wall": {"radius_angstrom": float(wall_angstrom),
                 "stiffness_kcal_mol_angstrom2": float(wall_k_kcal_mol_angstrom2),
                 "pairs": ["Br-H", "H-Cl", "Br-Cl"], "retained_in_target": True},
        "measure": "classical density with respect to d(r_BrH) d(r_HCl)",
        "normalization": "conditional on the finite x/y analysis rectangle",
        "energy_unit": "hartree", "coordinate_unit": "bohr",
        "limitations": [
            "Classical P=1 reference; not a P>1 centroid-PMF oracle.",
            "Successive quadrature agreement is an estimated numerical error, not a rigorous bound.",
            "The hidden distance is integrated over the triangle interval intersected with the declared PES domain; no minimum-energy path is substituted.",
            "Excluded-core probability is not bounded; the result is conditional on the declared valid model domain.",
        ],
    }


def _reference_identity(settings, library_sha256, edges):
    """The shared scientific identity for both cache readers and writers."""
    from benchmarks.brhcl.pes import SOURCE

    if (not isinstance(library_sha256, str) or len(library_sha256) != 64
            or any(character not in "0123456789abcdef" for character in library_sha256)):
        raise ValueError("reference cache library_sha256 must be a lowercase SHA256 digest")
    if settings.get("beads") != 1 or isinstance(settings.get("beads"), bool):
        raise ValueError("classical reference cache is only a P=1 reference")
    if (settings.get("atomic_order") != ["Br", "H", "Cl"]
            or settings.get("cv_names") != ["r_HBr", "r_HCl"]
            or settings.get("minimum_pair_distance_bohr", SOURCE["minimum_pair_distance_bohr"])
                != SOURCE["minimum_pair_distance_bohr"]):
        raise ValueError("reference target atom, CV or core-domain identity differs")
    axes = [np.asarray(axis, float) for axis in edges]
    if len(axes) != 2 or any(axis.ndim != 1 or len(axis) < 2
                            or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)
                            for axis in axes):
        raise ValueError("reference cache requires two finite increasing bin-edge arrays")
    temperature, beta = settings["temperature_K"], settings["beta"]
    if (not np.isfinite([temperature, beta]).all() or min(temperature, beta) <= 0
            or not np.isclose(beta, 1/(KB_AU*temperature), rtol=1e-13, atol=0)):
        raise ValueError("reference target temperature and beta are inconsistent")
    expected = {
        "model": settings["model"], "library_sha256": library_sha256,
        "source_commit": SOURCE["commit"], "source_sha256": SOURCE["sha256"],
        "temperature_K": temperature, "beta": beta, "beads": 1,
        "wall_angstrom": settings["wall_angstrom"],
        "wall_k_kcal_mol_angstrom2": settings["wall_k_kcal_mol_angstrom2"],
        "minimum_pair_distance_bohr": SOURCE["minimum_pair_distance_bohr"],
        "atomic_order": ["Br", "H", "Cl"], "cv_names": ["r_HBr", "r_HCl"],
        "x_edges_bohr": axes[0].tolist(), "y_edges_bohr": axes[1].tolist(),
    }
    if expected["model"] != "BrHCl_LEPS1":
        raise ValueError("reference cache belongs to the BrHCl_LEPS1 model")
    return expected


def load_reference_cache(path, *, settings, library_sha256, edges):
    """Read a verified BrHCl classical reference matching this exact target.

    ``path`` is the .npz array archive; its .json sibling contains quadrature
    metadata. ``edges`` is the pair of *actual FES bin edges in bohr*, not the
    requested ranges. The source pin, binary hash, P=1, temperature, beta,
    retained wall, core domain, atom/CV order, and every edge must agree.
    Returns the same array/metadata keys as :func:`classical_pmf`, so multiple
    independently sampled runs may reuse one deterministic integral.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("reference cache must name its .npz archive")
    expected = _reference_identity(settings, library_sha256, edges)
    axes = [np.asarray(expected[key]) for key in ("x_edges_bohr", "y_edges_bohr")]
    beta = expected["beta"]
    with load_checkpoint(path, expected_kind="arrays", require_integrity=True) as saved:
        arrays = {name: saved[name].copy() for name in saved}
    try:
        summary = json.loads(path.with_suffix(".json").read_text())
        identity_array, checksum_array = arrays.pop("identity"), arrays.pop("identity_sha256")
        if (identity_array.shape != () or identity_array.dtype.kind != "U"
                or checksum_array.shape != () or checksum_array.dtype.kind != "U"):
            raise ValueError("identity must contain scalar JSON and checksum strings")
        identity = json.loads(str(identity_array))
        checksum = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (str(checksum_array) != checksum or summary["identity_sha256"] != checksum
                or summary["identity"] != identity):
            raise ValueError("archive and metadata identity checksums differ")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reference cache identity metadata is invalid") from error
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("reference cache scientific identity differs from the sampled target")
    required = {"x_edges", "y_edges", "free_energy", "density", "probability_mass",
                "log_density_unnormalized", "quadrature_error_kbt", "quadrature_converged",
                "xy_order_per_bin", "z_order_per_bin"}
    shape = (len(axes[0])-1, len(axes[1])-1)
    if (not required.issubset(arrays)
            or not np.array_equal(arrays["x_edges"], axes[0])
            or not np.array_equal(arrays["y_edges"], axes[1])
            or any(arrays[key].shape != shape for key in required-{"x_edges", "y_edges"})):
        raise ValueError("reference cache arrays or bin edges are inconsistent")
    if (any(arrays[key].dtype.kind not in "iuf" for key in
            ("free_energy", "density", "probability_mass", "log_density_unnormalized", "quadrature_error_kbt"))
            or any(arrays[key].dtype.kind not in "iu" or np.any(arrays[key] < 0)
                   for key in ("xy_order_per_bin", "z_order_per_bin"))):
        raise ValueError("reference cache array types or quadrature orders are invalid")
    try:
        metadata_matches = (
            summary["schema"] == "pyrpmd.brhcl.classical-reference.v1"
            and summary["beads"] == 1 and summary["beta"] == beta
            and summary["energy_unit"] == "hartree" and summary["coordinate_unit"] == "bohr"
            and summary["wall"]["radius_angstrom"] == expected["wall_angstrom"]
            and summary["wall"]["stiffness_kcal_mol_angstrom2"] == expected["wall_k_kcal_mol_angstrom2"]
            and summary["wall"]["retained_in_target"] is True
            and summary["domain"]["minimum_pair_distance_bohr"] == expected["minimum_pair_distance_bohr"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("reference cache target metadata is incomplete") from error
    if not metadata_matches:
        raise ValueError("reference cache target metadata contradicts its scientific identity")
    density, mass, logarithms = (arrays[key] for key in ("density", "probability_mass", "log_density_unnormalized"))
    volume = np.diff(axes[0])[:, None]*np.diff(axes[1])[None, :]
    log_partition = float(logsumexp(logarithms+np.log(volume)))
    if (not np.isfinite(density).all() or not np.isfinite(mass).all()
            or np.any(density < 0) or np.any(mass < 0)
            or not np.isclose(mass.sum(), 1., rtol=0, atol=1e-10)
            or not np.allclose(density*volume, mass, rtol=1e-10, atol=1e-14)
            or not np.isfinite(logarithms).any() or np.isnan(logarithms).any()
            or np.isposinf(logarithms).any()
            or not np.isclose(summary.get("log_partition_in_rectangle", np.nan), log_partition, rtol=0, atol=1e-10)
            or not np.allclose(density, np.exp(logarithms-log_partition), rtol=1e-10, atol=1e-14)
            or not np.allclose(arrays["free_energy"], (logarithms.max()-logarithms)/beta, rtol=1e-10, atol=1e-12)):
        raise ValueError("reference cache probability or free-energy arrays are inconsistent")
    converged, error = arrays["quadrature_converged"], arrays["quadrature_error_kbt"]
    tolerance = summary.get("tolerance_kbt", np.nan)
    if (not np.isfinite(tolerance) or tolerance <= 0
            or converged.dtype.kind != "b" or not np.isfinite(error[converged]).all()
            or np.any(error[converged] < 0)
            or np.any(error[converged] > tolerance)
            or summary.get("converged") is not bool(converged.all())):
        raise ValueError("reference cache quadrature convergence metadata is inconsistent")
    return {**summary, **arrays, "cache_path": str(path)}


def save_reference_cache(path, exact, *, settings, library_sha256):
    """Validate and save a classical reference as an NPZ/JSON cache pair.

    ``exact`` is the result of :func:`classical_pmf` for the stated target.
    The complete pair is staged and checked with :func:`load_reference_cache`
    before either final path is created. Existing outputs are never replaced;
    choose a new path when rebuilding an already saved reference.
    The returned dictionary has the same interface as the cache reader.
    """
    from pyrpmd.workflows.fes import write_json

    path = Path(path)
    sidecar = path.with_suffix(".json")
    if path.suffix != ".npz":
        raise ValueError("reference cache must name its .npz archive")
    if path.exists() or sidecar.exists():
        raise FileExistsError("reference cache output already exists; choose a new path")
    edges = [exact["x_edges"], exact["y_edges"]]
    identity = _reference_identity(settings, library_sha256, edges)
    checksum = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    arrays = {name: value for name, value in exact.items() if isinstance(value, np.ndarray)}
    summary = {name: value for name, value in exact.items()
               if not isinstance(value, np.ndarray) and name != "cache_path"}
    finite_errors = arrays["quadrature_error_kbt"][np.isfinite(arrays["quadrature_error_kbt"])]
    summary.update(identity=identity, identity_sha256=checksum, array_file=path.name,
                   converged_bins=int(np.count_nonzero(arrays["quadrature_converged"])),
                   maximum_order_error_kbt=float(finite_errors.max()) if finite_errors.size else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-cache-", dir=path.parent) as temporary:
        staged = Path(temporary)/path.name
        atomic_savez(staged, kind="arrays", **arrays,
                     identity=json.dumps(identity, sort_keys=True), identity_sha256=checksum)
        write_json(staged.with_suffix(".json"), summary)
        validated = load_reference_cache(staged, settings=settings,
                                         library_sha256=library_sha256, edges=edges)
        # Recheck immediately before publishing; an independent process may
        # have completed a cache while this one was validating its payload.
        if path.exists() or sidecar.exists():
            raise FileExistsError("reference cache output already exists; choose a new path")
        # Hard-link publication is atomic and refuses an existing destination,
        # including one created between the check above and this operation.
        os.link(staged, path)
        try:
            os.link(staged.with_suffix(".json"), sidecar)
        except BaseException:
            if path.exists() and os.path.samefile(path, staged):
                path.unlink()
            raise
    validated["cache_path"] = str(path)
    return validated


def main(argv=None):
    """Rebuild a cache directly from a recorded P=1 scan plan and PES binary."""
    from benchmarks.brhcl.pes import BrHClPotential

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="P=1 run containing plan.json and model.json")
    parser.add_argument("--output", type=Path, required=True, help="new .npz reference cache; also writes .json")
    parser.add_argument("--tolerance-kbt", type=float, default=.01)
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        raise ValueError("reference cache must name its .npz archive")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("reference cache output already exists; choose a new path")
    if not np.isfinite(args.tolerance_kbt) or args.tolerance_kbt <= 0:
        raise ValueError("tolerance-kbt must be finite and positive")
    settings = json.loads((args.run/"plan.json").read_text())
    model = json.loads((args.run/"model.json").read_text())
    bins, bounds = settings["bins"], np.asarray(settings["ranges_angstrom"], float)*ANGSTROM_TO_AU
    if (len(bins) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in bins)
            or bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("run requires two positive integer bin counts and finite increasing distance ranges")
    # Match reconstruct_fes: convert bounds to atomic units before linspace.
    edges = [np.linspace(*axis, count+1) for axis, count in zip(bounds, bins)]
    _reference_identity(settings, model["library_sha256"], edges)
    library = Path(model["library"])
    if hashlib.sha256(library.read_bytes()).hexdigest() != model["library_sha256"]:
        raise ValueError("reference PES binary no longer matches the recorded run")
    started = time.perf_counter()
    with BrHClPotential(library=library) as potential:
        exact = classical_pmf(potential, *edges, settings["beta"],
                              wall_angstrom=settings["wall_angstrom"],
                              wall_k_kcal_mol_angstrom2=settings["wall_k_kcal_mol_angstrom2"],
                              tolerance_kbt=args.tolerance_kbt)
    if hashlib.sha256(library.read_bytes()).hexdigest() != model["library_sha256"]:
        raise ValueError("reference PES binary changed during integration")
    exact.update(seconds=time.perf_counter()-started, source_run=str(args.run.resolve()))
    saved = save_reference_cache(args.output, exact, settings=settings,
                                 library_sha256=model["library_sha256"])
    print(json.dumps({"cache": str(args.output), "converged": saved["converged"],
                      "converged_bins": saved["converged_bins"],
                      "total_bins": int(saved["free_energy"].size),
                      "maximum_order_error_kbt": saved["maximum_order_error_kbt"],
                      "seconds": saved["seconds"], "identity_sha256": saved["identity_sha256"]}))


if __name__ == "__main__":
    main()
