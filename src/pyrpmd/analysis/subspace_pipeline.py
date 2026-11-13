"""One-command subspace quantum correction of a saved PyRPMD rate run.

    rate run (output.centroid_path = true)
      -> ring-polymer mean path xbar(xi) from the umbrella windows
      -> path to both asymptotes (sampled core + steepest-descent continuation,
         followed without a step budget until the energy is flat)
      -> curvature x width: Sigma(xi) where sampled, harmonic width elsewhere
      -> 1D vibrationally adiabatic path model           -> C_S(1D)
      -> if kappa*rho > threshold: 2D plane through the TS spanned by the
         rate dividing-surface normal and the curvature normal at the peak
         (the collinear plane for a collinear atom transfer)  -> C_S(2D)
      -> nested sensitivity ln(C_2D/C_1D), recommended C_S
      -> rate_quantum_corrected.json next to the run

Validity gates decide whether a model may carry a factor at all: the path
must have reached both asymptotes, the transverse (normal-plane) Hessian of
the path model must be stable, and the ring-polymer estimator must not have
sampled a gridded ``V_S`` outside its grid.  A model that fails is recorded
with its diagnostics and never applied; ``validity.* = "allow"`` is the
explicit override.

:func:`run_subspace_pipeline` takes a mapping or a TOML file (the
``pyrpmd subspace-correct`` command).  Every intermediate quantity is written
to ``subspace_pipeline.json`` in the output directory.  Nothing here marks a
rate as physically converged; the grid, absorber, bead and sampling settings
are recorded so a caller can converge them.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .centroid_path import mean_path_from_rate_run, rigid_body_modes
from .subspace_correction import (CORRECTED_SCHEMA, SubspaceCorrection, correct_rate_run,
                                  nested_subspace_sensitivity, subspace_correction, validity_gates,
                                  validity_policy)
from .subspace_model import (_vibrational_complement, assemble_reaction_path, path_curvature,
                             project_linear_subspace, project_path_subspace, relaxed_path)
from .subspace_quantum import _json_safe, quantum_route_plan, sinc_dvr_kinetic


def crossover_temperature(*args, **kwargs):
    """Lazy instanton crossover helper, kept patchable for pipeline tests."""
    from ..instanton import crossover_temperature as _crossover_temperature

    return _crossover_temperature(*args, **kwargs)


def optimize_instanton(*args, **kwargs):
    """Lazy instanton optimizer helper, kept patchable for pipeline tests."""
    from ..instanton import optimize_instanton as _optimize_instanton

    return _optimize_instanton(*args, **kwargs)

DEFAULTS = {
    "source": {"rate_run": None},
    # max_steps = None: the continuations run until the asymptote criterion
    # is met.  A finite value is a hard budget whose exhaustion fails the path gate.
    "path": {"source": "centroid", "step": 10.0, "max_steps": None, "spread_factor": 2.0,
             "asymptote_tolerance": 1e-6, "asymptote_patience": 50, "smoothing": 1.5,
             "least_action_points": 9},
    # hessian_displacement (bohr) is the largest Cartesian displacement of the
    # lightest atom in the finite-difference normal-plane Hessian.
    "subspace": {"dimension": "auto", "threshold": 0.5, "environment": "adiabatic", "curvature": "none",
                 "zero_point": "harmonic", "dvr_points": 81, "dvr_half_width": None,
                 "dvr_spacing": None, "hessian_displacement": 5e-3},
    # on_invalid = "stop": a model whose path or bath gate fails is not solved;
    # "diagnose": solve it anyway and keep the component rates (never applied).
    "validity": {"unconverged_path": "reject", "unstable_bath": "reject", "outside_grid": "reject",
                 "max_outside_fraction": 0.01, "soft_mode_frequency": 2.3e-4, "on_invalid": "stop"},
    "quantum": {"spacing_1d": None, "spacing_2d": None, "refinement": 3, "absorber_fraction": 0.2,
                "threshold_margin": 1.0, "truncation_kt": 30.0, "truncation_min": 0.0, "padding": None,
                "dense_limit": 8000,
                "reference_kt": 0.3, "product_drop_kt": 15.0},
    "rpmd": {"nbeads": None, "windows": 48, "walkers": 32, "burn_in": 200, "sweeps": 400,
             "trajectories": 512, "seed": 0},
    "output": {"directory": None, "reuse": False},
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class RateRun:
    """Saved rate run: resolved input, system, reaction coordinate and PES."""

    root: Path
    config: dict
    system: Any
    reaction: dict
    coordinate: Any
    provider: Any
    beta: float
    nbeads: int
    xi_star: float

    @property
    def cartesian_masses(self):
        return np.repeat(np.asarray(self.system.masses, dtype=float), 3)

    def energy(self, x):
        result = self.provider.evaluate(np.asarray(x, dtype=float).reshape(1, -1, 3))
        return float(np.asarray(result.energy).ravel()[0])

    def gradient(self, x):
        result = self.provider.evaluate(np.asarray(x, dtype=float).reshape(1, -1, 3))
        return -np.asarray(result.forces, dtype=float).ravel()

    def xi(self, x):
        return float(np.asarray(self.coordinate.centroid_value_gradient(np.asarray(x).reshape(-1, 3))[0]))

    def xi_gradient(self, x):
        return np.asarray(self.coordinate.centroid_value_gradient(np.asarray(x).reshape(-1, 3))[1]).ravel()


def load_rate_run(directory) -> RateRun:
    """Rebuild the PES, system and RPMDrate coordinate of a saved rate run."""
    from ..potentials import make_potential
    from ..system import System
    from ..workflows.rate_constraints import rate_coordinate
    from .rate_io import _root

    root = _root(directory, "input/resolved.json")
    config = json.loads((root / "input" / "resolved.json").read_text())
    system = System.from_dict(json.loads((root / "input" / "system.json").read_text()))
    preparation = json.loads((root / "preparation.json").read_text()) if (root / "preparation.json").is_file() else {}
    from ..workflows.prepare import prepared_system

    system = prepared_system(system, preparation)
    reaction = dict(config.get("reaction", {}))
    reaction.update(preparation.get("reaction", {}))
    for key in ("fragments", "forming_bonds", "breaking_bonds"):
        if key not in reaction:
            raise ValueError(f"the rate run does not record reaction.{key}")
    coordinate = rate_coordinate(dict(config, reaction=reaction), system)
    provider = make_potential(config["potential"], system)
    xi_star = 1.0
    pmf_path = root / "rate_pmf.json"
    if pmf_path.is_file():
        xi_star = float(json.loads(pmf_path.read_text()).get("xi_star") or 1.0)
    return RateRun(root, config, system, reaction, coordinate, provider, float(config["beta"]),
                   int(config["beads"]), xi_star)


def _harmonic_bead_width(omega, beta):
    """sqrt(2) x RMS bead spread about the centroid of a harmonic mode (P -> inf)."""
    omega = max(float(omega), 1e-12)
    variance = 0.5 / omega / np.tanh(0.5 * beta * omega) - 1.0 / (beta * omega**2)
    return float(np.sqrt(2.0 * max(variance, 0.0)))


def _static_curvature_width(run, path, threshold):
    """kappa(s) along the whole path, rho from the harmonic mode along the normal."""
    masses = run.cartesian_masses
    sqrt_m = np.sqrt(masses)
    mw = path * sqrt_m
    arc, kappa = path_curvature(path, masses, samples=len(path))
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mw, axis=0), axis=1))]
    rho = np.zeros(len(path))
    normals = np.zeros_like(mw)
    for k in range(1, len(path) - 1):
        t = mw[k + 1] - mw[k - 1]
        t /= np.linalg.norm(t)
        a = (mw[k + 1] - mw[k]) / max(chord[k + 1] - chord[k], 1e-12) - (mw[k] - mw[k - 1]) / max(chord[k] - chord[k - 1], 1e-12)
        a = a - (a @ t) * t
        rigid = rigid_body_modes(path[k].reshape(-1, 3), run.system.masses)
        a = a - rigid @ (rigid.T @ a)
        if np.linalg.norm(a) == 0:
            continue
        n = a / np.linalg.norm(a)
        normals[k] = n
        dx = 1e-3 * n / sqrt_m
        curvature = n @ ((run.gradient(path[k] + dx) - run.gradient(path[k] - dx)) / 2e-3 / sqrt_m)
        rho[k] = _harmonic_bead_width(np.sqrt(max(curvature, 1e-12)), run.beta)
    kappa_points = np.interp(chord, np.linspace(0, chord[-1], len(kappa)), kappa)
    product = kappa_points * rho
    peak = int(np.argmax(product))
    return {"arc_length": chord.tolist(), "curvature": kappa_points.tolist(), "width": rho.tolist(),
            "curvature_width": product.tolist(), "maximum": float(product[peak]),
            "maximum_arc_length": float(chord[peak]), "threshold": float(threshold),
            "recommended_dimension": 2 if product[peak] > threshold else 1,
            "candidate_direction": normals[peak].tolist(), "width_source": "harmonic mode along the normal"}


def _ordered_locked_indices(mean, mask):
    """Return the original window indices in increasing reaction-coordinate order."""
    order = np.argsort(np.asarray(mean.xi, dtype=float))
    return order[np.asarray(mask, dtype=bool)[order]]


def _path_tangent_and_curvature(points, masses):
    """Finite-difference tangent and curvature vectors in mass-weighted space."""
    points = np.asarray(points, dtype=float)
    sqrt_m = np.sqrt(np.asarray(masses, dtype=float))
    weighted = points * sqrt_m[None, :]
    tangent = np.zeros_like(weighted)
    curvature = np.zeros_like(weighted)
    for i in range(len(points)):
        if i == 0:
            tangent[i] = weighted[1] - weighted[0]
        elif i == len(points) - 1:
            tangent[i] = weighted[-1] - weighted[-2]
        else:
            tangent[i] = weighted[i + 1] - weighted[i - 1]
            curvature[i] = weighted[i + 1] - 2 * weighted[i] + weighted[i - 1]
        norm = np.linalg.norm(tangent[i])
        if norm > 0:
            tangent[i] /= norm
    for i in range(len(points)):
        curvature[i] -= (curvature[i] @ tangent[i]) * tangent[i]
    return tangent, curvature


def _bead_weighted_core(mean, mask, masses):
    """Shift each centroid towards the local corner using bead covariance.

    ``bead_covariance`` is already mass weighted and is the covariance of
    bead displacements about the centroid.  The displacement along its
    largest transverse eigenvector is therefore the bead RMS
    ``sqrt(lambda_max)`` in mass-weighted coordinates.  (The curvature-width
    diagnostic uses a separate ``sqrt(2)`` quantum-width convention.)  The
    arbitrary eigenvector sign is fixed by the local curvature vector, so the
    shift points into the bend.
    """
    indices = _ordered_locked_indices(mean, mask)
    if hasattr(mean, "mean_centroid"):
        core = np.asarray(mean.mean_centroid[indices], dtype=float).copy()
    else:
        core = np.asarray(mean.path(mask), dtype=float).copy()
    tangent, curvature = _path_tangent_and_curvature(core.reshape(len(core), -1),
                                                     np.repeat(np.asarray(masses, dtype=float), 3))
    sqrt_m = np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    shifts = np.zeros((len(indices), core.shape[1] * core.shape[2]))
    eigenvalues = []
    for k, index in enumerate(indices):
        covariance = np.asarray(mean.bead_covariance[index], dtype=float)
        if covariance.shape != (core.shape[1] * core.shape[2],) * 2 or not np.isfinite(covariance).all():
            eigenvalues.append(0.0)
            continue
        covariance = 0.5 * (covariance + covariance.T)
        # Project the covariance before diagonalizing: projecting a full-space
        # eigenvector but retaining its eigenvalue overcounts tangential spread
        # and need not select the largest transverse mode.
        projector = np.eye(len(tangent[k])) - np.outer(tangent[k], tangent[k])
        transverse = projector @ covariance @ projector
        values, vectors = np.linalg.eigh(0.5 * (transverse + transverse.T))
        vector, eigenvalue = vectors[:, -1], max(float(values[-1]), 0.0)
        sign = float(np.sign(vector @ curvature[k]))
        if sign == 0.0:
            # A locally straight section has no curvature sign.  Keep the
            # orientation deterministic and continuous with the previous
            # selected mode when possible.
            if k and np.dot(vector, shifts[k - 1]) < 0:
                sign = -1.0
            else:
                sign = 1.0
        shifts[k] = sign * np.sqrt(eigenvalue) * vector
        eigenvalues.append(eigenvalue)
    shifted = (core.reshape(len(core), -1) * sqrt_m[None, :] + shifts) / sqrt_m[None, :]
    return shifted.reshape(core.shape), {
        "windows": int(len(indices)),
        "principal_variance_range": [float(min(eigenvalues or [0.0])), float(max(eigenvalues or [0.0]))],
        "bead_rms_range": [float(np.sqrt(min(eigenvalues or [0.0]))),
                           float(np.sqrt(max(eigenvalues or [0.0])))],
        "shift_max_mass_weighted": float(np.max(np.linalg.norm(shifts, axis=1))) if len(shifts) else 0.0,
        "shift_convention": "sqrt(lambda_max) bead RMS along transverse principal mode; curvature sign",
    }


def _wkb_action(points, run, reference_energy=None):
    """One-dimensional Euclidean WKB action along a Cartesian path."""
    masses = run.cartesian_masses
    weighted = np.asarray(points, dtype=float).reshape(len(points), -1) * np.sqrt(masses)[None, :]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(weighted, axis=0), axis=1))]
    energies = np.asarray([run.energy(point) for point in np.asarray(points)], dtype=float)
    if reference_energy is None:
        reference_energy = float(min(energies[0], energies[-1]))
    integrand = np.sqrt(2.0 * np.maximum(energies - float(reference_energy), 0.0))
    from scipy.integrate import trapezoid

    return float(trapezoid(integrand, arc)), energies, arc


def _least_action_core(core, run, options):
    """Scan a MEP-to-chord family and return the minimum-action member."""
    core = np.asarray(core, dtype=float)
    flat = core.reshape(len(core), -1)
    chord = np.linspace(flat[0], flat[-1], len(flat)).reshape(core.shape)
    count = max(2, int(options.get("least_action_points", 9)))
    alphas = np.linspace(0.0, 1.0, count)
    records = []
    best = None
    for alpha in alphas:
        candidate = (1.0 - alpha) * core + alpha * chord
        action, energies, arc = _wkb_action(candidate, run)
        record = {"alpha": float(alpha), "action_hbar": float(action),
                  "barrier": float(np.max(energies) - min(energies[0], energies[-1])),
                  "arc_length": float(arc[-1])}
        records.append(record)
        if best is None or action < best[0]:
            best = (action, candidate, record)
    return best[1], {"criterion": "minimum one-dimensional WKB action",
                     "selected_alpha": float(best[2]["alpha"]), "selected_action_hbar": float(best[0]),
                     "scan": records}


def _instanton_seed_from_core(core, nbeads):
    """Create a noncollapsed link-reflected instanton seed from a path core."""
    core = np.asarray(core, dtype=float)
    p = max(8, int(nbeads))
    if p % 2:
        p += 1
    half = p // 2
    # Use the barrier-centred portion of the MEP; preserving endpoint
    # separation gives the optimizer a noncollapsed orbit without guessing a
    # transition-state normal mode.
    # The energy is not available here; use the middle of the sampled core.
    centre = len(core) // 2
    span = min(len(core), max(4, half))
    start = max(0, min(len(core) - span, centre - span // 2))
    sample = core[start:start + span]
    parameter = np.linspace(0.0, 1.0, len(sample))
    target = np.linspace(0.0, 1.0, half)
    folded_half = np.empty((half,) + sample.shape[1:])
    for atom in range(sample.shape[1]):
        for component in range(3):
            folded_half[:, atom, component] = np.interp(target, parameter, sample[:, atom, component])
    return np.concatenate([folded_half, folded_half[::-1]], axis=0)


def _instanton_core(run, core, options):
    """Optimize a thermal instanton, or return centroid core with diagnostics."""
    from ..constants import KB_AU

    temperature = 1.0 / (KB_AU * run.beta)
    details = {"temperature_kelvin": float(temperature), "method": "instanton"}
    try:
        crossover = float(crossover_temperature(run.provider, run.system.geometry,
                                                run.system.masses))
        details["crossover_temperature_kelvin"] = crossover
    except Exception as error:
        details["crossover_temperature_error"] = str(error)
        crossover = options.get("crossover_temperature")
        if crossover is not None:
            crossover = float(crossover)
            details["crossover_temperature_kelvin"] = crossover
    if crossover is not None and temperature >= crossover:
        details.update({"fallback": "centroid", "fallback_reason": "temperature_at_or_above_crossover"})
        return core, details
    beads = int(options.get("instanton_beads") or run.nbeads or 16)
    # Keep the requested ring size explicit in the persisted diagnostics.  A
    # folded half-ring is what the path builder consumes, but the full bead
    # number is the quantity needed to reproduce an optimizer run and to
    # interpret the physical mode count.
    seed = _instanton_seed_from_core(core, beads)
    details["requested_bead_number"] = beads
    details["bead_number"] = int(len(seed))
    try:
        result = optimize_instanton(run.provider, run.system.masses, temperature, seed,
                                    transition_state=run.system.geometry,
                                    max_iterations=int(options.get("instanton_max_iterations", 200)),
                                    raise_on_failure=False)
    except Exception as error:
        message = str(error)
        reason = ("temperature_at_or_above_crossover"
                  if "temperature below crossover" in message.lower()
                  else "optimizer_failure")
        details.update({"fallback": "centroid", "fallback_reason": reason, "error": message})
        return core, details
    details.update({"converged": bool(result.converged), "gradient_max": float(result.gradient_max),
                    "negative_modes": int(result.negative_modes),
                    "physical_negative_modes": int(result.negative_modes),
                    "physical_converged": False})
    if not result.converged:
        details.update({"fallback": "centroid", "fallback_reason": "optimizer_not_converged",
                        "optimizer_message": str(result.message)})
        return core, details
    orbit = np.asarray(result.path_bohr, dtype=float)
    folded = orbit[:len(orbit) // 2]
    # Orient the folded chain from reactants to products to match assemble's
    # continuation convention.
    masses = run.cartesian_masses
    if np.linalg.norm((folded[0] - core[0]).ravel() * np.sqrt(masses)) > \
            np.linalg.norm((folded[-1] - core[0]).ravel() * np.sqrt(masses)):
        folded = folded[::-1]
    details.update({"folded_beads": int(len(folded)), "fallback": None,
                    "core_source": "optimized folded half-ring"})
    return folded, details


def build_path(run: RateRun, options: Mapping[str, Any]):
    """Build and extend a reaction path selected by ``path.source``.

    ``centroid`` preserves the historical mean-centroid path.  The other
    sources only change the sampled core; all of them use the same
    :func:`assemble_reaction_path` continuation and therefore the same
    asymptotic reference geometry and potential.
    """
    options = dict(options or {})
    source = str(options.get("source", "centroid")).lower()
    allowed = {"centroid", "bead_weighted", "instanton", "least_action"}
    if source not in allowed:
        raise ValueError(f"path.source must be one of {sorted(allowed)}")
    mean = mean_path_from_rate_run(run.root)
    mask = mean.locked(factor=float(options.get("spread_factor", 2.0)))
    indices = _ordered_locked_indices(mean, mask)
    centroids = np.asarray(mean.mean_centroid[indices] if hasattr(mean, "mean_centroid")
                           else mean.path(mask), dtype=float)
    core = centroids.reshape(int(mask.sum()), -1)
    xi_core = np.sort(mean.xi[mask])
    sampled = {"source": source, "windows": int(len(mean.xi)), "locked_windows": int(mask.sum()),
               "locked_xi": [float(v) for v in xi_core],
               "orientation_spread": mean.orientation_spread.tolist()}
    if len(core) < 2:
        raise ValueError("fewer than two orientation-locked windows; add umbrella windows near the transition state")

    if source == "bead_weighted":
        core, details = _bead_weighted_core(mean, mask, run.system.masses)
        core = core.reshape(len(core), -1)
        sampled["bead_weighted"] = details
    elif source == "instanton":
        core, details = _instanton_core(run, core.reshape(len(core), -1, 3), options)
        core = np.asarray(core).reshape(len(core), -1)
        sampled["instanton"] = details

    smoothing = options.get("smoothing")
    if smoothing and len(core) >= 5 and source != "instanton":
        # Fit xbar(xi) against the sampling coordinate itself: window means
        # carry transverse noise, and a chord-length parametrization of noisy
        # points zig-zags.  Local quadratic regression in xi, evaluated on a
        # uniform xi grid, gives a smooth, monotonically parametrized core.
        bandwidth = float(smoothing) * 2.0 * float(np.median(np.diff(xi_core)))
        if bandwidth <= 0:
            raise ValueError("locked reaction-coordinate windows must have distinct xi values")
        grid = np.linspace(xi_core[0], xi_core[-1], 4 * len(xi_core))
        fitted = np.empty((len(grid), core.shape[1]))
        for i, centre in enumerate(grid):
            delta = xi_core - centre
            weight = np.sqrt(np.exp(-0.5 * (delta / bandwidth) ** 2))
            design = np.column_stack([np.ones_like(delta), delta, delta * delta]) * weight[:, None]
            fitted[i] = np.linalg.lstsq(design, core * weight[:, None], rcond=None)[0][0]
        # Local fits are one-sided at the ends; drop half a bandwidth there
        # and let the continuation take over.
        inner = (grid >= xi_core[0] + 0.5 * bandwidth) & (grid <= xi_core[-1] - 0.5 * bandwidth)
        core = fitted[inner] if inner.sum() >= 4 else fitted
        sampled["smoothing_bandwidth_xi"] = bandwidth
    if source == "least_action":
        # Optimize the same smoothed core used by the centroid source.  Any
        # smoothing after the scan would change the selected action and path.
        core, details = _least_action_core(core.reshape(len(core), -1, 3), run, options)
        core = np.asarray(core).reshape(len(core), -1)
        sampled["least_action"] = details
    path, info = assemble_reaction_path(
        core, run.energy, run.gradient, run.cartesian_masses, step=float(options.get("step", 10.0)),
        max_steps=None if options.get("max_steps") is None else int(options["max_steps"]), rigid_body=True,
        asymptote_tolerance=float(options.get("asymptote_tolerance", 1e-6)),
        patience=int(options.get("asymptote_patience", 50)),
    )
    info.update(sampled)
    return mean, mask, path, info


def _arc(path, masses):
    mw = path * np.sqrt(masses)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mw, axis=0), axis=1))]


def _dividing_arc(run, path, arc):
    xi = np.array([run.xi(p) for p in path])
    above = np.flatnonzero(xi >= run.xi_star)
    if not len(above) or above[0] == 0:
        return float(arc[int(np.argmin(abs(xi - run.xi_star)))]), xi
    k = above[0]
    w = (run.xi_star - xi[k - 1]) / (xi[k] - xi[k - 1])
    return float(arc[k - 1] + w * (arc[k] - arc[k - 1])), xi


def _truncation_offset(options, beta):
    """Truncation energy above the energy floor: max(truncation_kt / beta, truncation_min).

    The removed DVR points are hard walls; the rate converges once the margin above
    the valley floor is about six quanta of the stiffest transverse mode, which
    ``truncation_min`` (Eh) enforces independently of temperature.
    """
    return max(float(options["truncation_kt"]) / beta, float(options.get("truncation_min", 0.0) or 0.0))


def _hessian_step(options, masses):
    """Mass-weighted finite-difference step for a Cartesian displacement in bohr.

    A unit mass-weighted displacement moves an atom by ``1/sqrt(m)``; a fixed
    mass-weighted step of 1e-3 is therefore ~2e-5 bohr for hydrogen, where the
    gradient difference of a soft mode is comparable to the evaluation noise
    of a fitted (single-precision) potential and the sign of a small
    transverse curvature is not determined.
    """
    return float(options.get("hessian_displacement", 5e-3)) * float(np.sqrt(np.min(masses)))


def _model_provenance(path_info, validity, **extra):
    """What the validity gates need to travel with a model to its correction."""
    provenance = dict(extra)
    if path_info is not None:
        provenance["path_termination"] = {
            "reactant": path_info.get("reactant_termination"),
            "product": path_info.get("product_termination"),
        }
    provenance["validity_policy"] = validity_policy(validity)
    return provenance


def path_model(run: RateRun, path, options: Mapping[str, Any], *, path_source: str = "centroid"):
    """1D vibrationally adiabatic model on the arc length of the assembled path.

    ``options['path_info']`` (from :func:`build_path`) and ``options['validity']``
    travel into the model provenance, where the validity gates read them.
    """
    masses = run.cartesian_masses
    path_info, validity = options.get("path_info"), dict(options.get("validity") or {})
    arc = _arc(path, masses)
    energies = np.array([run.energy(p) for p in path])
    spacing = options.get("spacing_1d")
    if spacing is None:
        emax = float(np.ptp(energies)) + 30.0 / run.beta
        spacing = 0.5 * np.pi / np.sqrt(2.0 * emax) / float(options.get("refinement", 3))
    axis = np.arange(0.0, arc[-1] - 1e-9, float(spacing))
    rate_surface, _ = _dividing_arc(run, path, arc)
    # Bennett--Chandler needs a dividing surface the dynamics must cross: use
    # the barrier top of V_S (the quantum rate does not depend on it).
    environment = options.get("environment", "adiabatic")
    model = project_path_subspace(
        run.energy, run.gradient, path, masses, axis, environment=environment,
        dividing_surface=None, rigid_body=True,
        beta=run.beta if environment == "free_energy" else None,
        curvature=options.get("curvature", "none"),
        provenance=_model_provenance(path_info, validity, path_source=str(path_source)),
        zero_point=options.get("zero_point", "harmonic"),
        dvr_points=options.get("dvr_points", 81),
        dvr_half_width=options.get("dvr_half_width"),
        dvr_spacing=options.get("dvr_spacing"),
        finite_difference_step=_hessian_step(options, masses),
        soft_mode_frequency=float(validity.get("soft_mode_frequency", 2.3e-4)),
    )
    surface = model.dividing_surface
    fraction = float(options["absorber_fraction"])
    model_axis = model.axes[0]
    length = model_axis[-1] - model_axis[0]
    reference = model_axis[0] + (fraction + 0.05) * length
    model = type(model)(model.axes, model.values, model.masses, model.dividing_surface, model.reactant,
                        reference, model.provenance)
    product_reference = model_axis[-1] - (fraction + 0.05) * length
    thresholds = [float(np.interp(reference, model_axis, model.values)),
                  float(np.interp(product_reference, model_axis, model.values))]
    floor = max(thresholds) - float(options["threshold_margin"]) / run.beta
    quantum = {"cap_width": fraction * length, "energy_floor": floor, "dense_limit": int(options["dense_limit"])}
    info = {"grid_points": int(len(model_axis)), "spacing": float(model.spacing[0]), "dividing_surface": surface,
            "rate_dividing_surface_arc": rate_surface,
            "reactant_reference": reference, "channel_thresholds": thresholds, "energy_floor": floor,
            "model_construction": dict(model.provenance)}
    return model, quantum, info


def _saved_model_matches_options(saved: SubspaceCorrection, dimension: int,
                                 subspace_options: Mapping[str, Any]) -> bool:
    """Guard ``output.reuse`` against silently reusing a different H_S.

    The correction record stores :meth:`SubspaceModel.describe`, including the
    path construction provenance.  A selector change (harmonic/RPH/DVR), or a
    DVR grid change, changes H_S and therefore invalidates an old C_S record.
    Older records predate ``zero_point`` and are treated as historical
    harmonic projections for backwards compatibility.
    """
    construction = dict(saved.model.get("construction", {}))
    if int(saved.dimension) != int(dimension):
        return False
    if dimension == 1:
        expected_environment = subspace_options.get("environment", "adiabatic")
        if construction.get("environment", "adiabatic") != expected_environment:
            return False
        expected_zero_point = subspace_options.get("zero_point", "harmonic")
        if construction.get("zero_point", "harmonic") != expected_zero_point:
            return False
        if expected_zero_point in ("rph", "dvr") and construction.get("zero_point_definition") != "linear-normal-cuts-v2":
            return False
        if expected_zero_point == "dvr":
            if construction.get("dvr_points") != subspace_options.get("dvr_points", 81):
                return False
            for key in ("dvr_half_width", "dvr_spacing"):
                old, new = construction.get(key), subspace_options.get(key)
                if old is None or new is None:
                    if old is not None or new is not None:
                        return False
                elif not np.isclose(float(old), float(new), rtol=0.0, atol=1e-14):
                    return False
    elif construction.get("environment", "adiabatic") != subspace_options.get("environment", "adiabatic"):
        return False
    return True


def _in_plane_threshold(model, center, direction, *, window=0.05):
    """Lowest level of ``V_S`` across one channel along an in-plane line.

    The line through ``center`` along ``direction`` is restricted to the
    contiguous segment around ``center`` with ``V <= V(center) + window``
    (hartree), so a hairpin's other channel, which the same line may cross,
    is excluded.  Unit mass (mass-weighted coordinates).
    """
    step = 0.5 * float(min(model.spacing))
    extent = float(np.linalg.norm([a[-1] - a[0] for a in model.axes]))
    s = np.arange(-extent, extent + step, step)
    points = center[None, :] + s[:, None] * direction[None, :]
    inside = np.all((points >= [a[0] for a in model.axes]) & (points <= [a[-1] for a in model.axes]), axis=1)
    s, points = s[inside], points[inside]
    values = model.energy(points)
    middle = int(np.argmin(abs(s)))
    cap = values[middle] + window
    lo = hi = middle
    while lo > 0 and values[lo - 1] <= cap:
        lo -= 1
    while hi < len(s) - 1 and values[hi + 1] <= cap:
        hi += 1
    segment = slice(lo, hi + 1)
    if hi - lo < 8:
        raise ValueError("channel cross-section too short for a threshold; enlarge the 2D grid")
    levels = np.linalg.eigvalsh(sinc_dvr_kinetic(s[segment], 1.0) + np.diag(values[segment]))
    return float(levels[0])


def _path_box(arc, energies, k_ts, beta, options):
    """Reference slice, absorbers and box along the path (arc length, mass weighted).

    C_S needs the quantum and RPMD rates normalized on the same reference
    slice, so the slice only has to lie where the reactant channel is (nearly)
    free, not at the asymptote: the first point, going out from the TS, within
    `reference_kt` of the reactant asymptote.  Absorbers are two de Broglie
    wavelengths long.  Products are absorbed at the first point beyond the TS
    that has dropped `product_drop_kt` below the maximum or come within
    `reference_kt` of the product asymptote, whichever is first.  The second
    criterion bounds the box when the barrier is lower than `product_drop_kt`
    -- a thermoneutral reaction above V_barrier/(15 k_B), issue #195 -- where
    the first alone ran the box to the end of the path continuation, whose
    length is a numerical setting and not a property of the channel.
    """
    arc, energies = np.asarray(arc, dtype=float), np.asarray(energies, dtype=float)
    tolerance = float(options.get("reference_kt", 0.3)) / beta
    reactant_side = np.flatnonzero(np.abs(energies[:k_ts] - energies[0]) <= tolerance)
    k_ref = int(reactant_side[-1]) if len(reactant_side) else 0
    reference_arc = float(arc[k_ref])
    thermal_wavelength = 2.0 * np.pi / np.sqrt(2.0 / beta)
    absorb_lo = reference_arc - 0.25 * thermal_wavelength
    box_lo = absorb_lo - 2.0 * thermal_wavelength
    drop = float(options.get("product_drop_kt", 15.0)) / beta
    top = float(energies[k_ts:].max()) if k_ts < len(energies) else float(energies.max())
    beyond = np.arange(len(arc)) > k_ts
    dropped = beyond & (energies < top - drop)
    asymptotic = beyond & (np.abs(energies - energies[-1]) <= tolerance)
    product_side = np.flatnonzero(dropped | asymptotic)
    k_p = int(product_side[0]) if len(product_side) else len(arc) - 1
    criterion = ("path_end" if not len(product_side) else
                 "product_drop_kt" if dropped[k_p] else "reference_kt")
    product_wavelength = 2.0 * np.pi / np.sqrt(2.0 * max(top - energies[-1], 1.0 / beta))
    absorb_hi = float(arc[k_p])
    return {"reference_index": k_ref, "reference_arc": reference_arc, "product_index": k_p,
            "absorber_arc": [float(absorb_lo), absorb_hi],
            "box_arc": [float(box_lo), float(absorb_hi + 2.0 * product_wavelength)],
            "product_wavelength": float(product_wavelength), "product_absorber_criterion": criterion}


def plane_model(run: RateRun, path, direction, options: Mapping[str, Any]):
    """2D plane through the TS: rate dividing-surface normal x curvature normal."""
    path_info, validity = options.get("path_info"), dict(options.get("validity") or {})
    from scipy.spatial import cKDTree

    masses = run.cartesian_masses
    sqrt_m = np.sqrt(masses)
    ts = np.asarray(run.system.geometry, dtype=float).ravel()
    rigid = rigid_body_modes(ts.reshape(-1, 3), run.system.masses)
    e0 = run.xi_gradient(ts) / sqrt_m
    e0 -= rigid @ (rigid.T @ e0)
    e0 /= np.linalg.norm(e0)
    mw_path = (path - ts) * sqrt_m
    # Second axis: principal direction of the path displacement orthogonal to
    # the dividing-surface normal (and the rigid-body modes).  For a planar
    # path, e.g. a collinear atom transfer, this is exactly its plane.  The
    # curvature-normal candidate is reported as a cross-check.
    displaced = mw_path - np.outer(mw_path @ e0, e0)
    displaced -= (displaced @ rigid) @ rigid.T
    _, _, vt = np.linalg.svd(displaced, full_matrices=False)
    e1 = vt[0] - (vt[0] @ e0) * e0
    e1 /= np.linalg.norm(e1)
    candidate = np.asarray(direction, dtype=float)
    candidate = candidate - rigid @ (rigid.T @ candidate) - (candidate @ e0) * e0
    candidate_overlap = float(abs(candidate @ e1) / max(np.linalg.norm(candidate), 1e-300))
    plane = np.column_stack([mw_path @ e0, mw_path @ e1])
    off_plane = mw_path - plane @ np.vstack([e0, e1])
    off_plane = np.array([v - (r := rigid_body_modes(p.reshape(-1, 3), run.system.masses)) @ (r.T @ v)
                          for v, p in zip(off_plane, path)])
    residual = np.linalg.norm(off_plane, axis=1)
    arc = _arc(path, masses)
    energies = np.array([run.energy(p) for p in path])
    surface_arc, _ = _dividing_arc(run, path, arc)
    k_ts = int(np.argmin(abs(arc - surface_arc)))
    beta = run.beta
    box = _path_box(arc, energies, k_ts, beta, options)
    k_ref, reference_arc = box["reference_index"], box["reference_arc"]
    (absorb_lo, absorb_hi), (box_lo, box_hi) = box["absorber_arc"], box["box_arc"]
    product_wavelength, product_criterion = box["product_wavelength"], box["product_absorber_criterion"]
    inside = (arc >= box_lo) & (arc <= box_hi)
    padding = options.get("padding")
    if padding is None:
        padding = 40.0
    lo = plane[inside].min(axis=0) - padding
    hi = plane[inside].max(axis=0) + padding
    spacing = options.get("spacing_2d")
    floor_guess = float(max(energies[k_ref], energies[-1]))
    if spacing is None:
        # Transfer axis: a third of the narrowest transverse width along the
        # path (harmonic, in-plane); path axis: a quarter of the shortest de
        # Broglie wavelength up to the truncation energy.
        widths = []
        for k in np.flatnonzero(inside)[::5]:
            t = plane[min(k + 1, len(arc) - 1)] - plane[max(k - 1, 0)]
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            v = (n[0] * e0 + n[1] * e1)
            dx = 1e-3 * v / sqrt_m
            curvature = v @ ((run.gradient(path[k] + dx) - run.gradient(path[k] - dx)) / 2e-3 / sqrt_m)
            if curvature > 0:
                widths.append(1.0 / np.sqrt(2.0 * np.sqrt(curvature)))
        sigma = float(min(widths)) if widths else 5.0
        emax = (floor_guess - energies[inside].min()) + _truncation_offset(options, beta)
        spacing = [sigma / 3.0, max(sigma / 3.0, 0.5 * np.pi / np.sqrt(2.0 * emax))]
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    # One V_S for both halves of C_S: it is projected on a grid `refinement`
    # times finer than the quantum DVR grid.  The DVR uses every
    # `refinement`-th node (exact V_S values); RPMD uses the cubic spline of
    # the fine grid, which resolves the narrow transverse valleys that a
    # spline through the DVR nodes alone would ring across.
    refinement = max(1, int(options.get("refinement", 3)))
    fine = spacing / refinement
    counts = [int(np.ceil((hi[i] - lo[i]) / spacing[i])) * refinement + 1 for i in range(2)]
    axes = [lo[i] + fine[i] * np.arange(counts[i]) for i in range(2)]
    skip = (floor_guess + 2.0 * _truncation_offset(options, run.beta) + 0.05)
    environment = options.get("environment", "adiabatic")
    fine_model = project_linear_subspace(run.energy, run.gradient, ts, masses, np.column_stack([e0, e1]), axes,
                                         environment=environment, dividing_surface=0.0, rigid_body=True,
                                         skip_above=skip, wall_energy=skip + 1.0,
                                         finite_difference_step=_hessian_step(options, masses),
                                         beta=run.beta if environment == "free_energy" else None)
    reference_point = np.array([np.interp(reference_arc, arc, plane[:, 0]), np.interp(reference_arc, arc, plane[:, 1])])
    provenance = dict(fine_model.provenance, plane_basis=[e0.tolist(), e1.tolist()], refinement=refinement,
                      **_model_provenance(path_info, validity))
    rpmd_model = type(fine_model)(fine_model.axes, fine_model.values, fine_model.masses, 0.0, "asymptotic",
                                  float(reference_point[0]), provenance)
    model = type(fine_model)(tuple(a[::refinement] for a in fine_model.axes),
                             fine_model.values[::refinement, ::refinement], fine_model.masses, 0.0, "asymptotic",
                             float(reference_point[0]), provenance)
    tree = cKDTree(plane)
    width_lo, width_hi = absorb_lo - box_lo, box_hi - absorb_hi
    strength = 30.0 * np.sqrt(2.0 / run.beta)

    def absorber(points):
        _, index = tree.query(points)
        s = arc[index]
        eps = np.zeros(len(points))
        left, right = s < absorb_lo, s > absorb_hi
        eps[left] = strength / width_lo * ((absorb_lo - s[left]) / width_lo) ** 2
        eps[right] = strength / width_hi * ((s[right] - absorb_hi) / width_hi) ** 2
        return eps

    def tangent_normal(at):
        k = int(np.clip(np.searchsorted(arc, at), 1, len(arc) - 2))
        t = plane[k + 1] - plane[k - 1]
        t /= np.linalg.norm(t)
        return np.array([plane[k, 0], plane[k, 1]]), np.array([-t[1], t[0]])

    thresholds = []
    product_reference = min(absorb_hi + product_wavelength, float(arc[-1]))
    for at in (reference_arc, product_reference):
        center, normal = tangent_normal(at)
        thresholds.append(_in_plane_threshold(rpmd_model, center, normal))
    floor = max(thresholds) - float(options["threshold_margin"]) / run.beta
    truncation = floor + _truncation_offset(options, run.beta)
    quantum = {"absorber": absorber, "normalization": "slice", "energy_floor": floor,
               "truncation_energy": truncation, "dense_limit": int(options["dense_limit"])}
    retained = int(np.sum(model.values <= truncation))
    info = {"shape": list(model.shape), "spacing": spacing.tolist(), "retained_points": retained,
            "rpmd_grid_shape": list(rpmd_model.shape), "refinement": refinement,
            "curvature_normal_in_plane_overlap": candidate_overlap,
            "path_off_plane_max": float(residual.max()), "path_off_plane_rms": float(np.sqrt(np.mean(residual**2))),
            "absorber_arc": [float(absorb_lo), float(absorb_hi)], "box_arc": [float(box_lo), float(box_hi)],
            "product_absorber_criterion": product_criterion,
            "reference_arc": reference_arc, "reactant_reference": float(reference_point[0]),
            "channel_thresholds": thresholds, "energy_floor": floor, "truncation_energy": truncation}
    return model, quantum, info, rpmd_model


def _rpmd_options(options, run):
    rpmd = {k: v for k, v in options.items() if k != "nbeads"}
    return rpmd, int(options.get("nbeads") or run.nbeads)


def pipeline_identity(root, options):
    """Bind cached factors to samples, geometry, PES files and numerical settings."""
    from .. import __version__
    from ..io import file_sha256, fingerprint_payload, referenced_files

    names = ("input/resolved.json", "input/system.json", "preparation.json", "centroid_path.npz", "rate_pmf.json")
    files = {name: file_sha256(root / name) for name in names if (root / name).is_file()}
    saved = json.loads((root / "input" / "resolved.json").read_text())
    numerical = {k: v for k, v in options.items()
                 if k not in {"enabled", "mode", "reuse", "output", "source"}}
    # Hash the implementations too: a source checkout can change without a
    # version bump, and its old numerical result must not silently be reused.
    analysis = Path(__file__).resolve().parent
    implementation = {p.name: file_sha256(p) for p in
                      (analysis / name for name in ("subspace_pipeline.py", "subspace_model.py",
                       "subspace_quantum.py", "subspace_rpmd.py", "subspace_correction.py", "centroid_path.py"))}
    implementation["instanton.py"] = file_sha256(analysis.parent / "instanton.py")
    return fingerprint_payload({"schema": 1, "version": __version__, "files": files,
                                "potential_files": referenced_files({"potential": saved["potential"]}),
                                "options": numerical, "implementation": implementation})


def _read_specification(specification):
    base = None
    if isinstance(specification, (str, Path)):
        path = Path(specification)
        text = path.read_text()
        if path.suffix.lower() == ".json":
            spec = json.loads(text)
        else:
            import tomllib

            spec = tomllib.loads(text)
        base = path.resolve().parent
    else:
        spec = dict(specification)
    return _merge(DEFAULTS, spec), base


def _prepare_pipeline(options, base, reuse_key):
    """Open the saved run and bind its reuse identity before any model is built."""
    source = options["source"].get("rate_run")
    if not source:
        raise ValueError("source.rate_run is required")
    source = Path(source)
    if base is not None and not source.is_absolute():
        source = base / source
    run = load_rate_run(source)
    out_dir = options["output"].get("directory")
    out_dir = Path(out_dir) if out_dir else run.root / "subspace"
    if base is not None and not out_dir.is_absolute():
        out_dir = base / out_dir
    try:
        if reuse_key is None:
            reuse_key = pipeline_identity(run.root, options)
        return {"run": run, **_build_cases(run, out_dir, options, base, reuse_key)}
    except BaseException:
        _close_run(run)
        raise


def _with_capacity_gate(gates, plan):
    """Add the exact-solver capacity of a model to its validity gates.

    Capacity is not a validity override: an infeasible solve has no
    ``allow`` policy, because there is no rate to keep for diagnosis.
    """
    gates = dict(gates, gates=list(gates["gates"]), failures=list(gates["failures"]))
    reason = None if plan["feasible"] else f"{plan['code']}: {plan['message']}"
    gates["gates"].append({"gate": "capacity", "passed": bool(plan["feasible"]), "policy": "reject",
                           "overridden": False, "reason": reason,
                           "detail": {k: plan[k] for k in (
                               "code", "route", "requested_method", "retained_points", "grid_shape",
                               "dense_limit", "required_dense_limit", "estimated_dense_memory_mb",
                               "memory_limit_mb", "dense_only_features", "alternatives")}})
    if reason is not None:
        gates["failures"].append(reason)
        gates["valid"] = False
    return gates


def _build_cases(run, out_dir, options, base, reuse_key):
    """Build the path and subspace models and list the corrections still to solve."""
    from ..io import file_sha256

    out_dir.mkdir(parents=True, exist_ok=True)
    mean, mask, path, path_info = build_path(run, options["path"])
    np.save(out_dir / "path.npy", path)
    threshold = float(options["subspace"]["threshold"])
    sampled_cw = None
    try:
        sampled_cw = mean.curvature_width(mask, threshold=threshold)
    except ValueError as error:
        sampled_cw = {"error": str(error)}
    static_cw = _static_curvature_width(run, path, threshold)
    sampled_max = sampled_cw.get("maximum", 0.0) if isinstance(sampled_cw, dict) else 0.0
    verdict = 2 if max(static_cw["maximum"], sampled_max or 0.0) > threshold else 1
    requested = options["subspace"]["dimension"]
    dimensions = [1, 2] if requested == "auto" and verdict == 2 else ([1] if requested == "auto" else
                                                                     ([int(requested)] if np.ndim(requested) == 0 else
                                                                      [int(d) for d in requested]))
    rpmd_options, nbeads = _rpmd_options(options["rpmd"], run)
    corrections, models, cases = {}, {}, {}
    validity = dict(options.get("validity") or {})
    diagnose = validity.get("on_invalid", "stop") == "diagnose"
    if validity.get("on_invalid", "stop") not in ("stop", "diagnose"):
        raise ValueError("validity.on_invalid must be 'stop' or 'diagnose'")
    rejected, escalation = {}, None
    gating = {"path_info": path_info, "validity": validity,
              "hessian_displacement": options["subspace"].get("hessian_displacement", 5e-3)}

    # The task scheduler's per-solve memory budget (a node-wide upper bound
    # under MPI); the ordinary sequential route has none.
    budget = None
    if options.get("execution") is not None:
        from .subspace_execution import _options as _execution_options

        budget = _execution_options(options["execution"])["memory_limit_mb"]

    def admit(dimension, case, info):
        """Gate a model on what is known before any rate is solved (path, bath, solver capacity)."""
        gates = validity_gates(case["model"].provenance)
        # The solver's own routing rule on the model it would solve: a case the
        # exact solver cannot carry is never started, whatever `on_invalid` says.
        quantum = dict(case["quantum_options"])
        if budget is not None:
            quantum["memory_limit_mb"] = min(budget, quantum.get("memory_limit_mb") or budget)
        plan = quantum_route_plan(case["model"], **quantum)
        info["quantum_plan"] = plan
        gates = _with_capacity_gate(gates, plan)
        info["validity"] = gates
        models[dimension] = info
        if plan["feasible"] and (gates["valid"] or diagnose):
            cases[dimension] = case
        else:
            rejected[dimension] = gates
        return gates

    # With an execution table the task scheduler owns reuse through its own
    # identity-checked checkpoints; saved C_S records are reused only on the
    # ordinary sequential route.
    if options["output"].get("reuse") and options.get("execution") is None:
        previous = {}
        report_path = out_dir / "subspace_pipeline.json"
        if report_path.is_file():
            try:
                loaded = json.loads(report_path.read_text())
                previous = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError):
                pass  # A damaged report is not reusable evidence.
        matched = reuse_key is None or previous.get("reuse_key") == reuse_key
        for d in list(dimensions) if matched else []:
            record = out_dir / f"C_S_{d}d.json"
            if not record.is_file():
                continue
            if reuse_key is not None and previous.get("correction_files", {}).get(record.name) != file_sha256(record):
                continue
            try:
                saved = SubspaceCorrection.read(record)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (saved.dimension == d and saved.nbeads == nbeads
                    and np.isclose(saved.beta, run.beta, rtol=1e-9, atol=0)
                    and _saved_model_matches_options(saved, d, options["subspace"])):
                corrections[d] = saved
                models[d] = previous.get("models", {}).get(str(d), {"reused": True})
                dimensions.remove(d)
    if 1 in dimensions:
        environment = {k: options["subspace"].get(k) for k in
                       ("environment", "curvature", "zero_point", "dvr_points", "dvr_half_width", "dvr_spacing")}
        model1, quantum1, info1 = path_model(
            run, path, {**options["quantum"], **environment, **gating},
            path_source=path_info.get("source", "centroid"),
        )
        gates1 = admit(1, dict(model=model1, beta=run.beta, nbeads=nbeads, quantum_options=quantum1,
                               rpmd_options=rpmd_options), info1)
        bath = [g for g in gates1["gates"] if g["gate"] == "bath" and not g["passed"] and not g["overridden"]]
        if bath and requested == "auto" and 2 not in dimensions and 2 not in corrections:
            # An unstable normal plane means a transverse coordinate is not a
            # bath mode of this path: it belongs in S, whatever kappa*rho says.
            dimensions.append(2)
            escalation = {"from": 1, "to": 2, "reason": bath[0]["reason"]}
    if 2 in dimensions:
        candidate = None
        if isinstance(sampled_cw, dict) and "candidate_direction" in sampled_cw and sampled_max >= static_cw["maximum"]:
            candidate = np.asarray(sampled_cw["candidate_direction"])
        if candidate is None:
            candidate = np.asarray(static_cw["candidate_direction"])
        environment = {"environment": options["subspace"].get("environment", "adiabatic")}
        model2, quantum2, info2, rpmd_model2 = plane_model(run, path, candidate,
                                                           {**options["quantum"], **environment, **gating})
        admit(2, dict(model=model2, beta=run.beta, nbeads=nbeads, quantum_options=quantum2,
                      rpmd_options=rpmd_options, rpmd_model=rpmd_model2), info2)
    execution = options.get("execution")
    if execution is not None:
        execution = dict(execution)
        directory = Path(execution.get("checkpoint_dir") or out_dir / "tasks")
        if execution.get("checkpoint_dir") and base is not None and not directory.is_absolute():
            directory = base / directory
        execution["checkpoint_dir"] = str(directory)
    return dict(out_dir=out_dir, nbeads=nbeads, options=options, cases=cases, execution=execution,
                corrections=corrections, models=models, path_info=path_info, sampled_cw=sampled_cw,
                static_cw=static_cw, verdict=verdict, reuse_key=reuse_key, rejected=rejected,
                escalation=escalation)


def _close_run(run):
    close = getattr(run.provider, "close", None)
    if callable(close):
        close()


def _finish_pipeline(state, computed, apply):
    from ..io import file_sha256, write_json

    run, out_dir, nbeads = state["run"], state["out_dir"], state["nbeads"]
    options, models, corrections = state["options"], state["models"], state["corrections"]
    path_info, sampled_cw, static_cw = state["path_info"], state["sampled_cw"], state["static_cw"]
    verdict, reuse_key = state["verdict"], state["reuse_key"]
    rejected, escalation = state.get("rejected", {}), state.get("escalation")
    for dimension, correction in computed.items():
        correction.write(out_dir / f"C_S_{dimension}d.json")
        corrections[dimension] = correction
    # Only models that passed every gate are evidence about the defect; the
    # others keep their record (and component rates) but decide nothing.
    usable = {d: c for d, c in corrections.items() if c.valid}
    sensitivity = nested_subspace_sensitivity(list(usable.values())) if len(usable) > 1 else None
    if not usable:
        chosen, reason = None, "no subspace model passed the validity gates"
    elif sensitivity is None:
        chosen = max(usable)
        reason = "only dimension computed" if len(corrections) + len(rejected) == 1 else "only valid dimension"
    elif sensitivity["recommended_dimension"] in usable:
        chosen, reason = sensitivity["recommended_dimension"], "nested sensitivity"
    else:
        # Undetermined nested step: fall back on curvature x width.
        chosen = verdict if verdict in usable else max(usable)
        reason = "nested sensitivity undetermined; curvature x width verdict"
    target = out_dir / "rate_quantum_corrected.json"
    corrected = correct_rate_run(run.root, usable[chosen], output=target) if apply and chosen is not None else None
    if chosen is None and target.is_file():
        # A corrected rate from an earlier run must not outlive the model it came from.
        try:
            stale = json.loads(target.read_text()).get("schema") == CORRECTED_SCHEMA
        except (OSError, ValueError, AttributeError):
            stale = False
        if stale:
            target.unlink()
    dimension_validity = {str(d): c.validity for d, c in corrections.items()}
    dimension_validity.update({str(d): gates for d, gates in rejected.items()})
    failures = {d: v.get("failures", []) for d, v in dimension_validity.items() if v.get("failures")}
    # Models the exact solver could not carry, with retained points, limits,
    # memory and alternatives; never solved, never applied.
    capacity = {str(d): models[d]["quantum_plan"] for d in sorted(rejected)
                if isinstance(models.get(d), dict) and not models[d].get("quantum_plan", {}).get("feasible", True)}
    if chosen is not None:
        status = "complete"
    else:
        status = "capacity_exceeded" if capacity else "invalid_model"
    report = {
        "schema": "pyrpmd.subspace_pipeline", "schema_version": 1,
        "status": status,
        "rate_run": str(run.root), "beta": run.beta, "nbeads": nbeads, "xi_star": run.xi_star,
        "path": path_info, "curvature_width_sampled": sampled_cw, "curvature_width_path": static_cw,
        "recommended_dimension_from_curvature_width": verdict, "dimensions_computed": sorted(corrections),
        "dimension_escalation": escalation,
        "validity": {"valid": chosen is not None, "valid_dimensions": sorted(usable),
                     "rejected_before_solving": sorted(rejected), "failures": failures,
                     "capacity_exceeded": capacity,
                     "dimensions": dimension_validity, "policy": options.get("validity")},
        "models": models,
        # Keep the saved quantum/RPMD curves when reusing a scalar correction
        # object (SubspaceCorrection.read deliberately loads only its summary).
        "corrections": {str(d): json.loads((out_dir / f"C_S_{d}d.json").read_text()) for d in corrections},
        "nested_sensitivity": sensitivity,
        "applied_dimension": None if chosen is None else int(chosen), "applied_dimension_reason": reason,
        "corrected_rate": corrected,
        "mode": "apply" if apply else "factor_only",
        "reuse_key": reuse_key,
        "correction_files": {f"C_S_{d}d.json": file_sha256(out_dir / f"C_S_{d}d.json") for d in corrections},
        "options": {k: v for k, v in options.items()},
        "physical_converged": False,
    }
    report = _json_safe(report)
    write_json(out_dir / "subspace_pipeline.json", report)
    return report


def run_subspace_pipeline(specification, *, apply=True, reuse_key=None) -> dict:
    """Run from a mapping or TOML/JSON; ``apply=False`` writes factors only.

    Cached factors are tied to source data, numerical settings and solver
    implementations for native and standalone callers alike. With
    ``execution.backend='mpi'`` every rank participates in the rate solves.
    """
    from ..parallel import root_call
    from ..parallel.execution import _optional_world
    from .subspace_execution import subspace_corrections

    # Only small options/reports cross these collectives. The preparation state
    # contains a live PES and stays on root; task models travel through the task
    # scheduler after its absorber closures have been evaluated on the DVR grid.
    call = root_call if _optional_world() is not None else lambda function, *args: function(*args)
    options, base = call(_read_specification, specification)
    state = {}

    def prepare():
        state.update(_prepare_pipeline(options, base, reuse_key))
        return state["execution"]

    def release():
        if "run" in state:
            _close_run(state["run"])

    try:
        execution = call(prepare)
        if execution is not None and execution.get("backend") == "mpi":
            computed = subspace_corrections(state.get("cases"), execution=execution)
        else:
            def compute():
                if execution is not None:
                    return subspace_corrections(state["cases"], execution=execution)
                return {key: subspace_correction(**case) for key, case in state["cases"].items()}

            computed = call(compute)
        return call(_finish_pipeline, state, computed, apply)
    finally:
        call(release)


__all__ = ["RateRun", "build_path", "load_rate_run", "path_model", "plane_model", "run_subspace_pipeline"]
