"""One-command subspace quantum correction of a saved PyRPMD rate run.

    rate run (output.centroid_path = true)
      -> ring-polymer mean path xbar(xi) from the umbrella windows
      -> path to both asymptotes (sampled core + steepest-descent continuation)
      -> curvature x width: Sigma(xi) where sampled, harmonic width elsewhere
      -> 1D vibrationally adiabatic path model           -> C_S(1D)
      -> if kappa*rho > threshold: 2D plane through the TS spanned by the
         rate dividing-surface normal and the curvature normal at the peak
         (the collinear plane for a collinear atom transfer)  -> C_S(2D)
      -> nested sensitivity ln(C_2D/C_1D), recommended C_S
      -> rate_quantum_corrected.json next to the run

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
from .subspace_correction import (SubspaceCorrection, correct_rate_run, nested_subspace_sensitivity,
                                  subspace_correction)
from .subspace_model import (_vibrational_complement, assemble_reaction_path, path_curvature,
                             project_linear_subspace, project_path_subspace, relaxed_path)
from .subspace_quantum import _json_safe, sinc_dvr_kinetic

DEFAULTS = {
    "source": {"rate_run": None},
    "path": {"step": 10.0, "max_steps": 600, "spread_factor": 2.0, "asymptote_tolerance": 1e-6,
             "smoothing": 1.5},
    "subspace": {"dimension": "auto", "threshold": 0.5, "environment": "adiabatic"},
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

    root = _root(directory, "summary.json")
    config = json.loads((root / "input" / "resolved.json").read_text())
    system = System.from_dict(json.loads((root / "input" / "system.json").read_text()))
    preparation = json.loads((root / "preparation.json").read_text()) if (root / "preparation.json").is_file() else {}
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


def build_path(run: RateRun, options: Mapping[str, Any]):
    """Ring-polymer mean path (orientation-locked windows) extended to both asymptotes."""
    mean = mean_path_from_rate_run(run.root)
    mask = mean.locked(factor=float(options["spread_factor"]))
    core = mean.path(mask).reshape(int(mask.sum()), -1)
    xi_core = np.sort(mean.xi[mask])
    sampled = {"windows": int(len(mean.xi)), "locked_windows": int(mask.sum()),
               "locked_xi": [float(v) for v in np.sort(mean.xi[mask])],
               "orientation_spread": mean.orientation_spread.tolist()}
    if len(core) < 2:
        raise ValueError("fewer than two orientation-locked windows; add umbrella windows near the transition state")
    smoothing = options.get("smoothing")
    if smoothing and len(core) >= 5:
        # Fit xbar(xi) against the sampling coordinate itself: window means
        # carry transverse noise, and a chord-length parametrization of noisy
        # points zig-zags.  Local quadratic regression in xi, evaluated on a
        # uniform xi grid, gives a smooth, monotonically parametrized core.
        bandwidth = float(smoothing) * 2.0 * float(np.median(np.diff(xi_core)))
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
    path, info = assemble_reaction_path(
        core, run.energy, run.gradient, run.cartesian_masses, step=float(options["step"]),
        max_steps=int(options["max_steps"]), rigid_body=True,
        asymptote_tolerance=float(options["asymptote_tolerance"]),
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


def path_model(run: RateRun, path, options: Mapping[str, Any]):
    """1D vibrationally adiabatic model on the arc length of the assembled path."""
    masses = run.cartesian_masses
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
    model = project_path_subspace(run.energy, run.gradient, path, masses, axis, environment=environment,
                                  dividing_surface=None, rigid_body=True,
                                  beta=run.beta if environment == "free_energy" else None)
    surface = model.dividing_surface
    fraction = float(options["absorber_fraction"])
    length = axis[-1] - axis[0]
    reference = axis[0] + (fraction + 0.05) * length
    model = type(model)(model.axes, model.values, model.masses, model.dividing_surface, model.reactant,
                        reference, model.provenance)
    product_reference = axis[-1] - (fraction + 0.05) * length
    thresholds = [float(np.interp(reference, axis, model.values)),
                  float(np.interp(product_reference, axis, model.values))]
    floor = max(thresholds) - float(options["threshold_margin"]) / run.beta
    quantum = {"cap_width": fraction * length, "energy_floor": floor, "dense_limit": int(options["dense_limit"])}
    info = {"grid_points": int(len(axis)), "spacing": float(spacing), "dividing_surface": surface,
            "rate_dividing_surface_arc": rate_surface,
            "reactant_reference": reference, "channel_thresholds": thresholds, "energy_floor": floor}
    return model, quantum, info


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


def plane_model(run: RateRun, path, direction, options: Mapping[str, Any]):
    """2D plane through the TS: rate dividing-surface normal x curvature normal."""
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
    # Box along the path.  C_S needs the quantum and RPMD rates normalized on
    # the same reference slice, so the slice only has to lie where the
    # reactant channel is (nearly) free, not at the asymptote: the first
    # point, going out from the TS, within `reference_kt` of the asymptote.
    # Absorbers are two de Broglie wavelengths long; products are absorbed
    # once the path has dropped `product_drop_kt` below its maximum.
    tolerance = float(options.get("reference_kt", 0.3)) / beta
    reactant_side = np.flatnonzero(np.abs(energies[:k_ts] - energies[0]) <= tolerance)
    k_ref = int(reactant_side[-1]) if len(reactant_side) else 0
    reference_arc = float(arc[k_ref])
    thermal_wavelength = 2.0 * np.pi / np.sqrt(2.0 / beta)
    absorb_lo = reference_arc - 0.25 * thermal_wavelength
    box_lo = absorb_lo - 2.0 * thermal_wavelength
    drop = float(options.get("product_drop_kt", 15.0)) / beta
    top = float(energies[k_ts:].max()) if k_ts < len(energies) else float(energies.max())
    product_side = np.flatnonzero((np.arange(len(arc)) > k_ts) & (energies < top - drop))
    k_p = int(product_side[0]) if len(product_side) else len(arc) - 1
    product_wavelength = 2.0 * np.pi / np.sqrt(2.0 * max(top - energies[-1], 1.0 / beta))
    absorb_hi = float(arc[k_p])
    box_hi = absorb_hi + 2.0 * product_wavelength
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
                                         beta=run.beta if environment == "free_energy" else None)
    reference_point = np.array([np.interp(reference_arc, arc, plane[:, 0]), np.interp(reference_arc, arc, plane[:, 1])])
    provenance = dict(fine_model.provenance, plane_basis=[e0.tolist(), e1.tolist()], refinement=refinement)
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
            "reference_arc": reference_arc, "reactant_reference": float(reference_point[0]),
            "channel_thresholds": thresholds, "energy_floor": floor, "truncation_energy": truncation}
    return model, quantum, info, rpmd_model


def _rpmd_options(options, run):
    rpmd = {k: v for k, v in options.items() if k != "nbeads"}
    return rpmd, int(options.get("nbeads") or run.nbeads)


def run_subspace_pipeline(specification) -> dict:
    """Run the full correction from a mapping or a TOML/JSON file."""
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
    options = _merge(DEFAULTS, spec)
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
    corrections, models = {}, {}
    if options["output"].get("reuse"):
        # Re-select and re-apply from saved C_S records (same temperature)
        # without recomputing the subspace rates.
        previous = {}
        if (out_dir / "subspace_pipeline.json").is_file():
            previous = json.loads((out_dir / "subspace_pipeline.json").read_text()).get("models", {})
        for d in list(dimensions):
            record = out_dir / f"C_S_{d}d.json"
            if record.is_file():
                saved = SubspaceCorrection.read(record)
                if np.isclose(saved.beta, run.beta, rtol=1e-9, atol=0):
                    corrections[d] = saved
                    models[d] = previous.get(str(d), {"reused": True})
                    dimensions.remove(d)
    if 1 in dimensions:
        environment = {"environment": options["subspace"].get("environment", "adiabatic")}
        model1, quantum1, info1 = path_model(run, path, {**options["quantum"], **environment})
        corrections[1] = subspace_correction(model1, run.beta, nbeads, quantum_options=quantum1,
                                             rpmd_options=rpmd_options)
        corrections[1].write(out_dir / "C_S_1d.json")
        models[1] = info1
    if 2 in dimensions:
        candidate = None
        if isinstance(sampled_cw, dict) and "candidate_direction" in sampled_cw and sampled_max >= static_cw["maximum"]:
            candidate = np.asarray(sampled_cw["candidate_direction"])
        if candidate is None:
            candidate = np.asarray(static_cw["candidate_direction"])
        environment = {"environment": options["subspace"].get("environment", "adiabatic")}
        model2, quantum2, info2, rpmd_model2 = plane_model(run, path, candidate, {**options["quantum"], **environment})
        corrections[2] = subspace_correction(model2, run.beta, nbeads, quantum_options=quantum2,
                                             rpmd_options=rpmd_options, rpmd_model=rpmd_model2)
        corrections[2].write(out_dir / "C_S_2d.json")
        models[2] = info2
    sensitivity = nested_subspace_sensitivity(list(corrections.values())) if len(corrections) > 1 else None
    if sensitivity is None:
        chosen, reason = max(corrections), "only dimension computed"
    elif sensitivity["recommended_dimension"] in corrections:
        chosen, reason = sensitivity["recommended_dimension"], "nested sensitivity"
    else:
        # Undetermined nested step: fall back on curvature x width.
        chosen = verdict if verdict in corrections else max(corrections)
        reason = "nested sensitivity undetermined; curvature x width verdict"
    corrected = correct_rate_run(run.root, corrections[chosen], output=out_dir / "rate_quantum_corrected.json")
    report = {
        "schema": "pyrpmd.subspace_pipeline", "schema_version": 1,
        "rate_run": str(run.root), "beta": run.beta, "nbeads": nbeads, "xi_star": run.xi_star,
        "path": path_info, "curvature_width_sampled": sampled_cw, "curvature_width_path": static_cw,
        "recommended_dimension_from_curvature_width": verdict, "dimensions_computed": sorted(corrections),
        "models": models, "corrections": {str(d): c.to_dict() for d, c in corrections.items()},
        "nested_sensitivity": sensitivity, "applied_dimension": int(chosen), "applied_dimension_reason": reason,
        "corrected_rate": corrected,
        "options": {k: v for k, v in options.items()},
        "physical_converged": False,
    }
    report = _json_safe(report)
    (out_dir / "subspace_pipeline.json").write_text(json.dumps(report, indent=2, default=str))
    return report


__all__ = ["RateRun", "build_path", "load_rate_run", "path_model", "plane_model", "run_subspace_pipeline"]
