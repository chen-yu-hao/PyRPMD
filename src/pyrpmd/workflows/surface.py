"""Fixed-cell surface FES and basin-normalized first-order RPMD rates.

Affine and smooth distance centroid coordinates have a constant mobile
mass metric, making constrained-parent and delta-coordinate measures
equivalent up to a cancelling constant. The FES is a *position density*, not
the gas-phase RPMDrate flux-metric PMF. See docs/surface-catalysis.md.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..cv import Cartesian, CollectiveVariable, SurfaceHeight, SurfaceLateral, Distance, PeriodicDistance
from ..parallel import root_call, run_tasks
from ..units import parse_quantity
from .fes import resolve_umbrellas, run_fes
from .common import write_json, _atomic_npz, jsonable


class _AffineSurface(CollectiveVariable):
    units = ("bohr",)

    def __init__(self, gradient, offset):
        super().__init__(estimator="centroid", name="surface_dividing_surface")
        self.gradient = np.array(gradient, dtype=float, copy=True)
        self.offset = float(offset)

    def centroid_value_gradient(self, q):
        if np.shape(q) != self.gradient.shape:
            raise ValueError("surface coordinate atom count mismatch")
        return float(np.sum(self.gradient * q) + self.offset), self.gradient.copy()

    def centroid_value_gradient_hessian(self, q):
        value, gradient = self.centroid_value_gradient(q)
        return value, gradient, np.zeros(q.shape + q.shape)


class SurfaceCoordinate(_AffineSurface):
    """Affine coordinate with immobile atom components removed from its metric."""

    def __init__(self, cv, system, product_side):
        value, gradient = cv.centroid_value_gradient(system.geometry)
        gradient = np.array(gradient, dtype=float, copy=True)
        gradient[list(system.fixed_atoms)] = 0.0
        super().__init__(gradient, value - np.sum(gradient * system.geometry))
        self.product_sign = 1.0 if product_side == "higher" else -1.0
        self.mass_metric = float(np.sum(gradient**2 / system.masses[:, None]))
        if not np.isfinite(self.mass_metric) or self.mass_metric <= 0:
            raise ValueError("surface rate coordinate has no mobile degree of freedom")

    def dividing_surface(self, value):
        return _AffineSurface(
            self.product_sign * self.gradient,
            self.product_sign * (self.offset - float(value)),
        )


class DistanceSurface(CollectiveVariable):
    """Exact curved distance surface, with frozen coordinates excluded.

    This is not a tangent-plane approximation. Values are evaluated at the
    actual centroid and both derivatives follow that same distance function
    on the mobile subspace. With at least one mobile endpoint, the metric is
    the sum of inverse mobile endpoint masses, independent of orientation.
    """
    units = ("bohr",)

    def __init__(self, cv, system, *, sign=1.0, dividing=0.0):
        super().__init__(estimator="centroid", name="distance_dividing_surface")
        self.cv, self.system = cv, system
        self.sign, self.dividing = float(sign), float(dividing)

    def _positions(self, q):
        q = np.array(q, dtype=float, copy=True)
        if q.shape != self.system.geometry.shape:
            raise ValueError("surface coordinate atom count mismatch")
        q[list(self.system.fixed_atoms)] = self.system.geometry[list(self.system.fixed_atoms)]
        return q

    def centroid_value_gradient(self, q):
        value, gradient = self.cv.centroid_value_gradient(self._positions(q))
        gradient[list(self.system.fixed_atoms)] = 0.0
        return self.sign * (value - self.dividing), self.sign * gradient

    def centroid_value_gradient_hessian(self, q):
        value, gradient, hessian = self.cv.centroid_value_gradient_hessian(self._positions(q))
        fixed = list(self.system.fixed_atoms)
        gradient[fixed] = 0.0
        hessian[fixed, :, :, :] = 0.0
        hessian[:, :, fixed, :] = 0.0
        return self.sign * (value - self.dividing), self.sign * gradient, self.sign * hessian


class DistanceSurfaceCoordinate(DistanceSurface):
    def __init__(self, cv, system, product_side):
        super().__init__(cv, system)
        self.product_sign = 1.0 if product_side == "higher" else -1.0
        self.mass_metric = float(sum(1.0 / system.masses[i] for i in cv.atoms if i not in system.fixed_atoms))
        if self.mass_metric <= 0:
            raise ValueError("surface rate coordinate has no mobile degree of freedom")
        self.periodic_safe_radius_bohr = None
        if type(cv) is PeriodicDistance:
            basis = cv.cell[np.asarray(cv.pbc, bool)]
            if not len(basis):
                raise ValueError("periodic_distance surface requires a periodic lattice direction")
            # |n B| >= sigma_min(B) for every nonzero lattice integer n.
            # Half this lower bound is a conservative seam-free ball even
            # for skew, unreduced cells; no shortest-vector guess is used.
            self.periodic_safe_radius_bohr = .5 * float(np.linalg.svd(basis, compute_uv=False)[-1])

    def dividing_surface(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("distance dividing surface must be positive and finite")
        if self.periodic_safe_radius_bohr is not None and value >= self.periodic_safe_radius_bohr * (1.0 - 1e-12):
            raise ValueError("periodic_distance dividing surface must be below half the smallest singular value of the periodic lattice basis, away from image-switch seams")
        return DistanceSurface(self.cv, self.system, sign=self.product_sign, dividing=value)


def surface_definition(config, system):
    """Validate a basin and a centroid CV with constant mobile mass metric.

    All bare numbers in [surface] are canonical bohr, independently of the
    optional display/input unit for umbrella centers. Unit-bearing strings are
    recommended. The finite basin defines the population being normalized.
    """
    raw = config.get("surface", {})
    unknown = set(raw) - {"reactant_basin", "dividing_surface", "product_side", "max_relative_standard_error", "normalization", "free_energy"}
    if unknown:
        raise ValueError("unknown surface options: " + ", ".join(sorted(unknown)))
    if "reactant_basin" not in raw or "dividing_surface" not in raw:
        raise ValueError("surface_rate requires surface.reactant_basin and surface.dividing_surface")
    from ..analysis.surface import surface_rate_error_target
    error_target = surface_rate_error_target(raw.get("max_relative_standard_error", .2))
    if config.get("analysis", {}).get("measure", "cv_density") != "cv_density":
        raise ValueError("surface_rate requires the cv_density sampling measure")
    from_source = bool(raw.get("free_energy"))
    method = "source" if from_source else "mbar"
    if config.get("analysis", {}).get("method", method) != method:
        raise ValueError(f"surface_rate analysis.method must be {method}")
    # A cluster is permitted as well as a fixed-cell slab. Boundary handling
    # belongs to the PES; there is no implicit area or volume conversion here.
    if from_source:
        from .surface_source import source_context
        context = source_context(config, system)
        cv = context["coordinate"]
        analysis = config.get("analysis", {})
        if set(analysis) - {"method", "measure", "bootstrap"} or analysis.get("bootstrap", 0) != 0:
            raise ValueError("source PMF uses its saved grid; analysis accepts method, measure and bootstrap=0 only")
    else:
        cv, specs, centers, _ = resolve_umbrellas(config, system)
    if len(cv.cvs) != 1 or type(cv.cvs[0]) not in (Cartesian, SurfaceHeight, SurfaceLateral, Distance, PeriodicDistance):
        raise ValueError("surface_rate needs one cartesian/surface_height/surface_lateral/distance/periodic_distance CV with constant mass metric")
    component = cv.cvs[0]
    if any(name in component.__dict__ for name in (
        "value", "value_and_jacobian", "centroid_value_gradient", "centroid_value_gradient_hessian",
        "_affine_gradient",
    )):
        raise ValueError("surface_rate does not accept overridden CV methods; the mass metric must be constant")
    if component.estimator != "centroid" or component.periods != (None,):
        raise ValueError("surface_rate requires a centroid CV with no circular period")
    values = raw["reactant_basin"]
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 2:
        raise ValueError("surface.reactant_basin must contain two ordered finite endpoints")
    basin = [parse_quantity(v, "length") for v in values]
    dividing = parse_quantity(raw["dividing_surface"], "length")
    if not np.isfinite(basin + [dividing]).all() or not basin[0] < basin[1]:
        raise ValueError("surface basin must have positive finite width")
    side = raw.get("product_side", "higher")
    if side not in ("higher", "lower"):
        raise ValueError("surface.product_side must be higher or lower")
    if (side == "higher" and basin[1] > dividing) or (side == "lower" and basin[0] < dividing):
        raise ValueError("surface reactant basin must lie entirely on the reactant side of the dividing surface")
    domain = [min(basin[0], dividing), max(basin[1], dividing)]
    if not from_source and (centers[:, 0].min() > domain[0] or centers[:, 0].max() < domain[1]):
        raise ValueError("surface umbrella centers must cover the complete reactant basin through the dividing surface")
    recrossing_surface = config.get("recrossing", {}).get("dividing_surface")
    if recrossing_surface is not None:
        raise ValueError("set the single dividing surface in [surface], not [recrossing]")
    is_distance = type(component) in (Distance, PeriodicDistance)
    if is_distance and min(*basin, dividing) <= 0:
        raise ValueError("distance basin and dividing surface must be positive, away from coincident atoms")
    coordinate = (DistanceSurfaceCoordinate if is_distance else SurfaceCoordinate)(component, system, side)
    if is_distance:
        coordinate.dividing_surface(dividing)  # Validate the actual curved surface.
    safe_radius = getattr(coordinate, "periodic_safe_radius_bohr", None)
    from ..analysis.surface import surface_normalization
    normalization = surface_normalization(raw.get("normalization", {}), system)
    return {
        "coordinate": coordinate,
        "cv_name": component.name,
        "reactant_basin_bohr": basin,
        "dividing_surface_bohr": float(dividing),
        "product_side": side,
        "mass_metric_au": coordinate.mass_metric,
        "max_relative_standard_error": error_target,
        "periodic_safe_radius_bohr": safe_radius,
        "rate_unit": "s^-1",
        "observable": "selected_surface_flux_side",
        "rate_scope": "single_local_exit",
        "product_indicator": "s > dividing_surface" if side == "higher" else "s < dividing_surface",
        "normalization": "per reactant in the specified basin, through the selected dividing surface",
        "concentration_normalization": normalization,
        "boundary": {"cell_bohr": system.cell, "pbc": system.pbc, "fixed_atoms": system.fixed_atoms},
    }


def _pmf_and_parents(sampled, config, system, output, options):
    from ..analysis.surface import surface_rate_pmf

    definition = surface_definition(config, system)
    streams = sampled["streams"]
    analysis = config.get("analysis", {})
    ranges = analysis.get("ranges")
    if ranges is not None:
        ranges = [[parse_quantity(x, "length") for x in pair] for pair in ranges]
    bins = analysis.get("bins", 100)
    if not np.isscalar(bins) and len(bins) == 1:
        bins = bins[0]
    pmf = surface_rate_pmf(
        [s["samples"] for s in streams],
        [s["center"] for s in streams],
        [s["stiffness"] for s in streams],
        config["beta"],
        basin=definition["reactant_basin_bohr"],
        dividing_surface=definition["dividing_surface_bohr"],
        mass_metric=definition["mass_metric_au"],
        bins=bins,
        ranges=ranges,
        bootstrap=analysis.get("bootstrap", analysis.get("bootstrap_replicates", analysis.get("bootstrap_samples", 40))),
        seed=analysis.get("seed", config.get("seed", 0)),
        min_count=analysis.get("min_count", 5),
        overlap_threshold=analysis.get("overlap_threshold", 1e-6),
        reweighting=sampled["data"].get("_reweighting"),
    )
    pmf["surface"] = {k: v for k, v in definition.items() if k != "coordinate"}
    pmf["warnings"] = list(dict.fromkeys(sampled["summary"]["warnings"] + pmf.get("warnings", []) + [
        "The finite reactant basin, slab size, bead count, timestep and finite-time plateau require independent convergence studies.",
        "This is a first-order rate through one selected surface; no gas-phase, area, site-degeneracy or coverage prefactor is applied.",
    ]))
    path = Path(output)
    _atomic_npz(
        path / "surface_pmf.npz",
        xi=pmf["xi"], free_energy=pmf["free_energy"], mask=pmf["mask"],
        bootstrap_qtst_s_inv=np.asarray(pmf["pmf_bootstrap_qtst"]),
        weights=pmf["weights"],
    )
    np.savetxt(
        path / "surface_pmf.csv",
        np.column_stack([pmf["xi"], pmf["free_energy"], pmf["mask"]]),
        delimiter=",", comments="", header="coordinate_bohr,free_energy_hartree,masked",
    )
    # The JSON remains compact; full curve arrays live in NPZ/CSV.
    public = {k: v for k, v in pmf.items() if k not in ("xi", "free_energy", "mask", "weights")}
    public["format_version"] = 2
    public["sample_weights_file"] = str(path / "surface_pmf.npz")
    public["sample_weights_key"] = "weights"
    public = jsonable(public)
    write_json(path / "surface_pmf.json", public)
    if not pmf["ready"]:
        return public, []
    ordered = sorted(streams, key=lambda s: abs(float(s["center"][0]) - pmf["xi_star"]))
    tasks = [
        {
            "config": config, "system": system, "options": options, "chain": chain,
            "xi_star": pmf["xi_star"], "initial_q": ordered[chain % len(ordered)]["last_q"],
            "output": str(path),
        }
        for chain in range(options["parent_replicas"])
    ]
    return public, tasks


def run_surface_rate(config, system, output, executor=None):
    """Umbrella density -> basin QTST -> constrained parents -> paired flux."""
    from .rate import _recrossing_options, _parent_task, _child_tasks, _aggregate, _dispatch_children

    config = copy.deepcopy(config)
    config["task"] = "surface_rate"
    if config.get("surface", {}).get("free_energy"):
        from .surface_source import source_paths
        from ..io import file_sha256
        paths = source_paths(config["surface"]["free_energy"])
        config["_surface_source_sha256"] = {k: file_sha256(v) for k, v in paths.items() if k.endswith("_file") or k == "selection"}
    definition = surface_definition(config, system)
    options = _recrossing_options(config)
    sampling_config = copy.deepcopy(config)
    sampling_config.setdefault("analysis", {}).update(measure="cv_density", bootstrap=0)
    from_source = bool(config.get("surface", {}).get("free_energy"))
    if from_source:
        from .surface_source import source_pmf_and_parents
        pmf, parent_tasks = root_call(source_pmf_and_parents, config, system, str(output), options)
    else:
        sampled = run_fes(sampling_config, system, Path(output) / "umbrella", executor=executor, return_data=True)
        pmf, parent_tasks = root_call(_pmf_and_parents, sampled, config, system, str(output), options)

    def dispatch(function, tasks):
        return executor.map(function, tasks) if executor is not None else run_tasks(function, tasks, config.get("parallel", {}))

    children, parents = [], []
    if parent_tasks:
        parents = dispatch(_parent_task, parent_tasks)
        tasks = root_call(_child_tasks, parents, config, system, str(output), options, pmf)
        children = _dispatch_children(tasks, config, dispatch)
    result = root_call(_aggregate, children, pmf, config, str(output), options)
    if result is not None:
        if from_source:
            result["source_parent_sampling"] = root_call(
                lambda rows: [{"chain": p["chain"], "attempted": options["parent_samples"],
                               "accepted": len(p["snapshots"])} for p in rows], parents)
            # A point estimate and a recrossing plateau cannot certify an
            # imported finite-time FES or supply its missing uncertainty.
            result["rate_s_inv"] = None
            if (result.get("rate_estimate_s_inv") is not None
                    and result.get("dynamics") != "trpmd"):
                result["status"] = "projected_pmf_pilot"
        result.update(
            rate_unit="s^-1",
            surface={k: v for k, v in definition.items() if k != "coordinate"},
        )
        from ..analysis.surface import normalize_surface_rates
        result["normalized_rates"] = normalize_surface_rates(result, definition["concentration_normalization"])
        root_call(write_json, Path(output) / "summary.json", result)
    return result
