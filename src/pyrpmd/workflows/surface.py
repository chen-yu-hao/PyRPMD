"""Fixed-cell surface FES and basin-normalized first-order RPMD rates.

Only affine centroid coordinates are accepted for the rate estimator. Their
constant mass metric makes constrained-parent and delta-coordinate measures
equivalent up to a cancelling constant. The FES is a *position density*, not
the gas-phase RPMDrate flux-metric PMF. See docs/surface-catalysis.md.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..cv import Cartesian, CollectiveVariable, SurfaceHeight, SurfaceLateral
from ..parallel import root_call, run_tasks
from ..units import parse_quantity
from .fes import resolve_umbrellas, run_fes, write_json, _atomic_npz, jsonable


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


def surface_definition(config, system):
    """Validate the explicitly chosen basin and a single affine centroid CV.

    All bare numbers in [surface] are canonical bohr, independently of the
    optional display/input unit for umbrella centers. Unit-bearing strings are
    recommended. The finite basin defines the population being normalized.
    """
    raw = config.get("surface", {})
    unknown = set(raw) - {"reactant_basin", "dividing_surface", "product_side"}
    if unknown:
        raise ValueError("unknown surface options: " + ", ".join(sorted(unknown)))
    if "reactant_basin" not in raw or "dividing_surface" not in raw:
        raise ValueError("surface_rate requires surface.reactant_basin and surface.dividing_surface")
    if config.get("analysis", {}).get("measure", "cv_density") != "cv_density":
        raise ValueError("surface_rate requires the cv_density sampling measure")
    if config.get("analysis", {}).get("method", "mbar") != "mbar":
        raise ValueError("surface_rate uses the MBAR density estimator")
    # A cluster is permitted as well as a fixed-cell slab. Boundary handling
    # belongs to the PES; there is no implicit area or volume conversion here.
    cv, specs, centers, _ = resolve_umbrellas(config, system)
    if len(cv.cvs) != 1 or type(cv.cvs[0]) not in (Cartesian, SurfaceHeight, SurfaceLateral):
        raise ValueError("surface_rate needs one linear cartesian/surface_height/surface_lateral CV")
    component = cv.cvs[0]
    if any(name in component.__dict__ for name in (
        "value", "value_and_jacobian", "centroid_value_gradient", "centroid_value_gradient_hessian",
    )):
        raise ValueError("surface_rate does not accept overridden CV methods; the mass metric must be constant")
    if component.estimator != "centroid" or component.periods != (None,):
        raise ValueError("surface_rate requires an unwrapped, nonperiodic centroid coordinate")
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
    if centers[:, 0].min() > domain[0] or centers[:, 0].max() < domain[1]:
        raise ValueError("surface umbrella centers must cover the complete reactant basin through the dividing surface")
    recrossing_surface = config.get("recrossing", {}).get("dividing_surface")
    if recrossing_surface is not None:
        raise ValueError("set the single dividing surface in [surface], not [recrossing]")
    coordinate = SurfaceCoordinate(component, system, side)
    return {
        "coordinate": coordinate,
        "cv_name": component.name,
        "reactant_basin_bohr": basin,
        "dividing_surface_bohr": float(dividing),
        "product_side": side,
        "mass_metric_au": coordinate.mass_metric,
        "rate_unit": "s^-1",
        "observable": "selected_surface_flux_side",
        "rate_scope": "single_local_exit",
        "product_indicator": "s > dividing_surface" if side == "higher" else "s < dividing_surface",
        "normalization": "per reactant in the specified basin, through the selected dividing surface",
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
    )
    np.savetxt(
        path / "surface_pmf.csv",
        np.column_stack([pmf["xi"], pmf["free_energy"], pmf["mask"]]),
        delimiter=",", comments="", header="coordinate_bohr,free_energy_hartree,masked",
    )
    # The JSON remains compact; full curve arrays live in NPZ/CSV.
    public = {k: v for k, v in pmf.items() if k not in ("xi", "free_energy", "mask")}
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
    from .rate import _recrossing_options, _parent_task, _child_task, _child_tasks, _aggregate

    config = copy.deepcopy(config)
    config["task"] = "surface_rate"
    definition = surface_definition(config, system)
    options = _recrossing_options(config)
    sampling_config = copy.deepcopy(config)
    sampling_config.setdefault("analysis", {}).update(measure="cv_density", bootstrap=0)
    sampled = run_fes(sampling_config, system, Path(output) / "umbrella", executor=executor, return_data=True)
    pmf, parent_tasks = root_call(_pmf_and_parents, sampled, config, system, str(output), options)

    def dispatch(function, tasks):
        return executor.map(function, tasks) if executor is not None else run_tasks(function, tasks, config.get("parallel", {}))

    children = []
    if parent_tasks:
        parents = dispatch(_parent_task, parent_tasks)
        tasks = root_call(_child_tasks, parents, config, system, str(output), options, pmf)
        children = dispatch(_child_task, tasks)
    result = root_call(_aggregate, children, pmf, config, str(output), options)
    if result is not None:
        result.update(
            rate_unit="s^-1",
            surface={k: v for k, v in definition.items() if k != "coordinate"},
        )
        root_call(write_json, Path(output) / "summary.json", result)
    return result
