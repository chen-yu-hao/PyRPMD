"""Scientific option contracts shared by input and direct workflow calls."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .input_schema import InputError, _SECTIONS, _integer
from .units import parse_quantity


def validate_nonadiabatic(config, *, potential=None, parallel=None, system=None):
    """Normalize the experimental state-selected NRPMD contract, also on resume."""
    potential = config.get("potential", {}) if potential is None else potential
    na = config.get("nonadiabatic")
    if na is None:
        raise InputError("diabatic_python requires [nonadiabatic]")
    if not isinstance(na, dict) or not na:
        raise InputError("nonadiabatic must be a nonempty mapping")
    if set(na) - _SECTIONS["nonadiabatic"]:
        raise InputError("Unknown nonadiabatic fields")
    if config.get("task") != "drpmd":
        raise InputError("NRPMD currently supports state-selected drpmd scattering only; electronic equilibrium PMF/recrossing rate is not implemented")
    if system is not None:
        get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
        from .geometry import validate_cell_pbc
        _, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        if pbc.any() or get("fixed_atoms", ()):
            raise InputError("NRPMD scattering requires a nonperiodic system with all atoms mobile")
    if na.get("method", "nrpmd") != "nrpmd" or na.get("initialization", "focused") != "focused":
        raise InputError("nonadiabatic requires method='nrpmd', initialization='focused'")
    nstates = _integer(na.get("nstates"), "nonadiabatic.nstates")
    initial = _integer(na.get("initial_state", 0), "nonadiabatic.initial_state", 0)
    if initial >= nstates:
        raise InputError("initial_state must be smaller than nstates")
    labels = na.get("state_labels", [str(i) for i in range(nstates)])
    if (not isinstance(labels, list) or len(labels) != nstates
            or any(not isinstance(x, str) or not x.strip() for x in labels)
            or len(set(labels)) != nstates):
        raise InputError("state_labels must be nstates distinct nonempty strings")
    tolerance = parse_quantity(na.get("asymptotic_coupling_tolerance", "1e-8 hartree"), "energy")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise InputError("asymptotic_coupling_tolerance must be finite and nonnegative")
    if potential.get("provider", potential.get("type")) != "diabatic_python":
        raise InputError("NRPMD requires potential.provider='diabatic_python'")
    if potential.get("derivative", "gradient") != "gradient":
        raise InputError("diabatic_python requires positive matrix gradients")
    if str(potential.get("device", "cpu")) != "cpu":
        raise InputError("NRPMD supports CPU matrix potentials only")
    from .parallel import execution_plan
    plan = execution_plan(config.get("parallel", {}) if parallel is None else parallel)
    if plan.backend == "mpi" or plan.mode != "serial" or plan.bead_workers != 1:
        raise InputError("NRPMD supports serial or trajectory-process execution only; MPI, GPU and bead workers are not implemented")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise InputError("NRPMD matrix propagation requires loop_backend='auto' or 'python'; the free ring can still use native kernels")
    config["nonadiabatic"] = {
        "method": "nrpmd", "nstates": nstates, "initial_state": initial,
        "state_labels": list(labels), "initialization": "focused",
        "asymptotic_coupling_tolerance": float(tolerance),
    }
    return config["nonadiabatic"]



def rate_pmf_options(config):
    """Validate PMF selection for both frontends and direct workflow calls.

    Defaults are returned without changing the caller's configuration, so an
    existing implicit-UI input retains its checkpoint identity. The explicit
    rate table and requested surface are part of the normal science digest.
    """
    rate = config.get("rate", {})
    if not isinstance(rate, Mapping):
        raise InputError("rate must be a table")
    from .input_rate_pmf import rate_pmf_spec, rate_pmf_analysis

    spec = rate_pmf_spec(config)
    source = rate.get("pmf_source")
    sampler = "umbrella"
    if spec is not None:
        sampler = spec["method"]
        if sampler == "external":
            saved = rate.get("pmf_stage", {})
            source = saved.get("source") if isinstance(saved, Mapping) else None
            if source is None:
                source = {"directory": spec["directory"],
                          "allow_non_umbrella": spec.get("allow_non_umbrella", False)}
    if source == "umbrella":
        source = None
    if source is not None:
        if isinstance(source, str):
            if source in ("metadynamics", "opes"):
                raise InputError(
                    f"rate.pmf_source='{source}' requires an explicitly supplied reweighted PMF; "
                    "MetaD/OPES bias estimates are not rate-flux PMFs"
                )
            if not source.strip():
                raise InputError("rate.pmf_source must name a FES result directory")
        elif isinstance(source, Mapping):
            allowed = {"directory", "allow_non_umbrella", "data_file", "config_file", "system_file", "summary_file", "metadata_file", "manifest_file"}
            if (set(source) - allowed or not isinstance(source.get("directory"), str)
                    or not source["directory"].strip()):
                raise InputError("rate.pmf_source accepts directory and allow_non_umbrella")
            if not isinstance(source.get("allow_non_umbrella", False), bool):
                raise InputError("rate.pmf_source.allow_non_umbrella must be boolean")
        else:
            raise InputError("rate.pmf_source must be a FES directory or a table")
    method = spec.get("estimator", "ui") if spec is not None else rate.get("pmf_method", "ui")
    if method not in (("ui", "mbar", "reweighted_histogram") if spec is not None else ("ui", "mbar")):
        raise InputError("rate.pmf_method must be 'ui' or 'mbar'")
    if "recrossing" in config and "recrossing" in rate:
        raise InputError("Use either [recrossing] or [rate.recrossing], not both")
    recrossing = config.get("recrossing", rate.get("recrossing", {}))
    if not isinstance(recrossing, Mapping):
        raise InputError("recrossing must be a table")

    def coordinate(value, name):
        if isinstance(value, str) and value == "auto":
            return None
        try:
            return parse_quantity(value, "dimensionless")
        except ValueError as error:
            raise InputError(
                f"{name} must be a finite dimensionless value or 'auto'"
            ) from error

    requested = []
    for table, name in ((rate, "rate"), (recrossing, "recrossing")):
        if "dividing_surface" in table:
            requested.append(
                (
                    name,
                    coordinate(table["dividing_surface"], name + ".dividing_surface"),
                )
            )
    if len(requested) == 2 and requested[0][1] != requested[1][1]:
        raise InputError(
            "Conflicting rate.dividing_surface and recrossing.dividing_surface"
        )
    xi_star = requested[0][1] if requested else None
    search_range = rate.get("pmf_search_range")
    if search_range is not None:
        if (
            not isinstance(search_range, (list, tuple, np.ndarray))
            or len(search_range) != 2
        ):
            raise InputError("rate.pmf_search_range must contain two coordinates")
        search_range = [
            coordinate(value, "rate.pmf_search_range") for value in search_range
        ]
        if (
            any(value is None for value in search_range)
            or not 0 <= search_range[0] < search_range[1]
        ):
            raise InputError("rate.pmf_search_range must satisfy 0 <= lower < upper")
    compare_ui = rate.get("compare_ui", method == "mbar")
    if not isinstance(compare_ui, bool):
        raise InputError("rate.compare_ui must be boolean")
    if method != "mbar" and compare_ui:
        selector = "rate.pmf.estimator" if spec is not None else "rate.pmf_method"
        raise InputError(f"rate.compare_ui requires {selector}='mbar'")
    analysis = rate_pmf_analysis(config)
    if not isinstance(analysis, Mapping):
        raise InputError("analysis must be a table")
    nboot = _integer(
        analysis.get(
            "bootstrap",
            analysis.get("bootstrap_replicates", analysis.get("bootstrap_samples", 20)),
        ),
        "analysis.bootstrap",
        0,
    )
    if nboot == 1:
        raise InputError("analysis.bootstrap must be zero or at least 2")
    bootstrap_workers = analysis.get("bootstrap_workers")
    if bootstrap_workers is not None:
        bootstrap_workers = _integer(
            bootstrap_workers, "analysis.bootstrap_workers", 1
        )
    bins = analysis.get("bins", 100)
    if isinstance(bins, (list, tuple, np.ndarray)):
        if len(bins) != 1:
            raise InputError("Rate PMF analysis.bins must describe one dimension")
        bins = bins[0]
    bins = _integer(bins, "analysis.bins", 3 if method == "mbar" else 2)
    ranges = analysis.get("ranges")
    if ranges is not None:
        if not isinstance(ranges, (list, tuple, np.ndarray)) or len(ranges) != 1:
            raise InputError(
                "Rate PMF analysis.ranges must contain one [lower, upper] pair"
            )
        pair = ranges[0]
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise InputError(
                "Rate PMF analysis.ranges must contain one [lower, upper] pair"
            )
        pair = [coordinate(value, "analysis.ranges") for value in pair]
        if any(value is None for value in pair) or not pair[0] < pair[1]:
            raise InputError("Rate PMF analysis.ranges requires finite lower < upper")
        ranges = pair
    threshold = float(analysis.get("overlap_threshold", 1e-6))
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise InputError(
            "analysis.overlap_threshold must lie strictly between zero and one"
        )
    from .analysis.rate_support import (
        DEFAULT_MAX_CENTER_DISPLACEMENT_SIGMA, DEFAULT_MIN_ENDPOINT_ESS,
        DEFAULT_MIN_WEIGHT_ESS, preflight_message, uniform_edges,
    )

    def gate(name, default):
        value = rate.get(name, default)
        if value is None or (isinstance(value, str) and value == "off"):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(f"rate.{name} must be a positive number or 'off'")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise InputError(f"rate.{name} must be a positive number or 'off'")
        return value

    gates = {
        "min_weight_ess": gate("pmf_min_weight_ess", DEFAULT_MIN_WEIGHT_ESS),
        "min_endpoint_ess": gate("pmf_min_endpoint_ess", DEFAULT_MIN_ENDPOINT_ESS),
        "max_center_displacement_sigma": gate(
            "pmf_max_window_displacement", DEFAULT_MAX_CENTER_DISPLACEMENT_SIGMA),
    }
    if method == "mbar" and ranges is not None:
        # Histogram PMFs exist only at bin centers. Reject an endpoint beyond
        # the center range before any sampling instead of clipping it later.
        edges = uniform_edges(bins, ranges)
        option = f"analysis.ranges=[{ranges[0]:g}, {ranges[1]:g}] with analysis.bins={bins}"
        points = [("rate PMF reference xi", 0.0)]
        if search_range is not None and xi_star is None:
            points += [("rate.pmf_search_range lower endpoint", search_range[0]),
                       ("rate.pmf_search_range upper endpoint", search_range[1])]
        if xi_star is not None:
            points.append(("rate.dividing_surface", xi_star))
        for name, point in points:
            message = preflight_message(name, point, edges, option=option)
            if message:
                raise InputError(message)
    return {
        "method": method,
        "sampler": sampler if source is None or spec is not None else "external",
        "source": source,
        "xi_star": xi_star,
        "search_range": search_range,
        "compare_ui": compare_ui,
        "points": _integer(rate.get("pmf_points", 401), "rate.pmf_points", 20),
        "bins": bins,
        "ranges": ranges,
        "bootstrap": nboot,
        "bootstrap_workers": bootstrap_workers,
        "min_count": _integer(analysis.get("min_count", 1), "analysis.min_count"),
        "overlap_threshold": threshold,
        **gates,
        "analysis_seed": _integer(
            analysis.get("seed", config.get("seed", 0)), "analysis.seed", 0
        ),
    }
