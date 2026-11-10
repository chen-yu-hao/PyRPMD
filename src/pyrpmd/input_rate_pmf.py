"""Checked, independent PMF stages for the gas-phase rate workflow.

The public ``rate.pmf`` table is an input request.  ``rate.pmf_stage`` is an
internal, persisted resolution of that request; users cannot supply it.  In
particular, durations are converted using the PMF time step before the raw
input is discarded.  Legacy inputs never acquire this new namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path

from .input_schema import InputError, _SECTIONS, _integer, _unknown_fields


_METHODS = ("umbrella", "metadynamics", "opes", "external")
_COMMON = ("dynamics", "sampling", "analysis")
_METHOD_TABLES = ("umbrella", "metadynamics", "opes", "adaptive_umbrella")
_FIELDS = {"method", "estimator", "directory", "allow_non_umbrella", *_COMMON, *_METHOD_TABLES}
_BOOTSTRAP = ("bootstrap", "bootstrap_replicates", "bootstrap_samples")


def normalize_rate_pmf_stages(config):
    """Resolve operation-named umbrella controls on the PMF's own clock.

    This runs before the top-level stage normalizer. Otherwise an inherited
    umbrella.sampling duration would become a recrossing-clock step count
    before the independently selected PMF timestep was available.
    """
    rate = config.get("rate", {})
    if not isinstance(rate, Mapping) or not isinstance(rate.get("pmf"), Mapping):
        return  # The generic schema/spec validator reports malformed tables.
    spec = deepcopy(dict(rate["pmf"]))
    if spec.get("method", "umbrella") != "umbrella":
        return
    umbrella = spec.get("umbrella", config.get("umbrella", {}))
    if not isinstance(umbrella, Mapping) or not any(key in umbrella for key in ("sampling", "analysis")):
        return
    if "umbrella" in spec and "umbrella" in config:
        raise InputError("Use [rate.pmf.umbrella] or legacy [umbrella], not both")
    from .input_stages import normalize_sampling_stage, _umbrella_analysis
    from .units import parse_quantity

    umbrella = deepcopy(dict(umbrella))
    if "sampling" in umbrella:
        if "sampling" in spec:
            raise InputError("Use rate.pmf.umbrella.sampling or rate.pmf.sampling, not both")
        local_dynamics = spec.get("dynamics", {})
        if not isinstance(local_dynamics, Mapping):
            raise InputError("rate.pmf.dynamics must be a table")
        shared_dynamics = config.get("dynamics", {})
        if not isinstance(shared_dynamics, Mapping):
            raise InputError("dynamics must be a table")
        dynamics = {**shared_dynamics, **local_dynamics}
        try:
            dt = float(parse_quantity(dynamics.get("timestep", "0.1 fs"), "time"))
        except (TypeError, ValueError) as error:
            raise InputError(f"rate.pmf.dynamics.timestep: {error}") from error
        if not math.isfinite(dt) or dt <= 0:
            raise InputError("rate.pmf.dynamics.timestep must be positive and finite")
        spec["sampling"] = normalize_sampling_stage(
            umbrella.pop("sampling"), dt, path="rate.pmf.umbrella.sampling",
        )
    if "analysis" in umbrella:
        if "analysis" in spec:
            raise InputError("Use rate.pmf.umbrella.analysis or rate.pmf.analysis, not both")
        normalized = {"task": "rate"}
        _umbrella_analysis(normalized, umbrella.pop("analysis"))
        local_rate = normalized.get("rate", {})
        if "pmf_method" in local_rate:
            if "estimator" in spec:
                raise InputError("Use rate.pmf.estimator or rate.pmf.umbrella.analysis.estimator, not both")
            spec["estimator"] = local_rate["pmf_method"]
        if "pmf_points" in local_rate:
            if "pmf_points" in rate:
                raise InputError("Use rate.pmf_points or rate.pmf.umbrella.analysis.integration_points, not both")
            config["rate"]["pmf_points"] = local_rate["pmf_points"]
        spec["analysis"] = normalized["analysis"]
    spec["umbrella"] = umbrella
    config.pop("umbrella", None)
    config["rate"]["pmf"] = spec
    return True


def rate_pmf_spec(config):
    """Return a checked public request, or None for unchanged legacy input."""
    rate = config.get("rate", {})
    if not isinstance(rate, Mapping):
        raise InputError("rate must be a table")
    if "pmf" not in rate:
        return None
    if config.get("task", "rate") != "rate":
        raise InputError("[rate.pmf] requires task='rate'")
    raw = rate["pmf"]
    if not isinstance(raw, Mapping):
        raise InputError("rate.pmf must be a table")
    unknown = set(raw) - _FIELDS
    if unknown:
        raise _unknown_fields(unknown, _FIELDS, "rate.pmf")
    if "pmf_method" in rate or "pmf_source" in rate:
        raise InputError("Use [rate.pmf] or legacy rate.pmf_method/rate.pmf_source, not both")
    spec = deepcopy(dict(raw))
    method = spec.get("method", "umbrella")
    if not isinstance(method, str) or method not in _METHODS:
        raise InputError("rate.pmf.method must be umbrella, metadynamics, opes or external")
    spec["method"] = method
    for name in (*_COMMON, *_METHOD_TABLES):
        if name not in spec:
            continue
        table = spec[name]
        if not isinstance(table, Mapping):
            raise InputError(f"rate.pmf.{name} must be a table")
        unknown = set(table) - _SECTIONS[name]
        if unknown:
            raise _unknown_fields(unknown, _SECTIONS[name], "rate.pmf." + name)
    sampling = spec.get("sampling", {})
    if "thermostat_params" in sampling and not isinstance(sampling["thermostat_params"], Mapping):
        raise InputError("rate.pmf.sampling.thermostat_params must be a table")
    if sampling.get("engine", "legacy") != "legacy":
        raise InputError("rate PMF sampling does not support sampling.engine='native'; use dynamics.loop_backend")
    for name in _METHOD_TABLES:
        if name in spec and name in config:
            raise InputError(f"Use [rate.pmf.{name}] or legacy [{name}], not both")
        if name in spec or name in config:
            accepted = method == name or (method == "umbrella" and name == "adaptive_umbrella")
            if not accepted:
                raise InputError(f"rate.pmf.method='{method}' does not accept {name} parameters")
    if method != "external":
        selected_cvs = spec.get(method, config.get(method, {})).get("cvs", ["xi"])
        if isinstance(selected_cvs, str):
            selected_cvs = [selected_cvs]
        if not isinstance(selected_cvs, (list, tuple)) or list(selected_cvs) != ["xi"]:
            raise InputError("Rate PMF sampling uses the prepared RPMDrate coordinate; cvs must be ['xi']")
    if method == "external":
        if "estimator" in spec:
            raise InputError("rate.pmf.method='external' uses the source estimator; do not specify estimator")
        if any(name in spec for name in _COMMON):
            raise InputError("External rate PMFs do not accept local dynamics, sampling or analysis parameters")
        if not isinstance(spec.get("directory"), str) or not spec["directory"].strip():
            raise InputError("rate.pmf.method='external' requires a nonempty directory")
        if not isinstance(spec.get("allow_non_umbrella", False), bool):
            raise InputError("rate.pmf.allow_non_umbrella must be boolean")
    else:
        if "directory" in spec or "allow_non_umbrella" in spec:
            raise InputError("rate.pmf.directory/allow_non_umbrella require method='external'")
        expected = ("ui", "mbar") if method == "umbrella" else ("reweighted_histogram",)
        spec.setdefault("estimator", expected[0])
        if not isinstance(spec["estimator"], str) or spec["estimator"] not in expected:
            raise InputError(f"rate.pmf.estimator for {method} must be {' or '.join(expected)}")
    if method in ("metadynamics", "opes"):
        options = spec.get(method, config.get(method, {}))
        for name, minimum in (("freeze_after_steps", 1), ("freeze_equilibration_steps", 0)):
            if name not in options:
                raise InputError(f"rate.pmf.{method}.{name} is required for frozen-bias reweighting")
            _integer(options[name], f"rate.pmf.{method}.{name}", minimum)
        if options.get("shared_walkers", False) or options.get("multiple_walkers", False):
            raise InputError("Frozen-bias rate PMFs currently require independent bias histories")
        if options.get("walls"):
            raise InputError("Frozen-bias rate PMFs do not support static walls")
    return spec


def normalize_rate_pmf_input(config, base=None):
    """Validate new input and materialize external routing after schema checks."""
    spec = rate_pmf_spec(config)
    if spec is None:
        return
    config["rate"]["pmf"] = spec
    if spec["method"] == "external":
        from .workflows.rate_source import source_paths

        try:
            paths = source_paths({"directory": spec["directory"],
                                  "allow_non_umbrella": spec.get("allow_non_umbrella", False)}, base)
        except (ValueError, TypeError) as error:
            raise InputError(str(error)) from error
        spec["directory"] = paths["directory"]
        config["rate"]["pmf_stage"] = {
            "schema_version": 1, "sampler": "external", "source": paths,
        }


def _overlay(base, local, aliases=()):
    result = deepcopy(dict(base))
    for names in aliases:
        if any(name in local for name in names):
            for name in names:
                result.pop(name, None)
    result.update(deepcopy(dict(local)))
    return result


def rate_pmf_analysis(config):
    """Effective rate-PMF analysis options, without modifying shared defaults."""
    base = config.get("analysis", {})
    if not isinstance(base, Mapping):
        raise InputError("analysis must be a table")
    spec = rate_pmf_spec(config)
    if spec is None:
        # Legacy umbrella inputs used to reach this check in the generic FES
        # analyzer, after sampling. Keep typo rejection when that analyzer is
        # no longer part of the rate path; neither value selects the rate estimator.
        if (config.get("rate", {}).get("pmf_source") is None
                and base.get("method", "mbar") not in ("mbar", "umbrella_integration")):
            raise InputError(
                "Omit the legacy analysis.method setting; select the rate estimator "
                "with rate.pmf.estimator or rate.pmf_method"
            )
        return base
    analysis = _overlay(base, spec.get("analysis", {}), (_BOOTSTRAP,))
    if analysis.get("measure", "rate_flux") != "rate_flux":
        raise InputError("Standard RPMD rate PMFs require analysis.measure='rate_flux'")
    if "flux_axis" in analysis and _integer(analysis["flux_axis"], "rate.pmf.analysis.flux_axis", 0) != 0:
        raise InputError("Rate PMF analysis.flux_axis must be zero")
    if spec["method"] in ("metadynamics", "opes"):
        for name in _BOOTSTRAP:
            if name in analysis and _integer(analysis[name], "rate.pmf.analysis." + name, 0) != 0:
                raise InputError("Frozen-bias rate PMFs currently require analysis.bootstrap=0")
        analysis.setdefault("bootstrap", 0)
    return analysis


def resolve_rate_pmf_stage(raw, resolved, system):
    """Persist an independently normalized PMF stage from RAW time budgets."""
    spec = rate_pmf_spec(raw)
    if spec is None or spec["method"] == "external":
        return
    from .input_runtime import resolve_dynamics, resolve_sampling, resolve_output
    from .input_tasks import validate_input_preflight

    method = spec["method"]
    local_dynamics = spec.get("dynamics", {})
    local_sampling = spec.get("sampling", {})
    dynamics = _overlay(raw.get("dynamics", {}), local_dynamics)
    sampling = _overlay(raw.get("sampling", {}), local_sampling, (
        ("equilibration", "equilibration_steps"),
        ("production", "production_steps"),
        ("interval", "save_interval", "interval_steps", "sample_interval"),
    ))
    # Inherited splitting names and bath parameters belong to their original
    # operator. A local operator selection gets matching defaults instead.
    if "integrator" in local_dynamics or "thermostat" in local_sampling:
        if "integrator" not in local_sampling:
            sampling.pop("integrator", None)
    if "thermostat" in local_sampling:
        for key in ("thermostat_params", "gamma_centroid", "andersen_schedule"):
            if key not in local_sampling:
                sampling.pop(key, None)
    if "timestep" in local_dynamics and "timestep" not in local_sampling:
        sampling.pop("timestep", None)
    analysis = deepcopy(rate_pmf_analysis(raw))
    if method == "umbrella":
        # Sampling does not select a FES estimator. Accept the historical
        # explicit MBAR setting without letting it override the rate estimator.
        legacy_method = analysis.pop("method", "mbar")
        if legacy_method != "mbar":
            raise InputError(
                "Select the rate estimator with rate.pmf.estimator, not "
                "rate.pmf.analysis.method; omit the legacy analysis.method setting"
            )
    else:
        if analysis.get("method", "reweighted_histogram") != "reweighted_histogram":
            raise InputError("rate.pmf.analysis.method must be 'reweighted_histogram'; select the rate estimator with rate.pmf.estimator")
        analysis["method"] = "reweighted_histogram"
    analysis["measure"] = "rate_flux"
    stage = {
        "task": "fes", "fes": {"method": method}, "schema_version": raw.get("schema_version", 1),
        "preset": resolved["preset"], "temperature": resolved["temperature_K"],
        "beads": resolved["beads"], "seed": resolved["seed"],
        "dynamics": dynamics, "sampling": sampling, "analysis": analysis,
        "output": deepcopy(raw.get("output", {})),
    }
    for name in (method, "adaptive_umbrella"):
        if name in spec or name in raw:
            stage[name] = deepcopy(spec.get(name, raw.get(name)))
    if "native" in resolved:
        stage["native"] = resolved["native"]
    if "native" in local_dynamics:
        stage["native"] = local_dynamics["native"]
    validate_input_preflight(stage)
    runtime = resolve_dynamics(stage, selected=False, initial_ensemble=None)
    stage["sampling"] = resolve_sampling(
        stage, runtime, selected=False, materials=False, native_pimd=False,
    )
    stage.update(beta=resolved["beta"], temperature_K=resolved["temperature_K"],
                 dt_au=runtime.dt, _warnings=runtime.notes,
                 hamiltonian_convention=resolved["hamiltonian_convention"])
    stage["output"] = resolve_output(stage, Path(resolved["_base_directory"]), runtime.dt, selected=False)
    if "adaptive_umbrella" in stage:
        from .workflows.adaptive_umbrella import adaptive_options

        try:
            adaptive_options(stage)
        except (ValueError, TypeError) as error:
            raise InputError(str(error)) from error
    if method in ("metadynamics", "opes"):
        options = stage[method]
        pace = _integer(options.get("pace", 100), f"rate.pmf.{method}.pace")
        if options["freeze_after_steps"] % pace:
            raise InputError(f"rate.pmf.{method}.freeze_after_steps must be a multiple of pace")
        last_discarded = options["freeze_after_steps"] + options["freeze_equilibration_steps"]
        production = stage["sampling"]["production_steps"]
        stride = stage["sampling"]["interval_steps"]
        if production // stride - last_discarded // stride < 2:
            raise InputError("Frozen-bias rate PMF production must include at least two saved samples after freeze_after_steps + freeze_equilibration_steps")
    resolved["rate"]["pmf_stage"] = {"schema_version": 1, "sampler": method, "config": stage}


def rate_pmf_stage(config, system):
    """Get the prepared FES stage, or None for legacy/external PMF routes.

    Call after rate preparation so the stage inherits the authoritative
    reaction coordinate and system. Scientific stage settings are taken from
    its persisted resolution, while execution context follows the active run.
    """
    spec = rate_pmf_spec(config)
    if spec is None or spec["method"] == "external":
        return None
    saved = config["rate"].get("pmf_stage")
    if (not isinstance(saved, Mapping) or saved.get("schema_version") != 1
            or saved.get("sampler") != spec["method"] or not isinstance(saved.get("config"), Mapping)):
        raise InputError("rate.pmf requires resolved stage settings; load it through JobSpec or pyrpmd.run")
    stage = deepcopy(saved["config"])
    for name in ("beta", "temperature_K", "beads", "seed", "potential", "system", "reaction", "cvs",
                 "parallel", "_base_directory", "_input_path", "resume", "_resume", "hamiltonian_convention"):
        if name in config:
            stage[name] = deepcopy(config[name])
    stage["system"] = system.to_dict()
    if "directory" in config.get("output", {}):
        stage["output"]["directory"] = config["output"]["directory"]
    for key in ("loop_backend", "block_steps"):
        if key in spec.get("dynamics", {}):
            stage["dynamics"][key] = spec["dynamics"][key]
        elif key in config.get("dynamics", {}):
            stage["dynamics"][key] = config["dynamics"][key]
    if spec["method"] == "umbrella":
        umbrella = stage.setdefault("umbrella", {})
        selectors = ("centers", "centers_au", "grid", "start", "regions", "windows")
        if stage.get("schema_version", 1) >= 2:
            selectors += ("stop", "count")
        if not any(key in umbrella for key in selectors):
            umbrella.update(start=-0.05, stop=1.05, count=23)
        if not any(key in umbrella for key in ("stiffness", "stiffness_au", "kappa", "force_constant", "force_matrix")):
            umbrella["kappa"] = 500.0 / stage["beta"]
        umbrella["cvs"] = ["xi"]
    else:
        stage[spec["method"]]["cvs"] = ["xi"]
    return stage
