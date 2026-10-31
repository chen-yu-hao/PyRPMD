"""Validate task contracts and normalize reaction and collision settings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .input_options import rate_pmf_options
from .input_runtime import _steps
from .input_schema import InputError, _SECTIONS, _integer, _positive
from .units import parse_quantity


def validate_task_sections(d):
    task = d["task"]
    materials = task in ("pimd", "diffusion")
    if "barostat" in d and task != "pimd":
        raise InputError("[barostat] requires task='pimd'; RPMD transport uses fixed-cell unthermostatted production")
    if not materials and any(key in d.get("output", {}) for key in ("trajectory_storage", "chunk_frames")):
        raise InputError("output.trajectory_storage/chunk_frames require task='pimd' or 'diffusion'")
    engine = d.get("sampling", {}).get("engine", "legacy")
    if engine not in ("legacy", "native"):
        raise InputError("sampling.engine must be legacy or native")
    native_pimd = engine == "native"
    if native_pimd and task != "pimd":
        raise InputError("sampling.engine='native' currently requires task='pimd'")
    if "diffusion" in d and task != "diffusion":
        raise InputError("[diffusion] requires task='diffusion'")
    if "electronic_friction" in d and task != "diffusion":
        raise InputError("[electronic_friction] requires task='diffusion'; other protocols use the Python API")
    if "site_rate" in d and task != "site_rate":
        raise InputError("[site_rate] requires task='site_rate'")
    if materials and any(k in d for k in ("umbrella", "cvs", "reaction", "collision", "outcomes", "rate", "recrossing")):
        raise InputError("Standalone pimd/diffusion tasks do not accept reaction or bias settings")
    if materials and "analysis" in d:
        raise InputError("Standalone pimd/diffusion tasks do not use [analysis]; put transport options in [diffusion]")
    if task == "site_rate" and any(k in d for k in ("reaction", "rate", "collision", "outcomes")):
        raise InputError("site_rate uses basin normalization without gas-phase reaction/rate settings")
    if "pcet" in d:
        raise InputError("[pcet] requires task='pcet'")
    if "network" in d:
        raise InputError("[network] requires task='network'")
    if "metadynamics" in d and task != "metadynamics":
        raise InputError("[metadynamics] requires task='metadynamics'")
    if "opes" in d and task != "opes":
        raise InputError("[opes] requires task='opes'")
    if task == "opes" and any(
        key in d for key in ("umbrella", "reaction", "rate", "recrossing", "collision", "outcomes")
    ):
        raise InputError("OPES uses its own probability bias; umbrella and reaction/rate settings do not apply")
    if "adaptive_umbrella" in d and task not in ("fes", "rate"):
        raise InputError("Adaptive umbrellas require task='fes' or task='rate'")
    if "replica_exchange" in d and task != "fes":
        raise InputError("[replica_exchange] requires task='fes'")
    if task in ("metadynamics", "opes"):
        analysis = d.setdefault("analysis", {})
        analysis.setdefault("method", "kde" if task == "opes" else "bias_estimate")
        analysis.setdefault("measure", "cv_density")
        if not any(key in analysis for key in ("bootstrap", "bootstrap_replicates", "bootstrap_samples")):
            analysis["bootstrap"] = 0
    analysis = d.get("analysis", {})
    if analysis.get("measure", "cv_density") not in ("cv_density", "rate_flux"):
        raise InputError("analysis.measure must be cv_density or rate_flux")
    if "flux_axis" in analysis:
        if analysis.get("measure", "cv_density") != "rate_flux":
            raise InputError("analysis.flux_axis requires measure='rate_flux'")
        if _integer(analysis["flux_axis"], "analysis.flux_axis", 0) != 0:
            raise InputError("analysis.flux_axis currently supports only zero")
    source_pmf = task == "surface_rate" and bool(d.get("surface", {}).get("free_energy"))
    expected_analysis = "source" if source_pmf else "mbar"
    if task in ("fes", "surface_rate") and analysis.get("method", expected_analysis) != expected_analysis:
        raise InputError(
            f"This FES/surface workflow requires analysis.method='{expected_analysis}'"
        )
    if task == "rate" and analysis.get("measure", "rate_flux") != "rate_flux":
        raise InputError("Standard RPMD rates require analysis.measure='rate_flux'")
    if task == "surface_rate" and analysis.get("measure", "cv_density") != "cv_density":
        raise InputError("surface_rate requires analysis.measure='cv_density'")
    if "surface" in d and task != "surface_rate":
        raise InputError("[surface] is the basin/rate definition for task='surface_rate'")
    if task == "surface_rate" and (d.get("rate") or d.get("reaction") or d.get("collision") or d.get("outcomes")):
        raise InputError("surface_rate uses [surface] and [recrossing]; gas-phase reaction/rate/collision options do not apply")
    for name in (
        "bootstrap",
        "bootstrap_samples",
        "bootstrap_replicates",
        "seed",
        "min_count",
    ):
        if name in analysis:
            _integer(
                analysis[name], "analysis." + name, 1 if name == "min_count" else 0
            )
    if "recrossing" in d.get("rate", {}):
        nested = d["rate"]["recrossing"]
        if not isinstance(nested, Mapping) or set(nested) - _SECTIONS["recrossing"]:
            raise InputError(
                "rate.recrossing must contain only documented recrossing fields"
            )
        if "recrossing" in d:
            raise InputError(
                "Use either [recrossing] or [rate.recrossing], not both"
            )
    if task == "rate":
        rate_pmf_options(d)
    return materials, native_pimd, source_pmf


def resolve_reaction(d, system):
    reaction = deepcopy(d.get("reaction", {}))
    ew = float(reaction.get("electronic_weight", 1.0))
    if not math.isfinite(ew) or ew < 0:
        raise InputError("electronic_weight must be finite and nonnegative")
    reaction["electronic_weight"] = ew
    if "rinf" in reaction:
        reaction["rinf_au"] = parse_quantity(reaction["rinf"], "length")
    elif "Rinf" in reaction:
        reaction["rinf_au"] = parse_quantity(reaction["Rinf"], "length")
    for key in ("forming_bonds", "breaking_bonds"):
        for bond in reaction.get(key, []):
            if (
                len(bond) != 2
                or any(
                    isinstance(i, bool)
                    or not isinstance(i, int)
                    or i < 0
                    or i >= system.natoms
                    for i in bond
                )
                or bond[0] == bond[1]
            ):
                raise InputError(
                    f"reaction.{key} needs distinct valid 0-based atom pairs"
                )
    return reaction


def resolve_collision(d, system, sys_spec, runtime, *, selected):
    task = d["task"]
    preset, notes, dt = runtime.preset, runtime.notes, runtime.dt
    collision = deepcopy(d.get("collision", {}))
    outcomes = deepcopy(d.get("outcomes", {}))
    if task == "drpmd":
        if "separation_au" not in sys_spec:
            f0, f1 = system.fragments
            r0 = np.average(
                system.geometry[list(f0)], axis=0, weights=system.masses[list(f0)]
            )
            r1 = np.average(
                system.geometry[list(f1)], axis=0, weights=system.masses[list(f1)]
            )
            sys_spec["separation_au"] = float(np.linalg.norm(r1 - r0))
        r0 = _positive(sys_spec["separation_au"], "system.separation")
        collision["separation_au"] = r0
        collision.setdefault("sampling", "fixed_energy" if selected else "thermal_flux")
        collision.setdefault("impact_sampling", "uniform_area")
        collision.setdefault("orientation", "isotropic")
        if collision["sampling"] not in ("thermal_flux", "maxwell", "fixed_energy"):
            raise InputError("Unknown collision sampling distribution")
        if selected and collision["sampling"] != "fixed_energy":
            raise InputError("State-selected initial_ensemble requires collision.sampling='fixed_energy' and an explicit collision.energy")
        if (
            collision["impact_sampling"] != "uniform_area"
            or collision["orientation"] != "isotropic"
        ):
            raise InputError(
                "v0.1 launch supports uniform_area and isotropic orientations"
            )
        b = collision.get("bmax", "auto")
        if b == "auto":
            if preset == "production":
                raise InputError(
                    "production requires a converged, explicit collision.bmax"
                )
            collision["bmax_au"] = r0 / 4
            collision["bmax_unvalidated"] = True
            notes.append(
                "bmax auto starts at R0/4 for pilot; rate is withheld until explicit cutoff convergence"
            )
        else:
            collision["bmax_au"] = _positive(
                parse_quantity(b, "length"), "collision.bmax"
            )
            if collision["bmax_au"] >= r0:
                raise InputError("Radial separation must exceed bmax")
        collision["trajectories"] = _integer(
            collision.get("trajectories", 4), "collision.trajectories"
        )
        collision["max_steps"] = _steps(
            collision, "max_steps", "max_time", "100 fs", dt
        )
        if collision["sampling"] == "fixed_energy":
            if "energy" not in collision:
                raise InputError("fixed_energy requires collision.energy")
            collision["energy_au"] = _positive(
                parse_quantity(collision["energy"], "energy"), "collision.energy"
            )
        outcomes["exit_distance_au"] = _positive(
            parse_quantity(outcomes.get("exit_distance", r0), "length"),
            "outcomes.exit_distance",
        )
        outcomes["persistence_steps"] = _steps(
            outcomes, "persistence_steps", "persistence", "1 fs", dt
        )
        outcomes["check_interval"] = _integer(
            outcomes.get("check_interval", 1), "outcomes.check_interval"
        )
        outcomes["bond_cutoffs_au"] = {
            k: parse_quantity(v, "length")
            for k, v in outcomes.get("bond_cutoffs", {}).items()
        }
        from .workflows.drpmd import resolve_pilot_outcomes

        outcomes = resolve_pilot_outcomes(
            system, outcomes, separation_au=r0, preset=preset
        )
        if outcomes.get("classifier_unvalidated"):
            notes.append(
                "Pilot connectivity is inferred from covalent radii; candidate outcomes are diagnostic and chemical rate/cross-section estimates are withheld"
            )
    return collision, outcomes


def validate_workflow(resolved, system, *, materials, source_pmf):
    task = resolved["task"]
    try:
        if task in ("fes", "surface_rate") and not source_pmf:
            from .workflows.fes import validate_fes_request

            validate_fes_request(resolved, system)
        if task in ("rate", "surface_rate"):
            from .workflows.rate import _recrossing_options

            _recrossing_options(resolved)
        if materials:
            from .workflows.materials import validate_materials

            for unused in ("reaction", "collision", "outcomes"):
                resolved.pop(unused, None)
            if task == "diffusion":
                diffusion = resolved.setdefault("diffusion", {})
                if "fit_range" in diffusion:
                    if not isinstance(diffusion["fit_range"], (list, tuple)) or len(diffusion["fit_range"]) != 2:
                        raise InputError("diffusion.fit_range must be [start, stop] times")
                    diffusion["fit_range"] = [None if v is None else parse_quantity(v, "time")
                                              for v in diffusion["fit_range"]]
            validate_materials(resolved, system)
        if task == "site_rate":
            from .workflows.site_rate import validate_site_rate

            for unused in ("reaction", "collision", "outcomes"):
                resolved.pop(unused, None)
            validate_site_rate(resolved, system)
        if task == "metadynamics":
            from .workflows.metadynamics import validate_metadynamics

            validate_metadynamics(resolved, system)
        if task == "opes":
            from .workflows.opes import validate_opes

            validate_opes(resolved, system)
        if "replica_exchange" in resolved:
            from .workflows.replica_exchange import exchange_options

            exchange_options(resolved, system)
        if source_pmf:
            from .workflows.surface_source import source_context
            source_context(resolved, system, verify=True)
        if "adaptive_umbrella" in resolved:
            from .workflows.adaptive_umbrella import validate_adaptive_umbrella

            validate_adaptive_umbrella(resolved, system)
    except (ValueError, TypeError, KeyError) as error:
        raise InputError(str(error)) from error
