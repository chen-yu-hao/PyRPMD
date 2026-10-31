"""TOML/Python frontends with one checked, unit-resolved configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tomllib
from collections.abc import Mapping
import numpy as np

from .constants import KB_AU
from .units import parse_quantity, time_steps
from .system import System, system_from_config


class InputError(ValueError):
    """Actionable input error before any production simulation starts."""


def _unknown_fields(unknown, allowed, section):
    """Describe a misspelled field without silently accepting a different input."""
    names = sorted(str(key) for key in unknown)
    message = f"Unknown {section} fields: {', '.join(names)}"
    suggestions = []
    for name in names:
        matches = get_close_matches(name, sorted(allowed), n=1, cutoff=0.7)
        if matches:
            suggestions.append(f"{name!r} → {matches[0]!r}")
    if suggestions:
        message += ". Did you mean " + "; ".join(suggestions) + "?"
    return InputError(message)


_ROOT = set(
    "schema_version task preset temperature beads seed system potential dynamics sampling reaction collision outcomes cvs umbrella analysis parallel output native description rate recrossing network metadynamics opes adaptive_umbrella pcet diffusion site_rate surface nonadiabatic replica_exchange barostat electronic_friction initial_ensemble".split()
)
_SECTIONS = {
    "electronic_friction": set("method provider gamma tensor factory params".split()),
    "initial_ensemble": set("method beta fragments".split()),
    "nonadiabatic": set("method nstates initial_state state_labels initialization asymptotic_coupling_tolerance".split()),
    "system": set(
        "geometry transition_state reactants separation geometry_unit isotopes masses masses_amu symbols fragments charge multiplicity cell cell_unit pbc fixed_atoms".split()
    ),
    "dynamics": set("timestep integrator native loop_backend block_steps".split()),
    "sampling": set(
        "engine integrator thermostat timestep equilibration production save_interval interval interval_steps sample_interval equilibration_steps production_steps andersen_interval thermostat_params gamma_centroid replicas parents".split()
    ),
    "potential": set(
        "provider type name library manifest factory module function initializer initializer_params with_dimensions model weights device dtype units derivative params abi_version periodic virial".split()
    ),
    "parallel": set(
        "backend mode trajectory_groups trajectory_workers workers bead_workers pes_ranks threads_per_rank devices trajectories_per_gpu coordinator_ranks scheduling reuse_potentials".split()
    ),
    "collision": set(
        "sampling impact_sampling bmax trajectories max_time max_steps energy orientation".split()
    ),
    "outcomes": set(
        "factory params channels exit_distance persistence persistence_steps check_interval bond_cutoffs".split()
    ),
    "output": set(
        "directory checkpoint_interval checkpoint_steps trajectory_stride trajectory_storage chunk_frames diagnostics store_configurations".split()
    ),
    "reaction": set(
        "fragments forming_bonds breaking_bonds electronic_weight rinf Rinf equivalent_paths symmetry_factor preparation entry_side".split()
    ),
    "analysis": set(
        "method measure flux_axis bins bootstrap_samples bootstrap_replicates bootstrap seed overlap_threshold ranges min_count max_grid_points".split()
    ),
    "umbrella": set(
        "cvs grid centers centers_au kappa stiffness stiffness_au start stop count".split()
    ),
    "rate": set(
        "recrossing pmf_points pmf_method pmf_search_range dividing_surface compare_ui".split()
    ),
    "recrossing": set(
        "parent_replicas parent_samples equilibration equilibration_steps parent_interval parent_interval_steps child_pairs child_pairs_per_parent child_steps child_time dividing_surface plateau_fraction plateau_tolerance bootstrap constraint_tolerance child_batch_pairs dynamics friction_scale".split()
    ),
    "surface": set("reactant_basin dividing_surface product_side max_relative_standard_error normalization free_energy".split()),
    "metadynamics": set(
        "cvs mode sigma sigma_au height height_au pace bias_factor walls walls_during_equilibration grid".split()
    ),
    "opes": set(
        "cvs sigma sigma_au barrier barrier_au bias_factor pace freeze_after_steps freeze_equilibration_steps".split()
    ),
    "replica_exchange": set("enabled interval_steps pairs".split()),
    "barostat": set("method pressure time_constant friction mode tau gamma compressibility cell_mask reference_cell reference_cell_unit stress".split()),
    "adaptive_umbrella": set(
        "enabled pilot_steps pilot_equilibration_steps max_rounds max_windows max_pilot_steps target_overlap min_effective_samples min_spacing coverage_bins coverage_ranges min_bin_effective_samples target_coverage".split()
    ),
    "diffusion": set(
        "atom_indices axes reference_atoms fit_range max_lag bootstrap seed block_length".split()
    ),
    "site_rate": set("cv basin dividing_surface direction".split()),
}


def _integer(v, name, minimum=1):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")
    return int(v)


def _positive(v, name):
    if not math.isfinite(v) or v <= 0:
        raise InputError(f"{name} must be positive and finite")
    return float(v)


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
    method = rate.get("pmf_method", "ui")
    if method not in ("ui", "mbar"):
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
    if method == "ui" and compare_ui:
        raise InputError("rate.compare_ui requires rate.pmf_method='mbar'")
    analysis = config.get("analysis", {})
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
    return {
        "method": method,
        "xi_star": xi_star,
        "search_range": search_range,
        "compare_ui": compare_ui,
        "points": _integer(rate.get("pmf_points", 401), "rate.pmf_points", 20),
        "bins": bins,
        "ranges": ranges,
        "bootstrap": nboot,
        "min_count": _integer(analysis.get("min_count", 1), "analysis.min_count"),
        "overlap_threshold": threshold,
        "analysis_seed": _integer(
            analysis.get("seed", config.get("seed", 0)), "analysis.seed", 0
        ),
    }


def _path(base, value):
    return (
        str((base / Path(value)).resolve())
        if not Path(value).is_absolute()
        else str(Path(value).resolve())
    )


def _plugin_paths(value, base):
    """Resolve file-based plugins without changing normal module:object imports."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("factory", "module", "function", "initializer") and isinstance(
                item, str
            ):
                module, separator, attribute = item.partition(":")
                if module.endswith((".py", ".so")):
                    value[key] = _path(base, module) + (
                        separator + attribute if separator else ""
                    )
            else:
                _plugin_paths(item, base)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _plugin_paths(item, base)


def _steps(table, key, duration_key, default, dt, *, allow_zero=False):
    # Do not evaluate a duration which the caller has replaced with exact steps.
    if key in table:
        return _integer(table[key], key, 0 if allow_zero else 1)
    return time_steps(table.get(duration_key, default), dt, allow_zero=allow_zero)


def _initial_ensemble_spec(section, base):
    """Read references relative to the input and retain their canonical data."""
    from .initial_states import normalize_initial_ensemble

    section = deepcopy(section)
    fragments = section.get("fragments", ())
    if isinstance(fragments, (list, tuple)):
        for fragment in fragments:
            if isinstance(fragment, Mapping):
                for key in ("hessian_file", "modes_file"):
                    if key in fragment:
                        if not isinstance(fragment[key], (str, Path)) or not str(fragment[key]):
                            raise InputError(f"initial_ensemble {key} must be a file path")
                        fragment[key] = _path(base, fragment[key])
    try:
        return normalize_initial_ensemble(section)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise InputError(str(error)) from error


@dataclass
class JobSpec:
    data: dict
    base_directory: Path = Path(".")
    input_path: str | None = None

    @classmethod
    def from_mapping(cls, mapping, *, base_directory=None):
        if not isinstance(mapping, Mapping):
            raise InputError("build() must return JobSpec or a mapping")
        return cls(deepcopy(dict(mapping)), Path(base_directory or ".").resolve())

    @classmethod
    def from_file(cls, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise InputError(f"Input file not found: {path}")
        if path.suffix.lower() in (".inp", ".toml"):
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as e:
                raise InputError(f"{path}: {e}") from e
            result = cls.from_mapping(data, base_directory=path.parent)
        elif path.suffix.lower() == ".py":
            # Python inputs are ordinary trusted code, not a sandbox.
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            name = (
                "_pyrpmd_input_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
            )
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except SyntaxError as error:
                raise InputError(
                    f"{error.filename or path}:{error.lineno}:{error.offset}: {error.msg}"
                ) from error
            if not callable(getattr(module, "build", None)):
                raise InputError(
                    "Python input must define build() returning JobSpec or dict"
                )
            result = module.build()
            if isinstance(result, Mapping):
                result = cls.from_mapping(result, base_directory=path.parent)
            if not isinstance(result, cls):
                raise InputError("build() must return JobSpec or dict")
            result.base_directory = path.parent
        else:
            raise InputError("Input must be .inp/.toml or .py")
        result.input_path = str(path)
        return result

    def with_overrides(self, **overrides):
        d = deepcopy(self.data)

        def table(name):
            value = d.setdefault(name, {})
            if not isinstance(value, Mapping):
                raise InputError(f"{name} must be a table")
            return value

        for key, value in overrides.items():
            if value is None:
                continue
            if d.get("task") == "pcet" and key not in ("temperature", "output"):
                raise InputError(
                    f"PCET supports only temperature/output overrides; {key} is not applicable to this serial model"
                )
            if key == "pes":
                table("potential")["library"] = str(Path(value).resolve())
                if d["potential"].get("provider") not in ("rpmdrate", "f2py"):
                    d["potential"]["provider"] = "cabi"
            elif key == "pes_manifest":
                table("potential")["manifest"] = str(Path(value).resolve())
            elif key == "output":
                table("output")["directory"] = str(Path(value).resolve())
            elif key == "workers":
                p = table("parallel")
                if "trajectory_groups" in p and p["trajectory_groups"] != value:
                    p.pop("trajectory_groups")
                p["trajectory_workers"] = value
                if p.get("backend", "serial") != "mpi":
                    p["backend"] = "process" if value > 1 else "serial"
            else:
                d[key] = value
        return JobSpec(d, self.base_directory, self.input_path)

    def resolve(self):
        # Network and PCET have dedicated early-return resolvers; do not let
        # those routes silently ignore native PIMD-only ensemble controls.
        raw_sampling = self.data.get("sampling", {})
        if not isinstance(raw_sampling, Mapping):
            raise InputError("sampling must be a table")
        if self.data.get("task") != "pimd" and (
            "barostat" in self.data or raw_sampling.get("engine") == "native"
        ):
            raise InputError("native sampling/barostat controls currently require task='pimd'")
        if "initial_ensemble" in self.data and self.data.get("task") != "drpmd":
            raise InputError("initial_ensemble requires task='drpmd'")
        if self.data.get("task") == "pcet":
            from .pcet_input import resolve_pcet_job

            return resolve_pcet_job(
                self.data, base_directory=self.base_directory, input_path=self.input_path
            )
        if self.data.get("task") == "network":
            from .reactions import resolve_network_job

            try:
                resolved, system = resolve_network_job(
                    self.data, base_directory=self.base_directory, input_path=self.input_path
                )
            except (ValueError, TypeError, KeyError) as error:
                raise InputError(str(error)) from error
            if any(system.pbc) or system.fixed_atoms:
                raise InputError(
                    "Gas-phase network normalization does not apply to periodic or fixed-atom systems; "
                    "use fes or surface_rate for a slab"
                )
            return resolved, system
        d = deepcopy(self.data)
        unknown = set(d) - _ROOT
        if unknown:
            raise _unknown_fields(unknown, _ROOT, "top-level")
        for section, allowed in _SECTIONS.items():
            if section in d:
                if section == "system" and isinstance(d[section], System):
                    continue
                if not isinstance(d[section], Mapping):
                    raise InputError(f"{section} must be a table")
                unknown = set(d[section]) - allowed
                if unknown:
                    raise _unknown_fields(unknown, allowed, section)
        # These nested tables are consumed before provider construction. Reject
        # malformed values here, rather than leaking AttributeError or creating
        # an output directory before a later workflow fails.
        for section, fields in (
            ("potential", ("params", "initializer_params")),
            ("sampling", ("thermostat_params",)),
            ("outcomes", ("params", "bond_cutoffs")),
        ):
            table = d.get(section, {})
            for field in fields:
                if (section == "potential" and field == "initializer_params"
                        and table.get(field) is None):
                    # Python RPMDrate inputs use None for the default initializer.
                    continue
                if field in table and not isinstance(table[field], Mapping):
                    raise InputError(f"{section}.{field} must be a table")
        for table, label in ((d, "native"), (d.get("dynamics", {}), "dynamics.native")):
            if "native" in table:
                value = table["native"]
                if isinstance(value, (bool, np.bool_)):
                    table["native"] = bool(value)
                elif value is not None and not (isinstance(value, str) and value == "required"):
                    raise InputError(f"{label} must be a boolean, None or 'required'")
        if _integer(d.get("schema_version", 1), "schema_version") != 1:
            raise InputError("Supported schema_version is 1")
        task = d.get("task")
        if task not in ("rate", "drpmd", "fes", "metadynamics", "opes", "pimd", "diffusion", "site_rate", "surface_rate"):
            raise InputError("task must be rate, drpmd, fes, metadynamics, opes, network, pcet, pimd, diffusion, site_rate or surface_rate")
        selected = "initial_ensemble" in d
        initial_ensemble = None
        base = Path(self.base_directory).resolve()
        if selected:
            if "temperature" in d:
                raise InputError("State-selected initial_ensemble does not accept temperature; specify initial_ensemble.beta for NE-RPMD")
            if "nonadiabatic" in d or d.get("potential", {}).get(
                "provider", d.get("potential", {}).get("type")
            ) == "diabatic_python":
                raise InputError("initial_ensemble currently requires a single-surface potential; combining it with NRPMD is not implemented")
            initial_ensemble = _initial_ensemble_spec(d["initial_ensemble"], base)
            d["initial_ensemble"] = initial_ensemble
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
        preset = d.get("preset", "pilot")
        if preset not in ("pilot", "production"):
            raise InputError("preset must be pilot or production")
        notes = []
        if preset == "production":
            required = ("beads", "dynamics") if selected else ("temperature", "beads", "dynamics", "sampling")
            for k in required:
                if k not in d:
                    raise InputError(f"production requires explicit {k}")
        if selected:
            # beta controls the NE-RPMD reference rings. At P=1 it is only a
            # required positive integrator parameter and does not thermalize QCT.
            beta = initial_ensemble.get("beta", 1.0)
            temperature = _positive(1 / KB_AU / beta, "auxiliary inverse-beta scale")
            notes.append("State-selected initial ensemble: temperature_K is an internal inverse-beta compatibility scale, not a thermal temperature")
            if initial_ensemble["method"] == "qct" and "beta" not in initial_ensemble:
                notes.append("QCT P=1 uses beta=1 hartree^-1 as an unused integrator compatibility parameter")
        else:
            if "temperature" not in d:
                notes.append("Pilot default temperature: 300 K")
            temperature = _positive(
                parse_quantity(d.get("temperature", "300 K"), "temperature"), "temperature"
            )
            beta = 1 / (KB_AU * temperature)
        beads = d.get("beads", 1 if selected and initial_ensemble["method"] == "qct" else 32)
        if beads == "auto":
            if preset == "production":
                raise InputError(
                    "Resolve auto beads with convergence checks before production"
                )
            beads = 32
            notes.append(
                "Auto bead pilot uses P=32; bead convergence has not been established"
            )
        beads = _integer(beads, "beads")
        if selected and initial_ensemble["method"] == "qct" and beads != 1:
            raise InputError("qct initial ensembles require beads=1")
        if selected and initial_ensemble["method"] == "ne_rpmd" and beads < 2:
            raise InputError("ne_rpmd initial ensembles require beads>=2; use qct for a classical quantum-number ensemble")
        seed = _integer(d.get("seed", 0), "seed", 0)
        dyn = dict(d.get("dynamics", {}))
        if dyn.get("loop_backend", "auto") not in ("auto", "native", "python"):
            raise InputError("dynamics.loop_backend must be auto/native/python")
        block_steps = _integer(dyn.get("block_steps", 256), "dynamics.block_steps")
        if block_steps > 65536:
            raise InputError("dynamics.block_steps must be <= 65536")
        dt = _positive(
            parse_quantity(dyn.get("timestep", "0.1 fs"), "time"), "dynamics.timestep"
        )
        from .dynamics import normalize_integrator

        try:
            integrator = normalize_integrator(dyn.get("integrator", "cayley-bcb"))
        except ValueError as error:
            raise InputError(str(error)) from error
        # Preserve the identity of legacy inputs with an implicit default.
        if "integrator" in dyn:
            d["dynamics"]["integrator"] = integrator
        _plugin_paths(d, base)
        sys_spec = deepcopy(d.get("system", {}))
        if isinstance(sys_spec, System):
            system = sys_spec
            sys_spec = system.to_dict()
        else:
            for k in ("geometry", "transition_state"):
                if isinstance(sys_spec.get(k), str):
                    sys_spec[k] = _path(base, sys_spec[k])
            if "reactants" in sys_spec:
                sys_spec["reactants"] = [_path(base, x) for x in sys_spec["reactants"]]
            if "separation" in sys_spec:
                sys_spec["separation_au"] = parse_quantity(
                    sys_spec["separation"], "length"
                )
            if "reaction" in d and "fragments" in d["reaction"]:
                sys_spec["fragments"] = d["reaction"]["fragments"]
            try:
                system = system_from_config(sys_spec)
            except (ValueError, KeyError, OSError) as e:
                raise InputError(str(e)) from e
        if task == "drpmd" and len(system.fragments) != 2:
            raise InputError("dRPMD requires two disjoint reactant fragments")
        if task in ("rate", "drpmd") and (any(system.pbc) or system.fixed_atoms):
            raise InputError(
                "Gas-phase rate/drpmd normalization does not apply to periodic or fixed-atom systems; "
                "use fes, surface_rate or site_rate for a fixed cell"
            )
        if selected:
            from .initial_states import prepare_initial_ensemble

            try:
                prepare_initial_ensemble(initial_ensemble, system, beads, beta)
            except (ValueError, TypeError, KeyError) as error:
                raise InputError(str(error)) from error
        if task == "surface_rate":
            from .workflows.surface import surface_definition

            try:
                if source_pmf:
                    from .workflows.surface_source import source_paths
                    d["surface"]["free_energy"] = source_paths(d["surface"]["free_energy"], base)
                surface_definition(d, system)
            except (ValueError, KeyError, TypeError) as e:
                raise InputError(str(e)) from e
        potential = deepcopy(d.get("potential", {}))
        if not potential:
            raise InputError("Specify [potential] or --pes; no PES is guessed")
        for k in ("library", "manifest", "weights"):
            if k in potential:
                potential[k] = _path(base, potential[k])
        if "model" in potential and potential.get(
            "provider", potential.get("type")
        ) not in ("builtin", "fairchem", None):
            potential["model"] = _path(base, potential["model"])
        elif potential.get("provider", potential.get("type")) == "fairchem" and "model" in potential:
            # fairchem accepts registered names as well as local checkpoints.
            # Preserve names; resolve explicit paths relative to the input.
            model = potential["model"]
            if isinstance(model, str) and (model.startswith((".", "~", "/"))
                    or Path(model).suffix.lower() in (".pt", ".pth", ".ckpt", ".bin")
                    or (base / model).is_file()):
                potential["model"] = _path(base, model)
        # Model-specific path-valued parameters can be declared relative to input.
        for k in ("model", "weights", "reference"):
            if isinstance(potential.get("params", {}).get(k), str):
                potential["params"][k] = _path(base, potential["params"][k])
        from .potentials import PotentialError, validate_periodic_spec

        try:
            validate_periodic_spec(potential, system)
        except (ValueError, PotentialError) as error:
            raise InputError(str(error)) from error
        sampling = deepcopy(d.get("sampling", {}))
        if selected:
            if set(sampling) - {"parents"}:
                raise InputError("State-selected initial_ensemble allows only sampling.parents; thermal equilibration and thermostat settings do not apply")
        else:
            sampling_dt = _positive(
                parse_quantity(sampling.get("timestep", dt), "time"), "sampling.timestep"
            )
            sampling["dt_au"] = sampling_dt
            default_thermostat = "pile-l" if materials and integrator != "velocity-verlet" else "andersen"
            sampling["thermostat"] = sampling.get("thermostat", default_thermostat)
            if not isinstance(sampling["thermostat"], str):
                raise InputError("sampling.thermostat must be a name such as 'andersen', 'pile-l' or 'nose-hoover-chain'")
            sampler = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(integrator)
            expected_sampler = sampler if native_pimd or sampling["thermostat"] == "pile-l" else integrator + "-" + sampling["thermostat"]
            sampling["integrator"] = sampling.get("integrator", expected_sampler)
            if native_pimd:
                from .native_ensemble import _THERMOSTATS
                if sampling["thermostat"] not in _THERMOSTATS or sampling["thermostat"] in ("none", "trpmd"):
                    raise InputError("native PIMD requires an equilibrium sampling thermostat")
                if sampler is None or sampling["integrator"] != sampler:
                    raise InputError("Native PIMD requires matching BCOCB/BAOAB sampling")
                if "gamma_centroid" in sampling:
                    if sampling["thermostat"] != "pile-l":
                        raise InputError("sampling.gamma_centroid requires PILE-L")
                    sampling["gamma_centroid"] = _positive(parse_quantity(sampling["gamma_centroid"], "frequency"), "gamma_centroid")
            elif task == "metadynamics" and sampling["thermostat"] == "nose-hoover-chain":
                from .thermostats import nhc_options
                try:
                    nhc_options(sampling)
                except (ValueError, TypeError) as error:
                    raise InputError(str(error)) from error
                if sampling["integrator"] != expected_sampler:
                    raise InputError("NHC sampling.integrator must match dynamics.integrator + '-nose-hoover-chain'")
                if "gamma_centroid" in sampling:
                    raise InputError("sampling.gamma_centroid requires PILE-L sampling")
            elif (materials or task in ("fes", "metadynamics")) and sampling["thermostat"] == "pile-l":
                if sampler is None or sampling["integrator"] != sampler:
                    raise InputError("PILE-L requires BCOCB with Cayley or BAOAB with normal-mode dynamics")
                sampling["gamma_centroid"] = _positive(
                    parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency"),
                    "sampling.gamma_centroid",
                )
                if dyn.get("loop_backend", "auto") == "native":
                    raise InputError("PILE-L sampling requires loop_backend='auto' or 'python'; its deterministic substeps may still be native")
            elif sampling["integrator"] != integrator + "-andersen" or sampling["thermostat"] != "andersen":
                raise InputError(
                    "Workflow sampling requires sampling.integrator='"
                    + integrator + "-andersen' and thermostat='andersen'; "
                    "sampling and production use the same selected ring-polymer integrator"
                )
            elif "gamma_centroid" in sampling:
                raise InputError("sampling.gamma_centroid requires PILE-L sampling")
            if not math.isclose(sampling_dt, dt, rel_tol=1e-14):
                raise InputError(
                    "Workflow sampling and production time steps must match"
                )
            sampling["equilibration_steps"] = _steps(
                sampling,
                "equilibration_steps",
                "equilibration",
                "100 fs",
                dt,
                allow_zero=True,
            )
            adaptive_table = d.get("adaptive_umbrella", {})
            enhanced_sampling = task == "metadynamics" or (
                isinstance(adaptive_table, Mapping)
                and adaptive_table.get("enabled", bool(adaptive_table))
            )
            default_production = "1000 fs" if enhanced_sampling else "200 fs"
            if enhanced_sampling and not any(k in sampling for k in ("production_steps", "production")):
                notes.append("Enhanced-sampling default production: 1000 fs; inspect production ESS and CV coverage before increasing the budget or interpreting the FES.")
            sampling["production_steps"] = _steps(
                sampling, "production_steps", "production", default_production, dt
            )
            interval = sampling.get("interval_steps", sampling.get("sample_interval"))
            if interval is None:
                interval = time_steps(
                    sampling.get("interval", sampling.get("save_interval", "1 fs")), dt
                )
            sampling["interval_steps"] = sampling["sample_interval"] = _integer(
                interval, "sampling.interval_steps"
            )
            sampling["andersen_interval"] = _integer(
                sampling.get(
                    "andersen_interval",
                    sampling.get("thermostat_params", {}).get("interval_steps", 10),
                ),
                "sampling.andersen_interval",
            )
            sampling["replicas"] = _integer(
                sampling.get("replicas", 4 if task == "diffusion" else 1), "sampling.replicas"
            )
        if "parents" in sampling:
            sampling["parents"] = _integer(sampling["parents"], "sampling.parents")
        parallel = deepcopy(d.get("parallel", {}))
        if (
            "trajectory_workers" in parallel
            and "trajectory_groups" in parallel
            and parallel["trajectory_workers"] != parallel["trajectory_groups"]
        ):
            raise InputError("trajectory_workers and trajectory_groups disagree")
        # 'mode' names per-trajectory work; backend names task transport.
        parallel.setdefault("bead_workers", 1)
        parallel.setdefault(
            "mode",
            (
                "gpu"
                if str(potential.get("device", "")).startswith("cuda")
                else ("bead" if parallel["bead_workers"] > 1 else "serial")
            ),
        )
        default_workers = 1
        if parallel["mode"] == "gpu":
            devices = parallel.get("devices", [0])
            if isinstance(devices, (str, int)):
                devices = [devices]
            default_workers = len(devices) * _integer(
                parallel.get("trajectories_per_gpu", 1), "parallel.trajectories_per_gpu"
            )
        parallel.setdefault(
            "trajectory_workers",
            parallel.get("trajectory_groups", parallel.get("workers", default_workers)),
        )
        parallel.setdefault(
            "backend", "process" if parallel["trajectory_workers"] > 1 else "serial"
        )
        parallel.setdefault("threads_per_rank", 1)
        parallel.setdefault("pes_ranks", 1)
        parallel.setdefault("scheduling", "static")
        parallel.setdefault("coordinator_ranks", 1 if parallel["scheduling"] == "dynamic" else 0)
        for k in (
            "trajectory_workers",
            "bead_workers",
            "threads_per_rank",
            "pes_ranks",
        ):
            parallel[k] = _integer(parallel[k], f"parallel.{k}")
        from .parallel import execution_plan

        try:
            execution_plan(parallel)
        except ValueError as e:
            raise InputError(str(e)) from e
        if "nonadiabatic" in d or potential.get("provider", potential.get("type")) == "diabatic_python":
            validate_nonadiabatic(d, potential=potential, parallel=parallel, system=system)
            notes.append("Experimental NRPMD: state-selected scattering, not electronic thermal-equilibrium RPMD rate; real-chemistry validation is pending")
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
        output = deepcopy(d.get("output", {}))
        output["directory"] = _path(base, output.get("directory", f"runs/{task}"))
        output["checkpoint_steps"] = _steps(
            output, "checkpoint_steps", "checkpoint_interval", "10 fs", dt
        )
        if "trajectory_stride" in output:
            output["trajectory_stride"] = _integer(
                output["trajectory_stride"], "output.trajectory_stride"
            )
        if selected or "diagnostics" in output:
            if task != "drpmd":
                raise InputError("output.diagnostics currently requires task='drpmd'")
            diagnostics = output.get("diagnostics", {})
            if not isinstance(diagnostics, Mapping) or set(diagnostics) - {"energy_parts", "ring_radii"}:
                raise InputError("output.diagnostics must be a table containing only energy_parts and ring_radii")
            if any(not isinstance(value, (bool, np.bool_)) for value in diagnostics.values()):
                raise InputError("output.diagnostics flags must be boolean")
            output["diagnostics"] = {
                name: bool(diagnostics.get(name, selected))
                for name in ("energy_parts", "ring_radii")
            }
        if "store_configurations" in output and task != "metadynamics":
            raise InputError("output.store_configurations is supported only for metadynamics")
        resolved = {
            **d,
            "schema_version": 1,
            "preset": preset,
            "temperature_K": temperature,
            "beta": beta,
            "beads": beads,
            "seed": seed,
            "dt_au": dt,
            "system": sys_spec,
            "potential": potential,
            "sampling": sampling,
            "parallel": parallel,
            "reaction": reaction,
            "collision": collision,
            "outcomes": outcomes,
            "output": output,
            "_input_path": self.input_path,
            "_base_directory": str(base),
            "_warnings": notes,
            "hamiltonian_convention": "sum_beads_beta_over_P",
        }
        if selected:
            resolved["temperature_interpretation"] = (
                "auxiliary_ring_parameter_not_thermal_temperature"
                if initial_ensemble["method"] == "ne_rpmd"
                else "unused_classical_compatibility_parameter_not_thermal_temperature"
            )
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
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
            if "replica_exchange" in d:
                from .workflows.replica_exchange import exchange_options

                exchange_options(resolved, system)
            if source_pmf:
                from .workflows.surface_source import source_context
                source_context(resolved, system, verify=True)
            if "adaptive_umbrella" in d:
                from .workflows.adaptive_umbrella import validate_adaptive_umbrella

                validate_adaptive_umbrella(resolved, system)
        except (ValueError, TypeError, KeyError) as error:
            raise InputError(str(error)) from error
        return resolved, system


def load_input(path):
    return JobSpec.from_file(path)
