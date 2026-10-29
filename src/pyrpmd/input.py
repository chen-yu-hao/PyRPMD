"""TOML/Python frontends with one checked, unit-resolved configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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


_ROOT = set(
    "schema_version task preset temperature beads seed system potential dynamics sampling reaction collision outcomes cvs umbrella analysis parallel output native description rate recrossing".split()
)
_SECTIONS = {
    "system": set(
        "geometry transition_state reactants separation geometry_unit isotopes masses masses_amu symbols fragments charge multiplicity".split()
    ),
    "dynamics": set("timestep integrator native".split()),
    "sampling": set(
        "integrator thermostat timestep equilibration production save_interval interval interval_steps sample_interval equilibration_steps production_steps andersen_interval thermostat_params replicas parents".split()
    ),
    "potential": set(
        "provider type name library manifest factory module function initializer initializer_params with_dimensions model weights device dtype units derivative params".split()
    ),
    "parallel": set(
        "backend mode trajectory_groups trajectory_workers workers bead_workers pes_ranks threads_per_rank devices trajectories_per_gpu coordinator_ranks scheduling".split()
    ),
    "collision": set(
        "sampling impact_sampling bmax trajectories max_time max_steps energy orientation".split()
    ),
    "outcomes": set(
        "factory params channels exit_distance persistence persistence_steps check_interval bond_cutoffs".split()
    ),
    "output": set(
        "directory checkpoint_interval checkpoint_steps trajectory_stride".split()
    ),
    "reaction": set(
        "fragments forming_bonds breaking_bonds electronic_weight rinf Rinf equivalent_paths symmetry_factor preparation entry_side".split()
    ),
    "analysis": set(
        "method measure bins bootstrap_samples bootstrap_replicates bootstrap seed overlap_threshold ranges min_count".split()
    ),
    "umbrella": set(
        "cvs grid centers centers_au kappa stiffness stiffness_au start stop count".split()
    ),
    "rate": set("recrossing pmf_points dividing_surface".split()),
    "recrossing": set(
        "parent_replicas parent_samples equilibration equilibration_steps parent_interval parent_interval_steps child_pairs child_pairs_per_parent child_steps child_time dividing_surface plateau_fraction plateau_tolerance bootstrap constraint_tolerance".split()
    ),
}


def _integer(v, name, minimum=1):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")
    return int(v)


def _positive(v, name):
    if not math.isfinite(v) or v <= 0:
        raise InputError(f"{name} must be positive and finite")
    return float(v)


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
                raise InputError(str(e)) from e
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
            spec.loader.exec_module(module)
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
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "pes":
                d.setdefault("potential", {})["library"] = str(Path(value).resolve())
                if d["potential"].get("provider") not in ("rpmdrate", "f2py"):
                    d["potential"]["provider"] = "cabi"
            elif key == "pes_manifest":
                d.setdefault("potential", {})["manifest"] = str(Path(value).resolve())
            elif key == "output":
                d.setdefault("output", {})["directory"] = str(Path(value).resolve())
            elif key == "workers":
                p = d.setdefault("parallel", {})
                if "trajectory_groups" in p and p["trajectory_groups"] != value:
                    p.pop("trajectory_groups")
                p["trajectory_workers"] = value
                if p.get("backend", "serial") != "mpi":
                    p["backend"] = "process" if value > 1 else "serial"
            else:
                d[key] = value
        return JobSpec(d, self.base_directory, self.input_path)

    def resolve(self):
        d = deepcopy(self.data)
        unknown = set(d) - _ROOT
        if unknown:
            raise InputError(f"Unknown top-level fields: {', '.join(sorted(unknown))}")
        for section, allowed in _SECTIONS.items():
            if section in d:
                if section == "system" and isinstance(d[section], System):
                    continue
                if not isinstance(d[section], Mapping):
                    raise InputError(f"{section} must be a table")
                unknown = set(d[section]) - allowed
                if unknown:
                    raise InputError(
                        f"Unknown {section} fields: {', '.join(sorted(unknown))}"
                    )
        if _integer(d.get("schema_version", 1), "schema_version") != 1:
            raise InputError("Supported schema_version is 1")
        task = d.get("task")
        if task not in ("rate", "drpmd", "fes"):
            raise InputError("task must be rate, drpmd or fes")
        analysis = d.get("analysis", {})
        if analysis.get("measure", "cv_density") not in ("cv_density", "rate_flux"):
            raise InputError("analysis.measure must be cv_density or rate_flux")
        if task == "fes" and analysis.get("method", "mbar") != "mbar":
            raise InputError(
                "The FES workflow uses MBAR; umbrella integration is part of the rate workflow"
            )
        if task == "rate" and analysis.get("measure", "rate_flux") != "rate_flux":
            raise InputError("Standard RPMD rates require analysis.measure='rate_flux'")
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
        preset = d.get("preset", "pilot")
        if preset not in ("pilot", "production"):
            raise InputError("preset must be pilot or production")
        notes = []
        if preset == "production":
            for k in ("temperature", "beads", "dynamics", "sampling"):
                if k not in d:
                    raise InputError(f"production requires explicit {k}")
        if "temperature" not in d:
            notes.append("Pilot default temperature: 300 K")
        temperature = _positive(
            parse_quantity(d.get("temperature", "300 K"), "temperature"), "temperature"
        )
        beads = d.get("beads", 32)
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
        seed = _integer(d.get("seed", 0), "seed", 0)
        dyn = dict(d.get("dynamics", {}))
        dt = _positive(
            parse_quantity(dyn.get("timestep", "0.1 fs"), "time"), "dynamics.timestep"
        )
        if dyn.get("integrator", "cayley-bcb") != "cayley-bcb":
            raise InputError("Production integrator in v0.1 is cayley-bcb")
        base = Path(self.base_directory).resolve()
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
        potential = deepcopy(d.get("potential", {}))
        if not potential:
            raise InputError("Specify [potential] or --pes; no PES is guessed")
        for k in ("library", "manifest", "weights"):
            if k in potential:
                potential[k] = _path(base, potential[k])
        if "model" in potential and potential.get(
            "provider", potential.get("type")
        ) not in ("builtin", None):
            potential["model"] = _path(base, potential["model"])
        # Model-specific path-valued parameters can be declared relative to input.
        for k in ("model", "weights", "reference"):
            if isinstance(potential.get("params", {}).get(k), str):
                potential["params"][k] = _path(base, potential["params"][k])
        sampling = deepcopy(d.get("sampling", {}))
        sampling_dt = _positive(
            parse_quantity(sampling.get("timestep", dt), "time"), "sampling.timestep"
        )
        sampling["dt_au"] = sampling_dt
        sampling["integrator"] = sampling.get("integrator", "cayley-bcb-andersen")
        sampling["thermostat"] = sampling.get("thermostat", "andersen")
        if (
            sampling["integrator"] != "cayley-bcb-andersen"
            or sampling["thermostat"] != "andersen"
        ):
            raise InputError(
                "Workflow sampling v0.1 uses cayley-bcb-andersen/andersen; standalone Dynamics.sample_step exposes BCOCB"
            )
        if not math.isclose(sampling_dt, dt, rel_tol=1e-14):
            raise InputError(
                "Workflow sampling and production time steps must match in v0.1"
            )
        sampling["equilibration_steps"] = _steps(
            sampling,
            "equilibration_steps",
            "equilibration",
            "100 fs",
            dt,
            allow_zero=True,
        )
        sampling["production_steps"] = _steps(
            sampling, "production_steps", "production", "200 fs", dt
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
            sampling.get("replicas", 1), "sampling.replicas"
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
        parallel.setdefault("coordinator_ranks", 0)
        parallel.setdefault("scheduling", "static")
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
            collision.setdefault("sampling", "thermal_flux")
            collision.setdefault("impact_sampling", "uniform_area")
            collision.setdefault("orientation", "isotropic")
            if collision["sampling"] not in ("thermal_flux", "maxwell", "fixed_energy"):
                raise InputError("Unknown collision sampling distribution")
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
        resolved = {
            **d,
            "schema_version": 1,
            "preset": preset,
            "temperature_K": temperature,
            "beta": 1 / (KB_AU * temperature),
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
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        return resolved, system


def load_input(path):
    return JobSpec.from_file(path)
