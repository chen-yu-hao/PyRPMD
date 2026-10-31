"""TOML/Python frontends with one checked, unit-resolved configuration.

JobSpec owns loading, overrides and normalization order. Schema, scientific
options, runtime controls and task-specific sections live in focused modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import tomllib

from .system import System
from .input_schema import InputError, _ROOT, _SECTIONS, _integer, _positive
from .input_options import validate_nonadiabatic, rate_pmf_options
from .input_paths import _path, _plugin_paths, _initial_ensemble_spec
from .input_runtime import (
    _steps, resolve_dynamics, resolve_sampling, resolve_parallel, resolve_output,
)
from .input_system import resolve_system, resolve_potential
from .input_tasks import (
    validate_task_sections, resolve_reaction, resolve_collision, validate_workflow,
)


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
            if d.get("task") == "pcet" and key not in ("temperature", "output"):
                raise InputError(
                    f"PCET supports only temperature/output overrides; {key} is not applicable to this serial model"
                )
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
        materials, native_pimd, source_pmf = validate_task_sections(d)
        runtime = resolve_dynamics(d, selected=selected, initial_ensemble=initial_ensemble)
        _plugin_paths(d, base)
        system, sys_spec = resolve_system(
            d, base, runtime, initial_ensemble=initial_ensemble, source_pmf=source_pmf,
        )
        potential = resolve_potential(d, base, system)
        sampling = resolve_sampling(
            d, runtime, selected=selected, materials=materials, native_pimd=native_pimd,
        )
        parallel = resolve_parallel(d, potential)
        if "nonadiabatic" in d or potential.get("provider", potential.get("type")) == "diabatic_python":
            validate_nonadiabatic(d, potential=potential, parallel=parallel, system=system)
            runtime.notes.append("Experimental NRPMD: state-selected scattering, not electronic thermal-equilibrium RPMD rate; real-chemistry validation is pending")
        reaction = resolve_reaction(d, system)
        collision, outcomes = resolve_collision(d, system, sys_spec, runtime, selected=selected)
        output = resolve_output(d, base, runtime.dt, selected=selected)
        resolved = {
            **d,
            "schema_version": 1,
            "preset": runtime.preset,
            "temperature_K": runtime.temperature,
            "beta": runtime.beta,
            "beads": runtime.beads,
            "seed": runtime.seed,
            "dt_au": runtime.dt,
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
            "_warnings": runtime.notes,
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
        validate_workflow(resolved, system, materials=materials, source_pmf=source_pmf)
        return resolved, system


def load_input(path):
    return JobSpec.from_file(path)
