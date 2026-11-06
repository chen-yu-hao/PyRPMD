"""One dispatch point for the historical, native and tensor PIMD contracts.

An engine selects a numerical/RNG contract, not an automatic performance hint.
The legacy route is a compatibility backend of this workflow: selecting a route
never edits the input, upgrades its identity, or substitutes a different kernel.
Imports remain lazy so inspecting an input does not require optional runtimes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module


OBSERVABLE_NAMES = (
    "potential_energy",
    "kinetic_primitive",
    "kinetic_centroid_virial",
    "internal_energy_centroid_virial",
    "ring_polymer_hamiltonian",
)


def _validate_common(config, system, engine):
    """Shared preflight, before any backend can allocate an output or PES."""
    if select_engine(config).name != engine:
        raise ValueError("sampling.engine differs from the selected PIMD route")
    from ..potentials.lifecycle import active_learning_options
    from ..trajectory_batch import trajectory_batch_size

    if trajectory_batch_size(config) > 1 and "active_learning" in config:
        raise ValueError("materials trajectory batching does not support active-learning uncertainty callbacks")
    active_learning_options(config)
    if system.bond_constraints:
        from ..bond_constraints import BondConstraints
        from ..propagation import loop_options

        BondConstraints(system.masses, system.bond_constraints, cell=system.cell, pbc=system.pbc)
        loop = loop_options(config)
        if (engine != "legacy" or loop["integrator"] != "velocity-verlet"
                or loop["loop_backend"] == "native"):
            raise ValueError("bond constraints require the legacy Python RATTLE loop and velocity-verlet")
        if config.get("barostat") or config.get("electronic_friction"):
            raise ValueError("bond constraints currently require a fixed cell without electronic friction")
        if config.get("native", config.get("dynamics", {}).get("native", True)) == "required":
            raise ValueError("native='required' is not supported by the Python bond-constraint solver")


@dataclass(frozen=True)
class EngineRoute:
    """Lazy operations belonging to one explicitly selected execution contract."""

    name: str
    module: str
    resolver: str
    runner: str
    analyzer: str | None = None
    checker: str | None = None

    def _function(self, name):
        return getattr(import_module(self.module, __package__), name)

    def resolve(self, config, system):
        _validate_common(config, system, self.name)
        return self._function(self.resolver)(config, system)

    def run(self, config, system, output, executor=None, return_data=False):
        from .sampling import validate_andersen_schedule

        validate_andersen_schedule(config, "materials")
        _validate_common(config, system, self.name)
        return self._function(self.runner)(config, system, output, executor, return_data)

    def reanalyze(self, config, system, output, return_data=False):
        if config.get("task") != "diffusion":
            raise ValueError("reanalyze_diffusion requires a diffusion run")
        if self.analyzer is None:
            raise ValueError(f"sampling.engine={self.name!r} does not support diffusion reanalysis")
        if self.name == "native":
            # Keep missing native journals as a local preflight error.  The
            # actual analyzer runs through root_call so completed reanalysis
            # remains collective, but an absent run should not be wrapped in
            # a generic root-stage failure.
            from pathlib import Path

            options = self.resolve(config, system)
            missing = [Path(output) / "samples" / f"native_pimd_{replica:04d}" / "journal.json"
                       for replica in range(options["replicas"])
                       if not (Path(output) / "samples" / f"native_pimd_{replica:04d}" / "journal.json").exists()]
            if missing:
                raise ValueError(f"native diffusion journal is missing: {missing[0]}")
        from ..parallel import root_call

        def analyze():
            self.resolve(config, system)
            return self._function(self.analyzer)(config, system, output, return_data)

        return root_call(analyze)

    def check(self, config, system, provider):
        """Return a specialized PES check, or None for the generic legacy check."""
        self.resolve(config, system)
        if self.checker is None:
            return None
        return self._function(self.checker)(config, system, provider)


_ROUTES = {
    "legacy": EngineRoute("legacy", ".materials", "_resolve_legacy_materials",
                          "_run_legacy_materials", "_reanalyze_legacy_diffusion"),
    "native": EngineRoute("native", ".native_pimd", "resolve_native_pimd",
                          "run_native_pimd", analyzer="_reanalyze_native_diffusion",
                          checker="_check_native_pimd"),
    "torch": EngineRoute("torch", ".torch_materials", "resolve_torch_materials",
                         "run_torch_materials", "analyze_torch_materials", "check_torch_materials"),
}


def select_engine(config):
    """Select without normalizing config; an omitted engine remains omitted.

    In particular, omitted and explicit ``legacy`` retain their distinct old
    fingerprints. Unsupported choices fail rather than falling back.
    """
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a table")
    name = sampling.get("engine", "legacy")
    if not isinstance(name, str) or name not in _ROUTES:
        raise ValueError("sampling.engine must be legacy, native or torch")
    return _ROUTES[name]
