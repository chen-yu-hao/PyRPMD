"""Normalize systems and potential providers without starting production dynamics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .input_paths import _path
from .input_schema import InputError
from .system import System, system_from_config
from .units import parse_quantity


def resolve_system(d, base, runtime, *, initial_ensemble, source_pmf):
    task = d["task"]
    selected = initial_ensemble is not None
    beads, beta = runtime.beads, runtime.beta
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
    return system, sys_spec


def resolve_potential(d, base, system):
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
    return potential
