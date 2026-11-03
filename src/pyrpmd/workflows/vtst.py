"""Standalone VTST/ZCT/SCT workflow for validated gas-phase rate inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..input import InputError, JobSpec
from ..io import file_sha256, fingerprint_payload, jsonable
from ..units import parse_quantity
from ..validation import nonnegative_integer, positive_integer
from ..vtst import (
    MEPOptions, VTSTError, compute_vtst, fragment_species_from_path_endpoint,
    trace_mep, validate_reaction_path_endpoints,
)
from ._vtst_output import _publish, _scientific_identity
from .prepare import preparation_options, prepare_rate


def _job_spec(value: Any) -> JobSpec:
    if isinstance(value, JobSpec):
        return value
    if isinstance(value, (str, Path)):
        return JobSpec.from_file(value)
    if isinstance(value, Mapping):
        return JobSpec.from_mapping(value)
    raise InputError("vtst input must be a JobSpec, mapping, or .inp/.toml/.py path")


def _integer_list(value: Sequence[int] | None, name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise InputError(f"{name} must contain positive integers")
    result = []
    try:
        for item in value:
            result.append(positive_integer(item, f"{name} entry"))
    except TypeError as exc:
        raise InputError(f"{name} must contain positive integers") from exc
    if not result:
        raise InputError(f"{name} must not be empty")
    return result


def _temperatures(config: Mapping[str, Any], supplied: Sequence[Any] | None) -> list[float]:
    raw = [config["temperature_K"]] if supplied is None else list(supplied)
    if not raw:
        raise InputError("vtst requires at least one temperature")
    values = [parse_quantity(value, "temperature") for value in raw]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise InputError("vtst temperatures must be positive and finite")
    return values


def _output_target(output: str | Path) -> Path:
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise InputError(
            f"VTST output already exists: {target}; choose a new directory"
        )
    parent = target.parent
    if parent.exists() and not parent.is_dir():
        raise InputError(f"VTST output parent is not a directory: {parent}")
    return target


def _mep_options(
    *,
    step: Any,
    initial_displacement: Any,
    max_steps_per_side: int,
    gradient_tolerance: Any,
    hessian_step: Any,
    negative_eigenvalue_tolerance: float,
    minimum_steps_per_side: int,
) -> MEPOptions:
    return MEPOptions(
        step_bohr=parse_quantity(step, "length"),
        initial_displacement_bohr=parse_quantity(initial_displacement, "length"),
        max_steps_per_side=positive_integer(max_steps_per_side, "max_steps_per_side"),
        gradient_tolerance_hartree_per_bohr=parse_quantity(
            gradient_tolerance, "force"
        ),
        hessian_step_bohr=parse_quantity(hessian_step, "length"),
        negative_eigenvalue_tolerance=float(negative_eigenvalue_tolerance),
        minimum_steps_per_side=nonnegative_integer(
            minimum_steps_per_side, "minimum_steps_per_side"
        ),
    ).validated()


def _reaction_fragments(reaction: Mapping[str, Any], natoms: int) -> list[list[int]]:
    raw = reaction.get("fragments")
    if raw is None:
        return [list(range(natoms))]
    groups = [[int(atom) for atom in group] for group in raw]
    flat = [atom for group in groups for atom in group]
    if any(not group for group in groups) or sorted(flat) != list(range(natoms)):
        raise VTSTError("prepared reaction fragments must partition all atoms")
    return groups


def run_vtst(
    job: Any,
    *,
    output: str | Path,
    temperatures_kelvin: Sequence[Any] | None = None,
    mep_step: Any = "0.05 bohr",
    initial_displacement: Any = "0.02 bohr",
    max_steps_per_side: int = 80,
    gradient_tolerance: Any = "1e-5 hartree/bohr",
    hessian_step: Any = "1e-4 bohr",
    negative_eigenvalue_tolerance: float = 1.0e-9,
    minimum_steps_per_side: int = 2,
    reactant_symmetry_numbers: Sequence[int] | None = None,
    reactant_electronic_degeneracies: Sequence[int] | None = None,
    transition_state_symmetry_number: int = 1,
    transition_state_electronic_degeneracy: int = 1,
    reaction_degeneracy: int = 1,
    tunnelling_quadrature_points: int = 64,
    bond_scale: float | None = None,
) -> dict[str, Any]:
    """Run deterministic MEP/CVT/ZCT/SCT and atomically publish a new directory.

    The result is intentionally not restartable: there is no RNG and no
    checkpoint state.  Every option that changes a scientific value is included
    in the standalone v1 physics fingerprint.  Existing RPMD identities and
    checkpoint formats are untouched.
    """
    target = _output_target(output)
    spec = _job_spec(job)
    config, system = spec.resolve()
    if config.get("task") != "rate":
        raise InputError("pyrpmd vtst requires an input with task='rate'")
    if any(system.pbc) or system.fixed_atoms or system.bond_constraints:
        raise InputError(
            "VTST currently requires a nonperiodic gas-phase system without fixed atoms or bond constraints"
        )
    temperatures = _temperatures(config, temperatures_kelvin)
    mep = _mep_options(
        step=mep_step,
        initial_displacement=initial_displacement,
        max_steps_per_side=max_steps_per_side,
        gradient_tolerance=gradient_tolerance,
        hessian_step=hessian_step,
        negative_eigenvalue_tolerance=negative_eigenvalue_tolerance,
        minimum_steps_per_side=minimum_steps_per_side,
    )
    reactant_symmetry = _integer_list(
        reactant_symmetry_numbers, "reactant_symmetry_numbers"
    )
    reactant_electronic = _integer_list(
        reactant_electronic_degeneracies,
        "reactant_electronic_degeneracies",
    )
    ts_symmetry = positive_integer(
        transition_state_symmetry_number, "transition_state_symmetry_number"
    )
    ts_electronic = positive_integer(
        transition_state_electronic_degeneracy,
        "transition_state_electronic_degeneracy",
    )
    path_degeneracy = positive_integer(reaction_degeneracy, "reaction_degeneracy")
    quadrature = positive_integer(
        tunnelling_quadrature_points, "tunnelling_quadrature_points", minimum=16
    )
    if bond_scale is not None:
        if isinstance(bond_scale, (bool, np.bool_)) or not np.isscalar(bond_scale):
            raise InputError("bond_scale must be positive and finite")
        bond_scale = float(bond_scale)
        if not math.isfinite(bond_scale) or bond_scale <= 0.0:
            raise InputError("bond_scale must be positive and finite")

    source_path = Path(spec.input_path).resolve() if spec.input_path else None
    source_sha = file_sha256(source_path) if source_path is not None else None

    from ..potentials import make_potential

    potential = make_potential(config["potential"], system)
    try:
        prepared = prepare_rate(config, system, potential=potential, numerical=True)
        if not prepared.get("ready"):
            details = "; ".join(prepared.get("diagnostics", ()))
            raise VTSTError(f"reaction preparation is not ready for VTST: {details}")
        if prepared.get("saddle_valid") is not True:
            raise VTSTError(
                "VTST requires a numerically validated first-order saddle; set preparation.ts_action='validate' or 'refine'"
            )
        if "transition_state" not in prepared:
            raise VTSTError("reaction preparation did not return a transition-state geometry")
        path = trace_mep(
            potential,
            np.asarray(prepared["transition_state"], dtype=float),
            system.masses,
            options=mep,
        )
        prep_options = preparation_options(prepared["reaction"])
        endpoint_scale = (
            prep_options["bond_scale"] if bond_scale is None else float(bond_scale)
        )
        endpoint = validate_reaction_path_endpoints(
            path,
            system.symbols,
            prepared["reaction"],
            bond_scale=endpoint_scale,
            covalent_radii=prep_options.get("covalent_radii"),
        )
        if not endpoint["connected"]:
            raise VTSTError(
                "MEP termini do not realize the declared reactant/product connectivity; extend or refine the path"
            )
        if endpoint["orientation"] == "right_to_left":
            path = path.reversed()
            endpoint = validate_reaction_path_endpoints(
                path,
                system.symbols,
                prepared["reaction"],
                bond_scale=endpoint_scale,
                covalent_radii=prep_options.get("covalent_radii"),
            )
        groups = _reaction_fragments(prepared["reaction"], system.natoms)
        reactant_symmetry = reactant_symmetry or [1] * len(groups)
        reactant_electronic = reactant_electronic or [1] * len(groups)
        if len(reactant_symmetry) != len(groups):
            raise VTSTError(
                "reactant_symmetry_numbers must provide one value per reactant fragment"
            )
        if len(reactant_electronic) != len(groups):
            raise VTSTError(
                "reactant_electronic_degeneracies must provide one value per reactant fragment"
            )
        reactants = fragment_species_from_path_endpoint(
            path,
            system.masses,
            groups,
            symmetry_numbers=reactant_symmetry,
            electronic_degeneracies=reactant_electronic,
        )
        calculation = compute_vtst(
            path,
            system.masses,
            reactants,
            temperatures,
            transition_state_symmetry_number=ts_symmetry,
            transition_state_electronic_degeneracy=ts_electronic,
            reaction_degeneracy=path_degeneracy,
            endpoint_validation=endpoint,
            tunnelling_quadrature_points=quadrature,
        )
        result = calculation.to_dict(include_hessians=True)
        result["preparation"] = jsonable(prepared)
        result["mep_options"] = jsonable(mep.__dict__)
    finally:
        if hasattr(potential, "close"):
            potential.close()

    identity = _scientific_identity(
        config,
        system,
        temperatures,
        mep,
        reactant_symmetry_numbers=reactant_symmetry,
        reactant_electronic_degeneracies=reactant_electronic,
        transition_state_symmetry_number=ts_symmetry,
        transition_state_electronic_degeneracy=ts_electronic,
        reaction_degeneracy=path_degeneracy,
        tunnelling_quadrature_points=quadrature,
        bond_scale=endpoint_scale,
        source_sha256=source_sha,
    )
    identity = jsonable(identity, callables=True)
    fingerprint = fingerprint_payload(identity)
    result["physics_fingerprint"] = fingerprint
    return _publish(
        target,
        config=config,
        system=system,
        source_path=source_path,
        identity=identity,
        fingerprint=fingerprint,
        result=result,
    )


def _command(args: Any) -> int:
    summary = run_vtst(
        args.input,
        output=args.output,
        temperatures_kelvin=args.temperatures,
        mep_step=args.mep_step,
        initial_displacement=args.initial_displacement,
        max_steps_per_side=args.mep_steps,
        gradient_tolerance=args.gradient_tolerance,
        hessian_step=args.hessian_step,
        negative_eigenvalue_tolerance=args.negative_eigenvalue_tolerance,
        minimum_steps_per_side=args.minimum_mep_steps,
        reactant_symmetry_numbers=args.reactant_symmetry,
        reactant_electronic_degeneracies=args.reactant_electronic_degeneracy,
        transition_state_symmetry_number=args.transition_state_symmetry,
        transition_state_electronic_degeneracy=(
            args.transition_state_electronic_degeneracy
        ),
        reaction_degeneracy=args.reaction_degeneracy,
        tunnelling_quadrature_points=args.tunnelling_points,
        bond_scale=args.bond_scale,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


def register_parser(commands: Any) -> Any:
    """Register ``pyrpmd vtst`` without changing the shared input schema."""
    parser = commands.add_parser(
        "vtst",
        help="Compute MEP-based TST, CVT, ZCT and SCT comparison rates",
    )
    parser.add_argument("input", help="Validated task='rate' .inp/.toml or Python input")
    parser.add_argument("--output", required=True, help="New atomically published result directory")
    parser.add_argument(
        "--temperature",
        dest="temperatures",
        action="append",
        help="Temperature quantity; repeat for a grid (default: input temperature)",
    )
    parser.add_argument("--mep-step", default="0.05 bohr")
    parser.add_argument("--initial-displacement", default="0.02 bohr")
    parser.add_argument("--mep-steps", type=int, default=80, help="Maximum accepted points per MEP side")
    parser.add_argument("--minimum-mep-steps", type=int, default=2)
    parser.add_argument("--gradient-tolerance", default="1e-5 hartree/bohr")
    parser.add_argument("--hessian-step", default="1e-4 bohr")
    parser.add_argument(
        "--negative-eigenvalue-tolerance",
        type=float,
        default=1.0e-9,
        help="Mass-weighted omega^2 threshold in atomic units",
    )
    parser.add_argument("--reactant-symmetry", nargs="+", type=int)
    parser.add_argument("--reactant-electronic-degeneracy", nargs="+", type=int)
    parser.add_argument("--transition-state-symmetry", type=int, default=1)
    parser.add_argument(
        "--transition-state-electronic-degeneracy", type=int, default=1
    )
    parser.add_argument("--reaction-degeneracy", type=int, default=1)
    parser.add_argument("--tunnelling-points", type=int, default=64)
    parser.add_argument("--bond-scale", type=float)
    parser.set_defaults(_vtst_handler=_command)
    return parser
