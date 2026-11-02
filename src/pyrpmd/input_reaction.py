"""Explicit reaction identity and independently requested TS preparation.

The old schema-1 implicit preparation path is preserved unless an input selects
the new controls.  Schema 2 requires an explicit choice to infer a reaction.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

from .input_schema import InputError


def _fragments(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise InputError(f"{name} must be a nonempty list of atom-index lists")
    result = []
    seen = set()
    for fragment in value:
        if not isinstance(fragment, (list, tuple)) or not fragment:
            raise InputError(f"{name} requires nonempty fragments")
        group = []
        for atom in fragment:
            if isinstance(atom, bool) or not isinstance(atom, Integral) or atom < 0:
                raise InputError(f"{name} requires nonnegative 0-based integer atom indices")
            if atom in seen:
                raise InputError(f"{name} cannot repeat an atom within or across fragments")
            seen.add(atom)
            group.append(int(atom))
        result.append(group)
    return result


def normalize_reaction_input(data):
    """Mutate the copied front-end mapping to canonical reaction controls.

    This is deliberately a structural operation: it neither constructs a PES
    nor changes geometry.  Full atom coverage and saddle/channel checks occur
    once the molecular system is known.
    """
    reaction = data.get("reaction", {})
    if not isinstance(reaction, Mapping):
        raise InputError("reaction must be a table")
    nested = reaction.get("preparation", {})
    if not isinstance(nested, Mapping):
        raise InputError("reaction.preparation must be a table")
    selected = (
        "preparation" in data
        or any(key in reaction for key in ("definition", "reactant_fragments", "product_fragments"))
        or any(key in nested for key in ("ts_action", "check_channel"))
    )
    if data.get("task") != "rate":
        if selected:
            raise InputError("Explicit reaction/preparation controls require task='rate'")
        return
    if data.get("schema_version", 1) != 2 and not selected:
        return

    reaction = deepcopy(dict(reaction))
    if "reactant_fragments" in reaction:
        if "fragments" in reaction:
            raise InputError("Specify reaction.reactant_fragments or reaction.fragments, not both")
        reaction["fragments"] = reaction.pop("reactant_fragments")
    if "fragments" not in reaction:
        system = data.get("system", {})
        if isinstance(system, Mapping) and "fragments" in system:
            reaction["fragments"] = deepcopy(system["fragments"])
        elif getattr(system, "fragments", None):
            reaction["fragments"] = [list(group) for group in system.fragments]
    for key in ("fragments", "product_fragments"):
        if key in reaction:
            reaction[key] = _fragments(reaction[key], f"reaction.{key}")

    definition = reaction.get("definition", "explicit")
    if not isinstance(definition, str) or definition not in ("explicit", "infer"):
        raise InputError("reaction.definition must be 'explicit' or 'infer'")
    if definition == "explicit":
        required = ("fragments", "forming_bonds", "breaking_bonds")
        missing = [key for key in required if not reaction.get(key)]
        if missing:
            raise InputError(
                "reaction.definition='explicit' requires reactant fragments and paired "
                "forming_bonds/breaking_bonds; missing: " + ", ".join(missing)
                + ". Select reaction.definition='infer' explicitly to infer the entrance."
            )
    reaction["definition"] = definition

    if "preparation" in data:
        if "preparation" in reaction:
            raise InputError("Specify [preparation] or [reaction.preparation], not both")
        raw = data.pop("preparation")
        if not isinstance(raw, Mapping):
            raise InputError("preparation must be a table")
        preparation = deepcopy(dict(raw))
    else:
        preparation = deepcopy(dict(nested))
    if "ts_action" in preparation and "refine" in preparation:
        raise InputError("preparation.ts_action and legacy preparation.refine are mutually exclusive")
    if "refine" in preparation:
        refine = preparation.pop("refine")
        if not isinstance(refine, bool):
            raise InputError("preparation.refine must be boolean")
        preparation["ts_action"] = "refine" if refine else "validate"
    action = preparation.get("ts_action", "validate")
    if not isinstance(action, str) or action not in ("validate", "refine", "none"):
        raise InputError("preparation.ts_action must be 'validate', 'refine' or 'none'")
    check_channel = preparation.get("check_channel", definition == "infer")
    if not isinstance(check_channel, bool):
        raise InputError("preparation.check_channel must be boolean")
    if definition == "infer" and not check_channel:
        raise InputError("reaction.definition='infer' requires preparation.check_channel=true")
    if action == "none":
        system = data.get("system", {})
        endpoints = isinstance(system, Mapping) and any(key in system for key in ("reactant", "product"))
        if definition != "explicit" or check_channel or endpoints:
            raise InputError(
                "preparation.ts_action='none' requires an explicit reaction, a supplied TS, "
                "and check_channel=false"
            )
    preparation["ts_action"] = action
    preparation["check_channel"] = check_channel
    reaction["preparation"] = preparation
    data["reaction"] = reaction
