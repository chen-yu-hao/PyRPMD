"""Scientific workflows share the same system, PES and execution contracts."""

from collections.abc import Mapping


def validate_bond_constraint_workflow(system, workflow):
    """Reject unsupported constraints even when callers bypass JobSpec.resolve.

    Public specialized entrypoints pass their own workflow name, not a caller's
    ``config['task']``. Only the ordinary materials PIMD/diffusion implementation
    currently connects System constraints to a constrained propagator. Its own
    resolver additionally checks the engine, integrator and sampling choices.
    """
    bonds = (system.get("bond_constraints", ()) if isinstance(system, Mapping)
             else getattr(system, "bond_constraints", ()))
    if bonds and workflow not in ("pimd", "diffusion"):
        raise ValueError(
            f"bond constraints are unsupported by {workflow}; use the supported "
            "fixed-cell PIMD/diffusion workflow with velocity-verlet and Andersen sampling"
        )
