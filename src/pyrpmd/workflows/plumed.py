"""Input-file access to the explicitly clocked PLUMED bridge.

Existing single-trajectory centroid inputs are delegated byte-for-byte to the
legacy workflow. Replica, native-walker, or bead-resolved inputs use the
external-file v2 workflow in :mod:`pyrpmd.workflows.plumed_extended`.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import plumed_legacy as _legacy


# Keep private helpers imported by downstream code/tests available while the
# legacy implementation lives in a separate module.
for _name in dir(_legacy):
    if not _name.startswith("__") and _name not in {
        "resolve_plumed_job",
        "run_plumed",
        "FIELDS",
    }:
        globals()[_name] = getattr(_legacy, _name)


_EXTENDED_FIELDS = frozenset({"mode", "walkers", "walker_read_stride"})
FIELDS = frozenset(_legacy.FIELDS) | _EXTENDED_FIELDS


def _uses_extended(config) -> bool:
    table = config.get("plumed", {})
    return (
        config.get("sampling", {}).get("replicas", 1) != 1
        or bool(_EXTENDED_FIELDS.intersection(table))
    )


def _raw_uses_extended(data) -> bool:
    table = data.get("plumed", {}) if isinstance(data, Mapping) else {}
    sampling = data.get("sampling", {}) if isinstance(data, Mapping) else {}
    return (
        isinstance(table, Mapping)
        and bool(_EXTENDED_FIELDS.intersection(table))
    ) or (
        isinstance(sampling, Mapping)
        and sampling.get("replicas", 1) != 1
    )


def resolve_plumed_job(data, *, base_directory=None, input_path=None):
    """Resolve legacy inputs unchanged and validate opt-in extended inputs."""
    if not _raw_uses_extended(data):
        return _legacy.resolve_plumed_job(
            data, base_directory=base_directory, input_path=input_path
        )

    from ..input import JobSpec, InputError, _path
    from ..plumed_extended import (
        configure_multiple_walkers,
        ensure_checkpoint_flush,
        validate_managed_output_paths,
    )

    base = Path(base_directory or ".").resolve()
    if not isinstance(data, Mapping) or data.get("task") != "plumed":
        raise InputError("resolve_plumed_job requires task='plumed'")
    raw = deepcopy(dict(data))
    table = raw.pop("plumed", None)
    if not isinstance(table, dict) or set(table) - FIELDS:
        raise InputError(
            "[plumed] must contain file or input, optional kernel, charges, "
            "restart, mode, walkers and walker_read_stride"
        )
    if ("file" in table) == ("input" in table):
        raise InputError("choose exactly one of plumed.file or plumed.input")
    forbidden = {
        "path_integral",
        "active_learning",
        "barostat",
        "electronic_friction",
        "reaction",
        "collision",
        "outcomes",
        "cvs",
        "umbrella",
        "analysis",
        "rate",
        "recrossing",
        "network",
        "metadynamics",
        "opes",
        "adaptive_umbrella",
        "pcet",
        "diffusion",
        "site_rate",
        "surface",
        "nonadiabatic",
        "replica_exchange",
        "initial_ensemble",
    }
    if forbidden.intersection(raw):
        raise InputError(
            "PLUMED workflow does not support sections: "
            + ", ".join(sorted(forbidden.intersection(raw)))
        )
    if "file" in table:
        if not isinstance(table["file"], str) or not table["file"]:
            raise InputError("plumed.file must be a nonempty file path")
        table["file"] = _path(base, table["file"])
        if not Path(table["file"]).is_file():
            raise InputError("PLUMED input file is missing")
        text = Path(table["file"]).read_text(encoding="utf-8")
    else:
        text = table["input"]
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise InputError("PLUMED input must be nonempty text")
    if "restart" in table and table["restart"] != "files":
        raise InputError("plumed.restart must be 'files' when specified")
    mode = table.get("mode", "centroid")
    if mode not in ("centroid", "beads"):
        raise InputError("plumed.mode must be 'centroid' or 'beads'")
    walkers = table.get("walkers", "independent")
    if walkers not in ("independent", "files", "mpi"):
        raise InputError(
            "plumed.walkers must be 'independent', 'files' or 'mpi'"
        )
    if "walker_read_stride" in table:
        value = table["walker_read_stride"]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise InputError(
                "plumed.walker_read_stride must be an integer >= 1"
            )
        if walkers != "files":
            raise InputError(
                "plumed.walker_read_stride requires plumed.walkers='files'"
            )
        table["walker_read_stride"] = int(value)
    if "kernel" in table:
        if not isinstance(table["kernel"], str) or not table["kernel"]:
            raise InputError("plumed.kernel must be a nonempty library path")
        table["kernel"] = _path(base, table["kernel"])
        if not Path(table["kernel"]).is_file():
            raise InputError("PLUMED kernel library is missing")

    output = raw.setdefault("output", {})
    if not isinstance(output, Mapping):
        raise InputError("output must be a table")
    unsupported_output = {
        "trajectory_stride",
        "trajectory_storage",
        "chunk_frames",
        "diagnostics",
        "store_configurations",
    }
    if unsupported_output.intersection(output):
        raise InputError(
            "PLUMED workflow does not support output controls: "
            + ", ".join(sorted(unsupported_output.intersection(output)))
        )
    output.setdefault("directory", "runs/plumed")
    raw["task"] = "pimd"
    resolved, system = JobSpec(raw, base, input_path).resolve()

    from . import validate_bond_constraint_workflow

    try:
        validate_bond_constraint_workflow(system, "plumed")
    except ValueError as error:
        raise InputError(str(error)) from error
    replicas = resolved["sampling"].get("replicas", 1)
    if walkers != "independent" and replicas < 2:
        raise InputError(
            f"plumed.walkers={walkers!r} requires sampling.replicas>=2"
        )
    if system.fixed_atoms or (any(system.pbc) and not all(system.pbc)):
        raise InputError(
            "PLUMED workflow requires all-mobile atoms and all/no PBC"
        )
    if (
        resolved["sampling"].get("engine", "legacy") != "legacy"
        or resolved.get("dynamics", {}).get("loop_backend") == "native"
    ):
        raise InputError("PLUMED requires Python step orchestration")
    if resolved["sampling"]["dt_au"] != resolved["dt_au"]:
        raise InputError(
            "PLUMED requires sampling.timestep equal to dynamics.timestep "
            "for its fixed update clock"
        )
    if resolved["sampling"]["thermostat"] not in ("andersen", "pile-l"):
        raise InputError(
            "PLUMED workflow supports only Andersen or PILE-L sampling"
        )
    if {"parents", "thermostat_params"}.intersection(
        data.get("sampling", {})
    ):
        raise InputError(
            "PLUMED workflow does not use parents or NHC thermostat parameters"
        )
    if "charges" in table:
        charges = np.asarray(table["charges"])
        if (
            charges.dtype.kind not in "fiu"
            or charges.shape != (system.natoms,)
            or not np.isfinite(charges).all()
        ):
            raise InputError(
                "plumed.charges must be a finite real vector of length N"
            )

    parallel = resolved["parallel"]
    backend = parallel["backend"]
    scheduling = parallel.get("scheduling", "static")
    groups = parallel.get(
        "trajectory_groups", parallel.get("trajectory_workers", 1)
    )
    coordinators = parallel.get("coordinator_ranks", 0)
    if backend == "mpi":
        if scheduling != "static" or groups != replicas or coordinators != 0:
            raise InputError(
                "Extended PLUMED MPI execution requires static scheduling, "
                "trajectory_groups=replicas and coordinator_ranks=0"
            )
    if walkers == "files":
        if backend == "serial":
            raise InputError(
                "PLUMED file walkers require process or MPI execution"
            )
        if scheduling != "static":
            raise InputError(
                "PLUMED file walkers require parallel.scheduling='static'"
            )
        if (
            backend == "process"
            and parallel.get("trajectory_workers", 1) != replicas
        ):
            raise InputError(
                "PLUMED process file walkers require "
                "trajectory_workers=replicas"
            )
    elif walkers == "mpi" and backend != "mpi":
        raise InputError(
            "PLUMED MPI walkers require parallel.backend='mpi'"
        )

    try:
        configured = configure_multiple_walkers(
            text,
            walkers=walkers,
            count=replicas,
            walker_id=0,
            read_stride=table.get("walker_read_stride", 1),
        )
        if table.get("restart") == "files":
            configured = ensure_checkpoint_flush(configured)
        validate_managed_output_paths(configured)
    except ValueError as error:
        raise InputError(str(error)) from error

    resolved.update(task="plumed", plumed=table, _input_path=input_path)
    resolved["_warnings"].append(
        "Extended PLUMED mode gives each replica a private kernel, log and "
        "replica-NNNN working directory. File walkers share only the managed "
        "plumed/shared tree; MPI walkers use PLUMED's multi-sim communicator."
    )
    if mode == "beads":
        resolved["_warnings"].append(
            "PLUMED bead mode exposes P*N synthetic atoms in bead-major order; "
            "the run writes results/plumed-atom-map.json."
        )
    if table.get("restart") == "files":
        resolved["_warnings"].append(
            "PLUMED external-files-v2 checkpoints store atomic MD/RNG state "
            "and hash every managed external-state prefix; resume rejects "
            "missing, replaced, shortened or untracked files."
        )
    else:
        resolved["_warnings"].append(
            "Automatic continuation of extended PLUMED runs requires "
            "plumed.restart='files'."
        )
    return resolved, system


def run_plumed(config, system, output):
    """Dispatch only opt-in inputs to the extended workflow."""
    if _uses_extended(config):
        from .plumed_extended import run_extended_plumed

        return run_extended_plumed(config, system, output)
    return _legacy.run_plumed(config, system, output)


__all__ = ["FIELDS", "resolve_plumed_job", "run_plumed"]
