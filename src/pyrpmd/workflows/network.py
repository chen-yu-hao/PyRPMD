"""Resumable entrance x channel x transition-state Bennett--Chandler studies.

Each path runs the existing complete, independently normalized rate workflow.
Different dividing surfaces are deliberately not added: without an explicit
physical partition their fluxes can overlap or be alternative estimates of
the same rate. Path-level finite-sampling and convergence diagnostics remain
visible in the network matrix.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..io import file_sha256, referenced_files, write_json
from ..parallel import root_call
from ..reactions import network_jobs, validate_network
from .prepare import prepare_rate
from .rate import run_rate


def prepare_network(config, system, numerical=False):
    """Prepare every declared path, checking each effective PES when requested."""
    inventory = validate_network(config, system)
    paths = []
    for job, local, path_system in network_jobs(config, system):
        prepared = prepare_rate(local, path_system, numerical=numerical)
        item = {
            "id": job["id"], "entrance": job["entrance"], "channel": job["channel"],
            "fingerprint": job["fingerprint"], "reaction_preparation": prepared,
            "potential_scope": job["potential_scope"],
        }
        if numerical:
            from ..potentials import make_potential, validate_result

            provider = make_potential(local["potential"], path_system)
            try:
                q = path_system.geometry[None]
                evaluated = validate_result(provider.evaluate(q), q.shape)
                item["pes_check"] = {
                    "energy_hartree": evaluated.energy.tolist(),
                    "maximum_reference_force_hartree_per_bohr": float(np.max(np.abs(evaluated.forces))),
                    "status": evaluated.status.tolist(),
                }
            finally:
                if hasattr(provider, "close"):
                    provider.close()
        paths.append(item)
    return {
        **inventory, "ready": all(p["reaction_preparation"]["ready"] for p in paths),
        "paths": paths,
        "diagnostics": [
            "Every path has its own two-fragment entrance normalization and complete PMF/recrossing calculation.",
            "Channel labels are organizational labels, not a proof that path fluxes are disjoint. No total rate is inferred.",
        ],
    }


def _dependencies(job, config):
    files = referenced_files(config)
    source = job.get("reference_source")
    if source:
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Transition-state source is missing: {source}")
        files[str(path.resolve())] = file_sha256(path)
    return files


def _initialize_path(job, config, system, directory, resume):
    path = Path(directory)
    manifest_path = path / "network-path.json"
    files = _dependencies(job, config)
    manifest = {
        "format": "PyRPMD.network.path.v1", "fingerprint": job["fingerprint"],
        "id": job["id"], "entrance": job["entrance"], "channel": job["channel"],
        "files": files, "system": system.to_dict(),
        "potential_scope": job["potential_scope"],
    }
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Network path already exists: {path}; use resume")
        previous = json.loads(manifest_path.read_text())
        if previous.get("fingerprint") != job["fingerprint"]:
            raise ValueError(f"Network path science changed: {job['id']}")
        if previous.get("files") != files:
            raise ValueError(f"Network path input/PES dependency changed: {job['id']}")
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Nonempty path has no network manifest: {path}")
        path.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        write_json(path / "input" / "resolved.json", config)
        write_json(path / "input" / "system.json", system.to_dict())
    return manifest


def _load_completed(job, directory):
    path = Path(directory) / "network-complete.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved.get("fingerprint") != job["fingerprint"]:
        raise ValueError(f"Completed path science changed: {job['id']}")
    return saved["result"]


def _save_completed(job, directory, result):
    write_json(Path(directory) / "network-complete.json", {
        "fingerprint": job["fingerprint"], "result": result,
    })


def _normalization(local, system):
    reaction = local["reaction"]
    masses = [float(system.masses[list(f)].sum()) for f in reaction["fragments"]]
    reduced = float(np.prod(masses) / sum(masses))
    rinf = float(reaction.get("rinf_au", reaction["rinf"]))
    return {
        "kind": "independent_bimolecular_entrance", "fragments": reaction["fragments"],
        "fragment_masses_au": masses, "reduced_mass_au": reduced,
        "rinf_bohr": rinf, "beta_au": local["beta"],
        "reactant_prefactor_au": float(4 * np.pi * rinf**2 / np.sqrt(2 * np.pi * local["beta"] * reduced)),
        "electronic_weight": reaction.get("electronic_weight", 1.0),
        "symmetry_factor": reaction.get("symmetry_factor", 1.0),
        "rate_unit": "cm^3 molecule^-1 s^-1",
    }


def _path_result(job, local, system, result, directory):
    return {
        "id": job["id"], "entrance": job["entrance"], "channel": job["channel"],
        "fingerprint": job["fingerprint"], "output": str(directory),
        "status": result.get("status"), "converged": bool(result.get("converged", False)),
        "rate_cm3_s": result.get("rate_cm3_s"),
        "rate_estimate_cm3_s": result.get("rate_estimate_cm3_s"),
        "rate_standard_error_cm3_s": result.get("rate_standard_error_cm3_s"),
        "rate_percentile_95_ci_cm3_s": result.get("rate_percentile_95_ci_cm3_s"),
        "kappa": result.get("kappa"), "plateau_detected": result.get("plateau_detected", False),
        "normalization": _normalization(local, system),
        "result": result,
    }


def _summary(config, results, output, *, complete):
    network = config["network"]
    matrix = {
        entrance["id"]: {
            channel["id"]: {"transition_states": {}, "summed_rate_cm3_s": None}
            for channel in network["channels"]
        }
        for entrance in network["entrances"]
    }
    for result in results:
        matrix[result["entrance"]][result["channel"]]["transition_states"][result["id"]] = result
    completed = {r["id"] for r in results}
    answer = {
        "format": "PyRPMD.network.summary.v1", "task": "network",
        "status": "completed" if complete else "partial", "converged": False,
        "aggregation": "separate", "total_rate_cm3_s": None,
        "rate_unit": "cm^3 molecule^-1 s^-1",
        "entrances": network["entrances"], "channels": network["channels"],
        "transition_states": len(network["jobs"]), "completed_paths": len(results),
        "pending_paths": [job["id"] for job in network["jobs"] if job["id"] not in completed],
        "matrix": matrix,
        "aggregation_note": "No rates are added across entrances or TSs. Different entrance partitions have different reactant normalizations; multiple TS surfaces may measure overlapping or identical reactive flux. A verified physical partition is required for channel/total aggregation.",
        "convergence_note": "Each path retains its own overlap, PMF, recrossing, energy and uncertainty diagnostics. Completing a network is not physical convergence.",
    }
    write_json(Path(output) / ("summary.json" if complete else "network-progress.json"), answer)
    return answer


def run_network(config, system, output, executor=None):
    """Run all paths with durable stage checkpoints and completed-path reuse.

    Every MPI rank follows the same ordered path list. Existing trajectory/
    window executors parallelize each path; no nested process pools are added.
    """
    inventory = root_call(validate_network, config, system)
    if not inventory["transition_states"]:
        raise ValueError("Network contains no work")
    results = []
    for job, local, path_system in network_jobs(config, system, output):
        directory = Path(local["output"]["directory"])
        root_call(_initialize_path, job, local, path_system, str(directory), bool(config.get("resume", False)))
        result = root_call(_load_completed, job, str(directory))
        if result is None:
            # run_rate prepares every explicit TS, samples its PMF, samples its
            # constrained parents and propagates actual paired children.
            result = run_rate(local, path_system, directory, executor=executor)
            root_call(_save_completed, job, str(directory), result)
        entry = _path_result(job, local, path_system, result, directory)
        results.append(entry)
        root_call(_summary, config, results, str(output), complete=False)
    return root_call(_summary, config, results, str(output), complete=True)
