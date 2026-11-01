"""Replay saved H3 RPMD children at half dt with identical initial states.

This changes only unconstrained child propagation: the saved PMF, constrained
parents, dividing surface, 100 fs child duration and random streams are fixed.
It does not establish equilibrium, parent-timestep convergence, bead convergence
or a physical rate. Original child checkpoints do not contain momenta; those
are reconstructed using the original Dynamics sampler, its recorded seed and
the same NumPy version. All original checkpoint fingerprints and flux weights
must match before new propagation starts. The worker observes (without changing)
the actual momentum draw and checks it against the reconstructed full array.

Example, after the endpoint rate driver has completed its P=128 case::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python \
        benchmarks/transition_state_search/recrossing_timestep.py \
        --source /path/to/quantum-128 --output /path/to/fresh-halfdt \
        --evidence h3-recrossing-halfdt.json --workers 4

The output directory must be new. It retains initial q/p arrays, child
checkpoints and full transmission curves; the evidence JSON keeps paired
energy errors and hashes. No production source is modified.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyrpmd.analysis.recrossing import summarize_recrossing
from pyrpmd.constants import AU_TIME_FS
from pyrpmd.dynamics import Dynamics
from pyrpmd.io import file_sha256, write_json
from pyrpmd.parallel import run_tasks
from pyrpmd.system import System
from pyrpmd.workflows.prepare import prepared_system
from pyrpmd.workflows.rate import (
    _child_task, _child_tasks, _normal, _recrossing_options, rate_coordinate,
)


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def restored_configuration(source):
    """Restore saved preparation without running optimization or any sampler."""
    settings = json.loads((source / "settings.json").read_text())
    prepared = json.loads((source / "run/preparation.json").read_text())
    if not prepared["ready"]:
        raise ValueError("Original preparation was not ready")
    config = copy.deepcopy(settings["config"])
    system = prepared_system(System.from_dict(settings["initial_system"]), prepared)
    config["reaction"] = prepared["reaction"]
    reaction = config["reaction"]
    config["cvs"] = [{
        "name": "xi", "kind": "rpmdrate_xi", "estimator": "centroid",
        **{key: reaction[key] for key in ("fragments", "forming_bonds", "breaking_bonds", "rinf")},
    }]
    if reaction.get("equivalent_paths"):
        config["cvs"][0]["equivalent_paths"] = reaction["equivalent_paths"]
    config["umbrella"]["cvs"] = ["xi"]
    config["analysis"]["measure"] = "rate_flux"
    return config, system


def reconstructed_momenta(config, system, identity):
    # No PES is evaluated. Native and NumPy integrators use the same sampler;
    # native=False avoids allocating an unused native propagator here.
    dynamics = Dynamics(
        None, system.masses, config["beta"], config["beads"], config["dt_au"],
        seed=np.random.SeedSequence([int(config["seed"]), 307, *identity]),
        native=False, fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
    )
    return dynamics.sample_momenta()


def replay_child(payload):
    """Observe the production sampler's draw, preserving its returned values."""
    task = payload["task"]
    expected = payload["momenta"]
    original_sampler = Dynamics.sample_momenta
    calls = []

    def checked_sample(dynamics):
        momenta = original_sampler(dynamics)
        if not np.array_equal(momenta, expected):
            raise AssertionError("Actual half-dt child momenta differ from the matched original stream")
        calls.append(array_hash(momenta))
        return momenta

    try:
        Dynamics.sample_momenta = checked_sample
        result = _child_task(task)
        if len(calls) != 1:
            raise AssertionError("Fresh paired child must draw its unconstrained momenta exactly once")
        if result["denominator"] != payload["original_denominator"]:
            raise AssertionError("Half-dt flux weight differs from the saved original child")
        result["initial_momenta_sha256"] = calls[0]
        result["initial_geometry_sha256"] = array_hash(task["q"])
        return result
    except Exception as error:
        return {"chain": task["chain"], "parent": task["parent"], "pair": task["pair"],
                "error": {"type": type(error).__name__, "message": str(error),
                          "traceback": traceback.format_exc()}}
    finally:
        Dynamics.sample_momenta = original_sampler


def statistics(children, options, seed, *, indices=False):
    return summarize_recrossing(
        children, rng=np.random.default_rng(np.random.SeedSequence([seed, 503])),
        return_bootstrap_indices=indices,
        **{key: options[key] for key in ("child_steps", "plateau_fraction", "plateau_tolerance", "bootstrap")},
    )


def compact_statistics(result, children, config, options):
    errors = np.array([item["endpoint_energy_error_kbt"] for item in children])
    return {
        "timestep_fs": config["dt_au"] * AU_TIME_FS,
        "child_steps": options["child_steps"],
        "child_time_fs": config["dt_au"] * options["child_steps"] * AU_TIME_FS,
        "kappa": result["kappa"], "kappa_standard_error": result["standard_error"],
        "plateau_detected": result["plateau_detected"],
        "plateau_diagnostics": result["plateau_diagnostics"],
        "parent_block_lengths": result["parent_block_lengths"],
        "uncertainty": result["uncertainty"],
        "maximum_endpoint_energy_error_kbt_per_bead": float(errors.max()),
        "energy_error_quantiles_0_25_50_75_95_100_kbt_per_bead": np.quantile(errors, [0, .25, .5, .75, .95, 1]).tolist(),
        "child_pairs_exceeding_0_1_kbt_per_bead": int(np.count_nonzero(errors > 0.1)),
        "energy_gate_passed": bool(np.all(errors <= 0.1)),
        "energy_error_scope": "maximum absolute initial-to-final energy change of the two opposite-momentum children, divided by beads*kBT",
    }


def replay(source, output, workers):
    started = time.perf_counter()
    if output.exists():
        raise ValueError("Replay output must be a fresh directory")
    config, system = restored_configuration(source)
    if config["beads"] != 128 or system.symbols != ("H", "H", "H"):
        raise ValueError("This diagnostic is scoped to the saved P=128 H3 experiment")
    if config.get("recrossing", {}).get("dynamics", "rpmd") != "rpmd":
        raise ValueError("Energy-conserving RPMD children are required")
    manifest = json.loads((source / "run/manifest.json").read_text())
    if manifest["dependencies"]["numpy"] != np.__version__:
        raise ValueError("Use the original NumPy version for faithful momentum-stream reconstruction")
    original_case = json.loads((source / "case.json").read_text())
    for name in ("src/pyrpmd/workflows/rate.py", "src/pyrpmd/workflows/prepare.py"):
        if original_case["source_sha256"][name] != file_sha256(ROOT / name):
            raise ValueError(f"Original production source changed: {name}")
    pmf = json.loads((source / "run/rate_pmf.json").read_text())
    summary = json.loads((source / "run/summary.json").read_text())
    options = _recrossing_options(config)
    parents = []
    parent_hashes = {}
    for chain in range(options["parent_replicas"]):
        path = source / "run/checkpoints" / f"parent_{chain:04d}.npz"
        parent_hashes[str(path.relative_to(source))] = file_sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            snapshots = saved["snapshots"].copy()
            if len(snapshots) != options["parent_samples"]:
                raise ValueError("Original constrained-parent record is incomplete")
            parents.append({"chain": chain, "snapshots": snapshots,
                            "xi_star": float(saved["xi_star"]), "pmf_method": str(saved["pmf_method"])})
    original_tasks = _child_tasks(parents, config, system, source / "run", options, pmf)
    original_children = []
    child_hashes = {}
    # These calls are cache-only: explicitly require each checkpoint to exist.
    # The production reader verifies config, System, initial q and task identity.
    for task in original_tasks:
        path = source / "run/checkpoints" / f"child_{task['chain']:04d}_{task['parent']:05d}_{task['pair']:04d}.npz"
        if not path.is_file():
            raise ValueError(f"Missing original child; replay will not create it: {path}")
        child_hashes[str(path.relative_to(source))] = file_sha256(path)
        original_children.append(_child_task(task))
    fine = copy.deepcopy(config)
    fine["dt_au"] /= 2
    fine["dynamics"]["timestep"] = f"{fine['dt_au'] * AU_TIME_FS:.17g} fs"
    fine["sampling"]["dt_au"] = fine["dt_au"]
    fine["recrossing"]["child_steps"] = 2 * options["child_steps"]
    fine["parallel"].update(backend="process", trajectory_workers=workers, threads_per_rank=1)
    fine_options = _recrossing_options(fine)
    if fine_options["child_steps"] * fine["dt_au"] != options["child_steps"] * config["dt_au"]:
        raise AssertionError("Physical child propagation time changed")
    fine_tasks = _child_tasks(parents, fine, system, output, fine_options, pmf)
    surface = rate_coordinate(config, system).dividing_surface(pmf["xi_star"])
    payloads, momenta_arrays = [], []
    for old, new, child in zip(original_tasks, fine_tasks, original_children):
        identity = (old["chain"], old["parent"], old["pair"])
        original_p = reconstructed_momenta(config, system, identity)
        new_p = reconstructed_momenta(fine, system, identity)
        if not np.array_equal(old["q"], new["q"]) or not np.array_equal(original_p, new_p):
            raise AssertionError("Original and replay initial-state arrays differ")
        _, gradient, metric = _normal(surface, old["q"], system.masses)
        flux_scale = np.sqrt(metric / (2 * np.pi * config["beta"]))
        velocity = float(np.sum(gradient * original_p.mean(0) / system.masses[:, None]))
        if abs(velocity) / flux_scale != child["denominator"]:
            raise AssertionError("Reconstructed momenta do not reproduce the original saved flux weight")
        momenta_arrays.append(original_p)
        payloads.append({"task": new, "momenta": original_p, "original_denominator": child["denominator"]})
    output.mkdir(parents=True)
    initial_path = output / "matched-initial-states.npz"
    np.savez_compressed(initial_path,
                        ids=np.array([[task[key] for key in ("chain", "parent", "pair")] for task in original_tasks]),
                        q=np.array([task["q"] for task in original_tasks]), p=np.array(momenta_arrays))
    write_json(output / "settings.json", {"original_config": config, "replay_config": fine,
                                          "prepared_system": system.to_dict(), "parent_ensemble_reused": True})
    print(json.dumps({"event": "matched_initial_states_verified", "pairs": len(payloads),
                      "dt_fs": config["dt_au"] * AU_TIME_FS, "half_dt_fs": fine["dt_au"] * AU_TIME_FS,
                      "workers": workers}), flush=True)
    new_children = run_tasks(replay_child, payloads, fine["parallel"])
    failures = [child for child in new_children if child.get("error")]
    if failures:
        write_json(output / "replay-failures.json", failures)
        raise RuntimeError(f"{len(failures)} paired child replays failed; see replay-failures.json")
    original_stats = statistics(original_children, options, config["seed"], indices=True)
    fine_stats = statistics(new_children, fine_options, config["seed"])
    aligned = [{**child, "numerator": np.asarray(child["numerator"])[1::2]} for child in new_children]
    aligned_stats = statistics(aligned, options, config["seed"])
    if abs(original_stats["kappa"] - summary["kappa"]) > 1e-14:
        raise AssertionError("Recomputed original transmission does not match saved summary")
    if not np.array_equal(original_stats["parent_denominator"], aligned_stats["parent_denominator"]):
        raise AssertionError("Parent flux weights changed")
    # Use identical bootstrap parent selections for the paired difference.
    # This conditional diagnostic retains the original estimated block plan.
    delta_parent = aligned_stats["parent_numerator"] - original_stats["parent_numerator"]
    denominator = original_stats["parent_denominator"]
    start = original_stats["start"]
    delta_draws = np.array([
        (delta_parent[ix].sum(0) / denominator[ix].sum())[start:].mean()
        for ix in original_stats["bootstrap_parent_indices"]
    ])
    comparisons = []
    for old, new, task in zip(original_children, new_children, original_tasks):
        comparisons.append({
            "chain": old["chain"], "parent": old["parent"], "pair": old["pair"],
            "original_energy_error_kbt_per_bead": old["endpoint_energy_error_kbt"],
            "halfdt_energy_error_kbt_per_bead": new["endpoint_energy_error_kbt"],
            "original_exceeds_0_1": old["endpoint_energy_error_kbt"] > 0.1,
            "halfdt_exceeds_0_1": new["endpoint_energy_error_kbt"] > 0.1,
            "denominator_bitwise_equal": old["denominator"] == new["denominator"],
            "initial_geometry_sha256": new["initial_geometry_sha256"],
            "initial_momenta_sha256": new["initial_momenta_sha256"],
            "native_propagation_steps": new["propagation"]["native_steps"],
            "python_propagation_steps": new["propagation"]["python_steps"],
        })
    curves = output / "paired-transmission-curves.npz"
    np.savez_compressed(curves,
                        original_time_fs=np.arange(1, options["child_steps"] + 1) * config["dt_au"] * AU_TIME_FS,
                        halfdt_time_fs=np.arange(1, fine_options["child_steps"] + 1) * fine["dt_au"] * AU_TIME_FS,
                        original_kappa=original_stats["curve"], halfdt_kappa=fine_stats["curve"],
                        aligned_halfdt_kappa=aligned_stats["curve"], paired_bootstrap_delta=delta_draws)
    # Confirm that replay did not modify any source parent or child checkpoint.
    for relative, digest in {**parent_hashes, **child_hashes}.items():
        if file_sha256(source / relative) != digest:
            raise AssertionError("Original checkpoint changed during replay")
    source_files = (
        "benchmarks/transition_state_search/recrossing_timestep.py", "src/pyrpmd/workflows/rate.py",
        "src/pyrpmd/dynamics.py", "src/pyrpmd/propagation.py", "src/pyrpmd/analysis/recrossing.py",
        "src/pyrpmd/cv.py", "benchmarks/h3/pes.py",
    )
    return {
        "schema": "pyrpmd.h3-recrossing-halfdt.v1", "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_case": str(source), "raw_replay_output": str(output),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "trajectory_workers": workers, "threads_per_worker": 1},
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
        "original_input_sha256": {name: file_sha256(source / name) for name in (
            "settings.json", "case.json", "run/manifest.json", "run/preparation.json", "run/rate_pmf.json", "run/summary.json")},
        "original_parent_checkpoint_sha256": parent_hashes,
        "original_child_checkpoint_manifest_sha256": hashlib.sha256(json.dumps(child_hashes, sort_keys=True).encode()).hexdigest(),
        "initial_state_archive": {"path": str(initial_path), "sha256": file_sha256(initial_path)},
        "transmission_curve_archive": {"path": str(curves), "sha256": file_sha256(curves)},
        "matching": {
            "all_original_child_fingerprints_verified": True,
            "all_saved_parent_positions_reused_exactly": True,
            "original_and_halfdt_reconstructed_momenta_bitwise_equal": True,
            "all_actual_replay_momentum_draws_checked_bitwise": True,
            "all_original_flux_denominators_reconstructed_exactly": True,
            "all_replay_flux_denominators_equal_original_exactly": True,
            "original_checkpoints_unchanged": True,
            "momenta_provenance": "Original p arrays were not saved. Reconstructed with Dynamics.sample_momenta, recorded seed [914401,307,chain,parent,pair], same NumPy version, checked against original flux denominators; actual replay draws are checked in full.",
        },
        "parent_chains": len(parents), "parent_configurations": sum(len(p["snapshots"]) for p in parents),
        "child_pairs": len(new_children), "child_trajectories": 2 * len(new_children),
        "operational_replay_failures": 0,
        "original": compact_statistics(original_stats, original_children, config, options),
        "halfdt": compact_statistics(fine_stats, new_children, fine, fine_options),
        "matched_time_comparison": {
            "time_alignment": "new numerator[1::2] is at the original 0.1 fs grid; both end at 100 fs",
            "halfdt_kappa_on_original_grid": aligned_stats["kappa"],
            "kappa_difference_halfdt_minus_original": aligned_stats["kappa"] - original_stats["kappa"],
            "paired_difference_standard_error": float(delta_draws.std(ddof=1)),
            "paired_difference_percentile_95_interval": np.quantile(delta_draws, [.025, .975]).tolist(),
            "paired_bootstrap_scope": "Identical parent selections using the original estimated chain/block plan; conditional on that plan describing parent mixing",
            "paired_bootstrap_draws": len(delta_draws),
            "maximum_curve_difference_on_matched_times": float(np.max(np.abs(aligned_stats["curve"] - original_stats["curve"]))),
        },
        "pair_energy_errors_and_matching": comparisons,
        "physical_rate_converged": False,
        "limitations": [
            "Only child timestep changes. The original PMF and constrained-parent ensemble are reused, including their sampling and parent-timestep limitations.",
            "The original energy-gate failure is retained; improved child integration does not establish PMF stationarity or a physical rate.",
            "Reported energy errors are endpoint differences, not maximum drift over the trajectory, and are maxima within each opposite-momentum child pair.",
            "Original full momenta were reconstructed from deterministic streams, not recovered from an original stored p array.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        with threadpool_limits(1):
            result = replay(args.source.resolve(), args.output.resolve(), args.workers)
    except Exception as error:
        write_json(args.evidence, {"schema": "pyrpmd.h3-recrossing-halfdt.v1", "status": "failed",
                                  "physical_rate_converged": False,
                                  "error": {"type": type(error).__name__, "message": str(error),
                                            "traceback": traceback.format_exc()}})
        raise
    write_json(args.evidence, result)
    print(json.dumps({"event": "completed", "evidence": str(args.evidence),
                      "original_max_energy_error_kbt": result["original"]["maximum_endpoint_energy_error_kbt_per_bead"],
                      "halfdt_max_energy_error_kbt": result["halfdt"]["maximum_endpoint_energy_error_kbt_per_bead"],
                      "paired_kappa_change": result["matched_time_comparison"]["kappa_difference_halfdt_minus_original"]}), flush=True)


if __name__ == "__main__":
    main()
