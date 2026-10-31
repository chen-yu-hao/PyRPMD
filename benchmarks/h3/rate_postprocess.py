"""Real BKMP2 H + H2 regression for two-dimensional rate postprocessing.

The joint (xi, r12) histogram is sampled in the explicitly declared xi-flux
measure.  Integrating r12 therefore recovers the same-sample one-dimensional
flux marginal without multiplying by a second thermal velocity.  This is a
short implementation regression, not a converged H + H2 rate calculation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.constants import AU_RATE_TO_CM3_S, AU_TIME_FS
from pyrpmd.input import JobSpec
from pyrpmd.parallel import make_task_potential, run_tasks
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.propagation import loop_options
from pyrpmd.workflows.fes import run_fes, write_json, resolve_umbrellas
from pyrpmd.workflows.prepare import prepare_rate
from pyrpmd.workflows.rate import (
    _normal, _parent_task, _recrossing_options, rate_coordinate,
)

HERE = Path(__file__).resolve().parent


def configurations(output, *, workers=2, profile="quick", library=None):
    if profile not in ("quick", "pilot"):
        raise ValueError("profile must be quick or pilot")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    spec = JobSpec.from_file(HERE / "rate.inp")
    raw = spec.data
    raw["seed"] = 731241
    raw["beads"] = 8 if profile == "quick" else 16
    raw["potential"]["params"] = {
        "library": str(Path(library or HERE / "build/libbkmp2.so").resolve())
    }
    raw["parallel"] = {
        "backend": "process" if workers > 1 else "serial",
        "trajectory_workers": workers, "bead_workers": 1, "threads_per_rank": 1,
    }
    raw["sampling"].update(
        equilibration="20 fs" if profile == "quick" else "100 fs",
        production="100 fs" if profile == "quick" else "300 fs",
        interval="0.5 fs", replicas=2,
    )
    raw["rate"] = {"pmf_method": "mbar", "dividing_surface": 1.0}
    raw["recrossing"].update(
        parent_replicas=2, parent_samples=16 if profile == "quick" else 32,
        child_pairs=2, equilibration="50 fs", parent_interval="5 fs",
        child_time="80 fs", bootstrap=200,
    )
    raw["output"]["directory"] = str(Path(output).resolve() / "dynamics")
    rate_config, system = spec.resolve()
    rate_config["reaction"] = prepare_rate(rate_config, system)["reaction"]

    joint = copy.deepcopy(raw)
    joint["task"] = "fes"
    joint.pop("rate", None)
    joint.pop("recrossing", None)
    joint["cvs"] = [
        {
            "name": "xi", "kind": "rpmdrate_xi", "estimator": "centroid",
            "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]], "equivalent_paths": [[0, 2, 1]],
            "rinf": "10 bohr",
        },
        {"name": "r12", "kind": "distance", "estimator": "centroid",
         "atoms": [1, 2], "unit": "bohr"},
    ]
    auxiliary_centers = [1.4, 1.8] if profile == "quick" else [1.3, 1.6, 1.9]
    joint["umbrella"] = {
        "cvs": ["xi", "r12"],
        "grid": [
            {"start": -0.05, "stop": 1.1, "count": 24}, auxiliary_centers,
        ],
        "stiffness": ["0.4750217345 hartree", "0.15 hartree/bohr^2"],
    }
    # Exact xi-bin centers at 0 and 1 avoid conflating transverse integration
    # with the noncommutation of free-energy interpolation and marginalization.
    # Two unequal transverse bins test density-versus-bin-mass normalization.
    joint["analysis"] = {
        "method": "mbar", "measure": "rate_flux", "flux_axis": 0,
        "bins": [np.linspace(-0.025, 1.125, 24).tolist(), [0.0, 1.55, 4.0]],
        "ranges": [[-0.025, 1.125], [0.0, 4.0]],
        "bootstrap": 0, "min_count": 3,
    }
    joint["output"]["directory"] = str(Path(output).resolve() / "run")
    joint_spec = JobSpec(joint, base_directory=HERE)
    fes_config, joint_system = joint_spec.resolve()
    np.testing.assert_array_equal(joint_system.masses, system.masses)
    return fes_config, rate_config, system


def reference_rate(config, system):
    ma, mb = [system.masses[list(f)].sum() for f in config["reaction"]["fragments"]]
    mu = ma * mb / (ma + mb)
    rinf = config["reaction"]["rinf"]
    return float(4 * np.pi * rinf**2 / np.sqrt(2 * np.pi * config["beta"] * mu))


def surface_identity(config, system):
    """Bind channel records to the actual PES and complete rate definition."""
    library = Path(config["potential"]["params"]["library"])
    build = json.loads(library.with_suffix(".build.json").read_text())
    pinned = json.loads((HERE / "source.json").read_text())
    binary_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    if build["library_sha256"] != binary_sha or build["source"]["sha256"] != pinned["sha256"]:
        raise ValueError("BKMP2 identity does not match the checked source/library provenance")
    payload = {
        "schema": "pyrpmd.h3.surface-physics.v1",
        "coordinate": {
            "kind": "centroid_rpmdrate_xi", "formula": "xi=s0/(s0-s1)",
            "dividing_surface": 1.0,
            "reaction": config["reaction"],
            "sampling_measure": "xi_rate_flux", "flux_axis": 0,
            "equivalent_paths_already_explicit": True,
        },
        "system": system.to_dict(),
        "temperature_K": config["temperature_K"], "beta_au": config["beta"],
        "beads": config["beads"],
        "timestep_au": config["dt_au"],
        "integrator": config.get("dynamics", {}).get("integrator", "cayley-bcb"),
        "potential": {
            "model": pinned["model"], "pinned_source_commit": pinned["commit"],
            "pinned_source_sha256": pinned["sha256"], "library_sha256": binary_sha,
            "wrapper_sha256": build["wrapper_sha256"],
        },
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity, payload


def independent_marginal_check(data, streams, *, beta, reference_rate_au):
    """Direct weighted histogram oracle, independent of F -> flux integration."""
    samples = np.concatenate([s["samples"] for s in streams])
    weights = np.asarray(data["weights"])
    xedge, yedge = data["edges"]
    keep = (samples[:, 1] >= yedge[0]) & (samples[:, 1] <= yedge[-1])
    # A direct bin sum avoids cancellation from differences of cumulative
    # weighted sums in numpy.histogram when barrier probabilities are small.
    index = np.searchsorted(xedge, samples[:, 0], side="right") - 1
    index[samples[:, 0] == xedge[-1]] = len(xedge) - 2
    inside = keep & (index >= 0) & (index < len(xedge) - 1)
    mass = np.bincount(index[inside], weights=weights[inside], minlength=len(xedge) - 1)
    density = mass / np.diff(xedge)
    centers = (xedge[:-1] + xedge[1:]) / 2
    indices = [int(np.argmin(abs(centers - s))) for s in (0.0, 1.0)]
    if any(abs(centers[i] - s) > 1e-12 for i, s in zip(indices, (0.0, 1.0))):
        raise ValueError("benchmark requires reference/surface at exact bin centers")
    ref, surf = indices
    if min(density[ref], density[surf]) <= 0:
        raise ValueError("direct marginal has no endpoint support")
    marginal = np.sum(data["density"] * np.diff(yedge)[None, :], axis=1)
    np.testing.assert_allclose(marginal, density, rtol=2e-12, atol=1e-15)
    ratio = float(density[surf] / density[ref])
    return {
        "k_tst_au": float(reference_rate_au * ratio),
        "k_tst_cm3_s": float(reference_rate_au * ratio * AU_RATE_TO_CM3_S),
        "free_energy_barrier_hartree": float(-np.log(ratio) / beta),
        "reference_bin": ref, "surface_bin": surf,
        "reference_density": float(density[ref]), "surface_density": float(density[surf]),
        "maximum_density_marginal_error": float(np.max(abs(marginal - density))),
        "transverse_sample_fraction_outside": float(np.mean(~keep)),
        "transverse_weight_fraction_outside": float(weights[~keep].sum()),
        "scope": "Algebraically independent same-trajectory weighted histogram; not independent physical sampling",
    }


def channel_indicators(centroid, product_side):
    """Partition the union product side without adding a symmetry factor."""
    centroid = np.asarray(centroid, float)
    side = np.asarray(product_side, bool)
    if centroid.ndim != 3 or centroid.shape[1:] != (3, 3) or side.shape != (len(centroid),):
        raise ValueError("expected [time,3,3] H3 centroids and one product-side flag per time")
    if not np.isfinite(centroid).all():
        raise ValueError("nonfinite H3 centroid")
    r01 = np.linalg.norm(centroid[:, 0] - centroid[:, 1], axis=1)
    r02 = np.linalg.norm(centroid[:, 0] - centroid[:, 2], axis=1)
    exit01 = r01 <= r02
    return np.column_stack((side & exit01, side & ~exit01))


def _channel_child(task):
    """Paired signed flux resolved into the two atom-labelled H3 exits."""
    config, system = task["config"], task["system"]
    surface = rate_coordinate(config, system).dividing_surface(1.0)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            provider, system.masses, config["beta"], config["beads"], config["dt_au"],
            seed=np.random.SeedSequence([config["seed"], 307, task["chain"], task["parent"], task["pair"]]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=1, **loop_options(config),
        )
        q = np.asarray(task["q"])
        p = dynamics.sample_momenta()
        residual, gradient, metric = _normal(surface, q, system.masses)
        if abs(residual) > 1e-8:
            raise ValueError("H3 child does not start on xi=1 dividing surface")
        fs = np.sqrt(metric / (2 * np.pi * config["beta"]))
        velocity = float(np.sum(gradient * p.mean(0) / system.masses[:, None]))
        denominator = abs(velocity) / fs
        numerator = np.zeros((task["child_steps"], 2))
        max_error = 0.0
        positive_endpoints = uncommitted_endpoints = 0
        positive_endpoint_weight = uncommitted_endpoint_weight = 0.0
        for sign in (1.0, -1.0):
            state = State(q.copy(), sign * p)
            initial = dynamics.energy(state)
            block = dynamics.propagate_block(
                state, task["child_steps"], cv=surface, check_energy=True,
            )
            centroid = block.centroids
            product_side = block.values[:, 0] > 0
            weight = sign * velocity / fs
            numerator += weight * channel_indicators(centroid, product_side)
            error = abs(dynamics.energy(state) - initial) * config["beta"] / config["beads"]
            max_error = max(max_error, float(error))
            if product_side[-1]:
                positive_endpoints += 1
                positive_endpoint_weight += abs(weight)
                last = centroid[-1]
                partner = 1 if np.linalg.norm(last[0] - last[1]) <= np.linalg.norm(last[0] - last[2]) else 2
                spectator = 3 - partner
                bond = float(np.linalg.norm(last[0] - last[partner]))
                separation = last[spectator] - 0.5 * (last[0] + last[partner])
                velocities = state.p.mean(axis=0) / system.masses[:, None]
                relative_velocity = velocities[spectator] - 0.5 * (velocities[0] + velocities[partner])
                committed = bond < 2.5 and np.linalg.norm(separation) > 5.0 and np.dot(separation, relative_velocity) > 0
                if not committed:
                    uncommitted_endpoints += 1
                    uncommitted_endpoint_weight += abs(weight)
        return {
            "chain": task["chain"], "parent": task["parent"], "pair": task["pair"],
            "numerator": numerator, "denominator": denominator,
            "endpoint_error_kbt": max_error,
            "positive_endpoints": positive_endpoints,
            "uncommitted_endpoints": uncommitted_endpoints,
            "positive_endpoint_abs_flux_weight": positive_endpoint_weight,
            "uncommitted_endpoint_abs_flux_weight": uncommitted_endpoint_weight,
        }
    finally:
        if hasattr(provider, "close"):
            provider.close()


def channel_dynamics(config, system, streams, output, k_tst_au):
    from pyrpmd.analysis.rate_postprocess import channel_rates
    from pyrpmd.analysis.recrossing import summarize_recrossing

    options = _recrossing_options(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parents = []
    for chain in range(options["parent_replicas"]):
        candidates = [s for s in streams if s["replica"] == chain % 2]
        source = min(candidates, key=lambda s: abs(s["center"][0] - 1.0) + 0.1 * abs(s["center"][1] - 1.757))
        parents.append({"config": config, "system": system, "options": options,
                        "chain": chain, "xi_star": 1.0, "initial_q": source["last_q"],
                        "output": str(output)})
    parents = run_tasks(_parent_task, parents, config.get("parallel", {}))
    tasks = [
        {"config": config, "system": system, "chain": parent["chain"],
         "parent": index, "pair": pair, "q": q, "child_steps": options["child_steps"]}
        for parent in parents for index, q in enumerate(parent["snapshots"])
        for pair in range(options["child_pairs"])
    ]
    children = run_tasks(_channel_child, tasks, config.get("parallel", {}))
    grouped = {}
    for child in children:
        key = (child["chain"], child["parent"])
        item = grouped.setdefault(key, [np.zeros((options["child_steps"], 2)), 0.0])
        item[0] += child["numerator"]
        item[1] += child["denominator"]
    ids = sorted(grouped)
    numerator = np.asarray([grouped[k][0] for k in ids])
    denominator = np.asarray([grouped[k][1] for k in ids])
    time_au = np.arange(1, options["child_steps"] + 1) * config["dt_au"]
    identity, identity_payload = surface_identity(config, system)
    write_json(output / "surface-identity.json", {"surface_identity": identity, "payload": identity_payload})
    np.savez_compressed(
        output / "channel_recrossing.npz", time_au=time_au,
        parent_numerator=numerator, parent_denominator=denominator,
        parent_ids=np.asarray(ids), channel_names=np.asarray(["H0H1+H2", "H0H2+H1"]),
        schema=np.asarray("pyrpmd.channel_flux"), schema_version=np.asarray(1),
        estimator=np.asarray("signed_flux_side"), time_unit=np.asarray("atomic"),
        surface_identity=np.asarray(identity),
        surface_identity_payload=np.asarray(json.dumps(identity_payload, sort_keys=True)),
    )
    result = channel_rates(
        k_tst_au, time_au, numerator, denominator, np.asarray(ids),
        ["H0H1+H2", "H0H2+H1"], plateau_fraction=options["plateau_fraction"],
        plateau_tolerance=options["plateau_tolerance"], bootstrap=options["bootstrap"],
        seed=731242, source_identity=identity, flux_identity=identity,
    )
    total = summarize_recrossing(
        [dict(c, numerator=c["numerator"].sum(axis=1)) for c in children],
        child_steps=options["child_steps"], plateau_fraction=options["plateau_fraction"],
        plateau_tolerance=options["plateau_tolerance"], bootstrap=0,
        rng=np.random.default_rng(731243),
    )
    total_curve = numerator.sum(axis=(0, 2)) / denominator.sum()
    np.testing.assert_allclose(total_curve, total["curve"], rtol=2e-13, atol=2e-14)
    write_json(output / "channel-rates.json", result)
    return {
        "rates": result,
        "single_channel_kappa": total["kappa"],
        "single_channel_plateau_detected": total["plateau_detected"],
        "single_channel_candidate_rate_cm3_s": k_tst_au * total["kappa"] * AU_RATE_TO_CM3_S,
        "parent_configurations": len(ids), "child_trajectories": 2 * len(children),
        "maximum_endpoint_error_kbt": max(c["endpoint_error_kbt"] for c in children),
        "sum_channel_curve_equals_existing_single_channel_estimator": True,
        "classification": "Mutually exclusive signed product-side indicators r12>min(r01,r02), labelled by shorter forming bond; finite-time RPMD flux-side definition",
        "symmetry_note": "Two exchange arrangements are already explicit. No extra factor of two and no imposed 50:50 averaging.",
        "surface_identity": identity,
        "surface_identity_payload": identity_payload,
        "terminal_commitment": {
            "bond_threshold_bohr": 2.5, "separation_threshold_bohr": 5.0,
            "requires_outward_motion": True,
            "positive_side_endpoint_count": sum(c["positive_endpoints"] for c in children),
            "not_committed_endpoint_count": sum(c["uncommitted_endpoints"] for c in children),
            "positive_side_endpoint_abs_flux_weight": sum(c["positive_endpoint_abs_flux_weight"] for c in children),
            "not_committed_endpoint_abs_flux_weight": sum(c["uncommitted_endpoint_abs_flux_weight"] for c in children),
            "scope": "Endpoint separation diagnostic only; no persistence or long-time commitment proof. Reported branches are finite-time union-side partitions, not validated asymptotic product branching fractions.",
        },
        "artifact": str(output / "channel_recrossing.npz"),
    }


def run_benchmark(output, *, workers=2, profile="quick", library=None, skip_dynamics=False):
    from pyrpmd.analysis.rate_postprocess import flux_slice_rate_2d

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, rate_config, system = configurations(output, workers=workers, profile=profile, library=library)
    write_json(output / "run/input/resolved.json", config)
    write_json(output / "system.json", system.to_dict())
    print(json.dumps({"event": "start", "profile": profile, "beads": config["beads"],
                      "streams": len(resolve_umbrellas(config, system)[2]) * 2}), flush=True)
    existing_checkpoints = len(list((output / "run/checkpoints").glob("window_*.npz")))
    started = time.perf_counter()
    sampled = run_fes(config, system, output / "run", return_data=True)
    elapsed = time.perf_counter() - started
    data, streams = sampled["data"], sampled["streams"]
    kref = reference_rate(rate_config, system)
    spec = {
        "method": "flux_slice_2d", "surface": 1.0, "reference": 0.0,
        "reference_rate_au": kref,
        "reference_definition": "Free H + H2 incoming sphere at Rinf=10 bohr; 4*pi*Rinf^2/sqrt(2*pi*beta*mu)",
        "beta": config["beta"], "flux_axis": 0,
    }
    write_json(output / "rate-postprocess-spec.json", spec)
    result = flux_slice_rate_2d(
        data["free_energy"], data["edges"], beta=config["beta"],
        surface=1.0, reference=0.0, reference_rate_au=kref, mask=data["mask"],
        measure="rate_flux", flux_axis=0,
    )
    oracle = independent_marginal_check(data, streams, beta=config["beta"], reference_rate_au=kref)
    relative_error = abs(result["k_tst_au"] / oracle["k_tst_au"] - 1)
    np.testing.assert_allclose(result["k_tst_au"], oracle["k_tst_au"], rtol=2e-12, atol=0)
    write_json(output / "static-rate.json", result)
    report = {
        "schema": "pyrpmd.h3.rate-postprocess-regression.v1", "passed": True,
        "converged": False, "profile": profile, "temperature_K": config["temperature_K"],
        "beads": config["beads"], "timestep_fs": config["dt_au"] * AU_TIME_FS,
        "sampling_seconds": elapsed, "sampling": sampled["summary"],
        "timing_scope": "Current run_fes call, including checkpoint reuse/reanalysis when present; not aggregate wall time across interrupted attempts",
        "existing_window_checkpoints_at_start": existing_checkpoints,
        "surface": result, "independent_same_sample_1d_marginal": oracle,
        "relative_2d_vs_1d_error": relative_error,
        "symmetry_factor": 1.0,
        "scope": "Real BKMP2 ring-polymer sampling and rate postprocessing regression. Finite transverse domain [0,4] bohr and coarse two-bin transverse resolution; no bead, timestep, Rinf, tail, statistical or exact-quantum convergence claim.",
        "uncertainty": "PMF bootstrap disabled for this short regression; any dynamic interval is conditional on the measured static prefactor.",
    }
    if not skip_dynamics:
        print(json.dumps({"event": "start_channel_dynamics", "k_tst_cm3_s": result["k_tst_cm3_s"]}), flush=True)
        report["dynamics"] = channel_dynamics(rate_config, system, streams, output / "dynamics", result["k_tst_au"])
        report["passed"] = report["dynamics"]["maximum_endpoint_error_kbt"] < 0.1
        spec.update(channel_flux="../dynamics/channel_recrossing.npz",
                    surface_identity=report["dynamics"]["surface_identity"])
        write_json(output / "rate-postprocess-spec.json", spec)
    from pyrpmd.analysis.rate_io import postprocess_rate
    report["end_to_end_postprocess"] = postprocess_rate(
        output / "run", spec=output / "rate-postprocess-spec.json",
        output=output / "rate-postprocess.json", bootstrap=200, seed=731242,
    )
    write_json(output / "benchmark.json", report)
    print(json.dumps({"event": "complete", "passed": report["passed"],
                      "output": str(output), "k_tst_cm3_s": result["k_tst_cm3_s"],
                      "relative_2d_vs_1d_error": relative_error}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--profile", choices=("quick", "pilot"), default="quick")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--skip-dynamics", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(args.output, workers=args.workers, profile=args.profile,
                           library=args.library, skip_dynamics=args.skip_dynamics)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
