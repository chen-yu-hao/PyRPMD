"""Execute a fixed, independent-seed Ni-H convergence study after acceptance.

This is deliberately separate from the quick implementation gate. It always
retains failed arms; no point estimate, partial result or apparent speedup can
change the registered matrix. Passing its finite checks still requires a
scientific assessment of the potential and material before publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.materials.batch_acceptance import (
    configuration, execute, obtain_model, read_streams, source_identity, write_json,
)

MARGINS = {"potential_eV_per_atom": 0.005, "internal_energy_eV_per_atom": 0.005,
           "hydrogen_ring_radius_squared_bohr2": 0.002}


def confidence_interval(values):
    """Independent replicas, never saved frames, define the uncertainty unit."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        return {"available": False, "reason": "at_least_four_finite_independent_replica_means_required"}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(t.ppf(.975, len(values)-1) * se)
    return {"available": True, "replicas": len(values), "mean": mean,
            "standard_error": se, "CI95": [mean-half, mean+half], "halfwidth": half}


def equivalence(left, right, margin):
    """Welch interval for separate-seed arms; broad overlap is not a pass."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if not confidence_interval(a)["available"] or not confidence_interval(b)["available"]:
        return {"passed": False, "reason": "insufficient_independent_replicas", "margin": margin}
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    variance = va + vb
    df = variance**2 / (va**2/(len(a)-1) + vb**2/(len(b)-1)) if variance > 0 else float("inf")
    half = float(t.ppf(.975, df) * np.sqrt(variance)) if variance > 0 else 0.0
    delta = float(b.mean()-a.mean())
    interval = [delta-half, delta+half]
    return {"passed": bool(interval[0] >= -margin and interval[1] <= margin),
            "difference_control_minus_base": delta, "CI95": interval, "margin": margin,
            "method": "independent-seed Welch t interval", "degrees_of_freedom": None if not np.isfinite(df) else float(df)}


def observations(directory):
    from pyrpmd.analysis.diagnostics import correlated_mean_diagnostics
    from pyrpmd.constants import EV_TO_AU

    _, system, streams = read_streams(directory)
    if not streams or not all(s["complete"] for s in streams):
        raise ValueError("Convergence analysis requires every registered replica to finish")
    hydrogen = [i for i, s in enumerate(system.symbols) if s in ("H", "D", "T")]
    series = {
        "potential_eV_per_atom": [s["observables"][:, 0] / EV_TO_AU / system.natoms for s in streams],
        "internal_energy_eV_per_atom": [s["observables"][:, 3] / EV_TO_AU / system.natoms for s in streams],
        "hydrogen_ring_radius_squared_bohr2": [s["radii_gyration_squared"][:, hydrogen].mean(axis=1) for s in streams],
    }
    result = {}
    for name, records in series.items():
        means = [float(a.mean()) for a in records]
        differences = [float(a[len(a)//2:].mean()-a[:len(a)//2].mean()) for a in records]
        interval, stationarity = confidence_interval(means), confidence_interval(differences)
        margin = MARGINS[name]
        diagnostics = [correlated_mean_diagnostics(a) for a in records]
        result[name] = {
            "replica_means": means, "independent_replica_statistics": interval,
            "production_half_difference": stationarity,
            "precision_passed": bool(interval["available"] and interval["halfwidth"] <= margin/2),
            "stationarity_passed": bool(stationarity["available"] and
                                         stationarity["CI95"][0] >= -margin and stationarity["CI95"][1] <= margin),
            "correlation_diagnostics": diagnostics,
            "effective_sampling_passed": all(d["effective_samples"] >= 20 and d["complete_blocks"] >= 4 for d in diagnostics),
        }
    return result


def protocol(*, replicas=4, beads=4, dt_fs=.1, equilibration_steps=2000, production_steps=10000):
    for name, value, minimum in (("replicas", replicas, 4), ("beads", beads, 2),
                                 ("equilibration_steps", equilibration_steps, 4),
                                 ("production_steps", production_steps, 16)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(dt_fs) or dt_fs <= 0:
        raise ValueError("dt_fs must be finite and positive")
    base = {"replicas": replicas, "beads": beads, "dt_fs": dt_fs,
            "equilibration_steps": equilibration_steps, "production_steps": production_steps,
            "lateral_repeat": 1, "sample_interval": 2}
    overrides = {
        "base": {}, "beads": {"beads": 2*beads},
        # Preserve physical preparation, production and recording times.
        "timestep": {"dt_fs": dt_fs/2, "equilibration_steps": 2*equilibration_steps,
                     "production_steps": 2*production_steps, "sample_interval": 4},
        # Replicate the complete slab+adsorbate cell: identical coverage and
        # layer count, rather than changing H coverage with the cell size.
        "cell_size": {"lateral_repeat": 2},
        "preparation": {"equilibration_steps": 2*equilibration_steps},
        "production_length": {"production_steps": 2*production_steps},
        "replica_count": {"replicas": 2*replicas},
    }
    return {
        "schema": "pyrpmd.materials-batch-convergence-protocol.v1",
        "arms": {name: {**base, **change, "seed": 8261 + 1009*index}
                 for index, (name, change) in enumerate(overrides.items())},
        "targets": MARGINS, "confidence": .95,
        "uncertainty_unit": "independent replica mean; distinct root seed in every arm",
        "stopping": "complete all registered arms; never stop or select using observed values",
        "comparison": "each control versus base, entire Welch CI inside the predeclared absolute margin",
        "additional_gates": ["all arms complete", "all production-half CI within margins",
                             "all independent-replica CI halfwidths <= half-margin",
                             "all per-replica effective-sample estimates >=20 and >=4 complete blocks"],
        "limitations": ["Finite single-material study, no potential-model accuracy certification.",
                        "Correlation-time estimates and stationarity screens are diagnostics, not proof of equilibrium.",
                        "The PIMD arm assesses equilibrium observables; PIMD clocks are not transport time.",
                        "This fixed study does not establish a thermodynamic limit or a literature diffusion coefficient."],
    }


def run_convergence(model, output, specification):
    model, output = obtain_model(model), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty convergence output directory")
    output.mkdir(parents=True, exist_ok=True)
    # Freeze all scientific choices before the first Context/PES evaluation.
    write_json(output / "protocol.json", specification)
    report = {"schema": "pyrpmd.materials-batch-convergence.v1", "physical_converged": False,
              "convergence_checks_passed": False, "all_arms_complete": False,
              "source_at_start": source_identity(), "arms": {}, "comparisons": {}}
    write_json(output / "report.json", report)
    try:
        for name, options in specification["arms"].items():
            config = configuration(model, "pimd", mode="process_batch", **options)
            config["output"].update(checkpoint_steps=200, chunk_frames=128)
            measured = execute(config, output / name, output / "evidence" / name)
            observed = observations(output / name)
            report["arms"][name] = {"configuration": config, "measurement": measured, "observations": observed}
            write_json(output / "report.json", report)
        base = report["arms"]["base"]["observations"]
        for name, arm in report["arms"].items():
            if name == "base":
                continue
            report["comparisons"][name] = {
                metric: equivalence(base[metric]["replica_means"], arm["observations"][metric]["replica_means"], margin)
                for metric, margin in specification["targets"].items()}
        report["all_arms_complete"] = True
        report["convergence_checks_passed"] = all(
            check["passed"] for arm in report["comparisons"].values() for check in arm.values()
        ) and all(
            observable[gate] for arm in report["arms"].values() for observable in arm["observations"].values()
            for gate in ("precision_passed", "stationarity_passed", "effective_sampling_passed")
        )
        report["source_at_end"] = source_identity()
        if report["source_at_start"] != report["source_at_end"]:
            raise AssertionError("Scientific source changed during the registered study")
    except BaseException as error:
        report["convergence_checks_passed"] = False
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--dt-fs", type=float, default=.1)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--production-steps", type=int, default=10000)
    args = parser.parse_args()
    choices = vars(args).copy()
    model, output = choices.pop("model"), choices.pop("output")
    result = run_convergence(model, output, protocol(**choices))
    print(json.dumps({"all_arms_complete": result["all_arms_complete"],
                      "convergence_checks_passed": result["convergence_checks_passed"],
                      "physical_converged": False, "report": str(output / "report.json")}))
    # An inconclusive or failed scientific check is observable by automation.
    if not result["convergence_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
