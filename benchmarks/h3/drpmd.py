"""Physical H + H2 direct RPMD benchmark on the BKMP2 H3 potential.

Use as a Python input (``pyrpmd run benchmarks/h3/drpmd.py``), or run this
script for diagnostics, numerical gates and an automatic completed-run restart:

    python benchmarks/h3/drpmd.py --output runs/h3-drpmd --workers 2

The finite sample, bead count, impact cutoff and parent spacing make this a
physical integration benchmark, not a converged thermal-rate determination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.constants import AU_TIME_FS, KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import JobSpec
from pyrpmd.runner import resume_run, run
from pyrpmd.potentials import make_potential

HERE = Path(__file__).resolve().parent
H_MASS_AMU = 1.00782503223
H2_BOND_BOHR = 1.4014719755
HARTREE_KCAL_MOL = 627.5094740631
PAIRS = ((0, 1), (0, 2), (1, 2))


def _h3_bonded_pair(centroid, bond_cutoff):
    short = [pair for pair in PAIRS
             if np.linalg.norm(centroid[pair[0]] - centroid[pair[1]]) < bond_cutoff]
    return short[0] if len(short) == 1 else None


def _original_h3_methods(classifier):
    return type(classifier) is H3ExchangeClassifier and all(
        getattr(getattr(classifier, name, None), "__func__", None) is function
        for name, function in _H3_CLASSIFIER_METHODS.items()
    )


class H3ExchangeClassifier:
    """Identify H2 by its atom membership after atom--diatom separation.

    Exactly one centroid pair must be shorter than ``bond_cutoff_bohr``. The
    remaining atom must be farther than ``exit_distance_bohr`` from the pair's
    mass-weighted centroid and moving outward. The workflow additionally holds
    this label for the configured persistence interval before accepting it.
    """

    def __init__(self, system, bond_cutoff_bohr=2.5, exit_distance_bohr=10.0):
        get = (
            system.get if isinstance(system, dict) else lambda key: getattr(system, key)
        )
        if tuple(get("symbols")) != ("H", "H", "H"):
            raise ValueError(
                "The H3 benchmark classifier requires three hydrogen atoms"
            )
        if tuple(tuple(x) for x in get("fragments")) != ((0,), (1, 2)):
            raise ValueError(
                "Incoming atom is 0 and the original H2 atoms must be [1,2]"
            )
        self.masses = np.asarray(get("masses"), dtype=float)
        self.bond_cutoff = float(bond_cutoff_bohr)
        self.exit_distance = float(exit_distance_bohr)
        if not 0 < self.bond_cutoff < self.exit_distance:
            raise ValueError("Require 0 < bond cutoff < exit distance")

    def bonded_pair(self, q):
        centroid = np.asarray(q).mean(axis=0)
        return _h3_bonded_pair(centroid, self.bond_cutoff)

    def validate_reactants(self, q):
        if self.bonded_pair(q) != (1, 2):
            raise ValueError("Thermal H + H2 parent lost the original H2 bond [1,2]")

    def diagnostics(self, state):
        builtin = _original_h3_methods(self)
        if builtin:
            q = state.q.mean(axis=0)
            pair = _h3_bonded_pair(q, self.bond_cutoff)
        else:
            pair = self.bonded_pair(state.q)
        if pair is None:
            return {"pair": None, "separation_bohr": None, "radial_velocity_au": None}
        atom = next(i for i in range(3) if i not in pair)
        if not builtin:
            q = state.q.mean(axis=0)
        p = state.p.mean(axis=0)
        pair_mass = self.masses[list(pair)].sum()
        center = (
            np.sum(q[list(pair)] * self.masses[list(pair), None], axis=0) / pair_mass
        )
        pair_velocity = p[list(pair)].sum(axis=0) / pair_mass
        displacement = q[atom] - center
        separation = float(np.linalg.norm(displacement))
        radial_velocity = float(
            np.dot(displacement, p[atom] / self.masses[atom] - pair_velocity)
            / separation
        )
        return {
            "pair": list(pair),
            "separation_bohr": separation,
            "radial_velocity_au": radial_velocity,
            "bond_bohr": float(np.linalg.norm(q[pair[0]] - q[pair[1]])),
        }

    def __call__(self, state):
        details = self.diagnostics(state)
        if (
            details["pair"] is None
            or details["separation_bohr"] < self.exit_distance
            or details["radial_velocity_au"] <= 0
        ):
            return None
        pair = tuple(details["pair"])
        if pair == (1, 2):
            return {"status": "nonreactive", "channel": None}
        return {"status": "product", "channel": f"H2[{pair[0]},{pair[1]}]+H"}


    def native_outcome_descriptor(self):
        """Enable native checks only while every benchmark classification hook is unchanged."""
        if not _original_h3_methods(self):
            return None
        try:
            masses = np.asarray(self.masses, dtype=float)
            if (masses.shape != (3,) or not np.isfinite(masses).all()
                    or (masses <= 0).any()
                    or not np.isfinite([self.bond_cutoff, self.exit_distance]).all()
                    or not 0 < self.bond_cutoff < self.exit_distance):
                return None
            return {
                "kind": "h3_exchange", "masses": masses.tolist(),
                "bond_cutoff": float(self.bond_cutoff),
                "exit_distance": float(self.exit_distance),
                "channel_labels": ["H2[0,1]+H", "H2[0,2]+H"],
            }
        except (AttributeError, TypeError, ValueError):
            return None


_H3_CLASSIFIER_METHODS = {
    name: getattr(H3ExchangeClassifier, name)
    for name in ("__call__", "bonded_pair", "diagnostics", "validate_reactants")
}


def build_classifier(system, **params):
    return H3ExchangeClassifier(system, **params)


def job_mapping(
    *,
    trajectories=1024,
    workers=2,
    parents=8,
    beads=16,
    temperature=1500.0,
    timestep_fs=0.1,
    max_time_fs=1200.0,
    equilibration_fs=500.0,
    sample_interval_fs=50.0,
    bmax_bohr=3.0,
    separation_bohr=10.0,
    seed=81073,
):
    """Serializable public input; all distances below are explicitly atomic units."""
    r0, bond = float(separation_bohr), H2_BOND_BOHR
    geometry = [
        [0.0, 0.0, -2 * r0 / 3],
        [-bond / 2, 0.0, r0 / 3],
        [bond / 2, 0.0, r0 / 3],
    ]
    return {
        "task": "drpmd",
        "preset": "pilot",
        "temperature": f"{temperature} K",
        "beads": int(beads),
        "seed": int(seed),
        "native": "required",
        "system": {
            "symbols": ["H", "H", "H"],
            "geometry": geometry,
            "geometry_unit": "bohr",
            "masses_amu": [H_MASS_AMU] * 3,
            "fragments": [[0], [1, 2]],
            "separation": f"{separation_bohr} bohr",
            "multiplicity": 2,
        },
        "potential": {
            "provider": "python",
            "factory": str(HERE / "pes.py") + ":build",
            "params": {"library": str(HERE / "build" / "libbkmp2.so")},
        },
        "dynamics": {"integrator": "cayley-bcb", "timestep": f"{timestep_fs} fs"},
        "sampling": {
            "parents": int(parents),
            "equilibration": f"{equilibration_fs} fs",
            "interval": f"{sample_interval_fs} fs",
            "andersen_interval": round(5.0 / timestep_fs),
            "integrator": "cayley-bcb-andersen",
            "thermostat": "andersen",
        },
        "collision": {
            "trajectories": int(trajectories),
            "bmax": f"{bmax_bohr} bohr",
            "sampling": "thermal_flux",
            "impact_sampling": "uniform_area",
            "orientation": "isotropic",
            "max_time": f"{max_time_fs} fs",
        },
        "outcomes": {
            "factory": str(Path(__file__).resolve()) + ":build_classifier",
            "params": {"bond_cutoff_bohr": 2.5, "exit_distance_bohr": separation_bohr},
            "check_interval": max(1, round(0.5 / timestep_fs)),
            "persistence": "5 fs",
        },
        "reaction": {"electronic_weight": 1.0},
        "parallel": {
            "backend": "process" if workers > 1 else "serial",
            "mode": "serial",
            "trajectory_workers": int(workers),
            "threads_per_rank": 1,
        },
        "output": {
            "directory": str(HERE / "runs" / "drpmd-input"),
            "checkpoint_interval": "50 fs",
            "trajectory_stride": 10,
        },
    }


def build():
    return job_mapping()


def check_classifier_contract(system):
    """Verify all three atom memberships and the incoming/unresolved decisions."""
    classifier = H3ExchangeClassifier(system, exit_distance_bohr=10.0)
    checked = []
    for pair in PAIRS:
        atom = next(i for i in range(3) if i not in pair)
        q, p = np.zeros((1, 3, 3)), np.zeros((1, 3, 3))
        q[0, pair[0], 0], q[0, pair[1], 0] = -0.7, 0.7
        q[0, atom, 2] = 11.0
        p[0, atom, 2] = system.masses[atom] * 0.001
        state = State(q, p)
        answer = classifier(state)
        expected = "nonreactive" if pair == (1, 2) else "product"
        if answer is None or answer["status"] != expected:
            raise AssertionError(f"H2 membership classifier failed for {pair}")
        checked.append({"pair": list(pair), "classification": answer})
        state.p *= -1
        if classifier(state) is not None:
            raise AssertionError(
                "Incoming fragments must not count as resolved outgoing products"
            )
    stretched = State(
        np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]]),
        np.zeros((1, 3, 3)),
    )
    if classifier(stretched) is not None:
        raise AssertionError(
            "Three separated atoms must stay unresolved by this exchange classifier"
        )
    return checked


def _wilson(count, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = count / total
    denom = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [float(max(0.0, midpoint - half)), float(min(1.0, midpoint + half))]


def analyze_raw(raw_directory, summary, config, system):
    raw = Path(raw_directory)
    classifier = H3ExchangeClassifier(system, **config["outcomes"]["params"])
    records = summary["records"]
    drifts, endpoint_drifts, h2_bonds, temperatures, axes = [], [], [], [], []
    final_states, violations = [], []
    for record in records:
        tid = record["trajectory_id"]
        with np.load(
            raw / "trajectories" / f"trajectory-{tid:08d}.npz", allow_pickle=False
        ) as data:
            state = State(data["final_q"], data["final_p"])
            label = classifier(state)
            if record["status"] in ("product", "nonreactive") and label != {
                "status": record["status"],
                "channel": record["channel"],
            }:
                violations.append(tid)
            details = classifier.diagnostics(state)
            final_states.append(
                {"trajectory_id": tid, "status": record["status"], **details}
            )
        value = record["max_energy_drift_per_bead"]
        if value is not None:
            drifts.append(float(value))
        if (
            record["initial_energy_per_bead"] is not None
            and record["final_energy_per_bead"] is not None
        ):
            endpoint_drifts.append(
                abs(record["final_energy_per_bead"] - record["initial_energy_per_bead"])
            )
        with np.load(
            raw / "states" / f"initial-{tid:08d}.npz", allow_pickle=False
        ) as data:
            thermal_q, thermal_p = data["thermal_q"], data["thermal_p"]
            h2_bonds.append(
                float(
                    np.linalg.norm(
                        thermal_q.mean(axis=0)[1] - thermal_q.mean(axis=0)[2]
                    )
                )
            )
            kinetic = 0.5 * float(np.sum(thermal_p**2 / system.masses[None, :, None]))
            degrees = (
                3 * len(system.symbols) * config["beads"] - 6
            )  # Two fixed centroid COMs.
            temperatures.append(2 * kinetic / (degrees * config["beads"] * KB_AU))
            bond = data["q"].mean(axis=0)[1] - data["q"].mean(axis=0)[2]
            axes.append((bond / np.linalg.norm(bond)) ** 2)
    bmax = config["collision"]["bmax_au"]
    impacts = np.array([r["launch"]["impact_parameter_au"] for r in records])
    energies = np.array([r["launch"]["collision_energy_au"] for r in records])
    products = [r for r in records if r["status"] == "product"]
    shell = [r for r in records if r["launch"]["impact_parameter_au"] >= 0.8 * bmax]
    shell_products = sum(r["status"] == "product" for r in shell)
    shell_unknown = sum(r["status"] in ("unresolved", "failed") for r in shell)
    n = len(records)
    parents = min(config["sampling"]["parents"], n)
    expected_clocks = {
        str(i): config["sampling"]["equilibration_steps"]
        + (n // parents + (i < n % parents)) * config["sampling"]["interval_steps"]
        for i in range(parents)
    }
    per_parent = {}
    for parent in range(parents):
        subset = [record for record in records if record["parent_id"] == parent]
        counts = {
            status: sum(record["status"] == status for record in subset)
            for status in ("product", "nonreactive", "unresolved", "failed")
        }
        errors = [
            record["max_energy_drift_per_bead"]
            for record in subset
            if record["max_energy_drift_per_bead"] is not None
        ]
        per_parent[str(parent)] = {
            "trajectories": len(subset),
            "counts": counts,
            "reaction_fraction": counts["product"] / len(subset),
            "production_steps": sum(record["steps"] for record in subset),
            "thermal_steps": summary["parent_thermal_steps"][str(parent)],
            "max_energy_drift_hartree_per_bead": float(max(errors)) if errors else None,
        }
    return {
        "per_parent": per_parent,
        "energy_drift_hartree_per_bead": {
            "maximum": float(max(drifts)) if drifts else None,
            "median": float(np.median(drifts)) if drifts else None,
            "p95": float(np.quantile(drifts, 0.95)) if drifts else None,
            "definition": "max_t |H_P(t)/P-H_P(0)/P| along each unthermostatted child",
        },
        "energy_drift_max_kcal_mol_per_bead": (
            float(max(drifts)) * HARTREE_KCAL_MOL if drifts else None
        ),
        "energy_drift_max_over_kBT": (
            float(max(drifts)) * config["beta"] if drifts else None
        ),
        "endpoint_abs_energy_change_hartree_per_bead": {
            "maximum": float(max(endpoint_drifts)) if endpoint_drifts else None,
            "median": float(np.median(endpoint_drifts)) if endpoint_drifts else None,
            "p95": (
                float(np.quantile(endpoint_drifts, 0.95)) if endpoint_drifts else None
            ),
        },
        "final_classifier_mismatches": violations,
        "final_states": final_states,
        "parent_thermal_steps_expected": expected_clocks,
        "parent_thermal_steps_observed": summary["parent_thermal_steps"],
        "thermal_h2_centroid_bond_bohr": {
            "mean": float(np.mean(h2_bonds)),
            "std": float(np.std(h2_bonds)),
            "minimum": float(min(h2_bonds)),
            "maximum": float(max(h2_bonds)),
        },
        "parent_momentum_temperature_K": {
            "mean": float(np.mean(temperatures)),
            "std": float(np.std(temperatures)),
            "degrees_of_freedom": 3 * 3 * config["beads"] - 6,
        },
        "launch_orientation_mean_squared_components": np.mean(axes, axis=0).tolist(),
        "launch_mean_energy_over_kBT": float(np.mean(energies) * config["beta"]),
        "launch_mean_b2_over_bmax2": float(np.mean(impacts**2) / bmax**2),
        "reaction_fraction_iid_wilson95": _wilson(len(products), n),
        "maximum_reactive_impact_bohr": max(
            (r["launch"]["impact_parameter_au"] for r in products), default=None
        ),
        "outer_impact_shell": {
            "lower_bohr": 0.8 * bmax,
            "upper_bohr": bmax,
            "trajectories": len(shell),
            "products": shell_products,
            "unknown": shell_unknown,
            "reaction_fraction_iid_wilson95": (
                _wilson(shell_products, len(shell)) if not shell_unknown else None
            ),
        },
    }


def refine_worst_child(raw_directory, summary, config, system):
    """Replay the worst full child at dt and dt/2 for equal physical time.

    These diagnostic replays do not replace a collision or change its weight.
    """
    available = [
        r
        for r in summary["records"]
        if r["max_energy_drift_per_bead"] is not None and r["steps"] > 0
    ]
    if not available or summary["counts"]["failed"]:
        return None
    worst = max(available, key=lambda r: r["max_energy_drift_per_bead"])
    path = Path(raw_directory) / "states" / f"initial-{worst['trajectory_id']:08d}.npz"
    with np.load(path, allow_pickle=False) as data:
        initial = State(data["q"], data["p"])
    potential = make_potential(config["potential"], system)
    results = []
    try:
        for subdivision in (1, 2):
            dt = config["dt_au"] / subdivision
            dynamics = Dynamics(
                potential,
                system.masses,
                config["beta"],
                config["beads"],
                dt,
                native="required",
            )
            state = initial.copy()
            energy0 = dynamics.energy(state) / config["beads"]
            maximum = 0.0
            started = time.perf_counter()
            for _ in range(worst["steps"] * subdivision):
                dynamics.step(state)
                end = dynamics.energy(state) / config["beads"]
                maximum = max(maximum, abs(end - energy0))
            results.append(
                {
                    "dt_fs": dt * AU_TIME_FS,
                    "steps": state.step,
                    "duration_fs": state.time * AU_TIME_FS,
                    "max_abs_energy_drift_hartree_per_bead": maximum,
                    "endpoint_abs_energy_drift_hartree_per_bead": abs(end - energy0),
                    "max_drift_over_kBT": maximum * config["beta"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    finally:
        if hasattr(potential, "close"):
            potential.close()
    ratio = (
        results[1]["max_abs_energy_drift_hartree_per_bead"]
        / results[0]["max_abs_energy_drift_hartree_per_bead"]
    )
    return {
        "trajectory_id": worst["trajectory_id"],
        "initial_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "results": results,
        "half_step_to_original_drift_ratio": ratio,
        "coarse_replay_matches_saved": bool(
            np.isclose(
                results[0]["max_abs_energy_drift_hartree_per_bead"],
                worst["max_energy_drift_per_bead"],
                rtol=1e-10,
                atol=1e-12,
            )
        ),
        "smaller_timestep_reduces_energy_error": ratio < 1.0,
        "changes_statistical_sample": False,
    }


def run_benchmark(output, *, require_reaction=True, resume=False, **settings):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    source = Path(__file__).read_bytes()
    saved_source = provenance / "drpmd-source-used.py"
    if saved_source.exists() and saved_source.read_bytes() != source:
        raise ValueError(
            "Benchmark source changed since this output was created; retain the original version for resume or choose a new output directory"
        )
    saved_source.write_bytes(source)
    raw = output / "raw"
    mapping = job_mapping(**settings)
    spec = JobSpec.from_mapping(mapping, base_directory=HERE)
    config, system = spec.resolve()
    classifier_checks = check_classifier_contract(system)
    started = time.perf_counter()
    summary = run(spec, output=raw, resume=resume)
    elapsed = time.perf_counter() - started
    before = json.dumps(summary["records"], sort_keys=True)
    resume_started = time.perf_counter()
    resumed = resume_run(raw)
    resume_seconds = time.perf_counter() - resume_started
    resume_ok = before == json.dumps(resumed["records"], sort_keys=True)
    resume_ok = (
        resume_ok and summary["parent_thermal_steps"] == resumed["parent_thermal_steps"]
    )
    diagnostics = analyze_raw(raw, summary, config, system)
    refinement = refine_worst_child(raw, summary, config, system)
    diagnostics["worst_child_timestep_refinement"] = refinement
    counts = summary["counts"]
    n = summary["trajectories"]
    resolved_fraction = (counts["product"] + counts["nonreactive"]) / n
    gates = {
        "correct_total": n == config["collision"]["trajectories"]
        and sum(counts.values()) == n,
        "no_failed_trajectories": counts["failed"] == 0,
        "resolved_at_least_98_percent": resolved_fraction >= 0.98,
        "reaction_requirement_met": counts["product"] > 0 or not require_reaction,
        "max_energy_drift_below_5e_4_hartree_per_bead": diagnostics[
            "energy_drift_hartree_per_bead"
        ]["maximum"]
        is not None
        and diagnostics["energy_drift_hartree_per_bead"]["maximum"] < 5e-4,
        "final_identity_classifier_agrees": not diagnostics[
            "final_classifier_mismatches"
        ],
        "continuous_parent_clocks": diagnostics["parent_thermal_steps_expected"]
        == diagnostics["parent_thermal_steps_observed"],
        "completed_restart_identical": resume_ok,
        "worst_child_coarse_replay_matches": refinement is not None
        and refinement["coarse_replay_matches_saved"],
        "worst_child_timestep_refinement": refinement is not None
        and refinement["smaller_timestep_reduces_energy_error"],
    }
    report = {
        "benchmark": "physical H + H2 exchange on BKMP2",
        "task": "drpmd",
        "passed": all(gates.values()),
        "gates": gates,
        "converged": False,
        "benchmark_runner_sha256": hashlib.sha256(source).hexdigest(),
        "temperature_K": config["temperature_K"],
        "beads": config["beads"],
        "timestep_fs": config["dt_au"] * AU_TIME_FS,
        "seed": config["seed"],
        "bmax_bohr": config["collision"]["bmax_au"],
        "separation_bohr": config["system"]["separation_au"],
        "trajectories": n,
        "parents": summary["parents"],
        "counts": counts,
        "workers": config["parallel"]["trajectory_workers"],
        "parent_equilibration_fs": config["sampling"]["equilibration_steps"]
        * config["dt_au"]
        * AU_TIME_FS,
        "parent_sample_interval_fs": config["sampling"]["interval_steps"]
        * config["dt_au"]
        * AU_TIME_FS,
        "production_max_time_fs": config["collision"]["max_steps"]
        * config["dt_au"]
        * AU_TIME_FS,
        "reactive_exchange_required": require_reaction,
        "reactive_exchange_observed": counts["product"] > 0,
        "resolved_fraction": resolved_fraction,
        "rate_estimate": summary["total_reaction"],
        "channels": summary["channels"],
        "elapsed_seconds": elapsed,
        "completed_restart_seconds": resume_seconds,
        "classifier_contract_checks": classifier_checks,
        "diagnostics": diagnostics,
        "raw_directory": str(raw),
        "limits": [
            "This is a finite-sample physical workflow benchmark; T, bead count, timestep, impact cutoff, radial separation and parent decorrelation are not jointly converged.",
            "The two H2 exchange memberships are counted explicitly; no extra factor of two is applied.",
            "IID Wilson intervals neglect dependence within a thermal parent; the workflow also reports parent-cluster resampling with only a small number of clusters.",
            "The rate estimate is truncated at the stated impact cutoff. No observed outer-shell reactions is not proof of impact-cutoff convergence.",
        ],
    }
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/h3-drpmd")
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--beads", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1500.0)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--max-time-fs", type=float, default=1200.0)
    parser.add_argument("--equilibration-fs", type=float, default=500.0)
    parser.add_argument("--sample-interval-fs", type=float, default=50.0)
    parser.add_argument("--bmax-bohr", type=float, default=3.0)
    parser.add_argument("--separation-bohr", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=81073)
    parser.add_argument(
        "--allow-zero-reactions",
        action="store_true",
        help="For short tuning pilots only",
    )
    parser.add_argument("--resume", action="store_true")
    options = vars(parser.parse_args(argv))
    output = options.pop("output")
    options["require_reaction"] = not options.pop("allow_zero_reactions")
    report = run_benchmark(output, **options)
    brief = {
        key: report[key]
        for key in (
            "benchmark",
            "passed",
            "counts",
            "resolved_fraction",
            "elapsed_seconds",
            "gates",
        )
    }
    brief["energy_drift_hartree_per_bead"] = report["diagnostics"][
        "energy_drift_hartree_per_bead"
    ]
    brief["report"] = str(Path(output).resolve() / "benchmark.json")
    print(json.dumps(brief, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
