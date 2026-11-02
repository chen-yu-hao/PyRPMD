"""Bounded native/Python H+CH4 propagation and cost probe, never a rate run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from pyrpmd import _native
from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias
from pyrpmd.constants import AMU_TO_AU, AU_RATE_TO_CM3_S, FS_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import make_potential
from pyrpmd.workflows.fes import separate_reactants_to_xi
from pyrpmd.workflows.rate import project_momenta, project_positions, rate_coordinate

from benchmarks.hch4.reproduction_input import PROTOCOL, resolve_pilot
from benchmarks.hch4.verify import paper_geometries

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_state(geometry, masses, beta, beads, seed):
    rng = np.random.default_rng(seed)
    q = np.repeat(np.asarray(geometry)[None], beads, axis=0)
    deviations = rng.normal(scale=0.002, size=q.shape)
    q += deviations - deviations.mean(axis=0)
    p = rng.normal(size=q.shape) * np.sqrt(masses[None, :, None] * beads / beta)
    p -= masses[None, :, None] * (p.sum(axis=(0, 1)) / (beads * masses.sum()))
    return State(q, p)


def independent_xi(q, masses, reference):
    """Direct scalar four-channel equation, separate from the CV class."""
    com = np.average(q[:5], axis=0, weights=masses[:5])
    s0 = 30.0 - np.linalg.norm(q[5] - com)
    difference = np.linalg.norm(reference[5] - reference[3]) - np.linalg.norm(reference[3] - reference[1])
    s1 = max(difference - np.linalg.norm(q[5] - q[h]) + np.linalg.norm(q[h] - q[1])
             for h in (0, 2, 3, 4))
    return s0 / (s0 - s1)


def run(library):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    limits = PROTOCOL["native_probe_predeclared_gates"]
    seed = PROTOCOL["campaign"]["calibration_seed_excluded_from_primary"]
    report = {"schema": "pyrpmd.hch4.native-planning-probe.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "rate_validated": False, "equilibrium_sampled": False,
              "scope": "Static input/CV verification and short deterministic dynamics; no umbrella campaign or rate estimate",
              "seed_excluded_from_primary": seed, "limits": limits, "comparisons": [],
              "nve_checks": [], "timings": [], "gates": {}}

    def budget():
        if time.process_time() - started_cpu > limits["maximum_total_process_CPU_seconds"]:
            raise RuntimeError("Bounded native verification exceeded its preregistered CPU budget")

    raw, config, system = resolve_pilot(library)
    provider = make_potential(config["potential"], system)
    beta, masses = config["beta"], system.masses
    coordinate = rate_coordinate(config, system)
    q0 = system.geometry
    qoff = q0 + np.random.default_rng(seed + 1).normal(scale=0.01, size=q0.shape)
    actual, grad = coordinate.centroid_value_gradient(qoff)
    expected = independent_xi(qoff, masses, q0)
    fd = np.empty_like(qoff)
    for atom in range(6):
        for axis in range(3):
            plus, minus = qoff.copy(), qoff.copy()
            plus[atom, axis] += 1e-5
            minus[atom, axis] -= 1e-5
            fd[atom, axis] = (independent_xi(plus, masses, q0) - independent_xi(minus, masses, q0)) / 2e-5
    entrance = separate_reactants_to_xi(q0, coordinate, 0.0, masses)
    permutation_values = []
    for permutation in coordinate.permutations:
        permuted = np.empty_like(q0)
        permuted[list(permutation)] = q0
        permutation_values.append(float(coordinate.centroid_value_gradient(permuted)[0]))
    ma, mb = masses[:5].sum(), masses[5]
    reduced_mass = ma * mb / (ma + mb)
    prefactor = 4 * np.pi * 30**2 / np.sqrt(2 * np.pi * beta * reduced_mass)
    report["static"] = {
        "natoms": system.natoms, "symbols": list(system.symbols), "masses_amu": (masses / AMU_TO_AU).tolist(),
        "masses_au": masses.tolist(), "beta_hartree_inverse": beta,
        "Rinf_bohr": coordinate.rinf, "channels": len(coordinate.paths),
        "forming_breaking_paths_zero_based": coordinate.paths,
        "xi_at_each_equivalent_reference": permutation_values,
        "xi_entrance": float(coordinate.centroid_value_gradient(entrance)[0]),
        "xi_independent_scalar_error": float(abs(actual - expected)),
        "xi_gradient_independent_FD_max_error": float(np.max(abs(grad - fd))),
        "prefactor_au": prefactor, "prefactor_cm3_molecule_s": prefactor * AU_RATE_TO_CM3_S,
        "reduced_mass_au": reduced_mass,
        "resolved_pilot_sampling": config["sampling"],
        "native_binding_kind": provider.native_binding().kind,
        "source_grid_actual_count": len(np.arange(-0.05, 1.05, 0.01)),
        "source_grid_last_value": float(np.arange(-0.05, 1.05, 0.01)[-1]),
        "input": raw,
    }
    report["gates"]["static_equations_and_channels"] = bool(
        len(coordinate.paths) == 4 and abs(actual - expected) < 1e-14
        and np.max(abs(grad - fd)) < 1e-8
        and max(abs(np.array(permutation_values) - 1)) < 1e-13
        and abs(report["static"]["xi_entrance"]) < 1e-12)

    for beads in (1, 128):
        dynamics = Dynamics(provider, masses, beta, beads, 0.1 * FS_TO_AU,
                            native="required", threads=1, loop_backend="native", seed=seed)
        start = initial_state(q0, masses, beta, beads, seed + beads)
        bias = CompositeBias(UmbrellaBias(coordinate, [1.0], 30.0), FluxMetricBias(coordinate, masses, beta))
        for mode in ("unbiased", "umbrella_flux", "constrained_parent"):
            options = {"cv": coordinate, "record_states": True}
            first = start.copy()
            if mode == "umbrella_flux":
                options["bias"] = bias
            elif mode == "constrained_parent":
                surface = coordinate.dividing_surface(0.999 if beads == 1 else 1.007)
                first.q[:] = project_positions(first.q, surface, masses)
                project_momenta(first.p, first.q, surface, masses)
                options.update(constraint=surface, cv=surface)
            for steps in (1, 64):
                a, b = first.copy(), first.copy()
                got = dynamics.propagate_block(a, steps, backend="native", **options)
                reference = dynamics.propagate_block(b, steps, backend="python", **options)
                qerr = float(np.max(abs(got.q - reference.q)))
                perr = float(np.max(abs(got.p - reference.p)))
                eerr = float(np.max(abs(got.energies - reference.energies)) / beads)
                record = {"beads": beads, "mode": mode, "steps": steps,
                          "native_backend": got.backend, "native_completed": got.completed,
                          "python_backend": reference.backend,
                          "maximum_position_error_bohr": qerr,
                          "maximum_momentum_error_au": perr,
                          "maximum_energy_per_bead_error_hartree": eerr,
                          "maximum_observable_error": float(np.max(abs(got.values - reference.values))),
                          "passed": bool(got.backend == "native" and got.completed == steps
                                         and qerr <= limits["native_python_position_atol_bohr"]
                                         and perr <= limits["native_python_momentum_atol_au"]
                                         and eerr <= limits["native_python_energy_per_bead_atol_hartree"])}
                report["comparisons"].append(record)
                budget()

        reactant, _ = paper_geometries(30.0)
        nve_start = initial_state(reactant, masses, beta, beads, seed + 10 + beads)
        spans = []
        for dt_fs in (0.1, 0.05):
            dyn = Dynamics(provider, masses, beta, beads, dt_fs * FS_TO_AU,
                           native="required", threads=1, loop_backend="native")
            state = nve_start.copy()
            e0 = dyn.energy(state) / beads
            result = dyn.propagate_block(state, round(20 / dt_fs), backend="native")
            energies = np.r_[e0, result.energies / beads]
            span = float(np.ptp(energies))
            spans.append(span)
            report["nve_checks"].append({"beads": beads, "dt_fs": dt_fs, "duration_fs": 20,
                                         "energy_span_per_bead_hartree": span,
                                         "energy_end_minus_start_per_bead_hartree": float(energies[-1] - e0),
                                         "scope": "Non-equilibrium but identical initial state at both time steps"})
            budget()
        report["gates"][f"P{beads}_short_NVE"] = bool(
            max(spans) <= limits["nve_energy_span_per_bead_limit_hartree"]
            and spans[1] <= limits["half_dt_energy_span_ratio_limit"] * spans[0] + 1e-13)

        # Three fixed positions on the published reaction coordinate. Short
        # blocks restart from identical states: this is timing, not sampling.
        for center in (0.0, 0.9, 1.0):
            geometry = q0 if center == 1 else separate_reactants_to_xi(q0, coordinate, center, masses)
            timed_start = initial_state(geometry, masses, beta, beads, seed + 100 + beads)
            timed_bias = CompositeBias(UmbrellaBias(coordinate, [center], 30.0), FluxMetricBias(coordinate, masses, beta))
            assert dynamics.can_fuse(bias=timed_bias, observe=coordinate)
            for backend, steps, repeats in (("native", 256, 3), ("python", 64, 1)):
                cpu, wall = [], []
                for _ in range(repeats):
                    state = timed_start.copy()
                    t_cpu, t_wall = time.process_time(), time.perf_counter()
                    result = dynamics.propagate_block(state, steps, bias=timed_bias, cv=coordinate,
                                                      observe=np.arange(1, steps + 1) % 20 == 0,
                                                      check_energy=True, backend=backend)
                    cpu.append(time.process_time() - t_cpu)
                    wall.append(time.perf_counter() - t_wall)
                    if result.backend != backend or result.completed != steps:
                        raise AssertionError("Unexpected timing fallback or incomplete block")
                    budget()
                report["timings"].append({"beads": beads, "center": center, "backend": backend,
                                          "steps_per_repeat": steps, "repeats": repeats,
                                          "CPU_seconds": cpu, "wall_seconds": wall,
                                          "median_CPU_microseconds_per_step": float(np.median(cpu) * 1e6 / steps),
                                          "median_wall_microseconds_per_step": float(np.median(wall) * 1e6 / steps),
                                          "included": "PES, normal modes, four-channel CV/Hessian, umbrella, flux metric, energy check and 2-fs observations; excludes thermostat/checkpoint/task scheduling"})
    report["gates"]["native_python_all_paths"] = all(x["passed"] for x in report["comparisons"])
    reference = PROTOCOL["reference_protocol"]
    umbrella_steps = reference["umbrella_windows"] * reference["trajectories_per_window"] * int(
        (reference["equilibration_ps_per_trajectory"] + reference["production_ps_per_trajectory"]) * 1000 / reference["dt_fs"])
    pilot = PROTOCOL["campaign"]["pilot"]
    pilot_steps = 4 * 111 * pilot["replicas_per_window_per_seed"] * int(
        (pilot["equilibration_fs"] + pilot["production_fs"]) / 0.1)
    report["cost_extrapolation"] = {"reference_umbrella_steps": umbrella_steps,
                                    "reference_production_samples": 111 * 200000000,
                                    "reference_unconstrained_child_steps": 100000 * 500,
                                    "pilot_four_seed_umbrella_steps": pilot_steps,
                                    "not_a_forecast_confidence_interval": True, "by_beads": {}}
    for beads in (1, 128):
        costs = [r["median_CPU_microseconds_per_step"] * 1e-6 for r in report["timings"]
                 if r["beads"] == beads and r["backend"] == "native"]
        report["cost_extrapolation"]["by_beads"][str(beads)] = {
            "reference_umbrella_CPU_hours_range_of_three_centers": [umbrella_steps * min(costs) / 3600, umbrella_steps * max(costs) / 3600],
            "pilot_four_seed_umbrella_CPU_hours_range_of_three_centers": [pilot_steps * min(costs) / 3600, pilot_steps * max(costs) / 3600],
            "caveat": "Short local-block extrapolation, excludes warmup failures, thermostat, restart I/O, full ensemble geometry variation and analysis. Wall time depends on available workers."
        }
    files = ["benchmarks/hch4/pes.py", "benchmarks/hch4/wrapper.f90", "benchmarks/hch4/reproduction_protocol.json",
             "benchmarks/hch4/reproduction_input.py", "benchmarks/hch4/verify_native.py",
             "src/pyrpmd/input.py", "src/pyrpmd/system.py", "src/pyrpmd/cv.py", "src/pyrpmd/constants.py",
             "src/pyrpmd/bias.py", "src/pyrpmd/dynamics.py", "src/pyrpmd/propagation.py",
             "src/pyrpmd/block_propagation.py", "src/pyrpmd/potentials/native.py",
             "src/pyrpmd/potentials/python.py", "src/pyrpmd/workflows/fes.py", "src/pyrpmd/workflows/rate.py"]
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                         "native_binary_sha256": sha256(_native.__file__),
                         "PES_library_sha256": sha256(library),
                         "source_files": {p: sha256(ROOT / p) for p in files},
                         "CPU_seconds": time.process_time() - started_cpu,
                         "wall_seconds": time.perf_counter() - started_wall}
    report["gates"]["CPU_budget"] = report["runtime"]["CPU_seconds"] <= limits["maximum_total_process_CPU_seconds"]
    report["passed"] = all(report["gates"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "gates": report["gates"],
                      "runtime": {k: v for k, v in report["runtime"].items() if k != "source_files"},
                      "cost": report["cost_extrapolation"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
