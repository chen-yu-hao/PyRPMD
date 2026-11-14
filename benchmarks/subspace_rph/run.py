"""Validate the subspace correction models against exact full-dimensional rates.

For a model potential and temperature this computes, with the pipeline's own
model builders (``path_model``, ``plane_model``, ``rph_model``) on the default
MEP path:

* ``C_S = k_QM(H_S) / k_RPMD(H_S)`` for the 1D model, the 1D model with the
  small-curvature effective mass, the 2D plane model and the 2D RPH model;
* the ideal correction ``C = k_QM(full) / k_RPMD(full)`` of the whole model
  (not for ``embedded`` models, whose ideal correction is that of their base).

``C_S / C - 1`` is the relative error of the corrected rate
``k_RPMD(full) * C_S`` against the exact quantum rate.

    python run.py --model curved_2d --temperature 300 --nbeads 32 --out results/curved_2d_300K.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subspace_rph.models import CM, Curved, Embedded, model_run  # noqa: E402
from benchmarks.subspace_rph.reference import QUANTUM, build_path, subspace_models  # noqa: E402

from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

MODELS = {
    "curved_2d": lambda: Curved(2),
    "curved_3d": lambda: Curved(3, omega_z=1200.0 * CM),
    "embedded_8d": lambda: Embedded(Curved(3, omega_z=1200.0 * CM), (800 * CM, 1400 * CM, 2200 * CM,
                                                                       2900 * CM, 3600 * CM)),
}


def rpmd(model, beta, nbeads, options):
    t0 = time.time()
    result = ring_polymer_rate(model, beta, nbeads, **options)
    return {"rate": result.rate, "log_standard_error": result.rate_standard_error / result.rate,
            "transmission": result.transmission, "seconds": time.time() - t0}


def quantum(model, beta, options):
    t0 = time.time()
    result = exact_quantum_rate(model, beta, **options)
    return {"rate": result.rate, "method": result.method, "seconds": time.time() - t0,
            "retained": result.diagnostics.get("retained_points")}


def full_reference(potential, beta, args):
    """The whole model on a tensor grid, normalized far out in the reactant channel."""
    dimension = potential.dimension
    reference, cap = args.full_reference, args.full_cap
    length = reference + cap + 0.25 * args.full_spacing[0] * 8
    x = np.arange(-length, length + 1e-9, args.full_spacing[0])
    x = x - x[np.argmin(np.abs(x + reference))] - reference     # a node on the reference slice
    axes = [x]
    bounds = [(-35.0, 50.0), (-45.0, 45.0)]
    for i in range(1, dimension):
        lo, hi = bounds[i - 1]
        axes.append(np.arange(lo, hi + 1e-9, args.full_spacing[i]))
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    model = SubspaceModel(tuple(axes), potential.energy(points), np.ones(dimension), 0.0, "asymptotic",
                          -reference, {"model": "full"}, function=potential.energy, gradient=potential.gradient)
    options = {"cap_width": cap}
    if dimension == 2:
        floor = 0.5 * potential.omega_y - 1.0 / beta
        offset = max(args.truncation_kt / beta, args.truncation_min)
        options.update(energy_floor=floor, truncation_energy=floor + offset, dense_limit=12000, method="sylvester")
    else:
        options.update(method="flux_eigenstates")
    return model, options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--nbeads", type=int, required=True)
    parser.add_argument("--spacing-2d", type=float, nargs=2, default=None)
    parser.add_argument("--full", action="store_true", help="also compute the exact full-dimensional reference")
    parser.add_argument("--full-spacing", type=float, nargs="+", default=[2.0, 2.0, 3.0])
    parser.add_argument("--full-reference", type=float, default=100.0)
    parser.add_argument("--full-cap", type=float, default=40.0)
    parser.add_argument("--truncation-kt", type=float, default=30.0)
    parser.add_argument("--reference-density-tolerance", type=float, default=None,
                        help="quantum.reference_density_tolerance (default: the pipeline's, 1e-2)")
    parser.add_argument("--truncation-min", type=float, default=0.055,
                        help="hartree; about six quanta of the stiffest transverse mode (2000 cm^-1)")
    parser.add_argument("--models", nargs="+", default=["1d", "1d_sct", "plane", "rph"])
    parser.add_argument("--sweeps", type=int, default=800)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge", action="store_true",
                        help="keep the entries of an existing --out report that are not recomputed")
    args = parser.parse_args()

    potential = MODELS[args.model]()
    run = model_run(potential, args.temperature, args.nbeads)
    rpmd_options = dict(windows=48, walkers=32, burn_in=200, sweeps=args.sweeps, trajectories=args.trajectories,
                        seed=args.seed)
    report = {"model": args.model, "dimension": potential.dimension, "temperature_K": args.temperature,
              "nbeads": args.nbeads, "beta": run.beta, "rpmd_options": rpmd_options, "spacing_2d": args.spacing_2d,
              "truncation_kt": args.truncation_kt, "truncation_min": args.truncation_min,
              "reference_density_tolerance": args.reference_density_tolerance, "subspace": {}}
    if args.merge and args.out.is_file():
        previous = json.loads(args.out.read_text())
        report["subspace"].update({k: v for k, v in previous.get("subspace", {}).items() if k not in args.models})
        if "full" in previous and not args.full:
            report["full"] = previous["full"]
    t0 = time.time()
    path, info = build_path(run)
    record = info["mep"]
    report["path"] = {"points": len(path), "seconds": time.time() - t0,
                      "saddle_iterations": record["iterations"], "saddle_negative_modes": record["negative_modes"],
                      "imaginary_frequency_cm": record["imaginary_frequency"] * 219474.6313632,
                      "reactant": info["reactant_termination"]["status"],
                      "product": info["product_termination"]["status"]}
    extra = {"truncation_kt": args.truncation_kt, "truncation_min": args.truncation_min, "dense_limit": 12000}
    if args.reference_density_tolerance is not None:
        extra["reference_density_tolerance"] = args.reference_density_tolerance
    if args.spacing_2d is not None:
        extra["spacing_2d"] = list(args.spacing_2d)
    original = dict(QUANTUM)
    QUANTUM.update(extra)
    try:
        built = subspace_models(run, path, info, which=tuple(args.models))
    finally:
        QUANTUM.clear()
        QUANTUM.update(original)
    for name, (model, quantum_options, model_info, rpmd_model) in built.items():
        entry = {"shape": list(model.shape),
                 "captured_curvature": model_info.get("captured_curvature"),
                 "retained_points": model_info.get("retained_points"),
                 "bath_stable": model.provenance.get("bath_stable")}
        entry["quantum"] = quantum(model, run.beta, quantum_options)
        entry["rpmd"] = rpmd(rpmd_model, run.beta, args.nbeads, rpmd_options)
        entry["C_S"] = entry["quantum"]["rate"] / entry["rpmd"]["rate"]
        entry["log_standard_error"] = entry["rpmd"]["log_standard_error"]
        report["subspace"][name] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(name, entry["C_S"], "+-", entry["log_standard_error"], flush=True)
    if args.full:
        model, options = full_reference(potential, run.beta, args)
        full = {"shape": list(model.shape), "quantum": quantum(model, run.beta, options),
                "rpmd": rpmd(model, run.beta, args.nbeads, rpmd_options)}
        full["C"] = full["quantum"]["rate"] / full["rpmd"]["rate"]
        full["log_standard_error"] = full["rpmd"]["log_standard_error"]
        report["full"] = full
        for entry in report["subspace"].values():
            entry["error_vs_full"] = entry["C_S"] / full["C"] - 1.0
        print("full", full["C"], "+-", full["log_standard_error"], flush=True)
    if "full" in report:
        for entry in report["subspace"].values():
            entry["error_vs_full"] = entry["C_S"] / report["full"]["C"] - 1.0
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
