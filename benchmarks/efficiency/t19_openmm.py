"""T19 physical-force validation of serial, Context-pool and RPMD-copy APIs.

Default CUDA matrix (no CPU or precision fallback)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
      python benchmarks/efficiency/t19_openmm.py --output openmm-cuda.json

Explicit CPU diagnostic, including the exact float64 Reference platform::

    python benchmarks/efficiency/t19_openmm.py --device cpu --platform Reference \
      --beads 8 --atoms 64 --output openmm-reference.json

Every measured changing configuration is compared to OpenMM Reference float64.
This is an evaluator benchmark: it neither advances dynamics nor establishes
physical convergence. An RPMD copy bank retains coordinates on the selected
device, but its public set/get API still performs per-copy transfers. Model
construction/cold allocation are reported separately from warm evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.efficiency.force_engine_adapters import coordinate_sequence, environment, errors
from pyrpmd.potentials.openmm import BOHR_NM, OpenMMPotential


def hardware():
    """Record observed software/hardware, retaining unavailable GPU fields."""
    result = environment()
    result["gpu_driver_query"] = None
    result["nvidia_smi_summary"] = None
    result["cuda_toolkit_nvcc"] = None
    result["nvidia_smi_cuda_version_semantics"] = "maximum driver-supported CUDA API, not an installed runtime/toolkit"
    try:
        import openmm
        result["openmm_source_revision"] = getattr(openmm.version, "git_revision", None)
    except ImportError:
        result["openmm_source_revision"] = None
    if result["nvidia_smi"]:
        for key, command in (
            ("gpu_driver_query", [result["nvidia_smi"],
                                  "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            ("nvidia_smi_summary", [result["nvidia_smi"]]),
        ):
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
                result[key] = {"returncode": completed.returncode,
                               "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
            except (OSError, subprocess.TimeoutExpired) as exc:
                result[key] = {"error": str(exc)}
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            completed = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
            result["cuda_toolkit_nvcc"] = {"returncode": completed.returncode,
                                           "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["cuda_toolkit_nvcc"] = {"error": str(exc)}
    try:
        result["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["source_commit"] = None
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), ROOT / "src/pyrpmd/potentials/openmm.py",
                     ROOT / "benchmarks/efficiency/force_engine_adapters.py",
                     ROOT / "benchmarks/water/qtip4pf.py")
    }
    return result


def model_case(name, atoms):
    import openmm as mm

    if name == "water":
        from benchmarks.water.qtip4pf import initial_grid, make_system
        from pyrpmd.constants import ANGSTROM_TO_AU

        cells = round((atoms / 3)**(1 / 3))
        if atoms != 3 * cells**3:
            raise ValueError("water requires atoms=3*cells**3 (for example 24, 192, 2187)")
        q, box = initial_grid(cells_per_axis=cells)
        system = make_system(molecules=cells**3, box_angstrom=box,
                             cutoff_angstrom=.45 * box, ewald_tolerance=1e-7)
        return system, ["O", "H", "H"] * cells**3, q, {"cell": np.eye(3) * box * ANGSTROM_TO_AU}
    model = mm.System()
    for _ in range(atoms):
        model.addParticle(39.948)
    if name == "harmonic":
        force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
        force.addGlobalParameter("k", 12.)
        for atom in range(atoms):
            force.addParticle(atom, [])
        model.addForce(force)
        q = np.random.default_rng(179).normal(scale=.1, size=(atoms, 3)) / BOHR_NM
        return model, ["Ar"] * atoms, q, {}
    if name != "lj":
        raise ValueError(f"unknown benchmark model {name!r}")
    cells = int(np.ceil(atoms**(1 / 3)))
    while cells**3 < atoms:
        cells += 1
    side = .5 * max(cells, 2)
    index = np.arange(atoms)
    q = .5 * (np.column_stack((index % cells, index // cells % cells,
                              index // (cells * cells))) + .5)
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(min(1.2, .45 * side))
    force.setUseDispersionCorrection(False)
    force.setUseSwitchingFunction(False)
    for _ in range(atoms):
        force.addParticle(0., .34, .997)
    model.addForce(force)
    model.setDefaultPeriodicBoxVectors(mm.Vec3(side, 0, 0), mm.Vec3(0, side, 0), mm.Vec3(0, 0, side))
    return model, ["Ar"] * atoms, q / BOHR_NM, {"cell": np.eye(3) * side / BOHR_NM}


def _measure(options, mode, workers, frames, references, warmup, atol, rtol):
    started = time.perf_counter()
    provider = OpenMMPotential(**options, batch_mode=mode, max_workers=workers)
    construction = time.perf_counter() - started
    try:
        first = time.perf_counter()
        provider.evaluate(frames[0])
        cold = time.perf_counter() - first
        samples, maximum = [], {}
        scaled_error = 0.
        for index, frame in enumerate(frames):
            started = time.perf_counter()
            actual = provider.evaluate(frame)
            elapsed = time.perf_counter() - started
            if index >= warmup:
                samples.append(elapsed)
            reference = references[index]
            for name, value in errors(actual, reference).items():
                maximum[name] = max(maximum.get(name, 0.), value)
            for field in ("energy", "forces"):
                value, expected = getattr(actual, field), getattr(reference, field)
                scaled_error = max(scaled_error, float(np.max(
                    np.abs(value - expected) / (atol + rtol * np.abs(expected)))))
        return {"status": "measured" if scaled_error <= 1 else "correctness_failed",
                "construction_seconds": construction, "cold_evaluation_seconds": cold,
                "seconds_per_batch_samples": samples,
                "frames_per_batch": len(frames[0]),
                "median_seconds_per_batch": statistics.median(samples),
                "median_seconds_per_frame": statistics.median(samples) / len(frames[0]),
                "reference_errors": maximum, "maximum_scaled_error": scaled_error,
                "atol": atol, "rtol": rtol, "provider_metadata": provider.metadata,
                "resident_contexts": len(provider._contexts),
                "resident_rpmd_copies": provider.integrator.getNumCopies() if mode == "rpmd" else None,
                "synchronization": "getState returns host energy and forces before timing stops",
                "transfer_contract": "per_copy_upload_and_result_download"}
    finally:
        provider.close()


def _aggregate_rounds(records):
    if any(record["status"] == "failed" for record in records):
        return {"status": "failed", "rounds": records}
    samples = [value for record in records for value in record["seconds_per_batch_samples"]]
    result = {key: value for key, value in records[0].items()
              if key not in ("construction_seconds", "cold_evaluation_seconds", "round", "position_in_order")}
    result.update(
        status="measured" if all(record["status"] == "measured" for record in records) else "correctness_failed",
        rounds=records, seconds_per_batch_samples=samples,
        median_seconds_per_batch=statistics.median(samples),
        construction_seconds_samples=[record["construction_seconds"] for record in records],
        cold_evaluation_seconds_samples=[record["cold_evaluation_seconds"] for record in records],
        median_construction_seconds=statistics.median([record["construction_seconds"] for record in records]),
        median_cold_evaluation_seconds=statistics.median([record["cold_evaluation_seconds"] for record in records]),
        maximum_scaled_error=max(record["maximum_scaled_error"] for record in records),
        reference_errors={key: max(record["reference_errors"][key] for record in records)
                          for key in records[0]["reference_errors"]},
    )
    # Every sample has the same batch size. Avoid taking a median-of-medians
    # for per-frame cost when a caller chooses an even sample count.
    result["median_seconds_per_frame"] = result["median_seconds_per_batch"] / records[0]["frames_per_batch"]
    return result


def _model_parameters(name, model, q, extras):
    result = {"name": name, "physical_atoms": len(q),
              "openmm_particles": model.getNumParticles(),
              "initial_coordinates_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
              "cell_bohr": extras["cell"].tolist() if "cell" in extras else None,
              "periodic": bool(model.usesPeriodicBoundaryConditions()),
              "trajectory_initial_displacement_std_bohr": .002,
              "trajectory_step_displacement_std_bohr": .0004}
    if name == "lj":
        from openmm import unit
        force = model.getForce(0)
        result.update(sigma_nm=.34, epsilon_kj_mol=.997, charge_e=0.,
                      cutoff_nm=float(force.getCutoffDistance().value_in_unit(unit.nanometer)),
                      dispersion_correction=False, switching_function=False,
                      grid_spacing_nm=.5, mass_dalton=39.948)
    elif name == "harmonic":
        result.update(spring_constant_kj_mol_nm2=12., coordinate_seed=179,
                      coordinate_std_nm=.1, mass_dalton=39.948)
    else:
        from benchmarks.water.qtip4pf import PARAMETERS
        result.update(force_field_parameters=dict(PARAMETERS),
                      coordinate_seed=541, density_g_cm3=.997,
                      cutoff_box_fraction=.45, ewald_tolerance=1e-7,
                      dispersion_correction=False, switching_function=False)
    return result


def run_case(*, model_name, beads, atoms, device, platform=None, workers=(1, 4),
             warmup=2, repeats=7, rounds=3, seed=1903):
    """Measure a single shape; exceptions remain explicit per-mode records."""
    model, symbols, q, extras = model_case(model_name, atoms)
    frames = coordinate_sequence(q, beads, warmup + repeats, seed)
    started = time.perf_counter()
    with OpenMMPotential(model, symbols, platform="Reference", **extras) as reference:
        reference_construction = time.perf_counter() - started
        started = time.perf_counter()
        references = [reference.evaluate(frame) for frame in frames]
        reference_evaluation = time.perf_counter() - started
        reference_metadata = dict(reference.metadata)
    selected = platform or ("CPU" if device == "cpu" else "CUDA")
    properties = {"Threads": "1"} if selected == "CPU" else {}
    if selected == "CUDA":
        properties["Precision"] = "double"
    options = {"model": model, "symbols": symbols, "device": device,
               "platform": selected, "properties": properties, **extras}
    atol, rtol = ((1e-7, 3e-6) if selected == "CPU" else
                  ((2e-7, 2e-6) if model_name == "water" else (1e-12, 1e-10)))
    modes = [("serial", 1), *(("contexts", worker) for worker in workers), ("rpmd", 1)]
    raw = {f"contexts-{worker}" if mode == "contexts" else mode: [] for mode, worker in modes}
    mode_orders = []
    for round_index in range(rounds):
        ordered = modes if round_index % 2 == 0 else list(reversed(modes))
        mode_orders.append([f"contexts-{worker}" if mode == "contexts" else mode for mode, worker in ordered])
        for position, (mode, worker) in enumerate(ordered):
            label = f"contexts-{worker}" if mode == "contexts" else mode
            try:
                record = _measure(options, mode, worker, frames, references, warmup, atol, rtol)
            except Exception as exc:
                record = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)}
            record.update(round=round_index, position_in_order=position)
            raw[label].append(record)
    results = {label: _aggregate_rounds(records) for label, records in raw.items()}
    serial = results["serial"]
    if serial["status"] == "measured":
        for record in results.values():
            if record["status"] == "measured":
                record["speedup_over_serial"] = serial["median_seconds_per_batch"] / record["median_seconds_per_batch"]
    return {"model": model_name, "beads": beads, "atoms": atoms,
            "status": "measured" if all(mode["status"] == "measured" for mode in results.values()) else "partial_failure",
            "device": device, "platform": selected, "reference": reference_metadata,
            "reference_construction_seconds": reference_construction,
            "reference_evaluation_seconds": reference_evaluation,
            "reference_cost_excluded_from_measurements": True,
            "model_parameters": _model_parameters(model_name, model, q, extras),
            "warmup": warmup, "repeats": repeats, "rounds": rounds, "seed": seed,
            "mode_orders": mode_orders,
            "modes": results, "physical_converged": False}


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _device(value):
    if value == "cuda":
        return "cuda:0"
    if value == "cpu" or (value.startswith("cuda:") and value[5:].isdigit()):
        return value
    raise argparse.ArgumentTypeError("must be cpu or cuda:INDEX")


def _atomic_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", type=_device)
    parser.add_argument("--platform", choices=("CPU", "Reference", "CUDA"))
    parser.add_argument("--model", choices=("lj", "harmonic", "water"), default="lj")
    parser.add_argument("--beads", nargs="+", type=_positive, default=[8, 32, 64])
    parser.add_argument("--atoms", nargs="+", type=_positive, default=[64, 512, 4096])
    parser.add_argument("--workers", nargs="+", type=_positive, default=[1, 4])
    parser.add_argument("--warmup", type=_positive, default=2)
    parser.add_argument("--repeats", type=_positive, default=7)
    parser.add_argument("--rounds", type=_positive, default=3)
    parser.add_argument("--seed", type=int, default=1903)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selected = args.platform or ("CPU" if args.device == "cpu" else "CUDA")
    if (args.device == "cpu") != (selected != "CUDA"):
        parser.error("device and platform must agree; no platform fallback is allowed")
    info = hardware()
    report = {"schema": "pyrpmd.t19.openmm.v1", "environment": info,
              "invocation": {"argv": [sys.executable, str(Path(__file__).resolve()),
                                       *(sys.argv[1:] if argv is None else argv)],
                             "parsed_arguments": {key: str(value) if isinstance(value, Path) else value
                                                  for key, value in vars(args).items()}},
              "requested_device": args.device, "requested_platform": selected,
              "claims_cuda_measurement": False, "physical_converged": False, "cases": []}
    unavailable = selected not in info["openmm_platforms"]
    failed = False
    for atoms in args.atoms:
        for beads in args.beads:
            if unavailable:
                case = {"model": args.model, "beads": beads, "atoms": atoms,
                        "device": args.device, "platform": selected,
                        "status": "unmeasured", "reason": f"OpenMM {selected} platform is unavailable"}
            else:
                try:
                    case = run_case(model_name=args.model, atoms=atoms, beads=beads,
                                    device=args.device, platform=selected, workers=args.workers,
                                    warmup=args.warmup, repeats=args.repeats, rounds=args.rounds, seed=args.seed)
                    failed |= any(mode["status"] != "measured" for mode in case["modes"].values())
                    report["claims_cuda_measurement"] |= (selected == "CUDA" and any(
                        mode["status"] == "measured" for mode in case["modes"].values()))
                except Exception as exc:
                    case = {"model": args.model, "beads": beads, "atoms": atoms,
                            "device": args.device, "platform": selected, "status": "failed",
                            "error_type": type(exc).__name__, "reason": str(exc)}
                    failed = True
            report["cases"].append(case)
            _atomic_json(args.output, report)
            print(f"{args.model} P={beads} N={atoms}: {case['status']}", flush=True)
    return 2 if unavailable or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
