r"""Sensitivity runs for the frozen H + H2 state-selected reproduction driver.

Run one case into a new directory, for example::

    python -m benchmarks.h3.state_selected_sensitivity --kind timestep \
        --v 0 --etot 0.90 --samples 1024 --workers 4 --id-offset 100000 \
        --output runs/state_selected_reproduction/sensitivity/dt-v0-E090

Defaults are P=32, dt=0.01 fs for ``timestep``; P=64, dt=0.02 fs for
``beads``; and P=32, dt=0.02 fs for ``radial`` and ``annulus``. Explicit
P/bmax overrides are recorded, so combined parameter changes remain visible.
An annulus requires explicit bmin, bmax and id-offset; use IDs disjoint from
the reference batch for an independent comparison. At identical IDs the
timestep/radial variants share the original preparation random streams.
Changing P changes those streams because P is part of the driver's seed.

Marjollet & Welsch (2021), DOI 10.1002/qua.26447, Eq. 22 calls d(A,BC) a
distance, without fully specifying its Cartesian launch convention for b>0.
The frozen driver interprets d=10+Rg(A)+max(Rg(B),Rg(C)) as an axial distance.
The radial test instead makes the *norm* of the centroid separation equal d:
z=sqrt(d*d-b*b). Only constant fragment translations along z are changed;
momenta, impact parameter, molecular orientation and total COM are preserved.
This is a convention sensitivity test, not a claim about the authors' code.

Nonradial cases reuse driver.run_case. Radial cases use their own spawn pool
and the unmodified driver.trajectory. The radial worker initializer explicitly
rebinds driver.prepare_paper_state to the wrapper below *inside that worker*;
the saved original function is called first. No driver file, parent-process
preparation function, integrator, potential or native kernel is modified.
The hook and startup source/binary hashes are included in every run manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from benchmarks.h3 import state_selected_reproduction as driver


KINDS = ("timestep", "beads", "radial", "annulus")
_ORIGINAL_PREPARE = driver.prepare_paper_state
_SEED_RULE = (
    "SeedSequence([seed,v,round(Etot_eV*1e6),method_code,P,trajectory_id]).spawn(2)"
)


def reinterpret_launch_as_radial(state, launch):
    """Translate a fresh H/H2 launch in place, preserving p and internal q."""
    q, p = state.q, state.p
    if q.ndim != 3 or q.shape[1:] != (3, 3) or p.shape != q.shape:
        raise ValueError("Radial reinterpretation requires a three-H ring state")
    d, b = float(launch["initial_axial_bohr"]), float(launch["b_bohr"])
    if not (math.isfinite(d) and math.isfinite(b) and 0 <= b < d):
        raise ValueError("Radial launch requires finite 0 <= b < d")
    before_q, before_p = q.copy(), p.copy()
    before_separation = q[:, 1:].mean(axis=(0, 1)) - q[:, 0].mean(axis=0)
    if not (np.isclose(before_separation[2], d, rtol=0, atol=2e-12)
            and np.isclose(np.linalg.norm(before_separation[:2]), b,
                           rtol=0, atol=2e-12)):
        raise ValueError("Launch geometry does not match the axial metadata")
    axial = math.sqrt((d - b) * (d + b))
    delta = axial - d
    # R = COM(BC) - COM(A); the three H masses are equal.
    q[:, 0, 2] -= (2 / 3) * delta
    q[:, 1:, 2] += (1 / 3) * delta
    separation = q[:, 1:].mean(axis=(0, 1)) - q[:, 0].mean(axis=0)
    if not (np.array_equal(p, before_p)
            and np.array_equal(q[:, :, :2], before_q[:, :, :2])
            and np.allclose(q.mean(axis=1), before_q.mean(axis=1),
                            rtol=0, atol=2e-12)
            and np.allclose(q[:, 2] - q[:, 1], before_q[:, 2] - before_q[:, 1],
                            rtol=0, atol=2e-12)
            and np.allclose(q - q.mean(axis=0), before_q - before_q.mean(axis=0),
                            rtol=0, atol=2e-12)
            and np.isclose(np.linalg.norm(separation), d, rtol=0, atol=3e-12)):
        raise AssertionError("Radial launch failed its geometry/conservation checks")
    metadata = dict(launch)
    metadata.update(
        distance_convention="radial",
        original_axial_bohr=d,
        original_centroid_separation_bohr=float(np.linalg.norm(before_separation)),
        requested_radial_bohr=d,
        initial_axial_bohr=float(axial),
        initial_centroid_separation_bohr=float(np.linalg.norm(separation)),
        initial_separation_vector_bohr=separation.tolist(),
        radial_translation_delta_z_bohr=float(delta),
    )
    return state, metadata


def _prepare_radial_state(config, prepared, trajectory_id):
    state, launch = _ORIGINAL_PREPARE(config, prepared, trajectory_id)
    return reinterpret_launch_as_radial(state, launch)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_snapshot(files):
    for entry in files:
        if _sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Source/binary changed during sensitivity run: {entry['path']}")


def _source_snapshot():
    """Read actual sources and binaries before reserving or starting a case."""
    root = driver.ROOT
    native = importlib.import_module("pyrpmd._native")
    build_path = root / "benchmarks/h3/build/libbkmp2.build.json"
    build = json.loads(build_path.read_text())
    source = root / "benchmarks/h3/build/h3_bkmp2.f"
    library = root / "benchmarks/h3/build/libbkmp2.so"
    if (_sha256(source) != build["source"]["sha256"]
            or _sha256(library) != build["library_sha256"]
            or _sha256(root / "benchmarks/h3/wrapper.f90") != build["wrapper_sha256"]):
        raise RuntimeError("BKMP2 source/wrapper/library does not match its build manifest")
    paths = [Path(__file__), Path(driver.__file__), Path(native.__file__), source,
             library, build_path, root / "benchmarks/h3/source.json",
             root / "benchmarks/h3/wrapper.f90", root / "benchmarks/h3/pes.py",
             root / "benchmarks/h3/drpmd.py"]
    # An installed package may be imported instead of this checkout's src tree.
    # Record the files Python actually loaded, never an assumed source location.
    paths += [Path(importlib.import_module(name).__file__) for name in (
        "pyrpmd.initial_states", "pyrpmd.dynamics", "pyrpmd.propagation",
        "pyrpmd.constants", "pyrpmd.workflows.drpmd")]
    paths += sorted(p for p in (root / "native").rglob("*")
                    if p.suffix in (".h", ".hpp", ".cpp") and p.is_file())
    files = [{"path": str(p.resolve()), "sha256": _sha256(p)} for p in paths]
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_sha256": _sha256(__file__),
        "driver_sha256": _sha256(driver.__file__),
        "model_source_sha256": _sha256(source),
        "pes": build,
        "files": files,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__,
    }


def _init_radial_worker(config, files):
    _verify_snapshot(files)
    if driver.prepare_paper_state is not _ORIGINAL_PREPARE:
        raise RuntimeError("Unexpected preparation override before radial initialization")
    driver._init_worker(config)
    driver.prepare_paper_state = _prepare_radial_state


def _write_json(path, value, *, new=False):
    path = Path(path)
    target = path if new else path.with_name(path.name + ".tmp")
    with target.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    if not new:
        os.replace(target, path)


def _run_radial_case(config, workers, output, files):
    """Reuse the original trajectory and summary; customize only preparation."""
    _write_json(output / "config.json", asdict(config), new=True)
    started = time.perf_counter()
    rows = []
    with (output / "trajectories.jsonl").open("x") as stream:
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_radial_worker, initargs=(config, files)) as pool:
            futures = [pool.submit(driver.trajectory, i)
                       for i in range(config.id_offset, config.id_offset + config.samples)]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if len(rows) % max(32, config.samples // 8) == 0:
                    stream.flush()
                    print(json.dumps({"kind": "radial", "case": config.label,
                                      "complete": len(rows), "total": config.samples}), flush=True)
    report = driver.summarize(config, rows)
    report.update(workers=workers, elapsed_seconds=time.perf_counter() - started)
    return report


def _validate(kind, config, workers):
    if kind not in KINDS or config.method != "ne_rpmd" or config.v not in (0, 1):
        raise ValueError("Require a supported kind and NE-RPMD with v=0 or v=1")
    if (config.samples < 1 or workers < 1 or config.id_offset < 0 or config.seed < 0
            or config.beads < 2):
        raise ValueError("Require samples/workers >=1, seed/id-offset >=0, and beads >=2")
    if not (all(math.isfinite(x) for x in (config.total_energy_ev, config.bmin,
                                          config.bmax, config.dt_fs))
            and 0 <= config.bmin < config.bmax):
        raise ValueError("Require finite total energy and 0 <= bmin < bmax")
    if kind == "annulus" and config.bmin <= 0:
        raise ValueError("Annulus sensitivity requires bmin > 0")
    if kind != "annulus" and config.bmin != 0:
        raise ValueError("Nonzero bmin is reserved for the annulus sensitivity")
    if kind == "radial" and config.bmax >= config.initial_axial_bohr:
        raise ValueError("Radial sensitivity requires bmax < the base distance (10 bohr)")
    if not 0 < config.dt_fs <= config.time_cap_fs <= config.extended_cap_fs:
        raise ValueError("Require 0 < dt <= time cap <= extended cap")
    if not (config.collision_energy > 0 and math.isfinite(config.beta) and config.beta > 0):
        raise ValueError("Invalid collision energy or beta")


def run_sensitivity(kind, config, workers, output):
    """Run an explicitly identified variant; never reuse an existing directory."""
    _validate(kind, config, workers)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing sensitivity directory: {output}")
    snapshot = _source_snapshot()
    manifest = {
        "schema": "pyrpmd.h3.state-selected-sensitivity.v1", "status": "running",
        "kind": kind, "config": asdict(config), "workers": workers,
        "paper_doi": "10.1002/qua.26447", "startup_sources": snapshot,
        "seed_rule": _SEED_RULE,
        "trajectory_ids": {"first": config.id_offset,
                           "stop_exclusive": config.id_offset + config.samples},
        "random_stream_relationship": (
            "Same seed/v/energy/method/P/id reproduces the same preparation and launch draws. "
            "dt, radial convention and impact-parameter interval are absent from the seed. "
            "Use disjoint ID intervals for independent batches; overlap is not checked against "
            "external reference runs. Changing P changes the random stream."),
        "distance_convention": "radial" if kind == "radial" else "axial",
        "distance_definition": (
            "d=10+Rg(A)+max(Rg(B),Rg(C)); axial z=sqrt(d^2-b^2), centroid separation=d"
            if kind == "radial" else
            "d=10+Rg(A)+max(Rg(B),Rg(C)); axial z=d, centroid separation=sqrt(d^2+b^2)"),
        "radial_worker_hook": kind == "radial",
        "propagation": "Unmodified driver.trajectory and native dynamics/PES/outcome checks",
        "impact_sampling": "b=sqrt(bmin^2+(bmax^2-bmin^2)*U); equal area and equal weight",
        "annulus_pooling": "An annulus and a disk cannot be pooled as equal-weight raw rows",
        "reference_controls": {"beads": 32, "dt_fs": 0.02,
                               "distance_convention": "axial"},
    }
    # Atomic directory reservation protects config, ledger, summary and manifest together.
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "sensitivity_manifest.json"
    _write_json(manifest_path, manifest, new=True)
    try:
        if kind == "radial":
            report = _run_radial_case(config, workers, output, snapshot["files"])
        else:
            report = driver.run_case(config, workers, output)
        _verify_snapshot(snapshot["files"])
        if kind == "annulus":
            counts = report["counts"]
            possible = counts["product"] + counts["unresolved"] + counts["failed"]
            n = report["launched_attempts"]
            upper = 1.0 if possible == n else float(
                driver.beta_distribution.ppf(.95, possible + 1, n - possible))
            report["annulus_upper_bound"] = {
                "confidence": .95, "sidedness": "one-sided",
                "unknown_handling": "all unresolved and failed attempts treated as reactive",
                "probability_upper": upper,
                "sigma_upper_bohr2": report["area_bohr2"] * upper,
                "sigma_upper_angstrom2": report["area_bohr2"] * upper * driver.BOHR_ANGSTROM ** 2,
                "zero_event_formula_applies": possible == 0,
            }
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                        source_snapshot_verified_at_end=True)
        report["sensitivity"] = manifest
        report.setdefault("provenance", {}).update(
            paper_doi="10.1002/qua.26447", seed_rule=_SEED_RULE,
            wrapper_sha256=snapshot["wrapper_sha256"],
            driver_sha256_at_start=snapshot["driver_sha256"],
            model_source_sha256_at_start=snapshot["model_source_sha256"],
            pes=snapshot["pes"],
            method=("2021 triatomic preparation with explicit radial-distance sensitivity"
                    if kind == "radial" else "2021 triatomic preparation; axial launch convention"))
        _write_json(output / "summary.json", report)
        _write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat(),
                        error=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        raise
    print(json.dumps({"kind": kind, "output": str(output), "counts": report["counts"],
                      "sigma_angstrom2": report["sigma_angstrom2"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--v", type=int, choices=(0, 1), required=True)
    parser.add_argument("--etot", type=float, required=True, help="Total energy in eV")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--id-offset", type=int)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--bmin", type=float)
    parser.add_argument("--bmax", type=float)
    parser.add_argument("--beads", type=int)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for this single case; must not already exist")
    args = parser.parse_args(argv)
    if args.kind == "annulus" and any(
            getattr(args, name) is None for name in ("bmin", "bmax", "id_offset")):
        parser.error("annulus requires explicit --bmin, --bmax and --id-offset")
    config = driver.Config(
        v=args.v, total_energy_ev=args.etot, samples=args.samples, seed=args.seed,
        id_offset=0 if args.id_offset is None else args.id_offset,
        bmin=0.0 if args.bmin is None else args.bmin,
        bmax=3.5 if args.bmax is None else args.bmax,
        beads=args.beads if args.beads is not None else (64 if args.kind == "beads" else 32),
        dt_fs=0.01 if args.kind == "timestep" else 0.02,
    )
    try:
        _validate(args.kind, config, args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    run_sensitivity(args.kind, config, args.workers, args.output)


if __name__ == "__main__":
    main()
