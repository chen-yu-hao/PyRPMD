"""Offline PySCF -> original Caracal QMDFF/EVB/TREQ -> native PyRPMD PES.

An input path is reference data, not automatically a converged IRC. Fit residuals
are not a certification of transferability or of an RPMD rate.
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import tomllib

import numpy as np

from .build import bundle_native, load_build, run_command
from .formats import export_qmdff_reference, read_eshifts, read_xyz, validate_coordinates, write_grad_hess, write_xyz
from .reference import PySCFReference, PySCFSettings, ReferencePoint, atomic_json, canonical_hash, file_hash


def _only(table, allowed, name):
    unknown = set(table) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")


def load_spec(path):
    path = Path(path).resolve()
    spec = tomllib.loads(path.read_text())
    _only(spec, {"schema", "method", "endpoints", "path", "coordinates", "hessian_indices",
                 "optimize_endpoints", "gradient_tolerance", "native_build", "worker_timeout",
                 "qm", "fit", "validation", "force_step"}, "PES-generation")
    if spec.get("schema") != "pyrpmd.pesgen/1":
        raise ValueError("expected schema='pyrpmd.pesgen/1'")
    method = spec.get("method")
    if method not in ("qmdff", "de_evb", "treq"):
        raise ValueError("method must be qmdff, de_evb or treq")
    endpoints = spec.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != (1 if method == "qmdff" else 2):
        raise ValueError("qmdff requires one endpoint; EVB/TREQ require two ordered endpoints")
    files = {}
    def resolve(name):
        if not isinstance(name, str):
            raise ValueError("file paths must be strings")
        p = (path.parent / name).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        files[str(p)] = file_hash(p)
        return p
    endpoint_paths = [resolve(p) for p in endpoints]
    path_file = resolve(spec["path"]) if "path" in spec else None
    coord_file = resolve(spec["coordinates"]) if "coordinates" in spec else None
    if method != "qmdff" and path_file is None:
        raise ValueError("EVB/TREQ require a supplied ordered reference path")
    if method == "qmdff" and any(k in spec for k in ("path", "coordinates", "hessian_indices", "fit")):
        raise ValueError("path, coordinates, hessian_indices and fit are not QMDFF fields")
    if method == "de_evb" and any(k in spec for k in ("coordinates", "hessian_indices")):
        raise ValueError("coordinate/Hessian path fields are TREQ-only")
    if method == "treq" and coord_file is None:
        raise ValueError("TREQ requires an explicit nonredundant internal-coordinate definition")
    optimize = spec.get("optimize_endpoints", False)
    if type(optimize) is not bool:
        raise ValueError("optimize_endpoints must be Boolean")
    tolerance = float(spec.get("gradient_tolerance", 1e-5))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("gradient_tolerance must be positive, in hartree/bohr")
    timeout = float(spec.get("worker_timeout", 120))
    if not np.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("worker_timeout must be in (0,3600] seconds")
    step = float(spec.get("force_step", 1e-4))
    if not np.isfinite(step) or not 1e-7 <= step <= 1e-2 or ("force_step" in spec and method != "treq"):
        raise ValueError("force_step is TREQ-only, between 1e-7 and 1e-2 bohr")
    qm = PySCFSettings(**spec.get("qm", {}))
    # Charged systems can use the reference API, but the complete fitted
    # workflow awaits independent charged-QMDFF calibration.
    if qm.charge != 0:
        raise ValueError("this initial fitted-PES workflow supports neutral molecules only")
    fit = spec.get("fit", {})
    _only(fit, {"coupling"} if method == "de_evb" else {"exp_coefficient", "mid_total", "mid_transition"}, "fit")
    if method == "de_evb" and fit.get("coupling", "sp") not in ("sp", "1g"):
        raise ValueError("supported dE-EVB couplings are sp and 1g")
    if method == "treq":
        values = [float(fit.get("exp_coefficient", 35)), float(fit.get("mid_total", .6)), float(fit.get("mid_transition", .1))]
        if not np.isfinite(values).all() or values[0] <= 0 or not 0 < values[2] < values[1] < 1:
            raise ValueError("invalid TREQ decay/transition controls")
    validation = spec.get("validation", {})
    _only(validation, {"geometries", "energy_rmse_max", "force_rmse_max"}, "validation")
    heldout = resolve(validation["geometries"]) if "geometries" in validation else None
    if validation and set(validation) != {"geometries", "energy_rmse_max", "force_rmse_max"}:
        raise ValueError("held-out validation requires geometries and explicit energy/force RMSE bounds")
    for key in ("energy_rmse_max", "force_rmse_max"):
        if key in validation and (not np.isfinite(float(validation[key])) or float(validation[key]) <= 0):
            raise ValueError("validation bounds must be positive in atomic units")
    build = (path.parent / spec["native_build"]).resolve()
    native = load_build(build)
    fingerprint = canonical_hash({"spec": spec, "files": files, "build": native,
                                  "pyscf": importlib.metadata.version("pyscf"),
                                  "implementation": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}})
    return dict(spec=spec, path=path, method=method, endpoints=endpoint_paths,
                path_file=path_file, coordinates=coord_file, heldout=heldout,
                build=build, qm=qm, fingerprint=fingerprint, input_files=files,
                gradient_tolerance=tolerance, timeout=timeout)


def _reference(engine, symbols, q, cache, *, hessian, populations, optimize=False, tolerance=1e-5):
    key = canonical_hash({"symbols": symbols, "q": np.asarray(q).tolist(), "settings": asdict(engine.settings),
                          "pyscf": importlib.metadata.version("pyscf"),
                          "reference_code_sha256": file_hash(Path(__file__).with_name("reference.py")), "hessian": hessian,
                          "populations": populations, "optimize": optimize, "tolerance": tolerance})
    path = cache / (key + ".json")
    if path.exists():
        point = ReferencePoint.load(path)
        if point.provenance.get("cache_key") != key:
            raise ValueError("reference cache key mismatch")
        return point
    if optimize:
        q = engine.optimize(symbols, q, gradient_tolerance=tolerance)
    point = engine.calculate(symbols, q, hessian=hessian, populations=populations)
    point.provenance["cache_key"] = key
    point.save(path)
    return point


def _minimum(point, tolerance):
    if np.max(np.abs(point.gradient)) > tolerance:
        raise ValueError("QMDFF endpoint is not stationary at the requested gradient tolerance")
    if np.linalg.eigvalsh(point.hessian).min() < -1e-5:
        raise ValueError("QMDFF endpoint has negative curvature; supply/optimize a minimum, not a TS")


def fit_model(build, endpoints, path_points, stage, *, method, fit=None, coordinates=None, derivative_points=None):
    """Run original Caracal fitting on validated reference objects.

    Useful independently of PySCF: other engines may supply explicit ReferencePoint
    objects with provenance. No generic harmonic substitute is used.
    """
    stage, build = Path(stage), Path(build)
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    fit = fit or {}
    expected = 1 if method == "qmdff" else 2
    if method not in ("qmdff", "de_evb", "treq") or len(endpoints) != expected:
        raise ValueError("invalid model method/endpoint count")
    load_build(build)
    symbols = endpoints[0].symbols
    if any(p.symbols != symbols for p in list(endpoints) + list(path_points)):
        raise ValueError("reference atom identity/order mismatch")
    shifts, names = [], []
    for index, point in enumerate(endpoints):
        directory = stage / f"endpoint-{index}"; directory.mkdir()
        export_qmdff_reference(point, directory / "minimum")
        (directory / "generate.key").write_text("software O\n1qmdff minimum\n")
        run_command([str(build / "Caracal/bin/qmdffgen.x"), "generate.key"], directory, directory / "generate.log")
        artifact = directory / "minimum.qmdff"
        if not artifact.is_file() or artifact.stat().st_size < 40:
            raise RuntimeError("Caracal did not generate a QMDFF; inspect generate.log")
        name = f"minimum{index}.qmdff"; names.append(name)
        shutil.copy2(artifact, stage / name)
        shifts += read_eshifts(directory / "caracal.key", 1)
    key = f"pes {method}\nxyzstart start.xyz\nqmdff {{\n ffnames {' '.join(names)}\n"
    key += " eshift " + " ".join(f"{v:.17e}" for v in shifts) + "\n}\n"
    write_xyz(stage / "start.xyz", symbols, endpoints[0].positions)
    extras = []
    if method != "qmdff":
        if len(path_points) < 3:
            raise ValueError("EVB/TREQ require at least three ordered path frames")
        write_xyz(stage / "path.xyz", symbols, [p.positions for p in path_points])
        (stage / "path_energies.dat").write_text("\n".join(f"{p.energy:.17e}" for p in path_points) + "\n")
        extras += ["path.xyz", "path_energies.dat"]
    if method == "de_evb":
        coupling = fit.get("coupling", "sp")
        if coupling not in ("sp", "1g"):
            raise ValueError("unsupported dE-EVB coupling")
        key += f"de_evb {{\n coupling {coupling}\n}}\n"
        key = key.replace("\nqmdff {", "\nenergies_ref path_energies.dat\ncoords_ref path.xyz\nqmdff {")
        key = key.replace("\n eshift ", "\n shift_manual\n eshift ")
        (stage / "fit.key").write_text(key)
        run_command([str(build / "Caracal/bin/evbopt.x"), "fit.key"], stage, stage / "fit.log")
        if not (stage / "evb_pars.dat").is_file():
            raise RuntimeError("EVB optimization did not produce evb_pars.dat")
        extras += ["evb_pars.dat"]
    elif method == "treq":
        rows = validate_coordinates(coordinates, len(symbols))
        if any(np.linalg.matrix_rank(p.positions - p.positions.mean(axis=0), tol=1e-8) < 2 for p in path_points):
            raise ValueError("TREQ workflow currently excludes linear reference geometries")
        (stage / "coord_def.inp").write_text("\n".join(" ".join(map(str, row)) for row in rows)+"\n")
        derivative_points = list(derivative_points or [])
        if len(derivative_points) < 3:
            raise ValueError("TREQ requires at least three ordered gradient/Hessian reference points")
        write_grad_hess(derivative_points, stage / "grad_hess.dat")
        coefficient = float(fit.get("exp_coefficient", 35))
        total, transition = float(fit.get("mid_total", .6)), float(fit.get("mid_transition", .1))
        if not np.isfinite([coefficient,total,transition]).all() or coefficient <= 0 or not 0 < transition < total < 1:
            raise ValueError("invalid TREQ controls")
        key += (f"treq {{\n points {len(derivative_points)}\n irc_struc path.xyz\n irc_ens path_energies.dat\n"
                f" rp_exp_coeff {coefficient:.12g}\n rp_evb_mid {total:.12g} {transition:.12g}\n read_coord\n shift_manual\n}}\n")
        extras += ["coord_def.inp", "grad_hess.dat"]
    (stage / "model.key").write_text(key)
    return {name: (stage / name).read_text() for name in ["model.key", "start.xyz", *names, *extras]}


def evaluate_errors(library, points):
    # Validate in a fresh interpreter: dlopen keeps a library's original pathname
    # alive even after ctypes objects are collected. Keeping this outside the
    # parent permits the validated bundle directory to be renamed atomically.
    import sys
    with tempfile.TemporaryDirectory(prefix="pyrpmd-pes-check-") as td:
        td = Path(td)
        atomic_json(td / "coordinates.json", [p.positions.tolist() for p in points])
        code = """import json,sys,numpy as np
from pathlib import Path
from pyrpmd.potentials.cabi import CABIPotential
p=CABIPotential(sys.argv[1])
try:
    q=np.asarray(json.loads(Path(sys.argv[2]).read_text()),dtype=float)
    r=p.evaluate(q)
    if np.any(r.status): raise RuntimeError('fitted PES failed on reference geometry')
    Path(sys.argv[3]).write_text(json.dumps({'energy':r.energy.tolist(),'forces':r.forces.tolist()},allow_nan=False))
finally: p.close()
"""
        try:
            run_command([sys.executable, "-c", code, str(Path(library).resolve()),
                         str(td / "coordinates.json"), str(td / "result.json")], td, td / "evaluate.log")
            result = json.loads((td / "result.json").read_text())
        except BaseException:
            if (td / "evaluate.log").exists():
                shutil.copy2(td / "evaluate.log", Path(library).parent / "evaluation-failure.log")
            raise
    energy = np.asarray(result["energy"]) - np.array([p.energy for p in points])
    forces = np.asarray(result["forces"]) + np.array([p.gradient for p in points])
    return {"samples": len(points), "energy_rmse_hartree": float(np.sqrt(np.mean(energy**2))),
            "force_rmse_hartree_per_bohr": float(np.sqrt(np.mean(forces**2))),
            "energy_error_hartree": energy.tolist(), "energy_zero": "absolute, no post-fit alignment"}


def verify_bundle(directory):
    directory = Path(directory)
    data = json.loads((directory / "bundle.json").read_text())
    if data.get("schema") != "pyrpmd.caracal-bundle/1":
        raise ValueError("unknown bundle schema")
    if file_hash(directory / "libpes.so") != data["library_sha256"]:
        raise ValueError("PES library changed")
    if file_hash(directory / "caracal_worker.x") != data["metadata"]["worker_sha256"]:
        raise ValueError("native worker changed")
    for name, digest in data["files"].items():
        if canonical_hash((directory / "model" / name).read_text()) != digest:
            raise ValueError(f"model source changed: {name}")
    return data


def generate(input_file, output, *, resume=False):
    """Run/restart an offline PES job, atomically caching complete QM references."""
    cfg = load_spec(input_file)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("output is not empty; use resume explicitly")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another PES generation process owns this directory") from exc
        manifest = output / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if not resume or old.get("fingerprint") != cfg["fingerprint"]:
                raise ValueError("cannot resume with changed input, reference files, QM version or native build")
            if old.get("status") == "complete":
                return load_completed_run(output)
        elif resume:
            raise ValueError("no PES-generation manifest to resume")
        record = {"schema": "pyrpmd.pesgen-run/1", "fingerprint": cfg["fingerprint"],
                  "input_files": cfg["input_files"], "spec": cfg["spec"], "status": "running"}
        atomic_json(manifest, record)
        cache = output / "references"; cache.mkdir(exist_ok=True)
        engine = PySCFReference(cfg["qm"])
        try:
            endpoints, symbols = [], None
            for name in cfg["endpoints"]:
                s, frames = read_xyz(name)
                if len(frames) != 1 or (symbols is not None and s != symbols):
                    raise ValueError("endpoints require one frame each with identical atom order")
                symbols = s
                point = _reference(engine, symbols, frames[0], cache, hessian=True, populations=True,
                                   optimize=cfg["spec"].get("optimize_endpoints", False), tolerance=cfg["gradient_tolerance"])
                _minimum(point, cfg["gradient_tolerance"])
                endpoints.append(point)
            if len(endpoints) == 2 and np.max(np.abs(endpoints[0].positions-endpoints[1].positions)) < 1e-8:
                raise ValueError("two distinct endpoint geometries are required")
            path_points, derivatives = [], []
            if cfg["path_file"] is not None:
                s, frames = read_xyz(cfg["path_file"])
                if s != symbols:
                    raise ValueError("path atom identity/order differs from endpoints")
                indices = cfg["spec"].get("hessian_indices", list(range(len(frames)))) if cfg["method"] == "treq" else []
                if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                        or indices != sorted(set(indices)) or any(not 0 <= i < len(frames) for i in indices)):
                    raise ValueError("hessian_indices must be sorted unique zero-based path indices")
                if cfg["method"] == "treq" and (len(indices) < 3 or indices[0] != 0 or indices[-1] != len(frames)-1):
                    raise ValueError("TREQ derivative indices must include both path ends and an interior point")
                for i, q in enumerate(frames):
                    point = _reference(engine, symbols, q, cache, hessian=i in indices, populations=False)
                    path_points.append(point)
                    if i in indices:
                        derivatives.append(point)
            if cfg["method"] == "treq" and int(np.argmax([p.energy for p in path_points])) not in indices:
                raise ValueError("TREQ derivative indices must include the maximum-energy path point")
            # Each attempt preserves previous logs. Only a fully evaluated bundle is committed.
            stage = Path(tempfile.mkdtemp(prefix="attempt-", dir=output))
            files = fit_model(cfg["build"], endpoints, path_points, stage / "fit", method=cfg["method"],
                              fit=cfg["spec"].get("fit"), coordinates=cfg["coordinates"], derivative_points=derivatives)
            lib = bundle_native(cfg["build"], files, symbols, stage / "bundle", charge=cfg["qm"].charge,
                                spin=cfg["qm"].spin, timeout=cfg["timeout"],
                                force_mode="fd" if cfg["method"] == "treq" else "analytic",
                                force_step=float(cfg["spec"].get("force_step", 1e-4)),
                                provenance={"run_fingerprint": cfg["fingerprint"], "qm": asdict(cfg["qm"]), "method": cfg["method"]})
            training = evaluate_errors(lib, path_points or endpoints)
            heldout, qualifies = None, False
            if cfg["heldout"]:
                s, frames = read_xyz(cfg["heldout"])
                if s != symbols:
                    raise ValueError("validation atom order mismatch")
                train = np.array([p.positions for p in endpoints + path_points])
                if any(np.any(np.max(np.abs(train-q), axis=(1,2)) < 1e-10) for q in frames):
                    raise ValueError("held-out frames must not duplicate training/endpoints")
                points = [_reference(engine, symbols, q, cache, hessian=False, populations=False) for q in frames]
                heldout = evaluate_errors(lib, points)
                bounds = cfg["spec"]["validation"]
                qualifies = (heldout["energy_rmse_hartree"] <= bounds["energy_rmse_max"] and
                             heldout["force_rmse_hartree_per_bohr"] <= bounds["force_rmse_max"])
            summary = {"schema": "pyrpmd.pesgen-summary/1", "method": cfg["method"], "symbols": list(symbols),
                       "force_mode": "fourth_order_energy_difference" if cfg["method"] == "treq" else "analytic",
                       "training": training, "heldout": heldout, "within_requested_validation_bounds": qualifies,
                       "validated_physical_accuracy": False, "converged_rate": False, "experimental": cfg["method"] == "treq",
                       "potential": {"provider": "cabi", "library": str(output / "bundle/libpes.so")},
                       "scope": "supplied single-surface molecular references; no TS/IRC discovery, no PES extrapolation guarantee"}
            # A crash between these writes is recoverable from the committed bundle + manifest.
            final = output / "bundle"
            if final.exists():
                verify_bundle(final)
                shutil.rmtree(final)
            (stage / "bundle").rename(final)
            atomic_json(output / "summary.json", summary)
            atomic_json(manifest, {**record, "status": "complete",
                                  "summary_sha256": file_hash(output / "summary.json"),
                                  "bundle_manifest_sha256": file_hash(output / "bundle/bundle.json")})
            return summary
        except BaseException as exc:
            atomic_json(manifest, {**record, "status": "failed", "error": str(exc)})
            raise


def load_completed_run(directory):
    directory = Path(directory)
    record = json.loads((directory / "manifest.json").read_text())
    if record.get("schema") != "pyrpmd.pesgen-run/1" or record.get("status") != "complete":
        raise ValueError("PES generation is not complete")
    for name, key in (("summary.json", "summary_sha256"), ("bundle/bundle.json", "bundle_manifest_sha256")):
        if file_hash(directory / name) != record.get(key):
            raise ValueError(f"PES-generation result changed: {name}")
    verify_bundle(directory / "bundle")
    return json.loads((directory / "summary.json").read_text())


def run_generated_job(directory, job, *, output, allow_unvalidated=False, resume=False):
    """Hand a fitted molecular PES to an existing JobSpec; never invent a rate setup."""
    from ..input import JobSpec
    from ..runner import run
    directory = Path(directory).resolve()
    summary = load_completed_run(directory)
    if type(allow_unvalidated) is not bool:
        raise ValueError("allow_unvalidated must be Boolean")
    if (not summary["within_requested_validation_bounds"] or summary["experimental"]) and not allow_unvalidated:
        raise ValueError("unvalidated/experimental PES; explicit allow_unvalidated is required for pilots")
    spec = JobSpec.from_file(job) if isinstance(job, (str,Path)) else JobSpec.from_mapping(job)
    spec.data["potential"] = {"provider": "cabi", "library": str(directory / "bundle/libpes.so")}
    # JobSpec and the ABI enforce molecular identity, charge, spin and no PBC.
    return run(spec, output=output, resume=resume)
