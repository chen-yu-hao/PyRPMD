"""pyrpmd command line: one interface for Python and TOML inputs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from . import __version__
from .input import JobSpec, InputError
from .io import jsonable, write_json


def _input_options(p):
    p.add_argument("input", help="TOML .inp/.toml or Python file defining build()")
    p.add_argument("--pes", help="C ABI .so or .o with a manifest")
    p.add_argument("--pes-manifest")
    p.add_argument("--temperature", help="Quantity, e.g. '300 K'")
    p.add_argument("--beads", type=int)
    p.add_argument("--workers", type=int, help="Concurrent trajectory/window workers")


def parser():
    p = argparse.ArgumentParser(
        prog="pyrpmd",
        description="PyRPMD: ring-polymer dynamics, reaction networks, enhanced sampling and PCET models",
    )
    p.add_argument("--version", action="version", version=f"PyRPMD {__version__}")
    commands = p.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run", help="Execute a checked scientific task")
    _input_options(r)
    r.add_argument("--output")
    r.add_argument("--resume", action="store_true")
    v = commands.add_parser(
        "validate", help="Resolve input, units, system and resources"
    )
    _input_options(v)
    v.add_argument("--check-pes", action="store_true")
    v.add_argument("--write-resolved")
    q = commands.add_parser(
        "prepare", help="Prepare a task; --pilot permits numerical TS/PES checks"
    )
    _input_options(q)
    q.add_argument("--pilot", action="store_true")
    q.add_argument("--output", help="Write preparation JSON")
    c = commands.add_parser(
        "resume", help="Resume using a saved resolved input and exact physics identity"
    )
    c.add_argument("directory")
    c.add_argument("--workers", type=int)
    c.add_argument("--loop-backend", choices=("auto", "native", "python"))
    c.add_argument("--block-steps", type=int)
    a = commands.add_parser(
        "analyze", help="Read results or reanalyze saved free-energy/transport data"
    )
    a.add_argument("directory")
    a.add_argument(
        "--recompute",
        action="store_true",
        help="Reanalyze saved umbrella, metadynamics or diffusion data without new dynamics",
    )
    a.add_argument(
        "--bins", nargs="+", type=int, help="FES bins per CV (implies --recompute)"
    )
    a.add_argument(
        "--bootstrap",
        type=int,
        help="Analysis bootstrap replicates (implies --recompute)",
    )
    a.add_argument("--fit-range", nargs=2, metavar=("START", "STOP"),
                   help="Diffusion lag-time fit range, e.g. '10 fs' '100 fs'")
    a.add_argument("--axes", nargs="+", type=int, choices=(0, 1, 2),
                   help="Diffusion Cartesian axes (0=x, 1=y, 2=z)")
    a.add_argument("--atoms", nargs="+", type=int, help="Diffusion atom indices")
    a.add_argument("--max-lag", type=int, help="Maximum diffusion lag in saved frames")
    commands.add_parser("info", help="Report version and compiled native capabilities")
    study = commands.add_parser("study", help="Run temperature/isotope and numerical-convergence rate scans")
    study.add_argument("input", help="Study JSON or TOML file referencing an existing rate job")
    study.add_argument("--output", help="Study output directory; required unless --plan")
    study.add_argument("--resume", action="store_true")
    study.add_argument("--plan", action="store_true", help="Resolve and print cases without dynamics")
    checkpoint = commands.add_parser("checkpoint", help="Verify a portable NPZ checkpoint")
    checkpoint.add_argument("path")
    checkpoint.add_argument(
        "--require-integrity", action="store_true",
        help="Reject legacy files that predate the checksum envelope",
    )
    pes = commands.add_parser("pes", help="Inspect, build or test a potential adapter")
    sub = pes.add_subparsers(dest="pes_command", required=True)
    b = sub.add_parser("build")
    b.add_argument("object")
    b.add_argument("--manifest", required=True)
    b.add_argument("--output")
    i = sub.add_parser("inspect")
    i.add_argument("library")
    i.add_argument("--manifest")
    i.add_argument("--geometry")
    t = sub.add_parser("check")
    t.add_argument("library")
    t.add_argument("--manifest")
    t.add_argument("--geometry", required=True)
    t.add_argument(
        "--step",
        type=float,
        default=1e-5,
        help="Finite-difference displacement in bohr",
    )
    t.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute force tolerance in hartree/bohr",
    )
    return p


def _job(args):
    s = JobSpec.from_file(args.input)
    options = {
        k: getattr(args, k, None)
        for k in ("pes", "pes_manifest", "temperature", "beads", "workers")
    }
    if args.command == "run":
        options["output"] = args.output
    return s.with_overrides(**options)


def _print(value):
    from .parallel import is_root

    if is_root():
        print(
            json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        )


def _pes(args):
    from .potentials import make_potential
    from .system import read_xyz

    if args.pes_command == "build":
        from .potentials.cabi import build_object

        library = Path(build_object(args.object, args.manifest))
        if args.output:
            import shutil

            target = Path(args.output).resolve()
            if target.exists():
                raise InputError(f"Output already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)
            library = target
        return {"library": str(library)}
    system = read_xyz(args.geometry) if args.geometry else None
    provider = make_potential(
        {
            "provider": "cabi",
            "library": str(Path(args.library).resolve()),
            "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        },
        system,
    )
    try:
        if args.pes_command == "inspect":
            return {
                "metadata": getattr(provider, "metadata", {}),
                "library": args.library,
            }
        if (
            not np.isfinite(args.step)
            or not np.isfinite(args.tolerance)
            or args.step <= 0
            or args.tolerance <= 0
        ):
            raise InputError("step and tolerance must be positive and finite")
        q = system.geometry[None, :, :].copy()
        result = provider.evaluate(q)
        from .potentials import validate_result

        validate_result(result, q.shape)
        finite = np.empty_like(q)
        for a in range(system.natoms):
            for c in range(3):
                qp = q.copy()
                qm = q.copy()
                qp[0, a, c] += args.step
                qm[0, a, c] -= args.step
                plus = validate_result(provider.evaluate(qp), qp.shape)
                minus = validate_result(provider.evaluate(qm), qm.shape)
                finite[0, a, c] = -(plus.energy[0] - minus.energy[0]) / (2 * args.step)
        error = float(np.max(np.abs(result.forces - finite)))
        if not np.isfinite(error) or error > args.tolerance:
            raise InputError(
                f"PES energy/force finite-difference check failed: max error {error:g}"
            )
        return {
            "passed": True,
            "max_force_error_au": error,
            "energy_hartree": result.energy.tolist(),
            "tolerance": args.tolerance,
        }
    finally:
        provider.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            from .runner import run

            result = run(_job(args), resume=args.resume)
            if result is not None:
                _print(result)
        elif args.command == "validate":
            spec = _job(args)
            config, system = spec.resolve()
            if args.check_pes:
                from .runner import prepare

                result = prepare(spec, numerical=True)
            else:
                from .parallel import execution_plan

                if config["task"] == "fes":
                    from .workflows.fes import resolve_umbrellas

                    resolve_umbrellas(config, system)
                if config["task"] == "pcet":
                    from .workflows.pcet import prepare_pcet

                    result = {
                        "valid": True, "task": "pcet",
                        "model": prepare_pcet(config, system, numerical=False),
                        "temperature_K": config["temperature_K"],
                        "warnings": config["_warnings"],
                    }
                else:
                    result = {
                        "valid": True,
                        "task": config["task"],
                        "natoms": system.natoms,
                        "temperature_K": config["temperature_K"],
                        "beads": config["beads"],
                        "parallel": execution_plan(config["parallel"]).__dict__,
                        "warnings": config["_warnings"],
                    }
            if args.write_resolved:
                from .parallel import root_call

                root_call(
                    write_json,
                    args.write_resolved,
                    {"config": config, "model" if config["task"] == "pcet" else "system": system.to_dict()},
                )
            _print(result)
        elif args.command == "prepare":
            from .runner import prepare

            result = prepare(_job(args), numerical=args.pilot)
            if args.output:
                from .parallel import root_call

                root_call(write_json, args.output, result)
            _print(result)
        elif args.command == "resume":
            from .runner import resume_run

            override = (
                {"trajectory_workers": args.workers, "trajectory_groups": args.workers}
                if args.workers is not None
                else None
            )
            execution = {
                key: value for key, value in {
                    "loop_backend": args.loop_backend,
                    "block_steps": args.block_steps,
                }.items() if value is not None
            }
            result = resume_run(args.directory, parallel=override, execution=execution or None)
            if result is not None:
                _print(result)
        elif args.command == "analyze":
            from .runner import analyze_run

            _print(
                analyze_run(
                    args.directory,
                    recompute=args.recompute,
                    bins=args.bins,
                    bootstrap=args.bootstrap,
                    diffusion={key: value for key, value in {
                        "fit_range": args.fit_range, "axes": args.axes,
                        "atom_indices": args.atoms, "max_lag": args.max_lag,
                    }.items() if value is not None} or None,
                )
            )
        elif args.command == "info":
            from .dynamics import native_capabilities

            _print(
                {
                    "name": "PyRPMD",
                    "version": __version__,
                    "native": native_capabilities(),
                    "workflows": ["rate", "drpmd", "fes", "network", "metadynamics", "pcet", "pimd", "diffusion", "site_rate", "surface_rate"],
                    "materials": {"cell": "fixed", "trajectory": "unwrapped_centroid", "pressure_ensemble": False},
                    "integrators": ["cayley-bcb", "normal-mode-bab", "velocity-verlet"],
                    "enhanced_sampling": ["adaptive_umbrella", "standard_metadynamics", "well_tempered_metadynamics"],
                    "scientific_tools": ["general_centroid_site_rates", "temperature_isotope_studies", "ring_polymer_instanton", "takahashi_imada_thermodynamics", "open_path_momentum"],
                    "input_schema_version": 1,
                    "run_manifest_schema_version": 2,
                    "checkpoint_schema_version": 2,
                    "input_formats": ["TOML .inp", "Python build()"],
                }
            )
        elif args.command == "study":
            from .studies import plan_rate_study, run_rate_study

            if args.plan:
                if args.resume:
                    raise InputError("--plan and --resume cannot be combined")
                plan = plan_rate_study(args.input)
                _print({**plan, "cases": [case.to_dict() for case in plan["cases"]]})
            else:
                if not args.output:
                    raise InputError("study requires --output, or --plan to inspect the cases")
                _print(run_rate_study(args.input, output=args.output, resume=args.resume))
        elif args.command == "pes":
            _print(_pes(args))
        elif args.command == "checkpoint":
            from .checkpoint import inspect_checkpoint

            _print(inspect_checkpoint(args.path, require_integrity=args.require_integrity))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        from .parallel import is_root

        if is_root():
            print(f"PyRPMD error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
