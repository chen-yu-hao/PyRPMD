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


def _report_shortcut_options(p):
    p.add_argument("--report", action="store_true", help="Generate saved-result figures and an HTML/JSON report (requires the plots extra)")
    p.add_argument("--report-output", help="Report directory (default: RUN/analysis-report)")
    p.add_argument("--report-overwrite", action="store_true", help="Replace an existing generated report, preserving source results")
    p.add_argument("--report-style", choices=("jctc", "nature"), help="Figure style for --report (default: jctc)")


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
    _report_shortcut_options(r)
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
    state = commands.add_parser(
        "state-check", help="Check prepared vibrational states on the full PES without a collision kick"
    )
    _input_options(state)
    state.add_argument("--time", default="100 fs", help="NVE duration, e.g. '100 fs'")
    state.add_argument("--samples", type=int, default=8, help="Independent initial states")
    state.add_argument("--stride", type=int, default=10, help="Observe every N steps, plus initial/final frames")
    state.add_argument("--output", help="JSON report file relative to the current working directory")
    c = commands.add_parser(
        "resume", help="Resume using a saved resolved input and exact physics identity"
    )
    c.add_argument("directory")
    c.add_argument("--workers", type=int)
    c.add_argument("--loop-backend", choices=("auto", "native", "python"))
    c.add_argument("--block-steps", type=int)
    c.add_argument("--adaptive-max-rounds", type=int, help="Extend an exhausted adaptive pilot budget")
    c.add_argument("--adaptive-max-windows", type=int)
    c.add_argument("--adaptive-max-pilot-steps", type=int)
    extend = commands.add_parser(
        "extend", help="Raise the total time budget of unresolved dRPMD trajectories"
    )
    extend.add_argument("directory")
    extend.add_argument("--max-time", required=True,
                        help="New total duration from launch, e.g. '20 ps' (not added time)")
    a = commands.add_parser(
        "analyze", help="Read results or reanalyze saved free-energy/transport data"
    )
    a.add_argument("directory")
    a.add_argument(
        "--recompute",
        action="store_true",
        help="Reanalyze saved umbrella, metadynamics, OPES or diffusion data without new dynamics",
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
    rate = commands.add_parser(
        "rate-postprocess", help="Compute rates from saved PMF/FES and signed channel fluxes"
    )
    rate.add_argument("directory", help="Saved gas-rate or FES output directory")
    rate.add_argument("--spec", help="JSON specification for a two-dimensional dividing surface")
    rate.add_argument("--output", help="New result JSON (default: DIRECTORY/rate_postprocess.json)")
    rate.add_argument("--bootstrap", type=int, default=200)
    rate.add_argument("--seed", type=int, default=0)
    projection = commands.add_parser("project", help="Boltzmann-marginalize a saved CV-density FES")
    projection.add_argument("directory")
    projection.add_argument("--axes", type=int, nargs="+", required=True, help="Retained CV axes, zero based")
    projection.add_argument("--selection", help="Boolean .npy channel mask with the complete source FES shape")
    projection.add_argument("--output", required=True)
    _report_shortcut_options(a)
    report = commands.add_parser("report", help="Automatically analyze saved results and produce scientific figures and reports")
    report.add_argument("directories", nargs="+", metavar="RUN")
    report.add_argument("--output", "-o", help="Report destination; required for multiple runs")
    report.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    report.add_argument("--dpi", type=int, default=300, help="Raster DPI (default: 300 for color figures)")
    report.add_argument("--style", choices=("jctc", "nature"), default="jctc", help="Figure style and publication size (default: jctc)")
    report.add_argument("--energy-unit", choices=("kcal/mol", "kJ/mol", "eV", "hartree"), default="kcal/mol")
    report.add_argument("--length-unit", choices=("angstrom", "bohr", "nm"), default="angstrom")
    report.add_argument("--labels", nargs="+", help="One display label per run")
    report.add_argument("--series", nargs="+", help="One comparison group per run (same group may occur several times)")
    report.add_argument("--reference", help="CSV with series,temperature_K,rate,rate_unit,source and optional standard_error")
    report.add_argument("--include-estimates", action="store_true", help="Also show explicitly labelled withheld diagnostic rate estimates")
    report.add_argument("--fit-arrhenius", action="store_true", help="Opt-in descriptive log-rate fits, with >=3 distinct temperatures")
    report.add_argument("--kie", nargs=2, metavar=("NUMERATOR_SERIES", "DENOMINATOR_SERIES"))
    report.add_argument("--convergence", choices=("beads", "dt_fs", "rinf_bohr", "bmax_bohr", "samples"))
    report.add_argument("--overwrite", action="store_true", help="Replace only an owned report with an intact generated-file inventory")
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


def _rate_postprocess_summary(result):
    """Keep full curves/bootstrap in the output file, not thousands of stdout lines."""
    summary = {key: value for key, value in result.items()
               if key not in ("channels", "specification", "source_rate_uncertainty")}
    channels = result.get("channels")
    summary["channels"] = None if channels is None else {
        key: channels[key] for key in (
            "status", "channel_names", "kappa_estimates", "total_kappa",
            "channel_rates_cm3_s", "channel_rate_estimates_cm3_s",
            "total_rate_estimate_cm3_s", "branching_fractions",
            "branching_fraction_estimates", "branching_fraction_standard_error",
            "plateau_detected", "parent_configurations", "independent_parent_chains",
            "rate_uncertainty", "reasons", "converged", "physical_scope",
        ) if key in channels
    }
    return summary


def _report_summary(report):
    return {"output": report["output"], "html": str(Path(report["output"]) / "index.html"),
            "json": str(Path(report["output"]) / "report.json"), "figures": len(report["figures"]),
            "findings": len(report["findings"]), "skipped": report["skipped"]}


def _report_after_result(args, result, directory):
    if not args.report and (args.report_output or args.report_overwrite or args.report_style):
        raise InputError("--report-output, --report-overwrite and --report-style require --report")
    from .parallel import is_root
    if args.report and is_root() and result is not None:
        try:
            from .analysis.reporting import generate_report
            report = generate_report(directory, output=args.report_output, overwrite=args.report_overwrite,
                                     style=args.report_style or "jctc")
        except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
            import shlex

            command = ["pyrpmd", "report", str(directory)]
            if args.report_output:
                command.extend(["--output", args.report_output])
            if args.report_overwrite:
                command.append("--overwrite")
            if args.report_style:
                command.extend(["--style", args.report_style])
            stage = "Simulation" if args.command == "run" else "Analysis"
            raise RuntimeError(
                f"{stage} completed; numerical results are saved in {directory}. "
                f"Report generation failed: {error}. Fix the reporting issue, "
                f"then retry only the saved-data report: {shlex.join(command)}"
            ) from error
        return {**result, "analysis_report": _report_summary(report)}
    return result


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
        from .potentials import check_potential, validate_result

        checked = check_potential(provider, q, epsilon=args.step)
        error = checked["max_absolute_force_error"]
        if not np.isfinite(error) or error > args.tolerance:
            raise InputError(
                f"PES energy/force finite-difference check failed: max error {error:g}"
            )
        # Re-evaluate the original geometry: the diagnostic's displaced calls
        # may have overwritten provider-owned energy and result buffers.
        result = validate_result(provider.evaluate(q), q.shape)
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
        if args.command in ("run", "analyze") and not args.report and (args.report_output or args.report_overwrite or args.report_style):
            raise InputError("--report-output, --report-overwrite and --report-style require --report")
        if args.command == "run":
            from .runner import run

            result = run(_job(args), resume=args.resume)
            if result is not None:
                result = _report_after_result(args, result, result["output"])
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
            if config.get("initial_ensemble"):
                result["temperature_K"] = None
                result["beta_au"] = config["beta"]
                result["temperature_interpretation"] = config["temperature_interpretation"]
                result["initial_ensemble_method"] = config["initial_ensemble"]["method"]
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
        elif args.command == "state-check":
            from .parallel import root_call
            from .workflows.state_selected import check_state_selected

            result = root_call(
                check_state_selected, _job(args), duration=args.time,
                samples=args.samples, stride=args.stride, output=args.output,
            )
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
            adaptive_budget = {
                key: value for key, value in {
                    "max_rounds": args.adaptive_max_rounds,
                    "max_windows": args.adaptive_max_windows,
                    "max_pilot_steps": args.adaptive_max_pilot_steps,
                }.items() if value is not None
            }
            result = resume_run(args.directory, parallel=override, execution=execution or None,
                                adaptive_budget=adaptive_budget or None)
            if result is not None:
                _print(result)
        elif args.command == "extend":
            from .runner import extend_run

            result = extend_run(args.directory, max_time=args.max_time)
            if result is not None:
                _print(result)
        elif args.command == "analyze":
            from .runner import analyze_run

            result = analyze_run(
                    args.directory,
                    recompute=args.recompute,
                    bins=args.bins,
                    bootstrap=args.bootstrap,
                    diffusion={key: value for key, value in {
                        "fit_range": args.fit_range, "axes": args.axes,
                        "atom_indices": args.atoms, "max_lag": args.max_lag,
                    }.items() if value is not None} or None,
                )
            _print(_report_after_result(args, result, args.directory))
        elif args.command == "rate-postprocess":
            from .analysis.rate_io import postprocess_rate
            from .parallel import root_call

            result = root_call(
                postprocess_rate, args.directory, spec=args.spec, output=args.output,
                bootstrap=args.bootstrap, seed=args.seed,
            )
            if result is not None:
                _print(_rate_postprocess_summary(result))
        elif args.command == "project":
            from .analysis.projection import project_run
            from .parallel import root_call
            _print(root_call(project_run, args.directory, axes=args.axes, output=args.output, selection_file=args.selection))
        elif args.command == "report":
            from .analysis.reporting import generate_report
            from .parallel import root_call
            report = root_call(generate_report, args.directories, output=args.output,
                formats=args.formats, dpi=args.dpi, style=args.style, energy_unit=args.energy_unit, length_unit=args.length_unit,
                labels=args.labels, series=args.series, reference=args.reference,
                include_estimates=args.include_estimates, arrhenius_fit=args.fit_arrhenius,
                kie=args.kie, convergence=args.convergence, overwrite=args.overwrite)
            if report is not None:
                _print(_report_summary(report))
        elif args.command == "info":
            from .dynamics import native_capabilities

            _print(
                {
                    "name": "PyRPMD",
                    "version": __version__,
                    "native": native_capabilities(),
                    "workflows": ["rate", "drpmd", "fes", "network", "metadynamics", "opes", "pcet", "pimd", "diffusion", "site_rate", "surface_rate"],
                    "materials": {
                        "cell": "fixed_or_isotropic_bzp",
                        "trajectory": "unwrapped_centroid",
                        "pressure_ensemble": True,
                        "fixed_atoms": True,
                        "streaming_trajectories": True,
                    },
                    "integrators": ["cayley-bcb", "normal-mode-bab", "velocity-verlet"],
                    "enhanced_sampling": ["adaptive_umbrella", "replica_exchange_umbrella", "standard_metadynamics", "well_tempered_metadynamics", "opes"],
                    "sampling_python_apis": ["plumed_centroid_bias", "transition_interface_sampling"],
                    "scientific_tools": ["general_centroid_site_rates", "temperature_isotope_studies", "ring_polymer_instanton", "takahashi_imada_thermodynamics", "open_path_momentum"],
                    "initial_ensembles": ["thermal_parent", "ne_rpmd", "qct"],
                    "input_schema_version": 1,
                    "run_manifest_schema_version": 2,
                    "checkpoint_schema_version": 2,
                    "input_formats": ["TOML .inp", "Python build()"],
                    "postprocessing": ["rate-postprocess"],
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
