"""Explicit CLI for optional molecular PES generation."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-native", help="build a locally supplied pinned Caracal checkout")
    build.add_argument("source"); build.add_argument("destination")
    build.add_argument("--fortran", default="mpifort"); build.add_argument("--cc", default="mpicc")
    build.add_argument("--cxx", default="c++")
    fit = sub.add_parser("generate", help="PySCF references, fitting, native bundle, diagnostics")
    fit.add_argument("input"); fit.add_argument("--output", required=True)
    fit.add_argument("--resume", action="store_true")
    run = sub.add_parser("run-job", help="run an existing PyRPMD job with the fitted PES")
    run.add_argument("directory"); run.add_argument("job"); run.add_argument("--output", required=True)
    run.add_argument("--allow-unvalidated", action="store_true"); run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build-native":
        from .build import build_native
        print(build_native(args.source, args.destination, fortran=args.fortran, c_compiler=args.cc, cxx=args.cxx))
    elif args.command == "generate":
        from .workflow import generate
        print(json.dumps(generate(args.input, args.output, resume=args.resume), indent=2))
    else:
        from .workflow import run_generated_job
        print(json.dumps(run_generated_job(args.directory, args.job, output=args.output,
                        allow_unvalidated=args.allow_unvalidated, resume=args.resume), indent=2, default=str))


if __name__ == "__main__":
    main()
