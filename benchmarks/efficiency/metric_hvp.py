r"""Compare dense-Hessian and HVP checkouts with the same C ABI harmonic PES.

Build both revisions first, then run the candidate's harness, for example::

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
      benchmarks/efficiency/metric_hvp.py --baseline-source ../PyRPMD-before \
      --output metric-hvp.json

Each timing sample runs in a fresh subprocess with an explicit source checkout.
The two revisions alternate order; raw samples, peak process RSS, source hashes,
native-extension hashes, hardware and thread settings are retained in the JSON.
RSS includes the interpreter, imported libraries and work buffers, not just CVs.
Timing covers public propagation calls, including their descriptor construction.
Initial-state construction, PES compilation, warmup and state copies are excluded.

The production constrained-parent Hamiltonian contains no FluxMetricBias, and
the block API rejects simultaneous constraint and bias arguments. Accordingly:

* parent_native measures the unchanged production Cayley-RATTLE parent as a
  regression control. It does not exercise Hessians or the flux correction.
* parent_metric_python explicitly augments that Hamiltonian with FluxMetricBias
  in a benchmark-only PES wrapper and uses the unchanged constrained_step.
  This synthetic case is not a production parent or a physical-rate validation.
* flux_native and flux_fused measure the two native flux-biased propagation
  implementations (propagation.h and fused.hpp), without RATTLE constraints.

Failures, timeouts and resource-limit failures remain visible in the report.
No missing measurement is replaced with an extrapolation or a Python fallback.
This is an implementation benchmark, not a claim of physical convergence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
MODES = ("parent_native", "parent_metric_python", "flux_native", "flux_fused")
DEFAULT_STEPS = {3: 2048, 30: 1024, 300: 64, 3000: 2}
SOURCE_FILES = ("native/fused.hpp", "native/propagation.h", "src/pyrpmd/cv.py",
                "src/pyrpmd/cv_periodic.py", "src/pyrpmd/bias.py",
                "src/pyrpmd/propagation.py")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata(natoms):
    return {
        "schema": "pyrpmd.pes/1", "abi_version": 1,
        "provider_name": "metric-hvp-harmonic", "provider_version": "1",
        "artifact_sha256": {}, "natoms": natoms,
        "atom_ids": [str(i) for i in range(natoms)], "atomic_numbers": [1] * natoms,
        "state": {"label": "test", "charge": 0, "multiplicity": 1},
        "available_states": [{"label": "test", "charge": 0, "multiplicity": 1}],
        "layout": "B,N,3:C", "dtype": "float64", "position_unit": "bohr",
        "energy_unit": "hartree", "force_unit": "hartree/bohr",
        "boundary_conditions": "nonperiodic",
        "capabilities": {"native_batch": True, "force_kind": "analytic",
                         "thread_safety": "reentrant", "stateful": False,
                         "requires_bead_affinity": False, "execution_device": "cpu"},
    }


HARMONIC_SOURCE = r'''
#include "pyrpmd/pes_api_v1.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
struct pyrpmd_pes_context_v1 { int initialized; };
static const char metadata[] = __METADATA__;
int32_t pyrpmd_pes_abi_version_v1(void) { return 1; }
pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1 *options,
    pyrpmd_pes_context_v1 **out, pyrpmd_error_v1 *error) {
    (void)error;
    if (!out) return PYRPMD_INVALID_ARGUMENT_V1;
    *out = NULL;
    if (!options || options->abi_version != 1 ||
        options->struct_size != (int32_t)sizeof(*options)) return PYRPMD_ABI_MISMATCH_V1;
    *out = malloc(sizeof(**out));
    if (!*out) return PYRPMD_INTERNAL_ERROR_V1;
    (*out)->initialized = 1;
    return PYRPMD_OK_V1;
}
pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1 *context,
    char *out, int64_t capacity, int64_t *required, pyrpmd_error_v1 *error) {
    (void)error;
    if (!context || !required || capacity < 0) return PYRPMD_INVALID_ARGUMENT_V1;
    *required = sizeof(metadata);
    if (capacity < *required) return PYRPMD_BUFFER_TOO_SMALL_V1;
    if (!out) return PYRPMD_INVALID_ARGUMENT_V1;
    memcpy(out, metadata, sizeof(metadata));
    return PYRPMD_OK_V1;
}
pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1 *context,
    const pyrpmd_pes_input_v1 *x, pyrpmd_pes_output_v1 *y, pyrpmd_error_v1 *error) {
    (void)error;
    if (!context || !x || !y) return PYRPMD_INVALID_ARGUMENT_V1;
    if (x->abi_version != 1 || y->abi_version != 1 ||
        x->struct_size != (int32_t)sizeof(*x) ||
        y->struct_size != (int32_t)sizeof(*y)) return PYRPMD_ABI_MISMATCH_V1;
    const int64_t dimensions = 3 * __NATOMS__;
    const int64_t beads = x->nstructures;
    if (beads < 1 || beads > INT64_MAX / dimensions || x->natoms != __NATOMS__ ||
        x->layout != 1 || x->scalar_type != 1 || x->positions_count != beads * dimensions ||
        y->energies_count != beads || y->forces_count != beads * dimensions ||
        y->structure_status_count != beads || !x->positions || !y->energies ||
        !y->forces || !y->structure_status) return PYRPMD_INVALID_ARGUMENT_V1;
    for (int64_t b = 0; b < beads; ++b) {
        double energy = 0.;
        y->structure_status[b] = 0;
        for (int64_t j = 0; j < dimensions; ++j) {
            const int64_t i = b * dimensions + j;
            const double q = x->positions[i];
            energy += .005 * q * q;
            y->forces[i] = -.01 * q;
            if (!isfinite(q)) y->structure_status[b] = PYRPMD_STRUCTURE_NONFINITE_V1;
        }
        y->energies[b] = energy;
    }
    return PYRPMD_OK_V1;
}
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1 *context) { free(context); }
'''


def _build_harmonic(root, temporary, natoms, compiler):
    source = temporary / f"harmonic-{natoms}.c"
    library = temporary / f"harmonic-{natoms}.so"
    metadata = json.dumps(_metadata(natoms), separators=(",", ":"))
    source.write_text(HARMONIC_SOURCE.replace("__METADATA__", json.dumps(metadata))
                      .replace("__NATOMS__", str(natoms)), encoding="utf-8")
    command = compiler + ["-O3", "-shared", "-fPIC", "-I", str(root / "native/include"),
                          str(source), "-o", str(library)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return library


def _peak_rss_mib():
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (1024**2 if sys.platform == "darwin" else 1024)


def _source_identity(root):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True)
    return {"root": str(root), "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "sha256": {name: _sha256(root / name) for name in SOURCE_FILES}}


def _worker(args):
    # Imports occur only after the coordinator selected this worker's PYTHONPATH.
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    if args.memory_limit_mib:
        import resource
        limit = args.memory_limit_mib * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    import numpy as np
    import pyrpmd
    from pyrpmd import _native
    from pyrpmd.bias import FluxMetricBias
    from pyrpmd.cv import RPMDrateCoordinate
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.potentials import CABIPotential
    from pyrpmd.potentials.base import PESResult
    from pyrpmd.workflows.rate_constraints import constrained_step, project_momenta, project_positions

    root = args.worker_source.resolve()
    if not Path(pyrpmd.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError(f"worker imported a different checkout: {pyrpmd.__file__}")
    if not Path(_native.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError(f"worker imported a different native extension: {_native.__file__}")
    n, beads = args.worker_n, args.beads
    masses = np.full(n, 1800.)
    reference = np.zeros((n, 3))
    reference[:3] = [[-1.75, .1, 0.], [0., 0., 0.], [1.75, .1, 0.]]
    if n > 3:
        indices = np.arange(n - 3)
        reference[3:] = np.column_stack((.25 * np.cos(indices),
                                         .2 * np.sin(indices), .1 * np.cos(.7 * indices)))
    cv = RPMDrateCoordinate([[0], list(range(1, n))], masses, reference,
                           [[0, 1]], [[1, 2]], 10.)
    surface = cv.dividing_surface(1.001)
    beta, dt = 220., .1
    bias = FluxMetricBias(cv, masses, beta)
    rng = np.random.default_rng(702195)
    q = np.repeat(reference[None], beads, axis=0) + rng.normal(scale=.002, size=(beads, n, 3))
    p = rng.normal(scale=.5, size=q.shape)
    if args.worker_mode.startswith("parent_"):
        q = project_positions(q, surface, masses)
        project_momenta(p, q, surface, masses)

    class MetricPotential:
        """Benchmark-only ring-coupled Hamiltonian; never exposes a native binding."""

        def __init__(self, provider):
            self.provider = provider

        def evaluate(self, positions):
            result = self.provider.evaluate(positions)
            energy, force = bias.energy_force(positions)
            return PESResult(result.energy + energy / len(positions), result.forces + force,
                             result.status, result.messages)

    with CABIPotential(args.worker_library) as pes:
        provider = MetricPotential(pes) if args.worker_mode == "parent_metric_python" else pes
        dyn = Dynamics(provider, masses, beta, beads, dt, threads=args.threads, seed=581,
                       loop_backend="python" if args.worker_mode == "parent_metric_python" else "native")
        initial = State(q, p)

        def advance(state, steps):
            if args.worker_mode == "parent_metric_python":
                for _ in range(steps):
                    constrained_step(dyn, state, surface)
            elif args.worker_mode == "flux_fused":
                result = dyn.propagate(state, steps, bias=bias, backend="native",
                                       record_centroids=False, sample_interval=steps)
                if result.backend != "native":
                    raise RuntimeError("fused benchmark did not use the native backend")
            else:
                result = dyn.propagate_block(
                    state, steps, backend="native", record_energy=False,
                    observe=np.zeros(steps, dtype=np.uint8),
                    **({"constraint": surface} if args.worker_mode == "parent_native" else {"bias": bias}),
                )
                if result.backend != "native" or result.completed != steps:
                    raise RuntimeError("block benchmark did not complete in the native backend")

        before_rss = _peak_rss_mib()
        if args.warmup_steps:
            advance(initial.copy(), args.warmup_steps)
        elapsed, passes = 0., 0
        while passes == 0 or elapsed < args.min_seconds:
            state = initial.copy()
            start = time.perf_counter()
            advance(state, args.worker_steps)
            elapsed += time.perf_counter() - start
            passes += 1
        peak_rss = _peak_rss_mib()
        if not np.isfinite(state.q).all() or not np.isfinite(state.p).all():
            raise RuntimeError("benchmark ended with nonfinite state")
        residual = float(surface.value(state.q)[0]) if args.worker_mode.startswith("parent_") else None
        if residual is not None and abs(residual) > 1e-10:
            raise AssertionError(f"parent left the dividing surface: {residual}")
        output = {
            "status": "ok", "seconds": elapsed, "timed_steps": passes * args.worker_steps,
            "seconds_per_step": elapsed / (passes * args.worker_steps), "passes": passes,
            "steps_per_pass": args.worker_steps, "warmup_steps": args.warmup_steps,
            "peak_rss_mib": peak_rss, "peak_rss_before_propagation_mib": before_rss,
            "peak_rss_increase_mib": max(0., peak_rss - before_rss),
            "constraint_residual": residual, "endpoint_first_bead_reactive_q": state.q[0, :3].tolist(),
            "endpoint_first_bead_reactive_p": state.p[0, :3].tolist(),
            "source": _source_identity(root), "python": sys.version,
            "numpy": np.__version__, "native_extension": str(Path(_native.__file__).resolve()),
            "native_extension_sha256": _sha256(_native.__file__),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        }
    print(json.dumps(output, allow_nan=False))


def _sample(args, root, python, library, n, mode, steps):
    command = [str(python), str(Path(__file__).resolve()), "--worker-source", str(root),
               "--worker-library", str(library), "--worker-n", str(n), "--worker-mode", mode,
               "--worker-steps", str(steps), "--beads", str(args.beads),
               "--threads", str(args.threads), "--min-seconds", str(args.min_seconds),
               "--warmup-steps", str(args.warmup_steps),
               "--memory-limit-mib", str(args.memory_limit_mib)]
    if args.cpu is not None:
        command += ["--cpu", str(args.cpu)]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads)
    try:
        result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True,
                                timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": args.timeout}
    if result.returncode:
        return {"status": "failed", "returncode": result.returncode,
                "stderr": result.stderr[-6000:], "stdout": result.stdout[-2000:]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "invalid-worker-output", "stdout": result.stdout[-6000:],
                "stderr": result.stderr[-2000:]}


def _aggregate(samples):
    valid = [sample for sample in samples if sample["status"] == "ok"]
    out = {"samples": samples, "successful_samples": len(valid), "requested_samples": len(samples)}
    if valid:
        times = [sample["seconds_per_step"] for sample in valid]
        out.update(median_seconds_per_step=statistics.median(times),
                   min_seconds_per_step=min(times), max_seconds_per_step=max(times),
                   peak_rss_mib=max(sample["peak_rss_mib"] for sample in valid))
    return out


def _hardware():
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "cpu_model": model,
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-source", type=Path, help="built pre-change checkout")
    parser.add_argument("--candidate-source", type=Path, default=ROOT, help="built HVP checkout")
    parser.add_argument("--baseline-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--candidate-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 30, 300, 3000])
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--steps", type=int, nargs="+",
                        help="one common count or one per size; default 2048,1024,64,2")
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=.2,
                        help="repeat identical initialized passes until this timed duration")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, help="pin each worker to this CPU (Linux)")
    parser.add_argument("--timeout", type=float, default=300., help="timeout per isolated sample")
    parser.add_argument("--memory-limit-mib", type=int, default=0,
                        help="optional worker virtual-address limit; 0 means no imposed limit")
    parser.add_argument("--output", type=Path, help="complete JSON report; table goes to stdout")
    parser.add_argument("--work-dir", type=Path, default=Path.cwd(), help="temporary compiler workspace")
    for name, kind in (("source", Path), ("library", Path), ("n", int), ("mode", str), ("steps", int)):
        parser.add_argument("--worker-" + name, type=kind, help=argparse.SUPPRESS)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.worker_source is not None:
        _worker(args)
        return
    if args.baseline_source is None:
        parser.error("--baseline-source is required: both revisions are measured, never extrapolated")
    if (min(args.sizes) < 3 or min(args.beads, args.repeats, args.threads) < 1
            or args.warmup_steps < 0 or args.warmup_steps > 4096 or args.min_seconds < 0
            or not math.isfinite(args.min_seconds) or not math.isfinite(args.timeout)
            or args.timeout <= 0 or args.memory_limit_mib < 0):
        parser.error("invalid size, step count, repetition count, resource limit or timing option")
    if args.steps is None:
        counts = [DEFAULT_STEPS.get(n, max(2, min(2048, 20000 // n))) for n in args.sizes]
    elif len(args.steps) == 1:
        counts = args.steps * len(args.sizes)
    elif len(args.steps) == len(args.sizes):
        counts = args.steps
    else:
        parser.error("--steps needs one count or one count per --sizes entry")
    if min(counts) < 1 or max(counts) > 4096:
        parser.error("steps per pass must be between 1 and 4096")
    roots = {"baseline": args.baseline_source.resolve(), "candidate": args.candidate_source.resolve()}
    # Resolving an interpreter symlink can turn a venv executable into its base
    # Python and silently discard that environment's installed dependencies.
    pythons = {"baseline": args.baseline_python.expanduser().absolute(),
               "candidate": args.candidate_python.expanduser().absolute()}
    if any(not (root / "native/include/pyrpmd/pes_api_v1.h").is_file() for root in roots.values()):
        parser.error("both sources must be PyRPMD source checkouts")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or shutil.which(compiler[0]) is None:
        parser.error("a C compiler is required for the harmonic PES")
    compiler_version = subprocess.run(compiler + ["--version"], check=True, capture_output=True, text=True)
    report = {
        "format": "PyRPMD.metric-hvp-benchmark.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "hardware": _hardware(),
        "command": [sys.executable, *sys.argv],
        "compiler": compiler_version.stdout.splitlines()[0], "threads": args.threads,
        "beads": args.beads, "repeats": args.repeats, "cpu": args.cpu,
        "timeout_seconds": args.timeout, "memory_limit_mib": args.memory_limit_mib,
        "semantics": {"parent_native": "production RATTLE parent; no flux bias; regression control",
                      "parent_metric_python": "synthetic RATTLE parent augmented by FluxMetricBias; Python orchestration, C ABI PES",
                      "flux_native": "unconstrained rate-flux propagation via native/propagation.h",
                      "flux_fused": "unconstrained rate-flux propagation via native/fused.hpp"},
        "measurement": "Public-call wall time, no startup/compilation/warmup/state-copy time. Peak process RSS includes imports and all work buffers.",
         "cases": [],
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print("mode                     N    old us/step    new us/step    speedup    old/new peak MiB", flush=True)
    with tempfile.TemporaryDirectory(prefix="metric-hvp-", dir=args.work_dir) as temp:
        for n, steps in zip(args.sizes, counts, strict=True):
            library = _build_harmonic(ROOT, Path(temp), n, compiler)
            for mode in args.modes:
                samples = {"baseline": [], "candidate": []}
                for repeat in range(args.repeats):
                    order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                    for variant in order:
                        sample = _sample(args, roots[variant], pythons[variant], library, n, mode, steps)
                        samples[variant].append(sample)
                row = {"natoms": n, "mode": mode, "steps_per_pass": steps,
                       **{variant: _aggregate(values) for variant, values in samples.items()}}
                old, new = row["baseline"], row["candidate"]
                complete = old["successful_samples"] == new["successful_samples"] == args.repeats
                if complete:
                    row["speedup"] = old["median_seconds_per_step"] / new["median_seconds_per_step"]
                    if n == 3:
                        row["small_system_slowdown_fraction"] = 1 / row["speedup"] - 1
                        row["small_system_5_percent_gate_passed"] = row["small_system_slowdown_fraction"] <= .05
                    print(f"{mode:23s} {n:5d} {old['median_seconds_per_step']*1e6:14.3f} "
                          f"{new['median_seconds_per_step']*1e6:14.3f} {row['speedup']:10.3f} "
                          f"{old['peak_rss_mib']:8.1f}/{new['peak_rss_mib']:.1f}", flush=True)
                else:
                    print(f"{mode:23s} {n:5d} INCOMPLETE: baseline {old['successful_samples']}/{args.repeats}, "
                          f"candidate {new['successful_samples']}/{args.repeats}; see errors in JSON", flush=True)
                report["cases"].append(row)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.output:
        print(json.dumps(report, indent=2, allow_nan=False))
    failed = any(case[variant]["successful_samples"] != args.repeats
                 for case in report["cases"] for variant in ("baseline", "candidate"))
    if failed:
        raise SystemExit(1)
    slow = [case for case in report["cases"]
            if case.get("small_system_5_percent_gate_passed") is False]
    if slow:
        details = ", ".join(f"{case['mode']} ({case['small_system_slowdown_fraction']:+.1%})"
                            for case in slow)
        print("N=3 slowdown exceeds the 5% acceptance gate: " + details, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
