#!/usr/bin/env python3
"""Reproducible, deliberately unconverged gas-phase f-QCMD/PA-CMD comparison.

Run from an installed checkout, with a writable output directory:

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
    examples/qcmd/validate_monomer.py --output runs/qcmd-monomer-150 --temperature 150

The independent copies are *noninteracting* q-TIP4P/F monomers. Their spectra
are averaged without cross-monomer dipole products. The published 2019 QCMD
gas-phase calculation used the different Partridge--Schwenke PES; this is a
direction/scale diagnostic, not a reproduction of that paper's peak values.
Neither successful IBI iterations nor a positive blue shift certify physical
convergence. Increase beads, independent seeds, sampling length, IBI accuracy,
adiabatic separation and spectral duration and decrease both timesteps.

The model-training and diagnostic trajectory checkpoints are atomic and
authenticate numerical settings, source files, RNG and the saved trace prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.analysis.spectroscopy import relative_ir_spectrum
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cmd import PACMDDynamics
from pyrpmd.constants import FS_TO_AU, KB_AU, WAVENUMBER_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.system import System

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from water_monomer import MONOMER_MASSES, PARAMETERS, QTIP4PFMonomers, geometry, molecular_dipoles


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_identity():
    import pyrpmd.cmd
    import pyrpmd.dynamics
    paths = [Path(__file__), HERE / "water_monomer.py", Path(pyrpmd.cmd.__file__),
             Path(pyrpmd.dynamics.__file__)]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _peak(wavenumber, intensity, lower=2800., upper=4400.):
    """Largest Hann-windowed peak in a declared OH-stretch band, without tuning."""
    keep = np.flatnonzero((wavenumber >= lower) & (wavenumber <= upper))
    if not len(keep):
        raise ValueError("spectrum does not resolve the declared OH band")
    index = int(keep[np.argmax(intensity[keep])])
    # Report an actual FFT-bin peak; do not imply zero-padding adds resolution.
    return float(wavenumber[index])


def _trajectory(potential, *, method, beta, masses, initial, beads, seed,
                dt_fs, equilibration_fs, production_fs, omega_cm1, friction,
                sample_fs, checkpoint, model_identity, source_identity):
    dt = dt_fs * FS_TO_AU
    eq_steps = int(round(equilibration_fs / dt_fs))
    prod_steps = int(round(production_fs / dt_fs))
    stride = int(round(sample_fs / dt_fs))
    if stride < 1 or not np.isclose(stride * dt_fs, sample_fs):
        raise ValueError("sample_fs must be a positive integer multiple of dt_fs")
    settings = {"method": method, "beta": beta, "masses": masses.tolist(), "beads": beads,
                "seed": seed, "dt_fs": dt_fs, "eq_steps": eq_steps, "prod_steps": prod_steps,
                "sample_stride": stride, "omega_cm1": omega_cm1, "friction": friction,
                "model_identity": model_identity, "source_identity": source_identity,
                "initial": initial.tolist(), "parameters": PARAMETERS}
    identity = _identity(settings)
    rng = np.random.default_rng(seed)
    if method == "pa-cmd":
        engine = PACMDDynamics(potential, masses, beta, beads, dt,
                               target_frequency=omega_cm1 * WAVENUMBER_TO_AU,
                               friction_scale=friction, seed=rng)
    else:
        engine = Dynamics(potential, masses, beta, 1, dt, integrator="normal-mode-bab",
                          native=False, loop_backend="python", seed=rng)
    trace = []
    elapsed_previous = 0.
    if checkpoint.exists():
        with load_checkpoint(checkpoint, expected_kind="PyRPMD.qcmd-monomer-trace.v1",
                             require_integrity=True) as data:
            if str(data["identity"]) != identity:
                raise ValueError("monomer diagnostic checkpoint settings/source/model changed")
            state = State(data["q"], data["p"], int(data["step"]), float(data["time"]))
            rng.bit_generator.state = json.loads(str(data["rng"]))
            trace = list(np.asarray(data["dipole"]))
            elapsed_previous = float(data["elapsed_seconds"])
    else:
        state = State(np.repeat(initial[None], beads, axis=0), engine.sample_momenta())
    total = eq_steps + prod_steps
    if not 0 <= state.step <= total or state.q.shape != (beads, len(masses), 3):
        raise ValueError("invalid monomer diagnostic checkpoint phase/shape")
    expected_frames = 0 if state.step < eq_steps else (state.step - eq_steps) // stride + 1
    if len(trace) != expected_frames:
        raise ValueError("monomer diagnostic checkpoint has a missing/duplicate frame")
    started = time.perf_counter()

    def record():
        trace.append(molecular_dipoles(state.q).mean(axis=0))

    def save():
        atomic_savez(checkpoint, kind="PyRPMD.qcmd-monomer-trace.v1", identity=identity,
                     q=state.q, p=state.p, step=state.step, time=state.time,
                     rng=json.dumps(rng.bit_generator.state),
                     dipole=np.asarray(trace).reshape(-1, len(masses) // 3, 3),
                     elapsed_seconds=elapsed_previous + time.perf_counter() - started)

    if state.step == eq_steps and not trace:
        record()
    while state.step < total:
        if state.step < eq_steps:
            engine.sample_step(state, gamma_centroid=.01)
        else:
            engine.step(state)
        if state.step >= eq_steps and (state.step - eq_steps) % stride == 0:
            record()
        if state.step % 2000 == 0 or state.step == total:
            save()
    elapsed = elapsed_previous + time.perf_counter() - started
    dipole = np.asarray(trace).transpose(1, 0, 2)[:, :, None, :]
    spectrum = relative_ir_spectrum(dipole, sample_fs * FS_TO_AU, beta=beta,
                                    normalize=False, window="hann")
    peak = _peak(spectrum["wavenumber_cm1"], spectrum["relative_ir"])
    per_replica = spectrum["per_replica"] * beta * spectrum["angular_frequency_au"][None, :]**2
    peaks = [_peak(spectrum["wavenumber_cm1"], row) for row in per_replica]
    return spectrum, {"settings": settings, "peak_cm1": peak,
                      "independent_monomer_peaks_cm1": peaks,
                      "fft_bin_spacing_cm1": float(np.diff(spectrum["wavenumber_cm1"])[0]),
                      "elapsed_seconds": elapsed,
                      "mean_microseconds_per_step": 1e6 * elapsed / total,
                      "copies": len(masses) // 3, "steps": total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=150.)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--target-equilibration-steps", type=int, default=10000)
    parser.add_argument("--target-sampling-steps", type=int, default=40000)
    parser.add_argument("--ibi-equilibration-steps", type=int, default=4000)
    parser.add_argument("--ibi-sampling-steps", type=int, default=20000)
    parser.add_argument("--ibi-iterations", type=int, default=5)
    parser.add_argument("--sampling-dt-fs", type=float, default=.1)
    parser.add_argument("--cmd-dt-fs", type=float, default=.025)
    parser.add_argument("--f-qcmd-dt-fs", type=float, default=.1)
    parser.add_argument("--production-fs", type=float, default=1000.)
    parser.add_argument("--equilibration-fs", type=float, default=500.)
    parser.add_argument("--target-frequency-cm1", type=float, default=14000.)
    parser.add_argument("--friction-scale", type=float, default=1.)
    parser.add_argument("--stage", choices=("all", "cmd", "fit", "f-qcmd"), default="all")
    args = parser.parse_args(argv)
    if args.temperature <= 0 or args.beads < 2 or args.copies < 1:
        parser.error("positive temperature/copies and at least two target beads are required")
    args.output.mkdir(parents=True, exist_ok=True)
    beta = 1 / (KB_AU * args.temperature)
    initial = geometry(args.copies)
    masses = np.tile(MONOMER_MASSES, args.copies)
    molecules = [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(args.copies)]
    system = System(("O", "H", "H") * args.copies, masses, initial)
    spec = {"provider": "python", "factory": str(HERE / "water_monomer.py") + ":build"}
    base_identity = _identity({"parameters": PARAMETERS, "masses": masses.tolist(),
                               "scope": "noninteracting_monomers"})
    if args.stage != "cmd":
        from pyrpmd.qcmd import QCMDModel, QuasiCentroidPotential, fit_fqcmd, qcmd_base_identity
        fit_base_identity = qcmd_base_identity(spec, system)
    source_identity = _source_identity()
    model_path = args.output / "qcmd-model.npz"
    report = {"physical_converged": False, "converged": False,
              "temperature_K": args.temperature, "source_identity": source_identity,
              "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                           "logical_cpus_available": os.cpu_count(),
                           "cpu_model": "not exposed by this execution environment",
                           "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                           "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS")},
              "reference_comparison": "QCMD gas-phase blue shift in Trenins et al. JCP 151, 054109 (2019), Fig. 4; different PES",
              "limitations": ["No liquid-water calculation.", "P=32/64, timestep, thermostat/adiabaticity, IBI grid, training length and seeds need convergence sweeps.",
                              "Raw maximum of an OH-band Hann periodogram; not fitted line positions or physical error bars.",
                              "q-TIP4P/F ordinary geometry dipole on f-QCMD coordinates is an observable approximation.",
                              "Finite-step PA-CMD is an approximate reference, not exact CMD or quantum dynamics."]}
    with threadpool_limits(limits=1):
        report["hardware"]["threadpools"] = threadpool_info()
        if args.stage in ("all", "fit"):
            started = time.perf_counter()
            model = fit_fqcmd(
                QTIP4PFMonomers(), masses, beta, np.repeat(initial[None], args.beads, axis=0),
                molecules, grids=(np.linspace(1.3, 2.5, 121), np.linspace(1.05, 2.75, 101)),
                dt=args.sampling_dt_fs * FS_TO_AU,
                equilibration_steps=args.target_equilibration_steps,
                sampling_steps=args.target_sampling_steps, sample_interval=5,
                iterations=args.ibi_iterations,
                ibi_equilibration_steps=args.ibi_equilibration_steps,
                ibi_sampling_steps=args.ibi_sampling_steps,
                gamma_centroid=.01, damping=.5, regularization=.01, smoothing=1.,
                seed=args.seed, base_identity=fit_base_identity,
                checkpoint=args.output / "fit-checkpoint.npz")
            model.save(model_path)
            report["fit"] = {"elapsed_seconds": time.perf_counter() - started,
                             "diagnostics": model.diagnostics, "training": model.training}
            print(json.dumps({"phase": "fit-complete", "validation": [
                {"iteration": entry["iteration"],
                 "raw_total_variation": {name: values["raw_total_variation"]
                                         for name, values in entry["coordinates"].items()}}
                for entry in model.diagnostics]}), flush=True)
        if args.stage in ("all", "cmd"):
            spec_cmd, result = _trajectory(
                QTIP4PFMonomers(), method="pa-cmd", beta=beta, masses=masses,
                initial=initial, beads=args.beads, seed=args.seed + 31,
                dt_fs=args.cmd_dt_fs, equilibration_fs=args.equilibration_fs,
                production_fs=args.production_fs, omega_cm1=args.target_frequency_cm1,
                friction=args.friction_scale, sample_fs=.1,
                checkpoint=args.output / "pa-cmd-trace.npz", model_identity=base_identity,
                source_identity=source_identity)
            report["pa-cmd"] = result
            atomic_savez(args.output / "pa-cmd-spectrum.npz", kind="PyRPMD.qcmd-monomer-spectrum.v1",
                         **{key: value for key, value in spec_cmd.items() if isinstance(value, np.ndarray)})
            print(json.dumps({"phase": "pa-cmd-complete", "peak_cm1": result["peak_cm1"],
                              "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        if args.stage in ("all", "f-qcmd"):
            model = QCMDModel.load(model_path)
            model.validate_system(system, beta, base_identity=fit_base_identity)
            potential = QuasiCentroidPotential(QTIP4PFMonomers(), model)
            spec_fq, result = _trajectory(
                potential, method="f-qcmd", beta=beta, masses=masses, initial=initial,
                beads=1, seed=args.seed + 59, dt_fs=args.f_qcmd_dt_fs,
                equilibration_fs=args.equilibration_fs, production_fs=args.production_fs,
                omega_cm1=args.target_frequency_cm1, friction=args.friction_scale,
                sample_fs=.1, checkpoint=args.output / "f-qcmd-trace.npz",
                model_identity=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                source_identity=source_identity)
            report["f-qcmd"] = result
            atomic_savez(args.output / "f-qcmd-spectrum.npz", kind="PyRPMD.qcmd-monomer-spectrum.v1",
                         **{key: value for key, value in spec_fq.items() if isinstance(value, np.ndarray)})
            print(json.dumps({"phase": "f-qcmd-complete", "peak_cm1": result["peak_cm1"],
                              "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        if "pa-cmd" in report and "f-qcmd" in report:
            report["f_qcmd_minus_pa_cmd_cm1"] = report["f-qcmd"]["peak_cm1"] - report["pa-cmd"]["peak_cm1"]
        target = args.output / f"report-{args.stage}.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"report": str(target), "physical_converged": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
