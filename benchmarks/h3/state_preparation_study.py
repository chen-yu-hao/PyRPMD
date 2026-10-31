"""Controlled initial-preparation comparisons on the real BKMP2 surface.

This research benchmark keeps the original native collision driver and seed
identity. It compares the published centroid cutoff, its removal, an isolated
BC smooth potential switch, and full-PES action-quantized classical orbits.
Smooth switching of a many-bead chain is experimental: it is not a pure-state
preparation or an established correction to state-selected RPMD.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from benchmarks.h3 import state_selected_reproduction as driver
from benchmarks.h3.state_selected_sensitivity import _source_snapshot, _verify_snapshot
from pyrpmd.constants import EV_TO_AU

ORIGINAL_PREPARE = driver.prepare_paper_state
_SPEC = _SWITCHER = _ORBIT = None
_KEEP_TRACE = False


@dataclass(frozen=True)
class LaunchConfig(driver.Config):
    """An explicit collision-energy override leaves RNG energy labels intact."""

    collision_energy_override_au: float | None = None

    @property
    def collision_energy(self):
        value = (super().collision_energy if self.collision_energy_override_au is None
                 else self.collision_energy_override_au)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("The actual collision energy must be positive and finite")
        return float(value)


@dataclass(frozen=True)
class PreparationSpec:
    kind: str = "direct"
    conditioning: str = "paper"
    switch_time_fs: float = 0.0
    hold_time_fs: float = 0.0
    preparation_dt_fs: float = 0.02
    trace_stride_fs: float = 0.2
    energy_convention: str = "fixed_collision"

    def validate(self, config):
        if self.kind not in ("direct", "switch", "action_qct"):
            raise ValueError("Unknown preparation kind")
        if self.conditioning not in ("paper", "none"):
            raise ValueError("Conditioning must be paper or none")
        if self.energy_convention not in ("fixed_collision", "fixed_total"):
            raise ValueError("Unknown collision-energy convention")
        values = (self.switch_time_fs, self.hold_time_fs,
                  self.preparation_dt_fs, self.trace_stride_fs)
        if not all(math.isfinite(x) for x in values) or min(values) < 0:
            raise ValueError("Preparation times must be finite and nonnegative")
        if self.preparation_dt_fs <= 0 or self.trace_stride_fs <= 0:
            raise ValueError("Preparation timestep and trace stride must be positive")
        if (self.kind == "switch") != (self.switch_time_fs > 0):
            raise ValueError("Only switch preparation has a positive switch duration")
        if self.kind == "action_qct":
            if config.method != "qct" or config.beads != 1 or self.conditioning != "none":
                raise ValueError("Action QCT requires one bead, qct method and no cutoff")
            if self.hold_time_fs:
                raise ValueError("Action QCT uses its stationary time-phase orbit without an extra hold")
        elif config.method != "ne_rpmd" or config.beads < 2:
            raise ValueError("Direct/switch comparisons require a multi-bead NE-RPMD ensemble")
        elif self.energy_convention != "fixed_collision":
            raise ValueError("NE comparisons retain the original collision energy; no retuning")


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _radial_components(state):
    bond = state.q[:, 2] - state.q[:, 1]
    axis = bond.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("Zero centroid bond in prepared molecule")
    axis /= norm
    radial = bond @ axis
    momentum = ((state.p[:, 2] - state.p[:, 1]) / 2) @ axis
    if not np.allclose(bond, radial[:, None] * axis, atol=5e-12, rtol=0):
        raise ValueError("The isolated preparation requires collinear molecular bead displacements")
    return radial, momentum, axis


def _diagnostics(r, p, config, potential):
    mu = driver.MASS / 2
    vmin = potential.radial([driver.BOND, 99.0, 99.0])[0]
    full = np.array([potential.radial([float(x), 99.0, 99.0])[0] for x in r])
    kinetic = float(np.mean(p * p) / (2 * mu))
    spring = float(mu * (config.beads / config.beta)**2
                   * np.mean((np.roll(r, -1) - r)**2) / 2)
    harmonic = float(mu * driver.OMEGA**2 * np.mean((r-driver.BOND)**2) / 2)
    full_relative = float(full.mean() - vmin)
    return {
        "kinetic_ev": kinetic / EV_TO_AU, "spring_ev": spring / EV_TO_AU,
        "harmonic_potential_ev": harmonic / EV_TO_AU,
        "full_potential_ev": full_relative / EV_TO_AU,
        "primitive_full_ev": (kinetic + full_relative - spring) / EV_TO_AU,
        "primitive_harmonic_ev": (kinetic + harmonic - spring) / EV_TO_AU,
        "ring_hamiltonian_relative_ev": (kinetic + full_relative + spring) / EV_TO_AU,
        "centroid_bond_bohr": float(np.mean(r)),
        "minimum_bead_bond_bohr": float(np.min(r)),
        "maximum_bead_bond_bohr": float(np.max(r)),
        "bond_spread_bohr": float(np.std(r)),
    }


def _replace_bc(state, radial, momentum, axis):
    before_q, before_p = state.q.copy(), state.p.copy()
    center = state.q[:, 1:].mean(axis=1)
    translation = state.p[:, 1:].mean(axis=1)
    state.q[:, 1] = center - radial[:, None] * axis / 2
    state.q[:, 2] = center + radial[:, None] * axis / 2
    state.p[:, 1] = translation - momentum[:, None] * axis
    state.p[:, 2] = translation + momentum[:, None] * axis
    if not (np.array_equal(state.q[:, 0], before_q[:, 0])
            and np.array_equal(state.p[:, 0], before_p[:, 0])
            and np.allclose(state.q[:, 1:].sum(axis=1), before_q[:, 1:].sum(axis=1), atol=3e-12, rtol=0)
            and np.allclose(state.p[:, 1:].sum(axis=1), before_p[:, 1:].sum(axis=1), atol=3e-11, rtol=0)):
        raise AssertionError("Internal preparation changed a fragment COM or incoming atom")


def prepare_variant(config, prepared, trajectory_id):
    """Reuse launch b/orientation/kick; change only the internal BC preparation."""
    condition = config.dmin_bohr if _SPEC.conditioning == "paper" else 0.0
    state, launch = ORIGINAL_PREPARE(replace(config, dmin_bohr=condition), prepared, trajectory_id)
    radial, momentum, axis = _radial_components(state)
    potential = driver._WORKER[2].potential
    initial = _diagnostics(radial, momentum, config, potential)
    metadata = {"kind": _SPEC.kind, "conditioning": _SPEC.conditioning,
                "before": initial, "energy_convention": _SPEC.energy_convention,
                "separation_policy": "Keep the pre-switch draw's axial separation and incoming atom ring; paired only within the same conditioning rule",
                "collision_energy_ev": config.collision_energy / EV_TO_AU,
                "nominal_internal_label_ev": (config.v + .5) * driver.OMEGA / EV_TO_AU}
    if _SPEC.kind == "action_qct":
        # Original QCT sampling consumes its first nonempty uniform draw for
        # the molecular phase. This mapping agrees exactly in a harmonic well:
        # old r=req+A*cos(phi), p=mu*omega*A*sin(phi); orbit u=0 starts inside.
        sequence = np.random.SeedSequence([config.seed, config.v,
                    round(config.total_energy_ev*1_000_000), 1, 1, trajectory_id])
        initial_rng = np.random.default_rng(sequence.spawn(2)[0])
        phase = float((.5-initial_rng.random()) % 1.0)
        r, p = _ORBIT.sample_phase(phase)
        radial, momentum = np.atleast_1d(r), np.atleast_1d(p)
        metadata.update(orbit_phase=phase, action_quantum_number=config.v+.5,
                        orbit_energy_ev=_ORBIT.energy_hartree/EV_TO_AU,
                        actual_asymptotic_total_energy_ev=(config.collision_energy+_ORBIT.energy_hartree)/EV_TO_AU)
    elif _SPEC.switch_time_fs or _SPEC.hold_time_fs:
        result = _SWITCHER.run(radial, momentum, beta=config.beta,
                 dt_fs=_SPEC.preparation_dt_fs, switch_time_fs=_SPEC.switch_time_fs,
                 hold_time_fs=_SPEC.hold_time_fs, trace_stride_fs=_SPEC.trace_stride_fs)
        radial, momentum = result.r, result.p
        metadata["switch"] = result.summary
        if _KEEP_TRACE:
            metadata["trace"] = result.trace
    _replace_bc(state, radial, momentum, axis)
    metadata["after"] = _diagnostics(radial, momentum, config, potential)
    metadata["delta_primitive_full_ev"] = metadata["after"]["primitive_full_ev"]-initial["primitive_full_ev"]
    metadata["final_rg_bohr"] = np.sqrt(np.mean(np.sum((state.q-state.q.mean(axis=0))**2,axis=2),axis=0)).tolist()
    launch["preparation_study"] = metadata
    return state, launch


def _initialize(config, spec, snapshot, keep_trace):
    global _SPEC, _SWITCHER, _ORBIT, _KEEP_TRACE
    _verify_snapshot(snapshot["files"])
    driver._init_worker(config)
    _SPEC, _KEEP_TRACE = spec, keep_trace
    potential = driver._WORKER[2].potential
    if spec.kind == "action_qct":
        from benchmarks.h3.diatomic_action import DiatomicAction
        _ORBIT = DiatomicAction.from_bkmp2(potential, reduced_mass=driver.MASS/2).quantize(config.v)
    elif spec.switch_time_fs or spec.hold_time_fs:
        from benchmarks.h3.diatomic_switch import DiatomicSwitcher
        _SWITCHER = DiatomicSwitcher(potential, mass=driver.MASS,
                                    omega=driver.OMEGA, equilibrium=driver.BOND)
    driver.prepare_paper_state = prepare_variant


def _audit_one(trajectory_id):
    config, prepared, _, _ = driver._WORKER
    start = time.perf_counter()
    try:
        _, launch = prepare_variant(config, prepared, trajectory_id)
        return {"id": trajectory_id, "status": "prepared", **launch,
                "wall_seconds": time.perf_counter()-start}
    except Exception as exc:
        return {"id": trajectory_id, "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.perf_counter()-start}


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "sem": float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None}


def summarize_audit(rows):
    accepted = [r for r in rows if r["status"] == "prepared"]
    quantities = {}
    for stage in ("before", "after"):
        if accepted:
            for key in accepted[0]["preparation_study"][stage]:
                quantities[stage+"_"+key] = _stats([r["preparation_study"][stage][key] for r in accepted])
    quantities["paired_delta_primitive_full_ev"] = _stats(
        [r["preparation_study"]["delta_primitive_full_ev"] for r in accepted])
    return {"attempts": len(rows), "prepared": len(accepted), "failed": len(rows)-len(accepted),
            "quantities": quantities,
            "statistics_scope": "Means conditional on successful preparation; failures retained and counted; SEM over independent trajectory IDs."}


def run(config, spec, output, *, workers=8, audit=False):
    spec.validate(config)
    if (config.samples < 1 or workers < 1 or config.id_offset < 0 or config.seed < 0
            or config.v not in (0, 1) or not all(math.isfinite(x) for x in
                (config.total_energy_ev, config.bmin, config.bmax, config.dt_fs))
            or not 0 <= config.bmin < config.bmax
            or not 0 < config.dt_fs <= config.time_cap_fs <= config.extended_cap_fs):
        raise ValueError("Invalid sample count, RNG identity, energy, area or timestep/budget")
    potential = driver.BKMP2Potential()
    orbit_metadata = None
    if spec.kind == "action_qct":
        from benchmarks.h3.diatomic_action import DiatomicAction
        orbit = DiatomicAction.from_bkmp2(potential, reduced_mass=driver.MASS/2).quantize(config.v)
        orbit_metadata = orbit.metadata
        if spec.energy_convention == "fixed_total":
            config = replace(config, collision_energy_override_au=config.total_energy_ev*EV_TO_AU-orbit.energy_hartree)
    _ = config.collision_energy
    native_preparation = None
    if spec.switch_time_fs or spec.hold_time_fs:
        from benchmarks.h3.diatomic_switch import DiatomicSwitcher
        switcher = DiatomicSwitcher(potential, mass=driver.MASS,
                                    omega=driver.OMEGA, equilibrium=driver.BOND)
        native_preparation = switcher.metadata
    snapshot = _source_snapshot()
    paths = [Path(__file__)]
    dependencies = (["diatomic_action.py"] if spec.kind == "action_qct" else
                    ["diatomic_switch.py", "diatomic_switch.cpp"]
                    if spec.switch_time_fs or spec.hold_time_fs else [])
    for name in dependencies:
        p = Path(__file__).with_name(name)
        if p.exists():
            paths.append(p)
    if native_preparation is not None:
        paths.append(Path(switcher.library_path))
    snapshot["files"].extend({"path": str(p.resolve()), "sha256": _sha(p)} for p in paths)
    snapshot["study_spec"] = asdict(spec)
    snapshot["orbit"] = orbit_metadata
    snapshot["native_preparation"] = native_preparation
    snapshot["seed_pairing"] = (
        "Original RNG identity and launch b/orientation retained. Within a conditioning rule, "
        "direct/switch variants share the atom ring and separation. Removing the cutoff uses "
        "the first raw draw: IDs rejected by the paper rule can also change the atom ring and "
        "separation because the original sampler redraws the entire state. NE Ecol is unchanged.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _json(output/"config.json", asdict(config))
    _json(output/"preparation.json", asdict(spec))
    _json(output/"provenance.json", snapshot)
    start = time.perf_counter()
    rows = []
    with (output/"trajectories.jsonl").open("x") as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize, initargs=(config, spec, snapshot, audit)) as pool:
            futures = [pool.submit(_audit_one if audit else driver.trajectory, i)
                       for i in range(config.id_offset, config.id_offset+config.samples)]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                rows.append(row)
                if len(rows) % max(32, config.samples//8) == 0:
                    stream.flush()
                    print(json.dumps({"case":output.name,"done":len(rows),"total":config.samples,
                                      "seconds":round(time.perf_counter()-start,1)}),flush=True)
    _verify_snapshot(snapshot["files"])
    summary = summarize_audit(rows) if audit else driver.summarize(config, rows)
    summary.update(config=asdict(config), preparation=asdict(spec), audit_only=audit,
                   elapsed_seconds=time.perf_counter()-start, workers=workers,
                   source_sha256=_sha(__file__), created_utc=datetime.now(timezone.utc).isoformat())
    _json(output/"summary.json", summary)
    print(json.dumps({"case":output.name,"audit":audit,"samples":len(rows),
                      "counts":summary.get("counts"),"failed":summary.get("failed"),
                      "seconds":summary["elapsed_seconds"]}),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=["direct","switch","action_qct"],default="direct")
    parser.add_argument("--conditioning",choices=["paper","none"],default="paper")
    parser.add_argument("--energy-convention",choices=["fixed_collision","fixed_total"],default="fixed_collision")
    parser.add_argument("--v",type=int,choices=[0,1],required=True)
    parser.add_argument("--etot",type=float,required=True)
    parser.add_argument("--beads",type=int,default=32)
    parser.add_argument("--samples",type=int,default=1024)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--id-offset",type=int,default=10000)
    parser.add_argument("--bmax",type=float)
    parser.add_argument("--switch-fs",type=float,default=0.)
    parser.add_argument("--hold-fs",type=float,default=0.)
    parser.add_argument("--preparation-dt-fs",type=float,default=.02)
    parser.add_argument("--trace-stride-fs",type=float,default=.2)
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=LaunchConfig(v=args.v,total_energy_ev=args.etot,samples=args.samples,id_offset=args.id_offset,
                        bmax=args.bmax if args.bmax is not None else (3.5 if args.v==0 else 4.),
                        method="qct" if args.kind=="action_qct" else "ne_rpmd",
                        beads=1 if args.kind=="action_qct" else args.beads)
    spec=PreparationSpec(kind=args.kind,conditioning=args.conditioning,
         switch_time_fs=args.switch_fs,hold_time_fs=args.hold_fs,
         preparation_dt_fs=args.preparation_dt_fs,trace_stride_fs=args.trace_stride_fs,
         energy_convention=args.energy_convention)
    run(config,spec,args.output,workers=args.workers,audit=args.audit)


if __name__=="__main__":
    main()
