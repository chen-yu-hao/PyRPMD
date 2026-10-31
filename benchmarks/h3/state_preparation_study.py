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
from pyrpmd.constants import EV_TO_AU, FS_TO_AU
from pyrpmd.dynamics import normalize_integrator

ORIGINAL_PREPARE = driver.prepare_paper_state
_SPEC = _SWITCHER = _ORBIT = None
_KEEP_TRACE = False
_QCT_PLACEHOLDER_BETA = 1500.0


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_number(value, name, *, positive=False, nonnegative=False):
    # Keep the persisted dataclasses JSON-serializable, and never treat True
    # as a scientific parameter or coerce a string into one.
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= 0) or (nonnegative and value < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {bound}")


def _grid_steps(value, dt, name, *, positive=False):
    """Match the native preparation grid without silently rounding budgets."""
    ratio = value / dt
    if not math.isfinite(ratio) or ratio > 2**52:
        raise ValueError(f"{name} contains too many steps")
    steps = round(ratio)
    tolerance = max(1e-9, 8*np.finfo(float).eps*abs(ratio))
    if ((positive and steps < 1) or abs(ratio-steps) > tolerance
            or (value > 0 and steps == 0)):
        raise ValueError(f"{name} must be an integer multiple of its timestep")
    return steps


def _validate_propagation(config):
    """Check the existing sampler/integrator's numerical domain without a PES."""
    beta = config.beta
    try:
        scale = 2*config.beads/beta
        maximum_frequency = scale*math.sin(math.pi*(config.beads//2)/config.beads)
        variance = config.beads/beta/driver.OMEGA**2
    except OverflowError as exc:
        raise ValueError("beads/beta overflows the propagation domain") from exc
    _finite_number(scale, "Ring frequency scale", nonnegative=True)
    _finite_number(maximum_frequency*maximum_frequency,
                   "Squared ring frequency", positive=config.beads > 1, nonnegative=True)
    _finite_number(variance, "Harmonic initial variance", positive=True)
    if config.beads > 1:
        minimum_frequency = scale*math.sin(math.pi/config.beads)
        squared_minimum = minimum_frequency*minimum_frequency
        _finite_number(squared_minimum, "Squared lowest ring frequency", positive=True)
        _finite_number(config.beads/(beta*squared_minimum),
                       "Free fragment ring variance", positive=True)
    phase = config.dt_fs*FS_TO_AU*maximum_frequency
    _finite_number(phase, "Ring propagation phase", nonnegative=True)
    if normalize_integrator(config.integrator) == "velocity-verlet" and phase >= 2:
        raise ValueError("velocity-Verlet ring stability requires dt*max(omega_k) < 2")


@dataclass(frozen=True)
class LaunchConfig(driver.Config):
    """An explicit collision-energy override leaves RNG energy labels intact."""

    collision_energy_override_au: float | None = None

    @property
    def beta(self):
        # Preserve the published runs' exact paper-beta values.  A positive
        # full-PES collision energy can nevertheless occur below the harmonic
        # threshold in P=1 action QCT.  In that case beta is only a required API
        # placeholder: there are no internal ring modes or spring forces.
        if self.beta_override is not None:
            _finite_number(self.beta_override, "beta_override", positive=True)
            return float(self.beta_override)
        _finite_number(self.total_energy_ev, "total_energy_ev", positive=True)
        total = self.total_energy_ev * EV_TO_AU
        paper_domain = (total > (self.v+.5)*driver.OMEGA
                        and 0 < .01*total < driver.OMEGA/2)
        if self.method == "qct" and self.beads == 1 and not paper_domain:
            # A placeholder must never hide a nonpositive physical launch.
            _ = self.collision_energy
            return _QCT_PLACEHOLDER_BETA
        return super().beta

    @property
    def auxiliary_beta(self):
        value = self.beta
        if self.beta_override is not None:
            convention = "explicit_override"
        else:
            total = self.total_energy_ev * EV_TO_AU
            paper_domain = (total > (self.v+.5)*driver.OMEGA
                            and 0 < .01*total < driver.OMEGA/2)
            convention = ("paper_2021" if paper_domain else
                          "qct_single_bead_placeholder_1500_au")
        return {"value_au": value, "convention": convention,
                "physical_role": (
                    "Auxiliary API parameter only: P=1 QCT has no internal ring modes or spring forces; this is not a physical temperature"
                    if self.method == "qct" and self.beads == 1 else
                    "Auxiliary NE-RPMD inverse energy controlling the initial ring ensemble and spring frequencies")}

    @property
    def collision_energy(self):
        value = (super().collision_energy if self.collision_energy_override_au is None
                 else self.collision_energy_override_au)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
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
        _integer(config.beads, "beads", minimum=1)
        if self.kind not in ("direct", "switch", "action_qct"):
            raise ValueError("Unknown preparation kind")
        if self.conditioning not in ("paper", "none"):
            raise ValueError("Conditioning must be paper or none")
        if self.energy_convention not in ("fixed_collision", "fixed_total"):
            raise ValueError("Unknown collision-energy convention")
        for name in ("switch_time_fs", "hold_time_fs"):
            _finite_number(getattr(self, name), name, nonnegative=True)
        for name in ("preparation_dt_fs", "trace_stride_fs"):
            _finite_number(getattr(self, name), name, positive=True)
        _finite_number(self.preparation_dt_fs*FS_TO_AU,
                       "preparation_dt_fs in atomic units", positive=True)
        switching = _grid_steps(self.switch_time_fs, self.preparation_dt_fs,
                                "switch_time_fs")
        holding = _grid_steps(self.hold_time_fs, self.preparation_dt_fs,
                             "hold_time_fs")
        stride = _grid_steps(self.trace_stride_fs, self.preparation_dt_fs,
                            "trace_stride_fs", positive=True)
        if switching or holding:
            if config.beads > 1000000:
                raise ValueError("Isolated preparation supports at most 1000000 beads")
            if (switching+holding)//stride+4 > 10000000:
                raise ValueError("Requested trace is too large; increase trace_stride_fs")
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


def validate_request(config, spec, *, workers=8, audit=False):
    """Reject invalid requests before PES loading, native builds or output.

    A fixed-total action orbit's actual collision energy is surface-dependent;
    ``run`` checks that after solving the orbit and before creating output or
    launching workers.  Every other input is checked here without computing it.
    """
    for name in ("samples", "beads"):
        _integer(getattr(config, name), name, minimum=1)
    for name in ("seed", "id_offset", "v"):
        _integer(getattr(config, name), name)
    _integer(workers, "workers", minimum=1)
    if not isinstance(audit, bool):
        raise ValueError("audit must be a boolean")
    if config.v not in (0, 1):
        raise ValueError("This benchmark supports v=0 or v=1")
    if config.method not in ("qct", "ne_rpmd"):
        raise ValueError("method must be qct or ne_rpmd")
    for name in ("total_energy_ev", "bmax", "dt_fs", "time_cap_fs",
                 "extended_cap_fs", "initial_axial_bohr", "exit_bohr",
                 "bond_cutoff_bohr", "check_fs"):
        _finite_number(getattr(config, name), name, positive=True)
    for name in ("bmin", "persistence_fs", "dmin_bohr"):
        _finite_number(getattr(config, name), name, nonnegative=True)
    for name in ("beta_override", "collision_energy_override_au"):
        value = getattr(config, name)
        if value is not None:
            _finite_number(value, name, positive=True)
    if not config.bmin < config.bmax:
        raise ValueError("Require 0 <= bmin < bmax")
    area = math.pi*(config.bmax*config.bmax-config.bmin*config.bmin)
    _finite_number(area, "Impact-parameter sampling area", positive=True)
    _finite_number(config.total_energy_ev*1_000_000,
                   "total_energy_ev RNG label", positive=True)
    _finite_number(config.dt_fs*FS_TO_AU, "dt_fs in atomic units", positive=True)
    if not config.dt_fs <= config.time_cap_fs <= config.extended_cap_fs:
        raise ValueError("Require 0 < dt_fs <= time_cap_fs <= extended_cap_fs")
    if not config.bond_cutoff_bohr < config.exit_bohr:
        raise ValueError("Require 0 < bond_cutoff_bohr < exit_bohr")
    normalize_integrator(config.integrator)
    for name in ("time_cap_fs", "extended_cap_fs", "check_fs", "persistence_fs"):
        _grid_steps(getattr(config, name), config.dt_fs, name,
                    positive=name != "persistence_fs")
    spec.validate(config)
    if spec.energy_convention == "fixed_collision" and config.collision_energy_override_au is not None:
        nominal = config.total_energy_ev*EV_TO_AU-(config.v+.5)*driver.OMEGA
        if (nominal <= 0 or not math.isclose(config.collision_energy_override_au, nominal,
                                           rel_tol=2e-12, abs_tol=1e-14)):
            raise ValueError("fixed_collision requires the original harmonic collision energy; an override cannot retune it")
    if spec.kind != "action_qct" or spec.energy_convention != "fixed_total":
        _ = config.collision_energy
        _validate_propagation(config)
    elif config.beta_override is not None:
        _validate_propagation(config)


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
                "auxiliary_beta": config.auxiliary_beta,
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
    validate_request(config, spec, workers=workers, audit=audit)
    potential = driver.BKMP2Potential()
    orbit_metadata = None
    if spec.kind == "action_qct":
        from benchmarks.h3.diatomic_action import DiatomicAction
        orbit = DiatomicAction.from_bkmp2(potential, reduced_mass=driver.MASS/2).quantize(config.v)
        orbit_metadata = orbit.metadata
        if spec.energy_convention == "fixed_total":
            config = replace(config, collision_energy_override_au=config.total_energy_ev*EV_TO_AU-orbit.energy_hartree)
    _ = config.collision_energy
    _validate_propagation(config)
    auxiliary_beta = config.auxiliary_beta
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
    snapshot["auxiliary_beta"] = auxiliary_beta
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
                   auxiliary_beta=auxiliary_beta,
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
