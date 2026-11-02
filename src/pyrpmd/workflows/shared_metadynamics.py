"""One atomic ensemble checkpoint for synchronous shared-bias MetaD."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import csv
import json
from pathlib import Path

import numpy as np

from ..analysis.metadynamics import bias_fes
from ..bias import CompositeBias
from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..potentials import make_potential
from ..propagation import loop_options
from ..shared_metadynamics import SharedMetadynamicsBias
from ..units import parse_quantity
from .common import _fingerprint, fingerprint_matches, write_json
from .metadynamics import resolve_metadynamics, resolve_walls
from .sampling import legacy_metadynamics_sampling, validate_andersen_schedule


KIND = "PyRPMD.shared-metadynamics.v1"


def validate_shared(config, options):
    validate_andersen_schedule(config, "metadynamics")
    if not isinstance(config["metadynamics"].get("shared_walkers", False), bool):
        raise ValueError("metadynamics.shared_walkers must be boolean")
    if config["metadynamics"].get("multiple_walkers", False):
        raise ValueError("metadynamics.shared_walkers and multiple_walkers are distinct modes; select only one")
    walkers = config["sampling"].get("replicas", 1)
    if walkers < 2:
        raise ValueError("shared metadynamics requires sampling.replicas >= 2")
    parallel = config.get("parallel", {})
    if (parallel.get("backend", "serial") != "serial" or parallel.get("mode", "serial") != "serial"
            or parallel.get("trajectory_workers", 1) != 1 or parallel.get("bead_workers", 1) != 1):
        raise ValueError("synchronous shared walkers currently require a serial coordinator")
    if options["grid"] is not None:
        raise ValueError("shared walkers currently use direct Gaussian hills, without a bias grid")
    if (config["sampling"].get("thermostat", "andersen") not in ("pile-l", "andersen")
            or not legacy_metadynamics_sampling(config["sampling"])):
        raise ValueError("shared walkers support legacy PILE-L or Andersen sampling")
    return walkers


def _bias(cv, config, options, hills=None):
    return SharedMetadynamicsBias(cv, options["sigma"], options["height"], config["beta"],
        walkers=config["sampling"]["replicas"], mode=options["mode"],
        bias_factor=options["bias_factor"], hills=hills)


def _read(config, system, output, cv, options):
    path = Path(output) / "checkpoints" / "shared_metadynamics.npz"
    walkers = validate_shared(config, options)
    with load_checkpoint(path, expected_kind=KIND, require_integrity=True) as saved:
        if not fingerprint_matches(saved["fingerprint"], config, system, {"shared_metadynamics": 1}):
            raise ValueError("shared metadynamics checkpoint physics differs")
        raw_elapsed = np.asarray(saved["elapsed"])
        if raw_elapsed.ndim != 0 or raw_elapsed.dtype.kind not in "iu":
            raise ValueError("shared metadynamics clock must be an integer scalar")
        elapsed = int(raw_elapsed)
        if not 0 <= elapsed <= options["neq"]+options["nprod"]:
            raise ValueError("invalid shared metadynamics clock")
        shape = (walkers, config["beads"], system.natoms, 3)
        if saved["q"].shape != shape or saved["p"].shape != shape:
            raise ValueError("shared metadynamics state shape mismatch")
        dt = config["sampling"].get("dt_au", config["dt_au"])
        time = float(saved["time"])
        if not np.isclose(time, elapsed*dt, rtol=1e-10, atol=1e-10):
            raise ValueError("shared metadynamics time disagrees with step")
        states = [State(saved["q"][i], saved["p"][i], elapsed, time) for i in range(walkers)]
        if system.fixed_atoms:
            fixed = np.asarray(system.fixed_atoms, dtype=int)
            for state in states:
                expected = np.broadcast_to(system.geometry[fixed], state.q[:, fixed].shape)
                if not np.array_equal(state.q[:, fixed], expected) or np.any(state.p[:, fixed] != 0.):
                    raise ValueError("shared checkpoint fixed atoms must retain reference positions and zero momenta")
        hills = json.loads(str(saved["hills"]))
        bias = _bias(cv, config, options, hills)
        production = max(0, elapsed-options["neq"])
        expected_hills = walkers*(production//options["pace"])
        expected_steps = np.arange(options["neq"]+options["interval"], elapsed+1, options["interval"])
        if bias.hill_count != expected_hills:
            raise ValueError("shared metadynamics history disagrees with clock")
        for event, hill in enumerate(hills):
            step = options["neq"]+(event//walkers+1)*options["pace"]
            if hill["step"] != step or not np.isclose(hill["time"], step*dt, rtol=1e-10, atol=1e-10):
                raise ValueError("shared hill clock disagrees with deposition schedule")
        samples = saved["samples"]
        if samples.shape != (len(expected_steps), walkers, cv.dimension) or not np.isfinite(samples).all():
            raise ValueError("invalid shared metadynamics samples")
        if not np.array_equal(saved["sample_steps"], expected_steps):
            raise ValueError("shared sample clock is inconsistent")
        sample_bias = saved["sample_bias"]
        if sample_bias.shape != (len(expected_steps), walkers) or not np.isfinite(sample_bias).all():
            raise ValueError("invalid shared metadynamics sampled bias")
        configurations = saved["configurations"]
        expected_shape = (len(expected_steps), *shape) if options["store_configurations"] else (0, *shape)
        if configurations.shape != expected_shape or not np.isfinite(configurations).all():
            raise ValueError("invalid shared metadynamics configurations")
        rng = json.loads(str(saved["rng"]))
        if not isinstance(rng, list) or len(rng) != walkers:
            raise ValueError("shared checkpoint requires one RNG per walker")
        return dict(states=states, rng=rng, bias=bias, elapsed=elapsed,
            samples=list(samples), sample_steps=list(saved["sample_steps"]),
            sample_bias=list(sample_bias), configurations=list(configurations))


def _analyze(config, system, output, cv, options, saved, return_data):
    output = Path(output)
    walkers = config["sampling"]["replicas"]
    samples = np.asarray(saved["samples"])
    if not samples.size:
        raise ValueError("shared metadynamics has no saved production samples")
    # A single history: pool only the occupancy histogram. Never report walker
    # spread as if the walkers were independent bias estimates.
    data = bias_fes([saved["bias"]], [samples.reshape(-1, cv.dimension)], bins=options["bins"],
        ranges=options["ranges"], min_count=options["min_count"], max_grid_points=options["max_grid_points"])
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    arrays.update({f"edges_{i}": edge for i, edge in enumerate(data["edges"])})
    atomic_savez(output / "fes.npz", kind="PyRPMD.shared-metadynamics-fes.v1", **arrays)
    centers = np.meshgrid(*[(x[1:]+x[:-1])/2 for x in data["edges"]], indexing="ij")
    np.savetxt(output / "fes.csv", np.column_stack([x.ravel() for x in centers]+[
        data["free_energy"].ravel(), data["mask"].ravel(), data["raw_count"].ravel()]),
        delimiter=",", comments="", header=",".join([f"cv{i}_{u}" for i,u in enumerate(cv.units)]+[
        "bias_estimated_free_energy_hartree", "masked", "raw_count"]))
    hill_file = output / "shared-hills.csv"
    with hill_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_au", "walker", "height_hartree", "bias_before_hartree"]+
                        [f"cv{i}_{u}" for i,u in enumerate(cv.units)])
        for hill in saved["bias"].hills:
            writer.writerow([hill[key] for key in ("step", "time", "walker", "height", "bias_before")]+hill["center"])
    complete = saved["elapsed"] == options["neq"]+options["nprod"]
    summary = {"task":"metadynamics", "status":"bias_estimate" if complete else "partial_bias_estimate", "mode":options["mode"],
        "shared_bias":True, "walkers":walkers, "independent_histories":1,
        "samples":int(samples.shape[0]*walkers), "hills":saved["bias"].hill_count,
        "dimension":cv.dimension, "units":cv.units, "measure":"cv_density", "energy_unit":"hartree",
        "estimator":"bias_estimate", "fes_factor":data["fes_factor"], "bias_factor":options["bias_factor"],
        "wall_ensemble":bool(config["metadynamics"].get("walls")),
        "stored_configurations":options["store_configurations"],
        "sample_bias_convention":"physical shared V before all simultaneous hills; FES includes the wall-defined ensemble",
        "converged":False, "uncertainty":None,
        "files":{"data":str(output/"fes.npz"), "table":str(output/"fes.csv"),
                 "hills":str(hill_file), "checkpoint":str(output/"checkpoints"/"shared_metadynamics.npz")},
        "warnings":["One shared history couples walkers; walker spread is not an independent-replica error bar.",
            "Finite-time bias FES only; sampling and CV sufficiency are not established.",
            "Each walker deposits the specified height, all tempered against the same pre-deposition bias."]}
    write_json(output / "summary.json", summary)
    return {"summary":summary, "data":data, "ensemble":saved} if return_data else summary


def run_shared_metadynamics(config, system, output, *, return_data=False):
    cv, _, options = resolve_metadynamics(config, system)
    walkers = validate_shared(config, options)
    path = Path(output) / "checkpoints" / "shared_metadynamics.npz"
    with ExitStack() as stack:
        dynamics = []
        for walker in range(walkers):
            provider = make_potential(config["potential"], system)
            stack.callback(provider.close)
            dynamics.append(Dynamics(provider, system.masses, config["beta"], config["beads"],
                config["sampling"].get("dt_au", config["dt_au"]),
                seed=np.random.SeedSequence([config.get("seed",0), 2267, walker]),
                native=config.get("native", config.get("dynamics",{}).get("native",True)),
                threads=config.get("parallel",{}).get("threads_per_rank",1),
                fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry, **loop_options(config)))
        if path.exists():
            saved = _read(config, system, output, cv, options)
            states, bias, elapsed = saved["states"], saved["bias"], saved["elapsed"]
            for dyn, state, rng in zip(dynamics, states, saved["rng"]):
                dyn.integrator._check(state.q, state.p)
                dyn._check_fixed(state.q, state.p)
                dyn.rng.bit_generator.state = rng
        else:
            states = [State(np.repeat(system.geometry[None],config["beads"],axis=0),dyn.sample_momenta()) for dyn in dynamics]
            bias, elapsed = _bias(cv,config,options), 0
            saved = dict(samples=[],sample_steps=[],sample_bias=[],configurations=[])
        walls = resolve_walls(config,system)
        total_bias = CompositeBias(bias,walls)
        shape = (walkers,config["beads"],system.natoms,3)
        def save():
            atomic_savez(path,kind=KIND,fingerprint=_fingerprint(config,system,{"shared_metadynamics":1}),
                elapsed=elapsed,time=states[0].time,q=np.stack([s.q for s in states]),p=np.stack([s.p for s in states]),
                rng=json.dumps([d.rng.bit_generator.state for d in dynamics]),hills=json.dumps(bias.hills),
                sample_steps=np.asarray(saved["sample_steps"],dtype=np.int64),
                samples=np.asarray(saved["samples"]).reshape(-1,walkers,cv.dimension),
                sample_bias=np.asarray(saved["sample_bias"]).reshape(-1,walkers),
                configurations=np.asarray(saved["configurations"]).reshape(-1,*shape))
        save()
        sampling = config["sampling"]
        while elapsed < options["neq"]+options["nprod"]:
            previous = [state.copy() for state in states]
            rng = [copy.deepcopy(dyn.rng.bit_generator.state) for dyn in dynamics]
            count = bias.hill_count
            try:
                active = total_bias if elapsed >= options["neq"] or config["metadynamics"].get("walls_during_equilibration",True) else bias
                for dyn,state in zip(dynamics,states):
                    if sampling.get("thermostat","andersen") == "pile-l":
                        dyn.sample_step(state,active,gamma_centroid=parse_quantity(sampling.get("gamma_centroid","0.01 fs^-1"),"frequency"))
                    else:
                        if state.step % sampling.get("andersen_interval",10) == 0:
                            state.p[:]=dyn.sample_momenta()
                        dyn.step(state,active)
                production = elapsed+1-options["neq"]
                record = production > 0 and production % options["interval"] == 0
                deposit = production > 0 and production % options["pace"] == 0
                values = np.asarray([cv.value(state.q) for state in states]) if record or deposit else None
                before = np.asarray(bias.physical_energy(values)) if record else None
                if deposit:
                    bias.deposit_walkers(values,step=states[0].step,time=states[0].time)
            except BaseException:
                states=previous
                for dyn,state_rng in zip(dynamics,rng):
                    dyn.rng.bit_generator.state=state_rng
                bias.truncate_history(count)
                save()
                raise
            elapsed+=1
            if record:
                saved["samples"].append(values.copy())
                saved["sample_steps"].append(elapsed)
                saved["sample_bias"].append(before.copy())
                if options["store_configurations"]:
                    saved["configurations"].append(np.stack([state.q for state in states]))
            if elapsed % config.get("output",{}).get("checkpoint_steps",1000) == 0:
                save()
        save()
    saved=_read(config,system,output,cv,options)
    return _analyze(config,system,output,cv,options,saved,return_data)


def reanalyze_shared_metadynamics(config, system, output, *, return_data=False):
    cv,_,options=resolve_metadynamics(config,system)
    saved=_read(config,system,output,cv,options)
    return _analyze(config,system,output,cv,options,saved,return_data)
