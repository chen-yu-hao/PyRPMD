"""GPU execution pieces for gas-phase US/AUS and RPMD recrossing.

The workflow keeps its existing NumPy files, MBAR/UI analysis and checkpoint
identities.  Only live q/p, physical forces, CV forces and ring propagation
remain on the selected device.  Host transfers are limited to checkpoints,
reduced CV/energy observations and the final recrossing arrays.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import atomic_savez as _atomic_npz, load_checkpoint
from ..input import rate_pmf_options
from ..io import fingerprint as _fingerprint
from ..potentials import make_potential
from ..potentials.torch_workflow import make_workflow_torch_pes
from ..torch_backend import TorchDynamics, TorchState
from ..torch_rate import (TorchBiasStack, TorchFluxMetricBias, TorchRateSurface,
                          TorchUmbrellaBias, torch_cv_from_cpu)
from .common import checked_integer
from .rate_constraints import _normal, rate_coordinate
from .rate_options import _pmf_method, _recrossing_options, _recrossing_spec
from .sampling import AndersenSchedule, validate_andersen_schedule


def enabled(config):
    return config.get("sampling", {}).get("engine") == "torch"


def _seed(config, *parts):
    return int(np.random.SeedSequence([int(config.get("seed", 0)), *parts]).generate_state(1, dtype=np.uint64)[0])


def _engine(provider, config, system, *, seed, bias=None, warmup_frames=None):
    sampling = config.get("sampling", {})
    pot_spec = config.get("potential", {})
    default_device = (
        pot_spec.get("device", getattr(provider, "device", "cpu"))
        if isinstance(pot_spec, dict)
        else getattr(provider, "device", "cpu")
    )
    device = sampling.get("device", default_device)
    layout = sampling.get("normal_mode_layout", "rfft")
    potential = make_workflow_torch_pes(provider, device=device)
    scheme = config.get("dynamics", {}).get("integrator", "cayley-bcb")
    if scheme == "velocity-verlet":
        raise ValueError("GPU rate workflow requires cayley-bcb or normal-mode-bab")
    engine = TorchDynamics(
        potential, system.masses, config["beta"], config["beads"], config["dt_au"],
        integrator=scheme, thermostat=None, seed=seed, bias=bias,
        normal_mode_layout=layout,
    )
    warmup = getattr(provider, "warmup_torch", None)
    if callable(warmup):
        warmup(int(warmup_frames or config["beads"]), system.natoms)
    return potential, engine


def _torch_rng(engine):
    return engine.generator.get_state().cpu().numpy()


def _restore_torch_rng(engine, value):
    engine.generator.set_state(torch.as_tensor(np.asarray(value), dtype=torch.uint8))


def _read_checkpoint(path, identity):
    if not path.exists():
        return None
    with load_checkpoint(path) as saved:
        if str(saved["fingerprint"]) != identity:
            raise ValueError(f"GPU rate checkpoint physics differs: {path}")
        return {name: saved[name].copy() for name in saved.files if name != "_pyrpmd_checkpoint"}


def _sampling_bias(config, cv, center, stiffness, system):
    tensor_cv = torch_cv_from_cpu(cv)
    biases = [TorchUmbrellaBias(tensor_cv, center, stiffness, cv.periods)]
    from .fes import _flux_coordinate
    flux = _flux_coordinate(config, cv)
    if flux is not None:
        flux_cv = torch_cv_from_cpu(flux)
        biases.append(TorchFluxMetricBias(flux_cv, system.masses, config["beta"]))
    return TorchBiasStack(biases), tensor_cv


def sample_window(task):
    """GPU replacement for the scalar fixed-window sampler."""
    from .fes import (_make_tasks, _window_checkpoint, _window_fingerprint,
                      _centroid_accumulator, separate_reactants_to_xi,
                      steer_to_center)
    from ..cv import make_cvs, RPMDrateCoordinate
    config, system = task["config"], task["system"]
    cv = make_cvs(task["specs"], system)
    for component in cv.cvs:
        if isinstance(component, RPMDrateCoordinate):
            component.regular_branch_only = True
    center, stiffness = np.asarray(task["center"]), np.asarray(task["stiffness"])
    sampling = config["sampling"]
    if sampling.get("thermostat", "andersen") != "andersen":
        raise ValueError("GPU rate US/AUS currently requires Andersen sampling")
    neq = checked_integer(sampling["equilibration_steps"], "sampling.equilibration_steps", 0)
    nprod = checked_integer(task.get("production_steps", sampling["production_steps"]), "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    checkpoint_steps = checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    checkpoint = _window_checkpoint(task)
    identity = _window_fingerprint(task)
    saved = _read_checkpoint(checkpoint, identity)
    bias, tensor_cv = _sampling_bias(config, cv, center, stiffness, system)
    provider = make_potential(config["potential"], system)
    try:
        _, engine = _engine(provider, config, system,
                            seed=_seed(config, 101, task["window"], task["replica"]),
                            bias=bias, warmup_frames=config["beads"])
        rng = np.random.default_rng(_seed(config, 101, task["window"], task["replica"], 991))
        if saved is None:
            geometry = system.geometry
            initialization = task.get("initialization", {})
            if "q" in initialization:
                q = np.array(initialization["q"], dtype=np.float64, copy=True)
            elif config.get("warm_start"):
                from ..warm_start import initial_ring
                q = initial_ring(config, system)
            elif isinstance(cv.cvs[0], RPMDrateCoordinate) and not system.fixed_atoms:
                geometry = separate_reactants_to_xi(geometry, cv.cvs[0], center[0], system.masses)
                q = np.repeat(geometry[None], config["beads"], axis=0)
            else:
                q = np.repeat(geometry[None], config["beads"], axis=0)
            q = steer_to_center(q, cv, center, system.masses, fixed_atoms=system.fixed_atoms)
            state = TorchState.from_arrays(q[None], engine.sample_momenta(), device=engine.device)
            elapsed, values, energies = 0, [], []
        else:
            state = TorchState.from_arrays(saved["q"], saved["p"], device=engine.device,
                                           step=int(saved["step"]), time=float(saved["time"]))
            elapsed = int(saved["elapsed"])
            values = [row.copy() for row in np.asarray(saved["values"])]
            energies = [float(x) for x in np.asarray(saved["energies"])]
            rng.bit_generator.state = json.loads(str(saved["rng"]))
            _restore_torch_rng(engine, saved["torch_rng"])

        clock = AndersenSchedule(
            checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval"),
            validate_andersen_schedule(config, "rate"),
        )
        if saved is not None:
            clock.restore(saved, elapsed)

        def save():
            _atomic_npz(
                checkpoint, q=state.q.cpu().numpy(), p=state.p.cpu().numpy(),
                step=np.asarray(state.step, dtype=np.int64), time=np.asarray(state.time),
                elapsed=np.asarray(elapsed, dtype=np.int64), values=np.asarray(values),
                energies=np.asarray(energies), rng=json.dumps(rng.bit_generator.state),
                torch_rng=_torch_rng(engine), fingerprint=identity,
                window_id=task.get("window_id", ""), center=center, stiffness=stiffness,
                **clock.checkpoint_fields(),
                propagation=json.dumps({"backend": "torch", "device": str(engine.device),
                                        "dtype": "float64", "torch_steps": elapsed}),
            )

        save()
        while elapsed < neq + nprod:
            if clock.due(state.step):
                state.p.copy_(engine.sample_momenta())
                clock.refreshed(state.step, rng)
            count = min(neq + nprod - elapsed, checkpoint_steps - elapsed % checkpoint_steps,
                        clock.steps_until_collision(state.step))
            for _ in range(count):
                engine.propagate(state, 1)
                elapsed += 1
                production = elapsed - neq
                if production > 0 and production % interval == 0:
                    value = tensor_cv.evaluate_tensor(state.q)[0].detach().cpu().numpy()[0]
                    energy = float(engine.energy(state).detach().cpu().numpy()[0])
                    if not np.isfinite(value).all() or not np.isfinite(energy):
                        raise FloatingPointError("nonfinite GPU umbrella observable")
                    values.append(value.copy())
                    energies.append(energy)
            if elapsed % checkpoint_steps == 0:
                save()
        save()
        if len(values) < 2:
            raise ValueError("at least two samples per umbrella stream required")
        return {
            "window": task["window"], **({"window_id": task["window_id"]} if "window_id" in task else {}),
            "replica": task["replica"], "samples": np.asarray(values), "center": center,
            "stiffness": stiffness, "last_q": state.q.cpu().numpy()[0],
            "last_p": state.p.cpu().numpy()[0], "energies": np.asarray(energies),
            "checkpoint": str(checkpoint), "native": False, "engine": "torch",
            "propagation": {"backend": "torch", "device": str(engine.device),
                            "dtype": "float64", "torch_steps": elapsed},
            "centroid_moments": None,
        }
    finally:
        provider.close()


def _surface_engine(task, *, seed, warmup_frames=None):
    from .rate_constraints import rate_coordinate
    surface = rate_coordinate(task["config"], task["system"]).dividing_surface(task["xi_star"])
    tensor_surface = TorchRateSurface(surface)
    provider = make_potential(task["config"]["potential"], task["system"])
    _, engine = _engine(provider, task["config"], task["system"], seed=seed,
                        warmup_frames=warmup_frames)
    return provider, engine, tensor_surface, surface


@torch.no_grad()
def _constrained_step(state, engine, surface):
    q, p = state.q, state.p
    mass = engine._masses[None, :, None]
    dt = engine.dt
    value0, g0 = surface.evaluate_centroid(q.mean(dim=1))
    force = engine.potential.evaluate_tensor(q).forces
    p = p + 0.5 * dt * force
    q, p = engine._free(q, p)
    free = q.clone()
    direction = g0 / mass
    lam = torch.zeros_like(value0)
    for _ in range(50):
        value, _ = surface.evaluate_centroid(q.mean(dim=1))
        derivative = (direction * surface.evaluate_centroid(q.mean(dim=1))[1]).sum(dim=(1, 2))
        delta = -value / derivative
        lam = lam + delta
        q = free + lam[:, None, None, None] * direction[:, None]
        if torch.max(torch.abs(value)).item() < 1e-10:
            break
    else:
        raise RuntimeError("GPU centroid RATTLE failed to converge")
    p = p + lam[:, None, None, None] / dt * g0[:, None]
    force = engine.potential.evaluate_tensor(q).forces
    p = p + 0.5 * dt * force
    value, gradient = surface.evaluate_centroid(q.mean(dim=1))
    z = (gradient.square() / mass).sum(dim=(1, 2))
    velocity = (gradient * p.mean(dim=1) / mass).sum(dim=(1, 2))
    p = p - (velocity / z)[:, None, None, None] * gradient[:, None]
    state.q.copy_(q)
    state.p.copy_(p)
    state.step += 1
    state.time += dt
    return state


def parent_task(task):
    """GPU constrained parent chain used by the rate recrossing stage."""
    from .rate_recrossing import _parent_thermostat
    from ..checkpoint import load_checkpoint
    config, system = task["config"], task["system"]
    options = task["options"]
    chain = int(task["chain"])
    checkpoint = Path(task["output"]) / "checkpoints" / f"parent_{chain:04d}.npz"
    identity = _fingerprint(config, system, {"stage": "parent", "chain": chain,
                                             "xi": task["xi_star"], "options": options,
                                             **({"generation_identity": task["generation_identity"],
                                                "initial_q": task["initial_q"]} if "generation_identity" in task else {})})
    provider, engine, surface, _ = _surface_engine(task, seed=_seed(config, 211, chain),
                                                   warmup_frames=config["beads"])
    try:
        saved = _read_checkpoint(checkpoint, identity)
        rng = np.random.default_rng(_seed(config, 211, chain, 991))
        if saved is None:
            q = np.asarray(task["initial_q"], dtype=np.float64)
            state = TorchState.from_arrays(q[None], engine.sample_momenta(), device=engine.device)
            snapshots, elapsed = [], 0
        else:
            state = TorchState.from_arrays(saved["q"], saved["p"], device=engine.device,
                                           step=int(saved["step"]), time=float(saved["time"]))
            snapshots, elapsed = [x.copy() for x in np.asarray(saved["snapshots"])], int(saved["elapsed"])
            rng.bit_generator.state = json.loads(str(saved["rng"]))
            _restore_torch_rng(engine, saved["torch_rng"])
        clock = AndersenSchedule(*_parent_thermostat(config))
        if saved is not None:
            clock.restore(saved, elapsed)
        target = options["equilibration_steps"] + options["parent_samples"] * options["parent_interval_steps"]
        while elapsed < target:
            if clock.due(state.step):
                state.p.copy_(engine.sample_momenta())
                # Parent momentum must be tangent to the dividing surface.
                _, gradient = surface.evaluate_centroid(state.q.mean(dim=1))
                mass = engine._masses[None, :, None]
                z = (gradient.square() / mass).sum(dim=(1, 2))
                v = (gradient * state.p.mean(dim=1) / mass).sum(dim=(1, 2))
                state.p.sub_((v / z)[:, None, None, None] * gradient[:, None])
                clock.refreshed(state.step, rng)
            count = min(target - elapsed, clock.steps_until_collision(state.step), 4096)
            for _ in range(count):
                _constrained_step(state, engine, surface)
                elapsed += 1
                production = elapsed - options["equilibration_steps"]
                if production > 0 and production % options["parent_interval_steps"] == 0:
                    snapshots.append(state.q.cpu().numpy()[0].copy())
        _atomic_npz(checkpoint, q=state.q.cpu().numpy()[0], p=state.p.cpu().numpy()[0],
                    step=np.asarray(state.step), time=np.asarray(state.time), elapsed=np.asarray(elapsed),
                    snapshots=np.asarray(snapshots), rng=json.dumps(rng.bit_generator.state),
                    torch_rng=_torch_rng(engine), fingerprint=identity, xi_star=task["xi_star"],
                    pmf_method=_pmf_method(config), **clock.checkpoint_fields(),
                    propagation=json.dumps({"backend": "torch", "torch_steps": elapsed}))
        return {"chain": chain, "snapshots": np.asarray(snapshots), "propagation": {"backend": "torch", "torch_steps": elapsed},
                "propagation_this_call": {"torch_steps": elapsed}, "checkpoint": str(checkpoint),
                "xi_star": task["xi_star"], "pmf_method": _pmf_method(config)}
    finally:
        provider.close()


def _child_batch(tasks):
    """Vectorize opposite-momentum child pairs on one device."""
    if not tasks:
        return []
    config, system = tasks[0]["config"], tasks[0]["system"]
    options = _recrossing_options(config)
    provider, engine, surface, cpu_surface = _surface_engine(
        tasks[0], seed=_seed(config, 307, len(tasks)),
        warmup_frames=2 * len(tasks) * config["beads"])
    try:
        pending, cached_results = [], {}
        for original_index, task in enumerate(tasks):
            checkpoint = Path(task["output"]) / "checkpoints" / f"child_{task['chain']:04d}_{task['parent']:05d}_{task['pair']:04d}.npz"
            identity = _fingerprint(config, system, {"stage": "child", "chain": task["chain"],
                                                     "parent": task["parent"], "pair": task["pair"],
                                                     "xi": task["xi_star"], "child_steps": task["child_steps"], "q": task["q"]})
            cached = _read_checkpoint(checkpoint, identity)
            if cached is not None:
                cached_results[original_index] = (task, cached, checkpoint, identity)
                continue
            q = np.asarray(task["q"], dtype=np.float64)
            value, gradient, metric = _normal(cpu_surface, q, system.masses)
            if abs(value) > max(1e-8, 10 * options["constraint_tolerance"]):
                raise ValueError("child starting configuration is not on the dividing surface")
            fs = np.sqrt(metric / (2 * np.pi * config["beta"]))
            generator = torch.Generator(device=engine.device).manual_seed(_seed(config, 307, task["chain"], task["parent"], task["pair"]))
            momentum = torch.randn((1, config["beads"], system.natoms, 3), dtype=torch.float64,
                                   device=engine.device, generator=generator) * engine._sigma
            velocity = float(np.sum(gradient * momentum[0].mean(dim=0).cpu().numpy() / system.masses[:, None]))
            pending.append({"index": original_index, "task": task, "checkpoint": checkpoint,
                            "identity": identity,
                            "q": torch.as_tensor(q, dtype=torch.float64, device=engine.device),
                            "p": momentum[0].detach(),
                            "weights": (velocity / fs, -velocity / fs),
                            "numerator": np.zeros(task["child_steps"], dtype=np.float64),
                            "initial_energy": None})
        if pending:
            q = torch.stack([item["q"] for item in pending for _ in (0, 1)])
            p = torch.stack([item["p"] if sign == 0 else -item["p"]
                             for item in pending for sign in (0, 1)])
            state = TorchState(q, p)
            initial_energy = engine.energy(state).detach().cpu().numpy()
            for item, pair_energy in zip(pending, initial_energy.reshape(-1, 2)):
                item["initial_energy"] = pair_energy
            for step in range(max(len(item["numerator"]) for item in pending)):
                engine.propagate(state, 1)
                for local, item in enumerate(pending):
                    if step >= len(item["numerator"]):
                        continue
                    values = surface.evaluate_centroid(state.q[2 * local:2 * local + 2].mean(dim=1))[0]
                    hit = values.detach().cpu().numpy() > 0
                    item["numerator"][step] = item["weights"][0] * hit[0] + item["weights"][1] * hit[1]
            finals = engine.energy(state).detach().cpu().numpy().reshape(-1, 2)
            for item, final_energy in zip(pending, finals):
                delta = np.abs(final_energy - item["initial_energy"])
                item["energy_drift"] = float(np.max(delta / np.maximum(
                    np.abs(item["initial_energy"]), config["beads"] / config["beta"])))
                item["endpoint_error"] = float(np.max(delta) / config["beads"])
        results = []
        for original_index, task in enumerate(tasks):
            if original_index in cached_results:
                _task, cached, checkpoint, identity = cached_results[original_index]
                result = {"chain": task["chain"], "parent": task["parent"], "pair": task["pair"],
                          "xi_star": task["xi_star"], "pmf_method": _pmf_method(config),
                          "numerator": cached["numerator"], "denominator": float(cached["denominator"]),
                          "energy_drift": float(cached["energy_drift"]),
                          "endpoint_energy_error_hartree_per_bead": float(cached["endpoint_energy_error_hartree_per_bead"]) if "endpoint_energy_error_hartree_per_bead" in cached else None,
                          "endpoint_energy_error_kbt": float(cached["endpoint_energy_error_kbt"]) if "endpoint_energy_error_kbt" in cached else None,
                          "propagation": json.loads(str(cached["propagation"]))}
            else:
                item = next(value for value in pending if value["index"] == original_index)
                denominator = abs(item["weights"][0])
                _atomic_npz(checkpoint, numerator=item["numerator"], denominator=denominator,
                            energy_drift=item["energy_drift"],
                            endpoint_energy_error_hartree_per_bead=item["endpoint_error"],
                            endpoint_energy_error_kbt=item["endpoint_error"] * config["beta"],
                            fingerprint=item["identity"], xi_star=task["xi_star"],
                            pmf_method=_pmf_method(config), propagation=json.dumps({"backend": "torch"}))
                result = {"chain": task["chain"], "parent": task["parent"], "pair": task["pair"],
                          "xi_star": task["xi_star"], "pmf_method": _pmf_method(config),
                          "numerator": item["numerator"], "denominator": denominator,
                          "energy_drift": item["energy_drift"],
                          "endpoint_energy_error_hartree_per_bead": item["endpoint_error"],
                          "endpoint_energy_error_kbt": item["endpoint_error"] * config["beta"],
                          "propagation": {"backend": "torch"}}
            results.append(result)
        return results
    finally:
        provider.close()


def child_task(task):
    return _child_batch([task])[0]


def dispatch_children(tasks, config, dispatch):
    size = checked_integer(_recrossing_spec(config).get("child_batch_pairs", 1),
                           "recrossing.child_batch_pairs")
    groups = [tasks[i:i + size] for i in range(0, len(tasks), size)]
    grouped = dispatch(_child_batch, groups)
    return None if grouped is None else [item for group in grouped for item in group]


__all__ = ["enabled", "sample_window", "parent_task", "child_task", "dispatch_children"]
