"""Explicit tensor PIMD and fixed-cell RPMD transport with atomic batch restart.

All replicas share one device-resident [B,P,N,3] state and one pair of owned
generators. The batch size is scientific input, never a scheduler decision.
A single checkpoint commits state, both RNGs, the launch phase and every
replica's immutable trajectory prefix together. Observations transfer reduced
records; coordinates and momenta leave the device only at checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import platform
from pathlib import Path
import re

import numpy as np

from ..checkpoint import CheckpointError, atomic_savez, load_checkpoint
from ..io import referenced_files
from ..parallel import root_call
from ..propagation import loop_options
from ..trajectory import TrajectoryStream, _descriptor, _walk_reverse
from .common import _fingerprint, checked_integer, write_json


KIND = "PyRPMD.materials-torch.v1"
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")
_FIELDS = {"fingerprint", "sources", "runtime", "q", "p", "step", "time",
           "production_started", "rng_sampling", "rng_production", "trajectories"}


def resolve_torch_materials(config, system):
    """Validate structural choices without importing Torch or loading a model."""
    if config.get("task") not in ("pimd", "diffusion"):
        raise ValueError("sampling.engine='torch' requires pimd or diffusion")
    if system.fixed_atoms or system.bond_constraints:
        raise ValueError("Torch materials currently require all atoms mobile and no bond constraints")
    for name in ("barostat", "path_integral", "active_learning", "electronic_friction"):
        if name in config:
            raise ValueError(f"Torch materials do not support {name}; select a compatible engine explicitly")
    sampling = config.get("sampling", {})
    if sampling.get("thermostat", "pile-l") != "pile-l":
        raise ValueError("Torch materials sampling requires thermostat='pile-l'")
    if sampling.get("integrator", "bcocb") != "bcocb":
        raise ValueError("Torch materials sampling.integrator must be 'bcocb'")
    if sampling.get("thermostat_params") or "andersen_schedule" in sampling:
        raise ValueError("Torch materials PILE-L uses gamma_centroid, not thermostat_params/andersen_schedule")
    loop = loop_options(config)
    if loop["integrator"] not in ("cayley-bcb", "normal-mode-bab"):
        raise ValueError("Torch materials production requires cayley-bcb or normal-mode-bab")
    if loop["loop_backend"] != "auto":
        raise ValueError("Torch materials require dynamics.loop_backend='auto'; sampling.engine selects Torch")
    if config.get("native") == "required" or config.get("dynamics", {}).get("native") == "required":
        raise ValueError("Torch materials cannot satisfy native='required'")
    block = checked_integer(loop["block_steps"], "dynamics.block_steps")
    if block > 65536:
        raise ValueError("dynamics.block_steps must be <= 65536")
    potential = config.get("potential", {})
    provider = potential.get("provider", potential.get("type", "builtin"))
    analytic = ("free", "harmonic", "periodic_cosine")
    bridges = ("torch", "mace-batched", "nequip-batched", "fairchem-batched", "deepmd-batched", "metatomic")
    if provider == "builtin" or provider in analytic:
        model = potential.get("name", potential.get("model", "harmonic" if provider == "builtin" else provider))
        if model not in analytic:
            raise ValueError("Torch materials builtin potential must be free, harmonic or periodic_cosine")
    elif provider not in (*analytic, *bridges):
        raise ValueError("Torch materials require a supported analytic, Torch or explicit batched tensor potential")
    device = sampling.get("device", potential.get("device", "cpu"))
    if not isinstance(device, str) or not _DEVICE.fullmatch(device):
        raise ValueError("sampling.device must be cpu, cuda or cuda:<index>")
    layout = sampling.get("normal_mode_layout", "rfft")
    if layout not in ("rfft", "rpmdrate"):
        raise ValueError("sampling.normal_mode_layout must be rfft or rpmdrate")
    parallel = config.get("parallel", {})
    from ..parallel import execution_plan
    plan = execution_plan(parallel)
    if (plan.backend != "serial" or plan.mode != "serial" or plan.trajectory_workers != 1
            or plan.trajectory_groups != 1 or plan.bead_workers != 1 or plan.devices
            or plan.trajectories_per_gpu != 1 or plan.coordinator_ranks != 0
            or plan.scheduling != "static" or parallel.get("trajectory_batch_size", 1) != 1
            or parallel.get("pes_ranks", 1) != 1):
        raise ValueError("Torch materials use one serial device batch; set sampling.replicas/device, not parallel batching, MPI or workers")
    if config.get("output", {}).get("trajectory_storage", "stream") != "stream":
        raise ValueError("Torch materials require output.trajectory_storage='stream'")
    # Reuse the mature schedule, transport-selection and analysis validation.
    # Only this local validation copy uses the legacy selector. No execution
    # or fallback can follow it, and the real scientific identity is untouched.
    legacy = copy.deepcopy(config)
    legacy.setdefault("sampling", {})["engine"] = "legacy"
    legacy.setdefault("dynamics", {})["integrator"] = "cayley-bcb"
    from .materials import resolve_materials
    options = resolve_materials(legacy, system)
    options.update(device=device, normal_mode_layout=layout, block_steps=block,
                   threads=plan.threads, production_integrator=loop["integrator"])
    return options


def _path(output):
    return Path(output) / "checkpoints" / "materials_torch.npz"


def _identity(config, system):
    physics = copy.deepcopy(config)
    physics.pop("diffusion", None)  # All atom centroids are already recorded.
    return _fingerprint(physics, system, {
        "materials_torch": 1, "rng_policy": "seedsequence-1306-two-owned-generators",
        "diffusion_launch": "independent_real_mass_maxwell",
    })


def _time(step, options):
    return (min(step, options["neq"]) * options["sample_dt"]
            + max(0, step - options["neq"]) * options["production_dt"])


def _scalar(saved, name, kind):
    value = np.asarray(saved[name])
    if value.shape != () or value.dtype.kind not in kind:
        raise CheckpointError(f"invalid Torch materials {name}")
    return value.item()


def _validate_runtime(runtime, options):
    required = {"torch", "device", "dtype", "platform", "machine", "threads", "interop_threads",
                "normal_mode_layout", "deterministic_algorithms", "cuda"}
    if not isinstance(runtime, dict):
        raise CheckpointError("invalid Torch materials runtime identity")
    device = runtime.get("device")
    cuda = isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device) is not None
    if (set(runtime) != required | ({"cuda_device"} if cuda else set())
            or runtime.get("dtype") != "float64" or (device != "cpu" and not cuda)
            or runtime.get("normal_mode_layout") != options["normal_mode_layout"]
            or runtime.get("deterministic_algorithms") is not True
            or any(not isinstance(runtime.get(key), str) or not runtime[key]
                   for key in ("torch", "platform", "machine"))
            or any(type(runtime.get(key)) is not int or runtime[key] < 1
                   for key in ("threads", "interop_threads"))
            or (runtime.get("cuda") is not None and not isinstance(runtime["cuda"], str))):
        raise CheckpointError("invalid Torch materials runtime identity")
    requested = options["device"]
    if (requested == "cpu" and device != "cpu") or (requested != "cpu" and not cuda) or (
            requested.startswith("cuda:") and requested != device):
        raise CheckpointError("Torch materials saved device differs from the requested device")
    if cuda:
        props = runtime["cuda_device"]
        if (not isinstance(props, dict) or set(props) != {"name", "major", "minor"}
                or not isinstance(props["name"], str) or not props["name"]
                or type(props["major"]) is not int or props["major"] < 1
                or type(props["minor"]) is not int or props["minor"] < 0):
            raise CheckpointError("invalid Torch materials CUDA runtime identity")
    return runtime


def _read_batch(config, system, output, options=None):
    """Read and authenticate the one authoritative batch, without a PES/Torch."""
    from .materials import _trajectory_schema, _stream_directory
    options = resolve_torch_materials(config, system) if options is None else options
    with load_checkpoint(_path(output), expected_kind=KIND, require_integrity=True) as saved:
        if set(saved.files) - {"_pyrpmd_checkpoint"} != _FIELDS:
            raise CheckpointError("Torch materials checkpoint fields differ from its schema")
        if _scalar(saved, "fingerprint", "US") != _identity(config, system):
            raise CheckpointError("Torch materials checkpoint scientific settings differ")
        sources = json.loads(_scalar(saved, "sources", "US"))
        if sources != referenced_files(config):
            raise CheckpointError("Torch materials model/input source changed or is missing")
        step = _scalar(saved, "step", "iu")
        time = _scalar(saved, "time", "f")
        started = _scalar(saved, "production_started", "b")
        total = options["neq"] + options["nprod"]
        if (not 0 <= step <= total or not np.isfinite(time)
                or time != _time(step, options)
                or (started and step < options["neq"])
                or (not started and step > options["neq"])):
            raise CheckpointError("invalid Torch materials phase clock")
        shape = (options["replicas"], config["beads"], system.natoms, 3)
        arrays = {}
        for name in ("q", "p"):
            value = saved[name]
            if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
                raise CheckpointError(f"invalid Torch materials {name} state")
            arrays[name] = value.copy()
        for name in ("rng_sampling", "rng_production"):
            value = saved[name]
            if value.dtype != np.uint8 or value.ndim != 1 or not value.size:
                raise CheckpointError(f"invalid Torch materials {name}")
            arrays[name] = value.copy()
        runtime = _validate_runtime(json.loads(_scalar(saved, "runtime", "US")), options)
        trajectories = json.loads(_scalar(saved, "trajectories", "US"))
        if not isinstance(trajectories, list) or len(trajectories) != options["replicas"]:
            raise CheckpointError("Torch materials trajectory count differs from the batch")
        schema = _trajectory_schema(system, options)
        count = (step - options["neq"]) // options["interval"] + 1 if started else 0
        for replica, descriptor in enumerate(trajectories):
            descriptor = _descriptor(descriptor)
            if descriptor["schema"] != schema or descriptor["frames"] != count:
                raise CheckpointError("Torch materials trajectory schema/phase count mismatch")
            seen = 0
            for _, metadata, chunk in _walk_reverse(_stream_directory(output, replica), descriptor):
                expected = (options["neq"] + np.arange(metadata["start"], metadata["stop"],
                                                     dtype=np.int64) * options["interval"])
                if not np.array_equal(chunk["sample_steps"], expected):
                    raise CheckpointError("Torch materials trajectory clock differs from the checkpoint")
                seen += len(expected)
            if seen != count:
                raise CheckpointError("Torch materials trajectory prefix is incomplete")
        return dict(**arrays, step=step, time=time, production_started=started,
                    runtime=runtime, sources=sources, trajectories=trajectories,
                    complete=step == total, frames=count)


@contextmanager
def _execution_context(threads):
    import torch
    old_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(threads)
        # Unsupported deterministic model operations fail explicitly. No
        # reduction of precision or change of device is an error recovery path.
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(old_threads)


def _runtime(device, options):
    import torch
    result = {"torch": str(torch.__version__), "device": str(device), "dtype": "float64",
              "platform": platform.system(), "machine": platform.machine(),
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "normal_mode_layout": options["normal_mode_layout"], "deterministic_algorithms": True,
              "cuda": torch.version.cuda}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result["cuda_device"] = {"name": props.name, "major": props.major, "minor": props.minor}
    return result


def _engines(potential, config, system, options):
    from ..torch_backend import TorchDynamics
    seed = checked_integer(config.get("seed", 0), "seed", 0)
    seeds = [int(np.random.SeedSequence([seed, 1306, stream]).generate_state(1, dtype=np.uint64)[0])
             for stream in (0, 1)]
    common = dict(potential=potential, masses=system.masses, beta=config["beta"],
                  nbeads=config["beads"], normal_mode_layout=options["normal_mode_layout"])
    sampler = TorchDynamics(**common, dt=options["sample_dt"], integrator="cayley-bcocb",
                            thermostat="pile-l", gamma_centroid=options["gamma_centroid"], seed=seeds[0])
    production = TorchDynamics(**common, dt=options["production_dt"],
                               integrator=options["production_integrator"], thermostat=None, seed=seeds[1])
    return sampler, production


def tensor_observables(state, dynamics):
    """Quantum estimators and reduced transport records, all still on device."""
    import torch
    q, p = state.q, state.p
    beads, atoms = q.shape[1:3]
    result = dynamics.potential.evaluate_tensor(q)
    mass = dynamics._mass
    offsets = q - q.mean(dim=1, keepdim=True)
    dq = q - torch.roll(q, -1, dims=1)
    spring = 0.5 * (beads / dynamics.beta)**2 * (mass * dq.square()).sum(dim=(1, 2, 3))
    potential = result.energy.mean(dim=1)
    primitive = 1.5 * atoms * beads / dynamics.beta - spring / beads
    virial = 1.5 * atoms / dynamics.beta - (offsets * result.forces).sum(dim=(1, 2, 3)) / (2 * beads)
    hamiltonian = result.energy.sum(dim=1) + spring + (p.square() / (2 * mass)).sum(dim=(1, 2, 3))
    values = torch.stack((potential, primitive, virial, potential + virial, hamiltonian), dim=1)
    radii = offsets.square().sum(dim=3).mean(dim=1)
    return dict(positions=q.mean(dim=1), velocities=(p / mass).mean(dim=1),
                observables=values, radii_gyration_squared=radii)


def _record(state, dynamics, writers):
    import torch
    values = tensor_observables(state, dynamics)
    # One transfer of the reduced batch per observation, not B*P transfers of
    # atomic positions/forces. Full q,p stay on the selected device.
    names = tuple(values)
    shapes = {name: value.shape[1:] for name, value in values.items()}
    packed = torch.cat([values[name].flatten(start_dim=1) for name in names], dim=1).detach().cpu().numpy()
    if not np.isfinite(packed).all():
        raise FloatingPointError("nonfinite Torch materials observable")
    start = 0
    records = {"sample_steps": np.full(len(writers), state.step, dtype=np.int64)}
    for name in names:
        size = int(np.prod(shapes[name]))
        records[name] = packed[:, start:start + size].reshape(len(writers), *shapes[name])
        start += size
    for replica, writer in enumerate(writers):
        writer.append({name: value[replica] for name, value in records.items()})


def _save(config, system, output, state, sampler, production, started, writers, runtime, sources):
    descriptors = [writer.prepare_commit() for writer in writers]
    snapshot = state.snapshot()
    # This single replace is the commit point for every replica and the
    # Maxwell refresh. Chunks prepared above remain uncommitted if it fails.
    atomic_savez(_path(output), kind=KIND, fingerprint=_identity(config, system),
                 sources=json.dumps(sources, sort_keys=True), runtime=json.dumps(runtime, sort_keys=True),
                 **snapshot, production_started=started,
                 rng_sampling=sampler.generator.get_state().cpu().numpy(),
                 rng_production=production.generator.get_state().cpu().numpy(),
                 trajectories=json.dumps(descriptors, sort_keys=True))


def _run_batch(config, system, output, options, *, saved=None):
    from .materials import _trajectory_schema, _stream_directory
    if saved is not None and saved["complete"]:
        return saved  # No provider construction or RNG consumption on replay.
    import torch
    from ..potentials import make_potential
    from ..potentials.torch_workflow import make_workflow_torch_pes
    from ..torch_backend import TorchState
    provider = make_potential(config["potential"], system)
    try:
        potential = make_workflow_torch_pes(provider, device=options["device"])
        sampler, production = _engines(potential, config, system, options)
        runtime = _runtime(potential.device, options)
        if saved is not None and saved["runtime"] != runtime:
            raise CheckpointError("Torch materials exact resume requires the saved Torch/device/runtime configuration")
        sources = referenced_files(config)
        if saved is None:
            q = torch.as_tensor(system.geometry, dtype=torch.float64, device=potential.device)
            q = q.expand(options["replicas"], config["beads"], system.natoms, 3)
            state = TorchState(q, sampler.sample_momenta(options["replicas"]))
            started = False
        else:
            state = TorchState.from_arrays(saved["q"], saved["p"], device=potential.device,
                                           step=saved["step"], time=saved["time"])
            sampler.generator.set_state(torch.from_numpy(saved["rng_sampling"]))
            production.generator.set_state(torch.from_numpy(saved["rng_production"]))
            started = saved["production_started"]
        writers = [TrajectoryStream(_stream_directory(output, replica),
                    schema=_trajectory_schema(system, options), chunk_frames=options["chunk_frames"],
                    descriptor=None if saved is None else saved["trajectories"][replica], validate=False)
                   for replica in range(options["replicas"])]
        def commit():
            _save(config, system, output, state, sampler, production, started, writers, runtime, sources)
        if saved is None:
            commit()
        total = options["neq"] + options["nprod"]
        while state.step < total:
            if state.step == options["neq"] and not started:
                if config["task"] == "diffusion":
                    # Endpoint BCOCB momenta have finite-step bias. A separate
                    # generator draws real-mass Maxwell momenta exactly once.
                    state.p.copy_(production.sample_momenta(options["replicas"]))
                _record(state, sampler, writers)
                started = True
                commit()
            equilibrating = state.step < options["neq"]
            engine = sampler if equilibrating or config["task"] == "pimd" else production
            stop = min(total, state.step + options["block_steps"],
                       (state.step // options["checkpoint_steps"] + 1) * options["checkpoint_steps"])
            if equilibrating:
                stop = min(stop, options["neq"])
            else:
                next_sample = options["neq"] + ((state.step - options["neq"]) // options["interval"] + 1) * options["interval"]
                stop = min(stop, next_sample)
            engine.propagate(state, stop - state.step)
            state.time = _time(state.step, options)  # Independent of block partition.
            if started and (state.step - options["neq"]) % options["interval"] == 0:
                _record(state, sampler, writers)
            if state.step == total or state.step % options["checkpoint_steps"] == 0:
                commit()
        return _read_batch(config, system, output, options)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _streams(saved, config, system, output, options):
    from ..dynamics import State
    from .materials import _stream_directory
    sampling_steps = saved["step"] if config["task"] == "pimd" else min(saved["step"], options["neq"])
    propagation = {"backend": "torch", "device": saved["runtime"]["device"], "dtype": "float64",
                   "batch_replicas": options["replicas"], "sampling_integrator": "bcocb",
                   "normal_mode_layout": options["normal_mode_layout"],
                   "sampling": {"torch_steps": sampling_steps},
                   "production": {"torch_steps": saved["step"] - sampling_steps},
                   "production_integrator": options["production_integrator"] if config["task"] == "diffusion" else "bcocb",
                   "momentum_refresh": "independent_real_mass_maxwell" if config["task"] == "diffusion" else None}
    return [dict(replica=replica, state=State(saved["q"][replica], saved["p"][replica],
                                            saved["step"], saved["time"]),
                 elapsed=saved["step"], complete=saved["complete"], frames=saved["frames"],
                 storage="stream", trajectory=descriptor,
                 trajectory_directory=str(_stream_directory(output, replica)),
                 checkpoint=str(_path(output)), propagation=copy.deepcopy(propagation), barostat=None)
            for replica, descriptor in enumerate(saved["trajectories"])]


def analyze_torch_materials(config, system, output, return_data=False, *, saved=None):
    from .materials import _analyze
    options = resolve_torch_materials(config, system)
    saved = _read_batch(config, system, output, options) if saved is None else saved
    result = _analyze(_streams(saved, config, system, output, options), config, system, output, return_data)
    summary = result["summary"] if return_data else result
    summary.update(engine="torch", tensor_runtime=saved["runtime"],
                   batch_checkpoint=str(_path(output)), sampling_integrator="bcocb")
    if config["task"] == "pimd":
        summary["integrator"] = "cayley-bcocb"
    else:
        summary["production_momenta"] = "independent_real_mass_maxwell"
    write_json(Path(output) / "summary.json", summary)
    return result


def run_torch_materials(config, system, output, executor=None, return_data=False):
    options = resolve_torch_materials(config, system)
    if executor is not None:
        raise ValueError("Torch materials own one device batch; external trajectory executors are unsupported")
    def run():
        # A complete replay remains usable for analysis without importing
        # Torch or constructing the model (the file hashes are still checked).
        saved = _read_batch(config, system, output, options) if _path(output).exists() else None
        if saved is None or not saved["complete"]:
            with _execution_context(options["threads"]):
                saved = _run_batch(config, system, output, options, saved=saved)
        return analyze_torch_materials(config, system, output, return_data, saved=saved)
    return root_call(run)


def check_torch_materials(config, system, provider):
    """Numerical preparation checks the same tensor contract, with no output."""
    options = resolve_torch_materials(config, system)
    import torch
    from ..potentials.torch_workflow import make_workflow_torch_pes
    with _execution_context(options["threads"]):
        potential = make_workflow_torch_pes(provider, device=options["device"])
        frames = options["replicas"] * config["beads"]
        q = torch.as_tensor(system.geometry, dtype=torch.float64, device=potential.device)
        result = potential.evaluate_tensor(q.expand(frames, system.natoms, 3).contiguous())
        energy, forces = result.energy.detach().cpu().numpy(), result.forces.detach().cpu().numpy()
        if not np.isfinite(energy).all() or not np.isfinite(forces).all():
            raise FloatingPointError("nonfinite Torch materials preflight energy/force")
        return {"engine": "torch", "device": str(potential.device), "dtype": "float64",
                "energy_hartree": energy.tolist(), "status": [0] * frames,
                "force_max_au": float(np.max(np.abs(forces))), "frames": frames, "step": 0}
