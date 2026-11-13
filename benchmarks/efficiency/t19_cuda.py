#!/usr/bin/env python3
"""Reproducible T19 CUDA measurements, without model or device fallbacks.

Run from an installed checkout, with locally supplied, trusted model files::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        benchmarks/efficiency/t19_cuda.py --manifest models.json \
        --output cuda.json --markdown cuda.md

The default matrix is P=8,32,64 and N=64,512,4096. ``--device cpu`` explicitly
requests a CPU diagnostic; it never constitutes CUDA evidence. Missing CUDA,
models, optional packages, or the unmerged PR #84 workflow are recorded, and
produce a nonzero exit code. No model is downloaded, cast to another precision,
or split into smaller batches after an out-of-memory failure.

Manifest example (paths are relative to the manifest; SHA256 is mandatory)::

    {"schema": "pyrpmd.t19-models.v1", "models": [
      {"name": "my-mace", "provider": "mace-batched",
       "model": "local.model", "sha256": "<64 lowercase hex digits>",
       "model_kind": "trained", "dtype": "float64", "params": {"virial": true},
       "geometry": {"species": ["Si"], "spacing_bohr": 5.0,
                    "jitter_bohr": 0.02, "pbc": [true, true, true]}}
    ]}

Use providers mace-batched, nequip-batched, fairchem-batched, deepmd-batched,
and metatomic. Provider-specific arguments, such as FAIR-Chem's task_name,
belong in params. Device-specific compiled artifacts that cannot execute in
CPU float64 are explicitly rejected; they cannot provide this reference.
Frameworks with incompatible dependencies must be measured in separate
environments, for example ``--suite ml --providers mace-batched``. Such a
report can complete its requested subset without claiming the whole T19
matrix is complete. Review all framework and workflow reports together for
final acceptance; this driver does not merge evidence from different runs.
Instead of geometry, supply ``structures: {"64": "n64.npz", ...}`` with
float64 ``positions[N,3]`` in bohr, string ``symbols[N]``, optional float64
``cell[3,3]`` in bohr, and boolean ``pbc[3]``. Frames receive reproducible
Gaussian displacements (jitter_bohr, default 0.02) from those base positions.
Generated lattices are performance probes, not validated molecular states.

Provider timings include neighbour construction and either the complete NumPy
boundary or an already resident tensor. Workflow timings call run_materials
and include construction, propagation, checkpoint/trajectory I/O and analysis.
CPU and CUDA RNG streams need not match: workflow errors compare each final
snapshot with independent NumPy float64 formulas, never stochastic paths
against each other. Diffusion additionally checks exact free-particle centroid
transport. None of these tests establishes physical convergence.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time

import numpy as np


PROVIDERS = ("mace-batched", "nequip-batched", "fairchem-batched",
             "deepmd-batched", "metatomic")
SCHEMA = "pyrpmd.t19-cuda.v1"


@dataclass(frozen=True)
class Settings:
    beads: tuple[int, ...] = (8, 32, 64)
    atoms: tuple[int, ...] = (64, 512, 4096)
    providers: tuple[str, ...] = PROVIDERS
    replicas: int = 1
    warmup: int = 2
    repeats: int = 5
    threads: int = 1
    seed: int = 19064
    device: str = "cuda:0"
    suite: str = "all"
    rtol: float = 1e-9
    atol: float = 1e-10
    production_steps: int = 64
    equilibration_steps: int = 16
    sample_interval: int = 8
    normal_mode_layout: str = "rfft"

    def __post_init__(self):
        if (not self.providers or any(value not in PROVIDERS for value in self.providers)
                or len(set(self.providers)) != len(self.providers)):
            raise ValueError("providers must contain unique supported provider names")
        for name in ("beads", "atoms"):
            values = getattr(self, name)
            if (not values or any(type(v) is not int or v < 1 for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain unique positive integers")
        for name, minimum in (("replicas", 1), ("warmup", 1), ("repeats", 3),
                              ("threads", 1), ("seed", 0), ("production_steps", 1),
                              ("equilibration_steps", 0), ("sample_interval", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.production_steps % self.sample_interval:
            raise ValueError("production_steps must be a multiple of sample_interval")
        if self.production_steps < 3 * self.sample_interval:
            raise ValueError("production_steps must contain at least three sample intervals")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must explicitly be cpu, cuda, or cuda:<index>")
        if self.suite not in ("all", "ml", "workflows"):
            raise ValueError("suite must be all, ml, or workflows")
        if self.normal_mode_layout not in ("rfft", "rpmdrate"):
            raise ValueError("normal_mode_layout must be rfft or rpmdrate")
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    if path is None:
        return []
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if (not isinstance(document, dict) or set(document) != {"schema", "models"}
            or document["schema"] != "pyrpmd.t19-models.v1"
            or not isinstance(document["models"], list)):
        raise ValueError("manifest must contain schema='pyrpmd.t19-models.v1' and a models list")
    models, names = [], set()
    allowed = {"name", "provider", "model", "sha256", "model_kind", "dtype", "params",
               "geometry", "structures", "jitter_bohr"}
    for original in document["models"]:
        if not isinstance(original, dict) or set(original) - allowed:
            raise ValueError("model entry has unsupported fields")
        spec = dict(original)
        name = spec.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("model names must be nonempty and unique")
        names.add(name)
        if spec.get("provider") not in PROVIDERS or spec.get("dtype") != "float64":
            raise ValueError("each model needs a supported provider and explicit dtype='float64'")
        if spec.get("model_kind") not in ("trained", "untrained"):
            raise ValueError("model_kind must explicitly be trained or untrained")
        if (not isinstance(spec.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
            raise ValueError("each model needs its expected SHA256")
        if not isinstance(spec.get("model"), str) or "://" in spec["model"]:
            raise ValueError("model must be a local file path")
        spec["model"] = str((path.parent / spec["model"]).resolve())
        # Missing files are case failures, not a reason to lose all other cases.
        params = spec.setdefault("params", {})
        if not isinstance(params, dict) or set(params) & {"device", "dtype", "symbols", "model", "cell", "pbc"}:
            raise ValueError("params must not override device, dtype, model, or system binding")
        if ("geometry" in spec) == ("structures" in spec):
            raise ValueError("supply exactly one of geometry or structures")
        if "structures" in spec:
            if not isinstance(spec["structures"], dict) or not spec["structures"]:
                raise ValueError("structures must map atom counts to local NPZ files")
            if any(not isinstance(k, str) or not k.isdecimal() or int(k) < 1
                   or not isinstance(v, str) or "://" in v for k, v in spec["structures"].items()):
                raise ValueError("structures must map positive atom counts to local NPZ files")
            spec["structures"] = {k: str((path.parent / v).resolve())
                                  for k, v in spec["structures"].items()}
        models.append(spec)
    return models


def geometry(spec, atoms, frames, seed):
    provenance = {}
    if "geometry" in spec:
        definition = spec["geometry"]
        if not isinstance(definition, dict) or set(definition) - {"species", "spacing_bohr", "jitter_bohr", "pbc"}:
            raise ValueError("geometry has unsupported fields")
        species = definition.get("species")
        if not isinstance(species, list) or not species or any(not isinstance(s, str) or not s for s in species):
            raise ValueError("geometry.species must be a nonempty list of element symbols")
        spacing = definition.get("spacing_bohr")
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("geometry.spacing_bohr must be positive and finite")
        side = math.ceil(atoms ** (1 / 3))
        while side**3 < atoms:
            side += 1
        lattice = np.stack(np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), axis=-1)
        base = np.asarray(lattice.reshape(-1, 3)[:atoms] * spacing, dtype=np.float64)
        symbols = [species[i % len(species)] for i in range(atoms)]
        flags = np.asarray(definition.get("pbc", [True] * 3))
        cell = np.eye(3, dtype=np.float64) * side * spacing
        jitter = definition.get("jitter_bohr", 0.02)
        provenance["kind"] = "synthetic_lattice_performance_probe"
    else:
        source = Path(spec["structures"][str(atoms)])
        with np.load(source, allow_pickle=False) as values:
            if set(values.files) - {"positions", "symbols", "cell", "pbc"}:
                raise ValueError("structure NPZ has unsupported fields")
            base, raw_symbols = values["positions"], values["symbols"]
            if base.dtype != np.float64 or base.shape != (atoms, 3) or not np.isfinite(base).all():
                raise ValueError("structure positions must be finite float64[N,3] in bohr")
            if raw_symbols.shape != (atoms,) or raw_symbols.dtype.kind not in "US":
                raise ValueError("structure symbols must be a string array [N]")
            symbols = raw_symbols.astype(str).tolist()
            flags = values["pbc"] if "pbc" in values else np.zeros(3, dtype=bool)
            cell = values["cell"] if "cell" in values else None
        jitter = spec.get("jitter_bohr", 0.02)
        provenance.update(kind="user_structure", path=str(source), sha256=sha256(source))
    if flags.shape != (3,) or flags.dtype != np.bool_:
        raise ValueError("pbc must contain three booleans")
    if cell is not None and (cell.dtype != np.float64 or cell.shape != (3, 3)
                             or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0):
        raise ValueError("cell must be finite right-handed float64[3,3] in bohr")
    if cell is None and flags.any():
        raise ValueError("periodic coordinates require a cell")
    if isinstance(jitter, bool) or not isinstance(jitter, (int, float)) or not math.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter_bohr must be finite and nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, atoms, frames]))
    q = base[None] + rng.normal(0.0, jitter, size=(frames, atoms, 3))
    provenance.update(coordinates_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                      jitter_bohr=jitter, symbols=symbols, pbc=flags.tolist(),
                      cell_bohr=None if cell is None else cell.tolist())
    return q, symbols, cell, flags, provenance


def error_metrics(actual, reference, *, rtol, atol):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.size == 0 or actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("correctness arrays must have equal nonempty shapes and finite values")
    with np.errstate(over="raise", invalid="raise"):
        difference = actual - reference
    scale = float(np.max(np.abs(difference), initial=0.0))
    reference_scale = float(np.max(np.abs(reference), initial=0.0))
    normalized = difference / scale if scale else difference
    rms = scale * float(np.sqrt(np.mean(normalized * normalized)))
    relative = 0.0 if scale == 0 else None
    if reference_scale and scale:
        relative = (scale / reference_scale) * float(np.linalg.norm(normalized)) / float(np.linalg.norm(reference / reference_scale))
        if not math.isfinite(relative):
            relative = None
    return {"max_absolute": scale, "rms": rms,
            "relative_l2": relative,
            "passed": bool(np.all(np.abs(difference) <= atol + rtol * np.abs(reference)))}


def timed_calls(function, synchronize, *, warmup, repeats):
    for _ in range(warmup):
        synchronize()
        function()
        synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        function()
        synchronize()
        samples.append(time.perf_counter() - start)
    return {"seconds": samples, "median_ms": 1000.0 * statistics.median(samples) if samples else None}


def _command(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment(device):
    cpu_model = platform.processor()
    if not cpu_model or cpu_model == platform.machine():
        cpu_model = None
    try:
        cpu_info = Path("/proc/cpuinfo").read_text()
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.splitlines()
                          if line.startswith("model name")), cpu_model)
    except OSError:
        pass
    record = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
              "affinity_cpus": None, "requested_device": device,
              "thread_environment": {k: os.environ.get(k) for k in
                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "versions": {}, "gpu": None, "cuda_available": False,
              "cuda_build_version": None, "cuda_driver_version": None,
              "nvidia_smi": None, "blocking_reasons": []}
    try:
        record["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    for package in ("numpy", "scipy", "torch", "mace-torch", "nequip", "fairchem-core",
                    "deepmd-kit", "metatomic-torch", "vesin-torch", "nvalchemiops"):
        try:
            record["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record["versions"][package] = None
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        record["blocking_reasons"].append(f"Torch is unavailable: {type(exc).__name__}: {exc}")
        return record
    record["versions"]["torch"] = str(torch.__version__)
    record["cuda_build_version"] = torch.version.cuda
    record["torch_threads"] = torch.get_num_threads()
    record["torch_interop_threads"] = torch.get_num_interop_threads()
    record["cuda_available"] = torch.cuda.is_available()
    if device != "cpu":
        if not record["cuda_available"]:
            record["blocking_reasons"].append("CUDA was requested but no CUDA device is available; no CPU fallback")
        else:
            try:
                resolved = torch.empty(0, device=device).device
                properties = torch.cuda.get_device_properties(resolved)
                record["gpu"] = {"device": str(resolved), "model": properties.name,
                                 "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
                                 "total_memory_bytes": properties.total_memory,
                                 "compute_capability": [properties.major, properties.minor]}
                record["nvidia_smi"] = _command([
                    "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                    "--format=csv,noheader,nounits"])
                if record["nvidia_smi"]:
                    versions = {row[3].strip() for row in csv.reader(record["nvidia_smi"].splitlines())
                                if len(row) == 5}
                    if len(versions) == 1:
                        record["cuda_driver_version"] = versions.pop()
            except (RuntimeError, AssertionError) as exc:
                record["blocking_reasons"].append(f"Requested device failed: {type(exc).__name__}: {exc}")
    return record


def source_snapshot():
    # The harness may live in one checkout while PYTHONPATH selects another.
    # Attribution follows the modules Python will actually execute, separately
    # from the benchmark source. Never label an integration tree as main.
    package = importlib.import_module("pyrpmd")
    package_path = Path(package.__file__).resolve()
    actual_root = _command(["git", "rev-parse", "--show-toplevel"], package_path.parent)
    script = Path(__file__).resolve()
    script_root = _command(["git", "rev-parse", "--show-toplevel"], script.parent)
    names = ["pyrpmd.potentials.batched_ml", "pyrpmd.torch_backend", "pyrpmd.torch_normal_modes",
             "pyrpmd.workflows.materials", "pyrpmd.workflows.torch_materials",
             "pyrpmd.potentials.torch_workflow"]
    names.extend("pyrpmd.potentials._batched_" + name for name in
                 ("ml_base", "mace", "nequip", "fairchem", "deepmd", "metatomic"))
    modules = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin is not None and Path(spec.origin).is_file():
            origin = Path(spec.origin).resolve()
            modules[name] = {"path": str(origin), "sha256": sha256(origin)}
    def checkout(path):
        return {"root": path, "git_head": None if path is None else _command(["git", "rev-parse", "HEAD"], path),
                "tracked_worktree_changes": None if path is None else
                _command(["git", "status", "--porcelain", "--untracked-files=no"], path)}
    return {**checkout(actual_root), "package_path": str(package_path), "modules": modules,
            "torch_workflow_available": "pyrpmd.workflows.torch_materials" in modules,
            "benchmark": {**checkout(script_root), "path": str(script), "sha256": sha256(script)}}


def _numpy_result(result, *, device=None):
    fields = {}
    for name in ("energy", "forces", "virial"):
        value = getattr(result, name, None)
        if value is None:
            if name != "virial":
                raise ValueError(f"missing {name}")
            continue
        if device is not None:
            import torch
            if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != torch.float64:
                raise ValueError(f"{name} must remain float64 on {device}")
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite float64")
        fields[name] = value
    return fields


def _compare(actual, reference, settings):
    if set(actual) != set(reference):
        raise ValueError("CPU and target must return the same energy/force/virial fields")
    return {name: error_metrics(actual[name], reference[name], rtol=settings.rtol, atol=settings.atol)
            for name in reference}


def _passed(value):
    if isinstance(value, dict):
        if not value:
            return False
        if "passed" in value:
            return value["passed"] is True
        return all(_passed(v) for v in value.values())
    return True


def stream_probe(provider, q, cpu, settings):
    """Exercise caller-stream graph construction with changing data and sizes."""
    import torch
    stream = torch.cuda.Stream(device=provider.device)
    checks = {}
    for attempt in range(3):
        # A rigid translation would leave most interatomic models unchanged
        # and could accidentally hide use of stale positions. Perturb every
        # frame/atom/component differently so the reference forces change.
        displacement = (attempt + 1) * .003 * np.sin(np.arange(q.size).reshape(q.shape) + 1.)
        shifted = q + displacement
        reference = _numpy_result(cpu.evaluate(shifted))
        # Build inputs and consume graph results on the same nondefault stream.
        # Synchronizing only after evaluation cannot repair an internal race.
        with torch.cuda.stream(stream):
            coordinates = torch.tensor(q, device=provider.device, dtype=torch.float64)
            coordinates.add_(torch.tensor(displacement, device=provider.device, dtype=torch.float64))
            provider.evaluate_tensor(coordinates[:1])  # Invalidate frame-count caches.
            result = provider.evaluate_tensor(coordinates)
        stream.synchronize()
        checks[str(attempt)] = _compare(_numpy_result(result, device=provider.device), reference, settings)
    return checks


def benchmark_provider(spec, atoms, beads, settings):
    import torch
    from pyrpmd.potentials.batched_ml import make_batched_ml_potential

    model_hash = sha256(spec["model"])
    if model_hash != spec["sha256"]:
        raise ValueError(f"model SHA256 mismatch: expected {spec['sha256']}, found {model_hash}")
    q, symbols, cell, pbc, provenance = geometry(spec, atoms, beads * settings.replicas, settings.seed)
    common = dict(symbols=symbols, model=spec["model"], dtype="float64", cell=cell, pbc=pbc, **spec["params"])
    target_device = torch.empty(0, device=settings.device).device
    synchronize = (lambda: torch.cuda.synchronize(target_device)) if target_device.type == "cuda" else (lambda: None)
    with ExitStack() as stack:
        cpu = stack.enter_context(make_batched_ml_potential(spec["provider"], device="cpu", **common))
        target = stack.enter_context(make_batched_ml_potential(spec["provider"], device=str(target_device), **common))
        tensor = torch.tensor(q, device=target_device, dtype=torch.float64)
        reference = _numpy_result(cpu.evaluate(q))
        correctness = {
            "numpy": _compare(_numpy_result(target.evaluate(q)), reference, settings),
            "tensor": _compare(_numpy_result(target.evaluate_tensor(tensor), device=target_device), reference, settings),
            "cpu_single_frame": _compare(_numpy_result(cpu.evaluate(q[:1])),
                                         {name: value[:1] for name, value in reference.items()}, settings),
            "permuted_tensor": _compare(_numpy_result(target.evaluate_tensor(tensor.flip(0)), device=target_device),
                                        {name: value[::-1] for name, value in reference.items()}, settings),
        }
        if target_device.type == "cuda":
            correctness["nondefault_stream"] = stream_probe(target, q, cpu, settings)
            torch.cuda.reset_peak_memory_stats(target_device)
        calls = {"cpu_numpy": lambda: cpu.evaluate(q),
                 "target_numpy": lambda: target.evaluate(q),
                 "target_tensor": lambda: target.evaluate_tensor(tensor)}
        # Rotate path order per repetition; imports/model construction and
        # correctness transfers are outside the force-evaluation measurements.
        timings = {name: {"seconds": []} for name in calls}
        for name, call in calls.items():
            timed_calls(call, synchronize, warmup=settings.warmup, repeats=0)
        names = tuple(calls)
        for repeat in range(settings.repeats):
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in order:
                synchronize()
                start = time.perf_counter()
                calls[name]()
                if name != "cpu_numpy":
                    synchronize()
                timings[name]["seconds"].append(time.perf_counter() - start)
        for values in timings.values():
            values["median_ms"] = statistics.median(values["seconds"]) * 1000
        peak = torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else None
        reserved = torch.cuda.max_memory_reserved(target_device) if target_device.type == "cuda" else None
        metadata = {"cpu": dict(cpu.metadata), "target": dict(target.metadata)}
    if sha256(spec["model"]) != model_hash:
        raise ValueError("model file changed during measurement")
    correct = _passed(correctness)
    return {"status": "passed" if correct else "incorrect", "correctness": correctness,
            "timings": timings, "model_sha256": model_hash, "model_kind": spec["model_kind"],
            "geometry": provenance, "provider_metadata": metadata, "peak_cuda_allocated_bytes": peak,
            "peak_cuda_reserved_bytes": reserved,
            "cpu_over_target_numpy": timings["cpu_numpy"]["median_ms"] / timings["target_numpy"]["median_ms"] if correct else None,
            "cpu_over_target_tensor": timings["cpu_numpy"]["median_ms"] / timings["target_tensor"]["median_ms"] if correct else None,
            "timing_scope": "one physical-PES batch evaluation; static coordinates; tensor inputs already resident"}


def workflow_configuration(task, beads, settings, device):
    config = {"task": task, "beta": 4.0, "beads": beads, "dt_au": 0.125, "seed": settings.seed,
              "potential": {"provider": "harmonic", "params": {"k": 2.0}} if task == "pimd" else {"provider": "free"},
              "sampling": {"engine": "torch", "device": device, "integrator": "bcocb",
                           "thermostat": "pile-l", "gamma_centroid": 1.0,
                           "normal_mode_layout": settings.normal_mode_layout,
                           "equilibration_steps": settings.equilibration_steps,
                           "production_steps": settings.production_steps,
                           "sample_interval": settings.sample_interval, "replicas": settings.replicas},
              "dynamics": {"integrator": "cayley-bcb", "block_steps": 256},
              "parallel": {"backend": "serial", "threads_per_rank": settings.threads},
              "output": {"checkpoint_steps": 32, "chunk_frames": 128}}
    if task == "diffusion":
        config["diffusion"] = {"bootstrap": 0}
    return config


def workflow_reference(result, *, task, beta, masses, settings):
    """Independent NumPy formulas evaluated on each saved float64 snapshot."""
    if not isinstance(result.get("streams"), list) or not result["streams"]:
        raise ValueError("workflow correctness requires nonempty saved streams")
    references = {}
    for index, stream in enumerate(result["streams"]):
        q, p = np.asarray(stream["state"].q), np.asarray(stream["state"].p)
        if q.dtype != np.float64 or p.dtype != np.float64:
            raise ValueError("workflow snapshot must retain float64")
        beads, atoms, _ = q.shape
        offsets = q - q.mean(axis=0, keepdims=True)
        delta = q - np.roll(q, -1, axis=0)
        spring = .5 * (beads / beta)**2 * np.sum(masses[None, :, None] * delta**2)
        energy = np.sum(q**2, axis=(1, 2)) if task == "pimd" else np.zeros(beads)
        forces = -2 * q if task == "pimd" else np.zeros_like(q)
        potential = np.mean(energy)
        primitive = 1.5 * atoms * beads / beta - spring / beads
        virial = 1.5 * atoms / beta - np.sum(offsets * forces) / (2 * beads)
        hamiltonian = np.sum(energy) + spring + np.sum(p**2 / (2 * masses[None, :, None]))
        expected = {"observables": np.array([potential, primitive, virial, potential + virial, hamiltonian]),
                    "positions": q.mean(axis=0), "velocities": (p / masses[None, :, None]).mean(axis=0),
                    "radii_gyration_squared": np.mean(np.sum(offsets**2, axis=2), axis=0)}
        actual = {name: stream[name][-1] for name in expected}
        checks = _compare(actual, expected, settings)
        if task == "diffusion":
            positions, velocities, times = stream["positions"], stream["velocities"], stream["times"]
            ballistic = positions[0] + (times - times[0])[:, None, None] * velocities[0]
            checks["ballistic_positions"] = error_metrics(positions, ballistic, rtol=settings.rtol, atol=settings.atol)
            checks["constant_centroid_velocity"] = error_metrics(
                velocities, np.broadcast_to(velocities[0], velocities.shape), rtol=settings.rtol, atol=settings.atol)
        references[str(index)] = checks
    return references


def validate_workflow_completion(result, config, atoms):
    """Reject partial, empty, or differently sized work before reporting cost/step."""
    if not isinstance(result, dict) or not isinstance(result.get("summary"), dict):
        raise ValueError("workflow must return its summary and complete materialized streams")
    sampling = config["sampling"]
    replicas, beads = sampling["replicas"], config["beads"]
    neq, nprod, interval = (sampling[key] for key in
                            ("equilibration_steps", "production_steps", "sample_interval"))
    total, frames = neq + nprod, nprod // interval + 1
    expected_steps = neq + np.arange(frames, dtype=np.int64) * interval
    expected_times = np.arange(frames, dtype=np.float64) * interval * config["dt_au"]
    streams = result.get("streams")
    if not isinstance(streams, list) or len(streams) != replicas:
        raise ValueError("workflow returned a different number of replicas than requested")
    for replica, stream in enumerate(streams):
        if (not isinstance(stream, dict) or stream.get("replica") != replica
                or stream.get("complete") is not True or stream.get("elapsed") != total
                or stream.get("frames") != frames):
            raise ValueError("workflow did not complete the requested replica and sample schedule")
        state = stream.get("state")
        if (getattr(state, "step", None) != total
                or getattr(state, "time", None) != total * config["dt_au"]):
            raise ValueError("workflow final step/time differs from the measured step count")
        for field in ("q", "p"):
            values = np.asarray(getattr(state, field, None))
            if values.dtype != np.float64 or values.shape != (beads, atoms, 3) or not np.isfinite(values).all():
                raise ValueError("workflow final state must have the requested float64[P,N,3] shape")
        steps = np.asarray(stream.get("sample_steps"))
        times = np.asarray(stream.get("times"))
        if (steps.dtype.kind not in "iu" or not np.array_equal(steps, expected_steps)
                or times.dtype != np.float64 or not np.array_equal(times, expected_times)):
            raise ValueError("workflow saved sample clock differs from the requested production schedule")
        for field, shape in (("observables", (frames, 5)), ("positions", (frames, atoms, 3)),
                             ("velocities", (frames, atoms, 3)), ("radii_gyration_squared", (frames, atoms))):
            values = np.asarray(stream.get(field))
            if values.dtype != np.float64 or values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"workflow {field} must contain every requested finite float64 frame")


def benchmark_workflow(task, atoms, beads, settings, scratch):
    import torch
    from pyrpmd.system import System
    from pyrpmd.workflows.materials import run_materials

    system = System(tuple(["H"] * atoms), np.full(atoms, 2.0), np.zeros((atoms, 3)))
    target = torch.empty(0, device=settings.device).device
    synchronize = (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else (lambda: None)
    devices = {"cpu": "cpu", "target": str(target)}
    timings, checks = {name: {"seconds": []} for name in devices}, {}
    runtime = {}
    peak = reserved = 0 if target.type == "cuda" else None
    with tempfile.TemporaryDirectory(prefix="t19-workflow-", dir=scratch) as temporary:
        directory = Path(temporary)
        for repeat in range(-settings.warmup, settings.repeats):
            order = ("cpu", "target") if repeat % 2 == 0 else ("target", "cpu")
            for name in order:
                config = workflow_configuration(task, beads, settings, devices[name])
                synchronize()
                if target.type == "cuda" and name == "target":
                    torch.cuda.reset_peak_memory_stats(target)
                start = time.perf_counter()
                result = run_materials(config, system, directory / f"{name}-{repeat}", return_data=True)
                if name == "target":
                    synchronize()
                elapsed = time.perf_counter() - start
                if target.type == "cuda" and name == "target":
                    peak = max(peak, torch.cuda.max_memory_allocated(target))
                    reserved = max(reserved, torch.cuda.max_memory_reserved(target))
                if repeat >= 0:
                    timings[name]["seconds"].append(elapsed)
                validate_workflow_completion(result, config, atoms)
                check = workflow_reference(result, task=task, beta=config["beta"], masses=system.masses, settings=settings)
                checks[f"{name}/{repeat}"] = check
                runtime[name] = result["summary"].get("tensor_runtime")
                if (result["summary"].get("engine") != "torch" or not isinstance(runtime[name], dict)
                        or runtime[name].get("device") != devices[name]
                        or runtime[name].get("dtype") != "float64"):
                    raise ValueError("workflow must execute the requested Torch device in float64; no fallback")
                del result
    total_steps = settings.equilibration_steps + settings.production_steps
    for values in timings.values():
        values["median_ms"] = 1000 * statistics.median(values["seconds"])
        values["median_ms_per_step"] = values["median_ms"] / total_steps
    correct = _passed(checks)
    return {"status": "passed" if correct else "incorrect", "correctness": checks,
            "correctness_scope": "independent NumPy float64 same-snapshot observables; CPU/CUDA stochastic paths are not compared",
            "potential": "harmonic k=2 hartree/bohr^2" if task == "pimd" else "free",
            "model_kind": "analytic_workflow_probe",
            "workflow_config": workflow_configuration(task, beads, settings, str(target)), "mass_electron": 2.0,
            "timings": timings, "runtime": runtime, 
            "peak_cuda_allocated_bytes": peak, "steps_per_replica": total_steps,
            "peak_cuda_reserved_bytes": reserved,
            "cpu_over_target": timings["cpu"]["median_ms"] / timings["target"]["median_ms"] if correct else None,
            "timing_scope": "public run_materials including construction, propagation, I/O, materialization and analysis"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(settings=Settings(), *, manifest=None, output=None, scratch=None, progress=None):
    models = load_manifest(manifest)
    snapshot, hardware = source_snapshot(), environment(settings.device)
    report = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": snapshot, "environment": hardware, "settings": asdict(settings),
              "manifest_sha256": None if manifest is None else sha256(manifest), "models": models,
              "effective_torch_threads": settings.threads,
               "execution_complete": False,
              "cuda_matrix_complete": False, "cases": []}
    if scratch is not None:
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
    def append(row):
        report["cases"].append(row)
        if output is not None:
            atomic_json(output, report)
        if progress is not None:
            progress(row)
    with ExitStack() as stack:
        if not hardware["blocking_reasons"]:
            import torch
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(settings.threads)
            stack.callback(torch.set_num_threads, previous_threads)
            try:
                from threadpoolctl import threadpool_info, threadpool_limits
            except ImportError:
                report["threadpools"] = {"controlled": False, "reason": "threadpoolctl unavailable; use explicit BLAS/OpenMP environment limits"}
            else:
                stack.enter_context(threadpool_limits(settings.threads))
                report["threadpools"] = {"controlled": True, "pools": threadpool_info()}
        entries = []
        if settings.suite != "workflows":
            for provider in settings.providers:
                matches = [spec for spec in models if spec["provider"] == provider]
                entries.extend(("ml", spec["name"], provider, spec) for spec in matches)
                if not matches:
                    entries.append(("ml", None, provider, None))
        if settings.suite != "ml":
            entries.extend(("workflow", task, None, None) for task in ("pimd", "diffusion"))
        for suite, name, provider, spec in entries:
            for atoms in settings.atoms:
                for beads in settings.beads:
                    row = {"suite": suite, "name": name, "provider": provider,
                           "atoms": atoms, "beads": beads, "replicas": settings.replicas,
                           "target_device": settings.device, "dtype": "float64"}
                    blockers = list(hardware["blocking_reasons"])
                    if suite == "ml" and spec is None:
                        blockers.append("No local float64 model was supplied for this provider")
                    if suite == "workflow" and not snapshot["torch_workflow_available"]:
                        blockers.append("PR #84 Torch materials workflow is not present in this source snapshot")
                    if blockers:
                        row.update(status="blocked", reasons=blockers)
                    else:
                        try:
                            row.update(benchmark_provider(spec, atoms, beads, settings) if suite == "ml" else
                                       benchmark_workflow(name, atoms, beads, settings, scratch))
                        except Exception as exc:
                            row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                            cause = exc
                            chain = []
                            while cause is not None:
                                chain.append(type(cause).__name__)
                                cause = cause.__cause__
                            row["exception_chain"] = chain
                            row["out_of_memory"] = any(name in chain for name in ("OutOfMemoryError", "MemoryError")) or "out of memory" in str(exc).lower()
                    append(row)
    report["execution_complete"] = True
    report["requested_cases_passed"] = bool(report["cases"]) and all(row["status"] == "passed" for row in report["cases"])
    report["cuda_matrix_complete"] = (report["requested_cases_passed"] and settings.device != "cpu"
                                      and settings.suite == "all" and set(settings.atoms) == {64, 512, 4096}
                                      and set(settings.beads) == {8, 32, 64} and bool(hardware.get("gpu"))
                                      and set(settings.providers) == set(PROVIDERS)
                                      and bool(hardware.get("cuda_driver_version"))
                                      and bool(hardware.get("cuda_build_version")))
    if output is not None:
        atomic_json(output, report)
    return report


def markdown(report):
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    hardware = report["environment"]
    lines = ["# T19 CUDA measurement record", "", f"Source: `{report['source']['git_head']}`.",
             f"Requested device: `{hardware['requested_device']}`; CUDA build version: `{hardware['cuda_build_version']}`.",
             f"GPU: `{None if hardware['gpu'] is None else hardware['gpu']['model']}`.",
             f"NVIDIA index / UUID / model / driver / MiB: `{hardware['nvidia_smi']}`.", "",
             "These are numerical/performance checks; physical convergence is false.", "",
             "| Scope/model | N | P | Status | CPU ms/step | Target NumPy or workflow ms/step | Target tensor ms/step |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in report["cases"]:
        timings = row.get("timings", {})
        ml = row["suite"] == "ml"
        cpu = timings.get("cpu_numpy" if ml else "cpu", {}).get("median_ms" if ml else "median_ms_per_step")
        target = timings.get("target_numpy" if ml else "target", {}).get("median_ms" if ml else "median_ms_per_step")
        tensor = timings.get("target_tensor", {}).get("median_ms")
        label = str(row["name"] or row["provider"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {row['atoms']} | {row['beads']} | {row['status']} | {number(cpu)} | {number(target)} | {number(tensor)} |")
    lines.extend(["", "## Correctness", "",
                  "Maximum absolute error against CPU float64, including the stream/permutation checks. Workflow observables use each device's own final snapshot.", "",
                  "| Scope/model | N | P | Energy/observables (hartree) | Force (hartree/bohr) | Virial (hartree) | Position (bohr) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in report["cases"]:
        fields = {}
        def collect(value, name=None):
            if isinstance(value, dict):
                if "max_absolute" in value:
                    fields[name] = max(fields.get(name, 0.), value["max_absolute"])
                else:
                    for key, item in value.items():
                        collect(item, key)
        collect(row.get("correctness", {}))
        label = str(row["name"] or row["provider"]).replace("|", "\\|").replace("\n", " ")
        energy = fields.get("energy", fields.get("observables"))
        positions = [fields[key] for key in ("positions", "ballistic_positions") if key in fields]
        lines.append(f"| {label} | {row['atoms']} | {row['beads']} | {number(energy)} | {number(fields.get('forces'))} | {number(fields.get('virial'))} | {number(max(positions, default=None))} |")
    lines.extend(["", "## Blockers and failures", ""])
    for row in report["cases"]:
        label = f"{row['name'] or row['provider']} N={row['atoms']} P={row['beads']}"
        if row["status"] != "passed":
            reason = "; ".join(row.get("reasons", [row.get("error", "No measurement")]))
            if row["status"] == "incorrect":
                reason = "CPU float64 numerical comparison exceeded the explicit tolerance; see every error in JSON"
            lines.append(f"- {label}: {reason}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", choices=("all", "ml", "workflows"), default="all")
    parser.add_argument("--providers", choices=PROVIDERS, nargs="+", default=list(PROVIDERS))
    parser.add_argument("--atoms", nargs="+", type=int, default=[64, 512, 4096])
    parser.add_argument("--beads", nargs="+", type=int, default=[8, 32, 64])
    for name, default in (("replicas", 1), ("warmup", 2), ("repeats", 5), ("threads", 1),
                          ("seed", 19064), ("production-steps", 64), ("equilibration-steps", 16), ("sample-interval", 8)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--normal-mode-layout", choices=("rfft", "rpmdrate"), default="rfft")
    args = vars(parser.parse_args(argv))
    manifest, output, markdown_path, scratch = (args.pop(name) for name in ("manifest", "output", "markdown", "scratch"))
    args["atoms"], args["beads"] = tuple(args["atoms"]), tuple(args["beads"])
    args["providers"] = tuple(args["providers"])
    try:
        settings = Settings(**args)
        report = run(settings, manifest=manifest, output=output, scratch=scratch,
                     progress=lambda row: print(f"{row['name'] or row['provider']} N={row['atoms']} P={row['beads']}: {row['status']}", flush=True))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown(report))
    print(json.dumps({"output": str(output), "cases": len(report["cases"]),
                      "requested_cases_passed": report["requested_cases_passed"],
                      "cuda_matrix_complete": report["cuda_matrix_complete"]}))
    return 0 if report["requested_cases_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
