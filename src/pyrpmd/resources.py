"""Read-only probes for the execution resources selected by an input."""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import subprocess

import numpy as np


def _mpi(plan):
    expected = plan.coordinator_ranks + plan.trajectory_groups * plan.bead_workers * plan.pes_ranks
    report = {"available": False, "required_ranks": expected,
              "hint": "Install the mpi extra and an MPI runtime, then use mpiexec -n REQUIRED_RANKS pyrpmd run INPUT."}
    try:
        # Importing mpi4py.MPI can initialize transports and abort the process.
        # Probe the library in a child interpreter with initialization disabled;
        # the calling interpreter must remain usable for a subsequent MPI run.
        # Launched processes provide their rank count through MPI's environment.
        if importlib.util.find_spec("mpi4py") is None or importlib.util.find_spec("mpi4py.MPI") is None:
            report["reason"] = "mpi4py is not importable"
            return report
        probe = subprocess.run(
            [sys.executable, "-c", "import mpi4py; mpi4py.rc.initialize=False; from mpi4py import MPI; print(MPI.Get_version())"],
            capture_output=True, text=True, timeout=10)
        if probe.returncode:
            report["reason"] = "mpi4py cannot load its MPI runtime: " + probe.stderr.strip()[-1000:]
            return report
        world = sys.modules.get("mpi4py.MPI")
        size = None
        if world is not None and world.Is_initialized() and not world.Is_finalized():
            size = world.COMM_WORLD.Get_size()
        if size is None:
            for variable in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
                if variable in os.environ:
                    size = int(os.environ[variable])
                    break
        report.update(launcher=shutil.which("mpiexec") or shutil.which("mpirun"), launched_ranks=size)
        report["available"] = size == expected
        report["reason"] = None if report["available"] else "MPI launcher context is absent or its rank count differs"
    except (ImportError, ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def _plumed(config):
    kernel = None
    report = {"available": False,
              "hint": "Provide the PLUMED Python wrapper and compatible libplumedKernel (plumed.kernel or PLUMED_KERNEL); see docs/plumed.md."}
    try:
        module = importlib.import_module("plumed")
        path = config.get("plumed", {}).get("kernel")
        kernel = module.Plumed() if path is None else module.Plumed(kernel=path)
        version = np.zeros(1, dtype=np.intc)
        kernel.cmd("getApiVersion", version)  # no init, input script, files or MD
        report.update(available=int(version[0]) >= 8, api_version=int(version[0]))
        if not report["available"]:
            report["reason"] = "PLUMED API >= 8 is required"
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    finally:
        if kernel is not None:
            kernel.finalize()
    return report


def _torch(devices):
    report = {"available": False, "devices": list(devices),
              "hint": "Install the gpu extra and select an available CPU/CUDA device with a compatible Torch runtime."}
    try:
        torch = importlib.import_module("torch")
        for value in devices:
            device = torch.device(value)
            if device.type == "cuda":
                if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
                    raise ValueError(f"Torch device {value!r} is unavailable")
            elif device.type != "cpu":
                raise ValueError(f"Torch float64 workflows require CPU or CUDA, got {value!r}")
        report["available"] = True
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def _native_gpu(devices):
    """Describe native GPU availability without importing a Python ORT wheel.

    The native ONNX provider is loaded by the C ABI library and may be built
    against a different ONNX Runtime installation than Python.  Resource
    probing therefore cannot validate its CUDA execution provider without
    constructing the user's native PES; that check remains explicit during
    provider initialization.
    """
    return {
        "available": True,
        "devices": list(devices),
        "deferred": True,
        "hint": (
            "The native provider will validate its CUDA/ONNX Runtime SDK "
            "when the PES context is created."
        ),
    }


def resource_report(config):
    """Check selected resources without constructing a potential or launching MPI."""
    from .parallel import execution_plan

    checks = {}
    parallel = config.get("parallel", {})
    if parallel.get("backend") == "mpi":
        checks["mpi"] = _mpi(execution_plan(parallel))
    if config.get("task") == "plumed":
        checks["plumed"] = _plumed(config)
    sampling, potential = config.get("sampling", {}), config.get("potential", {})
    provider = potential.get("provider", potential.get("type", ""))
    params = potential.get("params", {}) if isinstance(potential, dict) else {}
    native_gpu = provider in ("cabi", "native") and (
        parallel.get("mode") == "gpu"
        or str(potential.get("device", "")).startswith("cuda")
        or isinstance(params, dict) and str(params.get("device", "")).startswith("cuda")
    )
    torch_gpu = (
        sampling.get("engine") == "torch"
        or provider in ("torch", "mace", "nequip", "fairchem")
        or str(provider).endswith("-batched")
        or str(potential.get("device", "")).startswith("cuda")
        or parallel.get("mode") == "gpu"
    ) and not native_gpu
    if native_gpu or torch_gpu:
        devices = [sampling.get("device", potential.get("device", "cpu"))]
        if parallel.get("mode") == "gpu":
            devices = [str(value) if str(value).startswith("cuda") else f"cuda:{value}"
                       for value in parallel.get("devices", [0])]
        checks["native_gpu" if native_gpu else "torch"] = (
            _native_gpu(devices) if native_gpu else _torch(devices)
        )
    return {"available": all(item["available"] for item in checks.values()), "checks": checks}


def with_resources(result, config):
    resources = resource_report(config)
    return {**result, "resources": resources,
            "ready": bool(result.get("ready", True) and resources["available"])}
