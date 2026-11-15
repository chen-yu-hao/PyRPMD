"""Direct JAX GPU binding for fitted JaxPIP models.

The ONNX bridge is useful for a portable C ABI, but it necessarily copies the
coordinates through host memory.  This adapter is the device-resident route:
JaxPIP owns the compiled force graph, while the Torch workflow owns the ring
polymer state.  Torch and JAX exchange tensors with DLPack on the selected
device, so no coordinate or force array is materialized on the host inside a
force evaluation.

The adapter intentionally accepts only the checked JaxPIP fit artifacts used by
``pyrpmd.pesgen``.  A model directory may be supplied in place of ``basis``
and ``model``; its checked fit manifest then supplies the two files and the
model settings.
"""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import json

import numpy as np

from ..constants import BOHR_ANGSTROM, EV_TO_AU
from .base import BackendUnavailableError, PESResult, Potential, coordinates


def _units(position_unit, energy_unit):
    if position_unit == "bohr":
        length = 1.0
    elif position_unit == "angstrom":
        length = BOHR_ANGSTROM  # model length units per bohr
    else:
        raise ValueError("JaxPIP position_unit must be bohr or angstrom")
    if energy_unit == "hartree":
        energy = 1.0
    elif energy_unit == "eV":
        energy = EV_TO_AU  # hartree per model energy unit
    else:
        raise ValueError("JaxPIP energy_unit must be hartree or eV")
    return length, energy


def _load_runtime(*, basis, checkpoint, kind, alpha, device):
    try:
        import jax
        import equinox as eqx
        import jax.numpy as jnp
        from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork
    except ImportError as error:
        raise BackendUnavailableError(
            "the JaxPIP GPU route requires jax, equinox and jaxpip; "
            "install the pinned environment from tools/requirements-jaxpip-gpu.txt"
        ) from error
    jax.config.update("jax_enable_x64", True)
    if kind not in ("linear", "nn"):
        raise ValueError("JaxPIP kind must be linear or nn")
    devices = list(jax.devices())
    requested = str(device)
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda:"):
        if not any(d.platform == "gpu" and d.id == int(requested[5:]) for d in devices):
            raise BackendUnavailableError(f"requested JAX device {device!r} is unavailable")
        jax_device = next(d for d in devices if d.platform == "gpu" and d.id == int(requested[5:]))
    elif requested == "cpu":
        jax_device = next((d for d in devices if d.platform == "cpu"), None)
        if jax_device is None:
            raise BackendUnavailableError("JAX CPU device is unavailable")
    else:
        raise ValueError("JaxPIP device must be cpu, cuda or cuda:<index>")
    cls = PolynomialLinearModel if kind == "linear" else PolynomialNeuralNetwork
    model = cls.from_file(basis_file=str(basis), model_file=str(checkpoint))
    if jnp.dtype(model.dtype) != jnp.float64:
        raise ValueError("JaxPIP production requires float64 model parameters")

    @eqx.filter_jit
    def batch(q):
        return jax.vmap(model.get_energy_and_forces)(q)

    return jax, jnp, model, batch, jax_device


class _JaxPIPTorchModule:
    """Lazily imported ``torch.nn.Module`` around a compiled JaxPIP graph."""

    def __new__(cls, runtime, *, length, energy, max_frames_per_call=None):
        try:
            import torch
        except ImportError as error:
            raise BackendUnavailableError("the JaxPIP GPU route requires PyTorch") from error
        jax, jnp, model, batch, jax_device = runtime
        torch_device = (f"cuda:{jax_device.id}" if jax_device.platform == "gpu" else "cpu")

        class Module(torch.nn.Module):
            metadata = {"batch_independent": True, "trajectory_batch_safe": True}

            def __init__(self):
                super().__init__()
                self.jax_device = jax_device
                self.length = float(length)
                self.energy = float(energy)
                self.max_frames_per_call = max_frames_per_call
                self.metadata = {**self.metadata, "execution_device": str(jax_device),
                                 "dtype": "float64", "energy_dtype": "float64",
                                 "provider": "jaxpip-dlpack"}
                self.metadata["execution_device"] = torch_device

            def forward(self, q):
                if q.dtype != torch.float64 or q.device.type not in ("cpu", "cuda"):
                    raise ValueError("JaxPIP bridge expects a float64 CPU/CUDA tensor")
                if q.device.type == "cuda" and self.jax_device.platform != "gpu":
                    raise ValueError("Torch/JAX devices differ; no implicit device transfer")
                if q.device.type == "cpu" and self.jax_device.platform != "cpu":
                    raise ValueError("Torch/JAX devices differ; no implicit device transfer")
                q = q.detach().contiguous()
                # Both conversions are DLPack ownership transfers.  The JAX
                # executable is synchronous from Torch's perspective because
                # the returned tensors are consumed immediately by TorchPES.
                if self.max_frames_per_call is None or len(q) <= self.max_frames_per_call:
                    energy_jax, force_jax = batch(jax.dlpack.from_dlpack(q))
                    return (torch.from_dlpack(energy_jax),
                            torch.from_dlpack(force_jax))
                energies, forces = [], []
                for start in range(0, len(q), self.max_frames_per_call):
                    energy_jax, force_jax = batch(
                        jax.dlpack.from_dlpack(q[start:start + self.max_frames_per_call]))
                    energies.append(torch.from_dlpack(energy_jax))
                    forces.append(torch.from_dlpack(force_jax))
                return torch.cat(energies, dim=0), torch.cat(forces, dim=0)

        return Module()


class JaxPIPPotential(Potential):
    """Checked JaxPIP model with a NumPy contract and a Torch GPU bridge."""

    def __init__(self, basis, checkpoint, *, kind="linear", alpha=1.0,
                 position_unit="bohr", energy_unit="hartree", device="cuda",
                 atomic_numbers=None, charge=0, multiplicity=1,
                 max_frames_per_call=None, allow_unvalidated=False):
        self.basis = Path(basis).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not self.basis.is_file() or not self.checkpoint.is_file():
            raise FileNotFoundError("JaxPIP basis/checkpoint file is missing")
        self.kind, self.alpha = kind, float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("JaxPIP alpha must be positive and finite")
        self.position_unit, self.energy_unit = position_unit, energy_unit
        self.length, self.energy = _units(position_unit, energy_unit)
        self.device = "cuda:0" if device == "cuda" else str(device)
        if max_frames_per_call is not None:
            if (isinstance(max_frames_per_call, bool) or not isinstance(max_frames_per_call, int)
                    or max_frames_per_call < 1):
                raise ValueError("max_frames_per_call must be a positive integer")
        self.max_frames_per_call = max_frames_per_call
        if not isinstance(allow_unvalidated, (bool, np.bool_)):
            raise ValueError("allow_unvalidated must be boolean")
        self.allow_unvalidated = bool(allow_unvalidated)
        self.atomic_numbers = None if atomic_numbers is None else tuple(int(z) for z in atomic_numbers)
        self.charge, self.multiplicity = int(charge), int(multiplicity)
        runtime = _load_runtime(basis=self.basis, checkpoint=self.checkpoint,
                                kind=kind, alpha=self.alpha, device=self.device)
        self._runtime = runtime
        self._jax, self._jnp, self.model, self._batch, self.jax_device = runtime
        if (self.atomic_numbers is not None
                and self.model.descriptor.basis_info.num_atoms != len(self.atomic_numbers)):
            raise ValueError("JaxPIP basis atom count does not match atomic_numbers")
        self._torch_module = None
        self.closed = False
        self.metadata = {
            "provider": "jaxpip-dlpack", "batch_independent": True,
            "trajectory_batch_safe": True, "stateful": False,
            "execution_device": self.device, "dtype": "float64",
            "energy_dtype": "float64", "position_unit": position_unit,
            "energy_unit": energy_unit, "force_unit": f"{energy_unit}/{position_unit}",
            "atomic_numbers": None if self.atomic_numbers is None else list(self.atomic_numbers),
            "state": {"charge": self.charge, "multiplicity": self.multiplicity},
            "jax_version": importlib.metadata.version("jax"),
            "jaxpip_version": importlib.metadata.version("jaxpip"),
            "max_frames_per_call": self.max_frames_per_call,
            "allow_unvalidated": self.allow_unvalidated,
        }

    @classmethod
    def from_spec(cls, spec, system=None):
        spec = dict(spec)
        directory = spec.get("model_directory", spec.get("directory"))
        if directory is not None:
            directory = Path(directory).resolve()
            from ..pesgen.jaxpip_fit import load_fit
            manifest = load_fit(directory)
            if (not manifest.get("within_requested_test_bounds", False)
                    and not spec.get("allow_unvalidated", False)):
                raise ValueError(
                    "JaxPIP fit is outside its requested test bounds; set "
                    "potential.allow_unvalidated=true only for a documented pilot"
                )
            settings = manifest.get("settings", {})
            spec.setdefault("basis", str(directory / manifest["basis"]))
            spec.setdefault("checkpoint", str(directory / manifest.get("checkpoint", "model.eqx")))
            spec.setdefault("kind", settings.get("kind", "linear"))
            spec.setdefault("alpha", settings.get("alpha", 1.0))
            spec.setdefault("position_unit", settings.get("position_unit", "bohr"))
            spec.setdefault("energy_unit", settings.get("energy_unit", "hartree"))
            contract = manifest.get("contract", {})
            spec.setdefault("atomic_numbers", contract.get("atomic_numbers"))
            spec.setdefault("charge", contract.get("charge", 0))
            spec.setdefault("multiplicity", contract.get("multiplicity", 1))
        checkpoint = spec.get("checkpoint", spec.get("model"))
        if checkpoint is None or "basis" not in spec:
            raise ValueError("JaxPIP requires basis and checkpoint/model (or model_directory)")
        if system is not None:
            actual = tuple(int(z) for z in system.atomic_numbers)
            declared = spec.get("atomic_numbers", actual)
            if tuple(int(z) for z in declared) != actual:
                raise ValueError("JaxPIP atomic_numbers do not match System order")
            spec.setdefault("atomic_numbers", list(actual))
            spec.setdefault("charge", getattr(system, "charge", 0))
            spec.setdefault("multiplicity", getattr(system, "multiplicity", 1))
        return cls(basis=spec["basis"], checkpoint=checkpoint,
                   kind=spec.get("kind", "linear"), alpha=spec.get("alpha", 1.0),
                   position_unit=spec.get("position_unit", "bohr"),
                   energy_unit=spec.get("energy_unit", "hartree"),
                   device=spec.get("device", "cuda"),
                   atomic_numbers=spec.get("atomic_numbers"),
                   charge=spec.get("charge", 0), multiplicity=spec.get("multiplicity", 1),
                   max_frames_per_call=spec.get("max_frames_per_call"),
                   allow_unvalidated=spec.get("allow_unvalidated", False))

    def _check_open(self):
        if self.closed:
            raise RuntimeError("JaxPIP potential is closed")

    def evaluate(self, positions):
        self._check_open()
        q = coordinates(positions)
        q_model = self._jax.device_put(self._jnp.asarray(q) * self.length, self.jax_device)
        energy, forces = self._batch(q_model)
        energy, forces = self._jax.device_get((energy, forces))
        return PESResult(np.asarray(energy, dtype=np.float64) * self.energy,
                         np.asarray(forces, dtype=np.float64) * (self.energy * self.length))

    def as_torch_pes(self):
        self._check_open()
        if self._torch_module is None:
            self._torch_module = _JaxPIPTorchModule(
                self._runtime, length=self.length, energy=self.energy,
                max_frames_per_call=self.max_frames_per_call)
        from ..torch_backend import TorchPES
        # TorchPES uses bohr per model length unit.  The JAX module itself
        # receives model coordinates after this conversion.
        result = TorchPES(self._torch_module, device=self.device,
                          length_factor=1.0 / self.length,
                          energy_factor=self.energy)
        result.metadata.update(self.metadata)
        return result

    def warmup_torch(self, frames, atoms):
        """Compile the selected JAX batch shape before production sampling."""
        self._check_open()
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            raise ValueError("warmup frames must be a positive integer")
        if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
            raise ValueError("warmup atoms must be a positive integer")
        import torch
        adapter = self.as_torch_pes()
        q = torch.zeros((frames, atoms, 3), dtype=torch.float64, device=adapter.device)
        adapter.evaluate_tensor(q)
        if adapter.device.type == "cuda":
            torch.cuda.synchronize(adapter.device)

    def close(self):
        self.closed = True
        self._torch_module = None


__all__ = ["JaxPIPPotential"]
