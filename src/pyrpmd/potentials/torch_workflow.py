"""Explicit, float64 tensor bindings for the materials Torch workflow.

The bindings never call a NumPy evaluator. Analytic bindings own a snapshot of
their parameters; module/ML bindings borrow an already configured provider.
The caller owns that provider and must keep it open until propagation ends.
Constructing this module does not import Torch or initialize an accelerator.
"""
from __future__ import annotations

import numpy as np

from .base import BackendUnavailableError
from .builtin import (
    FreePotential, HarmonicPotential, PeriodicCosinePotential,
    _BUILTIN_EVALUATE,
)
from .optional import TorchPotential


_TORCH_EVALUATE = TorchPotential.evaluate
_BUILTINS = (FreePotential, HarmonicPotential, PeriodicCosinePotential)


def _independent(metadata, *, explicit):
    if not isinstance(metadata, dict):
        raise ValueError("tensor workflow provider metadata must be a mapping")
    capabilities = metadata.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise ValueError("tensor workflow provider capabilities must be a mapping")
    if ((explicit and metadata.get("batch_independent") is not True)
            or metadata.get("batch_independent") is False
            or metadata.get("trajectory_batch_safe") is False
            or any(metadata.get(key) for key in ("stateful", "coupled_frames", "ring_polymer_coupled"))
            or capabilities.get("stateful")):
        raise ValueError("tensor workflow requires an explicitly batch-independent, stateless provider")
    for name in ("dtype", "energy_dtype"):
        if name in metadata and metadata[name] != "float64":
            raise ValueError("tensor workflow requires float64; no precision conversion")


def _device(torch, value):
    try:
        requested = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("tensor workflow device must be cpu or cuda[:index]") from error
    if requested.type not in ("cpu", "cuda") or (
            requested.type == "cpu" and requested.index not in (None, 0)):
        raise ValueError("tensor workflow device must be cpu or cuda[:index]")
    try:
        # Resolve cuda to its current index and cpu:0 to cpu once at binding.
        return torch.empty(0, device=requested).device
    except (RuntimeError, AssertionError) as error:
        raise BackendUnavailableError(
            f"requested tensor workflow device {value!r} is unavailable; no device fallback"
        ) from error


def _snapshot(value, name, *, shape=None, nonnegative=False):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "iuf" or not np.isfinite(raw).all()
            or (shape is not None and raw.shape != shape)
            or (nonnegative and np.any(raw < 0))):
        raise ValueError(f"invalid tensor workflow {name} parameters")
    return np.array(raw, dtype=np.float64, copy=True)


def _builtin(provider, torch, TorchPES, device):
    kind = type(provider)
    if ("evaluate" in provider.__dict__ or kind.evaluate is not _BUILTIN_EVALUATE[kind]):
        raise ValueError("overridden built-in evaluator requires an explicit tensor binding")
    _independent(provider.metadata, explicit=False)

    class Free(torch.nn.Module):
        metadata = {"batch_independent": True}

        def forward(self, q):
            return {"energy": q.new_zeros(len(q)), "forces": torch.zeros_like(q),
                    "virial": q.new_zeros((len(q), 3, 3))}

    class Harmonic(torch.nn.Module):
        metadata = {"batch_independent": True}

        def __init__(self):
            super().__init__()
            for name in ("k", "center"):
                value = _snapshot(getattr(provider, name), "harmonic " + name,
                                  nonnegative=name == "k")
                if value.ndim > 2:
                    raise ValueError("tensor harmonic parameters cannot depend on the frame axis")
                self.register_buffer(name, torch.from_numpy(value))

        def forward(self, q):
            try:
                k = torch.broadcast_to(self.k, q.shape[1:])
                center = torch.broadcast_to(self.center, q.shape[1:])
            except RuntimeError as error:
                raise ValueError("harmonic k and center must broadcast to [atoms,3]") from error
            difference = q - center
            return 0.5 * (k * difference.square()).sum((1, 2)), -k * difference

    class PeriodicCosine(torch.nn.Module):
        metadata = {"batch_independent": True}

        def __init__(self):
            super().__init__()
            for name, shape in (("height", (3,)), ("inverse", (3, 3)),
                                ("perpendicular", (3, 3))):
                value = _snapshot(getattr(provider, name), "periodic cosine " + name,
                                  shape=shape, nonnegative=name == "height")
                self.register_buffer(name, torch.from_numpy(value))
            repeats = np.asarray(provider.repeats)
            if repeats.shape != (3,) or repeats.dtype.kind not in "iu" or np.any(repeats < 1):
                raise ValueError("periodic cosine repeats must be three positive integers")
            self.register_buffer("repeats", torch.from_numpy(repeats.astype(np.float64)))
            confinement = _snapshot(provider.confinement, "periodic cosine confinement",
                                    shape=(), nonnegative=True)
            self.confinement = float(confinement)
            self.maximum_atom = None
            atoms = None
            if provider.atoms is not None:
                atoms = np.asarray(provider.atoms)
                if (atoms.ndim != 1 or atoms.dtype.kind not in "iu" or not len(atoms)
                        or np.any(atoms < 0) or len(np.unique(atoms)) != len(atoms)
                        or np.any(atoms > np.iinfo(np.int64).max)):
                    raise ValueError("periodic cosine atoms must be distinct nonnegative indices")
                self.maximum_atom = int(atoms.max())
                atoms = torch.from_numpy(atoms.astype(np.int64, copy=True))
            self.register_buffer("atoms", atoms)

        def forward(self, q, cell, pbc):
            # This is the original fixed-lattice model. Its NumPy provider has
            # no variable-cell derivative or virial; neither is invented here.
            if self.maximum_atom is not None and self.maximum_atom >= q.shape[1]:
                raise ValueError("periodic_cosine atom index out of range")
            selected = q if self.atoms is None else q[:, self.atoms]
            phase = 2 * torch.pi * (selected @ self.inverse) * self.repeats
            energy = 0.5 * (self.height * (1 - torch.cos(phase))).sum((1, 2))
            gradient = torch.pi * self.height * self.repeats * torch.sin(phase)
            force = -gradient @ self.inverse.T
            perpendicular = selected @ self.perpendicular
            energy = energy + 0.5 * self.confinement * perpendicular.square().sum((1, 2))
            force = force - self.confinement * perpendicular
            if self.atoms is not None:
                full = torch.zeros_like(q)
                full[:, self.atoms] = force
                force = full
            return energy, force

    if kind is FreePotential:
        result = TorchPES(Free(), device=device, virial="provided")
    elif kind is HarmonicPotential:
        result = TorchPES(Harmonic(), device=device, virial="none")
    else:
        result = TorchPES(PeriodicCosine(), device=device, cell=provider.cell,
                          pbc=provider.pbc, periodic=True, virial="none")
    source_name = {FreePotential: "free", HarmonicPotential: "harmonic",
                   PeriodicCosinePotential: "periodic_cosine"}[kind]
    result.metadata.update(source_provider=source_name, dtype="float64",
                           energy_dtype="float64", periodic=kind is not HarmonicPotential,
                           supports_pbc=kind is not HarmonicPotential)
    return result


def make_workflow_torch_pes(provider, *, device=None):
    """Bind one provider to float64 tensor propagation without a host fallback.

    Free, harmonic, and fixed-lattice periodic-cosine bindings require the exact
    built-in class and its original evaluator. Harmonic parameters may broadcast
    over atoms/Cartesian axes but cannot select individual frames. Other models
    need an open module-backed :class:`TorchPotential`, or an explicit
    ``as_torch_pes()`` bridge such as the T04 batched ML providers. Both source
    and returned adapter must declare independent, stateless frames.

    ``device`` selects the device for analytic snapshots. For borrowed providers
    it must agree with their existing device: binding never moves a provider or
    changes its precision. No binding closes the source provider. This binding
    leaves finite-value checks to TorchDynamics's block boundary; its own
    shape/device/dtype checks introduce no per-evaluation host synchronization.
    Borrowed providers can still perform their existing validation/synchronization.
    """
    try:
        import torch
    except ImportError as error:
        raise BackendUnavailableError("tensor workflow requires PyTorch; install pyrpmd[gpu]") from error
    from ..torch_backend import TorchPES, TensorPESResult, _tensor

    if getattr(provider, "closed", False):
        raise ValueError("tensor workflow cannot bind a closed provider")
    if type(provider) in _BUILTINS:
        return _builtin(provider, torch, TorchPES, _device(torch, "cpu" if device is None else device))
    bridge = getattr(provider, "as_torch_pes", None)
    if type(provider) is not TorchPotential and not callable(bridge):
        raise ValueError("provider has no supported tensor workflow binding; no NumPy fallback")
    _independent(getattr(provider, "metadata", None), explicit=True)
    if not hasattr(provider, "device"):
        raise ValueError("a tensor workflow provider must declare its device")
    actual = _device(torch, provider.device)
    requested = actual if device is None else _device(torch, device)
    if requested != actual:
        raise ValueError("tensor workflow device differs from the configured provider; no device conversion")
    declaration = provider.metadata.get("execution_device")
    if declaration is not None and _device(torch, declaration) != actual:
        raise ValueError("tensor workflow provider device declaration is inconsistent")
    if callable(bridge):
        adapter = bridge()
    else:
        if ("evaluate" in provider.__dict__ or type(provider).evaluate is not _TORCH_EVALUATE):
            raise ValueError("overridden Torch evaluator requires an explicit tensor binding")
        if provider.delegate is not None or not isinstance(provider.model, torch.nn.Module):
            raise ValueError("tensor workflow requires a module-backed TorchPotential, not a NumPy delegate")
        for tensor in (*provider.model.parameters(), *provider.model.buffers()):
            if tensor.is_complex() or (tensor.is_floating_point() and tensor.dtype != torch.float64):
                raise ValueError("tensor workflow model must retain float64; no precision conversion")
            if tensor.device != actual:
                raise ValueError("tensor workflow model and provider device differ; no device conversion")
        adapter = TorchPES.from_provider(provider)
    if not isinstance(adapter, TorchPES):
        raise ValueError("as_torch_pes must return a TorchPES")
    _independent(getattr(adapter, "metadata", None), explicit=True)
    if _device(torch, adapter.device) != actual:
        raise ValueError("tensor bridge changed the provider device")
    declaration = adapter.metadata.get("execution_device")
    if declaration is not None and _device(torch, declaration) != actual:
        raise ValueError("tensor bridge device declaration is inconsistent")

    class Borrowed(TorchPES):
        def __init__(self):
            self.provider, self.adapter = provider, adapter
            self.device = actual
            self.metadata = {**provider.metadata, **adapter.metadata}
            self._source_device = str(provider.device)

        def evaluate_tensor(self, positions):
            if getattr(self.provider, "closed", False):
                raise ValueError("borrowed tensor workflow provider is closed")
            if str(self.provider.device) != self._source_device:
                raise ValueError("borrowed tensor workflow provider device changed")
            if not isinstance(positions, torch.Tensor) or positions.ndim != 3 or min(positions.shape) < 1:
                raise ValueError("positions must be a nonempty tensor [frames,atoms,3]")
            frames, atoms, _ = positions.shape
            _tensor(positions, "positions", (frames, atoms, 3), self.device)
            result = self.adapter.evaluate_tensor(positions)
            if not isinstance(result, TensorPESResult):
                raise ValueError("tensor bridge must return TensorPESResult")
            _tensor(result.energy, "energy", (frames,), self.device)
            _tensor(result.forces, "forces", positions.shape, self.device)
            if result.virial is not None:
                _tensor(result.virial, "virial", (frames, 3, 3), self.device)
            return TensorPESResult(result.energy.detach(), result.forces.detach(),
                                   None if result.virial is None else result.virial.detach())

    return Borrowed()
