"""Shared strict NumPy/Torch boundary for native ML batch providers.

Optional libraries are deliberately imported at construction time.  Neither
importing PyRPMD nor reading an input file initializes an accelerator.
"""

from __future__ import annotations

import importlib

import numpy as np

from ..geometry import validate_cell_pbc
from .adapters import _BoundPotential
from .base import BackendUnavailableError, PESResult, PotentialError, validate_result


def _optional(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"{extra} requires its optional dependencies; install 'pyrpmd[{extra}]' "
            f"in a compatible environment (could not import {module})"
        ) from exc


class _TensorBatchPotential(_BoundPotential):
    """One native model call, with explicit model precision and device.

    Subclasses return atomic-unit ``TensorPESResult`` from ``_evaluate_tensor``.
    The NumPy boundary alone transfers coordinates/results between host/device.
    Tensor calls require the declared input precision/device. Framework-native
    geometry and reduction precisions are explicit in provider metadata.
    """

    def _setup(self, symbols, provider, *, device="cpu", dtype="float64", cell=None,
               pbc=None, charge=0, multiplicity=1, virial=False,
               length_factor=1.0, energy_factor=1.0):
        torch = _optional("torch", provider)
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must explicitly be 'float32' or 'float64'")
        if not isinstance(virial, (bool, np.bool_)):
            raise ValueError("virial must be a boolean")
        self.torch, self.dtype = torch, getattr(torch, dtype)
        self.energy_dtype = self.dtype
        self.virial_dtype = self.dtype
        try:
            requested = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("device must be an explicit CPU or CUDA device") from exc
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("batched ML providers support device='cpu' or 'cuda[:index]'")
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA was requested but is unavailable; no CPU fallback")
        try:
            self.device = torch.empty(0, device=requested).device
        except (RuntimeError, AssertionError) as exc:
            raise BackendUnavailableError(f"requested device {device!r} is unavailable") from exc
        for value, name in ((length_factor, "length_factor"), (energy_factor, "energy_factor")):
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.length_factor, self.energy_factor = float(length_factor), float(energy_factor)
        self.virial = bool(virial)
        self._cached_cell = self._cached_pbc = None
        self._bind(symbols, cell, pbc, charge, multiplicity)
        self.metadata = {
            "provider": provider, "execution_device": str(self.device),
            "dtype": dtype, "energy_dtype": dtype, "virial_dtype": dtype,
            "native_batch": True, "batch_independent": True,
            "trajectory_batch_safe": True, "stateful": False,
            "thread_safety": "context_serial", "supports_pbc": True,
            "periodic": True, "energy_convention": "force-consistent",
        }

    def _prepare_module(self, model):
        """Take ownership of an inference module without changing its dtype."""
        torch = self.torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a supported torch.nn.Module or local exported model")
        metadata = getattr(model, "metadata", {}) or {}
        if isinstance(metadata, dict):
            capabilities = metadata.get("capabilities", {}) or {}
            if (metadata.get("batch_independent") is False
                    or metadata.get("trajectory_batch_safe") is False
                    or metadata.get("stateful") or metadata.get("coupled_frames")
                    or metadata.get("ring_polymer_coupled")
                    or (isinstance(capabilities, dict) and capabilities.get("stateful"))):
                raise PotentialError("model explicitly rejects independent stateless batching")
        parameters = [p for p in model.parameters() if p.is_floating_point()]
        if parameters and any(p.dtype != self.dtype for p in parameters):
            found = sorted({str(p.dtype) for p in parameters})
            raise ValueError(f"model dtype {found} does not match requested {self.dtype}; no implicit precision conversion")
        # Eval mode disables training-time randomness. Freezing parameters avoids
        # tracking parameter gradients; coordinate/strain gradients remain enabled.
        model = model.to(device=self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _cell_tensor(self):
        if self._cached_cell is None:
            cell = np.zeros((3, 3)) if self.cell is None else self.cell / self.length_factor
            self._cached_cell = self.torch.tensor(cell, dtype=self.dtype, device=self.device)
        return self._cached_cell

    def _pbc_tensor(self):
        if self._cached_pbc is None:
            self._cached_pbc = self.torch.tensor(self.pbc, dtype=self.torch.bool, device=self.device)
        return self._cached_pbc

    def bind_system(self, system):
        if self.closed:
            raise PotentialError("cannot rebind a closed batched ML provider")
        super().bind_system(system)
        self._cached_cell = self._cached_pbc = None

    def set_cell(self, cell):
        if self.closed:
            raise PotentialError("cannot set cell on a closed batched ML provider")
        new_cell, _ = validate_cell_pbc(cell, self.pbc)
        if new_cell is None or not self.pbc.all():
            raise PotentialError("variable cell requires full periodicity")
        self.cell = new_cell
        self._cached_cell = None

    def evaluate(self, positions):
        x = self._coordinates(positions)
        q = self.torch.as_tensor(x, dtype=self.dtype, device=self.device)
        result = self.evaluate_tensor(q)
        return validate_result(PESResult(
            result.energy.detach().cpu().numpy(),
            result.forces.detach().cpu().numpy(),
            virial=None if result.virial is None else result.virial.detach().cpu().numpy(),
        ), x.shape)

    def evaluate_tensor(self, positions):
        """Evaluate AU coordinates on the declared device, returning AU tensors.

        Float64 is directly compatible with the current ``TorchPES`` contract.
        Explicit float32 models retain their native result precision here
        (see ``energy_dtype`` and ``virial_dtype`` metadata for native reductions).
        The current float64 dynamics backend must not consume float32 forces.
        No model retry, precision change, device fallback, or frame splitting is
        performed after a backend failure, including CUDA out-of-memory errors.
        """
        from ..torch_backend import TensorPESResult

        torch = self.torch
        if self.closed:
            raise PotentialError(f"{self.metadata['provider']} provider is closed")
        if (not isinstance(positions, torch.Tensor) or positions.ndim != 3
                or tuple(positions.shape[1:]) != (len(self.symbols), 3)
                or positions.shape[0] < 1):
            raise PotentialError("positions must be a nonempty tensor [batch,atom,3] with the bound atom count")
        if positions.dtype != self.dtype or positions.device != self.device:
            raise PotentialError(f"positions must have dtype {self.dtype} on {self.device}; no implicit conversion")
        if torch.is_inference_mode_enabled():
            raise PotentialError("inference_mode is unsupported: forces require coordinate gradients; use no_grad")
        if torch.is_autocast_enabled(self.device.type):
            raise PotentialError("autocast is unsupported: batched ML evaluation must preserve the explicit model precision")
        if (self.device.type == "cuda" and self.dtype == torch.float32
                and torch.backends.cuda.matmul.allow_tf32):
            raise PotentialError("TF32 must be disabled for float32 batched ML evaluation; reduced precision is not implicit")
        if not torch.isfinite(positions).all().item():
            raise PotentialError("PES coordinates contain NaN/Inf")
        try:
            with torch.enable_grad():
                result = self._evaluate_tensor(positions)
            if not isinstance(result, TensorPESResult):
                raise PotentialError("batched ML backend must return TensorPESResult")
            fields = [(result.energy, "energy", (len(positions),), self.energy_dtype),
                      (result.forces, "forces", tuple(positions.shape), self.dtype)]
            if self.virial and result.virial is None:
                raise PotentialError("virial was requested but the model did not return it")
            if result.virial is not None:
                fields.append((result.virial, "virial", (len(positions), 3, 3), self.virial_dtype))
            for value, name, shape, dtype in fields:
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or value.dtype != dtype or value.device != self.device):
                    raise PotentialError(f"{name} must be {dtype} {shape} on {self.device}")
            if not torch.stack([torch.isfinite(v).all() for v, _, _, _ in fields]).all().item():
                raise PotentialError("batched ML model returned nonfinite energy/forces/virial")
            return TensorPESResult(result.energy.detach(), result.forces.detach(),
                                   None if result.virial is None else result.virial.detach())
        except PotentialError:
            raise
        except Exception as exc:
            raise PotentialError(f"{self.metadata['provider']} batch evaluation failed: {exc}") from exc

    def as_torch_pes(self):
        """Borrow this float64 provider through the existing ``TorchPES`` type.

        The current dynamics code uses an explicit TorchPES type check. This
        forwarding adapter satisfies it without editing ``torch_backend.py``,
        wrapping model inference again, moving data, or changing precision.
        Keep this provider open for the lifetime of the borrowed adapter.
        """
        from ..torch_backend import TorchPES

        if (self.dtype != self.torch.float64 or self.energy_dtype != self.torch.float64
                or (self.virial and self.virial_dtype != self.torch.float64)):
            raise ValueError("the current TorchPES dynamics contract requires float64; no precision conversion")
        if self.closed:
            raise PotentialError("cannot adapt a closed batched ML provider")

        class _BorrowedTorchPES(TorchPES):
            def __init__(self, provider):
                self.provider = provider
                self.device = provider.device
                self.metadata = dict(provider.metadata)

            def evaluate_tensor(self, positions):
                return self.provider.evaluate_tensor(positions)

        return _BorrowedTorchPES(self)

    def close(self):
        if not self.closed:
            self._reset_engine()
            self._cached_cell = self._cached_pbc = None
            self.closed = True
