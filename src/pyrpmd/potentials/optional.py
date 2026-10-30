"""Optional ASE and PyTorch adapters. Imports occur only when requested."""

import numpy as np
from .base import (
    Potential,
    PESResult,
    PotentialError,
    BackendUnavailableError,
    coordinates,
    validate_result,
)
from .python import (
    import_object,
    unit_factors,
    BOHR_ANGSTROM,
    EV_HARTREE,
    PythonPotential,
)


class ASEPotential(Potential):
    metadata = {
        "provider": "ase",
        "thread_safety": "context_serial",
        "execution_device": "cpu",
        "supports_pbc": True,
    }

    def __init__(self, symbols, calculator, *, cell=None, pbc=(False, False, False)):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise BackendUnavailableError(
                "ASE provider requires pip install 'pyrpmd[ase]'"
            ) from exc
        from ..geometry import validate_cell

        cell, pbc = validate_cell(cell, pbc)
        ase_symbols = (
            ("H" if symbols in ("D", "T") else symbols)
            if isinstance(symbols, str)
            else ["H" if s in ("D", "T") else s for s in symbols]
        )
        self.atoms = Atoms(
            symbols=ase_symbols,
            cell=None if cell is None else cell * BOHR_ANGSTROM,
            pbc=pbc,
        )
        self.atoms.calc = calculator

    def bind_system(self, system):
        """Bind canonical-AU cell and boundary flags without wrapping positions."""
        from ..geometry import validate_cell

        def get(key, default=None):
            return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)

        cell, pbc = validate_cell(get("cell"), get("pbc", (False, False, False)))
        symbols = get("symbols")
        if symbols is not None and ["H" if s in ("D", "T") else s for s in symbols] != self.atoms.get_chemical_symbols():
            raise PotentialError("ASE system atom identities differ from the provider")
        self.atoms.set_cell(np.zeros((3, 3)) if cell is None else cell * BOHR_ANGSTROM)
        self.atoms.set_pbc(pbc)

    def evaluate(self, x):
        x = coordinates(x)
        if x.shape[1] != len(self.atoms):
            raise PotentialError("ASE atom count mismatch")
        es = []
        fs = []
        for frame in x:
            self.atoms.set_positions(frame * BOHR_ANGSTROM)
            es.append(self.atoms.get_potential_energy() * EV_HARTREE)
            fs.append(self.atoms.get_forces() * EV_HARTREE * BOHR_ANGSTROM)
        return validate_result(PESResult(es, fs), x.shape)

    def close(self):
        close = getattr(self.atoms.calc, "close", None)
        if close:
            close()


class TorchPotential(Potential):
    def __init__(
        self,
        model=None,
        *,
        factory=None,
        params=None,
        device="cuda",
        units=None,
        dtype="float64",
    ):
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                "Torch provider requires PyTorch; install pyrpmd[gpu] with the appropriate CUDA wheel"
            ) from exc
        if dtype != "float64":
            raise ValueError("PyRPMD reference Torch provider requires float64")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError(
                "CUDA was requested but no CUDA device is available"
            )
        if self.device.type == "cuda":
            idx = 0 if self.device.index is None else self.device.index
            if idx >= torch.cuda.device_count():
                raise BackendUnavailableError(f"CUDA device index {idx} does not exist")
        self.delegate = None
        options = dict(params or {})
        options.pop("dtype", None)
        options.pop("forces", None)
        if factory:
            # Factories receive explicit device/model arguments; no cwd changes.
            options.setdefault("device", str(self.device))
            if model is not None:
                options.setdefault("model", model)
            candidate = import_object(factory)(**options)
            if hasattr(candidate, "evaluate") and not isinstance(
                candidate, torch.nn.Module
            ):
                md = getattr(candidate, "metadata", {})
                declared = (
                    md.get("execution_device", md.get("device", ""))
                    if isinstance(md, dict)
                    else ""
                )
                if self.device.type == "cuda" and not str(declared).startswith("cuda"):
                    raise PotentialError(
                        "GPU provider factory must declare CUDA execution in metadata"
                    )
                self.delegate = PythonPotential(candidate, units=units)
                self.model = None
            else:
                self.model = candidate
        else:
            if model is None:
                raise ValueError("Torch provider requires model or factory")
            self.model = torch.jit.load(str(model), map_location=self.device)
        if self.model is not None:
            if hasattr(self.model, "to"):
                self.model = self.model.to(device=self.device, dtype=torch.float64)
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(self.model, "parameters"):
                for p in self.model.parameters():
                    p.requires_grad_(False)
        self.length_factor, self.energy_factor = unit_factors(units)
        self.metadata = {
            "provider": "torch",
            "execution_device": str(self.device),
            "thread_safety": "context_serial",
            "native_batch": True,
            "dtype": "float64",
        }

    def evaluate(self, x):
        x = coordinates(x)
        if self.delegate:
            return self.delegate.evaluate(x)
        torch = self.torch
        with torch.enable_grad():
            q = torch.tensor(
                x / self.length_factor,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            out = self.model(q)
            direct = None
            if isinstance(out, dict):
                e = out.get("energy", out.get("energies"))
                direct = out.get("forces")
            elif isinstance(out, (tuple, list)):
                if len(out) != 2:
                    raise PotentialError("Torch tuple must be (energy,forces)")
                e, direct = out
            else:
                e = out
            if not isinstance(e, torch.Tensor) or tuple(e.shape) not in (
                (len(x),),
                (len(x), 1),
            ):
                raise PotentialError(
                    "Torch energy must be a Tensor with shape [B] or [B,1]"
                )
            if e.dtype != torch.float64:
                raise PotentialError("Torch energy must retain float64 precision")
            e = e.reshape(-1)
            if direct is None:
                if not e.requires_grad:
                    raise PotentialError(
                        "model detached energy; forces require a differentiable input graph"
                    )
                g = torch.autograd.grad(
                    e.sum(), q, allow_unused=False, create_graph=False
                )[0]
                f = -g
            else:
                f = direct
            if not isinstance(f, torch.Tensor) or f.dtype != torch.float64:
                raise PotentialError("Torch forces must be a float64 Tensor")
            energy = e.detach().cpu().numpy() * self.energy_factor
            force = f.detach().cpu().numpy() * self.energy_factor / self.length_factor
        return validate_result(PESResult(energy, force), x.shape)

    def close(self):
        if self.delegate:
            self.delegate.close()
        self.model = None
