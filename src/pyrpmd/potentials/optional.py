"""Optional ASE and PyTorch adapters. Imports occur only when requested."""

import numpy as np
from ..geometry import validate_cell_pbc
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
        "periodic": True,
        "supports_pbc": True,
    }

    def __init__(self, symbols, calculator, *, cell=None, pbc=None):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise BackendUnavailableError(
                "ASE provider requires pip install 'pyrpmd[ase]'"
            ) from exc
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        # Isotope masses are carried by Dynamics, while ASE calculators need
        # chemical element identities (ASE does not recognize D/T symbols).
        symbols = ("H" if symbols in ("D", "T") else symbols) if isinstance(symbols, str) else ["H" if s in ("D", "T") else s for s in symbols]
        self.atoms = Atoms(
            symbols=symbols,
            cell=None if self.cell is None else self.cell * BOHR_ANGSTROM,
            pbc=self.pbc,
        )
        self.atoms.calc = calculator

    def bind_system(self, system):
        """Bind canonical-AU cell and boundary flags without wrapping positions."""
        from ..geometry import validate_cell

        def get(key, default=None):
            return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)

        cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        symbols = get("symbols")
        if symbols is not None and ["H" if s in ("D", "T") else s for s in symbols] != self.atoms.get_chemical_symbols():
            raise PotentialError("ASE system atom identities differ from the provider")
        self.atoms.set_cell(np.zeros((3, 3)) if cell is None else cell * BOHR_ANGSTROM)
        self.atoms.set_pbc(pbc)
        self.cell, self.pbc = cell, pbc

    def evaluate(self, x):
        x = coordinates(x)
        if x.shape[1] != len(self.atoms):
            raise PotentialError("ASE atom count mismatch")
        es = []
        fs = []
        virials = [] if self.cell is not None and "stress" in getattr(self.atoms.calc, "implemented_properties", ()) else None
        for frame in x:
            self.atoms.set_positions(frame * BOHR_ANGSTROM)
            es.append(self.atoms.get_potential_energy() * EV_HARTREE)
            fs.append(self.atoms.get_forces() * EV_HARTREE * BOHR_ANGSTROM)
            if virials is not None:
                # ASE stress is tensile-positive eV/angstrom^3; W=-V*stress.
                virials.append(-self.atoms.get_stress(voigt=False) * self.atoms.get_volume() * EV_HARTREE)
        return validate_result(PESResult(es, fs, virial=virials), x.shape)

    def set_cell(self, cell):
        """Update the ASE cell in bohr, preserving unwrapped atom positions."""
        new_cell, _ = validate_cell_pbc(cell, self.pbc)
        if new_cell is None or not self.pbc.all():
            raise PotentialError("variable-cell ASE PES requires full periodicity")
        self.atoms.set_cell(new_cell * BOHR_ANGSTROM, scale_atoms=False)
        self.cell = new_cell

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
        cell=None,
        pbc=None,
        periodic=False,
        virial="auto",
    ):
        if not isinstance(periodic, (bool, np.bool_)):
            raise ValueError("Torch periodic must be a boolean")
        if virial not in ("auto", "autograd", "provided", "none"):
            raise ValueError("Torch virial must be auto, autograd, provided, or none")
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        self.periodic = bool(periodic)
        self.virial_mode = virial
        self.closed = False
        self._check_cell_contract()
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
                try:
                    md = getattr(candidate, "metadata", {})
                    declared = str(
                        md.get("execution_device", md.get("device", "cpu"))
                        if isinstance(md, dict) else "cpu"
                    )
                    requested = str(self.device)
                    requested = "cuda:0" if requested == "cuda" else requested
                    actual = "cuda:0" if declared == "cuda" else declared
                    if actual != requested:
                        raise PotentialError(
                            f"Torch provider requested {requested}, but factory declares {declared!r}"
                        )
                    self.delegate = PythonPotential(
                        candidate, units=units, cell=self.cell, pbc=self.pbc
                    )
                except BaseException:
                    close = getattr(candidate, "close", None)
                    if callable(close):
                        close()
                    raise
                self.model = None
            else:
                self.model = candidate
        else:
            if model is None:
                raise ValueError("Torch provider requires model or factory")
            self.model = (model if isinstance(model, torch.nn.Module)
                          else torch.jit.load(str(model), map_location=self.device))
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
            "periodic": self.periodic,
            "supports_pbc": self.periodic,
        }
        # A model's independent-frame promise is distinct from accepting a
        # tensor batch. Preserve declarations for opt-in trajectory batching;
        # never infer independence merely from ``native_batch=True``.
        batch_metadata = getattr(self.delegate if self.delegate is not None else self.model,
                                 "metadata", {}) or {}
        if isinstance(batch_metadata, dict):
            for name in ("trajectory_batch_safe", "batch_independent", "ring_polymer_coupled",
                         "coupled_frames", "stateful", "capabilities", "reuse_across_tasks"):
                if name in batch_metadata:
                    self.metadata[name] = batch_metadata[name]

    def _check_cell_contract(self):
        if self.pbc.any() and not self.periodic:
            raise PotentialError(
                "Periodic Torch models require periodic=True and forward(q, cell, pbc)"
            )
        if self.periodic and self.cell is None:
            raise PotentialError("Periodic Torch models require a nonsingular cell")

    def bind_system(self, system):
        get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
        cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        old_cell, old_pbc = self.cell, self.pbc
        self.cell, self.pbc = cell, pbc
        try:
            self._check_cell_contract()
            if self.delegate is not None:
                self.delegate.bind_system(system)
        except BaseException:
            self.cell, self.pbc = old_cell, old_pbc
            raise

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("Torch provider is closed")
        x = coordinates(x)
        if self.delegate is not None:
            return self.delegate.evaluate(x)
        torch = self.torch
        with torch.enable_grad():
            q = torch.tensor(
                x / self.length_factor,
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            # Each frame has its own strain, so batched virials do not mix beads.
            # Row-vector coordinates/cells deform as r' = r @ (I + eps).T.
            derive_virial = self.virial_mode == "autograd" or (
                self.virial_mode == "auto" and self.periodic
            )
            strain = None
            qe = q
            cell = None
            if self.cell is not None:
                cell = torch.tensor(
                    np.broadcast_to(self.cell / self.length_factor, (len(x), 3, 3)).copy(),
                    dtype=torch.float64, device=self.device,
                )
            if derive_virial:
                strain = torch.zeros((len(x), 3, 3), dtype=torch.float64,
                                     device=self.device, requires_grad=True)
                transform = torch.eye(3, dtype=torch.float64, device=self.device) + strain
                qe = torch.matmul(q, transform.transpose(-1, -2))
                if cell is not None:
                    cell = torch.matmul(cell, transform.transpose(-1, -2))
            if self.periodic:
                flags = torch.tensor(np.broadcast_to(self.pbc, (len(x), 3)).copy(),
                                     dtype=torch.bool, device=self.device)
                out = self.model(qe, cell, flags)
            else:
                out = self.model(qe)
            direct = provided_virial = stress = None
            if isinstance(out, dict):
                e = out.get("energy", out.get("energies"))
                direct = out.get("forces")
                provided_virial = out.get("virial")
                stress = out.get("stress")
                if provided_virial is not None and stress is not None:
                    raise PotentialError("Torch model must return either virial or stress")
            elif isinstance(out, (tuple, list)):
                if len(out) != 2:
                    raise PotentialError("Torch tuple must be (energy,forces)")
                e, direct = out
            else:
                e = out
            if not isinstance(e, torch.Tensor) or tuple(e.shape) not in (
                (len(x),), (len(x), 1),
            ):
                raise PotentialError("Torch energy must be a Tensor with shape [B] or [B,1]")
            if e.dtype != torch.float64:
                raise PotentialError("Torch energy must retain float64 precision")
            e = e.reshape(-1)
            w = provided_virial
            if stress is not None:
                if self.cell is None:
                    raise PotentialError("Torch stress requires a cell")
                if (not isinstance(stress, torch.Tensor) or stress.dtype != torch.float64
                        or tuple(stress.shape) != (len(x), 3, 3)):
                    raise PotentialError("Torch stress must be a float64 Tensor [B,3,3]")
                volume = abs(np.linalg.det(self.cell / self.length_factor))
                w = -stress * volume
            needs_strain = strain is not None and (
                self.virial_mode == "autograd" or w is None
            )
            f = direct
            targets = ([q] if f is None else []) + ([strain] if needs_strain else [])
            if targets:
                if not e.requires_grad:
                    if f is None or self.virial_mode == "autograd":
                        raise PotentialError(
                            "model detached energy; forces/virial require a differentiable input graph"
                        )
                else:
                    derivatives = torch.autograd.grad(
                        e.sum(), targets, allow_unused=True, create_graph=False,
                    )
                    if all(g is None for g in derivatives):
                        raise PotentialError("model energy graph is disconnected from coordinates/strain")
                    grads = iter(derivatives)
                    if f is None:
                        g = next(grads)
                        f = torch.zeros_like(q) if g is None else -g
                    if needs_strain:
                        g = next(grads)
                        w = torch.zeros_like(strain) if g is None else -g
            if not isinstance(f, torch.Tensor) or f.dtype != torch.float64:
                raise PotentialError("Torch forces must be a float64 Tensor")
            if self.virial_mode == "provided" and w is None:
                raise PotentialError("Torch virial='provided' requires model virial or stress")
            if self.virial_mode == "none":
                w = None
            if w is not None and (
                not isinstance(w, torch.Tensor) or w.dtype != torch.float64
                or tuple(w.shape) != (len(x), 3, 3)
            ):
                raise PotentialError("Torch virial must be a float64 Tensor [B,3,3]")
            energy = e.detach().cpu().numpy() * self.energy_factor
            force = f.detach().cpu().numpy() * self.energy_factor / self.length_factor
            virial = None if w is None else w.detach().cpu().numpy() * self.energy_factor
        return validate_result(PESResult(energy, force, virial=virial), x.shape)

    def close(self):
        if self.closed:
            return
        if self.delegate is not None:
            self.delegate.close()
        self.model = None
        self.closed = True
