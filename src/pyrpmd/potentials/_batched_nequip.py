"""Native NequIP graph batches; optional imports stay behind construction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

from ._batched_ml_base import _TensorBatchPotential
from .base import BackendUnavailableError, PotentialError
from .python import BOHR_ANGSTROM, EV_HARTREE
from ..validation import positive_number


class NequIPBatchedPotential(_TensorBatchPotential):
    """Evaluate all frames with one NequIP forward call.

    ``model`` is a NequIP GraphModel or a local ``.nequip.pth`` /
    ``.nequip.pt2`` artifact. AOTInductor artifacts must use ``--target batch``,
    PyTorch >=2.13, and ``AOTI_RUNTIME_CHECK_INPUTS=1``. Their compiled shape
    bounds remain authoritative; the native runtime validates every call.
    The model's precision is preserved and must match ``dtype``. CUDA graph
    construction uses NequIP's native Alchemiops batch cell list; CPU uses
    matscipy by default. There is no device or per-frame inference fallback.

    NequIP's energy/length units are defined by its training data. The two
    conversion arguments have the same meaning as in NequIPCalculator.
    """

    def __init__(self, symbols, model, *, device="cpu", dtype="float64",
                 cell=None, pbc=None, charge=0, multiplicity=1, virial=False,
                 chemical_species_to_atom_type_map=True,
                 energy_units_to_eV=1.0, length_units_to_A=1.0,
                 neighborlist_backend=None, neighborlist_backend_kwargs=None):
        energy_scale = positive_number(energy_units_to_eV, "energy_units_to_eV")
        length_scale = positive_number(length_units_to_A, "length_units_to_A")
        self._setup(
            symbols, "nequip-batched", device=device, dtype=dtype, cell=cell,
            pbc=pbc, charge=charge, multiplicity=multiplicity, virial=virial,
            length_factor=length_scale / BOHR_ANGSTROM,
            energy_factor=energy_scale * EV_HARTREE,
        )
        if self.charge != 0 or self.multiplicity != 1:
            raise ValueError("NequIP batched models do not accept charge/spin inputs; "
                             "only charge=0, multiplicity=1 is supported")
        try:
            from nequip.data import AtomicDataDict
            from nequip.data.transforms import NeighborListTransform
            from nequip.nn.embedding.utils import cutoff_str_to_fulldict
        except ImportError as exc:
            raise BackendUnavailableError(
                "NequIP batching requires the NequIP >=0.19.1 graph API"
            ) from exc
        self._aoti = False
        self.model, metadata = self._load_model(model)
        self._keys = AtomicDataDict
        # NequIP deliberately scales/shifts and sums energies in its global
        # float64 dtype even for float32 neural networks. Preserve that result.
        self.energy_dtype = self.torch.float64
        self.metadata["energy_dtype"] = "float64"
        if not isinstance(metadata, Mapping):
            raise ValueError("NequIP model must provide model metadata")
        self._check_precision(metadata)
        try:
            r_max = positive_number(float(metadata["r_max"]), "model r_max")
            type_names = metadata["type_names"]
            if isinstance(type_names, str):
                type_names = type_names.split()
            type_names = tuple(type_names)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("NequIP metadata must contain r_max and type_names") from exc
        if not type_names or len(set(type_names)) != len(type_names):
            raise ValueError("NequIP type_names must be nonempty and unique")
        mapping = chemical_species_to_atom_type_map
        if mapping is True:
            mapping = {name: name for name in type_names}
        if not isinstance(mapping, Mapping):
            raise ValueError("chemical_species_to_atom_type_map must be True or a mapping")
        if any(s not in mapping or mapping[s] not in type_names for s in self.symbols):
            raise ValueError("NequIP species mapping must cover every system element")
        self._types = self.torch.tensor(
            [type_names.index(mapping[s]) for s in self.symbols],
            dtype=self.torch.long, device=self.device,
        )
        backend = neighborlist_backend
        if backend is None:
            backend = "alchemiops" if self.device.type == "cuda" else "matscipy"
        if backend not in ("ase", "matscipy", "vesin", "alchemiops"):
            raise ValueError("unsupported NequIP neighborlist_backend")
        if self.device.type == "cuda" and backend != "alchemiops":
            raise ValueError("CUDA NequIP tensor evaluation requires neighborlist_backend="
                             "'alchemiops'; CPU neighbor lists would transfer positions to host")
        if backend == "alchemiops":
            try:
                from nequip.data._nl import _load_alchemiops_nl
                _load_alchemiops_nl()
            except (ImportError, AttributeError) as exc:
                raise BackendUnavailableError(
                    "NequIP alchemiops batching requires nvalchemiops and NequIP >=0.19.1"
                ) from exc
        cutoff = metadata.get("per_edge_type_cutoff")
        if isinstance(cutoff, str):
            cutoff = cutoff_str_to_fulldict(cutoff, list(type_names)) if cutoff else None
        self.neighborlist = NeighborListTransform(
            r_max=r_max, type_names=list(type_names), per_edge_type_cutoff=cutoff,
            backend=backend, backend_kwargs=dict(neighborlist_backend_kwargs or {}),
        ).to(device=self.device)
        self.model = self._prepare_module(self.model)
        self._layout = None
        self.metadata.update(neighborlist_backend=backend, force_consistent_energy=True)

    def _load_model(self, model):
        if isinstance(model, (str, Path)):
            path = Path(model)
            if not path.is_file():
                raise ValueError("NequIP batched provider requires an existing local model file")
            if str(path).endswith(".nequip.pth"):
                # NequIP's convenience loader drops model_dtype and applies
                # global TF32 settings. Request all precision metadata directly
                # through the public TorchScript loader, without changing it.
                metadata = {key: b"" for key in (
                    "r_max", "type_names", "per_edge_type_cutoff",
                    "model_dtype", "allow_tf32",
                )}
                module = self.torch.jit.load(
                    str(path), map_location=self.device, _extra_files=metadata,
                )
                metadata = {key: (value.decode() if isinstance(value, bytes) else value)
                            for key, value in metadata.items()}
                return self.torch.jit.freeze(module.eval()), metadata
            if str(path).endswith(".nequip.pt2"):
                self._check_aoti_runtime(check_version=True)
                from nequip.model.inference_models import load_aotinductor_model
                from nequip.scripts._compile_utils import BATCH_INPUTS, ASE_OUTPUTS
                # AOTI records the device *type* ("cuda"), not
                # an ordinal. Its runtime resolves that type to the current
                # CUDA device while loading; NequIP compares the type literally.
                # Select the requested ordinal for allocation without passing
                # "cuda:0" to that upstream type-only comparison.
                device_context = (self.torch.cuda.device(self.device)
                                  if self.device.type == "cuda" else nullcontext())
                with device_context:
                    module, metadata = load_aotinductor_model(
                        str(path), self.torch.device(self.device.type),
                        list(BATCH_INPUTS), list(ASE_OUTPUTS),
                    )
                # The upstream loader permits legacy artifacts with no field
                # metadata when keys are supplied. That fallback cannot prove
                # the artifact was compiled for independent frame batches.
                self._check_aoti_metadata(metadata)
                self._aoti = True
                return module, metadata
            raise ValueError("NequIP model must be .nequip.pth or .nequip.pt2 "
                             "compiled with --target batch")
        if not isinstance(model, self.torch.nn.Module):
            raise TypeError("NequIP model must be a framework module or local compiled artifact")
        if not getattr(model, "is_graph_model", False):
            raise TypeError("in-memory NequIP models must be NequIP GraphModel instances")
        from nequip.nn import ForceStressOutput
        derivative_model = getattr(model, "model", None)
        if not isinstance(derivative_model, ForceStressOutput) or not derivative_model.do_derivatives:
            raise ValueError("NequIP batched models require an enabled ForceStressOutput "
                             "to provide energy-gradient forces")
        return model, model.metadata

    def _check_aoti_runtime(self, *, check_version=False):
        if check_version:
            from packaging.version import Version

            if Version(self.torch.__version__) < Version("2.13"):
                raise BackendUnavailableError(
                    "NequIP AOTI batching requires PyTorch >=2.13; older CPU "
                    "compilers can specialize supposedly dynamic node counts. "
                    "Use eager/TorchScript inference or recompile with a supported runtime."
                )
        if os.environ.get("AOTI_RUNTIME_CHECK_INPUTS") != "1":
            raise PotentialError(
                "NequIP AOTI requires AOTI_RUNTIME_CHECK_INPUTS=1 before loading "
                "and throughout evaluation, so native shape/dtype/stride bounds "
                "are checked before compiled kernels execute."
            )

    def _check_aoti_metadata(self, metadata):
        from nequip.scripts._compile_utils import BATCH_INPUTS, ASE_OUTPUTS
        from nequip.utils.aoti_metadata import (
            NEQUIP_AOTI_INPUTS_KEY, NEQUIP_AOTI_OUTPUTS_KEY, parse_aoti_keys,
        )

        if not isinstance(metadata, Mapping):
            raise ValueError("NequIP AOTI artifact must include metadata; recompile with --target batch")
        for key, expected in ((NEQUIP_AOTI_INPUTS_KEY, BATCH_INPUTS),
                              (NEQUIP_AOTI_OUTPUTS_KEY, ASE_OUTPUTS)):
            declared = metadata.get(key)
            if not isinstance(declared, str) or parse_aoti_keys(declared) != list(expected):
                raise ValueError("NequIP AOTI artifact must explicitly record the batch input/output "
                                 "contract; recompile with --target batch")
        try:
            compile_device = self.torch.device(metadata["AOTI_DEVICE_KEY"])
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("NequIP AOTI artifact must declare its compile device; "
                             "recompile with --target batch for the requested device") from exc
        if (compile_device.type == self.device.type == "cuda"
                and compile_device.index is None):
            # Loading occurred inside cuda.device(self.device), so the AOTI
            # runtime resolved its unindexed device to this exact ordinal.
            compile_device = self.device
        if compile_device != self.device:
            raise ValueError("NequIP AOTI compile device differs from requested device; "
                             "recompile with --target batch for the requested device")
        if not metadata.get("model_dtype") or "allow_tf32" not in metadata:
            raise ValueError("NequIP AOTI artifact must explicitly record precision metadata; "
                             "recompile with --target batch and allow_tf32=False")

    def _check_precision(self, metadata):
        if str(metadata.get("allow_tf32", "0")) not in ("", "0", "False", "false"):
            raise ValueError("TF32 NequIP artifacts are unsupported; compile with allow_tf32=False")
        declared = metadata.get("model_dtype") or getattr(self.model, "model_dtype", None)
        if declared is not None and str(declared).removeprefix("torch.") != str(self.dtype).removeprefix("torch."):
            raise ValueError("NequIP model precision differs from requested dtype; "
                             "the provider does not convert model weights")
        for value in self.model.parameters():
            if value.is_floating_point() and value.dtype != self.dtype:
                raise ValueError("NequIP model precision differs from requested dtype")
        if declared is None and not any(True for _ in self.model.parameters()):
            raise ValueError("NequIP compiled artifact must include model_dtype metadata; recompile it")

    def _evaluate_tensor(self, q):
        from ..torch_backend import TensorPESResult
        if self._aoti:
            self._check_aoti_runtime()
        torch = self.torch
        batch, natoms, _ = q.shape
        if self._layout is None or self._layout[0] != batch:
            self._layout = (batch, {
                "atom_types": self._types.repeat(batch),
                "batch": torch.arange(batch, device=self.device).repeat_interleave(natoms),
                self._keys.NUM_NODES_KEY: torch.full((batch,), natoms, dtype=torch.long, device=self.device),
            })
        cell = self._cell_tensor()
        if cell is None:
            cell = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        data = dict(self._layout[1])
        data.update(
            pos=(q / self.length_factor).reshape(-1, 3),
            cell=cell.expand(batch, 3, 3).contiguous(),
            pbc=self._pbc_tensor().expand(batch, 3).contiguous(),
        )
        with torch.enable_grad():
            data = self.neighborlist(data)
            # Native AOTI checks require the strides used at compilation.
            # In particular expanded cells must be contiguous.
            data = {key: value.contiguous() for key, value in data.items()}
            output = self.model(data)
        energy, forces = output.get("total_energy"), output.get("forces")
        if (not isinstance(energy, torch.Tensor) or energy.shape not in ((batch,), (batch, 1))
                or not isinstance(forces, torch.Tensor) or forces.shape != (batch * natoms, 3)):
            raise PotentialError("NequIP must return total_energy[B] and forces[B*N,3]")
        energy = energy.reshape(batch).detach() * self.energy_factor
        forces = forces.reshape(batch, natoms, 3).detach() * (self.energy_factor / self.length_factor)
        virial = None
        if self.virial:
            raw = output.get("virial")
            if isinstance(raw, torch.Tensor) and raw.shape == (batch, 3, 3):
                virial = raw.detach() * self.energy_factor
            elif not self.pbc.any():
                # Isolated molecules have no periodic cell contribution.
                virial = torch.einsum("bna,bnc->bac", forces, q)
            else:
                stress = output.get("stress")
                if not isinstance(stress, torch.Tensor) or stress.shape != (batch, 3, 3):
                    raise PotentialError("NequIP model does not provide the requested virial/stress")
                virial = -stress.detach() * torch.linalg.det(cell).abs() * self.energy_factor
        return TensorPESResult(energy, forces, virial)

    def close(self):
        super().close()
        self.model = None
        self._layout = None
