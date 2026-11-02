"""Batched metatomic model evaluation with device-local neighbor lists.

The public re-export is in :mod:`pyrpmd.potentials.batched_ml`.  Imports of
metatomic, metatensor and vesin stay lazy so they remain optional dependencies.
"""

from __future__ import annotations

import importlib
import os

from ..system import _Z
from ._batched_ml_base import _TensorBatchPotential
from .base import BackendUnavailableError, PotentialError
from .python import BOHR_ANGSTROM, EV_HARTREE


class MetatomicBatchedPotential(_TensorBatchPotential):
    """Evaluate an exported metatomic energy model on all frames in one call.

    Supported models expose the standard ``energy`` output and differentiable
    coordinate dependence. Optional system-level ``charge`` and
    ``spin_multiplicity`` inputs follow the ASE adapter's charge/spin convention.
    Other requested inputs and non-conservative force heads are unsupported.

    ``dtype`` is an explicit precision choice and must agree with the exported
    model. Tensor inputs/outputs remain on the selected device and retain that
    dtype. The NumPy and tensor interfaces both use atomic units.
    """

    def __init__(self, symbols, model, *, device="cpu", dtype="float64", cell=None,
                 pbc=None, charge=0, multiplicity=1, virial=False,
                 check_consistency=True):
        self._setup(
            symbols, "metatomic", device=device, dtype=dtype, cell=cell, pbc=pbc,
            charge=charge, multiplicity=multiplicity, virial=virial,
            length_factor=1.0 / BOHR_ANGSTROM, energy_factor=EV_HARTREE,
        )
        if not isinstance(check_consistency, bool):
            raise ValueError("check_consistency must be a boolean")
        self.check_consistency = check_consistency
        self._compute_virial = virial
        try:
            self._mta = importlib.import_module("metatomic.torch")
            self._mts = importlib.import_module("metatensor.torch")
            self._torch = importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(
                "metatomic provider requires its supported optional API; "
                "install 'pyrpmd[metatomic]'"
            ) from exc

        if isinstance(model, (str, bytes, os.PathLike)):
            try:
                model = self._mta.load_atomistic_model(os.fspath(model))
            except Exception as exc:
                raise PotentialError(f"could not load metatomic model: {exc}") from exc
        if not self._mta.is_atomistic_model(model):
            raise TypeError("model must be an exported metatomic AtomisticModel or its path")

        capabilities = model.capabilities()
        expected_dtype = str(self.dtype).removeprefix("torch.")
        if capabilities.dtype != expected_dtype:
            raise ValueError(
                f"metatomic model dtype is {capabilities.dtype!r}, but dtype="
                f"{expected_dtype!r} was requested; model precision is never changed"
            )
        if self.device.type not in capabilities.supported_devices:
            raise ValueError(
                f"metatomic model does not support device {self.device.type!r}"
            )
        if "energy" not in capabilities.outputs:
            raise ValueError("metatomic provider requires the standard 'energy' output")
        if not capabilities.length_unit or not capabilities.outputs["energy"].unit:
            raise ValueError("metatomic model must declare length and energy units")
        missing_types = sorted({_Z[s] for s in self.symbols} - set(capabilities.atomic_types))
        if missing_types:
            raise ValueError(f"metatomic model does not support atomic numbers {missing_types}")
        self.model = self._prepare_module(model)
        for value in list(self.model.parameters()) + list(self.model.buffers()):
            if value.is_floating_point() and value.dtype != self.dtype:
                raise ValueError("metatomic model tensors disagree with its declared dtype")
        self._types = self._torch.tensor(
            [_Z[s] for s in self.symbols], dtype=self._torch.int32, device=self.device,
        )
        self._options = self._mta.ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={"energy": self._mta.ModelOutput(unit="eV", sample_kind="system")},
        )
        self._inputs = self._make_inputs(self.model.requested_inputs(use_new_names=True))

        requests = self.model.requested_neighbor_lists()
        self._neighbors = []
        if requests:
            try:
                NeighborList = importlib.import_module("vesin.metatomic").NeighborList
                importlib.import_module("vesin_torch")
                # No cross-frame Verlet state: changing batch order or batch size
                # must not alter the set of interactions used for a frame.
                self._neighbors = [
                    NeighborList(
                        request, length_unit="angstrom", torchscript=True,
                        check_consistency=check_consistency, skin=0.0,
                    ) for request in requests
                ]
            except (ImportError, AttributeError, OSError) as exc:
                raise BackendUnavailableError(
                    "metatomic neighbor lists require vesin and vesin-torch; "
                    "install 'pyrpmd[metatomic]'"
                ) from exc
        self.metadata = dict(
            self.metadata, neighbor_device=str(self.device),
            force_consistent_energy=True, check_consistency=check_consistency,
        )

    def _make_inputs(self, requested):
        """Construct immutable per-system electronic-state input TensorMaps."""
        torch, mts = self._torch, self._mts
        inputs = {}
        values = {"charge": self.charge, "spin_multiplicity": self.multiplicity}
        for name, option in requested.items():
            if name not in values or option.sample_kind != "system":
                raise ValueError(
                    "metatomic supports only system-level charge and spin_multiplicity "
                    f"inputs; model requested {name!r} ({option.sample_kind})"
                )
            zero = torch.zeros((1, 1), dtype=torch.int32, device=self.device)
            block = mts.TensorBlock(
                values=torch.tensor([[values[name]]], dtype=self.dtype, device=self.device),
                samples=mts.Labels(["system"], zero), components=[],
                properties=mts.Labels([name], zero),
            )
            data = mts.TensorMap(mts.Labels(["_"], zero), [block])
            data.set_info("unit", "e" if name == "charge" else "")
            inputs[name] = data
        if self.charge != 0 and "charge" not in inputs:
            raise ValueError("metatomic model does not request charge; nonzero charge is unsupported")
        if self.multiplicity != 1 and "spin_multiplicity" not in inputs:
            raise ValueError(
                "metatomic model does not request spin_multiplicity; multiplicity != 1 is unsupported"
            )
        return inputs

    def _evaluate_tensor(self, positions):
        from ..torch_backend import TensorPESResult

        torch = self._torch
        if torch.is_inference_mode_enabled():
            raise ValueError("metatomic forces require autograd; use no_grad, not inference_mode")
        with torch.enable_grad():
            q = positions.detach().requires_grad_(True)
            coordinates = q * BOHR_ANGSTROM
            frames = q.shape[0]
            cell = self._cell_tensor()
            if cell is None:
                cell = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
            pbc = self._pbc_tensor()
            # metatomic represents nonperiodic directions by zero lattice rows.
            # Coordinates stay unwrapped, including in partially periodic cells.
            cell = cell * pbc[:, None]
            cells = cell.expand(frames, 3, 3)
            strain = None
            if self._compute_virial:
                strain = torch.eye(3, dtype=self.dtype, device=self.device).expand(
                    frames, 3, 3,
                ).clone().requires_grad_(True)
                coordinates = torch.bmm(coordinates, strain.transpose(-1, -2))
                cells = torch.bmm(cells, strain.transpose(-1, -2))
            systems = [
                self._mta.System(
                    types=self._types, positions=coordinates[i], cell=cells[i], pbc=pbc,
                ) for i in range(frames)
            ]
            for system in systems:
                for name, value in self._inputs.items():
                    system.add_data(name, value)
            for calculator in self._neighbors:
                # One calculator is reused for every frame, so retain independent
                # neighbor storage until the shared backward pass has completed.
                calculator.add_neighbor_list(systems, copy=True)
            outputs = self.model(
                systems=systems, options=self._options,
                check_consistency=self.check_consistency,
            )
            block = outputs["energy"].block()
            if (tuple(block.values.shape) != (frames, 1)
                    or list(block.samples.names) != ["system"] or len(block.components)):
                raise PotentialError("metatomic energy must contain one scalar per system")
            indices = block.samples.column("system").to(dtype=torch.long)
            order = torch.argsort(indices)
            if not torch.equal(indices[order], torch.arange(frames, device=indices.device)):
                raise PotentialError("metatomic energy samples must identify each system exactly once")
            energy = block.values[:, 0][order] * EV_HARTREE
            if not energy.requires_grad:
                raise PotentialError(
                    "metatomic energy is detached; this provider requires an autograd energy model"
                )
            differentiated = (q,) if strain is None else (q, strain)
            gradients = torch.autograd.grad(energy.sum(), differentiated, allow_unused=True)
            if any(value is None for value in gradients):
                raise PotentialError("metatomic energy is disconnected from positions or strain")
            forces = -gradients[0]
            virial = None if strain is None else -gradients[1]
            return TensorPESResult(
                energy=energy.detach(), forces=forces.detach(),
                virial=None if virial is None else virial.detach(),
            )

    def close(self):
        if not self.closed:
            super().close()
            self.model = None
            self._neighbors.clear()
            self._inputs.clear()
            self._types = None
