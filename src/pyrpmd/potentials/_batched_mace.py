"""MACE inference on one disconnected graph containing all physical frames."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from ..system import _Z
from ._batched_ml_base import _TensorBatchPotential, _optional
from .base import PotentialError
from .python import BOHR_ANGSTROM, EV_HARTREE


class MACEBatchedPotential(_TensorBatchPotential):
    """Native energy MACE/ScaleShiftMACE batch inference (MACE 0.3.17 API).

    ``model`` is a local ``.model`` file or an initialized MACE module.  The
    module is owned by this provider, moved to the explicitly selected device,
    put in eval mode and frozen. Its stored precision is never changed.
    Cell-list neighbor construction runs on that same device; no ASE Atoms or
    per-frame model/force calls are made. A single cached composition template
    is reused while the number of frames is unchanged.
    """

    def __init__(self, symbols, model, *, device="cpu", dtype="float64", cell=None,
                 pbc=None, charge=0, multiplicity=1, virial=False, head=None):
        self._setup(symbols, "mace-batched", device=device, dtype=dtype, cell=cell,
                    pbc=pbc, charge=charge, multiplicity=multiplicity, virial=virial,
                    length_factor=1 / BOHR_ANGSTROM, energy_factor=EV_HARTREE)
        if self.charge != 0 or self.multiplicity != 1:
            raise ValueError("mace-batched energy models do not select charge/spin; require charge=0, multiplicity=1")
        modules = _optional("mace.modules", "mace-batched")
        neighbors = _optional("vesin_torch", "mace-batched")
        if isinstance(model, (str, Path)):
            path = Path(model)
            if not path.is_file():
                raise ValueError(f"MACE model must be an existing local file: {path}")
            model = self.torch.load(str(path), map_location=self.device, weights_only=False)
        if type(model) not in (modules.MACE, modules.ScaleShiftMACE):
            raise ValueError("mace-batched supports energy MACE and ScaleShiftMACE modules; polar/magnetic/dipole models need a separate contract")
        if hasattr(model, "joint_embedding"):
            raise ValueError("MACE models with additional embedding inputs are not supported by mace-batched")
        self.model = self._prepare_module(model)
        self.cutoff = float(self.model.r_max.item())
        if not 0 < self.cutoff < float("inf"):
            raise ValueError("MACE model cutoff must be positive and finite")
        species = self.model.atomic_numbers.detach().cpu().tolist()
        if len(set(species)) != len(species) or any(_Z[s] not in species for s in self.symbols):
            raise ValueError("MACE model atomic numbers must uniquely cover every system element")
        heads = list(getattr(self.model, "heads", ["Default"]))
        if head is None:
            if "Default" in heads:
                head = "Default"
            elif len(heads) == 1:
                head = heads[0]
            else:
                raise ValueError("multi-head MACE requires an explicit params.head")
        if head not in heads:
            raise ValueError(f"MACE head {head!r} is not one of {heads!r}")
        self.head = head
        self._head_index = heads.index(head)
        atom_types = self.torch.tensor([species.index(_Z[s]) for s in self.symbols],
                                      dtype=self.torch.long, device=self.device)
        self._node_attributes = self.torch.nn.functional.one_hot(
            atom_types, num_classes=len(species)
        ).to(dtype=self.dtype)
        self._numbers = self.torch.tensor([_Z[s] for s in self.symbols],
                                         dtype=self.torch.long, device=self.device)
        self._neighbors = neighbors.NeighborList(self.cutoff, full_list=True, sorted=True,
                                                algorithm="cell_list", skin=0.0)
        self._batch_cache = None
        self.metadata.update(model_version=version("mace-torch"), head=self.head,
                             neighbor_backend="vesin-cell-list", model_calls_per_batch=1)

    def _template(self, frames):
        if self._batch_cache is None or self._batch_cache[0] != frames:
            torch, atoms = self.torch, len(self.symbols)
            self._batch_cache = (frames, {
                "batch": torch.arange(frames, device=self.device).repeat_interleave(atoms),
                "ptr": torch.arange(frames + 1, device=self.device) * atoms,
                "node_attrs": self._node_attributes.repeat(frames, 1),
                "atomic_numbers": self._numbers.repeat(frames),
                "head": torch.full((frames,), self._head_index, dtype=torch.long, device=self.device),
            })
        return dict(self._batch_cache[1])

    def _evaluate_tensor(self, q):
        from ..torch_backend import TensorPESResult

        torch, frames, atoms = self.torch, len(q), len(self.symbols)
        positions = (q.detach() / self.length_factor).contiguous()
        cell, flags = self._cell_tensor(), self._pbc_tensor()
        indices, shifts = [], []
        # Vesin's native cell list is O(N + E), supports triclinic/partial PBC
        # and runs on CPU or CUDA without moving coordinates to the host.
        # Its API is per-system; only topology assembly loops over systems.
        # Inference and force/strain differentiation below are one whole batch.
        for frame in range(frames):
            i, j, shift = self._neighbors.compute(positions[frame], cell, flags,
                                                  quantities="ijS", copy=True)
            indices.append(torch.stack((i, j)) + frame * atoms)
            shifts.append(shift)
        unit_shifts = torch.cat(shifts).to(dtype=self.dtype)
        graph = self._template(frames)
        graph.update(positions=positions.reshape(-1, 3),
                     edge_index=torch.cat(indices, dim=1),
                     unit_shifts=unit_shifts, shifts=unit_shifts @ cell,
                     cell=cell.expand(frames, 3, 3).reshape(-1, 3),
                     pbc=flags.expand(frames, 3))
        output = self.model(graph, training=False, compute_force=True,
                            compute_virials=self.virial, compute_stress=False)
        if not isinstance(output, dict) or output.get("energy") is None or output.get("forces") is None:
            raise PotentialError("MACE energy model must return energy and energy-gradient forces")
        virial = output.get("virials") if self.virial else None
        return TensorPESResult(
            output["energy"] * self.energy_factor,
            output["forces"].reshape(frames, atoms, 3) * (self.energy_factor / self.length_factor),
            None if virial is None else virial * self.energy_factor,
        )

    def _reset_engine(self):
        self._batch_cache = None

    def close(self):
        super().close()
        self.model = self._neighbors = None
