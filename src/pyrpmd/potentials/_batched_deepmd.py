"""Sparse, tensor-native evaluation of conservative DeePMD PyTorch models."""

from __future__ import annotations

import json
import os

from ._batched_ml_base import _TensorBatchPotential, _optional
from .base import BackendUnavailableError, PotentialError
from .python import unit_factors


class DeepMDBatchedPotential(_TensorBatchPotential):
    """One batched atomic-energy call with sparse device-local neighbors.

    Supports standard float32/float64 DeePMD ``EnergyModel`` instances or their
    TorchScript exports. Learned parameters keep the requested precision;
    DeePMD's native geometry, total-energy and virial precision remains float64.
    Forces retain the input precision. Frame/atom parameters, spin models, data
    modifiers and direct-force fitting heads are rejected. The existing
    ``deepmd`` provider remains the general NumPy ``DeepPot`` interface.

    DeePMD's trained ``sel`` is respected: choose the nearest ``sum(sel)``
    neighbors, then retain each type's nearest ``sel[type]`` entries for a
    type-dependent descriptor. An exactly tied selection boundary is rejected,
    since the upstream dense ``topk`` has no reproducible tie policy.
    """

    def __init__(self, symbols, model, *, device="cpu", dtype="float64", cell=None,
                 pbc=None, charge=0, multiplicity=1, virial=False, units=None):
        length_factor, energy_factor = unit_factors(
            {"positions": "angstrom", "energy": "eV"} if units is None else units
        )
        self._setup(
            symbols, "deepmd-batched", device=device, dtype=dtype, cell=cell, pbc=pbc,
            charge=charge, multiplicity=multiplicity, virial=virial,
            length_factor=length_factor, energy_factor=energy_factor,
        )
        if self.charge != 0 or self.multiplicity != 1:
            raise ValueError("deepmd-batched supports neutral, multiplicity-one models only")
        try:
            deepmd_models = _optional("deepmd.pt.model.model", "deepmd-batched")
            neighbors = _optional("vesin_torch", "deepmd-batched")
            region = _optional("deepmd.pt.utils.region", "deepmd-batched")
        except (ImportError, OSError, RuntimeError) as exc:
            raise BackendUnavailableError(
                "deepmd-batched requires a DeePMD PyTorch build compatible with the "
                f"installed Torch and vesin-torch: {exc}"
            ) from exc
        torch = self.torch
        if isinstance(model, (str, bytes, os.PathLike)):
            extra = {"data_modifier.pth": ""}
            try:
                model = torch.jit.load(
                    os.fspath(model), map_location=self.device, _extra_files=extra,
                )
            except Exception as exc:
                raise PotentialError(
                    f"deepmd-batched requires a frozen TorchScript .pth model: {exc}"
                ) from exc
            if extra["data_modifier.pth"]:
                raise ValueError("deepmd-batched does not support exported data modifiers")
        is_script = isinstance(model, torch.jit.ScriptModule)
        if not (isinstance(model, deepmd_models.EnergyModel) or
                (is_script and model.original_name == "EnergyModel")):
            raise TypeError("deepmd-batched requires a standard DeePMD EnergyModel")
        if not model.do_grad_r("energy") or not model.do_grad_c("energy"):
            raise ValueError("deepmd-batched requires conservative energy-derived forces and virial")
        if model.get_dim_fparam() or model.get_dim_aparam():
            raise ValueError("deepmd-batched does not support frame or atomic model parameters")
        if model.has_chg_spin_ebd() or model.get_dim_chg_spin():
            raise ValueError("deepmd-batched does not support charge/spin-dependent models")
        precision_names = model.reverse_precision_dict
        for precision in (model.global_pt_float_precision, model.global_pt_ener_float_precision):
            if precision_names[precision] != "float64":
                raise ValueError("DeePMD global geometry and energy precision must be float64")
        # Native forward promotes input geometry to this precision, then returns
        # unreduced forces in the input dtype and reduced energy/virial in float64.
        # Preserve that model contract without casting any learned parameters.
        self.geometry_dtype = self.energy_dtype = self.virial_dtype = torch.float64
        definition = model.get_model_def_script()
        if definition:
            try:
                metadata = json.loads(definition)
            except (ValueError, TypeError) as exc:
                raise ValueError("DeePMD model definition metadata is not valid JSON") from exc
            self._check_precision_metadata(metadata)
        species = list(model.get_type_map())
        if (not species or len(species) != len(set(species))
                or any(symbol not in species for symbol in self.symbols)):
            raise ValueError("DeePMD model type_map must uniquely cover every system element")
        self.model = self._prepare_module(model)
        self._normalize_coord = region.normalize_coord
        self._types = torch.tensor([species.index(s) for s in self.symbols],
                                   dtype=torch.long, device=self.device)
        self._sel = tuple(int(value) for value in self.model.get_sel())
        self._nsel = sum(self._sel)
        if self._nsel < 1 or any(value < 0 for value in self._sel):
            raise ValueError("DeePMD model must have a positive trained neighbor selection")
        self._mixed_types = self.model.mixed_types()
        if not self._mixed_types and len(self._sel) != len(species):
            raise ValueError("DeePMD type-dependent sel must match the model type_map")
        self._type_count = len(species)
        self._selection = torch.tensor(self._sel, dtype=torch.long, device=self.device)
        self._offsets = self._selection.cumsum(0) - self._selection
        self._neighbors = neighbors.NeighborList(
            cutoff=float(self.model.get_rcut()), full_list=True, algorithm="cell_list",
            skin=0.0,
        )
        self.metadata = dict(
            self.metadata, neighbor_backend="vesin-cell-list",
            model_calls_per_batch=1, neighbor_selection="trained-nearest-sel",
            geometry_dtype="float64", energy_dtype="float64", virial_dtype="float64",
        )

    def _check_precision_metadata(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "precision":
                    aliases = {"double": "float64", "single": "float32", "default": "float64"}
                    normalized = aliases.get(item, item) if isinstance(item, str) else None
                    if normalized != str(self.dtype).removeprefix("torch."):
                        raise ValueError("DeePMD model definition precision disagrees with requested dtype")
                self._check_precision_metadata(item)
        elif isinstance(value, list):
            for item in value:
                self._check_precision_metadata(item)

    def _cell_tensor(self):
        if self._cached_cell is None:
            # Round model inputs to the public dtype first, as native forward
            # does, then cache the model's promoted geometry representation.
            self._cached_cell = super()._cell_tensor().to(dtype=self.geometry_dtype)
        return self._cached_cell

    def _check_boundary(self, pbc):
        if pbc.any() and not pbc.all():
            raise PotentialError("DeePMD supports full or no PBC; partial PBC is unsupported")

    def _sorted_ranks(self, groups, distances, group_count):
        """Sort by group then distance without a dense pair-distance matrix."""
        torch = self.torch
        order = torch.argsort(distances, stable=True)
        order = order[torch.argsort(groups[order], stable=True)]
        groups = groups[order]
        counts = torch.bincount(groups, minlength=group_count)
        starts = counts.cumsum(0) - counts
        ranks = torch.arange(len(order), device=self.device) - starts[groups]
        return order, ranks

    def _selection_ties(self, distances, ranks, capacity):
        # Only inspect the boundary where one equally distant neighbor would be
        # kept and the other dropped. Ties wholly within the selection are fine.
        torch = self.torch
        at_boundary = (ranks == capacity) & (ranks > 0)
        return torch.any((distances[1:] == distances[:-1]) & at_boundary[1:])

    def _sparse_inputs(self, positions, cell):
        torch = self.torch
        batch, atoms, _ = positions.shape
        slots = atoms * self._nsel
        # Ghost coordinates are stored once per selected directed edge. No
        # [B,N,Nimages] allocation, dynamic host copy, or host-built image grid.
        extended = positions.new_zeros((batch, atoms + slots, 3))
        extended[:, :atoms] = positions
        types = torch.full((batch, atoms + slots), -1, dtype=torch.long, device=self.device)
        types[:, :atoms] = self._types
        mapping = torch.zeros_like(types)
        mapping[:, :atoms] = torch.arange(atoms, device=self.device)
        nlist = torch.full((batch, slots), -1, dtype=torch.long, device=self.device)
        flags = self._pbc_tensor()
        selection_ties = []
        for frame in range(batch):
            i, j, shifts, vectors = self._neighbors.compute(
                positions[frame].detach(), cell.detach(), periodic=flags,
                quantities="ijSD", copy=True,
            )
            i, j = i.to(torch.long), j.to(torch.long)
            distance2 = (vectors * vectors).sum(-1)
            order, ranks = self._sorted_ranks(i, distance2, atoms)
            i, j, shifts, distance2 = i[order], j[order], shifts[order], distance2[order]
            selection_ties.append(self._selection_ties(distance2, ranks, self._nsel))
            selected = ranks < self._nsel
            i, j, shifts, distance2, ranks = (
                i[selected], j[selected], shifts[selected], distance2[selected], ranks[selected]
            )
            if self._mixed_types:
                target = i * self._nsel + ranks
            else:
                neighbor_type = self._types[j]
                groups = i * self._type_count + neighbor_type
                order, ranks = self._sorted_ranks(groups, distance2, atoms * self._type_count)
                i, j, shifts, distance2, neighbor_type = (
                    i[order], j[order], shifts[order], distance2[order], neighbor_type[order]
                )
                capacity = self._selection[neighbor_type]
                selection_ties.append(self._selection_ties(distance2, ranks, capacity))
                selected = ranks < capacity
                i, j, shifts, neighbor_type, ranks = (
                    i[selected], j[selected], shifts[selected], neighbor_type[selected], ranks[selected]
                )
                target = i * self._nsel + self._offsets[neighbor_type] + ranks
            ghost = atoms + target
            extended[frame, ghost] = positions[frame, j] + shifts.to(positions.dtype) @ cell
            types[frame, ghost] = self._types[j]
            mapping[frame, ghost] = j
            nlist[frame, target] = ghost
        if torch.stack(selection_ties).any().item():
            raise PotentialError(
                "DeePMD trained sel cuts through equidistant neighbors; the dense "
                "topk tie policy is unspecified, so this batch cannot be reproduced safely"
            )
        return extended, types, nlist.reshape(batch, atoms, self._nsel), mapping

    def _evaluate_tensor(self, positions):
        from ..torch_backend import TensorPESResult

        torch = self.torch
        with torch.enable_grad():
            q = (positions.detach() / self.length_factor).to(
                dtype=self.geometry_dtype,
            )
            cell = self._cell_tensor()
            if self.pbc.any():
                # Native DeePMD wraps in global precision before casting to a
                # descriptor's network dtype. Keep that order: large unwrapped
                # float32 coordinates otherwise lose neighbor displacements.
                # A single 3x3 cell broadcasts over all frames on the device.
                q = self._normalize_coord(q, cell)
            q.requires_grad_(True)
            extended, types, nlist, mapping = self._sparse_inputs(q, cell)
            strain = None
            if self.virial:
                strain = torch.eye(3, dtype=self.geometry_dtype, device=self.device).expand(
                    len(q), 3, 3,
                ).clone().requires_grad_(True)
                extended = torch.bmm(extended, strain.transpose(-1, -2))
            # Calling the atomic core avoids the upstream forward/forward_lower
            # dense neighbor preprocessing. It includes the exact trained energy
            # network, output bias, and atom/pair exclusion masks.
            outputs = self.model.atomic_model.forward_common_atomic(
                extended, types, nlist, mapping=mapping,
            )
            atomic_energy = outputs["energy"]
            if tuple(atomic_energy.shape) != (len(q), len(self.symbols), 1):
                raise PotentialError("DeePMD atomic energy must have shape [batch,atoms,1]")
            energy = atomic_energy.sum(dim=(1, 2), dtype=self.energy_dtype) * self.energy_factor
            differentiated = (q,) if strain is None else (q, strain)
            if not energy.requires_grad:
                raise PotentialError("DeePMD atomic energy must remain differentiable")
            gradients = torch.autograd.grad(energy.sum(), differentiated)
            return TensorPESResult(
                energy.detach(), (-gradients[0] / self.length_factor).to(self.dtype).detach(),
                None if strain is None else (-gradients[1]).detach(),
            )

    def close(self):
        if not self.closed:
            super().close()
            self.model = self._neighbors = None
            self._normalize_coord = None
            self._types = self._selection = self._offsets = None
