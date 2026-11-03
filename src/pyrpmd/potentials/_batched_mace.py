"""MACE inference on one disconnected graph containing all physical frames."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np

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

    def _graph(self, q):
        """Build one disconnected graph with owned neighbor tensors."""
        torch = self.torch
        if self.device.type == "cuda":
            caller = torch.cuda.current_stream(self.device)
            default = torch.cuda.default_stream(self.device)
            if caller != default:
                # Vesin 0.6.2 launches on the legacy default stream. Its
                # completion sync follows output writes, so it cannot protect
                # borrowed buffers from copies queued on another stream.
                # Keep topology inputs, dtype promotion and ownership copies
                # on its stream; inference still runs on the caller's stream.
                default.wait_stream(caller)
                q.record_stream(default)
                try:
                    with torch.cuda.stream(default):
                        graph = self._graph_on_current_stream(q)
                finally:
                    caller.wait_stream(default)
                for value in graph.values():
                    value.record_stream(caller)
                return graph
        return self._graph_on_current_stream(q)

    def _graph_on_current_stream(self, q):
        torch, frames, atoms = self.torch, len(q), len(self.symbols)
        positions = (q.detach() / self.length_factor).contiguous()
        cell, flags = self._cell_tensor(), self._pbc_tensor()
        indices, shifts = [], []
        # Vesin's native cell list is O(N + E), supports triclinic/partial PBC
        # and runs on CPU or CUDA without moving coordinates to the host.
        # Its API is per-system; only topology assembly loops over systems.
        # Inference and force/strain differentiation below are one whole batch.
        for frame in range(frames):
            pairs, shift = self._neighbors.compute(positions[frame], cell, flags,
                                                   quantities="PS", copy=False)
            # Own data before Vesin reuses its buffers. Adding the offset also
            # copies both index columns in one operation. The final cat makes
            # edge_index contiguous; shifts go directly to the model dtype.
            indices.append((pairs + frame * atoms).T)
            shifts.append(shift.to(dtype=self.dtype, copy=True))
        unit_shifts = torch.cat(shifts)
        graph = self._template(frames)
        graph.update(positions=positions.reshape(-1, 3),
                     edge_index=torch.cat(indices, dim=1),
                     unit_shifts=unit_shifts, shifts=unit_shifts @ cell,
                     cell=cell.expand(frames, 3, 3).reshape(-1, 3),
                     pbc=flags.expand(frames, 3))
        return graph

    def _evaluate_tensor(self, q):
        from ..torch_backend import TensorPESResult

        frames, atoms = len(q), len(self.symbols)
        graph = self._graph(q)
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

    def _second_derivatives(self, positions, direction=None):
        """Use second-order force autograd on one disconnected MACE graph.

        Compute a full Cartesian Hessian with 3N VJPs (one per coordinate
        shared across independent frames), or one VJP for a matrix-free
        Hessian-vector product. No force finite differences or dense HVP
        allocation. The graph neighbor list is fixed for each derivative.
        """
        from .base import coordinates

        raw_positions = np.asarray(positions)
        if raw_positions.dtype != np.dtype("float64"):
            raise PotentialError("MACE second-derivative positions must be float64; no implicit precision conversion")
        x = coordinates(raw_positions)
        if x.shape[1] != len(self.symbols):
            raise PotentialError("MACE second-derivative atom count mismatch")
        if self.closed:
            raise PotentialError("mace-batched provider is closed")
        torch = self.torch
        if self.dtype != torch.float64:
            raise PotentialError("MACE second derivatives require float64 model precision")
        if torch.is_inference_mode_enabled():
            raise PotentialError("inference_mode is unsupported for MACE second derivatives")
        if torch.is_autocast_enabled(self.device.type):
            raise PotentialError("autocast is unsupported for MACE second derivatives")
        v = None
        if direction is not None:
            raw = np.asarray(direction)
            if (raw.dtype != np.dtype("float64")
                    or raw.shape != x.shape or not np.isfinite(raw).all()):
                raise PotentialError("HVP direction must be finite float64 [B,N,3], matching coordinates")
            v = torch.as_tensor(raw, dtype=self.dtype, device=self.device)
        try:
            with torch.enable_grad():
                # _graph owns detached topology input; MACE activates gradients
                # on graph["positions"], so the position tensor is the VJP target.
                graph = self._graph(torch.as_tensor(x, dtype=self.dtype, device=self.device))
                point = graph["positions"]
                output = self.model(graph, training=True, compute_force=True,
                                    compute_virials=False, compute_stress=False)
                forces = output.get("forces") if isinstance(output, dict) else None
                frames, n, dimension = x.shape[0], x.shape[1], x.shape[1] * 3
                if (not isinstance(forces, torch.Tensor)
                        or forces.dtype != torch.float64
                        or tuple(forces.shape) != (frames * n, 3)
                        or not forces.requires_grad):
                    raise PotentialError("MACE second derivatives require differentiable float64 forces")
                if not torch.isfinite(forces).all().item():
                    raise PotentialError("MACE forces contain NaN/Inf")
                # The MACE forward takes positions as an input leaf and makes
                # them require grad in prepare_graph. The only reverse-mode
                # target is this flattened model position tensor.
                if v is not None:
                    product = torch.autograd.grad(
                        forces, point, grad_outputs=v.reshape(-1, 3),
                        create_graph=False, retain_graph=False, allow_unused=False,
                    )[0]
                    result = -product.reshape(frames, n, 3)
                else:
                    f = forces.reshape(frames, dimension)
                    rows = []
                    for component in range(dimension):
                        seed = torch.zeros_like(f)
                        seed[:, component] = 1.0
                        row = torch.autograd.grad(
                            f, point, grad_outputs=seed,
                            create_graph=False, retain_graph=component < dimension - 1,
                            allow_unused=False,
                        )[0]
                        rows.append(-row.reshape(frames, dimension))
                    result = torch.stack(rows, dim=1)
                factor = self.energy_factor / self.length_factor**2
                result = (result * factor).detach().cpu().numpy()
            if not np.isfinite(result).all():
                raise PotentialError("MACE second derivatives returned NaN/Inf")
            if v is None:
                if not np.allclose(
                    result, result.transpose(0, 2, 1), rtol=1e-8, atol=1e-12,
                ):
                    raise PotentialError("MACE Hessian is not symmetric within float64 tolerance")
                result = (result + result.transpose(0, 2, 1)) * 0.5
            return np.ascontiguousarray(result)
        except PotentialError:
            raise
        except Exception as exc:
            raise PotentialError(f"MACE second-derivative evaluation failed: {exc}") from exc

    def hessian(self, x):
        """float64[B,3N,3N] Cartesian PES Hessians in hartree/bohr**2."""
        return self._second_derivatives(x)

    def hessian_vector_product(self, x, v):
        """float64[B,N,3] exact Hessian-vector products, v in bohr."""
        return self._second_derivatives(x, v)

    def _reset_engine(self):
        self._batch_cache = None

    def close(self):
        super().close()
        self.model = self._neighbors = None
