"""Conservative local FAIR-Chem inference on one native AtomicData batch."""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import random
import re

import numpy as np

from ._batched_ml_base import _TensorBatchPotential
from .base import BackendUnavailableError, PotentialError
from .python import BOHR_ANGSTROM, EV_HARTREE
from ..system import _Z


class FairchemBatchedPotential(_TensorBatchPotential):
    """FAIR-Chem 2.25 local MLIPPredictUnit batching.

    The supported subset uses energy-gradient forces, full float32/float64,
    native graph generation and the general execution mode. Reduced-precision
    modes, direct-force models and automatic MOLE/compilation fallbacks are
    deliberately rejected. Checkpoints must already have the requested dtype.
    No pretrained weights are downloaded by this provider.
    """

    def __init__(self, symbols, model, *, task_name, device="cpu", dtype="float64",
                 cell=None, pbc=None, charge=0, multiplicity=1, virial=False):
        self._setup(
            symbols, "fairchem-batched", device=device, dtype=dtype, cell=cell,
            pbc=pbc, charge=charge, multiplicity=multiplicity, virial=virial,
            length_factor=1.0 / BOHR_ANGSTROM, energy_factor=EV_HARTREE,
        )
        # UMA/eSCN reduce_node_to_system sums native per-atom predictions
        # in float64 even when the neural network and forces are float32.
        self.energy_dtype = self.torch.float64
        self.metadata["energy_dtype"] = "float64"
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("fairchem-batched requires an explicit task_name")
        self.task_name = task_name
        if task_name != "omol" and (self.charge != 0 or self.multiplicity != 1):
            raise ValueError("fairchem tasks other than omol require charge=0 and multiplicity=1")
        if not -100 <= self.charge <= 100 or self.multiplicity > 100:
            raise ValueError("fairchem charge must be -100..100 and multiplicity 1..100")
        try:
            from fairchem.core.datasets.atomic_data import AtomicData
            from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        except ImportError as exc:
            raise BackendUnavailableError(
                "FAIR-Chem batching requires fairchem-core 2.25's local MLIPPredictUnit API"
            ) from exc
        self._AtomicData = AtomicData
        if isinstance(model, (str, Path)):
            path = Path(model)
            if not path.is_file():
                raise ValueError("fairchem-batched requires an existing local inference checkpoint")
            # InferenceSettings would otherwise silently cast a checkpoint's
            # weights at construction. Inspect the mmap-backed state first.
            checkpoint = self.torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True,
            )
            state = getattr(checkpoint, "model_state_dict", None)
            if not isinstance(state, dict):
                raise ValueError("fairchem model must be an MLIPInferenceCheckpoint")
            ema = getattr(checkpoint, "ema_state_dict", None)
            if not isinstance(ema, dict):
                raise ValueError("fairchem inference checkpoint must contain EMA weights")
            for weights in (state, ema):
                for name, value in weights.items():
                    if not isinstance(value, self.torch.Tensor) or not value.is_floating_point():
                        continue
                    # eSCN stores its fixed Wigner reference matrices in
                    # float64, including in native float32 checkpoints. They
                    # are constants, not learned network weights.
                    reference_matrix = (value.dtype == self.torch.float64 and
                                        re.fullmatch(r"(?:module\.)?backbone\.Jd_\d+", name))
                    if value.dtype != self.dtype and not reference_matrix:
                        raise ValueError("fairchem checkpoint dtype differs from requested dtype; "
                                         "the provider does not convert model weights")
            del state, ema, checkpoint
            settings = InferenceSettings(
                tf32=False, activation_checkpointing=False, merge_mole=False,
                compile=False, nitro=False, external_graph_gen=False,
                internal_graph_gen_version=2, execution_mode="general",
                base_precision_dtype=self.dtype,
                auto_add_default_untrained_tasks=False,
                predict_untrained_forces={task_name},
                predict_untrained_stress={task_name} if virial and self.pbc.any() else set(),
            )
            from fairchem.core.common.distutils import CURRENT_DEVICE_TYPE_STR
            old_device_type = os.environ.get(CURRENT_DEVICE_TYPE_STR)
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            # FAIR-Chem's constructor seeds global Python/NumPy/Torch RNGs and
            # writes its current-device environment variable. Isolate those
            # effects from dynamics and other providers; manual_seed_all affects
            # every CUDA device, so fork all of their RNG states as well.
            cuda_devices = list(range(self.torch.cuda.device_count()))
            device_context = (self.torch.cuda.device(self.device)
                              if self.device.type == "cuda" else nullcontext())
            try:
                with self.torch.random.fork_rng(devices=cuda_devices), device_context:
                    model = MLIPPredictUnit(
                        str(path), device=self.device.type, inference_settings=settings,
                    )
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                if old_device_type is None:
                    os.environ.pop(CURRENT_DEVICE_TYPE_STR, None)
                else:
                    os.environ[CURRENT_DEVICE_TYPE_STR] = old_device_type
        if not isinstance(model, MLIPPredictUnit):
            raise TypeError("fairchem model must be a local MLIPPredictUnit or inference checkpoint")
        self.predictor = model
        self._check_predictor()
        self._layout = None
        self.metadata.update(task_name=task_name, force_consistent_energy=True,
                             inference_settings="general-batch-ieee")

    def _check_predictor(self):
        torch = self.torch
        predictor = self.predictor
        settings = predictor.inference_settings
        if settings.base_precision_dtype != self.dtype:
            raise ValueError("fairchem predictor dtype differs from requested dtype")
        prohibited = ("tf32", "merge_mole", "compile", "nitro", "external_graph_gen")
        if any(getattr(settings, name, False) for name in prohibited):
            raise ValueError("fairchem batched predictor must disable TF32, MOLE merging, "
                             "compilation, Nitro and external graph generation")
        if (getattr(settings, "execution_mode", None) != "general"
                or settings.internal_graph_gen_version != 2):
            raise ValueError("fairchem batching requires execution_mode='general' and "
                             "internal_graph_gen_version=2")
        actual_device = torch.empty(0, device=predictor.device).device
        if actual_device != self.device:
            raise ValueError("fairchem predictor device differs from requested device; "
                             "configure its local CUDA rank explicitly")
        module = predictor.model.module
        config = getattr(module.backbone, "regress_config", None)
        if config is None or config.direct_forces or not config.forces:
            raise ValueError("fairchem batching requires conservative energy-gradient forces")
        if getattr(config, "hessian", False):
            raise ValueError("fairchem Hessian prediction is incompatible with frame batching")
        if self.virial and getattr(config, "direct_stress", False):
            raise ValueError("fairchem virial requires an energy-gradient stress model")
        if getattr(module.backbone, "always_use_pbc", False):
            raise ValueError("fairchem predictor must honor input PBC (always_use_pbc=False)")
        tasks = predictor.dataset_to_tasks.get(self.task_name)
        if not tasks or not {"energy", "forces"} <= {t.property for t in tasks}:
            raise ValueError("fairchem task must expose energy and forces")
        self._check_normalization(tasks, ("forces",))
        predictor.model = self._prepare_module(predictor.model)
        self._supports_single_atoms = module.supports_single_atoms
        self._check_boundary(self.pbc)

    def _check_normalization(self, tasks, properties):
        torch = self.torch
        energy_tasks = [task for task in tasks if task.property == "energy"]
        if len(energy_tasks) != 1:
            raise ValueError("fairchem task must have exactly one energy output")
        energy_scale = energy_tasks[0].normalizer.rmsd
        for prop in properties:
            derivatives = [task for task in tasks if task.property == prop]
            if len(derivatives) != 1:
                raise ValueError(f"fairchem task must have exactly one {prop} output")
            task = derivatives[0]
            if (task.element_references is not None
                    or not torch.equal(task.normalizer.rmsd, energy_scale)
                    or torch.count_nonzero(task.normalizer.mean).item()):
                raise ValueError(f"fairchem {prop} normalization is inconsistent with energy gradients")

    def _check_boundary(self, pbc):
        if (len(self.symbols) == 1 and not pbc.any()
                and not getattr(self, "_supports_single_atoms", True)):
            raise ValueError("this fairchem checkpoint does not natively support isolated atoms; "
                             "lookup-table inference fallback is disabled")
        predictor = getattr(self, "predictor", None)
        if predictor is not None and self.virial and pbc.any():
            tasks = predictor.dataset_to_tasks[self.task_name]
            if "stress" not in {task.property for task in tasks}:
                raise ValueError("periodic fairchem virial requires a stress output")
            # Rebinding a molecular provider to PBC enables stress for the
            # first time; validate its normalization before changing the cell.
            self._check_normalization(tasks, ("stress",))

    def _evaluate_tensor(self, q):
        from ..torch_backend import TensorPESResult
        torch = self.torch
        batch, natoms, _ = q.shape
        if self._layout is None or self._layout[0] != batch:
            numbers = torch.tensor([_Z[s] for s in self.symbols],
                                   dtype=torch.long, device=self.device)
            self._layout = (batch, dict(
                atomic_numbers=numbers.repeat(batch),
                natoms=torch.full((batch,), natoms, dtype=torch.long, device=self.device),
                batch=torch.arange(batch, device=self.device).repeat_interleave(natoms),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                cell_offsets=torch.empty((0, 3), dtype=self.dtype, device=self.device),
                nedges=torch.zeros(batch, dtype=torch.long, device=self.device),
                fixed=torch.zeros(batch * natoms, dtype=torch.long, device=self.device),
                tags=torch.zeros(batch * natoms, dtype=torch.long, device=self.device),
                charge=torch.full((batch,), self.charge, dtype=torch.long, device=self.device),
                spin=torch.full((batch,), self.multiplicity if self.task_name == "omol" else 0,
                                dtype=torch.long, device=self.device),
                sid=[""] * batch, dataset=[self.task_name] * batch,
            ))
        cell = self._cell_tensor()
        if self.cell is None:
            # Match ASE's complete_cell for molecules; PBC stays false.
            cell = torch.eye(3, dtype=self.dtype, device=self.device)
        positions = q / self.length_factor
        if self.pbc.any():
            fractional = positions @ torch.linalg.inv(cell)
            fractional = fractional - torch.floor(fractional) * self._pbc_tensor()
            positions = fractional @ cell
        data = self._AtomicData(
            pos=positions.reshape(-1, 3),
            cell=cell.expand(batch, 3, 3).contiguous(),
            pbc=self._pbc_tensor().expand(batch, 3).contiguous(),
            **self._layout[1],
        )
        # Public predictor API includes trained normalization and element
        # references. Calling the raw neural network would omit both.
        with torch.enable_grad():
            output = self.predictor.predict(data)
        energy, forces = output.get("energy"), output.get("forces")
        if (not isinstance(energy, torch.Tensor) or energy.shape not in ((batch,), (batch, 1))
                or not isinstance(forces, torch.Tensor) or forces.shape != (batch * natoms, 3)):
            raise PotentialError("fairchem must return energy[B] and forces[B*N,3]")
        energy = energy.reshape(batch).detach() * self.energy_factor
        forces = forces.reshape(batch, natoms, 3).detach() * (self.energy_factor / self.length_factor)
        virial = None
        if self.virial:
            if not self.pbc.any():
                virial = torch.einsum("bna,bnc->bac", forces, q)
            else:
                stress = output.get("stress")
                if not isinstance(stress, torch.Tensor) or stress.shape not in ((batch, 3, 3), (batch, 9)):
                    raise PotentialError("fairchem model does not provide the requested stress")
                virial = -stress.reshape(batch, 3, 3).detach() * torch.linalg.det(cell).abs() * self.energy_factor
        return TensorPESResult(energy, forces, virial)

    def close(self):
        super().close()
        self.predictor = None
        self._layout = None
