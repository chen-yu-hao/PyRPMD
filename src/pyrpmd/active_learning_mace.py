"""Small, optional CPU MACE training hook for the active-learning reference loop.

This deliberately limited implementation trains fixed-composition, nonperiodic
molecules in float64. It uses one graph batch and one energy/force backward pass
per epoch. It is a reproducible reference hook, not MACE's production training
CLI or a foundation-model downloader. Public data and options use atomic units;
only the explicit MACE boundary converts to angstrom and eV.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
import os
import random

import numpy as np

from .constants import BOHR_ANGSTROM, EV_TO_AU
from .potentials.adapters import _symbols
from .potentials.base import BackendUnavailableError, PESResult, Potential, coordinates, validate_result
from .validation import finite_real


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _versions():
    try:
        return {name: version(name) for name in ("mace-torch", "torch", "e3nn", "ase", "numpy")}
    except PackageNotFoundError as error:
        raise BackendUnavailableError("MACE training needs the optional 'pyrpmd[mace]' dependencies") from error


def _optional():
    # Recent MACE releases set this process-wide flag on import. Our codec never
    # loads pickle models, and importing the hook must not change later loads.
    serialization_flag = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    try:
        import torch
        from ase import Atoms
        from mace import data, modules, tools
        from e3nn import o3
        from mace.tools.torch_geometric import Batch
    except ImportError as error:
        raise BackendUnavailableError("MACE training needs the supported optional 'pyrpmd[mace]' API") from error
    finally:
        if serialization_flag is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = serialization_flag
    return torch, Atoms, o3, data, modules, tools, Batch


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    value = finite_real(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


@contextmanager
def _runtime(torch, num_threads, seed=None):
    """Restore caller precision, threads and all RNGs, even after interruption."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_default_dtype(torch.float64)
            torch.set_num_threads(num_threads)
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                np.random.seed(seed % (2**32))
                random.seed(seed)
            yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def _build(config):
    torch, _, o3, _, modules, _, _ = _optional()
    from .system import _Z
    species = sorted({_Z[s] for s in config["symbols"]})
    model = modules.MACE(
        r_max=config["cutoff"] * BOHR_ANGSTROM,
        num_bessel=config["num_bessel"], num_polynomial_cutoff=5,
        max_ell=config["max_ell"],
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=config["interactions"], num_elements=len(species),
        hidden_irreps=o3.Irreps(f"{config['channels']}x0e"),
        MLP_irreps=o3.Irreps(f"{config['channels']}x0e"),
        atomic_energies=np.asarray(config["atomic_energies"]) / EV_TO_AU,
        avg_num_neighbors=1.0, atomic_numbers=species,
        correlation=config["correlation"], gate=torch.nn.functional.silu,
        radial_MLP=[config["radial_channels"], config["radial_channels"]],
        radial_type="bessel",
    )
    return model.to(device="cpu", dtype=torch.float64)


def _batch(q, config):
    _, Atoms, _, data, _, tools, Batch = _optional()
    from .system import _Z
    table = tools.AtomicNumberTable(sorted({_Z[s] for s in config["symbols"]}))
    graphs = [data.AtomicData.from_config(
        data.config_from_atoms(Atoms(config["symbols"], positions=frame * BOHR_ANGSTROM)),
        z_table=table, cutoff=config["cutoff"] * BOHR_ANGSTROM,
    ) for frame in q]
    return Batch.from_data_list(graphs)


def _check_model(model, torch):
    for value in list(model.parameters()) + list(model.buffers()):
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float64):
            raise ValueError("MACE reference models require CPU float64; conversion is not implicit")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("MACE model has nonfinite parameters or buffers")


class MACEModelPotential(Potential):
    """Batched finite-molecule potential produced by :class:`MACETrainer`."""

    metadata = {"provider": "active_learning_mace", "native_batch": True,
                "execution_device": "cpu", "precision": "float64",
                "thread_safety": "context_serial", "batch_independent": True,
                "trajectory_batch_safe": False, "periodic": False,
                "units": {"energy": "hartree", "length": "bohr"}}

    def __init__(self, model, config, *, provenance=None):
        torch, *_ = _optional()
        _check_model(model, torch)
        self._model = model.eval()
        self._config = json.loads(_json(config))
        self._provenance = json.loads(_json(provenance or {}))
        self._versions = _versions()

    @property
    def provenance(self):
        return json.loads(_json(self._provenance))

    def evaluate(self, positions):
        q = coordinates(positions)
        if q.shape[1] != len(self._config["symbols"]):
            raise ValueError("MACE reference model atom count differs from its fixed composition")
        torch, *_ = _optional()
        _check_model(self._model, torch)
        with _runtime(torch, self._config["num_threads"]), torch.enable_grad():
            output = self._model(_batch(q, self._config).to_dict(), training=False, compute_force=True)
        return validate_result(PESResult(
            output["energy"].detach().numpy() * EV_TO_AU,
            output["forces"].detach().numpy().reshape(q.shape) * (EV_TO_AU * BOHR_ANGSTROM),
        ), q.shape)


class MACEModelCodec:
    """Numeric state-dict codec; the driver owns hashing and atomic persistence.

    No pickle, arbitrary Python import, downloaded model or changed precision is
    admitted. Runtime versions are pinned to make resumed training explicit.
    """

    @property
    def identity(self):
        return {"codec": "PyRPMD.mace-state.v1", "versions": _versions()}

    def encode(self, model):
        if not isinstance(model, MACEModelPotential):
            raise TypeError("MACEModelCodec only supports MACEModelPotential")
        torch, *_ = _optional()
        _check_model(model._model, torch)
        if model._versions != _versions():
            raise ValueError("MACE model runtime versions changed")
        state = model._model.state_dict()
        names = sorted(state)
        arrays = {f"tensor_{index:05d}": state[name].detach().numpy().copy()
                  for index, name in enumerate(names)}
        return dict(arrays, metadata=np.asarray(_json({
            "schema": "PyRPMD.mace-state.v1", "config": model._config,
            "provenance": model._provenance, "versions": model._versions, "names": names,
        })))

    def decode(self, payload):
        try:
            metadata = np.asarray(payload["metadata"])
            if metadata.shape != () or metadata.dtype.kind not in "US":
                raise ValueError("metadata must be scalar JSON text")
            record = json.loads(str(metadata))
            if set(record) != {"schema", "config", "provenance", "versions", "names"}:
                raise ValueError("invalid metadata fields")
            if record["schema"] != "PyRPMD.mace-state.v1":
                raise ValueError("unsupported schema")
            if record["versions"] != _versions():
                raise ValueError("runtime versions differ from the saved model")
            checked = MACETrainer(**record["config"], seeds=(0, 1), epochs=1)._config
            if checked != record["config"]:
                raise ValueError("noncanonical model configuration")
            names = record["names"]
            if not isinstance(names, list) or any(not isinstance(name, str) for name in names) or names != sorted(set(names)):
                raise ValueError("invalid state inventory")
            if set(payload) != {"metadata", *(f"tensor_{index:05d}" for index in range(len(names)))}:
                raise ValueError("model payload inventory differs")
            torch, *_ = _optional()
            with _runtime(torch, checked["num_threads"], 0):
                model = _build(checked)
            expected = model.state_dict()
            if names != sorted(expected):
                raise ValueError("state keys differ from the architecture")
            state = {}
            for index, name in enumerate(names):
                value = np.asarray(payload[f"tensor_{index:05d}"])
                template = expected[name].numpy()
                if value.dtype != template.dtype or value.shape != template.shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid dtype, shape or value for {name}")
                state[name] = torch.from_numpy(value.copy())
            model.load_state_dict(state, strict=True)
            return MACEModelPotential(model, checked, provenance=record["provenance"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid MACE model payload: {error}") from error


class MACETrainer:
    """``train(dataset, previous_models) -> models`` for a small CPU committee.

    ``cutoff`` is bohr; ``atomic_energies`` are hartree in increasing atomic-number
    order. Loss is ``energy_weight * mean(dE_eV**2) + force_weight *
    mean(dF_eV_per_angstrom**2)`` over successful frames. Every call starts a fresh
    Adam optimizer; with ``fine_tune=True`` it starts from the previous weights.
    Epochs are a transactional callback: an interrupted call is replayed from
    its original inputs and seeds. This hook does not resume halfway through an
    optimizer epoch or certify physical accuracy.
    """

    def __init__(self, symbols, *, seeds=(17, 29), epochs=40, learning_rate=1e-3,
                 cutoff=6.0, channels=8, max_ell=1, correlation=2, interactions=2,
                 num_bessel=4, radial_channels=16, atomic_energies=None,
                 energy_weight=1.0, force_weight=1.0, fine_tune=True,
                 device="cpu", dtype="float64", num_threads=1):
        from .system import _Z
        symbols = list(_symbols(symbols))
        if device != "cpu" or dtype != "float64":
            raise ValueError("MACE reference trainer supports only device='cpu', dtype='float64'")
        if not isinstance(fine_tune, (bool, np.bool_)):
            raise ValueError("fine_tune must be boolean")
        self.seeds = tuple(_integer(seed, "seed", 0) for seed in seeds)
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds) or any(seed >= 2**63 for seed in self.seeds):
            raise ValueError("at least two distinct seeds in [0, 2**63) are required")
        self.epochs = _integer(epochs, "epochs")
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.energy_weight = _positive(energy_weight, "energy_weight")
        self.force_weight = _positive(force_weight, "force_weight")
        self.fine_tune = bool(fine_tune)
        species = sorted({_Z[s] for s in symbols})
        e0 = np.zeros(len(species)) if atomic_energies is None else np.asarray(atomic_energies)
        if e0.dtype.kind not in "fiu" or e0.shape != (len(species),) or not np.isfinite(e0).all():
            raise ValueError("atomic_energies must be finite values in increasing atomic-number order")
        self._config = dict(symbols=symbols, cutoff=_positive(cutoff, "cutoff"),
            channels=_integer(channels, "channels"), max_ell=_integer(max_ell, "max_ell"),
            correlation=_integer(correlation, "correlation"), interactions=_integer(interactions, "interactions"),
            num_bessel=_integer(num_bessel, "num_bessel"), radial_channels=_integer(radial_channels, "radial_channels"),
            atomic_energies=e0.astype(float).tolist(), device=device, dtype=dtype,
            num_threads=_integer(num_threads, "num_threads"))

    @property
    def identity(self):
        return json.loads(_json(dict(trainer="PyRPMD.mace-trainer.v1", architecture=self._config,
            seeds=self.seeds, epochs=self.epochs, learning_rate=self.learning_rate,
            energy_weight=self.energy_weight, force_weight=self.force_weight, fine_tune=self.fine_tune,
            optimizer="Adam", optimizer_restart="each_call", versions=_versions())))

    def initial_models(self):
        """Create deterministic untrained members to bootstrap an acquisition loop."""
        torch, *_ = _optional()
        models = []
        for seed in self.seeds:
            with _runtime(torch, self._config["num_threads"], seed):
                models.append(MACEModelPotential(_build(self._config), self._config,
                    provenance={"trainer": self.identity, "seed": seed, "trained": False}))
        return tuple(models)

    def __call__(self, dataset, previous_models):
        from .potentials.lifecycle import LabelledDataset
        if not isinstance(dataset, LabelledDataset):
            raise TypeError("MACETrainer requires a LabelledDataset")
        if not len(dataset) or np.any(dataset.status != 0):
            raise ValueError("MACETrainer requires nonempty, successful labels only")
        if dataset.positions.shape[1] != len(self._config["symbols"]):
            raise ValueError("training dataset atom count differs from the fixed composition")
        previous_models = tuple(previous_models)
        if previous_models and len(previous_models) != len(self.seeds):
            raise ValueError("previous MACE committee size differs from seeds")
        if self.fine_tune and any(not isinstance(model, MACEModelPotential) or model._config != self._config for model in previous_models):
            raise ValueError("fine-tuning requires matching MACEModelPotential architectures")
        torch, *_ = _optional()
        models = []
        for index, seed in enumerate(self.seeds):
            with _runtime(torch, self._config["num_threads"], seed):
                model = (deepcopy(previous_models[index]._model)
                         if self.fine_tune and previous_models else _build(self._config))
                model.train()
                batch = _batch(dataset.positions, self._config)
                target_e = torch.tensor(np.array(dataset.energy) / EV_TO_AU)
                target_f = torch.tensor(np.array(dataset.forces).reshape(-1, 3) / (EV_TO_AU * BOHR_ANGSTROM))
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                initial_loss = None
                for _ in range(self.epochs):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch.clone().to_dict(), training=True, compute_force=True)
                    loss = (self.energy_weight * (output["energy"] - target_e).square().mean()
                            + self.force_weight * (output["forces"] - target_f).square().mean())
                    if not torch.isfinite(loss):
                        raise ValueError("MACE training produced a nonfinite loss")
                    if initial_loss is None:
                        initial_loss = float(loss.detach())
                    loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                        raise ValueError("MACE training produced a nonfinite gradient")
                    optimizer.step()
                models.append(MACEModelPotential(model, self._config, provenance={
                    "trainer": self.identity, "seed": seed, "trained": True,
                    "dataset_sha256": dataset.sha256, "initial_loss": initial_loss,
                }))
        return tuple(models)
