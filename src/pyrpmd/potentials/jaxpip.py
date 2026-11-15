"""Direct JAX GPU binding for fitted JaxPIP models.

The ONNX bridge is useful for a portable C ABI, but it necessarily copies the
coordinates through host memory.  This adapter is the device-resident route:
JaxPIP owns the compiled force graph, while the Torch workflow owns the ring
polymer state.  Torch and JAX exchange tensors with DLPack on the selected
device via :class:`~pyrpmd.potentials.jax_dlpack.JAXDLPackPotential`, so no
coordinate or force array is materialized on the host inside a force evaluation
and non-default CUDA streams remain synchronized.

The adapter accepts the checked JaxPIP fit artifacts used by ``pyrpmd.pesgen``.
A model directory may be supplied in place of ``basis`` and ``model``; its
checked fit manifest then supplies the two files and the model settings.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from ..constants import BOHR_ANGSTROM, EV_TO_AU
from .base import BackendUnavailableError, Potential, PotentialError
from .jax_dlpack import JAXDLPackPotential, _load_jax_runtime, _normalize_device_request


def _units(position_unit, energy_unit):
    if position_unit == "bohr":
        length = 1.0
    elif position_unit == "angstrom":
        length = BOHR_ANGSTROM  # model length units per bohr
    else:
        raise ValueError("JaxPIP position_unit must be bohr or angstrom")
    if energy_unit == "hartree":
        energy = 1.0
    elif energy_unit == "eV":
        energy = EV_TO_AU  # hartree per model energy unit
    else:
        raise ValueError("JaxPIP energy_unit must be hartree or eV")
    return length, energy


def _load_jaxpip_model(*, basis, checkpoint, kind, device):
    if kind not in ("linear", "nn"):
        raise ValueError("JaxPIP kind must be linear or nn")
    try:
        canonical_device = _normalize_device_request(device)
    except ValueError as error:
        raise ValueError("JaxPIP device must be cpu, cuda or cuda:<index>") from error
    try:
        import equinox  # noqa: F401
        from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork
    except ImportError as error:
        raise BackendUnavailableError(
            "the JaxPIP GPU route requires jax, equinox and jaxpip; "
            "install the pinned environment from tools/requirements-jaxpip-gpu.txt"
        ) from error
    jax, jnp, canonical_device, jax_device = _load_jax_runtime(
        canonical_device, backend_label="JaxPIP"
    )
    cls = PolynomialLinearModel if kind == "linear" else PolynomialNeuralNetwork
    with jax.default_device(jax_device):
        model = cls.from_file(basis_file=str(basis), model_file=str(checkpoint))
    if jnp.dtype(model.dtype) != jnp.float64:
        raise ValueError("JaxPIP production requires float64 model parameters")
    return jax, jnp, model, canonical_device, jax_device


class JaxPIPPotential(Potential):
    """Checked JaxPIP model delegating its DLPack bridge to :class:`JAXDLPackPotential`."""

    def __init__(self, basis, checkpoint, *, kind="linear", alpha=1.0,
                 position_unit="bohr", energy_unit="hartree", device="cuda",
                 atomic_numbers=None, charge=0, multiplicity=1,
                 max_frames_per_call=None, allow_unvalidated=False):
        self.basis = Path(basis).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not self.basis.is_file() or not self.checkpoint.is_file():
            raise FileNotFoundError("JaxPIP basis/checkpoint file is missing")
        self.kind, self.alpha = kind, float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("JaxPIP alpha must be positive and finite")
        self.position_unit, self.energy_unit = position_unit, energy_unit
        self.length, self.energy = _units(position_unit, energy_unit)
        if max_frames_per_call is not None:
            if (isinstance(max_frames_per_call, (bool, np.bool_))
                    or not isinstance(max_frames_per_call, (int, np.integer))
                    or max_frames_per_call < 1):
                raise ValueError("max_frames_per_call must be a positive integer")
            max_frames_per_call = int(max_frames_per_call)
        self.max_frames_per_call = max_frames_per_call
        if not isinstance(allow_unvalidated, (bool, np.bool_)):
            raise ValueError("allow_unvalidated must be boolean")
        self.allow_unvalidated = bool(allow_unvalidated)
        self.atomic_numbers = (
            None if atomic_numbers is None else tuple(int(z) for z in atomic_numbers)
        )
        self.charge, self.multiplicity = int(charge), int(multiplicity)

        jax, jnp, model, canonical_device, jax_device = _load_jaxpip_model(
            basis=self.basis, checkpoint=self.checkpoint, kind=kind, device=device,
        )
        self.device = canonical_device
        self._jax, self._jnp, self.model, self.jax_device = jax, jnp, model, jax_device
        if (self.atomic_numbers is not None
                and self.model.descriptor.basis_info.num_atoms != len(self.atomic_numbers)):
            raise ValueError("JaxPIP basis atom count does not match atomic_numbers")

        self._bridge = JAXDLPackPotential(
            self.model.get_energy_and_forces,
            device=self.device,
            position_unit=position_unit,
            energy_unit=energy_unit,
            batched=False,
            max_frames_per_call=self.max_frames_per_call,
            atomic_numbers=self.atomic_numbers,
            charge=self.charge,
            multiplicity=self.multiplicity,
            _provider_name="jaxpip-dlpack",
        )
        self.closed = False
        self.metadata = {
            **self._bridge.metadata,
            "provider": "jaxpip-dlpack",
            "jaxpip_version": importlib.metadata.version("jaxpip"),
            "allow_unvalidated": self.allow_unvalidated,
        }
        self._bridge.metadata.update(self.metadata)

    @classmethod
    def from_spec(cls, spec, system=None):
        spec = dict(spec)
        directory = spec.get("model_directory", spec.get("directory"))
        if directory is not None:
            directory = Path(directory).resolve()
            from ..pesgen.jaxpip_fit import load_fit
            manifest = load_fit(directory)
            if (not manifest.get("within_requested_test_bounds", False)
                    and not spec.get("allow_unvalidated", False)):
                raise ValueError(
                    "JaxPIP fit is outside its requested test bounds; set "
                    "potential.allow_unvalidated=true only for a documented pilot"
                )
            settings = manifest.get("settings", {})
            spec.setdefault("basis", str(directory / manifest["basis"]))
            spec.setdefault("checkpoint", str(directory / manifest.get("checkpoint", "model.eqx")))
            spec.setdefault("kind", settings.get("kind", "linear"))
            spec.setdefault("alpha", settings.get("alpha", 1.0))
            spec.setdefault("position_unit", settings.get("position_unit", "bohr"))
            spec.setdefault("energy_unit", settings.get("energy_unit", "hartree"))
            contract = manifest.get("contract", {})
            spec.setdefault("atomic_numbers", contract.get("atomic_numbers"))
            spec.setdefault("charge", contract.get("charge", 0))
            spec.setdefault("multiplicity", contract.get("multiplicity", 1))
        checkpoint = spec.get("checkpoint", spec.get("model"))
        if checkpoint is None or "basis" not in spec:
            raise ValueError("JaxPIP requires basis and checkpoint/model (or model_directory)")
        if system is not None:
            actual = tuple(int(z) for z in system.atomic_numbers)
            declared = spec.get("atomic_numbers", actual)
            if tuple(int(z) for z in declared) != actual:
                raise ValueError("JaxPIP atomic_numbers do not match System order")
            spec.setdefault("atomic_numbers", list(actual))
            spec.setdefault("charge", getattr(system, "charge", 0))
            spec.setdefault("multiplicity", getattr(system, "multiplicity", 1))
        return cls(basis=spec["basis"], checkpoint=checkpoint,
                   kind=spec.get("kind", "linear"), alpha=spec.get("alpha", 1.0),
                   position_unit=spec.get("position_unit", "bohr"),
                   energy_unit=spec.get("energy_unit", "hartree"),
                   device=spec.get("device", "cuda"),
                   atomic_numbers=spec.get("atomic_numbers"),
                   charge=spec.get("charge", 0), multiplicity=spec.get("multiplicity", 1),
                   max_frames_per_call=spec.get("max_frames_per_call"),
                   allow_unvalidated=spec.get("allow_unvalidated", False))

    def _check_open(self):
        if self.closed:
            raise PotentialError("JaxPIP potential is closed")

    def evaluate(self, positions):
        self._check_open()
        return self._bridge.evaluate(positions)

    def as_torch_pes(self):
        self._check_open()
        result = self._bridge.as_torch_pes()
        result.metadata.update(self.metadata)
        return result

    def warmup_torch(self, frames, atoms):
        """Compile the selected JAX batch shape before production sampling."""
        self._check_open()
        self._bridge.warmup_torch(frames, atoms)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._bridge.close()


__all__ = ["JaxPIPPotential"]
