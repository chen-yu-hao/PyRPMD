"""Exact MACE Cartesian second derivatives at the NumPy PES boundary.

The ASE implementation uses the upstream get_hessian and a reverse-over-
reverse force VJP for HVPs. Neither route finite differences forces, materializes
a full Hessian for HVP, changes inference precision, or changes PES identity.
Coordinates/directions are bohr; results are hartree/bohr**2 (or its action
on a direction in bohr). MACE's graph topology is fixed during differentiation.
"""

from __future__ import annotations

import numpy as np

from .adapters import ModelASEPotential
from .base import PotentialError, coordinates
from .python import BOHR_ANGSTROM, EV_HARTREE


def _direction(x, value):
    """Validate a physical Cartesian HVP direction without casting its dtype."""
    v = np.asarray(value)
    if v.dtype != np.dtype("float64") or v.shape != x.shape or not np.isfinite(v).all():
        raise PotentialError("HVP direction must be finite float64 [B,N,3], matching coordinates")
    return np.ascontiguousarray(v)


def _hessian_matrix(value, natoms):
    """Validate the upstream MACE Cartesian ordering, dtype and symmetry."""
    h = np.asarray(value)
    d = 3 * natoms
    if h.dtype != np.dtype("float64"):
        raise PotentialError("MACE Hessian must be float64; no implicit precision conversion")
    if h.shape in ((d, natoms, 3), (natoms, 3, natoms, 3)):
        h = h.reshape(d, d)
    if h.shape != (d, d) or not np.isfinite(h).all():
        raise PotentialError("MACE Hessian must be finite [3N,3N] Cartesian components")
    if not np.allclose(h, h.T, rtol=1e-8, atol=1e-12):
        raise PotentialError("MACE Hessian is not symmetric within float64 tolerance")
    return np.ascontiguousarray((h + h.T) * 0.5)


class MACEASEPotential(ModelASEPotential):
    """MACE ASE provider with exact Hessian and matrix-free HVP APIs.

    hessian(x) -> float64[B,3N,3N] in hartree/bohr**2.
    hessian_vector_product(x,v) -> float64[B,N,3], using a force VJP;
    it allocates no Cartesian Hessian. Each batch frame is independent.
    """

    def _mace_setup(self, x):
        if self.closed:
            raise PotentialError("mace provider is closed")
        q = coordinates(x)
        if q.shape[1] != len(self.atoms):
            raise PotentialError("MACE derivative atom count mismatch")
        calculator = self.atoms.calc
        if getattr(calculator, "model_type", None) != "MACE":
            raise PotentialError("MACE second derivatives require model_type='MACE'")
        if str(getattr(calculator, "default_dtype", None)) != "float64":
            raise PotentialError("MACE second derivatives require default_dtype='float64'")
        if getattr(calculator, "use_compile", False):
            raise PotentialError("MACE compiled/padded calculator Hessians are not supported")
        if getattr(calculator, "num_models", None) != 1:
            raise PotentialError("MACE second derivatives require exactly one energy model")
        energy_scale = float(getattr(calculator, "energy_units_to_eV", 1.0))
        length_scale = float(getattr(calculator, "length_units_to_A", 1.0))
        if (not np.isfinite(energy_scale) or energy_scale <= 0
                or not np.isfinite(length_scale) or length_scale <= 0):
            raise PotentialError("MACE energy/length scales must be positive finite values")
        conversion = EV_HARTREE * energy_scale * (BOHR_ANGSTROM / length_scale) ** 2
        return q, calculator, conversion

    def hessian(self, x):
        q, calc, factor = self._mace_setup(x)
        method = getattr(calc, "get_hessian", None)
        if not callable(method):
            raise PotentialError("MACECalculator.get_hessian is unavailable in the installed MACE version")
        matrices = []
        for index, frame in enumerate(q):
            try:
                self.atoms.set_positions(frame * BOHR_ANGSTROM)
                raw = method(atoms=self.atoms)
                matrices.append(_hessian_matrix(raw, len(self.atoms)) * factor)
            except PotentialError:
                raise
            except Exception as exc:
                raise PotentialError(f"MACE Hessian frame {index} failed: {exc}") from exc
        return np.stack(matrices)

    def hessian_vector_product(self, x, v):
        q, calc, factor = self._mace_setup(x)
        directions = _direction(q, v)
        try:
            import torch
            from mace.tools import torch_tools
        except ImportError as exc:
            raise PotentialError("MACE HVP requires torch and mace.tools.torch_tools") from exc
        if torch.is_inference_mode_enabled():
            raise PotentialError("inference_mode is incompatible with analytic MACE derivatives")
        if not all(callable(getattr(calc, name, None)) for name in
                   ("_atoms_to_batch", "_clone_batch")):
            raise PotentialError("MACE HVP requires the supported MACECalculator graph interface")
        models = getattr(calc, "models", None)
        if not isinstance(models, (tuple, list)) or len(models) != 1:
            raise PotentialError("MACE HVP requires exactly one accessible energy model")
        products = []
        for index, (frame, direction) in enumerate(zip(q, directions)):
            try:
                self.atoms.set_positions(frame * BOHR_ANGSTROM)
                with torch_tools.default_dtype(calc.default_dtype), torch.enable_grad():
                    batch = calc._clone_batch(calc._atoms_to_batch(self.atoms))
                    graph = batch.to_dict()
                    positions = graph.get("positions")
                    if (not isinstance(positions, torch.Tensor)
                            or positions.dtype != torch.float64
                            or tuple(positions.shape) != (len(self.atoms), 3)):
                        raise PotentialError("MACE HVP graph requires float64 [N,3] positions without padding")
                    result = models[0](graph, training=True, compute_force=True,
                                       compute_virials=False, compute_stress=False)
                    forces = result.get("forces") if isinstance(result, dict) else None
                    if (not isinstance(forces, torch.Tensor)
                            or forces.dtype != torch.float64
                            or tuple(forces.shape) != tuple(positions.shape)
                            or not forces.requires_grad):
                        raise PotentialError("MACE HVP requires differentiable float64 model forces")
                    seed = torch.as_tensor(direction, dtype=positions.dtype, device=positions.device)
                    derivative = torch.autograd.grad(
                        forces, positions, grad_outputs=seed,
                        retain_graph=False, create_graph=False, allow_unused=False,
                    )[0]
                    if derivative is None or derivative.dtype != torch.float64:
                        raise PotentialError("MACE HVP returned an invalid force derivative")
                    answer = (-derivative * factor).detach().cpu().numpy()
                    if answer.shape != frame.shape or not np.isfinite(answer).all():
                        raise PotentialError("MACE HVP returned nonfinite or incorrectly shaped values")
                    products.append(answer)
            except PotentialError:
                raise
            except Exception as exc:
                raise PotentialError(f"MACE HVP frame {index} failed: {exc}") from exc
        return np.ascontiguousarray(np.stack(products))
