"""Explicit potential references for a one-electron reservoir (atomic units)."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from ..constants import EV_TO_AU
from ..validation import finite_real

_finite_real = partial(
    finite_real,
    finite_message="{name} must be finite{and_sign}",
    bounds_message="{name} must be finite{and_sign}",
)


def _electron_bounds(value):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != (2,):
        raise ValueError("electron_bounds must contain two finite increasing numbers")
    lower, upper = map(float, array)
    if not np.isfinite(array).all() or lower >= upper or not np.isfinite(upper - lower):
        raise ValueError("electron_bounds must contain two finite increasing numbers")
    return lower, upper


@dataclass(frozen=True)
class ElectronReservoir:
    """A calibrated electron chemical potential, with no implicit SHE/RHE offset.

    ``mu(U) = mu_ref - e * (U - U_ref)``. Voltages use the *same* explicitly
    named reference. Energies are hartree and electron numbers are dimensionless
    (possibly excess numbers relative to a fixed reference). More positive U
    lowers the electron chemical potential. The calibration must match the
    number-dependent PES's energy convention.
    """

    potential_V: float
    reference_potential_V: float
    chemical_potential_at_reference_au: float
    potential_reference: str

    def __post_init__(self):
        for name in ("potential_V", "reference_potential_V", "chemical_potential_at_reference_au"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if not isinstance(self.potential_reference, str) or not self.potential_reference.strip():
            raise ValueError("potential_reference must explicitly name the voltage reference")
        if not np.isfinite(self.chemical_potential_au):
            raise ValueError("reservoir chemical potential overflow")

    @property
    def chemical_potential_au(self):
        return self.chemical_potential_at_reference_au - (
            self.potential_V - self.reference_potential_V
        ) * EV_TO_AU
