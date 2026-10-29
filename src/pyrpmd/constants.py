"""Atomic-unit conversions from SciPy's CODATA physical-constant table.

All simulation arrays use bohr, electron masses, hartree and atomic time.
"""

from scipy.constants import physical_constants, k, N_A, electron_volt, c, h

HARTREE_J = physical_constants["Hartree energy"][0]
BOHR_M = physical_constants["Bohr radius"][0]
ATOMIC_TIME_S = physical_constants["atomic unit of time"][0]
ELECTRON_MASS_KG = physical_constants["electron mass"][0]
AMU_KG = physical_constants["atomic mass constant"][0]
BOHR_ANGSTROM = BOHR_M / 1e-10
ANGSTROM_TO_AU = 1.0 / BOHR_ANGSTROM
AMU_TO_AU = AMU_KG / ELECTRON_MASS_KG
FS_TO_AU = 1e-15 / ATOMIC_TIME_S
PS_TO_AU = 1e-12 / ATOMIC_TIME_S
KB_AU = k / HARTREE_J
KCAL_MOL_TO_AU = 4184.0 / N_A / HARTREE_J
EV_TO_AU = electron_volt / HARTREE_J
WAVENUMBER_TO_AU = h * c * 100.0 / HARTREE_J
AU_RATE_TO_CM3_S = BOHR_M**3 / ATOMIC_TIME_S * 1e6
AU_TIME_FS = 1.0 / FS_TO_AU
AU_ENERGY_KCAL_MOL = 1.0 / KCAL_MOL_TO_AU
