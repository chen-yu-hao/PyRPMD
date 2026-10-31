"""Illustrative reversible HER network; input coefficients are not predictions.

Run from any directory with an installed PyRPMD. CSV goes to stdout. The rate
ratios are constructed to obey equilibrium at U=0 and unit activities. These
simple Butler--Volmer-like coefficients are independent of the toy slab PES.
"""

import csv
import sys
import numpy as np

from pyrpmd.constants import KB_AU, EV_TO_AU
from pyrpmd.electrochemistry import HERRates, steady_state_her


def main():
    writer = csv.writer(sys.stdout)
    writer.writerow(["model_potential_V", "coverage_H", "H2_TOF_s", "anodic_current_A_m2", "coverage_balance_s"])
    for potential in np.linspace(-0.2, 0.2, 9):
        exponent = potential * EV_TO_AU / (2 * KB_AU * 300)
        rates = HERRates(
            volmer_forward_s=float(np.exp(-exponent)),
            volmer_reverse_s=float(np.exp(exponent)),
            heyrovsky_forward_s=float(0.1 * np.exp(-exponent)),
            heyrovsky_reverse_s=float(0.1 * np.exp(exponent)),
            tafel_forward_s=0.02, tafel_reverse_s=0.02,
        )
        flux = steady_state_her(rates, proton_activity=1.0, hydrogen_activity=1.0, site_density_mol_m2=2.5e-5)
        writer.writerow([potential, flux.coverage, flux.hydrogen_tof_s, flux.current_density_A_m2, flux.coverage_rate_s])


if __name__ == "__main__":
    main()
