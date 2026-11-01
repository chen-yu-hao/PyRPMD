"""Independent classical H + H2 transition-state quadrature on BKMP2.

This integrates the two equivalent geometrical transition-state surfaces
directly, without an umbrella, reaction-coordinate implementation, dynamics,
or free-energy estimator. It is not a quantum RPMD or recrossing calculation.
The integration is restricted to the physical reaction channel: legacy BKMP2
has unphysical extrapolation at some exceptionally compact triangles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import AU_RATE_TO_CM3_S, KB_AU

MASS_AU = 1837.1526474396187
SOURCE_DOI = "10.1016/j.cpc.2012.10.017"
PUBLISHED_CLASSICAL_TST = 1.56e-18


def quadrature(potential, *, temperature=300.0, mass=MASS_AU, order=256,
               domain=(0.5, 8.0), energy_origin=0.0):
    """Return the equal-mass, two-channel classical TST volume/time.

    Let x=r_AB, y=r_BC, z=r_AC. The internal Cartesian volume element is
    8*pi**2*x*y*z*dx*dy*dz. On one active surface x=y=r, the other forming
    distance satisfies r<=z<=2*r, and |grad(y-x)|_M**2=(2+(z/r)**2)/m.
    Including both channels and dividing by the reactant diatom integral
    4*pi*integral(r**2*exp(-beta*V_H2)*dr) leaves the prefactor 4*pi.
    Transform z=r*t to integrate on a rectangle, with Jacobian dz=r*dt.
    """
    if (isinstance(order, bool) or not isinstance(order, int) or order < 8
            or not np.isfinite([temperature, mass, energy_origin]).all()
            or temperature <= 0 or mass <= 0):
        raise ValueError("positive finite temperature/mass and integer order >= 8 required")
    if len(domain) != 2 or not np.isfinite(domain).all() or not 0.4 <= domain[0] < domain[1] < 99:
        raise ValueError("physical-channel radial domain must satisfy 0.4 <= lower < upper < 99 bohr")
    beta = 1.0 / (KB_AU * temperature)
    nodes, weights = roots_legendre(order)
    lower, upper = map(float, domain)
    r = (upper-lower)*nodes/2 + (upper+lower)/2
    wr = weights*(upper-lower)/2
    t, wt = nodes/2+1.5, weights/2
    rr, tt = np.meshgrid(r, t, indexing="ij")
    cosine = 1-tt*tt/2
    coordinates = np.zeros((order*order, 3, 3))
    coordinates[:, 0, 0] = rr.ravel()
    coordinates[:, 2, 0] = (rr*cosine).ravel()
    coordinates[:, 2, 1] = (rr*np.sqrt(1-cosine*cosine)).ravel()
    result = potential.evaluate(coordinates)
    if np.any(result.status) or not np.isfinite(result.energy).all():
        raise ValueError("PES failed on a transition-state quadrature point")
    transition_energies = result.energy.reshape(order, order)
    asymptote = np.zeros((order, 3, 3))
    asymptote[:, 0, 0] = -r/2
    asymptote[:, 1, 0] = r/2
    asymptote[:, 2, 2] = np.sqrt(99**2-(r/2)**2)
    result = potential.evaluate(asymptote)
    if np.any(result.status) or not np.isfinite(result.energy).all():
        raise ValueError("PES failed on a reactant quadrature point")
    with np.errstate(over="raise", invalid="raise"):
        numerator = np.sum(wr[:, None]*wt[None, :]*rr**4*tt
                           * np.exp(-beta*(transition_energies-energy_origin))
                           * np.sqrt((2+tt*tt)/(2*np.pi*beta*mass)))
        denominator = np.sum(wr*r*r*np.exp(-beta*(result.energy-energy_origin)))
    rate_au = float(4*np.pi*numerator/denominator)
    if not np.isfinite(rate_au) or rate_au <= 0:
        raise ValueError("nonpositive/nonfinite quadrature result")
    return {"order_per_axis": order, "domain_bohr": [lower, upper],
            "temperature_K": float(temperature), "equal_nuclear_mass_au": float(mass),
            "equivalent_channels": 2, "rate_au": rate_au,
            "classical_tst_cm3_molecule_s": rate_au*AU_RATE_TO_CM3_S,
            "transition_integral_scaled": float(numerator),
            "reactant_radial_integral_scaled": float(denominator),
            "energy_origin_hartree": float(energy_origin)}


def audit(library):
    potential = BKMP2Potential(library)
    origin = potential.radial([1.4014719755, 99.0, 99.0])[0]
    grid = [quadrature(potential, order=n, energy_origin=origin) for n in (64, 128, 192, 256)]
    domains = [quadrature(potential, domain=d, energy_origin=origin)
               for d in ((0.4, 8.0), (0.6, 8.0), (0.5, 6.0), (0.5, 10.0))]
    reference = grid[-1]["classical_tst_cm3_molecule_s"]
    grid_difference = abs(grid[-2]["classical_tst_cm3_molecule_s"]/reference-1)
    domain_difference = max(abs(row["classical_tst_cm3_molecule_s"]/reference-1) for row in domains)
    numerical_converged = grid_difference < 1e-7 and domain_difference < 1e-7
    # Diagnose the model extrapolation instead of silently including its
    # unphysical compact configurations in a global partition integral.
    compact = [{"distances_bohr": r, "energy_hartree": potential.radial(r)[0]}
               for r in ([0.2, 0.2, 0.3], [0.3, 0.3, 0.45], [0.4, 0.4, 0.6])]
    return {
        "schema": "pyrpmd.h3.classical-tst-quadrature.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "direct Gauss-Legendre quadrature of the two classical transition-state surfaces",
        "scope": "physical-reaction-channel classical TST only; no RPMD or recrossing simulation",
        "temperature_K": 300.0, "classical_tst_cm3_molecule_s": reference,
        "numerical_quadrature_converged": numerical_converged,
        "numerical_relative_tolerance": 1e-7,
        "grid_relative_change": grid_difference, "maximum_domain_relative_change": domain_difference,
        "grid_refinement": grid, "domain_checks": domains,
        "paper": {"doi": SOURCE_DOI, "table": "Table 2, H+H2, classical k_TST column",
                  "value_cm3_molecule_s": PUBLISHED_CLASSICAL_TST,
                  "relative_difference": reference/PUBLISHED_CLASSICAL_TST-1,
                  "reported_sampling_precision_about_percent": 5,
                  "precision_note": "Paper convergence statement; not a supplied statistical confidence interval."},
        "quantum_rpmd_validated": False, "thermal_rate_validated": False,
        "limitations": [
            "The input mass is the repository's neutral 1H atomic-mass convention.",
            "The spectator is held at equal 99-bohr distances in the reactant integral.",
            "No transmission coefficient is calculated; comparison is to k_TST, not k_classical or k_RPMD.",
            "The lower radial cutoff excludes unphysical short-range extrapolation of the legacy PES; this is not its unrestricted global canonical partition integral."],
        "compact_extrapolation_diagnostic": compact,
        "provenance": {"pes": potential.metadata, "build": potential.provenance,
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("numerical_quadrature_converged",
          "classical_tst_cm3_molecule_s", "grid_relative_change", "maximum_domain_relative_change", "paper")}, indent=2))
    raise SystemExit(0 if report["numerical_quadrature_converged"] else 1)


if __name__ == "__main__":
    main()
