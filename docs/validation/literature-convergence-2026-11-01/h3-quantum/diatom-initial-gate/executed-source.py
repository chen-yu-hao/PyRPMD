"""Independent H2 rovibrational partition function on the pinned BKMP2 PES.

This solves the radial Schrodinger equation with a finite sine DVR, sums all
rotational degeneracies, and compares with the classical phase-space integral.
It is a continuum quantum *reactant* check, not a reaction rate calculation or
a finite-bead RPMD result. Identical-particle exchange and nuclear-spin weights
are omitted consistently with the distinguishable-nucleus benchmark.

The sine-basis transformation is explicit; see Colbert and Miller, J. Chem.
Phys. 96, 1982 (1992), DOI 10.1063/1.462100, for the DVR construction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, roots_legendre

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json


def sine_kinetic(points, lower, upper, mass):
    """Dirichlet radial kinetic operator at interior equispaced DVR nodes."""
    if (isinstance(points, bool) or not isinstance(points, (int, np.integer))
            or points < 8 or not 0 <= lower < upper or mass <= 0):
        raise ValueError('positive mass, >=8 interior nodes and 0<=lower<upper required')
    index = np.arange(1, points+1, dtype=float)
    radius = lower + (upper-lower)*index/(points+1)
    transform = np.sqrt(2/(points+1))*np.sin(np.pi*index[:, None]*index[None]/(points+1))
    eigenvalues = (np.pi*index/(upper-lower))**2/(2*mass)
    kinetic = (transform*eigenvalues[None])@transform.T
    return radius, kinetic


def spectrum(potential, *, mass, points=256, domain=(.4, 8.), jmax=24, levels=12):
    """Lowest radial levels for each J, including the centrifugal potential."""
    if jmax < 0 or not 1 <= levels <= points:
        raise ValueError('nonnegative jmax and 1<=levels<=points required')
    r, kinetic = sine_kinetic(points, *domain, mass)
    values = np.asarray(potential(r), float)
    if values.shape != r.shape or not np.isfinite(values).all():
        raise ValueError('radial potential returned invalid values')
    energies = np.empty((jmax+1, levels))
    for j in range(jmax+1):
        operator = kinetic.copy()
        operator.flat[::points+1] += values+j*(j+1)/(2*mass*r*r)
        energies[j] = eigh(operator, eigvals_only=True, subset_by_index=(0, levels-1),
                           check_finite=False)
    return energies


def quantum_log_partition(energies, beta, *, origin=0.):
    energies = np.asarray(energies, float)
    if energies.ndim != 2 or not np.isfinite(energies).all() or beta <= 0:
        raise ValueError('finite energies[J,v] and positive beta required')
    log_terms = np.log(2*np.arange(len(energies))+1)[:, None]-beta*(energies-origin)
    result = float(logsumexp(log_terms))
    return dict(log_partition=result,
                last_rotational_shell_fraction=float(np.exp(logsumexp(log_terms[-1])-result)),
                last_retained_radial_level_fraction=float(np.exp(logsumexp(log_terms[:, -1])-result)))


def classical_log_partition(potential, beta, mass, *, domain=(.4, 8.), order=384, origin=0.):
    nodes, weights = roots_legendre(order)
    lower, upper = domain
    r = lower+(nodes+1)*(upper-lower)/2
    wr = weights*(upper-lower)/2
    radial = logsumexp(np.log(wr)+2*np.log(r)-beta*(np.asarray(potential(r))-origin))
    return float(1.5*np.log(mass/(2*np.pi*beta))+np.log(4*np.pi)+radial)


def audit(library):
    provider = BKMP2Potential(library)
    beta = 1/(KB_AU*300.)
    mass = MASS_AU/2
    def potential(r):
        # Equal 99-bohr spectator distances give the same reactant definition
        # as the independent classical TS quadrature.
        r = np.asarray(r)
        return np.array([provider.radial([float(x), 99., 99.])[0] for x in r.flat]).reshape(r.shape)
    minimum = minimize_scalar(lambda r: float(potential(np.array(r))), bounds=(1.2, 1.6),
                              method='bounded', options={'xatol': 1e-13})
    origin = float(minimum.fun)
    rows = []
    for points, domain in [(128, (.4, 8.)), (192, (.4, 8.)), (256, (.4, 8.)),
                           (384, (.4, 8.)), (384, (.5, 8.)), (384, (.6, 8.)),
                           (384, (.4, 6.)), (384, (.4, 10.))]:
        energy = spectrum(potential, mass=mass, points=points, domain=domain)
        quantum = quantum_log_partition(energy, beta, origin=origin)
        classical = classical_log_partition(potential, beta, mass, domain=domain, origin=origin)
        rows.append(dict(points=points, domain_bohr=domain,
            energies_hartree=energy,
            ground_zero_point_hartree=float(energy[0, 0]-origin),
            fundamental_hartree=float(energy[0, 1]-energy[0, 0]),
            classical_log_partition=classical, quantum=quantum,
            log_quantum_classical_ratio=quantum['log_partition']-classical,
            quantum_classical_ratio=float(np.exp(quantum['log_partition']-classical))))
    reference = rows[3]['log_quantum_classical_ratio']
    grid_change = abs(rows[2]['log_quantum_classical_ratio']-reference)
    domain_change = max(abs(row['log_quantum_classical_ratio']-reference) for row in rows[4:])
    truncation = max(rows[3]['quantum']['last_rotational_shell_fraction'],
                     rows[3]['quantum']['last_retained_radial_level_fraction'])
    return dict(schema='pyrpmd.h3.diatom-partition.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        temperature_K=300., reduced_mass_au=mass,
        energy_origin_hartree=origin, equilibrium_radius_bohr=float(minimum.x),
        log_quantum_classical_ratio=reference, quantum_classical_ratio=float(np.exp(reference)),
        grid_log_change=grid_change, maximum_domain_log_change=domain_change,
        last_retained_shell_fraction=truncation,
        numerical_converged=bool(grid_change < 1e-7 and domain_change < 1e-7 and truncation < 1e-12),
        refinement=rows,
        limitations=['Continuum quantum reactant normalization only; no transition state or recrossing.',
            'No nuclear-spin/exchange statistical weights; all J have multiplicity 2J+1.',
            'Finite radial Dirichlet boxes and retained levels are checked by refinement, not assumed exact.',
            'A finite-bead ring-polymer comparison needs its own bead extrapolation.'],
        provenance={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'pes':provider.metadata, 'build':provider.provenance})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    report=audit(a.library);report['elapsed_seconds']=time.perf_counter()-started
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,report)
    print(json.dumps({k:report[k] for k in ['numerical_converged','quantum_classical_ratio',
          'grid_log_change','maximum_domain_log_change','elapsed_seconds']}),flush=True)
    raise SystemExit(0 if report['numerical_converged'] else 1)


if __name__=='__main__':
    main()
