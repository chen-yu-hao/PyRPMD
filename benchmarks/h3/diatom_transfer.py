"""Independent finite-bead H2 partition function by radial transfer matrices.

The exact 3D free-particle heat kernel is expanded into angular momentum
channels before the primitive potential splitting. This is different from
splitting a centrifugal potential off a one-dimensional radial kinetic map:
the latter has a different finite-P error. The present trace is exactly the
primitive ring integral in the radial quadrature limit, with beadwise radial
walls. No Monte Carlo observations or published reaction rate enter it.
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
from scipy.special import ive, logsumexp, roots_legendre

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.diatom_partition import classical_log_partition, quantum_log_partition, spectrum
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json


def free_radial_kernel(radius, tau, mass, angular_momentum):
    """Reduced radial free kernel for measure dr, evaluated without overflow.

    K_J = mu*sqrt(r*r')/tau * I_(J+1/2)(mu*r*r'/tau)
          * exp[-mu*(r*r+r'*r')/(2*tau)].
    For J=0 the half-integer Bessel identity gives the exact half-line image
    kernel. See NIST DLMF 10.39.1 for I_(1/2), and SciPy's ive documentation
    for the exponential scaling used below.
    """
    r=np.asarray(radius,float)
    if (r.ndim!=1 or not len(r) or not np.isfinite(r).all() or np.any(r<=0)
            or tau<=0 or mass<=0 or not isinstance(angular_momentum,(int,np.integer))
            or angular_momentum<0):
        raise ValueError('positive radial nodes, tau, mass and nonnegative integer J required')
    product=r[:,None]*r[None]
    difference=r[:,None]-r[None]
    z=mass*product/tau
    return (mass/tau*np.sqrt(product)*ive(angular_momentum+.5,z)
            *np.exp(-mass*difference*difference/(2*tau)))


def primitive_partition(potential, *, beta, mass, beads, order=256,
                        domain=(.5,8.), jmax=24, origin=0.):
    """Trace of a symmetric primitive transfer matrix, retaining all eigenvalues."""
    if (not isinstance(beads,(int,np.integer)) or isinstance(beads,bool) or beads<1
            or beta<=0 or mass<=0 or order<16 or not 0<=domain[0]<domain[1]
            or jmax<0):
        raise ValueError('invalid primitive partition parameters')
    nodes,weights=roots_legendre(order)
    lower,upper=domain;r=lower+(nodes+1)*(upper-lower)/2;wr=weights*(upper-lower)/2
    values=np.asarray(potential(r),float)-origin
    if values.shape!=r.shape or not np.isfinite(values).all():
        raise ValueError('invalid radial potential values')
    tau=beta/beads
    factor=np.sqrt(wr)*np.exp(-tau*values/2)
    partial=[];negative_logs=[];minimum=0.;negative_count=0
    for j in range(jmax+1):
        matrix=free_radial_kernel(r,tau,mass,j)*factor[:,None]*factor[None]
        if beads==1:
            # The full trace is direct and cannot omit small spectral terms.
            trace=float(np.trace(matrix))
            logtrace=np.log(trace)
        else:
            eigenvalues=eigh(matrix,eigvals_only=True,check_finite=False)
            minimum=min(minimum,float(eigenvalues.min()))
            scale=max(float(eigenvalues.max()),np.finfo(float).tiny)
            if eigenvalues.min() < -1e-12*scale:
                raise ValueError('transfer kernel has negative eigenvalues beyond roundoff')
            negative=eigenvalues[eigenvalues<0]
            negative_count+=len(negative)
            if len(negative):
                negative_logs.append(np.log(2*j+1)+logsumexp(beads*np.log(-negative)))
            positive=eigenvalues[eigenvalues>0]
            logtrace=logsumexp(beads*np.log(positive))
        partial.append(np.log(2*j+1)+logtrace)
    logq=float(logsumexp(partial))
    classical=classical_log_partition(potential,beta,mass,domain=domain,order=max(384,order),origin=origin)
    negative_logbound=float(logsumexp(negative_logs)-logq) if negative_logs else None
    return dict(beads=int(beads),radial_order=int(order),domain_bohr=list(domain),jmax=int(jmax),
        free_kernel_width_bohr=float(np.sqrt(tau/mass)),maximum_radial_grid_spacing=float(np.diff(r).max()),
        log_partition=logq,classical_log_partition=classical,
        log_quantum_classical_ratio=logq-classical,quantum_classical_ratio=float(np.exp(logq-classical)),
        last_rotational_shell_fraction=float(np.exp(partial[-1]-logq)),
        minimum_transfer_eigenvalue=minimum,roundoff_negative_eigenvalues=negative_count,
        log_relative_absolute_negative_trace_bound=negative_logbound,
        relative_absolute_negative_trace_bound=float(np.exp(negative_logbound)) if negative_logbound is not None else 0.,
        partial_log_traces=partial)


def audit(library):
    provider=BKMP2Potential(library);beta=1/(KB_AU*300.);mass=MASS_AU/2
    def potential(r):
        r=np.asarray(r)
        return np.array([provider.radial([float(x),99.,99.])[0] for x in r.flat]).reshape(r.shape)
    minimum=minimize_scalar(lambda r:float(potential(np.array(r))),bounds=(1.2,1.6),method='bounded')
    origin=float(minimum.fun)
    rows=[primitive_partition(potential,beta=beta,mass=mass,beads=p,order=384,origin=origin)
          for p in (1,8,16,32,64,128,256)]
    grids=[primitive_partition(potential,beta=beta,mass=mass,beads=p,order=n,origin=origin)
           for p in (128,256) for n in (256,512)]
    domains=[primitive_partition(potential,beta=beta,mass=mass,beads=128,order=384,
                                domain=d,origin=origin)
             for d in ((.4,8.),(.6,8.),(.5,6.),(.5,10.))]
    angular=primitive_partition(potential,beta=beta,mass=mass,beads=128,order=384,jmax=32,origin=origin)
    continuum_energy=spectrum(potential,mass=mass,points=384,domain=(.5,8.))
    continuum=quantum_log_partition(continuum_energy,beta,origin=origin)['log_partition']
    continuum-=classical_log_partition(potential,beta,mass,domain=(.5,8.),origin=origin)
    for row in rows:
        row['log_difference_from_continuum']=row['log_quantum_classical_ratio']-continuum
    reference=rows[-2]['log_quantum_classical_ratio']
    errors=[abs(row['log_quantum_classical_ratio']-next(x['log_quantum_classical_ratio']
            for x in rows if x['beads']==row['beads'])) for row in grids]
    grid_change=max(errors)
    p1_error=abs(rows[0]['log_quantum_classical_ratio'])
    angular_change=abs(angular['log_quantum_classical_ratio']-reference)
    upper_change=max(abs(row['log_quantum_classical_ratio']-reference) for row in domains[2:])
    return dict(schema='pyrpmd.h3.finite-bead-diatom-partition.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),temperature_K=300.,reduced_mass_au=mass,
        energy_origin_hartree=origin,primary_beads=128,
        primary_quantum_classical_ratio=float(np.exp(reference)),
        primary_log_quantum_classical_ratio=reference,
        continuum_log_quantum_classical_ratio=continuum,
        primary_log_finite_bead_error=reference-continuum,
        maximum_radial_quadrature_log_change=grid_change,p1_classical_log_error=p1_error,
        angular_truncation_log_change=angular_change,upper_radial_box_log_change=upper_change,
        numerical_converged=bool(grid_change<1e-7 and p1_error<1e-7 and angular_change<1e-10
                                  and upper_change<1e-7),
        bead_refinement=rows,radial_quadrature_refinement=grids,
        wall_stress_checks=domains,angular_refinement=angular,
        method='Exact free 3D radial partial-wave kernel plus primitive potential factors and trace(A**P)',
        limitations=['Only reactant normalization is computed; this is not a reaction-rate reproduction.',
          'Finite-P beadwise radial bounds differ from a centroid upper bound; upper-box refinement quantifies the remote tail.',
          'The isolated equal-distance 99-bohr spectator limit differs slightly from finite Rinf=30 H3 normalization.',
          'All J have multiplicity 2J+1, with no exchange or nuclear-spin statistical factors.'],
        provenance={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'diatom_dvr_source_sha256':hashlib.sha256((ROOT/'benchmarks/h3/diatom_partition.py').read_bytes()).hexdigest(),
                    'pes':provider.metadata,'build':provider.provenance})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();report=audit(a.library)
    report['elapsed_seconds']=time.perf_counter()-started
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,report)
    print(json.dumps({k:report[k] for k in ['numerical_converged','primary_quantum_classical_ratio',
          'primary_log_finite_bead_error','maximum_radial_quadrature_log_change','elapsed_seconds']}),flush=True)
    raise SystemExit(0 if report['numerical_converged'] else 1)


if __name__=='__main__':main()
