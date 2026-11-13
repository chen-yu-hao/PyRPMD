"""Equilibrium H3 umbrella sampler with Gaussian-prior ring moves.

This is a benchmark sampler, not a replacement for RPMD real-time dynamics.
For equal ordinary-H masses, Jacobi vectors R=q0-(q1+q2)/2 and r=q2-q1
have masses 2m/3 and m/2. The decoupled beadwise total center of mass is
integrated out; it cancels from all rate-PMF ratios. Every retained degree
of freedom is sampled with the physical ring-polymer equilibrium measure.

The zero-centroid spring action is m*P/(2*beta)*sum_j|x[j+1]-x[j]|^2.
Projected Gaussian increments with variance beta/(m*P) therefore generate
its exact periodic Gaussian reference. The pCN proposal preserves that
reference; its Metropolis ratio contains only beta*mean(V)+beta*W-log(fs).
Cartesian centroid shifts and rigid rotations preserve the spring action
and volume. Proposal settings are fixed before production. There is no
comparison-to-paper stopping rule and no numerical rate prefactor here.

pCN reference: Cotter et al., Statistical Science 28 (2013) 424-446,
DOI 10.1214/13-STS421, equations 4.8 and 4.11.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json


def free_ring(rng, shape, beta, mass):
    """Draw zero-centroid periodic paths; shape is (..., P, 3)."""
    beads = shape[-2]
    if beads < 1 or shape[-1] != 3 or beta <= 0 or mass <= 0:
        raise ValueError("positive beta/mass and (..., P, 3) shape required")
    delta = rng.normal(size=shape) * np.sqrt(beta / (mass * beads))
    delta -= delta.mean(axis=-2, keepdims=True)
    path = np.cumsum(delta, axis=-2)
    path -= path.mean(axis=-2, keepdims=True)
    return path


def spring_action(x, beta, mass):
    delta = x - np.roll(x, 1, axis=-2)
    return mass * x.shape[-2] / (2 * beta) * np.sum(delta * delta, axis=(-2, -1))


def cartesian(R, r):
    """Inverse equal-mass Jacobi map, with beadwise total COM zero."""
    return np.stack((2 * R / 3, -R / 3 - r / 2, -R / 3 + r / 2), axis=-2)


def jacobi(q):
    return q[..., 0, :] - (q[..., 1, :] + q[..., 2, :]) / 2, q[..., 2, :] - q[..., 1, :]


def coordinate_metric(R, r, beta, mass=MASS_AU, rinf=30.):
    """Vectorized H3 xi and flux metric, independently checked against PyRPMD."""
    R = np.asarray(R); r = np.asarray(r)
    nr = np.linalg.norm(r, axis=-1); nR = np.linalg.norm(R, axis=-1)
    d1 = R + r / 2; d2 = R - r / 2
    n1 = np.linalg.norm(d1, axis=-1); n2 = np.linalg.norm(d2, axis=-1)
    branch = n1 <= n2
    d = np.where(branch[..., None], d1, d2)
    nd = np.minimum(n1, n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        eR = R / nR[..., None]; er = r / nr[..., None]; ed = d / nd[..., None]
        s0 = rinf - nR; s1 = nr - nd; denominator = s0 - s1
        g0 = np.stack((-eR, eR / 2, eR / 2), axis=-2)
        g1 = np.stack((-ed, -er + np.where(branch[..., None], ed, 0.),
                        er + np.where(branch[..., None], 0., ed)), axis=-2)
        gradient = (-s1[..., None, None] * g0 + s0[..., None, None] * g1) / denominator[..., None, None]**2
        metric = np.sum(gradient * gradient, axis=(-2, -1)) / mass
        xi = s0 / denominator
        log_flux = .5 * np.log(metric / (2 * np.pi * beta))
        cosine = np.sum(R * r, axis=-1) / (nR * nr)
    # The audited reaction channel is the positive-denominator component.
    # Continuous umbrella dynamics cannot cross the xi pole; global moves
    # must not jump across it into a disconnected asymptotic channel.
    valid = ((nR > 1e-10) & (nr > 1e-10) & (nd > 1e-10)
             & (denominator > 1e-10) & np.isfinite(log_flux))
    return xi, log_flux, valid, np.stack((nR, nr, cosine), axis=-1)


def remaining_action(provider, R, r, beta, centers, stiffness, mass=MASS_AU, rinf=30.,
                     minimum_pair=.5, maximum_centroid_bond=8.):
    q = np.ascontiguousarray(cartesian(R, r))
    count, beads = q.shape[:2]
    evaluated = provider.evaluate(q.reshape(-1, 3, 3))
    energies = evaluated.energy.reshape(count, beads)
    valid_pes = np.all(evaluated.status.reshape(count, beads) == 0, axis=1)
    xi, logflux, valid_cv, geometry = coordinate_metric(R.mean(axis=1), r.mean(axis=1), beta, mass, rinf)
    base = beta * energies.mean(axis=1) - logflux
    action = base + .5 * beta * stiffness * (xi - centers)**2
    pair_minimum = np.minimum(np.linalg.norm(r,axis=-1),
        np.minimum(np.linalg.norm(R+r/2,axis=-1),np.linalg.norm(R-r/2,axis=-1))).min(axis=1)
    # Legacy BKMP2 has a nonphysical compact extrapolation. The explicit
    # physical-channel domain is independently checked by cutoff variation;
    # no unrestricted global BKMP2 partition integral is claimed.
    valid = (valid_pes & valid_cv & np.isfinite(action) & (pair_minimum>=minimum_pair)
             & (geometry[:,1]<=maximum_centroid_bond))
    geometry = np.column_stack((geometry,pair_minimum))
    return np.where(valid, action, np.inf), xi, base, energies.mean(axis=1), geometry


def random_rotations(rng, count, angle=.45, global_probability=.15):
    """Symmetric mixture of normal axis-angle and Haar SO(3) proposals."""
    axis = rng.normal(size=(count, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    theta = rng.normal(scale=angle, size=count)
    quat = np.column_stack((np.cos(theta/2), axis*np.sin(theta/2)[:, None]))
    global_moves = rng.random(count) < global_probability
    haar = rng.normal(size=(int(global_moves.sum()), 4))
    haar /= np.linalg.norm(haar, axis=1)[:, None]
    quat[global_moves] = haar
    w, x, y, z = quat.T
    return np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), axis=-1).reshape(count, 3, 3)


def initial_paths(centers, beads, rinf=30.):
    # Same collinear entrance construction as the production workflow, with
    # an equilibrium H2 length at the far end and continuous TS interpolation.
    # It is only a starting state and never enters an estimator as a sample.
    rlength = 1.4014719755049045 + np.clip(centers, 0, 1) * (1.7570000251-1.4014719755049045)
    # For R collinear and on the entrance branch: s1=1.5*r-R.
    radius = rinf*(1-centers) + 1.5*rlength*centers
    product = centers > 1
    forming = 1.7570000251 + np.clip((centers[product]-1)/.1, 0, 1)*(1.4014719755-1.7570000251)
    rlength[product] = (rinf*(centers[product]-1)+forming)/(1.5*centers[product]-.5)
    radius[product] = rlength[product]/2+forming
    R = np.zeros((len(centers), beads, 3)); r = np.zeros_like(R)
    R[:, :, 0] = radius[:, None]; r[:, :, 0] = rlength[:, None]
    return R, r


def provenance(library):
    paths = [Path(__file__), ROOT/'src/pyrpmd/cv.py', ROOT/'src/pyrpmd/bias.py',
             ROOT/'src/pyrpmd/constants.py', ROOT/'src/pyrpmd/analysis/fes.py',
             ROOT/'benchmarks/h3/pes.py', ROOT/'benchmarks/h3/classical_tst.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} | {
        'bkmp2_library': hashlib.sha256(Path(library).read_bytes()).hexdigest()}


def run(output, library, *, beads=128, seed=211013, warmup=5000, production=30000,
        centers=None, stiffness=30., centroid_R=.45, centroid_r=.085,
        alpha_R=.5, alpha_r=.3, rotation_angle=.45, swap_interval=10,
        checkpoint_interval=1000, diagnostic_interval=10, resume=False,
        minimum_pair=.5, maximum_centroid_bond=8., stop_after=None):
    output = Path(output).resolve()
    if min(beads, production, checkpoint_interval, diagnostic_interval) < 1 or warmup < 0:
        raise ValueError('invalid sample counts')
    if not 0 < alpha_R <= 1 or not 0 < alpha_r <= 1:
        raise ValueError('invalid fixed proposal parameters')
    if centers is None:
        centers = np.linspace(-.05, 1.05, 111)
    centers = np.asarray(centers, float)
    if centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('finite one-dimensional centers required')
    stiffness = np.broadcast_to(np.asarray(stiffness,float),centers.shape).copy()
    if not np.isfinite(stiffness).all() or np.any(stiffness<=0):
        raise ValueError('positive finite stiffness per window required')
    if not .4<=minimum_pair<=.7 or not 6<=maximum_centroid_bond<=12:
        raise ValueError('explicit physical-channel bounds are out of audited range')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('use a new output directory, or --resume for identical settings')
    output.mkdir(parents=True, exist_ok=True)
    beta = 1 / (KB_AU * 300.)
    settings = dict(schema='pyrpmd.h3.pcn-umbrella.v1', beads=beads, seed=seed,
        warmup=warmup, production=production, centers=centers.tolist(), stiffness=stiffness.tolist(),
        centroid_R=centroid_R, centroid_r=centroid_r, alpha_R=alpha_R, alpha_r=alpha_r,
        rotation_angle=rotation_angle, swap_interval=swap_interval,
        checkpoint_interval=checkpoint_interval, diagnostic_interval=diagnostic_interval,
        temperature_K=300., beta=beta, mass_au=MASS_AU, rinf_bohr=30.,
        minimum_bead_pair_bohr=minimum_pair,maximum_centroid_breaking_bond_bohr=maximum_centroid_bond,
        positive_coordinate_denominator=True,
        source_sha256=provenance(library), sampling_is_real_time_dynamics=False,
        gaussian_prior='zero-centroid Jacobi free rings; beadwise total COM analytically marginalized')
    settings_path = output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('settings/source identity changed; use a new run')
    write_json(settings_path, settings)
    before = settings['source_sha256']
    rng = np.random.default_rng(seed)
    R, r = initial_paths(centers, beads)
    n = len(centers); completed = 0
    samples = np.empty((production, n), dtype=float)
    diagnostics = np.empty(((production+diagnostic_interval-1)//diagnostic_interval, n, 7))
    accepted = np.zeros((6, n), dtype=np.int64)
    proposed = np.zeros_like(accepted)
    swaps = np.zeros((2, max(n-1, 0)), dtype=np.int64)
    lineage = np.arange(n)
    saved = output/'checkpoint.npz'
    if resume and saved.exists():
        with np.load(saved, allow_pickle=False) as old:
            R=old['R'].copy(); r=old['r'].copy(); completed=int(old['completed'])
            samples[:len(old['samples'])] = old['samples']
            diagnostics[:len(old['diagnostics'])] = old['diagnostics']
            accepted=old['accepted'].copy(); proposed=old['proposed'].copy(); swaps=old['swaps'].copy()
            lineage=old['lineage'].copy(); rng.bit_generator.state=json.loads(str(old['rng']))
    provider = BKMP2Potential(library)
    def evaluate(R,r):
        return remaining_action(provider,R,r,beta,centers,stiffness,
            minimum_pair=minimum_pair,maximum_centroid_bond=maximum_centroid_bond)
    action, xi, base, energy, geometry = evaluate(R,r)
    if not np.isfinite(action).all():
        raise ValueError('invalid initial/restart state')
    start = time.perf_counter()
    def save():
        count = max(0, completed-warmup)
        nd = (count+diagnostic_interval-1)//diagnostic_interval
        tmp = output/'checkpoint.tmp.npz'
        np.savez_compressed(tmp,R=R,r=r,completed=completed,samples=samples[:count],
            diagnostics=diagnostics[:nd],accepted=accepted,proposed=proposed,swaps=swaps,
            lineage=lineage,rng=np.array(json.dumps(rng.bit_generator.state)))
        tmp.replace(saved)
        write_json(output/'progress.json',dict(completed_sweeps=completed,
            production_sweeps=count,elapsed_seconds=time.perf_counter()-start,
            acceptance=(accepted/np.maximum(proposed,1)).mean(axis=1),
            xi_mean=xi.mean(),finite=True))
        print(json.dumps({'completed':completed,'production':count,
            'elapsed_seconds':time.perf_counter()-start,
            'acceptance':(accepted/np.maximum(proposed,1)).mean(axis=1).tolist()}),flush=True)
    try:
        limit=warmup+production if stop_after is None else min(warmup+production,completed+stop_after)
        while completed < limit:
            for move in range(6):
                if beads == 1 and move in (2,3):
                    continue
                Rp=R; rp=r
                if move == 0:
                    Rp=R+rng.normal(scale=centroid_R,size=(n,1,3))
                elif move == 1:
                    rp=r+rng.normal(scale=centroid_r,size=(n,1,3))
                elif move in (2,3):
                    x, alpha, mass = (R,alpha_R,2*MASS_AU/3) if move==2 else (r,alpha_r,MASS_AU/2)
                    center=x.mean(axis=1,keepdims=True)
                    candidate=center+np.sqrt(1-alpha*alpha)*(x-center)+alpha*free_ring(rng,x.shape,beta,mass)
                    if move==2: Rp=candidate
                    else: rp=candidate
                else:
                    rotation=random_rotations(rng,n,angle=rotation_angle)
                    if move==4: Rp=np.einsum('nij,npj->npi',rotation,R)
                    else: rp=np.einsum('nij,npj->npi',rotation,r)
                newaction,newxi,newbase,newenergy,newgeom=evaluate(Rp,rp)
                accept=np.log(rng.random(n)) < action-newaction
                proposed[move] += 1; accepted[move] += accept
                if Rp is not R: R[accept]=Rp[accept]
                if rp is not r: r[accept]=rp[accept]
                action[accept]=newaction[accept]; xi[accept]=newxi[accept]; base[accept]=newbase[accept]
                energy[accept]=newenergy[accept]; geometry[accept]=newgeom[accept]
            if swap_interval and (completed+1)%swap_interval==0 and n>1:
                parity=int(rng.integers(2)); left=np.arange(parity,n-1,2); right=left+1
                logratio=.5*beta*(
                    stiffness[left]*((xi[left]-centers[left])**2-(xi[right]-centers[left])**2)
                    +stiffness[right]*((xi[right]-centers[right])**2-(xi[left]-centers[right])**2))
                accept=np.log(rng.random(len(left))) < logratio
                swaps[0,left] += 1; swaps[1,left] += accept
                a=left[accept]; b=right[accept]
                for array in (R,r,xi,base,energy,geometry,lineage):
                    array[a],array[b]=array[b].copy(),array[a].copy()
                action=base+.5*beta*stiffness*(xi-centers)**2
            completed += 1
            if completed > warmup:
                index=completed-warmup-1; samples[index]=xi
                if index%diagnostic_interval==0:
                    radius=np.mean(np.sum((r-r.mean(axis=1,keepdims=True))**2,axis=-1),axis=1)
                    diagnostics[index//diagnostic_interval]=np.column_stack((geometry,energy,radius,lineage))
            if completed%checkpoint_interval==0:
                save()
        save()
    finally:
        provider.close()
    after=provenance(library)
    report=dict(schema=settings['schema'],completed=completed==warmup+production,completed_sweeps=completed,
        elapsed_seconds=time.perf_counter()-start,literature_reproduced=False,
        seed=seed,beads=beads,source_unchanged=before==after,
        checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        diagnostics_columns=['R_centroid_bohr','r_centroid_bohr','signed_cosine','minimum_bead_pair_bohr','mean_energy_hartree',
                             'r_ring_radius_squared_bohr2','lineage'],
        move_names=['R_centroid','r_centroid','R_pcn','r_pcn','R_rotation','r_rotation'],
        acceptance=(accepted/np.maximum(proposed,1)).mean(axis=1))
    write_json(output/'run-report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--beads',type=int,default=128)
    p.add_argument('--seed',type=int,default=211013)
    p.add_argument('--warmup',type=int,default=5000)
    p.add_argument('--production',type=int,default=30000)
    p.add_argument('--stiffness',type=float,default=30.)
    p.add_argument('--window-count',type=int,default=111)
    p.add_argument('--schedule',type=Path,help='JSON with centers and stiffness arrays; overrides uniform grid')
    p.add_argument('--checkpoint-interval',type=int,default=1000)
    p.add_argument('--swap-interval',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--minimum-pair',type=float,default=.5)
    p.add_argument('--stop-after',type=int)
    a=vars(p.parse_args());a['centers']=np.linspace(-.05,1.05,a.pop('window_count'))
    schedule=a.pop('schedule')
    if schedule is not None:
        grid=json.loads(schedule.read_text());a['centers']=grid['centers'];a['stiffness']=grid['stiffness']
    run(**a)


if __name__=='__main__':
    main()
