"""Shared RPMDrate xi: native workflow replay plus an actual restricted quantum core.

The native smoke uses a synthetic free three-particle input, not a molecular
rate benchmark. The quantum core is an explicit fixed-transverse double well;
its finite-time Cfs kernel is compared to an independent Cartesian FD solver.
No isolated quantum transmission is substituted for RPMD kappa.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh_tridiagonal
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.system import System
from pyrpmd.potentials import PESResult
from pyrpmd.analysis.shared_rate_coordinate import SharedRateCoordinate, XiSlice
from pyrpmd.analysis.shared_quantum_flux import assemble_slice_side, kubo_flux_side
from pyrpmd.analysis.curvilinear_quantum import solve_curvilinear
from pyrpmd.workflows.shared_rate_quantum import run_shared_rate
from pyrpmd.constants import KB_AU


class RestrictedDoubleWell:
    """Physical Cartesian model used only as a coordinate/quantum kernel control."""
    def evaluate(self,q):
        x=q[:,1,0];force=np.zeros_like(q);force[:,1,0]=-160*x*(x*x-.49)
        return PESResult(40*(x*x-.49)**2,force)


def quantum_control(out,counts=(81,161,321),states=32):
    beta=2.;times=np.linspace(0,10,501);anchor=np.array([[-1.5,0,0],[0,0,0],[1.5,0,0.]])
    masses=np.full(3,2.)
    native=RPMDrateCoordinate([[0],[1,2]],masses,anchor,[[0,1]],[[1,2]],10.)
    shared=SharedRateCoordinate(native,beta=beta,xi_star=1.,pes_identity='explicit Cartesian double-well control',normalization_id='finite box Z; not an RPMD denominator')
    tangent=np.zeros_like(anchor);tangent[1,0]=-1.;path=XiSlice(shared,anchor,tangent,[-1.25,1.25]);pes=RestrictedDoubleWell()
    # Independently discretized Cartesian coordinate z: T=-1/(2m) d_z^2.
    nfd=2400;dz=2.5/(nfd+1);z=-1.25+dz*np.arange(1,nfd+1);hopping=1/(2*path.path_mass*dz*dz)
    E,U=eigh_tridiagonal(2*hopping+40*(z*z-.49)**2,np.full(nfd-1,-hopping),select='i',select_range=(0,2*states-1))
    h=U[z>0].T@U[z>0]
    extended=kubo_flux_side(E,h,times,beta=beta)
    exact=kubo_flux_side(E[:states],h[:states,:states],times,beta=beta)
    state_change=float(np.linalg.norm(exact['flux_side_per_partition']-extended['flux_side_per_partition'])/np.linalg.norm(extended['flux_side_per_partition']))
    xi0=shared.geometry(anchor-1.25*tangent)['xi'];xi1=shared.geometry(anchor+1.25*tangent)['xi']
    records=[];curves={'time_au':times,'fd_flux':exact['flux_side_per_partition'],'fd_energies':E}
    for n in counts:
        start=time.perf_counter();half=(n-1)//2
        nodes=np.r_[np.linspace(xi0,1.,half+1),np.linspace(1.,xi1,half+1)[1:]]
        qref=solve_curvilinear([nodes],path.geometry,lambda u:path.potential(u,pes),states=states)
        side=assemble_slice_side(qref,nodes,path)
        actual=kubo_flux_side(qref.energies,side,times,beta=beta)
        diff=actual['flux_side_per_partition']-exact['flux_side_per_partition']
        rec={'xi_nodes':len(nodes),'states':states,'first_two_energies':qref.energies[:2].tolist(),
             'doublet_gap':float(qref.energies[1]-qref.energies[0]),
             'low_energy_max_absolute_error':float(np.max(abs(qref.energies[:2]-E[:2]))),
             'flux_relative_L2':float(np.linalg.norm(diff)/np.linalg.norm(exact['flux_side_per_partition'])),
             'side_min_eigenvalue':float(np.linalg.eigvalsh(side).min()),'side_max_eigenvalue':float(np.linalg.eigvalsh(side).max()),
             'mass_orthogonality_error':qref.diagnostics['mass_orthogonality_error'],
             'reactant_population':actual['reactant_population'],'seconds':time.perf_counter()-start}
        curves[f'xi_flux_{n}']=actual['flux_side_per_partition'];records.append(rec)
        print('quantum control',rec,flush=True)
    report={'scope':'restricted physical Cartesian double well, reparameterized by actual native RPMDrate xi',
        'full_molecular_rate_test':False,'bath_coupling_implemented':False,'physical_converged':False,
        'shared_coordinate':shared.manifest(),'cartesian_fd_grid':nfd,'beta':beta,
        'FD_state_doubling_relative_flux_change':state_change,'fd_doublet_gap':float(E[1]-E[0]),'fd_ground':float(E[0]),'barrier':40*.49**2,
        'normalization':'Cfs_K/Z of a finite closed model; Cfs(0)=0; not RPMD kappa',
        'native_side':'S=(1-xi_star)*s0+xi_star*s1 > 0','records':records}
    np.savez_compressed(out/'quantum_control.npz',**curves)
    (out/'quantum_control.json').write_text(json.dumps(report,indent=2))
    return report


def native_control(out,input_path):
    cfg=json.loads(Path(input_path).read_text());system_record=cfg.pop('test_system')
    system=System(tuple(system_record['symbols']),np.array(system_record['masses']),np.array(system_record['geometry']),tuple(tuple(x) for x in cfg['reaction']['fragments']))
    cfg['temperature_K']=1/(KB_AU*cfg['beta'])
    def noop(reference):
        return {'active_rp':reference.full_flux,'active_quantum':reference.full_flux,'evidence':{}}
    result=run_shared_rate(cfg,system,out,quantum_provider=noop,bootstrap=64)
    native=json.loads((out/'rpmd/summary.json').read_text())
    with np.load(out/'quantum/shared_flux.npz') as z:
        discrepancy=float(np.max(abs(z['kappa_curve']-z['baseline_kappa_curve'])))
    receipt={'scope':'actual native PMF, RATTLE parents and paired children; NOOP provider checks transport only',
        'kappa_native':native['kappa'],'kappa_shared':result['kappa_estimate'],
        'curve_max_difference':discrepancy,'qtst_unchanged':result['k_qtst_cm3_s']==native['pmf']['k_qtst_cm3_s'],
        'primary_corrected_rate':result['rate_cm3_s'],'physical_converged':False,
        'native_children':native['child_trajectories'],'unmet_gates':result['unmet_gates']}
    (out/'control.json').write_text(json.dumps(receipt,indent=2));print('native control',receipt,flush=True)
    return receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--input',type=Path,default=Path(__file__).parent/'inputs/native_smoke.json')
    ap.add_argument('--skip-native',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    quantum_control(a.output)
    if not a.skip_native:native_control(a.output/'native_control',a.input)
