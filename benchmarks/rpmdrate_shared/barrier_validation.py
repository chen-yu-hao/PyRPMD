"""Executed nonlinear-xi kinetic/flux validation, NOT a gas-phase rate benchmark.

A physical 1D sech^2 barrier, x in [-L,L], is reparameterized with the actual
RPMDrate scalar function for a three-atom constrained geometry. The physical
Hamiltonian is unchanged. Mass-lumped P1 FEM retains dx/dxi in kinetic and
measure. Finite-box Kubo flux is compared with independent analytical Eckart
transmission. No PMF is substituted for the physical barrier potential.

Default uses installed native pyrpmd.cv. --formula-oracle is an explicit,
recorded fallback for patch-only environments; it is never used in production.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from pyrpmd.analysis.rpmdrate_shared import SharedRPMDrateCoordinate
from pyrpmd.analysis.rpmdrate_quantum import kubo_flux_side,shared_side_operator


def run(output,formula_oracle=False,input_path=None):
    if formula_oracle:
        from coordinate_oracle import FormulaOracle as Coordinate
    else:
        from pyrpmd.cv import RPMDrateCoordinate as Coordinate
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    input_path=Path(input_path or Path(__file__).with_name('inputs')/'barrier.json')
    spec=json.loads(input_path.read_text())
    masses=np.asarray(spec['coordinate_atomic_masses'],float)
    heavy=float(spec['coordinate_geometry_heavy_position']);rinf=float(spec['coordinate_rinf'])
    beta=float(spec['beta']);V0=float(spec['V0']);a=float(spec['a']);m=float(spec['mass'])
    L=float(spec['box_half_width'])
    if not (len(masses)==3 and np.all(masses>0) and min(beta,V0,a,m,L)>0 and heavy>L and rinf>2*heavy and 2*m*V0/a**2>.25):
        raise ValueError('Invalid Eckart benchmark input')
    if m!=masses[1]:raise ValueError('The active physical mass must be the moving middle atom mass')
    reference=np.array([[-heavy,0,0],[0,0,0],[heavy,0,0]])
    coordinate=Coordinate([[0],[1,2]],masses,reference,[[0,1]],[[1,2]],rinf)
    shared=SharedRPMDrateCoordinate(coordinate,1.)
    c=masses[1]/masses[1:].sum();A=rinf-heavy-heavy*masses[2]/masses[1:].sum()
    def xi_of_x(x):return (A-c*x)/(A+(2-c)*x)
    def x_of_xi(xi):return A*(1-xi)/(c+(2-c)*xi)
    def dx_dxi(xi):return -2*A/(c+(2-c)*xi)**2
    def transmission(E):
        if E<0:return 0.
        z=np.pi*np.sqrt(2*m*E)/a
        if z>40:return 1.
        return np.sinh(z)**2/(np.sinh(z)**2+np.cosh(np.pi*np.sqrt(2*m*V0/a**2-.25))**2)
    analytic_integral=quad(lambda E:np.exp(-beta*E)*transmission(E),0,20,epsabs=1e-12)[0]/(2*np.pi)
    classical_integral=np.exp(-beta*V0)/(2*np.pi*beta)
    records=[];curves={};times=np.linspace(spec['time_start'],spec['time_stop'],spec['time_points'])
    for nodes in spec['grid_nodes']:
        xi=np.linspace(xi_of_x(L),xi_of_x(-L),nodes);h=np.diff(xi);x=x_of_xi(xi)
        # Cellwise stiffness and mass-lumped measure from Gauss quadrature.
        gauss,weights=np.polynomial.legendre.leggauss(4)
        localK=np.zeros(nodes-1);mass=np.zeros(nodes)
        for z,w in zip(gauss,weights):
            f=(1+z)/2;u=xi[:-1]+h*f;rho=abs(dx_dxi(u))
            localK+=.5*w*h*(.5/(m*rho*h*h))
            mass[:-1]+=.5*w*h*rho*(1-f);mass[1:]+=.5*w*h*rho*f
        K=np.diag(np.r_[localK,0]+np.r_[0,localK])-np.diag(localK,1)-np.diag(localK,-1)
        interior=slice(1,-1);mass=mass[interior];K=K[interior,interior]
        H=K/np.sqrt(mass[:,None]*mass[None])+np.diag(V0/np.cosh(a*x[interior])**2)
        geometries=np.broadcast_to(reference,(nodes-2,3,3)).copy();geometries[:,1,0]=x[interior]
        P=shared_side_operator(shared,geometries)
        kernel=kubo_flux_side(H,P,beta,times)
        Z=np.exp(kernel['log_partition']);Qr=kernel['reactant_population']*Z
        quantum_rate=kernel['rate_curve_per_atomic_time'];analytic_rate=analytic_integral/Qr
        reference_rate=classical_integral/Qr;lo,hi=spec['primary_window'];plateau=(times>=lo)&(times<=hi)
        # This finite-time comparison is a core test; no general plateau theorem.
        estimate=float(quantum_rate[plateau].mean())
        rec={'nodes':nodes,'beta_au':beta,'box_half_width':L,'quantum_flux_window':spec['primary_window'],
             'finite_box_reactant_partition':float(Qr),'quantum_rate_window':estimate,
             'independent_scattering_reference_rate':float(analytic_rate),
             'relative_flux_error':float(estimate/analytic_rate-1),
             'classical_flux_reference_rate':float(reference_rate),
             'quantum_flux_over_classical_positive_reference':float(estimate/reference_rate),
             'canonical_flux_at_t0':float(kernel['canonical_flux_side'][0]),
             'xi_roundtrip_max':float(np.max(abs(x_of_xi(xi)-x))),
             'nonlinear_coordinate_error_vs_native':float(max(abs(shared.evaluate(g).xi-u) for g,u in zip(geometries,xi[interior]))),
             'physical_converged':False,
             'fixed_window_sensitivity':{f'{lo}-{hi}':{
                 'relative_error':float(quantum_rate[(times>=lo)&(times<=hi)].mean()/analytic_rate-1),
                 'peak_to_peak_over_reference':float(np.ptp(quantum_rate[(times>=lo)&(times<=hi)])/analytic_rate)}
                 for lo,hi in spec['sensitivity_windows']}}
        records.append(rec);curves[f'rate_{nodes}']=quantum_rate
        print(json.dumps(rec),flush=True)
    # Independent P=1 flux-launch check, not a substitute for full native PMF.
    rng=np.random.default_rng(spec['seed']);v=np.sqrt(-2*np.log(rng.random(spec['P1_flux_launches']))/(beta*m));q=np.zeros(len(v));p=m*v
    dt=spec['P1_time_step'];check=[]
    for step in range(1,round(spec['time_stop']/dt)+1):
        f=2*a*V0*np.tanh(a*q)/np.cosh(a*q)**2;p+=.5*dt*f;q+=dt*p/m
        f=2*a*V0*np.tanh(a*q)/np.cosh(a*q)**2;p+=.5*dt*f
        if step%10==0:check.append(float(np.mean(q>0)))
    summary={'input':spec,'coordinate_definition':shared.definition(),'coordinate_implementation':'independent formula oracle' if formula_oracle else 'native pyrpmd.cv.RPMDrateCoordinate',
             'test_scope':'constrained 1D barrier quantum kernel in nonlinear xi; NOT a bimolecular rate or environment validation',
             'beta_au':beta,'barrier_height':V0,'barrier_inverse_width':a,'active_mass':m,
             'PMF_used_as_quantum_potential':False,'P1_independent_flux_launches':spec['P1_flux_launches'],
             'P1_min_transmission_on_test_window':min(check),'results':records,
             'native_run_rate_executed':False,'physical_converged':False}
    (output/'barrier_summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(output/'barrier_curves.npz',time_au=times,scattering_reference_rate=records[-1]['independent_scattering_reference_rate'],**curves)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--formula-oracle',action='store_true');parser.add_argument('--input',type=Path)
    args=parser.parse_args();run(args.output,args.formula_oracle,args.input)
