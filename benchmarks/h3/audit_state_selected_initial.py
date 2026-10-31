r"""Audit the 2021 H + H2 harmonic initial ensemble without propagation.

Reproduce the archived initial_audit.json from the repository root::

    PYTHONPATH=src:. python benchmarks/h3/audit_state_selected_initial.py \
        --output runs/state_selected_reproduction/initial_audit_reproduced.json \
        --samples 8192 4096 4096

Prerequisites are the built native PyRPMD/BKMP2 libraries and
runs/state_selected_reproduction/h2_reference.json, produced by
benchmarks/h3/h2_reference.py. The driver, sampler and PES versions must
match the hashes in the archived report for numerical reproducibility.
Numerical reductions are preserved; timestamps, elapsed times and added
execution/provenance metadata will differ.

The fixed cases, in --samples order, are (v, nominal total energy/eV)
(0, 0.55), (0, 0.90), (1, 1.10), all with P=32. Each case uses seed
20261031 and independent trajectory IDs starting at 900000. The imported
prepare_paper_state preserves the archived SeedSequence construction and
centroid bond rejection. No thermostat or propagation is invoked.

For each accepted state we remove BC's beadwise COM position and momentum
(equal H masses). An isolated BC energy is evaluated on BKMP2 with the
spectator H at laboratory z=-99 bohr. Both potential models use the same
well minimum as their energy zero. The primitive estimator is K + V - S,
where K=sum_beads,atoms(p**2/(2m))/P and
S=sum_beads,atoms(m*(P/beta)**2*(q_next-q)**2/2)/P.
Centroid energies are separate diagnostics, not this primitive estimator.
The actual launched full-PES energy is also evaluated to quantify residual
reactant interaction; no collision propagation is needed.

The unconditional finite-P harmonic mean is
sum_k omega**2/[beta*(omega**2+omega_k**2)] + v*omega,
with omega_k=2P/beta*sin(pi*k/P). Conditioning on the centroid bond changes
only the centroid distribution. Its two Gaussian tails are integrated
analytically for each excitation phase, then averaged over 16384 midpoint
phases. The reported low-variance mean is this conditional harmonic
expectation plus the measured, paired (V_BKMP2 - V_min - V_harmonic).
This is a coefficient-one control variate, not an energy adjustment to
the sampled trajectories. Its SEM is the paired potential-difference SEM.

All SEMs are over independent trajectories, never individual beads.
Primitive per-trajectory values can be broad or negative and must not be
interpreted as exact microcanonical energies or quantum-state assignments.
The embedded Figure 2 annotation records a prior visual source inspection;
it is not an automatic digitization or a runtime PDF dependency.

The numerical body below is retained from the executed inline audit.
The CLI/path wrapper does not alter its formulas, sampling order, batch
size, energy reductions or conditional-expectation quadrature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SAMPLES = (8192, 4096, 4096)


def run_audit(output=None, samples=DEFAULT_SAMPLES):
    """Run a fixed initial-state audit and write the JSON report.

    Relative output paths are resolved against the caller's working
    directory. A report is written after each completed case, then marked
    complete after the final case. An existing output file is replaced.
    """
    import json, math, time, hashlib, platform
    from pathlib import Path
    from dataclasses import asdict
    from datetime import datetime, timezone
    import numpy as np
    from scipy.special import ndtr
    from benchmarks.h3.state_selected_reproduction import Config, setup, prepare_paper_state, MASS, BOND, OMEGA
    from pyrpmd.constants import EV_TO_AU, WAVENUMBER_TO_AU
    root=ROOT
    samples=tuple(samples)
    if len(samples)!=3 or any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in samples):
        raise ValueError("samples must contain exactly three integers, each at least 2")
    output=(Path(output) if output is not None else root/'runs/state_selected_reproduction/initial_audit.json').expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    ID_OFFSET=900000
    CHUNK=256
    ref=json.loads((root/'runs/state_selected_reproduction/h2_reference.json').read_text())
    thresholds=[0.2,0.5,0.8,1.0,1.078]
    quantile_probs=[0,0.001,0.01,0.05,0.5,0.95,0.99,0.999,1]
    def stats(x):
        a=np.asarray(x,dtype=float)
        n=a.size; std=float(a.std(ddof=1)) if n>1 else None
        return {'n':int(n),'mean':float(a.mean()),'std':std,'sem':None if std is None else std/math.sqrt(n),'min':float(a.min()),'max':float(a.max()),'quantile_probabilities':quantile_probs,'quantiles':np.quantile(a,quantile_probs).tolist()}
    def analytic(config):
        beta=config.beta; P=config.beads; nu=config.v
        wk=2*P/beta*np.sin(np.arange(P)*np.pi/P)
        finite=float(np.sum(OMEGA**2/(beta*(OMEGA**2+wk**2)))+nu*OMEGA)
        continuum=OMEGA*(.5+nu+1/np.expm1(beta*OMEGA))
        phases=(np.arange(16384)+.5)*2*np.pi/16384
        mq=np.sqrt(2*nu/OMEGA)*np.cos(phases)
        mp=np.sqrt(2*nu*OMEGA)*np.sin(phases)
        sq=1/(np.sqrt(beta)*OMEGA)
        low=np.sqrt(MASS/2)*(config.dmin_bohr-BOND)
        neg=np.sqrt(MASS/2)*(-config.dmin_bohr-BOND)
        z=(low-mq)/sq; zn=(neg-mq)/sq
        density=np.exp(-z*z/2)/np.sqrt(2*np.pi)
        densityn=np.exp(-zn*zn/2)/np.sqrt(2*np.pi)
        accept=ndtr(-z)+ndtr(zn)
        q2=(mq**2+sq**2)*accept+sq*(mq+low)*density-sq*(mq+neg)*densityn
        p2=(mp**2+1/beta)*accept
        cc=float(.5*np.mean(p2+OMEGA**2*q2)/np.mean(accept))
        uc=1/beta+nu*OMEGA
        conditional=finite+cc-uc
        return {'finite_P_harmonic_primitive_unconditioned_eV':finite/EV_TO_AU,
          'infinite_P_harmonic_primitive_unconditioned_eV':float(continuum/EV_TO_AU),
          'zero_temperature_harmonic_label_eV':(nu+.5)*OMEGA/EV_TO_AU,
          'finite_P_harmonic_primitive_conditional_eV':conditional/EV_TO_AU,
          'harmonic_centroid_unconditioned_eV':uc/EV_TO_AU,
          'harmonic_centroid_conditional_eV':cc/EV_TO_AU,
          'predicted_acceptance_probability':float(np.mean(accept)),
          'conditioning_energy_shift_eV':float((conditional-finite)/EV_TO_AU),
          'conditional_formula':'Gaussian centroid Q/P plus random-phase kick; exact two-sided bond-norm cutoff, phase quadrature with 16384 equal bins; internal ring modes remain unconditional',
          'finite_P_formula':'sum_k omega^2/[beta*(omega^2+omega_k^2)] + v*omega; omega_k=2P/beta*sin(pi*k/P)'}
    report={'schema':'pyrpmd.h3.state-selected-initial-audit.v1','complete':False,
     'created_utc':datetime.now(timezone.utc).isoformat(),
     'purpose':'Initial-state diagnosis only; no propagation, parameter fitting, or scattering modification',
     'workload_policy':'Sample counts fixed before inspecting energies. Defaults 8192/4096/4096 reproduce the archived audit workload; seed and trajectory IDs are unchanged.',
     'independence':'SEM is over independent trajectory RNG streams, never over beads',
     'seed_and_ids':{'seed':20261031,'id_offset':ID_OFFSET,'case_sample_counts':list(samples)},
     'definitions':{
      'BC_rest_frame':'Remove beadwise BC coordinate and momentum COM; equal hydrogen masses. This exactly removes the launch velocity and translation.',
      'kinetic_bc':'mean_beads sum_BC |p_i|^2/(2m_i); fictitious ring momentum kinetic energy',
      'spring_bc':'mean_beads sum_BC m_i*(P/beta)^2*|q_(a+1,i)-q_(a,i)|^2/2',
      'primitive_bc_bkmp2':'K_BC + mean_beads V_BKMP2(isolated BC, spectator at lab z=-99 bohr) - Vmin - spring_BC',
      'primitive_bc_harmonic':'K_BC + mean_beads (mu*omega^2/2)*(signed_bond-BOND)^2 - spring_BC',
      'centroid_bc_energy_bkmp2':'centroid BC internal kinetic + V_BKMP2(BC centroid geometry, 99-bohr spectator)-Vmin',
      'centroid_bc_energy_harmonic':'centroid BC internal kinetic + (mu*omega^2/2)*(centroid_bond-BOND)^2',
      'switch_potential_delta':'mean_beads V_BKMP2(isolated BC)-Vmin - mean_beads V_harmonic; paired sample difference relative to the same minimum',
      'launch_residual_interaction':'mean_beads V_BKMP2(actual launched system) - mean_beads V_BKMP2(isolated BC)',
      'variance_reduced_mean':'Exact conditional harmonic expectation plus measured paired V_BKMP2-V_harm; unbiased control variate with coefficient 1 for this linear diatom ensemble',
      'warning':'Primitive estimators can be negative and very broad. Their per-sample distribution is not a microcanonical physical-energy distribution. Conditioning changes the expected initial energy.'},
     'paper_figure2':{'doi':'10.1002/qua.26447','file':'/tmp/pyrpmd-literature-reference/paper2021_publisher_rg.pdf','page':9,
      'visually_inspected':True,'x_axis':'time (fs), 0 to 1000','y_axis':'energy (cm^-1)',
      'ground_state_beta_au':[200,300,400],'excited_state_beta_au':[200,300],
      'harmonic_reference_cm_minus1':[2203.5,6610.5],
      'quantum_reference_description':'Dashed reference lines slightly below harmonic ground level and near 6340 cm^-1 for v1; exact plotted means not digitized.',
      'scope':'Figure averages 50000 initial configurations during propagation on full PES; current audit is t=0 only and mostly different beta.'},
     'provenance':{'python':platform.python_version(),'numpy':np.__version__,
      'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'audit_implementation':'Exact archived inline numerical implementation with CLI/path wrapper; formulas, RNG order, chunk size and reductions preserved.',
      'benchmark_script_sha256':hashlib.sha256((root/'benchmarks/h3/state_selected_reproduction.py').read_bytes()).hexdigest(),
      'sampler_sha256':hashlib.sha256((root/'src/pyrpmd/initial_states.py').read_bytes()).hexdigest(),
      'reference_json_sha256':hashlib.sha256((root/'runs/state_selected_reproduction/h2_reference.json').read_bytes()).hexdigest()},
     'cases':[]}
    started=time.perf_counter()
    for (nu,total),N in zip(((0,.55),(0,.90),(1,1.10)),samples):
        config=Config(v=nu,total_energy_ev=total,beads=32,samples=N,seed=20261031,id_offset=ID_OFFSET)
        _,prepared,dynamics,classifier=setup(config)
        potential=dynamics.potential
        P=config.beads; beta=config.beta; mu=MASS/2
        vmin=float(potential.radial([BOND,99.,99.])[0])
        cart_ref=float(potential.evaluate(np.array([[[0.,0.,-99.],[0.,0.,-BOND/2],[0.,0.,BOND/2]]])).energy[0])
        energies_raw={key:[] for key in ['primitive_bc_bkmp2','primitive_bc_harmonic','switch_potential_delta','centroid_bc_energy_bkmp2','centroid_bc_energy_harmonic','centroid_switch_delta','potential_bc_bkmp2','potential_bc_harmonic','kinetic_bc','spring_bc','centroid_bc_kinetic','launch_residual_interaction','primitive_total_system']}
        geometry_raw={key:[] for key in ['centroid_bond_bohr','minimum_bead_bond_bohr','maximum_bead_bond_bohr','mean_bead_bond_bohr','bead_bond_spread_bohr','initial_axial_bohr','impact_bohr']}
        short_beads={str(x):0 for x in thresholds}; short_trajectories={str(x):0 for x in thresholds}
        all_minima=[]; draws=0; max_draws=0; negative_signed_beads=0
        for start in range(0,N,CHUNK):
            m=min(CHUNK,N-start)
            fq=np.empty((m,P,3,3)); fp=np.empty_like(fq); launches=[]
            for j in range(m):
                state,launch=prepare_paper_state(config,prepared,ID_OFFSET+start+j)
                fq[j]=state.q; fp[j]=state.p; launches.append(launch)
                draws+=launch['preparation_draws']; max_draws=max(max_draws,launch['preparation_draws'])
            q=fq[:,:,1:,:].copy(); p=fp[:,:,1:,:].copy()
            q-=q.mean(axis=2,keepdims=True); p-=p.mean(axis=2,keepdims=True)
            bonds=q[:,:,1,:]-q[:,:,0,:]
            rv=bonds.mean(axis=1); rb=np.linalg.norm(rv,axis=1)
            axes=rv/rb[:,None]; signed=np.einsum('bpi,bi->bp',bonds,axes)
            distance=np.linalg.norm(bonds,axis=2)
            pbar=p.mean(axis=1)
            kinetic=np.sum(p*p,axis=(1,2,3))/(2*MASS*P)
            spring=MASS*(P/beta)**2*np.sum((np.roll(q,-1,axis=1)-q)**2,axis=(1,2,3))/(2*P)
            kc=np.sum(pbar*pbar,axis=(1,2))/(2*MASS)
            vh=.5*mu*OMEGA**2*np.mean((signed-BOND)**2,axis=1)
            vhc=.5*mu*OMEGA**2*(rb-BOND)**2
            iso=np.empty_like(fq); iso[:,:,0,:]=np.array([0.,0.,-99.]); iso[:,:,1:,:]=q
            ic=np.empty((m,3,3)); ic[:,0,:]=np.array([0.,0.,-99.]); ic[:,1:,:]=q.mean(axis=1)
            result=potential.evaluate(np.concatenate((iso.reshape(-1,3,3),ic,fq.reshape(-1,3,3)),axis=0))
            if np.any(result.status): raise RuntimeError('Nonzero PES status in initial audit')
            vi=result.energy[:m*P].reshape(m,P).mean(axis=1)-vmin
            vc=result.energy[m*P:m*P+m]-vmin
            vf=result.energy[m*P+m:].reshape(m,P).mean(axis=1)-vmin
            kt=np.sum(fp*fp,axis=(1,2,3))/(2*MASS*P)
            st=MASS*(P/beta)**2*np.sum((np.roll(fq,-1,axis=1)-fq)**2,axis=(1,2,3))/(2*P)
            values={'primitive_bc_bkmp2':kinetic+vi-spring,'primitive_bc_harmonic':kinetic+vh-spring,
              'switch_potential_delta':vi-vh,'centroid_bc_energy_bkmp2':kc+vc,
              'centroid_bc_energy_harmonic':kc+vhc,'centroid_switch_delta':vc-vhc,
              'potential_bc_bkmp2':vi,'potential_bc_harmonic':vh,'kinetic_bc':kinetic,'spring_bc':spring,
              'centroid_bc_kinetic':kc,'launch_residual_interaction':vf-vi,'primitive_total_system':kt+vf-st}
            for key,value in values.items(): energies_raw[key].extend((value/EV_TO_AU).tolist())
            minima=distance.min(axis=1)
            geom={'centroid_bond_bohr':rb,'minimum_bead_bond_bohr':minima,'maximum_bead_bond_bohr':distance.max(axis=1),
              'mean_bead_bond_bohr':distance.mean(axis=1),'bead_bond_spread_bohr':distance.std(axis=1),
              'initial_axial_bohr':np.array([x['initial_axial_bohr'] for x in launches]),'impact_bohr':np.array([x['b_bohr'] for x in launches])}
            for key,value in geom.items(): geometry_raw[key].extend(value.tolist())
            all_minima.extend(minima.tolist())
            negative_signed_beads+=int(np.count_nonzero(signed<0))
            for threshold in thresholds:
                short_beads[str(threshold)]+=int(np.count_nonzero(distance<threshold))
                short_trajectories[str(threshold)]+=int(np.count_nonzero(minima<threshold))
            if (start+m)%2048==0 or start+m==N:
                print(json.dumps({'case':config.label,'accepted':start+m,'target':N,'seconds':round(time.perf_counter()-started,1)}),flush=True)
        ae=analytic(config)
        energies={key:stats(value) for key,value in energies_raw.items()}
        for value in energies.values():
            value['mean_cm_minus1']=value['mean']*EV_TO_AU/WAVENUMBER_TO_AU
            value['sem_cm_minus1']=value['sem']*EV_TO_AU/WAVENUMBER_TO_AU
        measured=energies['primitive_bc_harmonic']; theory=ae['finite_P_harmonic_primitive_conditional_eV']
        short_mask=np.array(all_minima)<config.dmin_bohr
        shifts=np.array(energies_raw['switch_potential_delta'])
        vr=theory+energies['switch_potential_delta']['mean']
        target=ref['levels'][nu]['energy_above_well_minimum_eV']
        stratified=[]
        for label,mask in [('min_bead_bond < dmin',short_mask),('min_bead_bond >= dmin',~short_mask)]:
            stratified.append({'criterion':label,'count':int(mask.sum()),
              'mean_switch_eV':float(shifts[mask].mean()) if mask.any() else None,
              'contribution_to_total_mean_switch_eV':float(shifts[mask].sum()/N),
              'mean_primitive_harmonic_eV':float(np.array(energies_raw['primitive_bc_harmonic'])[mask].mean()) if mask.any() else None,
              'mean_primitive_bkmp2_eV':float(np.array(energies_raw['primitive_bc_bkmp2'])[mask].mean()) if mask.any() else None})
        case={'config':asdict(config),'beta_au':beta,'collision_energy_eV':config.collision_energy/EV_TO_AU,
          'accepted':N,'preparation_draws':draws,'preparation_rejections':draws-N,
          'rejection_fraction':(draws-N)/draws,'max_preparation_draws':max_draws,
          'reference_minimum_hartree':vmin,'cartesian_99bohr_reference_minus_radial_reference_hartree':cart_ref-vmin,
          'energies_eV':energies,'geometry':{key:stats(value) for key,value in geometry_raw.items()},
          'short_beads':{'bead_count':N*P,'trajectory_count':N,'count_by_threshold_bohr':short_beads,'trajectory_count_by_threshold_bohr':short_trajectories,'negative_signed_bond_beads':negative_signed_beads},
          'switch_stratified_by_minimum_bead_bond':stratified,
          'analytic_expectations':ae,
          'comparison':{'harmonic_conditional_mean_minus_theory_eV':measured['mean']-theory,
           'harmonic_conditional_z_score':(measured['mean']-theory)/measured['sem'],
           'numerical_BKMP2_quantum_level_eV':target,
           'mean_primitive_bkmp2_minus_quantum_level_eV':energies['primitive_bc_bkmp2']['mean']-target,
           'mean_primitive_bkmp2_minus_harmonic_label_eV':energies['primitive_bc_bkmp2']['mean']-ae['zero_temperature_harmonic_label_eV'],
           'variance_reduced_expected_primitive_bkmp2_eV':vr,
           'variance_reduced_expected_primitive_bkmp2_sem_eV':energies['switch_potential_delta']['sem'],
           'variance_reduced_minus_quantum_level_eV':vr-target,
           'variance_reduced_minus_harmonic_label_eV':vr-ae['zero_temperature_harmonic_label_eV']},
          'propagation_steps':dict(dynamics.propagation_stats)}
        report['cases'].append(case)
        report['wall_seconds']=time.perf_counter()-started
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'done':config.label,'rejection':case['rejection_fraction'],'harmonic_mean_eV':measured['mean'],'harmonic_theory_eV':theory,'harmonic_z':case['comparison']['harmonic_conditional_z_score'],'full_mean_eV':energies['primitive_bc_bkmp2']['mean'],'primitive_sem_eV':energies['primitive_bc_bkmp2']['sem'],'switch_mean_eV':energies['switch_potential_delta']['mean'],'switch_sem_eV':energies['switch_potential_delta']['sem'],'variance_reduced_full_mean_eV':vr}),flush=True)
    report['complete']=True
    report['wall_seconds']=time.perf_counter()-started
    report['provenance']['benchmark_script_sha256_at_end']=hashlib.sha256((root/'benchmarks/h3/state_selected_reproduction.py').read_bytes()).hexdigest()
    report['provenance']['source_unchanged_during_audit']=report['provenance']['benchmark_script_sha256']==report['provenance']['benchmark_script_sha256_at_end']
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'saved':str(output),'bytes':output.stat().st_size,'seconds':report['wall_seconds']}),flush=True)

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        help="JSON output (default: repository runs/state_selected_reproduction/initial_audit.json)",
    )
    parser.add_argument(
        "--samples", type=int, nargs=3, default=DEFAULT_SAMPLES,
        metavar=("V0_ET055", "V0_ET090", "V1_ET110"),
        help="Fixed accepted sample counts for the three cases (default: 8192 4096 4096)",
    )
    args = parser.parse_args(argv)
    if any(n < 2 for n in args.samples):
        parser.error("each sample count must be at least 2 to estimate a sample SEM")
    run_audit(output=args.output, samples=args.samples)


if __name__ == "__main__":
    main()
