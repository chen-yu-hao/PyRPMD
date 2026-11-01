"""Rebuild the audit's compact data table and static scientific figure.

Only saved calculations are read. No force evaluation or trajectory is run.
For the separate archive use --data-root RUNS --ui-root REPLAY; --prefix is
only needed for the original session's directory naming.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from pyrpmd.constants import AU_ENERGY_KCAL_MOL,AU_RATE_TO_CM3_S,AU_TIME_FS,KB_AU


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--ui-root',type=Path,required=True)
    parser.add_argument('--prefix',default='')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    labels=['smoke','pilot','long-a','long-b','classical']; rows=[];raw={};uis={}
    for label in labels:
        folder=args.data_root/(args.prefix+label)
        settings=read(folder/'settings.json');c=settings['config'];s=read(folder/'run/summary.json');p=s['pmf']
        raw[label]={'folder':folder,'summary':s,'config':c}
        classical=c['beads']==1;reference_q=1.56e-18 if classical else 2.51e-16
        reference_k=1.55e-18 if classical else 1.95e-16
        ui_file=args.ui_root/(label+'-ui')/'ui-report.json'
        ui=read(ui_file) if ui_file.exists() else None
        if ui is not None: uis[label]=ui
        barrier_ci=p['barrier_percentile_95_ci_hartree']
        qtst_ci=(ui['diagnostic_qtst_percentile_95_cm3_molecule_s'] if ui else
                 [p['reactant_prefactor_au']*AU_RATE_TO_CM3_S*np.exp(-c['beta']*v) for v in barrier_ci[::-1]])
        mbar_file=folder/'recovery-mbar.json'
        mb=read(mbar_file) if mbar_file.exists() else None
        row={'case':label,'temperature_K':c['temperature_K'],'beads':c['beads'],'seed':c['seed'],
             'equilibration_fs':c['sampling']['equilibration_steps']*c['sampling']['dt_au']*AU_TIME_FS,
             'production_fs':c['sampling']['production_steps']*c['sampling']['dt_au']*AU_TIME_FS,
             'streams':222,'saved_cv_values':222*c['sampling']['production_steps']//c['sampling']['sample_interval'],
             'qtst_cm3_molecule_s':p['k_qtst_cm3_s'],'paper_tst_or_qtst_cm3_molecule_s':reference_q,
             'qtst_ratio_to_paper':p['k_qtst_cm3_s']/reference_q,
             'ui_barrier_hartree':p['free_energy_barrier_hartree'],'mbar_barrier_hartree':mb['free_energy_barrier_hartree'] if mb else None,
             'kappa':s['kappa'],'kappa_standard_error':s['kappa_standard_error'],
             'paper_kappa':.994 if classical else .776,
             'rate_cm3_molecule_s':s['rate_estimate_cm3_s'],'paper_rate_cm3_molecule_s':reference_k,
             'rate_ratio_to_paper':s['rate_estimate_cm3_s']/reference_k,
             'diagnostic_qtst_95_low':qtst_ci[0],'diagnostic_qtst_95_high':qtst_ci[1],
             'diagnostic_rate_95_low':s['rate_percentile_95_ci_cm3_s'][0],
             'diagnostic_rate_95_high':s['rate_percentile_95_ci_cm3_s'][1],
             'bootstrap_block_status':s['rate_uncertainty']['pmf_block_status'],
             'first_half_qtst':ui['rows']['first_half']['qtst_candidate_cm3_molecule_s'] if ui else None,
             'second_half_qtst':ui['rows']['second_half']['qtst_candidate_cm3_molecule_s'] if ui else None,
             'parent_configurations':s['parent_configurations'],'child_trajectories':s['child_trajectories'],
             'physical_converged':False,'numerical_quadrature_converged':None,'workflow_status':s['status'],
             'summary_sha256':hashlib.sha256((folder/'run/summary.json').read_bytes()).hexdigest()}
        rows.append(row)
    quad=read(args.output/'classical-tst-quadrature.json')
    oracle_row={key:None for key in rows[0]}
    oracle_row.update(case='classical-quadrature',temperature_K=300.,beads=1,streams=0,saved_cv_values=0,
                      qtst_cm3_molecule_s=quad['classical_tst_cm3_molecule_s'],
                      paper_tst_or_qtst_cm3_molecule_s=quad['paper']['value_cm3_molecule_s'],
                      qtst_ratio_to_paper=quad['classical_tst_cm3_molecule_s']/quad['paper']['value_cm3_molecule_s'],
                      physical_converged=False,numerical_quadrature_converged=True,
                      workflow_status='numerically_converged_classical_tst_only')
    table_rows=rows+[oracle_row]
    record={'schema':'pyrpmd.h3.literature-audit-summary.v1','temperature_K':300.,
            'physical_rpmd_rate_validated':False,'interval_scope':'diagnostic bootstrap at fixed dividing surface; limited blocks and nonstationary sampling',
            'paper_doi':'10.1016/j.cpc.2012.10.017','rows':table_rows,
            'independent_classical_tst':{'cm3_molecule_s':quad['classical_tst_cm3_molecule_s'],
                 'paper_cm3_molecule_s':quad['paper']['value_cm3_molecule_s'],
                 'relative_difference':quad['paper']['relative_difference'],
                 'numerical_quadrature_converged':quad['numerical_quadrature_converged'],
                 'thermal_rate_validated':False},
            'independent_seed_rate_ratio_a_over_b':rows[2]['rate_cm3_molecule_s']/rows[3]['rate_cm3_molecule_s'],
            'render_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'compact-results.json').write_text(json.dumps(record,indent=2)+'\n')
    with (args.output/'compact-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(table_rows)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    colors={'pilot':'#777777','long-a':'#285b9a','long-b':'#c56a2d','classical':'#258474'}
    names={'pilot':'P=128, 100 fs pilot','long-a':'P=128, 1 ps, seed A','long-b':'P=128, 1 ps, seed B','classical':'P=1, 1 ps control'}
    ax=axes[0,0]
    for label in ['pilot','long-a','long-b']:
        if label in uis:
            a=np.genfromtxt(args.ui_root/(label+'-ui')/'ui-curves.csv',delimiter=',',names=True)
            ax.plot(a['xi'],a['all_hartree']*AU_ENERGY_KCAL_MOL,color=colors[label],label=names[label])
            ax.fill_between(a['xi'],a['bootstrap_p025_hartree']*AU_ENERGY_KCAL_MOL,
                            a['bootstrap_p975_hartree']*AU_ENERGY_KCAL_MOL,color=colors[label],alpha=.12)
        else:
            a=np.genfromtxt(raw[label]['folder']/'run/rate_pmf.csv',delimiter=',',names=True)
            ax.plot(a['xi'],(a['free_energy_hartree']-a['free_energy_hartree'][0])*AU_ENERGY_KCAL_MOL,
                    color=colors[label],label=names[label],ls='--')
    pref=raw['long-a']['summary']['pmf']['reactant_prefactor_au']*AU_RATE_TO_CM3_S
    paper_barrier=-np.log(2.51e-16/pref)*KB_AU*300.
    ax.scatter([1],[paper_barrier*AU_ENERGY_KCAL_MOL],s=95,marker='*',color='black',zorder=6,label='Paper QTST mapped to barrier')
    ax.set(title='A  Quantum PMF: sampling still matters',xlabel='Reaction coordinate ξ',ylabel='W(ξ) − W(0) / kcal mol⁻¹',xlim=(0,1.04))
    ax.legend(loc='upper left',fontsize=8)
    ax=axes[0,1]
    a=np.genfromtxt(args.ui_root/'classical-ui/ui-curves.csv',delimiter=',',names=True)
    for key,ls,name in [('all_hartree','-','Full 1 ps'),('first_half_hartree','--','First 0.5 ps'),('second_half_hartree',':','Second 0.5 ps')]:
        ax.plot(a['xi'],a[key]*AU_ENERGY_KCAL_MOL,color=colors['classical'],ls=ls,label=name)
    oracle_barrier=-np.log(quad['classical_tst_cm3_molecule_s']/pref)*KB_AU*300.
    ax.scatter([1],[oracle_barrier*AU_ENERGY_KCAL_MOL],s=95,marker='*',color='black',zorder=6,label='Independent TST mapped to barrier')
    ax.set(title='B  Classical control also drifts',xlabel='Reaction coordinate ξ',ylabel='W(ξ) − W(0) / kcal mol⁻¹',xlim=(0,1.04))
    ax.legend(loc='upper left',fontsize=8)
    ax=axes[1,0]
    chosen=rows[1:];ratio=[r['qtst_ratio_to_paper'] for r in chosen]
    for i,r in enumerate(chosen):
        y=r['qtst_ratio_to_paper'];low=r['diagnostic_qtst_95_low']/r['paper_tst_or_qtst_cm3_molecule_s'];high=r['diagnostic_qtst_95_high']/r['paper_tst_or_qtst_cm3_molecule_s']
        ax.errorbar(i,y,yerr=[[y-low],[high-y]],fmt='o',color=colors[r['case']],capsize=4)
    ax.scatter([4],[quad['classical_tst_cm3_molecule_s']/1.56e-18],marker='*',color='black',s=100)
    ax.axhline(1,color='black',ls='--',lw=1)
    ax.set_yscale('log');ax.set_xticks(range(5),['128-bead\npilot','128-bead\nseed A','128-bead\nseed B','1-bead\ncontrol','Classical\nquadrature'])
    ax.set(title='C  TST factor versus its matching paper value',ylabel='Calculated TST factor / paper TST factor',ylim=(1e-8,2e2))
    ax.text(.03,.04,'Bars: diagnostic bootstrap percentiles, not established physical CIs.\nSmoke ratio 2.56×10⁻³⁶ is retained in the data table.',transform=ax.transAxes,fontsize=8)
    ax=axes[1,1]
    for label in ['pilot','long-a','long-b','classical']:
        a=np.genfromtxt(raw[label]['folder']/'run/recrossing.csv',delimiter=',',names=True)
        t=a['time_au']*AU_TIME_FS
        ax.plot(t,a['kappa'],color=colors[label],label=names[label])
        ax.fill_between(t,a['kappa']-a['standard_error'],a['kappa']+a['standard_error'],color=colors[label],alpha=.10)
    ax.axhline(.776,color='black',ls='--',lw=.9,label='Paper RPMD κ = 0.776')
    ax.axhline(.994,color='black',ls=':',lw=.9,label='Paper classical κ = 0.994')
    ax.set(title='D  Separately measured transmission factors',xlabel='Child propagation time / fs',ylabel='κ(t)',xlim=(0,100),ylim=(.45,1.04))
    ax.legend(loc='lower right',fontsize=7.5,ncol=2)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('H + H₂ / BKMP2 at 300 K: numerical evidence, physical convergence pending',fontsize=15)
    fig.supxlabel('Reference: CPC 184 (2013), Table 2, DOI 10.1016/j.cpc.2012.10.017.  PMF bands: diagnostic 95%; κ bands: ±1 SE.',fontsize=9)
    fig.savefig(args.output/'h3-literature-comparison.png',dpi=180)
    fig.savefig(args.output/'h3-literature-comparison.pdf')
    plt.close(fig)
    print(json.dumps({'cases':len(table_rows),'independent_seed_rate_ratio':record['independent_seed_rate_ratio_a_over_b'],
                      'output':str(args.output)},indent=2))

if __name__=='__main__':main()
