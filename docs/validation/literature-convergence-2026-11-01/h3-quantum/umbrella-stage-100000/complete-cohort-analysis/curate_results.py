"""Preserve complete-cohort estimates and the unresolved operational anomaly."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'PyRPMD'
PREFIX = Path('docs/validation/literature-convergence-2026-11-01/h3-quantum')
SOURCE = ROOT / 'literature-convergence-quantum/full-cohort-stage-100000'
CAMPAIGN = ROOT / 'literature-convergence-quantum/campaign-run'
OUT = REPO / PREFIX / 'umbrella-stage-100000/complete-cohort-analysis'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    report_path = SOURCE / 'campaign-comparison.json'
    assert sha(report_path) == 'db76aa53b848e9b24ee32c6e4ad7b199c151c3547f28d7954c9f49388d8bb16a'
    report = json.loads(report_path.read_text())
    execution = json.loads((SOURCE / 'execution.json').read_text())
    seeds = [211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039]
    assert report['expected_independent_seeds'] == report['completed_independent_seeds'] == seeds
    assert report['production_sweeps_per_ladder'] == 100000
    assert not report['criteria']['statistical_precision'] and not report['registered_primary_qtst_gates_passed']
    assert execution['returncode'] == 0 and execution['sources_unchanged'] and execution['completion_unchanged']
    assert execution['completion_sha256'] == sha(CAMPAIGN / 'stage-100000.json')
    sources = {report_path: 'campaign-comparison.json', SOURCE / 'execution.json': 'analysis-execution.json',
               CAMPAIGN / 'stage-100000.json': 'global-completion.json',
               ROOT / 'run_complete_umbrella_100000_analysis.py': 'executed_analysis_waiter.py',
               ROOT / 'launch_umbrella_stage100k.py': 'executed_sampling_launcher.py'}
    for seed in seeds:
        sources[SOURCE / f'seed-{seed}.json'] = f'per-seed/seed-{seed}.json'
    for name in ('umbrella-stage100000-root-observation.json', 'umbrella-analysis-100000-root-observation.json',
                 'umbrella-stage100000-launch.json', 'umbrella-stage100000.log', 'umbrella-analysis-100000.log'):
        sources[ROOT / 'literature-convergence-quantum' / name] = 'execution/' + name
    for key, session in [('umbrella-stage100000-root-observation.json', 57942), ('umbrella-analysis-100000-root-observation.json', 1094)]:
        observed = json.loads((ROOT / 'literature-convergence-quantum' / key).read_text())
        assert observed['exec_session_id'] == session and observed['tool_result']['exit_code'] == 0
        assert 'session_id' not in observed['tool_result']
    OUT.mkdir(exist_ok=False)
    provenance = []
    for source, name in sources.items():
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        provenance.append({'source': str(source), 'destination': str(target.relative_to(REPO)), 'bytes': target.stat().st_size, 'sha256': sha(target)})
    old = json.loads((REPO / PREFIX / 'umbrella-stage-30000/campaign-comparison.json').read_text())
    rows = []
    for stage, value in [(30000, old), (100000, report)]:
        rows.append({'production_sweeps_per_ladder': stage,
                     'qtst_cm3_molecule_s': value['pooled_wham']['qtst_cm3_molecule_s'],
                     'nominal_95_interval': value['statistics']['primary_student_t_95_interval'],
                     'primary_log_standard_error': value['statistics']['primary_log_standard_error'],
                     'paper_relative_difference': value['relative_difference_from_paper'],
                     'registered_precision_passed': value['criteria']['statistical_precision']})
    write_new(OUT / 'comparison-plot-data.json', {
        'rows': rows, 'paper_qtst_cm3_molecule_s': report['paper_qtst_cm3_molecule_s'],
        'paper_neighborhood_relative_halfwidth': 0.05,
        'paper_neighborhood_is_not_a_confidence_interval': True,
        'prefixes_are_correlated_and_not_independent_replications': True,
        'uncertainty_units': 'eight independently seeded complete exchange ladders',
    })
    write_new(OUT / 'curation-provenance.json', {
        'schema': 'pyrpmd.h3.umbrella100000-complete-result-curation.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_file_identities': provenance,
        'raw_checkpoints_or_PES_loaded_by_this_curation': False,
        'analysis_rerun_by_this_curation': False,
        'new300000_sampling_admitted': False,
        'operational_anomaly': 'The original completed scheduler lease survived normal execution. One reviewed rename preserved its bytes and briefly released the already authorized analyzer; a new inode with the same old owner bytes subsequently reappeared. All completion/source hashes stayed unchanged. Durable lease absence is not established and raw archive/new-writer admission remains unresolved.',
        'complete_analysis_source_execution_returncode': 0,
        'scientific_gate_status': 'precision failure retained; point agreement alone does not stop the registered refinement plan',
    })
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.3, 3.8), constrained_layout=True)
    paper = report['paper_qtst_cm3_molecule_s'] / 1e-16
    ax.axvspan(paper * .95, paper * 1.05, color='#d9e9df', label='Paper ±5% comparison neighborhood')
    ax.axvline(paper, color='#38674c', linestyle='--', label='Paper QTST: 2.51')
    for y, row in enumerate(rows):
        point = row['qtst_cm3_molecule_s'] / 1e-16
        low, high = np.asarray(row['nominal_95_interval']) / 1e-16
        ax.errorbar(point, y, xerr=[[point - low], [high - point]], fmt='o', color='#1f5e94', capsize=5, markersize=7)
        ax.annotate(f"log SE {row['primary_log_standard_error']:.3f} > 0.030: FAIL", (3.42, y), fontsize=9, va='center')
    ax.set_yticks([0, 1], ['30,000 sweeps / ladder', '100,000 sweeps / ladder'])
    ax.set_ylim(-.55, 1.55)
    ax.set_xlim(2.15, 4.42)
    ax.set_xlabel(r'$k_{\mathrm{QTST}}$  ($10^{-16}$ cm$^3$ molecule$^{-1}$ s$^{-1}$)')
    ax.set_title('H + H₂ at 300 K, P128: complete independent umbrella cohorts')
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower right', fontsize=8)
    fig.supxlabel('Eight whole ladders; nominal Student 95% intervals. Correlated refinement prefixes.\nPaper band is not a confidence interval; no transmission coefficient or full RPMD rate.', fontsize=8)
    fig.savefig(OUT / 'complete-prefix-comparison.png', dpi=180)
    fig.savefig(OUT / 'complete-prefix-comparison.svg')
    plt.close(fig)
    (OUT / 'README.md').write_text('''# H + H₂: complete100,000-sweep umbrella stage

All eight registered P128 exchange ladders completed the fixed100,000-sweep
production prefix after5,000 warmup sweeps. These are Monte Carlo sweeps,
not femtoseconds or molecular-dynamics steps. The unchanged complete-cohort
analyzer ran once after the actual sampling process had returned normally.

The pooled300 K QTST estimate is **2.5873968042 ×10⁻¹⁶ cm³ molecule⁻¹ s⁻¹**,
with nominal whole-ladder Student95% interval **[2.2941329563,2.9181491874]
×10⁻¹⁶**. The paper's2.51 ×10⁻¹⁶ value lies inside that interval; the point
estimate is3.083538% higher. The registered log standard-error limit still
**fails:0.05087375 >0.030**. Complete-cohort, overlap/residual, bin width,
local-fit width, temporal-half, seed-consistency and paper-point predicates
pass. This is **not** a passing overall equilibrium validation.

The initial30,000-sweep stage remains unchanged: its point estimate was
2.8007250269 ×10⁻¹⁶ and log SE0.07689760. The refinement reduced uncertainty;
point agreement never triggers or stops continuation. The plotted prefixes
share samples and are not independent replications. No transmission
coefficient was calculated here, and this report establishes no full RPMD
rate, bead convergence or physical-domain convergence.

The analyzed eight-ladder execution used35.924906 CPU seconds and36.004667
wall seconds. Actual root tool observations are preserved for sampling
session57942 and analysis session1094. Source, model and original prefixes
remain unchanged. The complete reports and exact command/log identities
are retained; this compact bundle contains no raw trajectory or PES asset.

## Retained lease anomaly

The original scheduler lease remained after a normal exit even though the
frozen source's normal cleanup should remove it. Independent review proved
the exact eight submissions/completions, drained worker pool and original
owner identity. Root performed one explicitly adopted no-overwrite rename
of the exact completed lease into quarantine. Immediate absence released
the already authorized analyzer; no second analyzer was launched. A new
directory/owner inode subsequently appeared at the original path with the
same historical owner bytes. The original quarantined inodes remain intact.

This anomaly's cause is **unexplained**. The first action's immediate success
is retained separately from the failed durable-absence postcondition.
No repeated cleanup, new sampler or300,000-sweep stage has been admitted.
Raw archive admission requires a new explicit read-only review binding the
anomaly, both actual normal exits, unchanged complete data and absence of
any admitted writer. Scientific precision failure remains visible while
that operational preservation step is handled.

![Complete-prefix comparison](complete-prefix-comparison.png)

The numerical paper reference is Table2 of Suleimanov, Allen and Green,
*Computer Physics Communications*184 (2013),833–840,
[doi:10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).
The approximate5% paper comparison neighborhood is not a reported95%
confidence interval. Our intervals are nominal estimates over eight whole
ladders; registered repeated-prefix diagnostics do not create additional
independent samples.
'''.replace('complete100', 'complete 100').replace('fixed100', 'fixed 100').replace('after5', 'after 5').replace('pooled300', 'pooled 300').replace('Student95', 'Student 95').replace("paper's2.51", "paper's 2.51").replace('is3.083', 'is 3.083').replace('fails:0.05', 'fails: 0.05').replace('>0.030', '> 0.030').replace('initial30', 'initial 30').replace('SE0.076', 'SE 0.076').replace('used35.', 'used 35.').replace('and36.', 'and 36.').replace('session57942', 'session 57942').replace('session1094', 'session 1094').replace('or300,000', 'or 300,000').replace('Table2', 'Table 2').replace('Communications*184', 'Communications* 184').replace('(2013),833', '(2013), 833').replace('approximate5%', 'approximate 5%').replace('reported95%', 'reported 95%'))
    shutil.copy2(Path(__file__), OUT / 'curate_results.py')
    files = [{'path': str(p.relative_to(REPO)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
    write_new(OUT / 'publication-inventory.json', {
        'schema': 'pyrpmd.immutable-validation-inventory.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'file_count': len(files), 'total_bytes': sum(r['bytes'] for r in files),
        'inventory_excludes_itself': True,
    })
    print(json.dumps({'inventory_sha256': sha(OUT / 'publication-inventory.json'), 'files': len(files), 'report_sha256': sha(OUT / 'campaign-comparison.json'), 'all_primary_gates_passed': False}))


if __name__ == '__main__':
    main()
