"""Copy complete four-pair evidence; never analyze or curate a partial cohort."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
DATA = BASE / 'literature-convergence-quantum/recrossing-primary'
ANALYSIS = BASE / 'literature-convergence-quantum/recrossing-analysis-200-four-pairs'
DEST = REPO / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/recrossing-stage-200-pairs-4'
SEEDS = [721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323]
PROTOCOL_SHA = '53203aeca4fdf66d9fbf1a23fe7a430ab99e24d18c351e85bd2eb383a87d3589'
ANALYZER_SHA = '4d1925461219e87d85d4e1f4ac98263c4bb58ff0d7713075782145ba909649c1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report = json.loads((ANALYSIS / 'quantum-rate-comparison.json').read_text())
    execution = json.loads((ANALYSIS / 'execution.json').read_text())
    completion = json.loads((DATA / 'schedule-200-pairs-1-completed.json').read_text())
    if not (report['completed_seeds'] == report['expected_seeds'] == SEEDS
            and report['pair_batches'] == [0, 1] and report['horizon_fs'] == 200
            and report['criteria']['complete_registered_cohort'] is True
            and report['protocol_sha256'] == PROTOCOL_SHA
            and report['analysis_source_sha256'] == ANALYZER_SHA
            and execution['returncode'] == 0 and execution['sources_unchanged'] is True
            and execution['complete_cohort_required'] is True
            and execution['available_only_used'] is False
            and completion['inventory']['seeds'] == SEEDS
            and [row['seed'] for row in completion['completed']] == SEEDS
            and all(row['complete'] for row in completion['completed'])):
        raise ValueError('Complete analyzed four-pair cohort is required')
    if sha(DATA / 'protocol.json') != PROTOCOL_SHA:
        raise ValueError('Protocol changed')
    copies = []

    def copy(source, name):
        source, destination = Path(source), DEST / name
        if destination.exists():
            raise FileExistsError(f'Never overwrite a previous stage artifact: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha(source) != sha(destination):
            raise ValueError('Copy identity differs')
        copies.append({'path':str(destination.relative_to(REPO)), 'source_path':str(source),
                       'sha256':sha(destination), 'bytes':destination.stat().st_size})

    for name in ['quantum-rate-comparison.json', 'kappa-curve.csv', 'execution.json']:
        copy(ANALYSIS / name, name)
    for name in ['protocol.json', 'schedule-200-pairs-1.json', 'schedule-200-pairs-1-completed.json']:
        copy(DATA / name, name)
    copy(BASE / 'run_complete_quantum_four_pairs_analysis.py', 'analysis_driver.py')
    copy(Path(__file__), 'curation_driver.py')
    for seed in SEEDS:
        for dt in ['0.1', '0.05']:
            directory = DATA / f'extra-pair-batch-1/seed-{seed}/dt-{dt}/horizon-200'
            detail = json.loads((directory / 'report.json').read_text())
            if not detail['complete'] or not detail['source_unchanged']:
                raise ValueError('Extra-pair run did not finish unchanged')
            for name in ['settings.json', 'report.json', 'kappa.csv']:
                copy(directory / name, f'extra-pair-batch-1/seed-{seed}/dt-{dt}/{name}')
    prior = DEST.parent / 'recrossing-stage-200-pairs-2/publication-inventory.json'
    prior_archive = DEST.parent / 'recrossing-stage-200-pairs-2/raw-archive/publication-inventory.json'
    if not prior_archive.exists():
        prior_archive = DEST.parent / 'recrossing-stage-200-pairs-2/raw-archive/publishable-files.json'
    receipt = {'schema':'pyrpmd.h3.complete-four-pair-curation.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'files':copies,
        'complete_registered_seeds':SEEDS, 'pair_batches':[0,1],
        'base_stage_inventory':{'path':str(prior.relative_to(REPO)),'sha256':sha(prior)},
        'base_raw_archive_inventory':{'path':str(prior_archive.relative_to(REPO)),'sha256':sha(prior_archive)},
        'base_metadata_not_duplicated':True, 'no_raw_arrays_or_external_models_copied':True,
        'scientific_gates_modified':False, 'figure_and_independent_review_are_separate':True}
    with (DEST / 'curation.json').open('x') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files_copied':len(copies), 'bytes':sum(x['bytes'] for x in copies),
                     'source_report_sha256':sha(ANALYSIS/'quantum-rate-comparison.json'),
                     'fixed_P_gates_passed':report['fixed_P_independent_rate_gates_passed']}))


if __name__ == '__main__':
    main()
