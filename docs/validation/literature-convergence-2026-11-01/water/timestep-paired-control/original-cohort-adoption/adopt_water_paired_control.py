"""Bind completed original evidence before the independent numerical control."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'PyRPMD'
BASE = REPO / 'docs/validation/literature-convergence-2026-11-01/water/timestep-paired-control'
DRAFT = BASE / 'original-cohort-adoption-draft'
OUT = BASE / 'original-cohort-adoption'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    inventory = DRAFT / 'publication-inventory.json'
    assert sha(inventory) == '3dc6fd9739e4d6b22b1a3beb01e6b3aa2d24fc0cf0079aef43fe9060b0407564'
    guards = {str(inventory): sha(inventory)}
    for row in json.loads(inventory.read_text())['files']:
        path = REPO / row['path']
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        guards[str(path)] = row['sha256']
    draft_path = DRAFT / 'pending-adoption.json'
    assert sha(draft_path) == '36bd3fea357df4f37d81f45956af85ba01ce2314387914ea89c6aa121d61db40'
    draft = json.loads(draft_path.read_text())
    assert draft['decision'] == 'pending_root_review_not_adopted'
    assert draft['fixed_seeds'] == list(range(1101, 1109))
    assert draft['total_NVE_ps'] == 64 and draft['total_physical_steps'] == 384000
    for key in ['original_complete_assessment', 'original_normal_exit', 'raw_archive_index', 'protocol', 'selected_inputs']:
        row = draft[key]
        assert sha(row['path']) == row['sha256']
        guards[row['path']] = row['sha256']
    for name, expected in draft['frozen_source_sha256'].items():
        path = REPO / name
        assert sha(path) == expected
        guards[str(path)] = expected
    archive = draft['raw_archive']
    assert Path(archive['path']).stat().st_size == archive['bytes']
    assert sha(archive['path']) == archive['sha256']
    selected = json.loads(Path(draft['selected_inputs']['path']).read_text())
    assert selected['all_eight_original_chains_complete'] is True
    assert selected['all_original_leases_acquired_together_then_released'] is True
    assert selected['PES_evaluations'] == selected['RNG_draws'] == selected['contexts_created'] == 0
    assert [r['seed'] for r in selected['selected']] == list(range(1101, 1109))
    for row in selected['selected']:
        assert row['active_segment'] == -1 and row['state_step'] == 400000
        directory = Path(row['raw_directory'])
        for path, expected in [(directory / 'manifest.json', row['manifest_sha256']),
                               (directory / row['checkpoint']['file'], row['checkpoint']['sha256'])]:
            assert sha(path) == expected
            guards[str(path)] = expected
    assessment = json.loads(Path(draft['original_complete_assessment']['path']).read_text())
    assert assessment['all_predeclared_chains_complete'] is True
    assert assessment['literature_comparison'] is None and assessment['physical_converged'] is False
    complete_umbrella = []
    pending_umbrella = []
    for seed in [211017,211019,211021,211023,211027,211031,211033,211039]:
        progress = ROOT / f'literature-convergence-quantum/campaign-run/seed-{seed}/progress.json'
        j = json.loads(progress.read_text())
        (complete_umbrella if j['production_sweeps'] == 100000 else pending_umbrella).append(seed)
    assert len(complete_umbrella) >= 6 and len(pending_umbrella) <= 2
    output = ROOT / 'water-paired-timestep-control-2026-11-01'
    assert not output.exists()
    launcher = ROOT / 'launch_water_paired_control.py'
    guards[str(launcher)] = sha(launcher)
    OUT.mkdir(exist_ok=False)
    decision = dict(schema='pyrpmd.root-water-paired-timestep-adoption.v1',
        adopted_utc=datetime.now(timezone.utc).isoformat(),
        decision='adopted_fixed_numerical_control',
        reason='The complete original cohort has exited and been archived. Its preparation and early-MSD failures remain unchanged. Execute the independently preregistered all-eight paired timestep property check before considering a separate preparation extension.',
        pending_adoption=dict(path=str(draft_path), sha256=sha(draft_path)),
        launcher_sha256=sha(launcher), control_output=str(output),
        input_and_source_guards=guards,
        archive_verified=dict(path=archive['path'], sha256=archive['sha256'], bytes=archive['bytes']),
        resources=dict(integration_workers=1, CPU_affinity=[4], library_threads=1,
            maximum_invocations=4, bounded_invocation_wall_seconds=1800,
            total_admission_wall_seconds=7200,
            checkpoint_boundary_budget_semantics_retained=True,
            environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', OPENMM_CPU_THREADS='1',
                PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1'),
            other_registered_long_workers=dict(P256_children=2, umbrella_remaining=len(pending_umbrella)),
            umbrella_complete_seed_observation=complete_umbrella,
            umbrella_pending_seed_observation=pending_umbrella,
            original_water_workers=0, total_long_worker_ceiling=5,
            separate_bounded_nonphysical_work='Current-head pytest and archive replay may use their previously assigned short-work resources.'),
        fixed_original_failures_retained=assessment['eligibility_checks'],
        original_data_overwritten=False, physical_parameters_changed=False,
        additional_preparation_adopted=False, quantum_campaign_adopted=False)
    with (OUT / 'root-adoption.json').open('x') as stream:
        stream.write(json.dumps(decision, indent=2, sort_keys=True) + '\n')
    for source in [launcher, Path(__file__)]:
        with (OUT / source.name).open('xb') as stream:
            stream.write(source.read_bytes())
    print(json.dumps(dict(adopted=True, adoption_sha256=sha(OUT / 'root-adoption.json'),
                         CPU_affinity=[4], integration_workers=1, NVE_ps=64, steps=384000)))

if __name__ == '__main__':
    main()
