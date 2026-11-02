"""Adopt the fixed child cohort only after the complete reviewed input gates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys

BASE=Path('/workspace/scratch/a5e4820daedd')
REPO=BASE/'PyRPMD'
INTEGRATED=BASE/'pyrpmd-convergence-integration'
DOC=Path('docs/validation/literature-convergence-2026-11-01')
ROOT=BASE/'literature-convergence-quantum/p256-fixed-eight-chain-v1'
OUT=ROOT/'child-execution'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'src'))
sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def identity(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)

def bound(path,digest):
    if sha(path)!=digest:
        raise ValueError(f'Frozen prerequisite changed: {path}')
    return read(path)

def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-report',required=True)
    parser.add_argument('--calibration-sha256',required=True)
    args=parser.parse_args()
    coordinator_path=REPO/'benchmarks/h3/p256_child_campaign.py'
    if sha(coordinator_path)!='0304915726116f062cee1cbb23a649d2c4e4086a787e4206d6dc7ec939ef89bb':
        raise ValueError('Reviewed coordinator changed')
    coordinator_review=REPO/DOC/'h3-quantum/p256-child-coordinator/independent-review.json'
    review=bound(coordinator_review,'06aaa12cd2302c078a6969cf95ef9d9b928a9ffceff7f92f3edf7de0a247f67d')
    if review['remaining_blocking_findings']:
        raise ValueError('Independent coordinator review is not clear')
    synthetic=REPO/DOC/'h3-quantum/p256-child-coordinator/final/synthetic-gate.json'
    bound(synthetic,'76a5d5e64b23ba67a246e50b329e0a4eb1e58828dd30386800206d82bd87e4df')
    scientific=REPO/DOC/'h3-quantum/p256-child-tool/independent-analysis-review.json'
    science=bound(scientific,'4bb9d6a68527f225d490ff5fedbd5ff7d7913da821dac39f6b5aa15efa43ce88')
    if science['remaining_blocking_findings']:
        raise ValueError('Scientific analyzer review has a blocker')
    parent_audit_path=REPO/DOC/'h3-quantum/p256-parent-stage/complete-audit.json'
    parent_audit=bound(parent_audit_path,'71e811ed90e866c19f7b90668e65bc0c3332925090f4bed2f09447497576d4f1')
    if not (parent_audit['all_parent_gates_passed'] is True and parent_audit['complete_independent_parent_chains']==8
            and parent_audit['parent_configurations']==1600 and parent_audit['historical_array_comparisons']==64):
        raise ValueError('Complete parent review is required')
    exit_path=ROOT/'execution/root-exec-exit.json'
    exited=bound(exit_path,'b0b596e4bf513b6fe4251d28e75e8fa979da0e23545224a775bfae19301011a4')
    if not (exited['exit_code']==0 and exited['all_workers_drained'] is True and exited['complete_registered_parent_chains']==8):
        raise ValueError('Parent worker slots are not confirmed released')
    gate_dir=INTEGRATED/DOC/'integration-4698c12/four-pair-p256-compatibility'
    gate_inventory=bound(gate_dir/'publication-inventory.json','27b099d5fefe02e27d9e16c9dd86290924459c9df9b20ee224a3fd6fef26f36a')
    for row in gate_inventory['files']:
        path=INTEGRATED/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Frozen cross-version evidence changed')
    compatibility=read(gate_dir/'summary.json')
    if compatibility['distinct_new_cases']!=239 or compatibility['combined_distinct_cases_per_interpreter']!=554:
        raise ValueError('Cross-version case scope changed')
    for label in ['py312','py313','py314']:
        row=compatibility['interpreters'][label]
        if row['junit_counts']!={'tests':239,'failures':0,'errors':0,'skipped':0} or not row['guard_after']['all_unchanged']:
            raise ValueError('A required compatibility gate did not pass')
    calibration_path=Path(args.calibration_report).resolve()
    if not calibration_path.is_relative_to(BASE):
        raise ValueError('Calibration evidence must be inside this workspace')
    calibration=bound(calibration_path,args.calibration_sha256)
    if calibration.get('all_checks_passed') is not True:
        raise ValueError('Actual complete excluded P256 calibration must pass')
    control_path=BASE/'literature-convergence-quantum/umbrella-worker-control.json'
    if read(control_path)!={'active_workers':4}:
        raise ValueError('Reassess actual resource allocation after control change')
    umbrella_admission=BASE/'literature-convergence-quantum/umbrella-worker-admission-2-to-4.json'
    bound(umbrella_admission,'94d0a3d49450db0571e59391cc78a9d84dfb016da257f7732c859d97a888ddd0')
    water_continuation=REPO/DOC/'water/progress/original-continuation-20261101T1945Z.json'
    bound(water_continuation,'412b2e66d687caf9d0bb055f5dbb6e4e35c14e03f2aadfa804b3f0374b73d0c2')
    from benchmarks.h3 import p256_child_campaign as coordinator
    specification=bound(coordinator.DESIGN,coordinator.DESIGN_SHA)
    producer=coordinator.producer()
    manifest=producer.verify_manifest(producer.PARENTS,specification)
    sources,physical=coordinator.source_inventory(specification,producer)
    runtime=coordinator.runtime_identity()
    if not {6,7}.issubset(os.sched_getaffinity(0)):
        raise ValueError('Required CPU IDs unavailable')
    OUT.mkdir(exist_ok=False)
    snapshot=OUT/'umbrella-control-at-adoption.json'
    with snapshot.open('xb') as stream:stream.write(control_path.read_bytes())
    resource_path=OUT/'resource-decision.json'
    resource=dict(schema='pyrpmd.p256-child.root-resource-decision.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),safe_to_launch_child_workers=True,
        child_workers=2,declared_project_active_workers=8,prerequisite_queues_drained=True,
        prerequisite_scope='Completed P256 parent workers; their two released slots are reused. Water and umbrella remain active under their unchanged ceilings.',
        allocation=dict(quantum_umbrella_worker_ceiling=4,original_water_remaining_integration_workers=2,P256_children=2),
        CPU_affinity=[6,7],whole_campaign_wall_seconds=36000,finalization_reserve_seconds=120,
        evidence=[identity(p) for p in [exit_path,parent_audit_path,umbrella_admission,snapshot,water_continuation,
            gate_dir/'summary.json',calibration_path]],
        active_logs_or_mutable_checkpoints_bound_as_resource_guards=False,
        no_water_extension_or_new_umbrella_stage_adopted=True)
    write(resource_path,resource)
    authorization=dict(schema=coordinator.AUTH_SCHEMA,parent_adoption=True,
        design_sha256=coordinator.DESIGN_SHA,P128_baseline_sha256=coordinator.BASELINE_SHA,
        source_files=sources,runtime=runtime,output=str(coordinator.OUTPUT.resolve()),
        seed_order=list(coordinator.SEEDS),branch_order=[[batch,dt] for batch,dt in coordinator.BRANCHES],
        workers=2,cpu_affinity=[6,7],whole_campaign_wall_seconds=36000,finalization_reserve_seconds=120,
        wall_budget_origin='root explicit child-campaign choice; not the parent-generation proposal',
        automatic_resume=False,automatic_analysis=False,
        parent_manifest=identity(producer.PARENTS/'parents-complete.json'),
        P128_complete_four_pair_report=identity(producer.BASELINE),
        library=specification['parent_generation']['external_library'],
        library_build_receipt=specification['parent_generation']['external_library_build_receipt'],
        synthetic_gate=identity(synthetic),independent_coordinator_review=identity(coordinator_review),
        scientific_review=identity(scientific),resource_decision=identity(resource_path))
    auth_path=OUT/'authorization.json';write(auth_path,authorization)
    context=coordinator.preflight(auth_path)
    preflight=dict(schema='pyrpmd.p256-child.root-readonly-preflight.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),passed=True,authorization=identity(auth_path),
        guarded_files=len(context['guards']),guard_sha256=context['guards'],all_eight_parents_eligible=True,
        registered_output_created=False,PES_evaluations=0,child_trajectories_launched=0,
        root_driver=identity(Path(__file__)),calibration_report=identity(calibration_path))
    write(OUT/'preflight.json',preflight)
    print(json.dumps(dict(authorization_sha256=sha(auth_path),coordinator_sha256=sha(coordinator_path),
        resource_decision_sha256=sha(resource_path),preflight_sha256=sha(OUT/'preflight.json'),
        guarded_files=len(context['guards']),children_launched=False)))

if __name__=='__main__':
    main()
