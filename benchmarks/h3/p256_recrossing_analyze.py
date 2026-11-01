"""Complete-eight-chain P256/P128 full-rate comparison with independent groups."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3 import p256_surface_recrossing as producer
from benchmarks.h3.quantum_recrossing_analyze import (
    load_seed, paired_rate_statistics, equivalent_paired_difference, interval,
)
BASE_ANALYZER = ROOT / 'benchmarks/h3/quantum_recrossing_analyze.py'
BASE_ANALYZER_SHA = '4d1925461219e87d85d4e1f4ac98263c4bb58ff0d7713075782145ba909649c1'


def check_analysis_dependencies(baseline):
    producer.require(producer.sha(BASE_ANALYZER) == BASE_ANALYZER_SHA
                     and baseline['analysis_source_sha256'] == BASE_ANALYZER_SHA,
                     'The imported frozen P128 analysis helpers changed')
    return {str(BASE_ANALYZER.relative_to(ROOT)): BASE_ANALYZER_SHA,
            str(Path(__file__).relative_to(ROOT)): producer.sha(__file__)}


def nonfinite_json_record(value):
    """Preserve invalid outcomes explicitly in JSON, without dropping a chain."""
    observations = []
    def convert(item, path):
        if isinstance(item, np.ndarray):
            return convert(item.tolist(), path)
        if isinstance(item, np.generic):
            return convert(item.item(), path)
        if isinstance(item, float) and not np.isfinite(item):
            observations.append({'path':path,'value_representation':repr(item)})
            return None
        if isinstance(item, dict):
            return {key:convert(member,path+'/'+str(key)) for key,member in item.items()}
        if isinstance(item, (list,tuple)):
            return [convert(member,path+'/'+str(i)) for i,member in enumerate(item)]
        return item
    result = convert(value, '')
    return result, observations


def independent_difference(first, second, margin=None):
    """Welch interval for independent complete groups, never paired by index."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != (8,) or second.shape != (8,) or not np.isfinite([first, second]).all():
        raise ValueError('Two complete finite eight-chain groups are required')
    pieces = np.array([first.var(ddof=1)/8, second.var(ddof=1)/8])
    variance = float(pieces.sum())
    df = float(variance**2 / np.sum(pieces**2/7)) if variance else 14.
    delta = float(first.mean()-second.mean())
    half = float(student_t.ppf(.975, df)*np.sqrt(variance))
    low, high = delta-half, delta+half
    result = {'mean_difference': delta, 'standard_error': float(np.sqrt(variance)),
              'CI95': [low, high], 'halfwidth95': half, 'Welch_df': df,
              'independent_chain_counts': [8, 8], 'groups_paired_by_index': False,
              'degenerate_observed_variance': variance == 0,
              'scope': 'Nominal independent-chain interval, not sequentially adjusted; zero observed variance is not a rare-event bound.'}
    if margin is not None:
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('Positive finite equivalence margin required')
        result.update(absolute_margin=float(margin), CI95_wholly_within_margin=bool(low >= -margin and high <= margin))
    return result


def finite_P_reactant(reference, protocol, specification):
    rows = [row for row in reference['bead_refinement'] if row['beads'] == 256]
    producer.require(len(rows) == 1, 'Exactly one P256 reactant reference is required')
    row = rows[0]
    beta = row['free_kernel_width_bohr']**2 * reference['reduced_mass_au'] * 256
    producer.require(reference['numerical_converged'] is True
        and reference['temperature_K'] == protocol['temperature_K'] == 300.
        and reference['reduced_mass_au'] == protocol['mass_au']/2
        and reference['provenance']['pes'] == protocol['model']
        and row['domain_bohr'] == [.5, 8.]
        and np.isclose(beta, protocol['beta'], rtol=1e-14, atol=0)
        and row['log_quantum_classical_ratio'] == specification['statistics_and_acceptance']['finite_P_reactant_log_ratio']['256'],
        'Reactant mass, temperature, finite P, domain, normalization or PES differs')
    return row


def full_rate_statistics(log_ts, kappas, reactant_log_ratio, classical_tst, baseline):
    """Retain nonpositive outcomes as failed inference, never omit a chain."""
    values = np.asarray(kappas, float)
    old = baseline['statistics']
    if values.shape != (8,) or np.asarray(log_ts).shape != (8,):
        raise ValueError('All eight P256 chain outcomes are required')
    if not np.isfinite(values).all() or np.any(values <= 0):
        return {'positive_finite_kappa_for_every_chain': False,
                'per_chain_kappa': [float(x) if np.isfinite(x) else None for x in values],
                'per_chain_log_TS': np.asarray(log_ts, float).tolist(),
                'statistics': None, 'bead_comparison': None,
                'failure': 'At least one nonpositive/nonfinite plateau; all outcomes retained, log-rate inference unavailable.'}
    statistics = paired_rate_statistics(log_ts, values, reactant_log_ratio, classical_tst)
    comparison = independent_difference(statistics['per_seed_log_rate'], old['per_seed_log_rate'], np.log(1.05))
    comparison.update(rate_ratio=float(np.exp(comparison['mean_difference'])),
                      rate_ratio_CI95=np.exp(comparison['CI95']).tolist(),
                      rate_ratio_acceptance_band=[1/1.05, 1.05])
    old_kappa = np.exp(np.asarray(old['per_seed_log_rate'])-np.asarray(old['per_seed_log_qtst']))
    return {'positive_finite_kappa_for_every_chain': True, 'per_chain_kappa': values.tolist(),
            'per_chain_log_TS': np.asarray(log_ts, float).tolist(), 'statistics': statistics,
            'bead_comparison': comparison,
            'independent_kappa_difference': independent_difference(values, old_kappa),
            'independent_log_kappa_ratio': independent_difference(np.log(values), np.log(old_kappa)),
            'no_independence_assumption_within_parent_chain': True}


def analyze(data, output):
    data, output = Path(data), Path(output)
    producer.require(not output.exists(), 'Never replace a completed or failed P256 analysis')
    raw_protocol = json.loads((data/'protocol.json').read_text())
    protocol = dict(raw_protocol)
    protocol.pop('created_before_primary_children_utc')
    specification = producer.bound_json(producer.DESIGN, producer.DESIGN_SHA)
    producer.require(protocol['schema'] == 'pyrpmd.h3.p256-surface-recrossing.v1'
        and protocol['design_sha256'] == producer.DESIGN_SHA
        and tuple(protocol['seeds']) == producer.SEEDS and protocol['beads'] == 256
        and protocol['parents_per_seed'] == 200 and protocol['timestep_fs'] == [.1, .05]
        and protocol['primary_timestep_fs'] == .05 and protocol['horizon_fs'] == 200
        and protocol['registered_primary_pair_batches'] == [0, 1]
        and protocol['momentum_pair_indices'] == [0, 1, 2, 3]
        and protocol['parent_completed_sweeps'] == list(range(2100, 22001, 100))
        and protocol['source_sha256'] == producer.source_identity(), 'Child protocol/source differs')
    parent_root = Path(protocol['parent_root'])
    manifest = producer.verify_manifest(parent_root, specification)
    producer.require(producer.sha(parent_root/'parents-complete.json') == protocol['parent_manifest_sha256'],
                     'Parent manifest changed after child registration')
    baseline = producer.bound_json(producer.BASELINE, producer.BASELINE_SHA)
    dependency_identities = check_analysis_dependencies(baseline)
    producer.require(protocol['P128_complete_four_pair_report']['sha256'] == producer.BASELINE_SHA
        and tuple(baseline['completed_seeds']) == tuple(baseline['expected_seeds']) == producer.P128_SEEDS
        and baseline['pair_batches'] == [0, 1] and baseline['horizon_fs'] == 200,
        'Complete fixed P128 baseline required')
    missing = []
    for seed in producer.SEEDS:
        for dt in (.1, .05):
            for batch in (0, 1):
                path = producer.run_directory(data, seed, dt, 200, batch)/'report.json'
                if not path.exists() or not json.loads(path.read_text())['complete']:
                    missing.append(str(path))
    producer.require(not missing, 'All8seeds/bothdt/bothpairbatches must complete before inference: '+repr(missing))
    reference_identity = specification['statistics_and_acceptance']['reactant_reference']
    reference_path = ROOT / reference_identity['path']
    reference = producer.bound_json(reference_path, reference_identity['sha256'])
    finite = finite_P_reactant(reference, protocol, specification)
    classical_path = producer.EVIDENCE/'h3-classical/transmission/transmission-comparison.json'
    classical = producer.bound_json(classical_path, baseline['classical_reference_sha256'])
    classical_tst = classical['independent_classical_tst_cm3_molecule_s']
    per_dt, log_ts = {'0.1': [], '0.05': []}, []
    for source in specification['parent_generation']['cohorts']:
        seed = source['seed']
        _, _, identity = producer.load_parents(parent_root/source['relative_directory'], protocol)
        log_ts.append(identity['BAR_log_TS_quantum_classical_ratio'])
        for dt in (.1, .05):
            per_dt[str(dt)].append(load_seed(data, protocol, seed, dt, 200, (0, 1), identity))
        producer.require(per_dt['0.1'][-1]['momentum_pairing_hashes'] == per_dt['0.05'][-1]['momentum_pairing_hashes'],
                         'The two timesteps did not use identical parent/channel/Maxwell draws')
    fine, coarse = per_dt['0.05'], per_dt['0.1']
    sanitized_rows, invalid_observations = nonfinite_json_record(per_dt)
    total_native_steps = sum(row['unique_native_steps'] for rows in per_dt.values() for row in rows)
    producer.require(total_native_steps == 76800000, 'Incomplete native step accounting')
    if invalid_observations:
        producer.require(dependency_identities == check_analysis_dependencies(baseline)
                         and protocol['source_sha256'] == producer.source_identity(),
                         'Analysis or producer source changed')
        report = {'schema':'pyrpmd.h3.complete-P256-dynamic-bead-comparison.v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'completed_seeds':list(producer.SEEDS),'expected_seeds':list(producer.SEEDS),
            'beads':256,'horizon_fs':200,'pair_batches':[0,1],
            'protocol_sha256':producer.sha(data/'protocol.json'),
            'analysis_source_sha256':producer.sha(__file__),'analysis_dependencies_sha256':dependency_identities,
            'P128_baseline_sha256':producer.BASELINE_SHA,'parent_manifest_sha256':protocol['parent_manifest_sha256'],
            'criteria':{'complete_registered_cohort':True,'all_outcomes_finite':False},
            'registered_full_rate_bead_control_passed':False,'rate_analysis':None,
            'failure':'Nonfinite complete-cohort observations: inference unavailable; every chain and invalid field is retained.',
            'nonfinite_observations':invalid_observations,'per_timestep_per_seed':sanitized_rows,
            'native_steps_both_timesteps':total_native_steps,'kappa_curve_written':False,
            'production_MD_umbrella_validated':False}
        output.mkdir(parents=True,exist_ok=False)
        producer.write_json(output/'bead-comparison.json',report)
        return report
    kappas = [row['plateaus']['180-200'] for row in fine]
    comparison = full_rate_statistics(log_ts, kappas, finite['log_quantum_classical_ratio'], classical_tst, baseline)
    plateau = equivalent_paired_difference(kappas, [row['plateaus']['80-100'] for row in fine], .01)
    timestep = equivalent_paired_difference([row['plateaus']['180-200'] for row in coarse], kappas, .01)
    halves = equivalent_paired_difference([row['parent_time_half_plateaus']['180-200'][1] for row in fine],
                                         [row['parent_time_half_plateaus']['180-200'][0] for row in fine], .03)
    halves['scope'] = 'Supplementary parent-time-half equivalence; failures retained separately from registered primary criteria.'
    statistics = comparison['statistics']
    criteria = {'complete_registered_cohort': True, 'all_outcomes_finite':True,
        'all_parent_sampling_flags_passed': all(row['sampling_flags_passed'] for row in manifest['parents']),
        'P128_registered_primary_criteria_passed': baseline['fixed_P_independent_rate_gates_passed'] is True,
        'positive_finite_kappa_for_every_chain': comparison['positive_finite_kappa_for_every_chain'],
        'P256_time_plateau_equivalence': plateau['CI95_wholly_within_margin'],
        'P256_paired_timestep_equivalence': timestep['CI95_wholly_within_margin'],
        'P256_primary_energy_screen': max(row['maximum_energy_error_kbt_per_bead'] for row in fine) <= .1,
        'P256_no_sampled_domain_crossings': all(row['children_with_sampled_domain_crossing'] == 0 for rows in per_dt.values() for row in rows),
        'P256_kappa_relative95_halfwidth_at_most03': bool(statistics is not None and statistics['kappa_statistics']['halfwidth95']/statistics['kappa_arithmetic_mean'] <= .03),
        'full_rate_bead_equivalence': bool(comparison['bead_comparison'] is not None and comparison['bead_comparison']['CI95_wholly_within_margin'])}
    curves = np.array([row['curve'] for row in fine])
    half = student_t.ppf(.975, 7)*curves.std(axis=0, ddof=1)/np.sqrt(8)
    for rows in per_dt.values():
        for row in rows:
            row.pop('curve')
    report = {'schema':'pyrpmd.h3.complete-P256-dynamic-bead-comparison.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'completed_seeds':list(producer.SEEDS),
        'expected_seeds':list(producer.SEEDS), 'beads':256, 'horizon_fs':200, 'pair_batches':[0,1],
        'protocol_sha256':producer.sha(data/'protocol.json'), 'analysis_source_sha256':producer.sha(__file__),
        'analysis_dependencies_sha256':dependency_identities,
        'parent_manifest_sha256':protocol['parent_manifest_sha256'], 'P128_baseline_sha256':producer.BASELINE_SHA,
        'finite_P_reactant_reference_sha256':reference_identity['sha256'],
        'selected_P256_reactant_log_ratio':finite['log_quantum_classical_ratio'],
        'P128_finite_P_denominator_not_reused_for_P256':True,
        'rate_analysis':comparison, 'time_plateau_comparison':plateau, 'paired_timestep_comparison':timestep,
        'supplementary_parent_time_halves':halves,
        'mean_denominator_diagnostic':interval([row['mean_denominator'] for row in fine]),
        'per_timestep_per_seed':per_dt, 'criteria':criteria,
        'registered_full_rate_bead_control_passed':bool(all(criteria.values())),
        'native_steps_both_timesteps':sum(row['unique_native_steps'] for rows in per_dt.values() for row in rows),
        'paper_comparison_scope':'The original paper usesP128; P256 tests numerical bead convergence independently of closeness to that point.',
        'production_MD_umbrella_validated':False,
        'limitations':['Nominal intervals exclude PES uncertainty and are not sequentially adjusted.',
                       'Domain minima are recorded at1fs boundaries; native hard guards and Hamiltonian extrema are evaluated each physical step.',
                       'A sampled plateau and supplementary stationarity outcome do not bound arbitrarily rare recrossing.']}
    producer.require(report['native_steps_both_timesteps'] == 76800000, 'Incomplete native step accounting')
    producer.require(dependency_identities == check_analysis_dependencies(baseline), 'Analysis dependencies changed during analysis')
    producer.require(protocol['source_sha256'] == producer.source_identity(), 'Producer source changed during analysis')
    output.mkdir(parents=True, exist_ok=False)
    producer.write_json(output/'bead-comparison.json', report)
    np.savetxt(output/'kappa-curve.csv', np.column_stack((np.arange(1,201),curves.mean(axis=0),curves.mean(axis=0)-half,curves.mean(axis=0)+half)),
               delimiter=',',header='time_fs,kappa_seed_mean,pointwise_CI95_low,pointwise_CI95_high',comments='')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = analyze(args.data,args.output)
    print(json.dumps({'completed_seeds':report['completed_seeds'],'criteria':report['criteria'],
                      'registered_full_rate_bead_control_passed':report['registered_full_rate_bead_control_passed']}))
