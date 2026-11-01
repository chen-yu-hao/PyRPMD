"""Independent statistical and finite-P identity checks; no trajectories run."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy import stats

from benchmarks.h3 import p256_recrossing_analyze as analysis
from benchmarks.h3 import p256_surface_recrossing as producer


def test_independent_welch_interval_matches_scipy_unequal_variance_oracle():
    first = np.array([.02, .05, -.04, .01, .09, -.07, .03, .04])
    second = np.array([.005, -.003, .012, -.006, .009, -.010, .013, -.008])
    expected = stats.ttest_ind(first, second, equal_var=False).confidence_interval(.95)
    observed = analysis.independent_difference(first, second, np.log(1.05))
    np.testing.assert_allclose(observed['CI95'], [expected.low, expected.high], rtol=1e-13, atol=1e-15)
    assert observed['groups_paired_by_index'] is False
    assert 7 < observed['Welch_df'] < 14
    assert observed['CI95_wholly_within_margin'] is False


def test_group_reordering_does_not_artificially_pair_independent_seeds():
    a = np.array([1., 3., 7., 2., 8., 3., -2., 6.])/100
    b = np.array([3., -1., 2., -2., 4., 1., 5., 1.])/100
    first = analysis.independent_difference(a, b)
    second = analysis.independent_difference(a, b[::-1])
    np.testing.assert_allclose(first['CI95'], second['CI95'], atol=1e-16)


@pytest.mark.parametrize('shift,passes', [(0., True), (.04, True), (.06, False)])
def test_full_interval_equivalence_and_degenerate_observation_are_explicit(shift, passes):
    result = analysis.independent_difference(np.full(8, shift), np.zeros(8), np.log(1.05))
    assert result['degenerate_observed_variance'] is True
    assert result['CI95_wholly_within_margin'] is passes
    assert result['CI95'] == [shift, shift]


@pytest.mark.parametrize('a,b', [(np.ones(7), np.ones(8)), (np.ones(8), np.ones(9)),
                                (np.r_[np.ones(7), np.nan], np.ones(8)),
                                (np.ones((8,1)), np.ones(8))])
def test_incomplete_or_nonfinite_groups_cannot_form_primary_interval(a, b):
    with pytest.raises(ValueError): analysis.independent_difference(a, b)


def baseline(log_qtst, kappa):
    return {'statistics': {'per_seed_log_qtst': np.asarray(log_qtst).tolist(),
                          'per_seed_log_rate': (np.asarray(log_qtst)+np.log(kappa)).tolist()}}


def test_joint_chain_rate_retains_partition_transmission_covariance():
    delta = np.arange(8)*.005
    log_ts = -2.1+delta
    kappa = .77*np.exp(-delta)
    old = baseline(log_ts, kappa)
    result = analysis.full_rate_statistics(log_ts, kappa, 0., 1., old)
    assert result['positive_finite_kappa_for_every_chain'] is True
    statistics = result['statistics']
    assert statistics['log_rate_statistics']['standard_error'] < 1e-14
    assert statistics['same_chain_log_partition_log_kappa_covariance_of_mean'][0][1] < 0
    assert statistics['incorrect_independence_log_rate_variance_for_comparison_only'] > 1e-5
    assert result['bead_comparison']['CI95_wholly_within_margin'] is True


@pytest.mark.parametrize('invalid', [0., -.01, np.nan, np.inf])
def test_nonpositive_or_nonfinite_chain_retained_without_lograte_inference(invalid):
    kappa = np.full(8, .77)
    kappa[3] = invalid
    result = analysis.full_rate_statistics(np.full(8, -2.), kappa, -7., 1., baseline(np.full(8, -2.), np.full(8, .77)))
    assert result['positive_finite_kappa_for_every_chain'] is False
    assert result['statistics'] is None and result['bead_comparison'] is None
    assert len(result['per_chain_kappa']) == 8
    assert result['per_chain_kappa'][3] == (float(invalid) if np.isfinite(invalid) else None)


def reactant_inputs():
    design = json.loads(producer.DESIGN.read_text())
    reference = json.loads((producer.ROOT/design['statistics_and_acceptance']['reactant_reference']['path']).read_text())
    child = design['child_experiment_definition']
    protocol = {'temperature_K':300., 'mass_au':child['mass_au'], 'beta':child['beta'],
                'model':design['parent_generation']['cohorts'][0]['expected_settings']['model']}
    return reference, protocol, design


def test_P256_uses_its_own_finite_P_normalization():
    reference, protocol, design = reactant_inputs()
    row = analysis.finite_P_reactant(reference, protocol, design)
    assert row['beads'] == 256
    assert row['log_quantum_classical_ratio'] == -7.282252233219134
    assert row['log_quantum_classical_ratio'] != reference['primary_log_quantum_classical_ratio']


@pytest.mark.parametrize('change', ['mass','temperature','kernel','domain','PES','duplicate','unconverged'])
def test_mismatched_reactant_reference_rejected(change):
    reference, protocol, design = reactant_inputs()
    row = next(row for row in reference['bead_refinement'] if row['beads'] == 256)
    if change == 'mass': protocol['mass_au'] *= 2
    elif change == 'temperature': protocol['temperature_K'] = 301.
    elif change == 'kernel': row['free_kernel_width_bohr'] *= 1.01
    elif change == 'domain': row['domain_bohr'][0] = .4
    elif change == 'PES': protocol['model']['source_sha256'] = '0'*64
    elif change == 'duplicate': reference['bead_refinement'].append(deepcopy(row))
    else: reference['numerical_converged'] = False
    with pytest.raises(ValueError): analysis.finite_P_reactant(reference, protocol, design)


def test_analysis_dependency_guard_rejects_modified_imported_helper(tmp_path, monkeypatch):
    path = tmp_path/'wrong_analysis.py'
    path.write_text('changed scientific helper\n')
    monkeypatch.setattr(analysis, 'BASE_ANALYZER', path)
    with pytest.raises(ValueError, match='analysis helpers changed'):
        analysis.check_analysis_dependencies({'analysis_source_sha256':analysis.BASE_ANALYZER_SHA})


@pytest.fixture
def synthetic_complete_analysis(tmp_path, monkeypatch):
    data = tmp_path/'data'
    data.mkdir()
    parent_root = tmp_path/'parents'
    parent_root.mkdir()
    parent_manifest = parent_root/'parents-complete.json'
    parent_manifest.write_text('{}\n')
    actual_design = json.loads(producer.DESIGN.read_text())
    baseline_path=tmp_path/'baseline.json'
    synthetic_baseline=dict(baseline(np.full(8,-35.9),np.full(8,.77)),
        expected_seeds=list(producer.P128_SEEDS),completed_seeds=list(producer.P128_SEEDS),
        pair_batches=[0,1],horizon_fs=200,fixed_P_independent_rate_gates_passed=True,
        criteria={'complete_registered_cohort':True},analysis_source_sha256=analysis.BASE_ANALYZER_SHA,
        classical_reference_sha256=producer.sha(producer.EVIDENCE/'h3-classical/transmission/transmission-comparison.json'))
    baseline_path.write_text(json.dumps(synthetic_baseline))
    monkeypatch.setattr(producer,'BASELINE',baseline_path)
    monkeypatch.setattr(producer,'BASELINE_SHA',producer.sha(baseline_path))
    protocol = dict(schema='pyrpmd.h3.p256-surface-recrossing.v1', design_sha256=producer.DESIGN_SHA,
        seeds=list(producer.SEEDS), beads=256, parents_per_seed=200, timestep_fs=[.1,.05],
        primary_timestep_fs=.05, horizon_fs=200, registered_primary_pair_batches=[0,1],
        momentum_pair_indices=[0,1,2,3], parent_completed_sweeps=list(range(2100,22001,100)),
        source_sha256={}, parent_root=str(parent_root), parent_manifest_sha256=producer.sha(parent_manifest),
        P128_complete_four_pair_report={'sha256':producer.BASELINE_SHA},
        temperature_K=300., mass_au=producer.MASS_AU, beta=producer.BETA,
        model=actual_design['parent_generation']['cohorts'][0]['expected_settings']['model'])
    (data/'protocol.json').write_text(json.dumps(dict(protocol,created_before_primary_children_utc='synthetic')))
    for seed in producer.SEEDS:
        for dt in (.1,.05):
            for batch in (0,1):
                directory=producer.run_directory(data,seed,dt,200,batch)
                directory.mkdir(parents=True)
                (directory/'report.json').write_text('{"complete":true}\n')
    monkeypatch.setattr(producer,'source_identity',lambda:{})
    monkeypatch.setattr(producer,'verify_manifest',lambda root,spec:{'parents':[{'sampling_flags_passed':True} for _ in range(8)]})
    monkeypatch.setattr(producer,'load_parents',lambda directory,protocol:(None,None,{'BAR_log_TS_quantum_classical_ratio':-2.17}))
    state={'invalid':None,'load_calls':0}
    def row(data,protocol,seed,timestep,horizon,batches,parent_identity):
        state['load_calls']+=1
        value = state['invalid'] if seed==producer.SEEDS[3] and state['invalid'] is not None else .77
        return {'seed':seed,'timestep_fs':timestep,'horizon_fs':200,
            'plateaus':{'80-100':value,'180-200':value},
            'parent_time_half_plateaus':{'180-200':[value,value]},
            'mean_denominator':2.,'minimum_pair_at_1fs_bohr':.7,
            'children_with_sampled_domain_crossing':0,'maximum_energy_error_kbt_per_bead':.01,
            'unique_native_steps':3200000 if timestep==.1 else 6400000,
            'momentum_pairing_hashes':[{'batch':0,'synthetic':True},{'batch':1,'synthetic':True}],
            'curve':np.full(200,value),'receipts':[]}
    monkeypatch.setattr(analysis,'load_seed',row)
    return data,tmp_path/'analysis',state


def test_full_analyzer_retains_all_eight_nonfinite_outcomes_as_failed_inference(synthetic_complete_analysis):
    data,out,state=synthetic_complete_analysis
    state['invalid']=np.nan
    result=analysis.analyze(data,out)
    assert result['registered_full_rate_bead_control_passed'] is False
    assert result['criteria']['complete_registered_cohort'] is True
    assert result['criteria']['all_outcomes_finite'] is False
    assert result['rate_analysis'] is None
    assert len(result['per_timestep_per_seed']['0.05'])==8
    assert result['per_timestep_per_seed']['0.05'][3]['plateaus']['180-200'] is None
    assert result['nonfinite_observations'] and result['kappa_curve_written'] is False
    parsed=json.loads((out/'bead-comparison.json').read_text())
    assert parsed['native_steps_both_timesteps']==76800000
    assert not (out/'kappa-curve.csv').exists()


def test_full_analyzer_retains_nonpositive_finite_plateau(synthetic_complete_analysis):
    data,out,state=synthetic_complete_analysis
    state['invalid']=0.
    result=analysis.analyze(data,out)
    assert result['criteria']['positive_finite_kappa_for_every_chain'] is False
    assert result['registered_full_rate_bead_control_passed'] is False
    assert result['rate_analysis']['statistics'] is None
    assert result['rate_analysis']['per_chain_kappa'][3]==0.
    assert (out/'kappa-curve.csv').exists()


@pytest.mark.parametrize('missing', ['one_seed','one_timestep','one_batch'])
def test_full_analyzer_refuses_partial_cohort_before_loading_any_flux(synthetic_complete_analysis,missing):
    data,out,state=synthetic_complete_analysis
    seed=producer.SEEDS[-1]
    for dt in (.1,.05):
        for batch in (0,1):
            if missing=='one_seed' or (missing=='one_timestep' and dt==.05) or (missing=='one_batch' and batch==1):
                (producer.run_directory(data,seed,dt,200,batch)/'report.json').unlink()
    with pytest.raises(ValueError,match='must complete before inference'):
        analysis.analyze(data,out)
    assert state['load_calls']==0 and not out.exists()
