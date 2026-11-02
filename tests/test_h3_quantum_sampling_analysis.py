"""Independent uncertainty and immutable-prefix controls for the audit."""
import json
import numpy as np
import pytest
from scipy.stats import t as student_t
from benchmarks.h3.quantum_sampling_analyze import cluster_statistics, load_registered_prefix
from benchmarks.h3.quantum_sampling_campaign import array_hash


def test_cluster_jackknife_equals_known_mean_standard_error():
    x=np.array([-.2,0.,0.,.2]);n=len(x)
    loo=(x.sum()-x)/(n-1)
    halves=np.column_stack((x-.01,x+.01))
    result=cluster_statistics(x.mean(),x,loo,halves,np.ones(n)*.04)
    expected=x.std(ddof=1)/np.sqrt(n)
    assert result['jackknife_log_standard_error']==pytest.approx(expected)
    assert result['primary_log_standard_error']==pytest.approx(expected)
    assert result['primary_student_t_95_log_halfwidth']==pytest.approx(student_t.ppf(.975,3)*expected)
    assert result['paired_time_half_log_difference']==pytest.approx(.02)
    # Exact point agreement cannot make a wide uncertainty interval precise.
    assert result['primary_log_standard_error']>.03


def test_whole_ladder_statistics_are_invariant_to_seed_order_and_units():
    x=np.array([-2.,-1.9,-2.05,-2.02]);loo=(x.sum()-x)/3
    halves=np.column_stack((x-.02*np.arange(4),x+.02*np.arange(4)))
    a=cluster_statistics(x.mean(),x,loo,halves,np.full(4,.02))
    perm=[2,0,3,1];shift=13.
    b=cluster_statistics(x.mean()+shift,x[perm]+shift,loo[perm]+shift,
                         halves[perm]+shift,np.full(4,.02))
    for key in ['primary_log_standard_error','paired_time_half_log_difference',
                'paired_time_half_standard_error','heterogeneity_p_approximate']:
        assert a[key]==pytest.approx(b[key],abs=3e-14)
    np.testing.assert_allclose(np.array(b['primary_student_t_95_interval'])/np.exp(shift),
                               a['primary_student_t_95_interval'],rtol=2e-14)


def test_analysis_verifies_registered_prefix_even_after_refinement(tmp_path):
    samples=np.arange(80,dtype=float).reshape(20,4)
    stage=tmp_path/'stages'/'12';stage.mkdir(parents=True)
    receipt={'production_sweeps':12,'samples_prefix_sha256':array_hash(samples[:12])}
    (stage/'stage.json').write_text(json.dumps(receipt))
    np.savez_compressed(tmp_path/'checkpoint.npz',samples=samples)
    actual,metadata=load_registered_prefix(tmp_path,12)
    np.testing.assert_array_equal(actual,samples[:12]);assert metadata['receipt']==receipt
    samples[3,1]+=1
    np.savez_compressed(tmp_path/'checkpoint.npz',samples=samples)
    with pytest.raises(ValueError,match='checksum'):
        load_registered_prefix(tmp_path,12)
