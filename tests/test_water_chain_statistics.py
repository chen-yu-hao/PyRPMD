"""Chain-level uncertainty, explicit pairing, and independently matched RPMD maps."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.water.analyze_native import (
    OBSERVABLES,basic_rhat,chain_uncertainty,paired_check,
    read_sampling_and_segments,validate_source_contract,
)


def test_independent_chain_means_define_sem_and_student_t_interval():
    values=np.array([.16,.18,.20,.22,.24,.26,.28,.30])
    result=chain_uncertainty(values)
    assert result["chains"]==8 and result["degrees_of_freedom"]==7
    assert result["mean"]==pytest.approx(values.mean())
    assert result["standard_error"]==pytest.approx(values.std(ddof=1)/np.sqrt(8))
    assert result["CI95_halfwidth"]==pytest.approx(2.364624251*result["standard_error"],rel=1e-8)
    single=chain_uncertainty([.22])
    assert single["standard_error"] is None and single["CI95"] is None


def test_paired_comparison_uses_chain_differences_not_two_independent_errors():
    first=np.array([.1,.3,.2,.4])
    result=paired_check(first,first+.005)
    assert result["compatible"] and result["equivalent_within_margin"]
    assert result["paired_difference"]["standard_error"]<1e-16
    assert result["threshold_angstrom2_ps"]==.01
    assert not paired_check(first,first+.04)["compatible"]
    noisy=paired_check(first,first+np.array([-.1,.1,-.2,.2]))
    assert noisy["compatible"] and not noisy["equivalent_within_margin"]
    with pytest.raises(ValueError,match="same seed"):
        paired_check(first,first[:2])


def test_between_chain_offset_is_visible_in_rhat():
    base=np.array([0,1,-1,.2,-.2,1,-1,0,.1,-.1])
    mixed=np.stack([base,base[::-1],np.roll(base,2),np.roll(base,4)])
    separated=mixed+np.arange(4)[:,None]*10
    assert basic_rhat(mixed)<1.1
    assert basic_rhat(separated)>5
    assert basic_rhat([[1,1],[1,1]]) is None


def test_native_rpmd_frequency_convention_explains_short_map_residual():
    pytest.importorskip("openmm")
    from benchmarks.water.native_validation import native_rpmd_map
    default=native_rpmd_map(8,steps=3)
    matched=native_rpmd_map(8,steps=3,match_openmm_hbar=True)
    assert matched["maximum_position_difference_bohr"]<1e-9
    assert matched["maximum_momentum_difference_au"]<2e-8
    assert matched["openmm_to_modern_free_ring_frequency_ratio"]==pytest.approx(1.0000001798,abs=1e-9)
    assert default["maximum_momentum_difference_au"]>matched["maximum_momentum_difference_au"]*10


@pytest.mark.parametrize("bad",[[],[float('nan')],[[1,2],[3,4]]])
def test_invalid_chain_mean_sets_are_not_validated(bad):
    with pytest.raises(ValueError):
        chain_uncertainty(bad)


def test_analysis_only_accepts_registered_source_or_identified_checkpoint_revision():
    identity={"benchmarks/water/native_reference.py":"old-runner","benchmarks/water/qtip4pf.py":"model"}
    protocol={"source_sha256":identity}
    revised=dict(identity,**{"benchmarks/water/native_reference.py":"new-runner"})
    contract={"from_source_sha256":"old-runner","to_source_sha256":"new-runner"}
    validate_source_contract(identity,protocol)
    validate_source_contract(revised,protocol,contract)
    with pytest.raises(ValueError,match="preregistration"):
        validate_source_contract(revised,protocol)
    with pytest.raises(ValueError,match="preregistration"):
        validate_source_contract(dict(revised,**{"benchmarks/water/qtip4pf.py":"changed-model"}),protocol,contract)


def test_segment_stream_crosses_chunks_and_ignores_raw_checkpoint_scalars(tmp_path):
    records=[]
    for index,segments in enumerate(([-1,-1,0,0],[0,0,1,1])):
        frames=len(segments)
        arrays={name:np.zeros((frames,8,3)) if name in (
            "molecular_com_bohr","molecular_velocity_au","oxygen_centroid_bohr") else np.zeros(frames)
            for name in OBSERVABLES}
        arrays["segment"]=np.array(segments)
        arrays["segment_step"]=np.array([1,2,0,1] if index==0 else [2,3,0,1])
        filename=f"chunk-{index}.npz"
        np.savez(tmp_path/filename,**arrays,active_segment=np.array(1),engine_positions_nm=np.zeros((32,3)))
        records.append({"file":filename})
    output=list(read_sampling_and_segments(tmp_path,{"chunks":records,"config":{"cells_per_axis":2}}))
    assert [(kind,index) for kind,index,_ in output]==[("NVE",0),("NVE",1),("sampling",-1)]
    np.testing.assert_array_equal(output[0][2]["segment_step"],[0,1,2,3])
    np.testing.assert_array_equal(output[1][2]["segment_step"],[0,1])
    np.testing.assert_array_equal(output[2][2]["segment_step"],[1,2])
    broken={name:value.copy() for name,value in arrays.items()}
    broken["molecular_com_bohr"][0,0,0]=float("nan")
    np.savez(tmp_path/records[-1]["file"],**broken)
    with pytest.raises(ValueError,match="nonfinite"):
        list(read_sampling_and_segments(tmp_path,{"chunks":records,"config":{"cells_per_axis":2}}))
