"""A timestep control must preserve thermostat/observation/production times."""
import pytest
from benchmarks.h3.sampling_control import resolved_control
from pyrpmd.constants import AU_TIME_FS


def test_timestep_halving_preserves_all_physical_intervals(tmp_path):
    coarse, system = resolved_control(tmp_path, tmp_path/"offline.so", timestep_fs=.1)
    fine, other = resolved_control(tmp_path, tmp_path/"offline.so", timestep_fs=.05)
    assert coarse["beads"] == fine["beads"] == 1
    assert system.to_dict() == other.to_dict()
    for table, keys in {
        "sampling": ("equilibration_steps", "production_steps", "sample_interval", "interval_steps", "andersen_interval"),
        "recrossing": ("equilibration_steps", "parent_interval_steps", "child_steps"),
        "output": ("checkpoint_steps",),
    }.items():
        for key in keys:
            assert fine[table][key] == 2*coarse[table][key]
    assert fine["dt_au"]*AU_TIME_FS == pytest.approx(.05)
    assert fine["sampling"]["dt_au"] == fine["dt_au"]
    assert coarse["sampling"]["dt_au"] == coarse["dt_au"]
    assert coarse["reaction"]["rinf"] == fine["reaction"]["rinf"] == "30 bohr"


@pytest.mark.parametrize("kwargs", [{"beads":True}, {"beads":0}, {"timestep_fs":float("nan")},
                                     {"timestep_fs":.03}, {"production_fs":.5},
                                     {"equilibration_fs":-1.}])
def test_invalid_controls_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        resolved_control(tmp_path, tmp_path/"offline.so", **kwargs)
