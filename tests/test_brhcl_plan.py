"""The benchmark's saved clock must represent the requested physical work."""
import pytest

from benchmarks.brhcl.scan import plan


def test_long_run_can_save_at_thermostat_interval_without_growing_mbar_matrix():
    settings = plan(production_fs=10000., equilibration_fs=2000., save_interval_fs=10.)
    assert settings["production_steps"] * settings["dt_fs"] == 10000.
    assert settings["equilibration_steps"] * settings["dt_fs"] == 2000.
    assert settings["sample_interval"] == settings["andersen_interval"] == 100
    assert settings["samples_per_stream"] == 1000
    assert settings["planned_total_samples"] == 242000


@pytest.mark.parametrize("options", [
    {"save_interval_fs": 3.},
    {"save_interval_fs": 0.},
    {"save_interval_fs": float("nan")},
    {"save_interval_fs": 10., "production_fs": 1001.},
    {"save_interval_fs": 10., "equilibration_fs": 501.},
    {"dt_fs": .3},
    {"production_fs": 1000.04},
])
def test_unrepresentable_save_or_md_clocks_are_rejected(options):
    with pytest.raises(ValueError):
        plan(**options)


def test_zero_mc_plan_describes_the_sampler_actually_used():
    settings = plan(orientation_mc_trials=0)
    assert "Metropolis" not in settings["sampler"]
