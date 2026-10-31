"""Exercise public input parsing, not only direct workflow dictionaries."""
from pathlib import Path

import pytest

from pyrpmd.input import JobSpec, InputError


def rate_spec():
    return JobSpec.from_file(Path(__file__).resolve().parents[1] / "examples/rate/pilot.inp")


def test_public_batch_and_reuse_controls_are_retained():
    spec = rate_spec()
    spec.data.setdefault("parallel", {})["reuse_potentials"] = False
    spec.data.setdefault("recrossing", {})["child_batch_pairs"] = 4
    config, _ = spec.resolve()
    assert config["parallel"]["reuse_potentials"] is False
    assert config["recrossing"]["child_batch_pairs"] == 4


def test_public_trpmd_controls_are_retained():
    spec = rate_spec()
    spec.data.setdefault("recrossing", {}).update(dynamics="trpmd", friction_scale=.5)
    config, _ = spec.resolve()
    assert config["recrossing"]["dynamics"] == "trpmd"
    assert config["recrossing"]["friction_scale"] == .5


@pytest.mark.parametrize("settings", [
    {"child_batch_pairs": True}, {"child_batch_pairs": 0},
    {"child_batch_pairs": 2.5}, {"dynamics": "cmd"},
    {"dynamics": "trpmd", "friction_scale": -1},
    {"dynamics": "trpmd", "friction_scale": True},
    {"dynamics": "trpmd", "friction_scale": float("nan")},
    {"dynamics": "trpmd", "child_batch_pairs": 2},
    {"friction_scale": 1},
])
def test_invalid_controls_rejected_before_loading_potential(settings, monkeypatch):
    spec = rate_spec()
    spec.data.setdefault("recrossing", {}).update(settings)
    monkeypatch.setattr("pyrpmd.potentials.make_potential",
                        lambda *a, **k: pytest.fail("invalid input loaded a potential"))
    with pytest.raises(InputError):
        spec.resolve()


@pytest.mark.parametrize("setting", [
    {"integrator": "velocity-verlet"}, {"loop_backend": "native"},
])
def test_unsupported_trpmd_integrators_fail_during_input_resolution(setting):
    spec = rate_spec()
    spec.data.setdefault("recrossing", {})["dynamics"] = "trpmd"
    spec.data.setdefault("dynamics", {}).update(setting)
    with pytest.raises(InputError, match="TRPMD"):
        spec.resolve()


def test_reuse_flag_is_boolean():
    spec = rate_spec()
    spec.data.setdefault("parallel", {})["reuse_potentials"] = "false"
    with pytest.raises((InputError, ValueError), match="boolean"):
        spec.resolve()


def test_execution_only_nested_table_preserves_absent_table_fingerprint():
    from pyrpmd.propagation import scientific_config
    baseline = {"beads": 8}
    tuned = {"beads": 8, "rate": {"recrossing": {"child_batch_pairs": 4}}}
    assert scientific_config(tuned) == baseline


@pytest.mark.parametrize("parallel", [
    {"mode": "bead", "bead_workers": 2},
    {"backend": "process", "bead_workers": 2},
    {"backend": "bead", "bead_workers": 2},
    {"backend": "mpi", "trajectory_groups": 2, "bead_workers": 2},
])
def test_batch_rejects_bead_proxies_before_sampling(parallel):
    spec = rate_spec()
    spec.data.setdefault("recrossing", {})["child_batch_pairs"] = 4
    spec.data["parallel"] = parallel
    with pytest.raises(InputError, match="requires parallel.bead_workers=1"):
        spec.resolve()


@pytest.mark.parametrize("parallel", [
    {"mode": "bead", "bead_workers": 1},
    {"backend": "bead", "bead_workers": 1},
    {"backend": "process", "trajectory_workers": 2},
    {"backend": "mpi", "trajectory_groups": 2, "bead_workers": 1},
])
def test_batch_accepts_independent_trajectory_workers(parallel):
    spec = rate_spec()
    spec.data.setdefault("recrossing", {})["child_batch_pairs"] = 4
    spec.data["parallel"] = parallel
    config, _ = spec.resolve()
    assert config["recrossing"]["child_batch_pairs"] == 4


def test_batch_rejects_fused_native_loop_before_sampling():
    spec = rate_spec()
    spec.data.setdefault("recrossing", {})["child_batch_pairs"] = 4
    spec.data.setdefault("dynamics", {})["loop_backend"] = "native"
    with pytest.raises(InputError, match="Batched recrossing"):
        spec.resolve()
