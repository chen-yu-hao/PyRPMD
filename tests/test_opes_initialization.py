"""OPES initialization, versioned restart, and live PLUMED equation checks."""
import importlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.cv import Cartesian
from pyrpmd.opes import OPESBias


def make_bias(*, mode="metad", adaptive=False, state=None):
    return OPESBias(Cartesian(0), [.3], 2., barrier=4., bias_factor=5.,
                    mode=mode, adaptive_sigma=adaptive,
                    sigma_min=[.03] if adaptive else None, state=state)


def legacy_empty(**options):
    state = make_bias(**options).state_dict()
    state["schema_version"] = 1
    del state["initial_bias"]
    return state


@pytest.mark.parametrize("mode", ["metad", "explore"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_empty_bias_uses_equation_floor_and_first_kernel_weight(mode, adaptive):
    bias = make_bias(mode=mode, adaptive=adaptive)
    points = np.array([[-3.], [0.], [12.]])
    expected = bias.prefactor * bias.log_epsilon
    np.testing.assert_array_equal(bias.physical_energy(points), np.full(3, expected))
    energy, gradient = bias.physical_energy_gradient(points)
    np.testing.assert_array_equal(energy, np.full(3, -4.))
    np.testing.assert_array_equal(gradient, np.zeros_like(points))
    ring_energy, force = bias.energy_force(np.ones((7, 1, 3)))
    assert ring_energy == -28.
    np.testing.assert_array_equal(force, np.zeros((7, 1, 3)))
    state = bias.state_dict()
    assert state["schema_version"] == 2 and state["initial_bias"] == -4.
    restored = make_bias(mode=mode, adaptive=adaptive, state=state)
    assert restored.state_dict() == state
    event = restored.update([.2], step=1, time=.1)
    assert event["bias_before"] == -4.
    assert event["log_weight"] == (-8. if mode == "metad" else 0.)


@pytest.mark.parametrize("mode", ["metad", "explore"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_legacy_empty_and_nonempty_checkpoints_continue_original_algorithm(mode, adaptive):
    state = legacy_empty(mode=mode, adaptive=adaptive)
    bias = make_bias(mode=mode, adaptive=adaptive, state=state)
    assert bias.state_dict() == state
    assert bias.physical_energy([1.]) == 0.
    event = bias.update([-.3], step=1, time=.1)
    assert event["bias_before"] == event["log_weight"] == 0.
    for index, value in enumerate([.5, -.1, .8], 2):
        bias.update([value], step=index, time=index*.1)
    saved = bias.state_dict()
    assert saved["schema_version"] == 1 and "initial_bias" not in saved
    resumed = make_bias(mode=mode, adaptive=adaptive, state=saved)
    assert resumed.state_dict() == saved
    for index, value in enumerate([-.6, .25, 1.1], 5):
        left = bias.update([value], step=index, time=index*.1)
        right = resumed.update([value], step=index, time=index*.1)
        assert left == right
    assert resumed.state_dict() == bias.state_dict()


def test_legacy_nonempty_reference_uses_saved_weights_without_new_gauge_conversion():
    # Build a two-kernel v1 state from closed-form normalized Gaussians,
    # independently of update(), then check the next weighted deposition.
    state = legacy_empty()
    centers = np.array([-.3, .5])
    width, prefactor = .3, .4
    kernel = np.exp(-.5*((centers[:, None]-centers)/width)**2)/(np.sqrt(2*np.pi)*width)
    first_bias_at_second = prefactor*np.log(kernel[1, 0]/kernel[0, 0]+np.exp(-4/prefactor))
    logs = np.array([0., 2*first_bias_at_second])
    sums = kernel.sum(axis=0)
    z = np.mean(kernel@np.exp(logs)/np.exp(logs).sum())
    state.update(centers=centers[:, None].tolist(), log_weights=logs.tolist(),
                 steps=[1, 2], times=[.1, .2], log_center_sums=np.log(sums).tolist(), log_z=float(np.log(z)))
    bias = make_bias(state=state)
    probe = .1
    density = np.exp(-.5*((probe-centers)/width)**2)@np.exp(logs)/(np.sqrt(2*np.pi)*width*np.exp(logs).sum())
    expected = prefactor*np.log(density/z+np.exp(-4/prefactor))
    assert bias.physical_energy([probe]) == pytest.approx(expected, abs=2e-15)
    assert bias.update([probe], step=3, time=.3)["log_weight"] == pytest.approx(2*expected, abs=4e-15)
    assert bias.state_dict()["log_weights"][0] == 0.


@pytest.mark.parametrize("version", [True, False, 0, 3, 1., "2", None])
def test_ambiguous_or_unknown_state_versions_are_rejected(version):
    state = make_bias().state_dict()
    state["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        make_bias(state=state)


@pytest.mark.parametrize("change", ["v1_with_initial", "v2_without_initial", "wrong_initial", "nan_initial"])
def test_initial_bias_is_part_of_checkpoint_identity(change):
    state = make_bias().state_dict()
    if change == "v1_with_initial":
        state["schema_version"] = 1
    elif change == "v2_without_initial":
        del state["initial_bias"]
    else:
        state["initial_bias"] = 0. if change == "wrong_initial" else float("nan")
    with pytest.raises(ValueError, match="state fields|initial_bias"):
        make_bias(state=state)


@pytest.mark.parametrize("adaptive", [False, True])
def test_legacy_first_weight_cannot_be_relabelled_as_schema_two(adaptive):
    bias = make_bias(adaptive=adaptive)
    bias.update([.2], step=1, time=.1)
    state = bias.state_dict()
    # For one kernel, this change leaves the normalized density and Z
    # unchanged, so normalization checks alone cannot detect the conflict.
    state["log_weights"][0] = 0.
    with pytest.raises(ValueError, match="first kernel weight.*initial_bias"):
        make_bias(adaptive=adaptive, state=state)


def _live_plumed():
    required = os.environ.get("PYRPMD_REQUIRE_PLUMED") == "1"
    try:
        module = importlib.import_module("plumed")
    except ImportError as exc:
        if required:
            pytest.fail(f"PYRPMD_REQUIRE_PLUMED=1 but the Python wrapper is unavailable: {exc}", pytrace=False)
        pytest.skip("optional PLUMED Python wrapper unavailable")
    try:
        kernel = module.Plumed()
    except RuntimeError as exc:
        if "not available" not in str(exc):
            raise
        if required:
            pytest.fail(f"PYRPMD_REQUIRE_PLUMED=1 but libplumedKernel is unavailable: {exc}", pytrace=False)
        pytest.skip("optional libplumedKernel unavailable")
    kernel.finalize()


def test_real_optional_plumed_fixed_opes_identical_deposition_history(tmp_path):
    _live_plumed()
    path = Path(__file__).parents[1]/"benchmarks"/"tier2_literature"/"sampling.py"
    spec = importlib.util.spec_from_file_location("sampling_literature", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    results = benchmark.plumed_comparator(tmp_path, depositions=12, probes=11)
    for result in results:
        assert result["pyrpmd_state_schema_version"] == 2
        assert result["pyrpmd_initial_empty_bias"] == result["plumed_initial_empty_bias"] == -20.
        assert result["max_energy_difference"] < 2e-11
        assert result["max_force_difference"] < 2e-10
    assert results[0]["legacy_v1_max_energy_difference"] > 1.
    assert results[0]["legacy_v1_max_force_difference"] > 1.
    assert results[1]["legacy_v1_max_energy_difference"] < 2e-11
