"""Native composed blocks retain exactly the valid state/observation prefix."""
import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.native_ensemble import NativeEnsemble
from test_native_propagation import native_library, CompiledPotential
from test_native_python_bridge import PeriodicFree


@pytest.mark.parametrize("thermostat", ["pile-l", "svr", "nhc",
    {"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]]}}])
def test_native_prefix_rolls_back_failed_step_and_restarts_exactly(native_library, thermostat):
    q = np.random.default_rng(712).normal(scale=.1, size=(3, 3, 3))
    p = np.random.default_rng(713).normal(scale=.1, size=q.shape)
    options = dict(q=q, p=p, masses=[1., 2., 3.], beta=2., dt=.01,
                   thermostat=thermostat, potential_identity="prefix-harmonic-v1", seed=91)
    reference = NativeEnsemble(**options, potential=CompiledPotential(native_library))
    first = reference.run(4, stride=3, record_centroids=True, record_moments=True)
    expected_state = reference.state()
    provider = CompiledPotential(native_library)
    actual = NativeEnsemble(**options, potential=provider)
    provider.lib.reset(5, 0)  # Five new force calls: four accepted steps, then failure.
    result = actual.run_prefix(10, stride=3)
    assert result["completed"] == 4
    assert "native PES" in result["error"]
    assert actual.state()["digest"] == expected_state["digest"]
    for field in ("samples", "centroids", "velocities", "radii_gyration_squared"):
        np.testing.assert_array_equal(result[field], first[field])
    assert result["python_step_callbacks"] == 0
    provider.lib.reset(-1, 0)
    expected = reference.run(6, stride=3, record_centroids=True, record_moments=True)
    resumed = actual.run_prefix(6, stride=3)
    assert resumed["completed"] == 6 and resumed["error"] is None
    np.testing.assert_array_equal(resumed["samples"], expected["samples"])
    assert actual.state()["digest"] == reference.state()["digest"]


def test_observation_failure_keeps_only_prior_unobserved_steps():
    options = dict(q=np.zeros((1, 1, 3)), p=np.ones((1, 1, 3))*1e200,
                   masses=[1.], beta=2., dt=.01, thermostat="none")
    expected, actual = NativeEnsemble(**options), NativeEnsemble(**options)
    expected.run(2, stride=3)
    result = actual.run_prefix(10, stride=3)
    assert result["completed"] == 2
    assert "nonfinite ensemble observation" in result["error"]
    assert result["samples"].shape == (0, 18)
    assert actual.state()["digest"] == expected.state()["digest"]


def test_python_prefix_restores_provider_cell_to_last_accepted_step():
    options = dict(q=np.ones((3, 2, 3))*.2, p=np.ones((3, 2, 3))*.1,
                   masses=[1., 2.], beta=2., dt=.01, seed=91, cell=np.eye(3)*2,
                   thermostat={"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]]}},
                   barostat={"mode": "mttk", "pressure": .5, "tau": 1.},
                   allow_python=True, potential_identity="periodic-free-prefix-v1")
    reference = NativeEnsemble(**options, potential=PeriodicFree())
    records = reference.run(2, stride=2, record_centroids=True, record_moments=True)
    provider = PeriodicFree()
    actual = NativeEnsemble(**options, potential=provider)
    provider.fail_on = 4  # Initial call, two accepted endpoints, failed third step.
    result = actual.run_prefix(7, stride=2)
    assert result["completed"] == 2
    assert "controlled Python PES failure" in result["error"]
    assert result["python_step_callbacks"] == 3
    assert actual.state()["digest"] == reference.state()["digest"]
    np.testing.assert_array_equal(provider.cell, actual.state()["cell"])
    np.testing.assert_array_equal(result["samples"], records["samples"])
    provider.fail_on = None
    repeated = actual.run_prefix(5, stride=2)
    expected = reference.run(5, stride=2, record_centroids=True, record_moments=True)
    np.testing.assert_array_equal(repeated["samples"], expected["samples"])
    assert actual.state()["digest"] == reference.state()["digest"]


def test_empty_native_prefix_is_successful_and_unchanged():
    e = NativeEnsemble(np.zeros((1, 1, 3)), [1.], 2., .01)
    before = e.state()
    result = e.run_prefix(0)
    assert result["completed"] == 0 and result["error"] is None
    assert result["samples"].shape == (0, 18)
    assert e.state()["digest"] == before["digest"]


def extreme_free(**overrides):
    options = dict(q=np.zeros((1, 1, 3)), p=np.full((1, 1, 3), 1e200),
                   masses=[1.], beta=2., dt=.01, thermostat="none")
    options.update(overrides)
    return NativeEnsemble(**options)


def test_common_mask_retains_unobserved_prefix_before_energy_overflow():
    expected, actual = extreme_free(), extreme_free()
    expected.run_prefix(2, observe=[False, False])
    result = actual.run_prefix(5, observe=[False, False, True, False, True])
    assert result["completed"] == 2
    assert "nonfinite total ring energy" in result["error"]
    assert result["samples"].shape == (0, 18)
    assert result["centroid_momenta"].shape == (0, 1, 3)
    assert actual.state()["digest"] == expected.state()["digest"]


@pytest.mark.parametrize("record_energy", [False, True])
def test_common_unobserved_steps_skip_energy_reductions(record_energy):
    actual = extreme_free()
    result = actual.run_prefix(4, observe=[False]*4, record_energy=record_energy)
    assert result["completed"] == 4 and result["error"] is None
    assert result["samples"].shape == (0, 18)
    assert result["centroids"].shape == result["centroid_momenta"].shape == (0, 1, 3)
    assert result["radii_gyration_squared"] is None and result["energy_parts"] is None


def test_common_disabled_energy_records_finite_qp_without_squared_estimators():
    actual = extreme_free()
    result = actual.run_prefix(3, observe=[True, False, True], record_energy=False)
    assert result["completed"] == 3 and result["error"] is None
    np.testing.assert_array_equal(result["samples"][:, 0], [1, 3])
    assert np.isnan(result["samples"][:, 2:]).all()
    assert np.isfinite(result["centroids"]).all()
    np.testing.assert_array_equal(result["centroid_momenta"], np.full((2, 1, 3), 1e200))
    assert result["velocities"] is None and result["radii_gyration_squared"] is None


def test_common_momenta_do_not_require_finite_velocity():
    # p/m overflows, but dt*p/m, the physical drift actually requested, is finite.
    actual = extreme_free(p=np.full((1, 1, 3), 1e200), masses=[1e-200], dt=1e-200)
    result = actual.run_prefix(2, record_energy=False)
    assert result["completed"] == 2 and result["error"] is None
    np.testing.assert_array_equal(result["centroid_momenta"], np.full((2, 1, 3), 1e200))
    assert np.isfinite(result["centroids"]).all()


def test_common_check_energy_validates_initial_unobserved_state_atomically():
    actual = extreme_free()
    saved = actual.state()
    result = actual.run_prefix(4, observe=[False]*4, record_energy=False, check_energy=True)
    assert result["completed"] == 0 and "nonfinite total ring energy" in result["error"]
    assert actual.state()["digest"] == saved["digest"]


def test_common_check_energy_validates_each_unobserved_step():
    options = dict(q=np.full((1, 1, 3), 1e148), p=np.zeros((1, 1, 3)),
                   masses=[1.], beta=2., dt=10., thermostat="none",
                   potential={"kind": "harmonic", "k": 1.})
    unchecked, checked, reference = (NativeEnsemble(**options) for _ in range(3))
    no_check = unchecked.run_prefix(3, observe=[False]*3, record_energy=False)
    assert no_check["completed"] == 3 and no_check["error"] is None
    result = checked.run_prefix(3, observe=[False]*3, record_energy=False, check_energy=True)
    assert result["completed"] == 2 and "nonfinite total ring energy" in result["error"]
    reference.run_prefix(2, observe=[False]*2, record_energy=False)
    assert checked.state()["digest"] == reference.state()["digest"]


def test_common_ring_radius_validation_is_opt_in_and_masked():
    q = np.zeros((3, 1, 3));q[:, 0, 0] = [-1e200, 0., 1e200]
    options = dict(q=q, p=np.zeros_like(q))
    skipped = extreme_free(**options)
    result = skipped.run_prefix(3, record_energy=False)
    assert result["completed"] == 3 and result["error"] is None
    checked = extreme_free(**options)
    result = checked.run_prefix(3, observe=[False, True, False], record_energy=False, ring_radii=True)
    assert result["completed"] == 1 and "nonfinite ring radius" in result["error"]
    reference = extreme_free(**options)
    reference.run_prefix(1, observe=[False], record_energy=False)
    assert checked.state()["digest"] == reference.state()["digest"]


def test_common_requested_energy_parts_validate_even_without_record_energy():
    actual = extreme_free()
    saved = actual.state()
    result = actual.run_prefix(1, record_energy=False, energy_parts=True)
    assert result["completed"] == 0 and "nonfinite total ring energy" in result["error"]
    assert result["energy_parts"].shape == (0, 4)
    assert actual.state()["digest"] == saved["digest"]


def test_common_sparse_diagnostics_match_dense_records_and_restart(native_library):
    q = np.random.default_rng(77).normal(scale=.1, size=(3, 3, 3))
    options = dict(q=q, p=q*.2, masses=[1., 2., 3.], beta=2., dt=.01,
                   thermostat="pile-l", seed=44, potential_identity="prefix-common-harmonic-v1")
    reference = NativeEnsemble(**options, potential=CompiledPotential(native_library))
    expected = reference.run_prefix(4, record_energy=True, energy_parts=True, ring_radii=True)
    saved = reference.state()
    provider = CompiledPotential(native_library)
    actual = NativeEnsemble(**options, potential=provider)
    provider.lib.reset(5, 0)
    mask = np.array([False, True, False, True, True, False, True])
    result = actual.run_prefix(7, observe=mask, energy_parts=True, ring_radii=True)
    assert result["completed"] == 4 and "native PES" in result["error"]
    assert actual.state()["digest"] == saved["digest"]
    for field in ("samples", "centroids", "centroid_momenta", "energy_parts", "radii_gyration_squared"):
        np.testing.assert_array_equal(result[field], expected[field][mask[:4]])
    parts = result["energy_parts"]
    hamiltonian = result["samples"][:, result["columns"].index("ring_polymer_hamiltonian")]
    np.testing.assert_allclose(parts.sum(axis=1)*3, hamiltonian, rtol=2e-15)


@pytest.mark.parametrize("options", [
    {"observe": [True]}, {"observe": [0., 1.]}, {"observe": [0, 2]},
    {"record_energy": "yes"}, {"check_energy": 1}, {"energy_parts": 1},
    {"ring_radii": 1}, {"observe": [True, False], "stride": 2},
    {"observe": [True, False], "record_moments": False},
])
def test_common_observation_validation_is_atomic(options):
    actual = extreme_free(p=np.ones((1, 1, 3)))
    saved = actual.state()
    with pytest.raises(ValueError):
        actual.run_prefix(2, **options)
    assert actual.state()["digest"] == saved["digest"]
