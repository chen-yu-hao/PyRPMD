"""Private compact observations preserve the public propagation contract."""
import numpy as np
import pytest

thermal = pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd.potentials import HarmonicPotential


def engine(rng="mt19937_64", beads=32, atoms=9):
    q = np.random.default_rng(115).normal(scale=.1, size=(beads, atoms, 3))
    return NativeEnsemble(q, np.linspace(.7, 3.1, atoms), 2.3, .01,
                          thermostat={"mode": "pile-l", "rng": rng}, seed=115,
                          potential={"kind": "harmonic", "k": .5})


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()  # includes signed zero / NaN bits


@pytest.mark.parametrize("rng", ["mt19937_64", "philox-v1"])
@pytest.mark.parametrize("beads,atoms", [(1, 9), (7, 9), (32, 9), (32, 385)])
@pytest.mark.parametrize("record_centroids", [False, True])
def test_compact_observer_matches_full_common_records(rng, beads, atoms, record_centroids):
    full, compact = engine(rng, beads, atoms), engine(rng, beads, atoms)
    mask = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    reference = full.run_prefix(7, observe=mask, record_energy=True, check_energy=True)
    result = compact.run_prefix(7, observe=mask, record_energy=True, check_energy=True,
                                record_centroids=record_centroids, record_moments=False,
                                _positions_only=True)
    exact(result["samples"], reference["samples"])
    assert result["centroid_momenta"] is None
    assert result["radii_gyration_squared"] is None
    if record_centroids:
        exact(result["centroids"], reference["centroids"])
    else:
        assert result["centroids"] is None
    assert result["completed"] == reference["completed"] == 7
    assert compact.state()["digest"] == full.state()["digest"]


@pytest.mark.parametrize("record_centroids", [False, True])
@pytest.mark.parametrize("stride", [1, 3, 11])
@pytest.mark.parametrize("rng", ["mt19937_64", "philox-v1"])
def test_public_propagation_matches_legacy_observer(monkeypatch, record_centroids, stride, rng):
    def run():
        dyn = Dynamics(HarmonicPotential(k=.5), np.linspace(.7, 3.1, 13), 2.3, 32, .01,
                       thermostat={"mode": "pile-l", "rng": rng}, seed=115, block_steps=4)
        q = np.random.default_rng(115).normal(scale=.1, size=(32, 13, 3))
        state = State(q, np.zeros_like(q), step=5, time=.05)
        result = dyn.propagate(state, 10, sample_interval=stride, record_centroids=record_centroids)
        return state, result, dyn.operator_state_dict()
    monkeypatch.setattr(thermal, "COMMON_POSITION_OBSERVATIONS_API", 0)
    reference, expected, reference_ops = run()
    monkeypatch.setattr(thermal, "COMMON_POSITION_OBSERVATIONS_API", 1)
    current, actual, actual_ops = run()
    exact(current.q, reference.q); exact(current.p, reference.p)
    exact(actual.energies, expected.energies)
    if record_centroids:
        exact(actual.centroids, expected.centroids)
    assert (actual.initial_energy, actual.minimum_energy, actual.maximum_energy) == (
        expected.initial_energy, expected.minimum_energy, expected.maximum_energy)
    assert (current.step, current.time) == (reference.step, reference.time)
    assert actual_ops == reference_ops


def test_position_only_requires_capability(monkeypatch):
    e = engine()
    before = e.state()["digest"]
    monkeypatch.setattr(thermal, "COMMON_POSITION_OBSERVATIONS_API", 0)
    with pytest.raises(RuntimeError, match="rebuilt"):
        e.run_prefix(3, record_moments=False, _positions_only=True)
    assert e.state()["digest"] == before


@pytest.mark.parametrize("options", [{}, {"record_moments": False, "energy_parts": True},
                                     {"record_moments": False, "ring_radii": True}])
def test_position_only_rejects_mixed_record_contract(options):
    e = engine(); before = e.state()["digest"]
    with pytest.raises(ValueError, match="position-only"):
        e.run_prefix(3, _positions_only=True, **options)
    assert e.state()["digest"] == before


def test_position_only_checks_energy_without_records():
    q = np.zeros((8, 9, 3)); p = np.full_like(q, 1e155)
    e = NativeEnsemble(q, np.ones(9), 2., .01, p=p, thermostat="pile-l",
                       potential={"kind": "harmonic", "k": .5})
    before = e.state()["digest"]
    result = e.run_prefix(3, observe=np.zeros(3, dtype=np.uint8), record_energy=False,
                          check_energy=True, record_centroids=False, record_moments=False,
                          _positions_only=True)
    assert result["completed"] == 0
    assert result["error"] and "nonfinite" in result["error"]
    assert result["samples"].shape[0] == 0
    assert e.state()["digest"] == before
