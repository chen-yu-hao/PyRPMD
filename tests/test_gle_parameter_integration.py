"""Parameter provenance, immutable plans and exact restart integration."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from pyrpmd.engine import PropagationPlan, normalize_thermostat
from pyrpmd.gle_parameters import generate_optimal_gle, load_gle4md


@pytest.fixture
def generated():
    return generate_optimal_gle([0.25, 4.0])


@pytest.fixture
def drift_file(tmp_path):
    # An original, canonical two-variable test bath; no published fit data.
    path = tmp_path / "drift.dat"
    path.write_text("1.0 2.0\n-2.0 1.0\n")
    return path


def make(parameters, *, seed=41, mode="nm-gle", beads=2):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.native_ensemble import NativeEnsemble

    return NativeEnsemble(
        np.zeros((beads, 1, 3)), [1.0], 2.0, 0.025,
        thermostat={"mode": mode, "gle": parameters},
        potential={"kind": "harmonic", "k": 0.7}, seed=seed,
    )


def test_generator_spec_is_resolved_once_and_detached(generated):
    spec = {"mode": "NM_GLE", "gle": {
        "generator": "optimal-sampling", "frequency_range": [0.25, 4.0],
    }}
    before = copy.deepcopy(spec)
    plan = PropagationPlan(thermostat=spec)
    assert spec == before
    assert plan.thermostat_spec == {"mode": "nm-gle", "gle": generated}
    identity = plan.state_dict()
    spec["gle"]["frequency_range"][0] = 0.5
    detached = plan.thermostat_spec
    detached["gle"]["parameter_metadata"]["objective"] = "changed"
    assert plan.state_dict() == identity


def test_piglet_generator_cannot_select_a_canonical_gle_measure():
    with pytest.raises(ValueError, match="(?i)piglet"):
        normalize_thermostat({"mode": "gle", "gle": {
            "generator": "piglet", "frequency_range": [0.2, 2.0],
            "beta": 8.0, "beads": 4,
        }})


def test_imported_plan_retains_content_after_source_changes(drift_file):
    spec = {"mode": "nm-gle", "gle": {
        "file": str(drift_file), "drift_unit": "autime^-1",
    }}
    before = copy.deepcopy(spec)
    plan = PropagationPlan(thermostat=spec)
    saved = plan.state_dict()
    assert spec == before
    drift_file.write_text("2.0 2.0\n-2.0 2.0\n")
    assert plan.state_dict() == saved
    assert PropagationPlan(thermostat=spec).state_dict() != saved


@pytest.mark.parametrize("key,value", [
    ("frequency_range_au", [0.125, 8.0]),
    ("objective", "a different harmonic sampling objective"),
    ("residuals", {"maximum_relative_error": 0.025}),
])
def test_calibration_metadata_changes_native_restart_identity(generated, key, value):
    changed = copy.deepcopy(generated)
    changed["parameter_metadata"][key] = value
    first, second = make(generated), make(changed)
    # The propagated OU map is identical, but its scientific record differs.
    np.testing.assert_array_equal(first._cfg["gle_T"], second._cfg["gle_T"])
    np.testing.assert_array_equal(first._cfg["gle_S"], second._cfg["gle_S"])
    assert first.fingerprint != second.fingerprint
    before = second.state()
    with pytest.raises(ValueError, match="checkpoint physics/backend differs"):
        second.restore(first.state())
    assert second.state()["digest"] == before["digest"]


def test_new_parameter_provenance_is_part_of_native_identity(generated):
    changed = copy.deepcopy(generated)
    changed["provenance"] += "; independent provenance revision"
    assert make(generated).fingerprint != make(changed).fingerprint


def test_native_output_contains_detached_parameter_record(generated):
    engine = make(generated)
    record = engine.run(3)["gle_parameters"]
    assert record == generated
    record["parameter_metadata"]["objective"] = "changed after sampling"
    assert engine.plan.thermostat_spec["gle"] == generated
    assert "gle_parameters" not in make({"A": [[1.0]]}).run(1)


def test_calibration_record_is_in_dynamics_operator_identity(generated):
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.potentials.builtin import HarmonicPotential

    def dynamics(parameters):
        return Dynamics(HarmonicPotential(0.7), [1.0], 2.0, 2, 0.025,
                        thermostat={"mode": "nm-gle", "gle": parameters},
                        native=False, seed=37)

    first = dynamics(generated)
    saved = first.operator_state_dict()
    changed = copy.deepcopy(generated)
    changed["parameter_metadata"]["objective"] = "a different objective"
    second = dynamics(changed)
    with pytest.raises(ValueError, match="checkpoint parameters differ"):
        second.load_operator_state_dict(saved)


@pytest.mark.parametrize("source", ["generated", "imported"])
def test_parameter_records_resume_bitwise(source, generated, drift_file, tmp_path):
    parameters = (generated if source == "generated" else
                  load_gle4md(drift_file, drift_unit="autime^-1", mode="nm-gle"))
    first, split = make(parameters), make(parameters)
    whole = first.run(79, stride=3, record_centroids=True)
    prefix = split.run(23, stride=3, record_centroids=True)
    checkpoint = tmp_path / "parameters.npz"
    split.save(checkpoint)
    resumed = make(parameters, seed=999)
    resumed.load(checkpoint)
    suffix = resumed.run(56, stride=3, record_centroids=True)
    for field in ("samples", "centroids"):
        np.testing.assert_array_equal(
            np.concatenate((prefix[field], suffix[field])), whole[field])
    for field in ("q", "p", "aux", "rng", "digest"):
        np.testing.assert_array_equal(resumed.state()[field], first.state()[field])
    assert not list(tmp_path.glob("*.tmp"))


def test_native_config_resolves_file_relative_to_input_directory(drift_file, monkeypatch, tmp_path):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.native_ensemble import ensemble_from_config

    config = {
        "positions": [[0.0, 0.0, 0.0]], "masses": [1.0],
        "beads": 2, "beta": 2.0, "timestep": 0.025, "seed": 41,
        "potential": {"kind": "harmonic", "k": 0.7},
        "thermostat": {"mode": "NM_GLE", "gle": {
            "file": drift_file.name, "drift_unit": "autime^-1",
        }},
    }
    before = copy.deepcopy(config)
    elsewhere = tmp_path / "unrelated-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    actual = ensemble_from_config(config, base_directory=drift_file.parent)
    expected = make(load_gle4md(drift_file, drift_unit="autime^-1", mode="nm-gle"))
    assert config == before
    assert actual.fingerprint == expected.fingerprint
    actual.run(11)
    expected.run(11)
    assert actual.state()["digest"] == expected.state()["digest"]


def test_native_config_accepts_generation_without_changing_caller(generated):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.native_ensemble import ensemble_from_config

    config = {
        "positions": [[0.0, 0.0, 0.0]], "masses": [1.0],
        "beads": 2, "beta": 2.0, "timestep": 0.025,
        "potential": {"kind": "harmonic", "k": 0.7},
        "thermostat": {"mode": "nm-gle", "gle": {
            "generator": "optimal-sampling", "frequency_range": [0.25, 4.0],
        }},
    }
    before = copy.deepcopy(config)
    actual = ensemble_from_config(config)
    assert config == before
    assert actual.fingerprint == make(generated).fingerprint


def test_legacy_explicit_matrices_keep_their_original_fingerprint_schema():
    # Captured scientific schema before parameter imports/generation existed.
    # Evaluate the scalar OU entries locally rather than pinning libm's final
    # bit or a platform-specific C++ RNG ABI in a hexadecimal hash fixture.
    spec = {"A": [[1.0]], "C": [[1.0]]}
    assert normalize_thermostat({"mode": "gle", "gle": spec}) == {
        "mode": "gle", "gle": spec,
    }
    engine = make(spec, mode="gle", beads=1)
    transition = float(np.exp(-0.025))
    legacy = {
        "atoms": 1, "baro_gamma": 1.0, "baro_tau": 1.0, "barostat": 0,
        "beads": 1, "beta": 2.0, "cayley": True,
        "cell_mask": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        "compressibility": 0.0, "dt": 0.025, "external_stress": [0.0] * 9,
        "fixed": [0.0], "fixed_positions": [],
        "format": "pyrpmd.native-ensemble.v1", "fuse_modes": True,
        "gamma": 1.0, "gamma0": 1.0, "gle_Croot": [1.0],
        "gle_S": [float(np.sqrt(1.0 - transition * transition))],
        "gle_T": [transition], "gle_size": 1, "has_cell": False,
        "initial_cell": np.eye(3).tolist(), "interval": 10, "masses": [1.0],
        "parameters": [0.7], "pbc": [0.0, 0.0, 0.0], "pile_scale": 1.0,
        "potential": 1, "potential_identity": {"kind": "harmonic", "parameters": [0.7]},
        "pressure": 0.0, "reference_cell": np.eye(3).ravel().tolist(),
        "rng_abi": getattr(engine._native, "rng_abi", "std-cxx17-local-build-v1"),
        "tau": 1.0, "thermostat": 6,
    }
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    assert engine.fingerprint == expected
