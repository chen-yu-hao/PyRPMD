"""Reaction identity does not imply that a saddle or channel was checked."""

import copy
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.input_reaction import normalize_reaction_input
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.potentials import PESResult
from pyrpmd.potentials.python import import_object
from pyrpmd.workflows.prepare import prepare_rate


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rate"
ToyExchange = import_object(str(EXAMPLE / "toy_exchange.py") + ":ToyExchange")
SADDLE_DISTANCE = 1.2 + np.log(3.0) / 1.2


def _job(*, displaced=False):
    distance = SADDLE_DISTANCE
    geometry = [[-distance, 0.0, 0.0], [0.0, 0.0, 0.0], [distance, 0.0, 0.0]]
    if displaced:
        geometry[0][0] -= 0.15
        geometry[0][1] = 0.03
        geometry[2][0] += 0.08
    return {
        "schema_version": 2,
        "task": "rate",
        "temperature": "1000 K",
        "beads": 1,
        "system": {"symbols": ["H", "H", "H"], "geometry": geometry, "geometry_unit": "bohr"},
        "potential": {"provider": "python", "factory": str(EXAMPLE / "toy_exchange.py") + ":build"},
        "reaction": {
            "definition": "explicit",
            "reactant_fragments": [[0], [1, 2]],
            "product_fragments": [[0, 1], [2]],
            "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]],
            "rinf": "5 angstrom",
        },
        "preparation": {"ts_action": "validate", "check_channel": False},
        "dynamics": {"native": False, "timestep": "0.1 fs"},
        "sampling": {"equilibration_steps": 0, "production_steps": 2, "sample_interval": 1},
        "umbrella": {"centers": [0.0, 0.5, 1.0], "stiffness": "0.16 hartree"},
        "analysis": {"bins": 4, "bootstrap": 0},
        "parallel": {"backend": "serial"},
    }


def test_explicit_definition_checks_saddle_without_refining_or_inferring_channel():
    job = _job()
    original = copy.deepcopy(job)
    config, system = JobSpec.from_mapping(job).resolve()
    geometry = system.geometry.copy()
    provider = Mock(wraps=ToyExchange())
    result = prepare_rate(config, system, provider, numerical=True)

    assert result["ready"] and result["saddle_valid"]
    assert result["channel_consistent"] is None
    assert result["ts_action"] == "validate"
    assert not result["refinement_performed"]
    assert "refinement" not in result and "trace_endpoints" not in result
    assert provider.evaluate.call_count >= 1 + 2 * geometry.size
    np.testing.assert_array_equal(result["transition_state"], geometry)
    np.testing.assert_array_equal(system.geometry, geometry)
    assert job == original


def test_explicit_bad_ts_no_longer_passes_when_new_controls_selected():
    config, system = JobSpec.from_mapping(_job(displaced=True)).resolve()
    with pytest.raises(ValueError, match="not stationary"):
        prepare_rate(config, system, ToyExchange(), numerical=True)


def test_schema_two_defaults_to_saddle_validation_even_without_preparation_table():
    job = _job(displaced=True)
    job.pop("preparation")
    job["reaction"].pop("definition")
    config, system = JobSpec.from_mapping(job).resolve()
    assert config["reaction"]["definition"] == "explicit"
    assert config["reaction"]["preparation"]["ts_action"] == "validate"
    with pytest.raises(ValueError, match="not stationary"):
        prepare_rate(config, system, ToyExchange(), numerical=True)


def test_new_controls_opt_in_on_schema_one():
    job = _job(displaced=True)
    job["schema_version"] = 1
    config, system = JobSpec.from_mapping(job).resolve()
    with pytest.raises(ValueError, match="not stationary"):
        prepare_rate(config, system, ToyExchange(), numerical=True)


def test_explicit_nonnumerical_prepare_defers_without_evaluating_or_constructing_pes():
    config, system = JobSpec.from_mapping(_job()).resolve()
    provider = Mock()
    with patch("pyrpmd.potentials.make_potential") as factory:
        result = prepare_rate(config, system, provider, numerical=False)
        owned = prepare_rate(config, system, numerical=False)
    assert not result["ready"] and not owned["ready"]
    assert result["requires_numerical_preparation"]
    assert result["saddle_valid"] is None and result["channel_consistent"] is None
    provider.evaluate.assert_not_called()
    factory.assert_not_called()


def test_refinement_is_an_independent_explicit_action():
    job = _job(displaced=True)
    job["preparation"]["ts_action"] = "refine"
    config, system = JobSpec.from_mapping(job).resolve()
    original = system.geometry.copy()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert result["ready"] and result["saddle_valid"]
    assert result["refinement_performed"] and result["refinement"]["converged"]
    assert result["channel_consistent"] is None
    assert not np.array_equal(result["transition_state"], original)
    np.testing.assert_array_equal(system.geometry, original)


def test_channel_check_respects_explicit_atom_mapping_and_direction():
    job = _job()
    job["preparation"]["check_channel"] = True
    config, system = JobSpec.from_mapping(job).resolve()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert result["ready"] and result["saddle_valid"] and result["channel_consistent"]
    assert result["reaction"]["fragments"] == [[0], [1, 2]]
    assert result["reaction"]["forming_bonds"] == [[0, 1]]
    assert result["reaction"]["breaking_bonds"] == [[1, 2]]
    assert len(result["trace_endpoints"]) == 2
    assert not result["refinement_performed"]


def test_reversed_explicit_direction_is_not_overwritten_by_mode_sign():
    job = _job()
    job["preparation"]["check_channel"] = True
    job["reaction"].update(
        reactant_fragments=[[0, 1], [2]], product_fragments=[[0], [1, 2]],
        forming_bonds=[[1, 2]], breaking_bonds=[[0, 1]],
    )
    config, system = JobSpec.from_mapping(job).resolve()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert result["ready"] and result["channel_consistent"]
    assert result["reaction"]["fragments"] == [[0, 1], [2]]
    assert result["reaction"]["forming_bonds"] == [[1, 2]]


@pytest.mark.parametrize("definition", ["explicit", "infer"])
def test_channel_validation_preserves_declared_bond_pairing_and_fragment_order(definition):
    geometry = np.array([[0., 0., 0.], [1., 1., 0.], [2., -.5, 0.], [3., 1., .2]])

    class Saddle:
        def evaluate(self, positions):
            delta = positions - geometry
            curvature = np.ones((4, 3))
            curvature[0, 0] = -1.
            return PESResult(
                np.sum(0.5 * curvature * delta**2, axis=(1, 2)),
                -curvature * delta,
            )

    job = _job()
    job["system"].update(symbols=["H"] * 4, geometry=geometry.tolist())
    job["reaction"].update(
        definition=definition,
        reactant_fragments=[[1, 2, 3], [0]],
        product_fragments=[[3], [0, 1, 2]],
        forming_bonds=[[0, 1], [0, 2]],
        breaking_bonds=[[2, 3], [1, 3]],
    )
    job["preparation"]["check_channel"] = True
    config, system = JobSpec.from_mapping(job).resolve()
    requested = copy.deepcopy(config["reaction"])
    with patch("pyrpmd.workflows.prepare.downhill_trace", side_effect=[
        {"geometry": geometry}, {"geometry": geometry},
    ]), patch("pyrpmd.workflows.prepare.bonding_graph", side_effect=[
        {(1, 2), (1, 3), (2, 3)}, {(0, 1), (0, 2), (1, 2)},
    ]):
        result = prepare_rate(config, system, Saddle(), numerical=True)
    assert result["ready"] and result["saddle_valid"] and result["channel_consistent"]
    prepared = result["reaction"]
    for key in ("fragments", "product_fragments", "forming_bonds", "breaking_bonds"):
        assert prepared[key] == requested[key]

    def coordinate(reaction):
        return RPMDrateCoordinate(
            reaction["fragments"], system.masses, geometry,
            reaction["forming_bonds"], reaction["breaking_bonds"], reaction["rinf_au"],
        )

    probe = geometry.copy()
    probe[1] += [0.1, -0.3, 0.2]
    np.testing.assert_array_equal(
        coordinate(prepared).centroid_value_gradient(probe)[0],
        coordinate(requested).centroid_value_gradient(probe)[0],
    )


def test_saddle_valid_does_not_certify_a_wrong_product_partition():
    job = _job()
    job["preparation"]["check_channel"] = True
    job["reaction"]["product_fragments"] = [[0, 2], [1]]
    config, system = JobSpec.from_mapping(job).resolve()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert result["saddle_valid"]
    assert not result["ready"] and not result["channel_consistent"]
    assert result["candidates"] == []


def test_stationary_minimum_is_rejected_despite_explicit_reaction():
    job = _job()
    center = np.asarray(job["system"]["geometry"])

    class Harmonic:
        def evaluate(self, positions):
            delta = positions - center
            return PESResult(0.5 * np.sum(delta**2, axis=(1, 2)), -delta)

    config, system = JobSpec.from_mapping(job).resolve()
    result = prepare_rate(config, system, Harmonic(), numerical=True)
    assert not result["ready"] and not result["saddle_valid"]
    assert result["channel_consistent"] is None


def test_inference_requires_an_explicit_mode_and_reports_derived_reaction():
    job = _job()
    job["reaction"] = {"rinf": "5 angstrom"}
    job.pop("preparation")
    with pytest.raises(InputError, match="definition='infer'"):
        JobSpec.from_mapping(job).resolve()
    job["reaction"]["definition"] = "infer"
    config, system = JobSpec.from_mapping(job).resolve()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert result["ready"] and result["saddle_valid"] and result["channel_consistent"]
    assert result["reaction_definition"] == "infer"
    assert result["reaction"]["product_fragments"]


def test_none_is_an_explicit_unchecked_path_and_does_not_call_pes():
    job = _job(displaced=True)
    job["preparation"]["ts_action"] = "none"
    config, system = JobSpec.from_mapping(job).resolve()
    provider = Mock()
    result = prepare_rate(config, system, provider, numerical=True)
    assert result["ready"] and result["ts_action"] == "none"
    assert result["saddle_valid"] is None and result["channel_consistent"] is None
    provider.evaluate.assert_not_called()


def test_old_explicit_input_still_has_legacy_semantics():
    job = _job(displaced=True)
    job["schema_version"] = 1
    job.pop("preparation")
    reaction = job["reaction"]
    reaction["fragments"] = reaction.pop("reactant_fragments")
    reaction.pop("product_fragments")
    reaction.pop("definition")
    config, system = JobSpec.from_mapping(job).resolve()
    provider = Mock()
    result = prepare_rate(config, system, provider, numerical=True)
    assert result["ready"] and result["ts_action"] == "legacy"
    assert result["saddle_valid"] is None
    provider.evaluate.assert_not_called()


@pytest.mark.parametrize("preparation", [
    {"ts_action": "guess"},
    {"ts_action": "validate", "refine": False},
    {"check_channel": "yes"},
    {"ts_action": "none", "check_channel": True},
])
def test_conflicting_or_invalid_preparation_controls_fail_before_pes(preparation):
    job = _job()
    job["preparation"] = preparation
    with patch("pyrpmd.potentials.make_potential") as factory:
        with pytest.raises(InputError):
            JobSpec.from_mapping(job).resolve()
    factory.assert_not_called()


def test_inference_cannot_disable_channel_check():
    job = _job()
    job["reaction"] = {"definition": "infer"}
    with pytest.raises(InputError, match="check_channel=true"):
        JobSpec.from_mapping(job).resolve()


def test_fragment_aliases_and_preparation_locations_do_not_silently_override():
    job = _job()
    job["reaction"]["fragments"] = [[0], [1, 2]]
    with pytest.raises(InputError, match="not both"):
        JobSpec.from_mapping(job).resolve()
    job = _job()
    job["reaction"]["preparation"] = {"refine": True}
    with pytest.raises(InputError, match="not both"):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("fragments", [[[0], [1, 1, 2]], [[False], [1, 2]], [[], [0, 1, 2]]])
def test_fragment_identity_rejects_duplicates_booleans_and_empty_groups(fragments):
    job = _job()
    job["reaction"]["reactant_fragments"] = fragments
    with pytest.raises(InputError):
        JobSpec.from_mapping(job).resolve()


def test_product_partition_coverage_is_checked_before_numerical_work():
    job = _job()
    config, system = JobSpec.from_mapping(job).resolve()
    job["reaction"]["product_fragments"] = [[0], [1]]
    with patch("pyrpmd.potentials.make_potential") as factory:
        with pytest.raises(InputError, match="partition all atoms"):
            JobSpec.from_mapping(job).resolve()
    factory.assert_not_called()
    config["reaction"]["product_fragments"] = [[0], [1]]
    provider = Mock()
    with pytest.raises(ValueError, match="partition all atoms"):
        prepare_rate(config, system, provider, numerical=True)
    provider.evaluate.assert_not_called()


def test_reaction_normalization_is_idempotent():
    job = _job()
    normalize_reaction_input(job)
    normalized = copy.deepcopy(job)
    normalize_reaction_input(job)
    assert job == normalized


def test_explicit_reaction_can_reuse_fragments_from_a_system_object():
    source = _job()
    source["system"]["fragments"] = [[0], [1, 2]]
    _, molecular_system = JobSpec.from_mapping(source).resolve()
    job = _job()
    job["system"] = molecular_system
    job["reaction"].pop("reactant_fragments")
    config, _ = JobSpec.from_mapping(job).resolve()
    assert config["reaction"]["fragments"] == [[0], [1, 2]]


def test_python_fragment_indices_accept_numpy_integers_and_normalize_to_python_int():
    job = _job()
    job["reaction"]["reactant_fragments"] = [[np.int64(0)], [np.int64(1), np.int64(2)]]
    normalize_reaction_input(job)
    assert job["reaction"]["fragments"] == [[0], [1, 2]]
    assert all(type(atom) is int for fragment in job["reaction"]["fragments"] for atom in fragment)
