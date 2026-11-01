"""Regressions for reactant/product and approximate-TS rate preparation."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials.python import import_object
from pyrpmd.runner import prepare
from pyrpmd.system import system_from_config
from pyrpmd.workflows.prepare import bonding_graph, prepare_rate
from pyrpmd.workflows.rate import run_rate


_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rate"
ToyExchange = import_object(str(_EXAMPLE / "toy_exchange.py") + ":ToyExchange")
_SADDLE_DISTANCE = 1.2 + np.log(3.0) / 1.2


def _guess():
    return np.array([
        [-_SADDLE_DISTANCE - 0.15, 0.03, 0.0],
        [0.0, 0.0, 0.0],
        [_SADDLE_DISTANCE + 0.08, 0.0, 0.0],
    ])


def _endpoints():
    reactant = np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    product = np.array([[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    return reactant - reactant.mean(axis=0), product - product.mean(axis=0)


def _job(*, endpoints=False, explicit=False, preparation=None):
    system = {"symbols": ["H", "H", "H"], "geometry_unit": "bohr"}
    if endpoints:
        reactant, product = _endpoints()
        system.update(reactant=reactant.tolist(), product=product.tolist())
    else:
        system["geometry"] = _guess().tolist()
    reaction = {"rinf": "5 angstrom"}
    if explicit:
        reaction.update(
            fragments=[[0], [1, 2]],
            forming_bonds=[[0, 1]],
            breaking_bonds=[[1, 2]],
        )
    if preparation is not None:
        reaction["preparation"] = preparation
    return {
        "task": "rate",
        "temperature": "1000 K",
        "beads": 1,
        "native": False,
        "system": system,
        "potential": {
            "provider": "python",
            "factory": str(_EXAMPLE / "toy_exchange.py") + ":build",
        },
        "reaction": reaction,
        "dynamics": {"timestep": "0.1 fs"},
        "sampling": {
            "equilibration_steps": 0,
            "production_steps": 2,
            "sample_interval": 1,
        },
        "umbrella": {"centers": [0.0, 0.5, 1.0], "kappa": 0.16},
        "analysis": {"bins": 4, "bootstrap": 0},
        "parallel": {"backend": "serial"},
    }


def _assert_exchange_saddle(geometry, tolerance=1e-5):
    q = np.asarray(geometry, dtype=float)
    force = ToyExchange().evaluate(q[None]).forces
    assert np.max(np.abs(force)) <= tolerance
    np.testing.assert_allclose(
        [np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[1])],
        [_SADDLE_DISTANCE, _SADDLE_DISTANCE],
        atol=2e-3,
        rtol=0,
    )


def test_approximate_ts_refines_and_infers_exchange_without_mutating_inputs():
    job = _job()
    original_job = copy.deepcopy(job)
    config, system = JobSpec.from_mapping(job).resolve()
    original_geometry = system.geometry.copy()
    original_config = copy.deepcopy(config)

    result = prepare_rate(config, system, ToyExchange(), numerical=True)

    assert result["ready"]
    assert result["refinement"]["converged"]
    assert result["refinement"]["iterations"] > 0
    _assert_exchange_saddle(result["transition_state"])
    assert np.count_nonzero(np.array(result["hessian_eigenvalues"]) < -1e-8) == 1
    assert result["reaction"]["forming_bonds"]
    assert result["reaction"]["breaking_bonds"]
    np.testing.assert_array_equal(system.geometry, original_geometry)
    assert job == original_job
    assert config == original_config


def test_endpoint_pipeline_finds_ts_and_preserves_requested_reactant_direction():
    job = _job(endpoints=True)
    original_job = copy.deepcopy(job)
    config, system = JobSpec.from_mapping(job).resolve()
    original_geometry = system.geometry.copy()

    result = prepare_rate(config, system, ToyExchange(), numerical=True)

    assert result["ready"]
    assert result["neb"]["converged"]
    assert result["refinement"]["converged"]
    _assert_exchange_saddle(result["transition_state"])
    assert result["reaction"]["fragments"] == [[0], [1, 2]]
    assert result["reaction"]["forming_bonds"] == [[0, 1]]
    assert result["reaction"]["breaking_bonds"] == [[1, 2]]
    np.testing.assert_array_equal(system.geometry, original_geometry)
    assert job == original_job


def test_reverse_endpoints_reverse_the_labeled_reaction_channel():
    job = _job(endpoints=True)
    job["system"]["reactant"], job["system"]["product"] = (
        job["system"]["product"], job["system"]["reactant"]
    )
    config, system = JobSpec.from_mapping(job).resolve()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert result["ready"]
    assert result["reaction"]["fragments"] == [[0, 1], [2]]
    assert result["reaction"]["forming_bonds"] == [[1, 2]]
    assert result["reaction"]["breaking_bonds"] == [[0, 1]]


def test_runner_prepare_returns_optimized_geometry():
    job = _job()
    original = copy.deepcopy(job)
    result = prepare(job, numerical=True)
    assert result["ready"]
    _assert_exchange_saddle(result["system"]["geometry"])
    np.testing.assert_array_equal(
        result["system"]["geometry"],
        result["reaction_preparation"]["transition_state"],
    )
    assert job == original


@pytest.mark.parametrize("endpoints", [False, True])
def test_nonnumerical_prepare_defers_search_without_constructing_pes(endpoints):
    with patch("pyrpmd.potentials.make_potential") as build:
        result = prepare(_job(endpoints=endpoints), numerical=False)
    assert not result["ready"]
    assert result["reaction_preparation"]["requires_numerical_preparation"]
    build.assert_not_called()


def test_rate_sampling_receives_refined_geometry(tmp_path):
    config, system = JobSpec.from_mapping(_job()).resolve()
    original_geometry = system.geometry.copy()

    class SamplingReached(Exception):
        pass

    with patch("pyrpmd.workflows.rate.run_fes", side_effect=SamplingReached) as sample:
        with pytest.raises(SamplingReached):
            run_rate(config, system, tmp_path)

    sample.assert_called_once()
    sampled_system = sample.call_args.args[1]
    _assert_exchange_saddle(sampled_system.geometry)
    np.testing.assert_array_equal(system.geometry, original_geometry)
    assert (tmp_path / "preparation.json").is_file()


def test_explicit_reaction_keeps_legacy_bypass_unless_refinement_requested():
    config, system = JobSpec.from_mapping(_job(explicit=True)).resolve()
    unused_provider = Mock()
    unused_provider.evaluate.side_effect = AssertionError("legacy explicit path evaluated PES")
    result = prepare_rate(config, system, unused_provider, numerical=True)
    assert result["ready"]
    assert result["reaction"]["forming_bonds"] == [[0, 1]]
    unused_provider.evaluate.assert_not_called()

    config["reaction"]["preparation"] = {"refine": True}
    refined = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert refined["ready"]
    assert refined["refinement"]["converged"]
    _assert_exchange_saddle(refined["transition_state"])


def test_refinement_can_be_disabled_to_retain_strict_ts_validation():
    config, system = JobSpec.from_mapping(_job(preparation={"refine": False})).resolve()
    with pytest.raises(ValueError, match="not stationary"):
        prepare_rate(config, system, ToyExchange(), numerical=True)


def test_exhausted_refinement_is_not_ready_and_does_not_replace_input_geometry():
    config, system = JobSpec.from_mapping(
        _job(preparation={"max_steps": 1, "max_step": "0.00001 bohr"})
    ).resolve()
    original = system.geometry.copy()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert not result["ready"]
    assert not result["refinement"]["converged"]
    assert result["diagnostics"]
    np.testing.assert_array_equal(system.geometry, original)


def test_exhausted_neb_is_not_ready_and_does_not_run_refinement():
    config, system = JobSpec.from_mapping(
        _job(endpoints=True, preparation={"neb": {"max_steps": 1, "climb_after": 0}})
    ).resolve()
    result = prepare_rate(config, system, ToyExchange(), numerical=True)
    assert not result["ready"]
    assert not result["neb"]["converged"]
    assert "refinement" not in result


@pytest.mark.parametrize("endpoints", [False, True])
def test_unconverged_search_stops_rate_before_sampling_and_saves_diagnostics(tmp_path, endpoints):
    controls = {"neb": {"max_steps": 1}} if endpoints else {"max_steps": 1, "max_step": 1e-5}
    config, system = JobSpec.from_mapping(_job(endpoints=endpoints, preparation=controls)).resolve()
    with patch("pyrpmd.workflows.rate.run_fes") as sample:
        with pytest.raises(RuntimeError, match="did not produce a validated TS"):
            run_rate(config, system, tmp_path)
    sample.assert_not_called()
    report = json.loads((tmp_path / "preparation.json").read_text())
    assert not report["ready"]
    assert report["diagnostics"]
    assert not report["neb" if endpoints else "refinement"]["converged"]


def test_inline_endpoints_use_shared_units_and_leave_input_unchanged():
    spec = _job(endpoints=True)["system"]
    spec["geometry_unit"] = "angstrom"
    before = copy.deepcopy(spec)
    system = system_from_config(spec)
    np.testing.assert_allclose(system.geometry, np.array(spec["reactant"]) * ANGSTROM_TO_AU)
    assert spec == before


def _write_xyz(path, symbols, geometry):
    rows = [str(len(symbols)), "endpoint geometry"]
    rows.extend(
        f"{symbol} {position[0]:.12f} {position[1]:.12f} {position[2]:.12f}"
        for symbol, position in zip(symbols, geometry)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_relative_xyz_endpoints_resolve_and_reach_a_valid_saddle(tmp_path):
    reactant, product = _endpoints()
    _write_xyz(tmp_path / "reactant.xyz", ["H"] * 3, reactant)
    _write_xyz(tmp_path / "product.xyz", ["H"] * 3, product)
    job = _job(endpoints=True)
    job["system"] = {
        "reactant": "reactant.xyz", "product": "product.xyz", "geometry_unit": "bohr"
    }
    source = tmp_path / "job.py"
    source.write_text(f"def build():\n    return {job!r}\n", encoding="utf-8")
    result = prepare(source, numerical=True)
    assert result["ready"]
    _assert_exchange_saddle(result["system"]["geometry"])


@pytest.mark.parametrize("product_symbols", [["D", "H", "H"], ["H", "H"]])
def test_xyz_endpoints_require_identical_atom_identity_and_order(tmp_path, product_symbols):
    reactant, product = _endpoints()
    _write_xyz(tmp_path / "reactant.xyz", ["H", "D", "H"], reactant)
    _write_xyz(tmp_path / "product.xyz", product_symbols, product)
    job = _job(endpoints=True)
    job["system"] = {
        "reactant": str(tmp_path / "reactant.xyz"),
        "product": str(tmp_path / "product.xyz"),
        "geometry_unit": "bohr",
    }
    with pytest.raises(InputError, match="atom|symbol|order|identit"):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("missing", ["reactant", "product"])
def test_both_endpoints_are_required(missing):
    job = _job(endpoints=True)
    del job["system"][missing]
    with pytest.raises(InputError, match="reactant|product"):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("other", ["geometry", "transition_state", "reactants"])
def test_endpoint_and_existing_geometry_sources_are_mutually_exclusive(other):
    job = _job(endpoints=True)
    job["system"][other] = _guess().tolist() if other == "geometry" else "unused.xyz"
    with pytest.raises(InputError, match="reactant|product|exclusive|together|only"):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("product", [[[0, 0]], [[0, 0, 0]], [[0, 0, 0]] * 4,
                                     [[np.nan, 0, 0]] * 3])
def test_inline_endpoints_reject_bad_shape_and_nonfinite_coordinates(product):
    job = _job(endpoints=True)
    job["system"]["product"] = product
    with pytest.raises(InputError):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("options", [
    "invalid", [], {"refine": "yes"}, {"max_steps": 2.5},
    {"max_step": 0}, {"gradient_tolerance": np.nan}, {"hessian_step": -1},
    {"negative_eigenvalue_tolerance": "1 fs"},
    {"unknown_option": 1}, {"neb": []}, {"neb": {"images": True}},
    {"neb": {"images": 2}}, {"neb": {"spring_constant": -0.1}},
    {"neb": {"max_steps": 1.5}}, {"neb": {"unknown_option": 1}},
])
def test_preparation_options_are_checked_before_pes_construction(options):
    job = _job(endpoints=True, preparation=options)
    with patch("pyrpmd.potentials.make_potential") as build:
        with pytest.raises(InputError):
            JobSpec.from_mapping(job).resolve()
    build.assert_not_called()


def test_bonding_graph_uses_extended_radii_isotopes_and_explicit_heavy_override():
    q = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]) * ANGSTROM_TO_AU
    assert bonding_graph(q, ["Fe", "T"]) == {(0, 1)}
    assert bonding_graph(q, ["Og", "H"], radii={"Og": "1.8 angstrom"}) == {(0, 1)}
    with pytest.raises(ValueError, match="covalent_radii"):
        bonding_graph(q, ["Og", "H"])
