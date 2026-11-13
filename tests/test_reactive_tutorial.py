"""The exchange tutorial has analytic forces and a genuinely reactive trajectory."""
from pathlib import Path
import runpy
import shutil

import numpy as np
import pytest

from pyrpmd.constants import FS_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.input import JobSpec
from pyrpmd.potentials.cabi import CABIPotential
from pyrpmd.workflows.drpmd_classification import ConnectivityClassifier
from pyrpmd.workflows.drpmd import _trajectory

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "examples/tutorials/exchange_rate"


@pytest.fixture(scope="module")
def exchange_library(tmp_path_factory):
    if shutil.which("cc") is None:
        pytest.skip("existing C compiler required")
    build = runpy.run_path(str(TUTORIAL / "build.py"))["build"]
    return build(tmp_path_factory.mktemp("exchange-library") / "libexchange.so")


def test_analytic_cabi_matches_python_model_and_finite_difference(exchange_library):
    model = runpy.run_path(str(ROOT / "examples/rate/toy_exchange.py"))["ToyExchange"](
        depth=.15, coupling=.36)
    q = np.array([[[-1.8, .1, 0], [0, 0, 0], [1.9, -.2, .1]],
                  [[-8.0, .3, .2], [0, 0, 0], [1.3, 0, .1]],
                  [[-1.3, .1, 0], [0, 0, 0], [7.0, -.1, .2]]])
    with CABIPotential(exchange_library) as potential:
        result = potential.evaluate(q)
        reference = model.evaluate(q)
        np.testing.assert_allclose(result.energy, reference.energy, atol=2e-15)
        np.testing.assert_allclose(result.forces, reference.forces, atol=3e-15)
        step = 1e-5
        for atom in range(3):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[:, atom, axis] += step
                minus[:, atom, axis] -= step
                force = -(potential.evaluate(plus).energy - potential.evaluate(minus).energy) / (2 * step)
                np.testing.assert_allclose(result.forces[:, atom, axis], force, atol=3e-10, rtol=3e-7)
        assert potential.native_binding() is not None


@pytest.mark.parametrize("name", ["drpmd.inp", "drpmd-short.inp", "rate.inp"])
def test_inputs_resolve_without_changing_model_temperature_or_beads(name, exchange_library):
    spec = JobSpec.from_file(TUTORIAL / name).with_overrides(pes=str(exchange_library))
    config, system = spec.resolve()
    assert config["temperature_K"] == 1500 and config["beads"] == 4
    assert system.symbols == ("D", "H", "H")
    assert system.multiplicity == 2
    if config["task"] == "drpmd":
        assert config["collision"]["sampling"] == "thermal_flux"
        assert config["collision"]["impact_sampling"] == "uniform_area"
        assert config["sampling"]["parents"] >= 8
        assert config["sampling"]["equilibration_steps"] == 10000
        assert config["outcomes"]["channels"] == {"DH+H": ["DH", "H"]}


def test_directed_above_barrier_collision_forms_dh_and_separates(exchange_library, tmp_path):
    if not native_capabilities()["available"]:
        pytest.skip("native extension required for this propagation check")
    config, system = JobSpec.from_file(TUTORIAL / "drpmd-short.inp").with_overrides(pes=str(exchange_library)).resolve()
    mass = system.masses
    first, second = mass[0], mass[1:].sum()
    relative_speed = np.sqrt(2 * .035 / (first * second / (first + second)))
    p = np.zeros((4, 3, 3))
    p[:, 0, 0] = mass[0] * relative_speed * second / (first + second)
    p[:, 1:, 0] = -mass[1:] * relative_speed * first / (first + second)
    q = np.repeat(system.geometry[None], 4, axis=0)
    classifier = ConnectivityClassifier(system.symbols, mass, system.fragments,
        {"D-H": 2.5, "H-H": 2.5}, {"DH+H": ["DH", "H"]}, 9.0)
    with CABIPotential(exchange_library, system=system) as potential:
        dynamics = Dynamics(potential, mass, 1/(KB_AU*1500), 4, .1*FS_TO_AU,
                            loop_backend="native", threads=1)
        record = _trajectory(dynamics, State(q, p), classifier, config, tmp_path,
                             0, 0, {"purpose": "directed mechanism test; not thermal sampling"},
                             "exchange-directed-regression", False)
    assert record["status"] == "product"
    assert record["channel"] == "DH+H"
    assert record["propagation"]["native_steps"] > 0
    assert record["propagation"]["python_steps"] == 0
    assert abs(record["final_energy_per_bead"] - record["initial_energy_per_bead"]) < 1e-4


def test_comparison_does_not_promote_short_or_unresolved_runs(tmp_path):
    import csv
    import json
    compare = runpy.run_path(str(TUTORIAL / "compare.py"))["compare"]
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    summary = {"counts": {"product": 1, "nonreactive": 255, "failed": 0, "unresolved": 0},
               "parents": 8, "trajectories": 256, "sampling": "thermal_flux", "temperature_K": 1500,
               "beads": 4, "total_reaction": {"rate_cm3_molecule_s": 1e-11},
               "trajectory_table": "results/trajectories.csv"}
    path = result_dir / "summary.json"
    path.write_text(json.dumps(summary))
    assert compare(tmp_path, smoke=True)["passed"]
    assert not compare(tmp_path)["passed"]
    summary["counts"]["unresolved"] = 1
    path.write_text(json.dumps(summary))
    assert not compare(tmp_path, smoke=True)["passed"]
    summary.update(counts={"product": 24, "nonreactive": 776, "failed": 0, "unresolved": 0}, trajectories=800,
                   total_reaction={"rate_cm3_molecule_s": 1.142e-11, "event_fraction": 24/800})
    path.write_text(json.dumps(summary))
    with (result_dir / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parent_id", "status"])
        for parent, count in enumerate([1, 2, 3, 6, 1, 2, 3, 6]):
            writer.writerows([parent, "product" if i < count else "nonreactive"] for i in range(100))
    result = compare(tmp_path)
    assert result["passed"] and result["rate_comparison_performed"]
    assert result["parent_cluster_standard_error_cm3_s"] > 0
