"""Independent model checks and input/identity boundaries of instanton studies."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.instanton import instanton_seed
from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.base import PESResult
from pyrpmd.workflows.instanton import (load_instanton_input, register_parser,
                                        run_instanton, run_instanton_input)


class Quartic:
    def __init__(self, bath=0.0):
        self.bath, self.calls = bath, 0

    def evaluate(self, q):
        self.calls += 1
        x, y = q[:, 0, 0], q[:, 0, 1]
        force = np.zeros_like(q)
        force[:, 0, 0] = -0.04 * x * (x * x - 1)
        force[:, 0, 1] = -self.bath * y
        return PESResult(0.01 * (x * x - 1)**2 + 0.5 * self.bath * y * y, force)

    def hessian(self, q):
        hessian = np.zeros((len(q), 3, 3), dtype=np.float64)
        hessian[:, 0, 0] = 0.04 * (3 * q[:, 0, 0]**2 - 1)
        hessian[:, 1, 1] = self.bath
        return hessian


def rate_input(bath=0.0):
    potential = Quartic(bath)
    mask = np.array([[True, bool(bath), False]])
    masses = np.array([1822.0])
    hessian = np.diag([0.08, bath]) if bath else [[0.08]]
    state = HarmonicState([[-1.0, 0.0, 0.0]], masses, 0.0, hessian, mask)
    ts = np.zeros((1, 3))
    return dict(potential=potential, masses=masses,
                potential_identity={"model": "quartic", "height": 0.01, "bath": bath},
                initial_path=instanton_seed(ts, [[1.0, 0.0, 0.0]], 32, 0.95),
                bead_counts=[32, 64, 128], temperature_kelvin=0.55 * math.sqrt(0.04 / masses[0]) / (2 * math.pi * KB_AU),
                active_mask=mask, transition_state=ts, reactants=[state],
                optimizer_options={"method": "root"}, convergence_relative_tolerance=0.02)


def test_full_rate_study_separable_transverse_partition_cancels():
    one = run_instanton(**rate_input())
    two = run_instanton(**rate_input(bath=0.02))
    for a, b in zip(one["rows"], two["rows"]):
        assert a["rate"]["log_rate_au"] == pytest.approx(b["rate"]["log_rate_au"], abs=1e-8)
        assert a["orbit"]["gradient_max"] <= 1e-8
        assert a["rate"]["negative_modes"] == 1
    assert two["numerical_converged"] is True
    assert two["physical_converged"] is False
    assert two["bead_convergence"]["required_successive_refinements"] == 2
    assert two["rows"][-1]["orbit"]["nbeads"] == 128
    assert two["crossover_temperature_kelvin"] > two["scientific_identity"]["temperature_kelvin"]
    json.dumps(two, allow_nan=False)


def test_identity_covers_defaults_science_and_output_is_atomic(tmp_path):
    config = rate_input()
    destination = tmp_path / "study.json"
    report = run_instanton(**config, output=destination)
    assert json.loads(destination.read_text())["physics_fingerprint"] == report["physics_fingerprint"]
    assert not list(tmp_path.glob("*.tmp"))
    settings = report["scientific_identity"]
    assert settings["optimizer_options"]["retain_hessian"] is True
    assert settings["optimizer_options"]["hessian_step_bohr"] == 1e-4
    assert settings["rate_options"]["reaction_degeneracy"] == 1
    assert settings["potential"]["provider_source_sha256"]["evaluate"]
    # Changing only reaction path multiplicity changes both identity and rate.
    doubled = run_instanton(**config, rate_options={"reaction_degeneracy": 2})
    assert doubled["physics_fingerprint"] != report["physics_fingerprint"]
    assert doubled["final_rate"]["rate"] == pytest.approx(2 * report["final_rate"]["rate"])
    repeat = run_instanton(**config)
    assert repeat["physics_fingerprint"] == report["physics_fingerprint"]


def test_scientific_identity_rejects_nested_key_collisions_before_pes():
    config = rate_input()
    config["potential_identity"] = {"model": {1: "lost", "1": "retained"}}
    with pytest.raises(ValueError, match="string keys at every nesting level"):
        run_instanton(**config)
    assert config["potential"].calls == 0


def test_tiny_grid_increments_do_not_certify_discretization_convergence():
    config = rate_input()
    config["bead_counts"] = [32, 34, 36]
    report = run_instanton(**config)
    check = report["bead_convergence"]
    assert max(check["relative_changes"]) < config["convergence_relative_tolerance"]
    assert not check["sufficient_refinement_span"]
    assert not report["discretization_converged"]
    assert report["numerical_converged"]


@pytest.mark.parametrize("change,match", [
    ({"bead_counts": [32, 64]}, "three"),
    ({"bead_counts": [32, 64, 63]}, "increasing"),
    ({"bead_counts": [32, 64, 128.0]}, "integer"),
    ({"potential_identity": {}}, "identify the PES"),
    ({"optimizer_options": {"grad_tolerance": 1e-8}}, "unknown"),
    ({"optimizer_options": {"raise_on_failure": False}}, "workflow-owned"),
    ({"optimizer_options": {"method": "unavailable"}}, "method"),
    ({"optimizer_options": {"max_iterations": -1}}, "positive integer"),
    ({"optimizer_options": {"hessian_step_bohr": 0}}, "positive"),
    ({"rate_options": {"symmetry": 1}}, "unknown"),
    ({"rate_options": {"rigid_body": "true"}}, "boolean"),
    ({"convergence_relative_tolerance": True}, "positive"),
    ({"imaginary_times_au": [10, 20, 30]}, "rate mode"),
    ({"mode": "golden_rule"}, "mode"),
])
def test_invalid_inputs_fail_before_any_pes_call(change, match):
    config = rate_input()
    config.update(change)
    with pytest.raises(ValueError, match=match):
        run_instanton(**config)
    assert config["potential"].calls == 0


def test_splitting_duration_scan_preserves_dtau_and_matches_analytic_kink():
    mass, omega = 1822.0, math.sqrt(0.08 / 1822.0)
    durations = np.array([20.0, 24.0, 28.0]) / omega
    path = np.zeros((81, 1, 3))
    path[:, 0, 0] = np.tanh(omega * np.linspace(-durations[0] / 2, durations[0] / 2, 81) / 2)
    path[0, 0, 0], path[-1, 0, 0] = -1.0, 1.0
    report = run_instanton(Quartic(), [mass], path, mode="splitting",
                           potential_identity={"model": "quartic", "height": 0.01},
                           active_mask=np.array([[True, False, False]]),
                           bead_counts=[80, 160, 320], imaginary_times_au=durations,
                           splitting_options={"equivalent_wells": True},
                           convergence_relative_tolerance=0.02)
    s0 = 4 / 3 * math.sqrt(2 * mass * 0.01)
    exact_semiclassical = 2 * omega * math.sqrt(6 * s0 / math.pi) * math.exp(-s0)
    assert report["final_splitting"]["splitting_hartree"] == pytest.approx(exact_semiclassical, rel=0.001)
    assert report["final_splitting"]["action_hbar"] == pytest.approx(s0, rel=0.001)
    assert report["duration_comparisons_at_fixed_dtau"]
    for column in range(3):
        dtau = [report["rows"][row * 3 + column]["dtau_au"] for row in range(3)]
        np.testing.assert_allclose(dtau, dtau[0], rtol=1e-14)
    assert report["numerical_converged"]
    assert report["discretization_converged"]
    assert not report["physical_converged"]


def test_bad_splitting_grids_and_unasserted_wells_fail_before_pes():
    base = dict(potential=Quartic(), masses=[1822.0], initial_path=np.zeros((9, 1, 3)),
                mode="splitting", potential_identity={"model": "quartic"},
                bead_counts=[80, 160, 320], imaginary_times_au=[2000.0, 2400.0, 2800.0])
    with pytest.raises(ValueError, match="equivalent_wells"):
        run_instanton(**base)
    with pytest.raises(ValueError, match="integer scaled"):
        run_instanton(**dict(base, imaginary_times_au=[2000.0, 2401.0, 2800.0]),
                       splitting_options={"equivalent_wells": True})
    assert base["potential"].calls == 0


def test_python_input_tracks_source_and_command_registration(tmp_path, capsys):
    factory = tmp_path / "input.py"
    example = Path(__file__).resolve().parents[1] / "examples/instanton/production.py"
    factory.write_text("from pyrpmd.potentials.python import import_object\n"
                       f"def build():\n    return import_object({str(example) + ':build'!r})()\n")
    loaded = load_instanton_input(factory)
    assert len(loaded["potential_identity"]["input_sha256"]) == 64
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command", required=True))
    destination = tmp_path / "report.json"
    args = parser.parse_args(["instanton", str(factory), "--output", str(destination)])
    assert args._instanton_handler(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(destination.read_text())["physics_fingerprint"] == report["physics_fingerprint"]
    assert report["scientific_identity"]["optimizer_options"]["method"] == "bofill"
    assert report["discretization_converged"]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run([sys.executable, "-m", "pyrpmd.cli", "instanton", str(factory),
                                "--output", str(destination)], env=environment,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["physics_fingerprint"] == report["physics_fingerprint"]
    bad = tmp_path / "bad.py"
    bad.write_text("def build():\n    return {'unexpected': 3}\n")
    with pytest.raises(ValueError, match="unknown instanton input fields"):
        run_instanton_input(bad)
    with pytest.raises(ValueError, match="trusted .py"):
        load_instanton_input(tmp_path / "input.json")


def test_python_input_is_reexecuted_after_source_changes(tmp_path):
    factory = tmp_path / "changing.py"
    factory.write_text("def build():\n    return {'potential_identity': {'model': 'first'}}\n")
    before = load_instanton_input(factory)
    factory.write_text("def build():\n    return {'potential_identity': {'model': 'second'}}\n")
    after = load_instanton_input(factory)
    assert before["potential_identity"]["declared"]["model"] == "first"
    assert after["potential_identity"]["declared"]["model"] == "second"
    assert before["potential_identity"]["input_sha256"] != after["potential_identity"]["input_sha256"]
