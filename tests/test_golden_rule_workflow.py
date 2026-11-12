"""Golden-rule workflow: independent harmonic limits, grids and scientific identity."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.constants import ATOMIC_TIME_S, KB_AU
from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.base import PESResult
from pyrpmd.workflows.instanton import load_instanton_input, run_instanton


class HarmonicSurface:
    def __init__(self, center, curvature=0.02, offset=0.0, bath=0.0):
        self.center, self.curvature, self.offset = center, curvature, offset
        self.bath = bath
        self.calls = 0

    def evaluate(self, q):
        self.calls += 1
        x = q[:, 0, 0] - self.center
        force = np.zeros_like(q)
        force[:, 0, 0] = -self.curvature * x
        force[:, 0, 1] = -self.bath * q[:, 0, 1]
        return PESResult(0.5 * self.curvature * x * x + self.offset + 0.5*self.bath*q[:, 0, 1]**2, force)

    def hessian(self, q):
        h = np.zeros((len(q), 3, 3), dtype=np.float64)
        h[:, 0, 0] = self.curvature
        h[:, 1, 1] = self.bath
        return h


def study_input(*, temperature=300.0, offset=0.0, coupling=1e-5, bath=0.0,
                levels=((8, 8), (16, 16), (32, 32))):
    mass, curvature = 1822.0, 0.02
    omega = math.sqrt(curvature / mass)
    beta = 1 / (KB_AU * temperature)
    tau = beta / 2
    t = np.linspace(0.0, tau, 9)
    path0 = np.zeros((9, 1, 3))
    path0[:, 0, 0] = np.cosh(omega * (t - tau / 2)) / math.cosh(omega * tau / 2) - 1
    path1 = -path0[::-1].copy()
    mask = np.array([[True, bool(bath), False]])
    hessian = [[curvature]] if not bath else np.diag([curvature, bath])
    return {"potential": (HarmonicSurface(-1.0, offset=offset, bath=bath), HarmonicSurface(1.0, offset=offset, bath=bath)),
            "masses": [mass], "initial_path": (path0, path1), "mode": "golden_rule",
            "bead_counts": levels, "temperature_kelvin": temperature, "active_mask": mask,
            "reactants": [HarmonicState([[-1.0, 0.0, 0.0]], [mass], offset, hessian, mask)],
            "energy_reference_hartree": offset,
            "potential_identity": {f"surface{i}": {"model": "shifted_harmonic", "center_bohr": center,
                                                      "curvature_hartree_per_bohr2": curvature,
                                                      "offset_hartree": offset, "bath_curvature": bath}
                                   for i, center in enumerate((-1.0, 1.0))},
            "golden_rule_options": {"coupling_hartree": coupling},
            "convergence_relative_tolerance": 0.02}


@pytest.fixture(scope="module")
def reference_study():
    config = study_input()
    return config, run_instanton(**config)


def test_independent_harmonic_semiclassical_limit_and_segment_scans(reference_study):
    config, report = reference_study
    beta = 1 / (KB_AU * config["temperature_kelvin"])
    omega = math.sqrt(0.02 / config["masses"][0])
    reorganization = 0.04  # V1(-1)-V1(+1) for the specified parabolas.
    # Independent analytic imaginary-time harmonic correlation saddle, with
    # its complete curvature factor. This is a semiclassical, not exact
    # irreversible finite-bath quantum-rate reference.
    analytic = (1e-10 * math.sqrt(2 * math.pi * math.sinh(beta * omega / 2) / (reorganization * omega))
                * math.exp(-reorganization / omega * math.tanh(beta * omega / 4)) / ATOMIC_TIME_S)
    assert report["final_rate"]["rate"] == pytest.approx(analytic, rel=0.01)
    assert report["number_of_optimizations"] == 7
    points = {(row["nintervals0"], row["nintervals1"]) for row in report["rows"]}
    assert points == {(8, 8), (16, 16), (32, 32), (8, 32), (16, 32), (32, 8), (32, 16)}
    assert report["imaginary_time_reoptimized_at_every_grid"]
    for row in report["rows"]:
        orbit = row["orbit"]
        assert orbit["tau0_au"] + orbit["tau1_au"] == pytest.approx(beta, rel=1e-12)
        assert orbit["gradient_max"] <= 1e-8
        assert abs(orbit["energy_gap_hartree"]) <= 1e-10
        assert orbit["time_schur_curvature_hartree2"] < 0
        assert row["rate"]["rate_unit"] == "s^-1"
    assert report["numerical_converged"]
    assert report["surface0_convergence"]["refinement_ratios"] == [2.0, 2.0]
    assert report["surface1_convergence"]["refinement_ratios"] == [2.0, 2.0]
    json.dumps(report, allow_nan=False)


def test_coupling_squared_scaling_and_two_surface_identity(reference_study):
    _, base = reference_study
    doubled = run_instanton(**study_input(coupling=-2e-5))
    assert doubled["final_rate"]["rate"] == pytest.approx(4 * base["final_rate"]["rate"], rel=1e-10)
    assert doubled["physics_fingerprint"] != base["physics_fingerprint"]
    identity = doubled["scientific_identity"]
    assert len(identity["surfaces"]) == 2
    assert identity["surfaces"][0]["declared"]["center_bohr"] == -1
    assert identity["surfaces"][1]["declared"]["center_bohr"] == 1
    assert identity["optimizer_options"]["tau_fraction_bounds"] == [1e-4, 1 - 1e-4]
    assert identity["rate_options"]["coupling_hartree"] == -2e-5


def test_energy_reference_and_explicit_partition_match_harmonic_route(reference_study):
    config, base = reference_study
    shifted = run_instanton(**study_input(offset=0.13))
    assert shifted["final_rate"]["log_rate_au"] == pytest.approx(base["final_rate"]["log_rate_au"], abs=1e-7)
    beta = 1 / (KB_AU * config["temperature_kelvin"])
    explicit = study_input()
    explicit["golden_rule_options"]["log_reactant_partition"] = explicit["reactants"][0].log_partition(beta)
    explicit["golden_rule_options"]["partition_discretization"] = "external"
    explicit["reactants"] = None
    result = run_instanton(**explicit)
    continuous = study_input()
    continuous["golden_rule_options"]["partition_discretization"] = "continuous"
    continuous = run_instanton(**continuous)
    assert result["final_rate"]["log_rate_au"] == pytest.approx(continuous["final_rate"]["log_rate_au"], abs=1e-10)
    assert result["physics_fingerprint"] != continuous["physics_fingerprint"] != base["physics_fingerprint"]
    assert result["scientific_identity"]["reactant_partition"]["caller_must_use_the_action_energy_zero"]


def test_matched_partition_cancels_high_frequency_spectator_at_every_grid(reference_study):
    _, base = reference_study
    bath = run_instanton(**study_input(bath=0.8))
    for one, two in zip(base["rows"], bath["rows"]):
        assert one["rate"]["log_rate_au"] == pytest.approx(two["rate"]["log_rate_au"], abs=1e-8)
        assert two["reactant_partition"]["matched_to_optimized_segment_grid"]
    continuum = study_input(bath=0.8)
    continuum["golden_rule_options"]["partition_discretization"] = "continuous"
    continuum = run_instanton(**continuum)
    # A continuum denominator does not cancel a finite-grid high-frequency bath.
    assert abs(continuum["final_rate"]["rate"] / bath["final_rate"]["rate"] - 1) > 0.01


def test_high_temperature_limit_agrees_with_independent_marcus_formula():
    config = study_input(temperature=10000.0)
    report = run_instanton(**config)
    beta = 1 / (KB_AU * config["temperature_kelvin"])
    reorganization = 0.04
    marcus = 1e-10 * math.sqrt(math.pi * beta / reorganization) * math.exp(-beta * reorganization / 4) / ATOMIC_TIME_S
    assert report["final_rate"]["rate"] == pytest.approx(marcus, rel=0.002)
    assert report["discretization_converged"]


@pytest.mark.parametrize("change,match", [
    ({"potential_identity": {"surface0": {"model": "a"}}}, "surface0 and surface1"),
    ({"potential_identity": {"surface0": {}, "surface1": {"model": "b"}}}, "identify"),
    ({"bead_counts": [8, 16, 32]}, "pair"),
    ({"bead_counts": [(8, 8), (16, 16)]}, "three"),
    ({"bead_counts": [(8, 8), (16, 8), (32, 32)]}, "both surfaces"),
    ({"bead_counts": [(8, 8), (16, 16), (4096, 4096)]}, "4096"),
    ({"temperature_kelvin": 1e-310}, "nonfinite inverse temperature"),
    ({"golden_rule_options": {}}, "coupling_hartree"),
    ({"golden_rule_options": {"coupling_hartree": 0.0}}, "nonzero"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "regime": "inverted"}}, "normal"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "rigid_body": True}}, "rigid_body"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "initial_tau_fraction": 1.0}}, "inside"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "tau_fraction_bounds": (0.6, 0.4)}}, "bounds"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "ignored_option": True}}, "unknown"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "partition_discretization": "unknown"}}, "partition_discretization"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "partition_discretization": "external"}}, "requires explicit"),
    ({"golden_rule_options": {"coupling_hartree": 1e-5, "log_reactant_partition": 0.0}}, "mutually exclusive"),
    ({"reactants": None}, "partition"),
    ({"optimizer_options": {"method": "bofill"}}, "not thermal"),
    ({"rate_options": {}}, "not thermal"),
    ({"splitting_options": {}}, "not thermal"),
])
def test_invalid_scientific_inputs_are_rejected_before_pes_calls(change, match):
    config = study_input()
    config.update(change)
    with pytest.raises(ValueError, match=match):
        run_instanton(**config)
    assert all(surface.calls == 0 for surface in config["potential"])


def test_nonclosing_paths_and_colliding_nested_identity_keys_fail():
    config = study_input()
    config["initial_path"][1][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="opposite directions"):
        run_instanton(**config)
    config = study_input()
    config["potential_identity"]["surface0"]["parameters"] = {1: "a", "1": "b"}
    with pytest.raises(ValueError, match="string keys"):
        run_instanton(**config)
    assert all(surface.calls == 0 for surface in config["potential"])


def test_surface_declared_coupling_cannot_silently_differ_from_rate():
    config = study_input()
    config["potential"][0].coupling_hartree = 1e-4
    with pytest.raises(ValueError, match="coupling magnitude differs"):
        run_instanton(**config)
    assert all(surface.calls == 0 for surface in config["potential"])


def test_external_partition_must_be_declared_as_external():
    config = study_input()
    config["reactants"] = None
    config["golden_rule_options"]["log_reactant_partition"] = 0.0
    with pytest.raises(ValueError, match="partition_discretization='external'"):
        run_instanton(**config)
    assert all(surface.calls == 0 for surface in config["potential"])


def test_small_grid_changes_do_not_certify_continuum_convergence():
    report = run_instanton(**study_input(levels=((8, 8), (9, 9), (10, 10))))
    assert report["numerical_converged"]
    assert not report["discretization_converged"]
    assert not report["surface0_convergence"]["sufficient_refinement_span"]
    assert not report["surface1_convergence"]["sufficient_refinement_span"]


def test_existing_cli_dispatch_runs_diabatic_adapter_example_and_saves_atomically(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/instanton/golden_rule.py"
    config = load_instanton_input(example)
    for name in ("surface0", "surface1"):
        assert len(config["potential_identity"][name]["input_sha256"]) == 64
    output = tmp_path / "golden-rule.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-m", "pyrpmd.cli", "instanton", str(example),
                             "--output", str(output)], env=env, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert json.loads(output.read_text())["physics_fingerprint"] == report["physics_fingerprint"]
    assert report["mode"] == "golden_rule"
    assert report["number_of_optimizations"] == 10
    assert report["numerical_converged"]
    assert report["discretization_converged"]
    assert not list(tmp_path.glob("*.tmp"))
