"""Published-reference comparison must not turn pilots/QM differences into CI claims."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "published_rates" / "compare.py"
SPEC = importlib.util.spec_from_file_location("published_rate_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def manifest(reference, *, ratio=1.02, converged=False):
    result = {key: copy.deepcopy(reference[key]) for key in (
        "reaction", "pes", "temperature_K", "nuclear_spin_convention",
        "electronic_convention", "electronic_weight", "symmetry_convention",
        "symmetry_factor",
    )}
    result.update(
        method="RPMD", rate_unit=comparison.RATE_UNIT,
        rate_cm3_molecule_s=ratio * reference["rate_cm3_molecule_s"],
        convergence={key: converged for key in comparison.CONVERGENCE_FLAGS},
        uncertainty={
            "includes_pmf": converged, "includes_recrossing": True,
            "standard_error_cm3_molecule_s": .01 * reference["rate_cm3_molecule_s"],
            "independent_parent_chains": 3,
            "method": "paired static/bootstrap and independent-chain recrossing bootstrap",
        },
        provenance={"source": "synthetic-unit-test-calculation.json"},
    )
    if converged:
        result["convergence_evidence"] = {
            key: {"source": "synthetic-unit-test-convergence.json",
                  "criterion": f"Synthetic {key} convergence criterion satisfied"}
            for key in comparison.CONVERGENCE_FLAGS
        }
    return result


def h3_reference():
    return comparison.load_reference("h_h2_bkmp2_rpmd_300k")


def test_real_source_h3_reference_matched_synthetic_pilot_reports_ratio_not_validation():
    reference = h3_reference()
    assert reference["rate_cm3_molecule_s"] == 1.95e-16
    result = comparison.compare(reference, manifest(reference, ratio=1.2))
    assert result["compatible"] is True
    assert result["status"] == "pilot_comparison"
    assert result["ratio"] == pytest.approx(1.2)
    assert result["relative_difference"] == pytest.approx(.2)
    assert result["passed"] is None
    assert result["gate_eligible"] is False
    assert result["implementation_failure"] is None
    assert result["evidence_independently_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("reaction", "D + H2 -> HD + H"),
    ("pes", "BKMP1"),
    ("temperature_K", 301.),
    ("nuclear_spin_convention", "para_hydrogen_even_j"),
    ("electronic_convention", "spin_orbit_corrected"),
    ("electronic_weight", .5),
    ("symmetry_convention", "single_atom_labelled_exit"),
    ("symmetry_factor", 2.),
    ("method", "TRPMD"),
    ("rate_unit", "m3 molecule-1 s-1"),
])
def test_mismatched_physics_or_units_never_produce_ratio(field, value):
    reference = h3_reference()
    calculation = manifest(reference, converged=True)
    calculation[field] = value
    result = comparison.compare(reference, calculation)
    assert result["status"] == "incompatible_conventions"
    assert result["compatible"] is False
    assert result["ratio"] is None
    assert result["passed"] is None


@pytest.mark.parametrize("field", ["electronic_weight", "symmetry_factor", "nuclear_spin_convention"])
def test_unresolved_reference_is_not_assumed_unity_or_equal(field):
    reference = h3_reference()
    calculation = manifest(reference, converged=True)
    reference[field] = None
    result = comparison.compare(reference, calculation)
    assert "reference_convention_unresolved:" + field in result["compatibility_reasons"]
    assert result["ratio"] is None


def test_current_qm_spin_convention_and_null_factors_are_blocked():
    reference = comparison.load_reference("h_h2_bkmp2_qm_300k")
    # Even copying every unresolved label does not make it a matched convention.
    result = comparison.compare(reference, manifest(reference, converged=True))
    assert result["comparison_kind"] == "quantum_method_accuracy"
    assert result["compatible"] is False
    assert result["ratio"] is None
    assert result["passed"] is None
    assert result["implementation_failure"] is None


@pytest.mark.parametrize("reference", [None, comparison.load_reference("mu_h2_rpmd_pending")])
def test_missing_reference_never_fabricates_rate(reference):
    result = comparison.compare(reference, manifest(h3_reference()))
    assert result["status"] == "reference_not_available"
    assert result["reference_rate_cm3_molecule_s"] is None
    assert result["ratio"] is None
    assert result["passed"] is None


@pytest.mark.parametrize("ratio,passed", [(1.02, True), (1.1, True), (.7, False), (1.3, False)])
def test_converged_synthetic_rpmd_gate_passes_or_fails_point_tolerance(ratio, passed):
    reference = h3_reference()
    result = comparison.compare(reference, manifest(reference, ratio=ratio, converged=True), .1)
    assert result["status"] == "implementation_agreement"
    assert result["gate_eligible"] is True
    assert result["passed"] is passed
    assert result["implementation_failure"] is (not passed)


@pytest.mark.parametrize("missing", ["beads", "timestep", "rinf", "sampling", "plateau"])
def test_true_flags_without_corresponding_evidence_cannot_pass(missing):
    reference = h3_reference()
    calculation = manifest(reference, converged=True)
    calculation["convergence_evidence"].pop(missing)
    result = comparison.compare(reference, calculation)
    assert result["status"] == "pilot_comparison"
    assert result["passed"] is None
    assert "convergence_evidence_missing:" + missing in result["convergence_reasons"]


@pytest.mark.parametrize("key,value", [
    ("includes_pmf", False), ("includes_recrossing", False),
    ("standard_error_cm3_molecule_s", None),
    ("standard_error_cm3_molecule_s", 0.),
    ("standard_error_cm3_molecule_s", 1.95e-16),
    ("independent_parent_chains", 1), ("method", ""),
])
def test_incomplete_or_uninformative_uncertainty_keeps_gate_unresolved(key, value):
    reference = h3_reference()
    calculation = manifest(reference, converged=True)
    calculation["uncertainty"][key] = value
    result = comparison.compare(reference, calculation)
    assert result["compatible"] is True
    assert result["status"] == "pilot_comparison"
    assert result["ratio"] is not None
    assert result["passed"] is None


def test_verified_synthetic_qm_method_discrepancy_never_becomes_code_failure():
    reference = h3_reference()
    # Artificial matched-QM fixture tests classification; it adds no QM datum
    # to the published registry and is not a claim about actual H+H2 values.
    reference.update(id="synthetic_qm", reference_method="QM",
                     comparison_role="quantum_accuracy_reference")
    result = comparison.compare(reference, manifest(reference, ratio=4., converged=True))
    assert result["status"] == "quantum_accuracy_comparison"
    assert result["ratio"] == pytest.approx(4)
    assert result["within_tolerance"] is False
    assert result["passed"] is None
    assert result["implementation_failure"] is None
    assert result["gate_eligible"] is False


@pytest.mark.parametrize("fields", [
    {"method": None}, {"method": "unknown"}, {"method": "pending"},
    {"method": True}, {"method": "RPMD", "reference_method": 42},
    {"method": "RPMD", "reference_method": "unresolved"},
])
def test_qm_comparison_requires_a_resolved_calculation_method(fields):
    reference = h3_reference()
    reference.update(id="synthetic_qm", reference_method="QM",
                     comparison_role="quantum_accuracy_reference")
    calculation = manifest(reference, converged=True)
    calculation.update(fields)
    result = comparison.compare(reference, calculation)
    assert result["status"] == "incompatible_conventions"
    assert "calculation_method_missing" in result["compatibility_reasons"]
    assert result["ratio"] is None
    assert result["passed"] is None


def test_no_second_application_of_electronic_or_symmetry_factors():
    reference = h3_reference()
    reference.update(electronic_weight=.5, symmetry_factor=2.)
    calculation = manifest(reference, ratio=1.4)
    assert comparison.compare(reference, calculation)["ratio"] == pytest.approx(1.4)


def test_method_alias_and_contradictions():
    reference = h3_reference()
    calculation = manifest(reference)
    calculation["reference_method"] = calculation.pop("method")
    assert comparison.compare(reference, calculation)["compatible"]
    calculation["method"] = "TRPMD"
    with pytest.raises(ValueError, match="disagree"):
        comparison.compare(reference, calculation)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1e-16"])
@pytest.mark.parametrize("field", ["rate_cm3_molecule_s", "temperature_K"])
def test_nonpositive_nonfinite_or_nonnumeric_calculation_numbers_rejected(field, value):
    reference = h3_reference()
    calculation = manifest(reference)
    calculation[field] = value
    with pytest.raises(ValueError, match="positive number"):
        comparison.compare(reference, calculation)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_tolerance_rejected(value):
    reference = h3_reference()
    with pytest.raises(ValueError, match="relative_tolerance"):
        comparison.compare(reference, manifest(reference), value)


def test_reference_rate_and_uncertainty_are_validated():
    reference = h3_reference()
    calculation = manifest(reference)
    reference["rate_cm3_molecule_s"] = -1
    with pytest.raises(ValueError, match="reference.rate"):
        comparison.compare(reference, calculation)
    calculation["uncertainty"]["standard_error_cm3_molecule_s"] = -1
    with pytest.raises(ValueError, match="standard_error"):
        comparison.compare(h3_reference(), calculation)


def test_cli_emits_reproducible_pilot_report_and_protects_inputs(tmp_path):
    calculation = tmp_path / "calculation.json"
    calculation.write_text(json.dumps(manifest(h3_reference())), encoding="utf-8")
    original = calculation.read_bytes()
    output = tmp_path / "comparison.json"
    command = [sys.executable, str(SCRIPT), "--reference-id", "h_h2_bkmp2_rpmd_300k",
               "--calculation", str(calculation), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    saved = json.loads(output.read_text())
    assert saved["status"] == "pilot_comparison" and saved["passed"] is None
    assert len(saved["input_files"]["calculation_sha256"]) == 64
    command[-1] = str(calculation)
    blocked = subprocess.run(command, capture_output=True, text=True)
    assert blocked.returncode == 2
    assert calculation.read_bytes() == original
