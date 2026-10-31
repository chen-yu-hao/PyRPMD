#!/usr/bin/env python3
"""Convention-aware comparison with the source-verified rate registry.

This harness compares supplied calculation manifests. It does not run dynamics,
infer missing statistical conventions, or certify the provenance claims. A
matched pilot may report a ratio, but cannot pass an implementation-validation
gate. Quantum reference differences describe method error, not code failure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import tempfile


RATE_UNIT = "cm3 molecule-1 s-1"
CONVERGENCE_FLAGS = ("beads", "timestep", "rinf", "sampling", "plateau")
CONVENTION_FIELDS = (
    "reaction", "pes", "nuclear_spin_convention", "electronic_convention",
    "symmetry_convention",
)
FACTOR_FIELDS = ("electronic_weight", "symmetry_factor")
UNRESOLVED_WORDS = ("unresolved", "unmatched", "unspecified", "unknown", "pending", "not_available")


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _resolved_text(value):
    return (isinstance(value, str) and bool(value.strip())
            and not any(word in value.lower() for word in UNRESOLVED_WORDS))


def _method(record):
    values = [record[key] for key in ("method", "reference_method")
              if record.get(key) is not None]
    if not values or any(not _resolved_text(value) for value in values):
        return None
    names = [value.strip().upper() for value in values]
    if len(set(names)) > 1:
        raise ValueError("method and reference_method disagree")
    return names[0]


def load_reference(reference_id, registry_path=None):
    """Return one registry case with its explicit top-level rate unit attached."""
    path = Path(registry_path) if registry_path is not None else Path(__file__).with_name("references.json")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("cases"), list):
        raise ValueError("unsupported published-rate registry schema")
    matches = [case for case in registry["cases"] if case.get("id") == reference_id]
    if len(matches) != 1:
        raise ValueError(f"reference ID must identify exactly one case: {reference_id}")
    reference = copy.deepcopy(matches[0])
    reference.setdefault("rate_unit", registry.get("rate_unit"))
    return reference


def _convergence_reasons(calculation, relative_tolerance):
    reasons = []
    convergence = calculation.get("convergence", {})
    evidence = calculation.get("convergence_evidence", {})
    if not isinstance(convergence, dict) or not isinstance(evidence, dict):
        raise ValueError("convergence and convergence_evidence must be objects")
    for flag in CONVERGENCE_FLAGS:
        if flag in convergence and not isinstance(convergence[flag], bool):
            raise ValueError(f"convergence.{flag} must be a boolean")
        if convergence.get(flag) is not True:
            reasons.append("convergence_not_established:" + flag)
            continue
        item = evidence.get(flag)
        if (not isinstance(item, dict) or not _resolved_text(item.get("source"))
                or not _resolved_text(item.get("criterion"))):
            reasons.append("convergence_evidence_missing:" + flag)
    provenance = calculation.get("provenance", {})
    if not isinstance(provenance, dict) or not _resolved_text(provenance.get("source")):
        reasons.append("calculation_provenance_missing")
    uncertainty = calculation.get("uncertainty", {})
    if not isinstance(uncertainty, dict):
        raise ValueError("uncertainty must be an object")
    for flag in ("includes_pmf", "includes_recrossing"):
        if flag in uncertainty and not isinstance(uncertainty[flag], bool):
            raise ValueError(f"uncertainty.{flag} must be a boolean")
        if uncertainty.get(flag) is not True:
            reasons.append("uncertainty_incomplete:" + flag)
    if not _resolved_text(uncertainty.get("method")):
        reasons.append("uncertainty_method_missing")
    chains = uncertainty.get("independent_parent_chains")
    if chains is not None and (isinstance(chains, bool) or not isinstance(chains, int) or chains < 0):
        raise ValueError("uncertainty.independent_parent_chains must be a nonnegative integer")
    if chains is None or chains < 2:
        reasons.append("fewer_than_two_independent_parent_chains")
    error = uncertainty.get("standard_error_cm3_molecule_s")
    if error is None:
        reasons.append("rate_standard_error_missing")
    else:
        if (isinstance(error, bool) or not isinstance(error, numbers.Real)
                or not math.isfinite(error) or error < 0):
            raise ValueError("uncertainty.standard_error_cm3_molecule_s must be finite and nonnegative")
        if error == 0:
            reasons.append("zero_rate_standard_error_does_not_establish_precision")
        elif error / calculation["rate_cm3_molecule_s"] > relative_tolerance:
            reasons.append("rate_standard_error_exceeds_comparison_tolerance")
    return reasons


def compare(reference, calculation, relative_tolerance=.1):
    """Compare matched final rates without reapplying degeneracy factors.

    Required calculation identity fields are the registry's reaction, PES,
    temperature, spin/electronic/symmetry conventions and factors, plus method,
    ``rate_cm3_molecule_s`` and ``rate_unit``. ``reference_method`` is accepted
    as an alias for ``method`` in a calculation manifest.

    A positive finite rate is required for a supplied calculation. Missing
    reference values remain unavailable. Unresolved or unequal conventions
    produce no numeric ratio. Convergence flags alone never enable a gate:
    documented criteria/source references and complete rate uncertainty are
    required as well. Those records are caller assertions, not independently
    inspected files. For QM comparisons ``passed`` is always None.
    """
    tolerance = _positive(relative_tolerance, "relative_tolerance")
    if not isinstance(calculation, dict):
        raise ValueError("calculation must be an object")
    if reference is not None and not isinstance(reference, dict):
        raise ValueError("reference must be an object or None")
    rate = _positive(calculation.get("rate_cm3_molecule_s"), "calculation.rate_cm3_molecule_s")
    temperature = _positive(calculation.get("temperature_K"), "calculation.temperature_K")
    method = _method(calculation)
    convergence_reasons = _convergence_reasons(calculation, tolerance)
    result = {
        "schema_version": 1,
        "reference_id": None if reference is None else reference.get("id"),
        "reference_method": None,
        "calculation_method": method,
        "comparison_role": None,
        "comparison_kind": None,
        "status": "reference_not_available",
        "compatible": False,
        "compatibility_reasons": [],
        "convergence_reasons": convergence_reasons,
        "convergence_asserted_with_evidence": not convergence_reasons,
        "evidence_independently_verified": False,
        "gate_basis": "matched supplied metadata and declared convergence/uncertainty evidence",
        "rate_unit": RATE_UNIT,
        "reference_rate_cm3_molecule_s": None,
        "calculation_rate_cm3_molecule_s": rate,
        "ratio": None,
        "relative_difference": None,
        "absolute_relative_difference": None,
        "relative_tolerance": tolerance,
        "within_tolerance": None,
        "gate_eligible": False,
        "passed": None,
        "implementation_failure": None,
        "reference_source": None,
        "calculation_provenance": copy.deepcopy(calculation.get("provenance", {})),
        "uncertainty": copy.deepcopy(calculation.get("uncertainty", {})),
    }
    if reference is None:
        result["compatibility_reasons"].append("reference_missing")
        return result
    result["reference_source"] = copy.deepcopy(reference.get("source"))
    result["reference_method"] = reference_method = _method(reference)
    result["comparison_role"] = role = reference.get("comparison_role")
    if reference.get("status") == "not_available" or reference.get("rate_cm3_molecule_s") is None:
        result["compatibility_reasons"].append("reference_rate_not_available")
        return result
    reference_rate = _positive(reference["rate_cm3_molecule_s"], "reference.rate_cm3_molecule_s")
    reference_temperature = _positive(reference.get("temperature_K"), "reference.temperature_K")
    result["reference_rate_cm3_molecule_s"] = reference_rate
    reasons = result["compatibility_reasons"]
    if reference.get("status") not in ("verified_table", "verified_equation", "verified_dataset"):
        reasons.append("reference_not_source_verified")
    if reference.get("rate_unit", RATE_UNIT) != RATE_UNIT:
        reasons.append("reference_rate_unit_mismatch")
    if calculation.get("rate_unit") != RATE_UNIT:
        reasons.append("calculation_rate_unit_missing_or_mismatched")
    if not math.isclose(temperature, reference_temperature, rel_tol=1e-12, abs_tol=1e-9):
        reasons.append("temperature_K_mismatch")
    for field in CONVENTION_FIELDS:
        expected, actual = reference.get(field), calculation.get(field)
        if not _resolved_text(expected):
            reasons.append("reference_convention_unresolved:" + field)
        elif not _resolved_text(actual):
            reasons.append("calculation_convention_unresolved:" + field)
        elif expected != actual:
            reasons.append("convention_mismatch:" + field)
    for field in FACTOR_FIELDS:
        expected, actual = reference.get(field), calculation.get(field)
        if expected is None:
            reasons.append("reference_convention_unresolved:" + field)
            continue
        expected = _positive(expected, "reference." + field)
        if actual is None:
            reasons.append("calculation_convention_unresolved:" + field)
            continue
        actual = _positive(actual, "calculation." + field)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0):
            reasons.append("convention_mismatch:" + field)
    if reference_method == "QM":
        result["comparison_kind"] = "quantum_method_accuracy"
        if not isinstance(role, str) or not role.startswith("quantum_accuracy_reference"):
            reasons.append("QM_reference_cannot_be_an_implementation_target")
        elif "pending" in role:
            reasons.append("reference_comparison_role_pending")
        if method is None:
            reasons.append("calculation_method_missing")
    else:
        result["comparison_kind"] = "implementation_agreement"
        if reference_method != "RPMD" or role != "implementation_reference":
            reasons.append("unsupported_implementation_reference")
        if method != reference_method:
            reasons.append("calculation_method_mismatch")
    if reasons:
        result["status"] = "incompatible_conventions"
        return result
    ratio = rate / reference_rate
    difference = ratio - 1.
    if not math.isfinite(ratio) or not math.isfinite(difference):
        raise ValueError("rate ratio overflows; check rates and units")
    within_tolerance = abs(difference) <= tolerance or math.isclose(
        abs(difference), tolerance, rel_tol=1e-12, abs_tol=0)
    result.update(
        compatible=True, ratio=ratio, relative_difference=difference,
        absolute_relative_difference=abs(difference),
        within_tolerance=within_tolerance,
        status="pilot_comparison" if convergence_reasons else "quantum_accuracy_comparison",
    )
    if not convergence_reasons and reference_method == "RPMD":
        result.update(
            status="implementation_agreement", gate_eligible=True,
            passed=result["within_tolerance"],
            implementation_failure=not result["within_tolerance"],
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--calculation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=Path(__file__).with_name("references.json"))
    parser.add_argument("--relative-tolerance", type=float, default=.1)
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.calculation.resolve(), args.references.resolve()):
        parser.error("output must not overwrite the calculation or reference registry")
    try:
        reference = load_reference(args.reference_id, args.references)
        calculation_bytes = args.calculation.read_bytes()
        calculation = json.loads(calculation_bytes)
        result = compare(reference, calculation, args.relative_tolerance)
        result["input_files"] = {
            "calculation": str(args.calculation),
            "calculation_sha256": hashlib.sha256(calculation_bytes).hexdigest(),
            "references": str(args.references),
            "references_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest(),
        }
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=args.output.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            temporary.replace(args.output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(text, end="")
    if result["passed"] is False:
        return 1
    return 0 if result["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
