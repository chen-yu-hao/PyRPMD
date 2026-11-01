#!/usr/bin/env python3
"""Audit local evidence bytes and numerical convergence, not scientific truth.

Unlike a boolean ``converged`` field, each convergence check contains two sets
of independent-replica estimates. The whole Welch confidence interval for the
difference must fit inside the predeclared equivalence band. This checks the
supplied arithmetic and file integrity, not how a simulation was conducted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def load_campaign(path=None):
    campaign = json.loads((Path(path) if path else HERE / "campaign.json").read_text())
    if campaign.get("schema_version") != 1 or not isinstance(campaign.get("cases"), list):
        raise ValueError("unsupported campaign schema")
    ids = [case.get("id") for case in campaign["cases"]]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("campaign IDs must be nonempty and unique")
    return campaign


def _artifact(spec, root):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("an artifact needs exactly path and sha256")
    name = spec["path"]
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("artifact path must be relative to the evidence bundle")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("artifact must be a file inside the evidence bundle")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if spec["sha256"] != digest:
        raise ValueError(f"artifact checksum mismatch: {name}")
    return content, {"path": name, "sha256": digest, "bytes": len(content)}


def _replicas(record, name):
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be an object")
    ids, values = record.get("replica_ids"), record.get("values")
    if (not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values)
            or len(values) < 3 or any(not isinstance(x, str) or not x.strip() for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(f"{name} requires at least three distinct independent replica IDs and values")
    values = [_positive(v, f"{name}.values") for v in values]
    return ids, values


def check_convergence(record, *, control, unit, identity, relative_tolerance=.05):
    """Conservative unpaired Welch equivalence check on replica estimates.

    Identity metadata must stay exactly equal between settings. Independent
    replicas, not frames or time origins, are the sampling units. Independence
    is a caller assertion and must still be reviewed scientifically.
    """
    tolerance = _positive(relative_tolerance, "relative_tolerance")
    if tolerance >= 1:
        raise ValueError("relative_tolerance must be less than one")
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or record.get("control") != control or record.get("unit") != unit
            or record.get("identity") != identity):
        raise ValueError("convergence record schema/control/unit/identity mismatch")
    coarse, fine = record.get("baseline"), record.get("refined")
    ids_a, a = _replicas(coarse, "baseline")
    ids_b, b = _replicas(fine, "refined")
    if set(ids_a) & set(ids_b):
        raise ValueError("unpaired convergence checks require disjoint independent replica IDs")
    p_a = _positive(coarse.get("parameter"), "baseline.parameter")
    p_b = _positive(fine.get("parameter"), "refined.parameter")
    if control == "timestep":
        refined = p_b <= p_a / 2
    else:
        refined = p_b > p_a
    if control in ("beads", "sampling"):
        refined = p_b >= 2 * p_a
    if not refined:
        raise ValueError("refined control must increase (double beads/sampling) or halve timestep")
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    if va + vb == 0:
        return {"passed": False, "reason": "zero replica variance does not establish statistical precision"}
    df = (va + vb)**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half_width = float(student_t.ppf(.975, df)) * math.sqrt(va + vb)
    change = mean_b - mean_a
    upper = (abs(change) + half_width) / abs(mean_b)
    return {
        "passed": bool(upper <= tolerance), "control": control, "unit": unit,
        "baseline_parameter": p_a, "refined_parameter": p_b,
        "baseline_mean": mean_a, "refined_mean": mean_b,
        "baseline_standard_error": math.sqrt(va), "refined_standard_error": math.sqrt(vb),
        "difference_confidence_interval_95": [change - half_width, change + half_width],
        "relative_equivalence_upper_bound": upper, "relative_tolerance": tolerance,
        "criterion": "entire unpaired Welch 95% difference interval within +/- tolerance * |refined mean|",
        "degrees_of_freedom": df, "replica_independence_independently_verified": False,
    }


def audit_bundle(bundle_path, *, campaign_path=None, relative_tolerance=.05):
    path = Path(bundle_path).resolve()
    root = path.parent
    bundle = json.loads(path.read_text())
    if bundle.get("schema_version") != 1:
        raise ValueError("unsupported evidence-bundle schema")
    matches = [c for c in load_campaign(campaign_path)["cases"] if c["id"] == bundle.get("case_id")]
    if len(matches) != 1:
        raise ValueError("case_id must identify one campaign case")
    case = matches[0]
    identity = bundle.get("identity")
    required = ("method", "model_sha256", "temperature_K", "observable")
    if not isinstance(identity, dict) or any(k not in identity for k in required):
        raise ValueError("identity requires method, model_sha256, temperature_K and observable")
    if identity["method"] != case["method"] or identity["observable"] != case["observable"]:
        raise ValueError("case method/observable mismatch; method comparisons need a separate case")
    _positive(identity["temperature_K"], "temperature_K")
    raw_result, result_artifact = _artifact(bundle.get("result"), root)
    result = json.loads(raw_result)
    if result.get("identity") != identity:
        raise ValueError("result identity differs from evidence bundle")
    raw_model, model_artifact = _artifact(bundle.get("model"), root)
    if hashlib.sha256(raw_model).hexdigest() != identity["model_sha256"]:
        raise ValueError("identity model hash differs from the supplied model bytes")
    records = bundle.get("convergence", {})
    if not isinstance(records, dict):
        raise ValueError("convergence must be an object")
    if set(records) - set(case["numeric_convergence_controls"]):
        raise ValueError("unexpected convergence control")
    checks, artifacts = {}, [result_artifact, model_artifact]
    for control in case["numeric_convergence_controls"]:
        if control not in records:
            checks[control] = {"passed": False, "reason": "evidence file not supplied"}
            continue
        raw, artifact = _artifact(records[control], root)
        artifacts.append(artifact)
        checks[control] = check_convergence(json.loads(raw), control=control,
            unit=case["unit"], identity=identity, relative_tolerance=relative_tolerance)
    numeric_passed = bool(checks) and all(x["passed"] for x in checks.values())
    return {
        "schema_version": 1, "case_id": case["id"], "identity": identity,
        "artifact_integrity_passed": True, "artifacts": artifacts,
        "convergence_checks": checks, "numeric_convergence_checks_passed": numeric_passed,
        "literature_agreement": None, "physical_validated": False,
        "requires_scientific_review": True,
        "remaining_reviews": case["required_scientific_reviews"],
        "scope": "Verifies supplied local bytes and replica arithmetic, not trajectory authenticity, statistical independence or literature agreement.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bundle:
        report = audit_bundle(args.bundle, campaign_path=args.campaign,
                              relative_tolerance=args.relative_tolerance)
        inputs = [args.bundle.resolve(), *(args.bundle.resolve().parent / a["path"] for a in report["artifacts"])]
        status = 0 if report["numeric_convergence_checks_passed"] else 1
    else:
        campaign = load_campaign(args.campaign)
        report = {"schema_version": 1, "physical_validated": False,
                  "cases": [{k: c[k] for k in ("id", "status", "blockers", "source_urls")}
                            for c in campaign["cases"]]}
        inputs = []
        status = 0  # inventory only, not a validation result
    inputs.append((args.campaign or HERE / "campaign.json").resolve())
    if args.output.resolve() in [p.resolve() for p in inputs]:
        parser.error("output must not replace an input artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "physical_validated": False}))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
