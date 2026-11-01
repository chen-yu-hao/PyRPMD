"""Independent read-only arithmetic/provenance replay; never loads a PES."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import struct
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    inventory_path = HERE / "execution-inventory.json"
    assert identity(inventory_path)["sha256"] == "b137e41a669cdfdd51a6793f85225a85ffecfd9530e4e1f710cc8987b455b935"
    inventory = read(inventory_path)
    assert len(inventory["files"]) == inventory["file_count"] == 25
    assert sum(row["bytes"] for row in inventory["files"]) == inventory["total_bytes"] == 65542
    for row in inventory["files"]:
        assert identity(ROOT / row["path"]) == row
    protocol = read(ROOT / "benchmarks/fh2_reference/static_protocol.json")
    sources = read(ROOT / "benchmarks/fh2_reference/sources.json")
    report = read(HERE / "report.json")
    sw = read(HERE / "sw1-results.json")
    hsw = read(HERE / "hsw-results.json")
    for worker in (sw, hsw):
        assert worker["protocol_sha256"] == identity(ROOT / "benchmarks/fh2_reference/static_protocol.json")["sha256"]
        assert worker["runner_sha256"] == identity(ROOT / "benchmarks/fh2_reference/run_static_check.py")["sha256"]
        assert worker["dynamics_steps"] == 0
        assert len(worker["geometries"]) == 10
    expected = protocol["geometries"]
    for worker in (sw, hsw):
        for row, geometry in zip(worker["geometries"], expected, strict=True):
            assert row["id"] == geometry["id"]
            assert row["distances_bohr"] == geometry["distances_bohr"]
            assert math.isfinite(row["energy_hartree"])
    finite_error_rows = []
    for row in sw["geometries"]:
        assert [check["step_bohr"] for check in row["finite_differences"]] == protocol["finite_difference_steps_bohr"]
        gradient = row["analytic_gradient_hartree_per_bohr"]
        assert len(gradient) == 3 and all(math.isfinite(x) for x in gradient)
        for check in row["finite_differences"]:
            difference = max(abs(a - b) for a, b in zip(gradient, check["gradient_hartree_per_bohr"], strict=True))
            assert difference == check["max_absolute_error"]
            finite_error_rows.append(difference)
    a, b = sw["geometries"], hsw["geometries"]
    cross = [abs(x["energy_hartree"] - y["unmodified_sw1_component_hartree"])
             for x, y in zip(a, b, strict=True)]
    stdout = (HERE / "hsw-author-verify.stdout.txt").read_text()
    executable_values = [float(x.replace("D", "E")) for x in re.findall(r"your calculated value:\s*([+\-\d.EeDd]+)", stdout)]
    authors = [row for row in b if "author_decimal_hartree" in row]
    assert len(executable_values) == len(authors) == 2
    author_errors = [abs(row["energy_hartree"] - row["author_decimal_hartree"]) for row in authors]
    for row, difference in zip(authors, author_errors, strict=True):
        assert difference == row["author_decimal_absolute_difference_hartree"]
    limits = protocol["acceptance"]
    observed = {
        "SW1_two_energy_implementations": max(row["independent_energy_code_difference_hartree"] for row in a),
        "SW1_analytic_distance_gradient_all_three_steps": max(finite_error_rows),
        "SW1_H_exchange_gradient": max(row["H_exchange_gradient_error"] for row in a),
        "both_H_exchange_energies": max(row["H_exchange_energy_error_hartree"] for row in a + b),
        "same_SW1_component_in_both_archives": max(cross),
        "HSW_two_author_decimal_targets": max(author_errors),
        "HSW_unchanged_author_program_vs_wrapper": max(abs(x - row["energy_hartree"]) for x, row in zip(executable_values, authors, strict=True))}
    threshold = {
        "SW1_two_energy_implementations": limits["energy_absolute_hartree"],
        "SW1_analytic_distance_gradient_all_three_steps": limits["gradient_absolute_hartree_per_bohr"],
        "SW1_H_exchange_gradient": limits["gradient_symmetry_absolute_hartree_per_bohr"],
        "both_H_exchange_energies": limits["energy_absolute_hartree"],
        "same_SW1_component_in_both_archives": limits["energy_absolute_hartree"],
        "HSW_two_author_decimal_targets": limits["author_reference_absolute_hartree"],
        "HSW_unchanged_author_program_vs_wrapper": limits["energy_absolute_hartree"]}
    gates = {key: math.isfinite(value) and value <= threshold[key] for key, value in observed.items()}
    externals = []
    for row in report["external_assets"]:
        actual = identity(row["path"])
        assert actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"]
        externals.append(actual)
    for path, digest in report["source_sha256"].items():
        assert identity(path)["sha256"] == digest
    gates["sources_unchanged"] = True
    assert gates == report["gates"] and all(gates.values())
    assert cross == report["SW1_cross_archive_absolute_differences_hartree"]
    assert not report["potential_rate_reference_match"] and not report["thermal_rate_computed"]
    assert not report["paper_reproduction"] and report["dynamics_steps"] == 0
    assert sources["variants"]["sw1"]["spin_orbit_correction"] is False
    assert sources["variants"]["hsw"]["spin_orbit_correction"] is True
    rounding = struct.unpack("f", struct.pack("f", .0006))[0] - .0006
    residuals = [difference - rounding for difference in author_errors]
    prior_rounding = read(HERE / "legacy-literal-rounding.json")
    assert rounding == prior_rounding["offset_difference_hartree"]
    assert residuals == [row["residual_after_accounting_for_literal_rounding_hartree"] for row in prior_rounding["author_points"]]
    compiler = read(HERE / "compiler-observation.json")
    compiler_matches_now = identity(compiler["resolved_path"])["sha256"] == compiler["sha256"]
    after = resource.getrusage(resource.RUSAGE_SELF)
    output = {
        "schema": "pyrpmd.fh2.independent-static-evidence-review.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_source": identity(__file__), "frozen_execution_inventory": identity(inventory_path),
        "reviewer": "independent benchmarks agent; executed static audit was prepared by root",
        "verdict": "No blocker found for the stated ten-geometry static GNU-Fortran audit. All eight fixed gates replay from the saved evidence. This is not an independent PES rerun or a corrected thermal-rate validation.",
        "checked_published_files": 25, "checked_published_bytes": 65542,
        "gate_maxima": observed, "fixed_gate_limits": threshold, "replayed_gates": gates,
        "gradient_difference_vectors_recomputed": 30,
        "cross_archive_SW1_values_exactly_equal_count": sum(x == 0 for x in cross),
        "author_decimal_errors_hartree": author_errors,
        "default_real32_offset_promotion_error_hartree": rounding,
        "remaining_rounding_residuals_hartree": residuals,
        "compiler_hash_matches_current_postexecution_observation": compiler_matches_now,
        "compiler_provenance_scope": "Compiler identity was observed after execution. The original report records commands and flags; this review does not upgrade it to a pre-execution hash guard.",
        "static_interface_review": [
            "SW1 source declares double-precision by-reference sw(x,y,z,v); the caller initializes inifit and prepot separately before energy and derivative entry points.",
            "POTCM declares three R values, one energy and three derivatives, all REAL*8, matching the seven-double COMMON view. -fno-automatic preserves the legacy PREPOT/POTE parameter state.",
            "HSW pot(rFH1,rHH,rFH2) is a double-precision return value. It initializes its own parameters on the first call; the SW1 component is queried after that initialization.",
            "The order F-H1,H1-H2,F-H2 and bohr/hartree units match original source/readmes. Hydrogen exchange swaps first/third distances and derivative components.",
            "SW1 and HSW run in separate processes with separate parameter-file working directories. Their original COMMON/ENTRY state and GNU symbol ABI are not advertised as a portable or thread-safe production provider.",
            "Original Fortran source uses an unsuffixed0.0006 offset and default-real ftrue literals. Float32 promotion explains the recorded decimal discrepancies; no model energy or gate is corrected by this diagnosis."
        ],
        "external_asset_current_identities": externals,
        "scope_limits": [
            "Static evidence only: ten predeclared triangles, three finite-difference spacings and two author verification points.",
            "Only SW1 analytic distance derivatives were tested. No HSW gradient, Cartesian-force integration, spline-branch smoothness or global-domain accuracy is validated.",
            "The two official source variants remain distinct. Corrected F+H2 RPMD table/PES/electronic-factor matching is still unresolved.",
            "No source or library is redistributed by this review; no external fetch, compiler execution, PES call or dynamics step occurred.",
            "Energy-implementation and exchange errors whose raw alternate values are not stored are checked as recorded observations plus source-interface audit, not claimed to be independently reevaluated."
        ],
        "new_execution": {"PES_calls": 0, "dynamics_steps": 0, "compiler_invocations": 0,
                          "wall_seconds": time.perf_counter() - start,
                          "process_CPU_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                          "peak_RSS_KiB": after.ru_maxrss}}
    for row in inventory["files"]:
        assert identity(ROOT / row["path"]) == row
    target = HERE / "independent-review.json"
    with target.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(identity(target), sort_keys=True))


if __name__ == "__main__":
    main()
