"""Additive exact stencil/reference audit; saved metadata only, no PES or RNG."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.constants import physical_constants, k


def run(data, reference_path, design_path, physics_path, output):
    data = Path(data)
    read = lambda p: json.loads(Path(p).read_text())
    identify = lambda p: {"path": str(Path(p).resolve()), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    inputs = [identify(p) for p in (data/"calibration-initial-ledger.json", data/"calibration-followup-ledger.json", data/"registration.json", reference_path, design_path, physics_path, __file__)]
    first, second = read(data/"calibration-initial-ledger.json"), read(data/"calibration-followup-ledger.json")
    registration = read(data/"registration.json")
    bindings = {}
    for path, suffix in ((reference_path, "/benchmarks/hch4/geometries.json"),
                         (physics_path, "/review/hch4-reference-feasibility/pilot-design/proposal.json")):
        candidates = [(name, value) for name, value in registration["source_sha256"].items() if name.endswith(suffix)]
        assert len(candidates) == 1 and identify(path)["sha256"] == candidates[0][1], "adopted fixed input differs"
        bindings[suffix] = {"adopted_source_path": candidates[0][0], "sha256": candidates[0][1]}
    assert identify(design_path)["sha256"] == registration["design_sha256"]
    design = read(design_path); fixed = read(physics_path)["physics"]["coordinate_constants"]
    angstrom = 1e-10/physical_constants["Bohr radius"][0]
    mass_conversion = physical_constants["atomic mass constant"][0]/physical_constants["electron mass"][0]
    masses = np.asarray(design["physics"]["masses_amu"])*mass_conversion
    beta = physical_constants["Hartree energy"][0]/(300*k)
    q = np.asarray(read(reference_path)["official_input"]["coordinates"])*angstrom
    q -= q[1]
    ez = q[3]/np.linalg.norm(q[3]); ex = q[0]-(q[0]@ez)*ez; ex /= np.linalg.norm(ex)
    frame = np.array([ex, np.cross(ez, ex), ez])
    z = np.concatenate(([np.linalg.norm(q[3]), np.linalg.norm(q[0]), q[0]@ez/np.linalg.norm(q[0])], frame@q[2], frame@q[4]))
    direction = frame@(q[5]-q[3]); direction /= np.linalg.norm(direction)
    if z[4] < 0:
        z[[4, 7]] *= -1; direction[1] *= -1
    transition = np.concatenate((z, direction[:2]/(1+direction[2])))
    r = 1.0890*angstrom
    tetra = np.concatenate(([r, r, -1/3],
        r*np.array([math.sqrt(8/9)*math.cos(2*math.pi/3), math.sqrt(8/9)*math.sin(2*math.pi/3), -1/3]),
        r*np.array([math.sqrt(8/9)*math.cos(4*math.pi/3), math.sqrt(8/9)*math.sin(4*math.pi/3), -1/3])))
    errors = {}
    def compare(a, b, name):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name
        error = float(np.max(np.abs(a-b), initial=0)); errors[name] = max(errors.get(name, 0.), error)
        assert np.allclose(a, b, atol=2e-12, rtol=2e-14), (name, error)
    compare(masses, fixed["masses_electron_mass"], "mass_units")
    compare(beta, fixed["beta_hartree_inverse"], "temperature_units")
    compare(np.linalg.norm(q[5]-q[3])-np.linalg.norm(q[3]), fixed["reference_difference_bohr"], "reference_bond_difference")
    checks = 0
    for target in ("alpha_0.999", "alpha_1.007", "entry"):
        entry = target == "entry"; spec = first["specifications"][target]
        expected_x = tetra if entry else transition
        scales = np.array(design["calibration"]["dimensionless_shape_scales_entry" if entry else "dimensionless_shape_scales_TS"]+([] if entry else [1/(2*math.sqrt(8))]*2))
        compare(expected_x, spec["x0"], "registered_reference_shape")
        compare(scales, spec["scales"], "registered_scales")
        for row in [r for r in first["rows"] if r["target"] == target]:
            bits = row["tag"].split(":"); y = np.zeros(len(expected_x))
            if bits[0] in ("diag", "refine"):
                y[int(bits[1])] = int(bits[2])*(.02 if bits[0] == "diag" else .01)
            elif bits[0] == "cross":
                y[int(bits[1])] = int(bits[3])*.02; y[int(bits[2])] = int(bits[4])*.02
            else:
                assert bits == ["center"]
            assert np.array_equal(y, row["y"]), "tag-to-stencil displacement differs"
            compare(expected_x+scales*y, row["x"], "actual_stencil_coordinate")
            checks += 1
        labels = ["shifted_center", *[f"probe:{i}:{s}" for i in range(len(expected_x)) for s in (-1, 1)]]
        labels += ["mirror:"+tag for tag in ["shifted_center", *[f"probe:{i}:{s}" for i in range(3) for s in (-1, 1)], "probe:3:1"]]
        actual = [r["tag"] for r in second["rows"] if r["target"] == target]
        assert actual == labels, "exact ordered followup/reflection inventory differs"
    assert checks == 711
    assert all(identify(row["path"]) == row for row in inputs)
    result = {"schema": "pyrpmd.hch4-calibration-reference-and-stencil-audit.v1", "all_checks_passed": True,
              "reference_shape_and_scales_reconstructed_from_pinned_metadata": True, "exact_initial_tag_displacements_checked": checks,
              "exact_ordered_followup_labels_checked": len(second["rows"]), "maximum_absolute_errors": errors,
              "tolerances": {"atol": 2e-12, "rtol": 2e-14}, "PES_calls": 0, "RNG_draws": 0,
              "pinned_physics_and_geometry_match_actual_registration": bindings,
              "original_inputs_unchanged": True, "inputs": inputs,
              "scope": "Additive input audit, preserving the already executed standalone replay and raw archive bytes. No physical or efficiency threshold change."}
    with Path(output).open("x") as f: json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    print(json.dumps({"all_checks_passed": True, "tag_displacements": checks, "maximum_absolute_errors": errors}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "design", "physics", "output"): parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args(); run(args.data, args.reference, args.design, args.physics, args.output)
