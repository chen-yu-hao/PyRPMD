"""Matrix of transfer-error points for .github/workflows/subspace-transfer.yml (prints JSON)."""

import json

TEMPERATURES = (600.0, 300.0, 200.0)  # 1.45, 0.73, 0.48 T_c of the symmetric barrier


def points(curvature_only=False):
    out = []
    for barrier in ("symmetric", "asymmetric"):
        for h in (0.0, 0.3, 0.6, 0.9):
            for t in TEMPERATURES:
                for curvature_mass in ("none", "small"):
                    out.append({"family": "curvature", "parameter": h, "barrier": barrier, "temperature": t,
                                "curvature_mass": curvature_mass, "beads": 0, "seed": 0})
    if not curvature_only:
        for w in (0.0035, 0.007, 0.014):
            for t in TEMPERATURES:
                out.append({"family": "frequency", "parameter": w, "barrier": "symmetric", "temperature": t,
                            "curvature_mass": "none", "beads": 0, "seed": 0})
    # Finite-bead test: Delta(P) + ln B_P should not depend on P.
    if not curvature_only:
        for p in (8, 16, 32):
            out.append({"family": "frequency", "parameter": 0.014, "barrier": "symmetric", "temperature": 200.0,
                        "curvature_mass": "none", "beads": p, "seed": 100 + p})
    for item in out:
        item["truncation"] = 0.07
    # Truncation check of the exact 2D rate at 0.10 Eh (spacings 1.0 and 0.8 are in every point).
    checks = (("curvature", 0.9, "symmetric", 200.0, "none"),
              ("curvature", 0.9, "asymmetric", 200.0, "none"))
    if curvature_only:
        checks = tuple((family, parameter, barrier, temperature, mass)
                       for family, parameter, barrier, temperature, mass in checks
                       for mass in ("none", "small"))
    else:
        checks += (("frequency", 0.0035, "symmetric", 600.0, "none"),)
    for family, parameter, barrier, t, curvature_mass in checks:
        out.append({"family": family, "parameter": parameter, "barrier": barrier, "temperature": t,
                    "curvature_mass": curvature_mass, "beads": 0, "seed": 0, "truncation": 0.10})
    out = [dict(item, zero_point=zp) for item in out
           for zp in (("harmonic", "rph", "dvr") if item["family"] == "curvature" else ("harmonic",))]
    for item in out:
        suffix = "-sct" if item.get("curvature_mass") == "small" else ""
        item["name"] = "{family}-{parameter:g}-{barrier}-{temperature:g}K-{zero_point}-P{beads}-t{truncation:g}{suffix}".format(**item, suffix=suffix)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--curvature-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(points(curvature_only=args.curvature_only)))
