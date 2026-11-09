"""Matrix of transfer-error points for .github/workflows/subspace-transfer.yml (prints JSON)."""

import json

TEMPERATURES = (600.0, 300.0, 200.0)  # 1.45, 0.73, 0.48 T_c of the symmetric barrier


def points():
    out = []
    for barrier in ("symmetric", "asymmetric"):
        for h in (0.0, 0.3, 0.6, 0.9):
            for t in TEMPERATURES:
                out.append({"family": "curvature", "parameter": h, "barrier": barrier, "temperature": t,
                            "beads": 0, "seed": 0})
    for w in (0.0035, 0.007, 0.014):
        for t in TEMPERATURES:
            out.append({"family": "frequency", "parameter": w, "barrier": "symmetric", "temperature": t,
                        "beads": 0, "seed": 0})
    # Finite-bead test: Delta(P) + ln B_P should not depend on P.
    for p in (8, 16, 32):
        out.append({"family": "frequency", "parameter": 0.014, "barrier": "symmetric", "temperature": 200.0,
                    "beads": p, "seed": 100 + p})
    for item in out:
        item["refine"] = 1.0
    # Convergence of the exact 2D rate: 0.75x spacing, truncation 33 kT (same RPMD seeds).
    for family, parameter, barrier, t in (("curvature", 0.9, "symmetric", 200.0), ("curvature", 0.9, "asymmetric", 600.0),
                                         ("frequency", 0.0035, "symmetric", 600.0)):
        out.append({"family": family, "parameter": parameter, "barrier": barrier, "temperature": t,
                    "beads": 0, "seed": 0, "refine": 0.75})
    for item in out:
        item["name"] = "{family}-{parameter:g}-{barrier}-{temperature:g}K-P{beads}-r{refine:g}".format(**item)
    return out


if __name__ == "__main__":
    print(json.dumps(points()))
