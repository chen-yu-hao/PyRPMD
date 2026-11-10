"""Collect transfer-error point records into one table (JSON + printed summary)."""

import argparse
import glob
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="point JSON files or globs")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    files = sorted({f for pattern in args.inputs for f in glob.glob(pattern)})
    records = [json.load(open(f)) for f in files]
    records.sort(key=lambda r: (r["family"], r["barrier"], r["parameter"], -r["temperature"],
                                r.get("curvature_mass", "none"), r.get("zero_point", "harmonic"), r["nbeads"],
                                r.get("quantum_grid", {}).get("truncation_offset", 0.0)))
    rows = []
    for r in records:
        row = {k: r[k] for k in ("family", "barrier", "parameter", "temperature", "nbeads", "curvature_width_max",
                                 "beta_omega_reactant", "environment_bead_factor", "C_full", "wall_seconds")}
        row["zero_point"] = r.get("zero_point", "harmonic")
        row["curvature_mass"] = r.get("curvature_mass", "none")
        row["refine"] = r.get("quantum_grid", {}).get("truncation_offset", 0.0)
        row["grid_change"] = r["full"]["quantum"].get("grid_relative_change")
        row["k_qm_full"] = r["full"]["quantum"]["rate"]
        row["k_rpmd_full"] = r["full"]["rpmd"]["rate"]
        row["kappa_full"] = r["full"]["rpmd"]["kappa"]
        for e, v in r["projections"].items():
            for key in ("C_S", "delta", "delta_bead_corrected", "delta_standard_error", "G_quantum", "G_rpmd"):
                row[f"{e}.{key}"] = v[key]
        rows.append(row)
    with open(args.output, "w") as handle:
        json.dump({"rows": rows, "records": records}, handle, indent=1, default=float)
    head = ("family", "barrier", "par", "T", "P", "mass", "zero_point", "trunc", "k_QM(2D)", "grid", "k*rho", "C_full", "C_S(zpe)", "Delta(zpe)", "+-",
            "Delta(fe)", "G_QM", "G_RP", "B_P")
    print("TABLE " + " | ".join(head))
    for row in rows:
        values = (row["family"], row["barrier"], f"{row['parameter']:g}", f"{row['temperature']:g}", row["nbeads"], row["curvature_mass"], row["zero_point"],
                  f"{row['refine']:g}", f"{row['k_qm_full']:.5e}", f"{(row['grid_change'] or 0):+.3f}",
                  f"{row['curvature_width_max']:.2f}", f"{row['C_full']:.3f}", f"{row.get('adiabatic.C_S', np.nan):.3f}",
                  f"{row.get('adiabatic.delta', np.nan):+.3f}", f"{row.get('adiabatic.delta_standard_error', np.nan):.3f}",
                  f"{row.get('free_energy.delta', np.nan):+.3f}", f"{row.get('adiabatic.G_quantum', np.nan):+.3f}",
                  f"{row.get('adiabatic.G_rpmd', np.nan):+.3f}", f"{row['environment_bead_factor']:.3f}")
        print("TABLE " + " | ".join(str(v) for v in values))
    print(f"POINTS {len(rows)}")


if __name__ == "__main__":
    main()
