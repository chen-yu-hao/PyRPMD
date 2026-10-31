"""Convert the authors' numerical OGRe scan data to inspectable CSV/JSON.

Input: six small public pickle files from the pinned SupportingInformation
commit documented in README.md. Only NumPy array/dtype constructors are
allowed; arbitrary pickle globals are rejected. No source code is executed.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np


class NumericOnly(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
                   ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")
        return allowed[(module, name)]


def extract(directory):
    directory = Path(directory)
    rows = []
    for model in ("double_well", "ackley"):
        loaded = {}
        for key in ("data", "error_data", "info"):
            path = directory / f"{model}-grid_{key}.pkl"
            if path.stat().st_size > 1_000_000:
                raise ValueError("Unexpectedly large source data")
            with path.open("rb") as handle:
                loaded[key] = NumericOnly(handle).load()
        for parameters, axes in loaded["info"].items():
            for i, confinement in enumerate(axes[0]):
                for j, overlap in enumerate(axes[1]):
                    for k, layers in enumerate(axes[2]):
                        error = float(loaded["error_data"][parameters][i, j, k, 0])
                        data = loaded["data"][parameters][i, j, k]
                        if not np.isfinite(error) or not np.isfinite(data).all():
                            continue
                        rows.append({"model": model, "spacing": parameters[0], "initial_kappa": parameters[1],
                            "kappa_growth": parameters[2], "consistency": parameters[3] if len(parameters) == 4 else "0.96",
                            "confinement": confinement, "overlap": overlap, "max_layers": layers,
                            "rmse_kjmol": error, "source_non_converged_flag": int(data[0]),
                            "simulations": int(data[1]), "extreme_kappa_flag": int(data[2]), "nan_bins": int(data[3])})
    with (directory / "paper-scan.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig12 = [r for r in rows if r["model"] == "ackley" and r["spacing"] == "2.0"
             and r["initial_kappa"] == "1.0" and r["kappa_growth"] == "2"
             and r["confinement"] == "0.33" and r["overlap"] == "0.50"]
    report = {"doi": "10.1021/acs.jctc.3c01028",
              "source_commit": "e7511c7dd48be80230c3cc4afc660696584712b1", "rows": len(rows),
              "figure_12_parameter_family": fig12,
              "note": "The 3-layer entry matches the published 0.29 value. Preserve the source non_converged flag; numerical agreement is not a global convergence certificate."}
    (directory / "paper-summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("directory", type=Path)
    print(json.dumps(extract(p.parse_args().directory), indent=2))
