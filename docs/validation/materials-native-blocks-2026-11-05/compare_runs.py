"""Compare two materials CLI run directories array-by-array (bitwise and max abs diff)."""
import json, sys
from pathlib import Path
import numpy as np

def load(run):
    out = {}
    for sub in ("checkpoints", "samples"):
        for f in sorted((Path(run) / sub).glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    out[f"{sub}/{f.name}:{k}"] = z[k]
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
report = {"only_left": sorted(set(a) - set(b)), "only_right": sorted(set(b) - set(a)), "arrays": {}}
for k in sorted(set(a) & set(b)):
    x, y = a[k], b[k]
    if x.dtype.kind in "fc" and x.shape == y.shape:
        report["arrays"][k] = {"identical": bool(np.array_equal(x, y)), "max_abs": float(np.max(np.abs(x - y))) if x.size else 0.0}
    else:
        report["arrays"][k] = {"identical": bool(x.shape == y.shape and np.array_equal(x, y))}
sa, sb = json.loads((Path(sys.argv[1]) / "summary.json").read_text()), json.loads((Path(sys.argv[2]) / "summary.json").read_text())
for key in ("propagation", "nve_energy_diagnostics", "diffusion", "diffusion_m2_s", "samples_per_replica"):
    report.setdefault("summary", {})[key] = sa.get(key) == sb.get(key)
bad = [k for k, v in report["arrays"].items() if not v["identical"]]
report["all_identical"] = not bad and not report["only_left"] and not report["only_right"] and all(report["summary"].values())
print(json.dumps(report, indent=1))
