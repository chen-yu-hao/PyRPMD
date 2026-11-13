"""Deterministic check: on a 2D system the RPH model's quantum rate is the full rate on the same slice."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subspace_rph.models import Curved, model_run
from benchmarks.subspace_rph.reference import build_path, QUANTUM
from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate

T, out = float(sys.argv[1]), sys.argv[2]
potential = Curved(2)
run = model_run(potential, T, 32)
path, info = build_path(run)
gating = {"path_info": info, "validity": {}, "hessian_displacement": 5e-3}
rows = []
for spacing in ([2.47, 2.47], [1.9, 1.9]):
    options = {**QUANTUM, "truncation_min": 0.055, "dense_limit": 12000, "environment": "adiabatic",
               "spacing_2d": spacing, **gating}
    model, quantum, model_info, _ = pipeline.rph_model(run, path, options)
    t0 = time.time()
    rph = exact_quantum_rate(model, run.beta, **quantum)
    row = {"rph_spacing": spacing, "rph_shape": list(model.shape), "reference": model.reactant_reference,
           "rph_rate": rph.rate, "rph_seconds": time.time() - t0, "full": []}
    # The developed axis 0 is x here (both channel directions are +x); put a full-grid node on the slice.
    for h in (2.0, 1.5):
        x = np.arange(-170.0, 170.0 + 1e-9, h)
        x = x - x[np.argmin(np.abs(x - model.reactant_reference))] + model.reactant_reference
        axes = (x, np.arange(-35.0, 50.0 + 1e-9, h))
        X, Y = np.meshgrid(*axes, indexing="ij")
        full = SubspaceModel(axes, potential.energy(np.stack([X, Y], -1)), (1.0, 1.0), 0.0, "asymptotic",
                             model.reactant_reference)
        t0 = time.time()
        reference = exact_quantum_rate(full, run.beta, cap_width=40.0, normalization="slice",
                                       energy_floor=quantum["energy_floor"],
                                       truncation_energy=quantum["truncation_energy"], dense_limit=12000)
        row["full"].append({"spacing": h, "rate": reference.rate, "seconds": time.time() - t0})
        print(spacing, h, rph.rate, reference.rate, rph.rate / reference.rate, flush=True)
    rows.append(row)
    json.dump({"temperature_K": T, "rows": rows}, open(out, "w"), indent=1)
