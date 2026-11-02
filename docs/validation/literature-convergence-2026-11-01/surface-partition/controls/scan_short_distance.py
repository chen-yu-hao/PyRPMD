"""Finite-grid diagnostic of the known short-distance BKMP2 extrapolation.

This is not a global PES bound or a statistical convergence test. The fixed
equilibrium-control domains and their registered acceptance gates are unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.pes import BKMP2Potential


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    directory = Path(__file__).parent
    first, second, cosine = np.meshgrid(np.linspace(.25, .8, 23),
        np.linspace(.25, 3., 45), np.linspace(-1., 1., 49), indexing="ij")
    q = np.zeros((first.size, 3, 3))
    q[:, 0, 0] = first.ravel()
    q[:, 2, 0] = (second*cosine).ravel()
    q[:, 2, 1] = (second*np.sqrt(np.maximum(0., 1.-cosine**2))).ravel()
    distances = np.stack([np.linalg.norm(q[:, i]-q[:, j], axis=1)
                          for i, j in ((0, 1), (1, 2), (0, 2))], axis=1)
    keep = distances.min(axis=1) >= .25-1e-14
    q, distances = q[keep], distances[keep]
    provider = BKMP2Potential()
    result = provider.evaluate(q)
    if np.any(result.status) or not np.isfinite(result.energy).all():
        raise ValueError("finite-grid PES evaluation failed")
    minimum = distances.min(axis=1)
    rows = []
    for cutoff in (.25, .3, .4, .5, .6):
        selected = np.flatnonzero(minimum >= cutoff-1e-14)
        near = np.flatnonzero((minimum >= cutoff-1e-14) & (minimum <= cutoff+.025+1e-14))
        index, boundary = selected[np.argmin(result.energy[selected])], near[np.argmin(result.energy[near])]
        rows.append({"minimum_pair_bohr": cutoff, "retained_grid_geometries": len(selected),
                     "minimum_energy_hartree": float(result.energy[index]),
                     "minimum_energy_pair_distances_bohr": distances[index].tolist(),
                     "boundary_band_bohr": [cutoff, cutoff+.025],
                     "boundary_band_geometries": len(near),
                     "boundary_band_minimum_energy_hartree": float(result.energy[boundary]),
                     "boundary_band_minimum_energy_pair_distances_bohr": distances[boundary].tolist()})
    anchors = []
    for d in ((.3, .3, .45), (.4, .4, .4), (.5, .5, .5), (.6, .6, .6)):
        energy, gradient = provider.radial(d)
        anchors.append({"pair_distances_bohr": list(d), "energy_hartree": energy,
                        "radial_gradient_hartree_per_bohr": gradient.tolist()})
    raw = directory / "short-distance-grid.npz"
    np.savez_compressed(raw, pair_distances_bohr=distances, energy_hartree=result.energy)
    report = {
        "schema": "pyrpmd.h3.finite-grid-short-distance-diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Known unphysical deep BKMP2 extrapolation around 0.3 bohr; document the separately registered cutoff stress variants.",
        "grid": {"r01_bohr": [.25, .8, 23], "r12_bohr": [.25, 3., 45],
                 "cos_angle": [-1., 1., 49], "minimum_retained_pair_bohr": .25,
                 "pair_order": [[0, 1], [1, 2], [0, 2]],
                 "retained_geometries": len(q)},
        "cutoff_summaries": rows, "explicit_radial_anchors": anchors,
        "source_sha256": sha(__file__), "PES": provider.metadata,
        "controls_protocol_sha256": sha(directory/"protocol.json"),
        "raw": {"file": raw.name, "sha256": sha(raw), "bytes": raw.stat().st_size},
        "sampling_settings_changed": False, "acceptance_gates_changed": False,
        "limitations": ["A finite geometry grid is not an exhaustive lower bound on the PES.",
                        "These energies do not establish the equilibrium probability of an omitted domain.",
                        "The registered .4/.5/.6-bohr stochastic controls remain the statistical domain comparison."],
    }
    (directory/"short-distance-domain-check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"cutoff_summaries": rows, "explicit_radial_anchors": anchors}))


if __name__ == "__main__":
    main()
