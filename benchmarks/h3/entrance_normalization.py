"""Compute the physical H+H2 entrance bridge from joint unbiased MBAR samples.

Required NPZ arrays (no pickle): centroids [N,P,3,3] or [N,3,3],
unbiased_weights [N], masses_au [3], beta_au scalar, canonical_identity
scalar, measure='joint_unbiased_canonical', overlap_connected bool, and
sample_set_sha256.  An independently gauge-fixed 2D PMF is NOT a joint sample
set and must not be passed as these weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pyrpmd.analysis.entrance import h3_entrance_from_centroid_samples


def run(path, *, Rinf, hR, hS, min_effective=20):
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        required = {"centroids", "unbiased_weights", "masses_au", "beta_au",
                    "canonical_identity", "measure", "overlap_connected",
                    "sample_set_sha256"}
        if not required.issubset(source.files):
            raise ValueError(f"joint ensemble fields missing: {sorted(required-set(source.files))}")
        if not str(source["sample_set_sha256"].item()).strip():
            raise ValueError("missing whole-sample-set source identity")
        result = h3_entrance_from_centroid_samples(
            source["centroids"], source["unbiased_weights"],
            source["masses_au"], float(source["beta_au"]),
            radius_bohr=Rinf, reference_width_bohr=hR,
            target_width_bohr=hS,
            canonical_identity=str(source["canonical_identity"].item()),
            measure=str(source["measure"].item()),
            connected=bool(source["overlap_connected"].item()),
            min_effective=min_effective)
    result["input_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["input_file_name"] = path.name
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--Rinf", type=float, default=10.)
    p.add_argument("--reference-half-width", type=float, default=.2)
    p.add_argument("--target-half-width", type=float, default=.1)
    p.add_argument("--min-effective", type=float, default=20.)
    args = p.parse_args()
    result = run(args.samples, Rinf=args.Rinf,
                 hR=args.reference_half_width,
                 hS=args.target_half_width,
                 min_effective=args.min_effective)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
