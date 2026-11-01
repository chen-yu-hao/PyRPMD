#!/usr/bin/env python3
"""Analytic, excluded-fixture noise audit for the unadopted water RDF gate.

No trajectory, diffusion estimate, or realized liquid RDF is read. Independent
Poisson shell counts are a deliberately simple null model, not a claim about
the correlated counts in liquid water. The Skellam probabilities are exact for
that model; their application to an MD histogram would need independent checks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import skellam


ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = Path("docs/validation/literature-convergence-2026-11-01/water/equilibration-extension-proposal.json")
PROPOSAL_SHA256 = "a5a9c85a2e2093839f3fd688516d65adef825e162b21351708fe5fa547acb1d6"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unordered_pair_exposure(edges_angstrom, *, molecules, box_angstrom, independent_frames):
    """Expected unordered counts in each shell when g(r)=1, including N-1."""
    edges = np.asarray(edges_angstrom, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("shell edges must be finite, nonnegative, and strictly increasing")
    if (isinstance(molecules, bool) or not isinstance(molecules, int) or molecules < 2
            or isinstance(independent_frames, bool) or not isinstance(independent_frames, int)
            or independent_frames < 1):
        raise ValueError("molecules >=2 and independent_frames >=1 must be integers")
    if not np.isfinite(box_angstrom) or box_angstrom <= 0 or edges[-1] > box_angstrom / 2:
        raise ValueError("spherical shells must fit inside half the cubic box")
    shell_volume = 4 * np.pi / 3 * np.diff(edges**3)
    return independent_frames * molecules * (molecules - 1) / 2 * shell_volume / box_angstrom**3


def unchanged_count_noise(exposure, *, g=1.0, margin=.1, chains=8):
    """Two independent windows, each count X~Poisson(exposure*g).

    The reported single-bin probability is P(|X2-X1|/exposure > margin).
    The familywise products additionally assume independent bins and chains.
    Union bounds need no bin/chain independence but still require these null
    marginal distributions. Neither is a calibrated false-alarm rate for MD.
    """
    exposure = np.asarray(exposure, dtype=float)
    intensity = np.broadcast_to(np.asarray(g, dtype=float), exposure.shape)
    if (exposure.ndim != 1 or not len(exposure) or not np.isfinite(exposure).all()
            or np.any(exposure <= 0) or not np.isfinite(intensity).all() or np.any(intensity < 0)):
        raise ValueError("exposure must be positive and g must be nonnegative finite vectors")
    if (not np.isfinite(margin) or margin < 0 or isinstance(chains, bool)
            or not isinstance(chains, int) or chains < 1):
        raise ValueError("margin must be nonnegative and chains must be a positive integer")
    rate = exposure * intensity
    probability = np.zeros_like(rate)
    occupied = rate > 0
    probability[occupied] = 2 * skellam.sf(np.floor(margin * exposure[occupied]),
                                           rate[occupied], rate[occupied])
    probability = np.clip(probability, 0, 1)
    with np.errstate(divide="ignore"):
        log_pass_one_chain = float(np.log1p(-probability).sum())
    return {
        "exposure": exposure,
        "g": intensity,
        "expected_counts_per_window": rate,
        "difference_standard_deviation": np.sqrt(2 * intensity / exposure),
        "single_bin_false_alarm_probability": probability,
        "one_chain_independent_bins_false_alarm_probability": float(-np.expm1(log_pass_one_chain)),
        "all_chains_independent_bins_and_chains_false_alarm_probability": float(-np.expm1(chains * log_pass_one_chain)),
        "one_chain_union_bound": float(min(1, probability.sum())),
        "all_chains_union_bound": float(min(1, chains * probability.sum())),
    }


@dataclass(frozen=True)
class Case:
    name: str
    independent_frames: int
    g: float
    interpretation: str


def generate(output):
    proposal = ROOT / PROPOSAL
    if sha256(proposal) != PROPOSAL_SHA256:
        raise ValueError("the original unadopted proposal changed; preserve it and version the design explicitly")
    declaration = json.loads(proposal.read_text())
    if (declaration["structural_diagnostics"]["histogram_bin_width_angstrom"] != .02
            or declaration["structural_diagnostics"]["fixed_g_difference_range_angstrom"] != [2.3, 6.0]
            or declaration["structural_diagnostics"]["per_seed_max_abs_g_difference_threshold"] != .1):
        raise ValueError("this audit is bound to the original 185-bin, 0.1-margin proposal")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("audit output must be new/empty")
    edges = np.linspace(2.3, 6.0, 186)
    # This fixed box is a model input, not a measured RDF or fitted observable.
    # It is the frozen H2O-216 box at 0.997 g/cm3 with the declared isotope masses.
    box = 18.642822871774577
    cases = (
        Case("uniform_g1_200_independent_frames", 200, 1.0,
             "Every recorded frame is independent; uniform g=1 in the assessed shells."),
        Case("uniform_g1_20_independent_frames", 20, 1.0,
             "Illustrative tenfold lower independent-count exposure; not an estimated effective sample size."),
        Case("uniform_g3_200_independent_frames", 200, 3.0,
             "Synthetic high-occupancy shell model g=3; not a fitted or physically validated liquid RDF."),
    )
    summaries, rows = [], []
    for case in cases:
        exposure = unordered_pair_exposure(edges, molecules=216, box_angstrom=box,
                                           independent_frames=case.independent_frames)
        result = unchanged_count_noise(exposure, g=case.g, margin=.1, chains=8)
        summaries.append({"case": case.name, "independent_frames_per_window": case.independent_frames,
                          "g_in_every_assessed_shell": case.g, "interpretation": case.interpretation,
                          "minimum_difference_standard_deviation": float(result["difference_standard_deviation"].min()),
                          "maximum_difference_standard_deviation": float(result["difference_standard_deviation"].max()),
                          "maximum_single_bin_false_alarm_probability": float(result["single_bin_false_alarm_probability"].max()),
                          **{key: value for key, value in result.items() if np.isscalar(value)}})
        for index, radius in enumerate((edges[:-1] + edges[1:]) / 2):
            rows.append({"case": case.name, "radius_angstrom": float(radius),
                         "shell_lower_angstrom": float(edges[index]),
                         "shell_upper_angstrom": float(edges[index+1]),
                         **{key: float(result[key][index]) for key in
                            ("exposure", "g", "expected_counts_per_window", "difference_standard_deviation",
                             "single_bin_false_alarm_probability")}})
    report = {
        "schema": "pyrpmd.water-rdf-analytic-noise-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(Path(__file__)), "scipy_version": scipy.__version__,
        "proposal_path": str(PROPOSAL), "proposal_sha256": PROPOSAL_SHA256,
        "real_trajectory_or_diffusion_values_read": False,
        "counts_are_excluded_analytic_fixture": True,
        "proposal_adopted": False,
        "design": {"molecules": 216, "box_angstrom": box, "chains": 8, "bin_count": 185,
                   "bin_width_angstrom": .02, "range_angstrom": [2.3, 6.0], "margin": .1,
                   "window_ps": 100, "recording_interval_ps": .5, "frames_per_window": 200},
        "equations": {
            "exposure": "A_i = F*N*(N-1)/2 * (4*pi/3)*(r_hi^3-r_lo^3)/V",
            "null": "X_1i and X_2i independently Poisson(A_i*g_i); estimated g_ji=X_ji/A_i",
            "variance_of_difference": "Var(g_2i-g_1i)=2*g_i/A_i",
            "strict_single_bin_tail": "P(|g_2i-g_1i|>d) = 2*Skellam.sf(floor(d*A_i), A_i*g_i, A_i*g_i)",
            "independent_familywise": "P(any false alarm in all chains)=1-product_i(1-p_i)^chains",
        },
        "cases": summaries,
        "limitations": [
            "The exact Skellam probabilities apply only to the stated independent-Poisson-count null.",
            "Liquid pair structure, mutual pair/bin dependencies and time correlations are not modeled or estimated.",
            "The g=1 calculation is not a lower bound for water: depleted core bins reduce noise and peaks increase it.",
            "The 20-frame case is a hypothetical exposure change, not an inference of an autocorrelation time.",
            "Products over bins and chains require extra independence; union bounds use only the toy null marginals.",
            "No lack of a significant change, and no noisy interval including zero, establishes structural equivalence.",
        ],
        "recommendation_before_adoption": (
            "Retain every per-seed RDF curve and the descriptive maximum difference, but do not adopt the new "
            "0.1 all-bin maximum as a mandatory equilibrium pass gate on the strength of the proposed sample count. "
            "Predeclare correlation-aware block and simultaneous uncertainty assessment, with insufficient precision "
            "reported as unresolved. Preserve the original temperature, energy, OH and HOH thresholds unchanged."
        ),
        "primary_implementation_reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skellam.html",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "bin-noise.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["bin_noise_sha256"] = sha256(output / "bin-noise.csv")
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps(result["cases"], indent=2))


if __name__ == "__main__":
    main()
