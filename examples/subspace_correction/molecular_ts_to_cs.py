"""Compute full-dimensional 3-atom (9D Cartesian) C_S and B_P directly from TS + PES.

Demonstrates the umbrella-free workflow when you already have (or plan to compute
separately) a full-dimensional rate k_RPMD and want the quantum subspace correction
factor C_S = k_QM(H_S) / k_RPMD(H_S) plus the complement finite-bead factor B_P
without running dozens of umbrella-sampling windows:

1. Refine an approximate, slightly bent 3D Cartesian transition-state guess to a
   first-order saddle with P-RFO while projecting out rigid translations and
   rotations (``rigid_body=True``), and compute T_c and normal-mode frequencies.
2. Assemble an in-memory 9D ``RateRun`` (``root=None``) and run
   ``run_subspace_pipeline(..., apply=False, run=run)`` with ``path.source="mep"``,
   solving both the 1D adiabatic MEP model and the 2D reaction-path Hamiltonian
   (``model_2d="rph"``) and anchoring both to the 9D Cartesian instanton relative
   to their bare skeletons (``anchor="skeleton"``).
3. Compute the complement-matched harmonic finite-bead factor ``B_P``
   (``harmonic_bead_factor``) for 1D (3 transverse TS modes vs 1 reactant stretch)
   and 2D RPH (only the 2 out-of-plane bends; the in-plane stretch is inside S),
   and optionally combine them with an external full-dimensional RPMD rate.

Usage:
    python examples/subspace_correction/molecular_ts_to_cs.py --out cs_9d.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from pyrpmd.analysis import (
    SubspaceCorrection,
    apply_subspace_correction,
    harmonic_bead_factor,
    run_subspace_pipeline,
)
from pyrpmd.analysis.subspace_paths import (
    _vibrational_basis,
    mass_weighted_hessian,
    refine_saddle,
    rigid_modes,
)
from pyrpmd.analysis.subspace_pipeline import RateRun
from pyrpmd.constants import AMU_TO_AU, KB_AU
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.instanton import crossover_temperature
from pyrpmd.potentials import PESResult

HARTREE_TO_CM = 219474.6313632


class TriatomicCartesianExchangePES:
    """3-atom (9D Cartesian) A + BC <-> AB + C exchange PES in atomic units.

    Invariant under 3D rigid translations and rotations: depends only on the pair
    distances r_01 = |x_0 - x_1|, r_12 = |x_2 - x_1| and bond angle cos(theta).
    """

    metadata = {
        "thread_safety": "reentrant",
        "execution_device": "cpu",
        "model": "analytic 9D Cartesian triatomic exchange",
    }

    def __init__(
        self,
        depth: float = 0.036,
        alpha: float = 2.0,
        equilibrium: float = 1.20,
        coupling: float = 0.144,
        bending: float = 0.14,
    ):
        self.depth = float(depth)
        self.alpha = float(alpha)
        self.equilibrium = float(equilibrium)
        self.coupling = float(coupling)
        self.bending = float(bending)

    def evaluate(self, positions: np.ndarray) -> PESResult:
        x = np.asarray(positions, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (3, 3):
            raise ValueError("TriatomicCartesianExchangePES expects shape (n_frames, 3, 3)")
        a = x[:, 0] - x[:, 1]
        b = x[:, 2] - x[:, 1]
        ra = np.linalg.norm(a, axis=1)
        rb = np.linalg.norm(b, axis=1)
        if np.any(ra < 1e-12) or np.any(rb < 1e-12):
            raise ValueError("coincident atoms")
        ua = a / ra[:, None]
        ub = b / rb[:, None]
        cos = np.sum(ua * ub, axis=1)
        ea = np.exp(-self.alpha * (ra - self.equilibrium))
        eb = np.exp(-self.alpha * (rb - self.equilibrium))
        pair = ea * eb
        coup = self.coupling + self.bending * (1.0 + cos)
        energy = self.depth * (ea * ea - 2.0 * ea + eb * eb - 2.0 * eb) + pair * coup
        dra = 2.0 * self.depth * self.alpha * ea * (1.0 - ea) - self.alpha * pair * coup
        drb = 2.0 * self.depth * self.alpha * eb * (1.0 - eb) - self.alpha * pair * coup
        ga = (
            dra[:, None] * ua
            + self.bending * pair[:, None] * (ub - cos[:, None] * ua) / ra[:, None]
        )
        gb = (
            drb[:, None] * ub
            + self.bending * pair[:, None] * (ua - cos[:, None] * ub) / rb[:, None]
        )
        forces = np.empty_like(x)
        forces[:, 0] = -ga
        forces[:, 2] = -gb
        forces[:, 1] = ga + gb
        return PESResult(energy=energy, forces=forces)


def _vibrational_frequencies_au(gradient_fn, x_cart: np.ndarray, masses_cart: np.ndarray) -> np.ndarray:
    """Signed vibrational frequencies (Eh, hbar=1) with rigid-body modes projected out."""
    basis = _vibrational_basis(x_cart, masses_cart, rigid_body=True)
    hess = basis.T @ mass_weighted_hessian(gradient_fn, x_cart, masses_cart) @ basis
    evals = np.linalg.eigvalsh(hess)
    return np.sign(evals) * np.sqrt(np.abs(evals))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=200.0, help="Temperature in K (default: 200 K < T_c)")
    parser.add_argument("--beads", type=int, default=24, help="Ring-polymer bead count P (default: 24)")
    parser.add_argument(
        "--masses-amu",
        type=float,
        nargs=3,
        default=[1.00782503223, 1.00782503223, 1.00782503223],
        help="Three atomic masses in amu (default: three 1H masses)",
    )
    parser.add_argument("--cpus", default="auto", help="CPU budget for parallel QM/RPMD tasks ('auto' or int)")
    parser.add_argument("--sweeps", type=int, default=300, help="Subspace HMC production sweeps per window")
    parser.add_argument("--trajectories", type=int, default=384, help="Paired subspace recrossing trajectories")
    parser.add_argument("--seed", type=int, default=17, help="RNG seed for subspace RPMD")
    parser.add_argument("--work-dir", type=Path, help="Optional directory to retain pipeline artifacts")
    parser.add_argument("--rpmd-rate", type=float, help="Optional full-dimensional k_RPMD to multiply by C_S / B_P")
    parser.add_argument("--rpmd-se", type=float, help="Optional standard error of --rpmd-rate")
    parser.add_argument("--out", type=Path, help="Optional JSON summary output path")
    args = parser.parse_args(argv)

    cpus = "auto" if str(args.cpus).strip().lower() == "auto" else int(args.cpus)
    pes = TriatomicCartesianExchangePES()
    masses_au = np.asarray(args.masses_amu, dtype=float) * AMU_TO_AU
    masses_cart = np.repeat(masses_au, 3)
    beta = 1.0 / (KB_AU * float(args.temperature))

    def energy_fn(x: np.ndarray) -> float:
        return float(pes.evaluate(np.asarray(x, dtype=float).reshape(1, 3, 3)).energy[0])

    def gradient_fn(x: np.ndarray) -> np.ndarray:
        return -np.asarray(pes.evaluate(np.asarray(x, dtype=float).reshape(1, 3, 3)).forces[0], dtype=float).ravel()

    # 1. Start from an approximate, slightly bent & asymmetric 3D Cartesian TS guess (bohr)
    #    and refine it in the 9D vibrational subspace (rigid translations/rotations removed).
    ts_guess = np.array([[-1.82, 0.06, -0.02], [0.01, -0.04, 0.01], [1.74, 0.05, 0.01]], dtype=float)
    t0 = time.perf_counter()
    saddle_flat, saddle_record = refine_saddle(
        gradient_fn, ts_guess.ravel(), masses_cart, rigid_body=True, gradient_tolerance=1e-9
    )
    if not saddle_record["saddle"]:
        raise RuntimeError(f"TS refinement did not converge to a first-order saddle: {saddle_record}")
    ts_bohr = saddle_flat.reshape(3, 3)
    rigid_ts = rigid_modes(saddle_flat, masses_cart, rigid_body=True)
    tc_kelvin = float(crossover_temperature(pes, ts_bohr, masses_au))

    ts_freqs = _vibrational_frequencies_au(gradient_fn, saddle_flat, masses_cart)
    ts_real = np.sort(ts_freqs[ts_freqs > 2.3e-4])  # [bend_1, bend_2, sym_stretch]

    # Asymptotic A + BC reactant geometry (atom 0 separated at 25 bohr, BC at r_eq).
    x_reactant = np.array([[-25.0, 0.0, 0.0], [0.0, 0.0, 0.0], [pes.equilibrium, 0.0, 0.0]], dtype=float)
    reactant_freqs = _vibrational_frequencies_au(gradient_fn, x_reactant.ravel(), masses_cart)
    reactant_real = np.sort(reactant_freqs[reactant_freqs > 2.3e-4])  # [diatomic_stretch]
    barrier_eh = energy_fn(saddle_flat) - energy_fn(x_reactant.ravel())

    # 2. Build an umbrella-free 9D RateRun and run 1D MEP + 2D RPH with 9D skeleton anchoring.
    equal_masses = np.allclose(masses_au, masses_au[0], rtol=0.0, atol=1e-8)
    eq_paths = [[0, 2, 1]] if equal_masses else None
    coordinate = RPMDrateCoordinate(
        [[0], [1, 2]], masses_au, ts_bohr, [[0, 1]], [[1, 2]], 10.0, equivalent_paths=eq_paths
    )
    system = SimpleNamespace(masses=masses_au, geometry=ts_bohr)
    reaction = {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]]}
    run = RateRun(
        root=None,
        config={},
        system=system,
        reaction=reaction,
        coordinate=coordinate,
        provider=pes,
        beta=beta,
        nbeads=int(args.beads),
        xi_star=1.0,
    )

    # Set truncation_min >= 6 quanta of the stiffest stretch mode so 2D DVR walls never squeeze ZPE.
    stiffest_omega = float(max(np.max(ts_real), np.max(reactant_real)))
    truncation_min = round(max(0.06, 6.0 * stiffest_omega), 4)

    temp_ctx = tempfile.TemporaryDirectory(prefix="pyrpmd_ts_cs_") if args.work_dir is None else None
    out_dir = args.work_dir if args.work_dir is not None else Path(temp_ctx.name)
    try:
        specification = {
            "output": {"directory": str(out_dir), "reuse": False},
            "path": {
                "source": "mep",
                "step": 4.0,
                "asymptote_tolerance": 1e-9,
                "instanton_beads": max(int(args.beads), 64),
            },
            "subspace": {
                "dimension": [1, 2],
                "threshold": 0.5,
                "environment": "adiabatic",
                "model_2d": "rph",
                "anchor": "skeleton",
            },
            "quantum": {
                "spacing_2d": [4.8, 4.2],
                "refinement": 3,
                "truncation_kt": 30.0,
                "truncation_min": truncation_min,
                "product_drop_kt": 18.0,
                "dense_limit": 8000,
            },
            "rpmd": {
                "nbeads": int(args.beads),
                "windows": 32,
                "walkers": 24,
                "burn_in": 100,
                "sweeps": int(args.sweeps),
                "trajectories": int(args.trajectories),
                "seed": int(args.seed),
            },
            "execution": {
                "backend": "process",
                "cpus": cpus,
                "blas_threads": 4,
                "memory_limit_mb": 16384.0,
            },
        }
        report = run_subspace_pipeline(specification, apply=False, run=run)
        if report["status"] != "complete" or set(report["validity"]["valid_dimensions"]) != {1, 2}:
            raise RuntimeError(
                f"Subspace pipeline did not validate both 1D and 2D ({report['status']}): {report['validity']['failures']}"
            )
        corr_objs = {
            int(d): SubspaceCorrection.read(out_dir / f"C_S_{d}d.json")
            for d in report["dimensions_computed"]
        }
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    elapsed = time.perf_counter() - t0

    # 3. Complement-matched harmonic finite-bead factors B_P = k_RPMD(P) / k_RPMD(inf).
    #    - 1D path model: all 3 transverse TS modes (2 bends + 1 symmetric stretch) and the
    #      reactant diatomic stretch belong to the orthogonal environment E = S_1D^perp.
    #    - 2D RPH model: the symmetric/reactant stretch is inside the 2D subspace S_2D (so its
    #      finite-P error is already corrected by C_S(2D)); only the 2 out-of-plane bends remain in E.
    bp_1d = harmonic_bead_factor(ts_real, reactant_real, beta, int(args.beads))
    bp_2d = harmonic_bead_factor(ts_real[:2], [], beta, int(args.beads))

    chosen_dim = int(report["applied_dimension"])
    chosen_corr = corr_objs[chosen_dim]
    chosen_bp = bp_2d if chosen_dim == 2 else bp_1d

    applied = None
    if args.rpmd_rate is not None:
        applied = apply_subspace_correction(
            args.rpmd_rate,
            chosen_corr,
            standard_error=args.rpmd_se,
            beta=beta,
            bead_factor=chosen_bp["factor"],
        )

    anchor_info = report["instanton_anchor"]
    c1 = report["corrections"]["1"]
    c2 = report["corrections"]["2"]
    a1 = anchor_info["dimensions"]["1"]
    a2 = anchor_info["dimensions"]["2"]

    print(
        f"System (9D Cartesian, rigid modes removed={rigid_ts.shape[1]}): "
        f"T={args.temperature:.1f} K (T_c={tc_kelvin:.1f} K, T/T_c={args.temperature / tc_kelvin:.2f}), "
        f"P={args.beads}, barrier={barrier_eh * 627.5095:.2f} kcal/mol"
    )
    print(
        f"Frequencies (cm^-1): TS imag={abs(ts_freqs[0]) * HARTREE_TO_CM:.1f}i, "
        f"TS real={[round(float(w * HARTREE_TO_CM), 1) for w in ts_real]}, "
        f"reactant={[round(float(w * HARTREE_TO_CM), 1) for w in reactant_real]}"
    )
    print(
        f"Path & curvature: MEP points={len(report['path']['reactant_termination'] and report['curvature_width_path']['arc_length'])}, "
        f"max(kappa*rho)={report['curvature_width_path']['maximum']:.3f} -> recommended {chosen_dim}D "
        f"({report['applied_dimension_reason']})"
    )
    print(
        f"9D instanton defect: eps_SD(9D)={anchor_info['full_defect']['factor']:.4f} "
        f"(A_H={anchor_info['full_defect']['log_factor']:+.4f})"
    )
    print(
        f"  1D MEP : C_S(unanch)={a1['unanchored_factor']:.4f} -> C_S(skel-anch)={c1['factor']:.4f} +- {c1['log_standard_error']:.4f} "
        f"[shift A_H - A_1D^0 = {a1['log_shift']:+.4f}], B_P(1D env)={bp_1d['factor']:.4f}"
    )
    print(
        f"  2D RPH : C_S(unanch)={a2['unanchored_factor']:.4f} -> C_S(skel-anch)={c2['factor']:.4f} +- {c2['log_standard_error']:.4f} "
        f"[shift A_H - A_2D^0 = {a2['log_shift']:+.4f}], B_P(2D env)={bp_2d['factor']:.4f}"
    )
    print(
        f"Recommended multiplier k_QM / k_RPMD(P) ~= C_S({chosen_dim}D) / B_P({chosen_dim}D) = "
        f"{chosen_corr.factor / chosen_bp['factor']:.4f}  (completed in {elapsed:.1f}s)"
    )
    if applied is not None:
        print(
            f"Applied to k_RPMD={args.rpmd_rate:.4e}: k_corr={applied['corrected_rate']:.4e}"
            + (f" +- {applied['corrected_rate_standard_error']:.4e}" if applied["corrected_rate_standard_error"] is not None else "")
        )

    summary = {
        "temperature_K": float(args.temperature),
        "crossover_temperature_K": tc_kelvin,
        "beads": int(args.beads),
        "masses_amu": [float(m) for m in args.masses_amu],
        "saddle": {
            "geometry_bohr": ts_bohr.tolist(),
            "rigid_modes_removed": int(rigid_ts.shape[1]),
            "barrier_hartree": barrier_eh,
            "imaginary_frequency_cm": float(abs(ts_freqs[0]) * HARTREE_TO_CM),
            "ts_real_frequencies_cm": [float(w * HARTREE_TO_CM) for w in ts_real],
            "reactant_frequencies_cm": [float(w * HARTREE_TO_CM) for w in reactant_real],
        },
        "curvature_width_max": report["curvature_width_path"]["maximum"],
        "applied_dimension": chosen_dim,
        "applied_dimension_reason": report["applied_dimension_reason"],
        "validity": report["validity"],
        "instanton_anchor": anchor_info,
        "corrections": {
            "1d": {
                "unanchored_factor": a1["unanchored_factor"],
                "anchored_factor": c1["factor"],
                "anchor_log_shift": a1["log_shift"],
                "log_standard_error": c1["log_standard_error"],
                "bead_factor_B_P": bp_1d["factor"],
                "effective_multiplier_C_over_B_P": float(c1["factor"] / bp_1d["factor"]),
            },
            "2d_rph": {
                "unanchored_factor": a2["unanchored_factor"],
                "anchored_factor": c2["factor"],
                "anchor_log_shift": a2["log_shift"],
                "log_standard_error": c2["log_standard_error"],
                "retained_dvr_points": report["solver_plans"]["2"]["retained_points"],
                "bead_factor_B_P": bp_2d["factor"],
                "effective_multiplier_C_over_B_P": float(c2["factor"] / bp_2d["factor"]),
            },
        },
        "nested_sensitivity": report["nested_sensitivity"],
        "applied_rate": applied,
        "wall_seconds": elapsed,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
