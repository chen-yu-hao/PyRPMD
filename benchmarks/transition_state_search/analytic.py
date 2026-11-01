"""Reproduce analytic saddle references and exercise P-RFO / CI-NEB.

Run from the repository root with ``PYTHONPATH=src python
benchmarks/transition_state_search/analytic.py --output analytic.json``.
No molecular rate is predicted by these dimensionless model surfaces.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import root

from pyrpmd.potentials import PESResult
from pyrpmd.workflows.neb import climbing_image_neb
from pyrpmd.workflows.saddle import refine_transition_state


SOURCES = {
    "banerjee1985": {
        "citation": "Banerjee, Adams, Simons and Shepard, J. Phys. Chem. 89, 52-57 (1985)",
        "doi": "10.1021/j100247a015",
        "url": "https://simons.hec.utah.edu/papers/108.pdf",
        "location": "Applications, Eq. (21), pp. 56-57",
    },
    "doye_wales2002": {
        "citation": "Doye and Wales, J. Chem. Phys. 116, 3777-3788 (2002)",
        "doi": "10.1063/1.1436470",
        "url": "https://arxiv.org/pdf/cond-mat/0108310",
        "location": "Section II, Eqs. (1)-(2); Table I on PDF page 3 (zero-based page 2)",
        "surface_origin_doi": "10.1007/BF00547608",
    },
    "henkelman2000": {
        "citation": "Henkelman, Uberuaga and Jonsson, J. Chem. Phys. 113, 9901-9904 (2000)",
        "doi": "10.1063/1.1329672",
        "url": "https://henkelmanlab.org/pubs/henkelman00_9901.pdf",
        "use": "CI-NEB method reference; no molecular result from this paper is reproduced here",
    },
    "mauro2005": {
        "citation": "Mauro, Loucks and Balakrishnan, J. Phys. Chem. A 109, 9578-9583 (2005)",
        "doi": "10.1021/jp053581t",
        "url": "https://pubs.acs.org/doi/10.1021/jp053581t",
        "location": "Eq. (37) confirms the explicit Adams formula",
        "use": "Formula only; retrieved saddle-coordinate text is not used as gold data",
    },
}


class Surface:
    """Dimensionless 2D surface embedded in [1,3] with a stable z direction.

    ``scale`` multiplies *all* energies and derivatives. Coordinates are passed
    numerically as bohr to the molecular API; energies are scale*V hartree.
    Comparisons to papers divide energies by scale and use the original x,y.
    """

    def __init__(self, name, scale, z_curvature=100.0):
        self.name, self.scale, self.z_curvature = name, scale, z_curvature
        self.evaluations = 0

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        energy, forces = [], []
        for position in q[:, 0]:
            value, gradient, _ = self.raw(position[:2])
            z = position[2]
            energy.append(self.scale * (value + 0.5 * self.z_curvature * z * z))
            forces.append(-self.scale * np.r_[gradient, self.z_curvature * z])
        self.evaluations += len(q)
        return PESResult(np.asarray(energy), np.asarray(forces)[:, None, :])

    def exact(self, geometry):
        x, y, z = np.asarray(geometry).reshape(3)
        energy, gradient, hessian = self.raw([x, y])
        full_hessian = np.zeros((3, 3))
        full_hessian[:2, :2] = hessian
        full_hessian[2, 2] = self.z_curvature
        return (
            self.scale * (energy + 0.5 * self.z_curvature * z * z),
            self.scale * np.r_[gradient, self.z_curvature * z],
            self.scale * full_hessian,
        )


class CerjanMiller(Surface):
    """Banerjee1985 Eq.21: a=b=c=1; saddles (+/-1,0)."""

    def __init__(self):
        super().__init__("cerjan_miller", 1.0, z_curvature=10.0)

    def raw(self, xy):
        x, y = np.asarray(xy)
        exponential = np.exp(-x * x)
        p = x * x * exponential
        dp = 2 * x * (1 - x * x) * exponential
        ddp = (2 - 10 * x * x + 4 * x**4) * exponential
        energy = (1 - y * y) * p + 0.5 * y * y
        gradient = np.array([(1 - y * y) * dp, y * (1 - 2 * p)])
        hessian = np.array([
            [(1 - y * y) * ddp, -2 * y * dp],
            [-2 * y * dp, 1 - 2 * p],
        ])
        return float(energy), gradient, hessian


class MullerBrown(Surface):
    """Doye-Wales2002 Eqs.1-2; original four-Gaussian Muller-Brown PES."""

    amplitude = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])

    def __init__(self):
        super().__init__("muller_brown", 0.001)

    def raw(self, xy):
        x, y = np.asarray(xy)
        dx, dy = x - self.x0, y - self.y0
        term = self.amplitude * np.exp(self.a * dx**2 + self.b * dx * dy + self.c * dy**2)
        gx, gy = 2 * self.a * dx + self.b * dy, self.b * dx + 2 * self.c * dy
        gradient = np.array([np.sum(term * gx), np.sum(term * gy)])
        hessian = np.array([
            [np.sum(term * (gx * gx + 2 * self.a)), np.sum(term * (gx * gy + self.b))],
            [np.sum(term * (gx * gy + self.b)), np.sum(term * (gy * gy + 2 * self.c))],
        ])
        return float(term.sum()), gradient, hessian


class Adams(Surface):
    """Adams formula in Banerjee1985 p.57; use independent roots only.

    Retrieved paper saddle coordinates conflict with this displayed formula.
    No purported paper-coordinate agreement is used as an acceptance test.
    """

    def __init__(self):
        super().__init__("adams", 0.01)

    def raw(self, xy):
        x, y = np.asarray(xy)
        t = 17 * np.exp(-(x * x + y * y) / 4)
        energy = 2 * x * x * (4 - x) + y * y * (4 + y) - x * y * (6 - t)
        gradient = np.array([
            16 * x - 6 * x * x - 6 * y + t * y * (1 - x * x / 2),
            8 * y + 3 * y * y - 6 * x + t * x * (1 - y * y / 2),
        ])
        cross = -6 + t * (1 - x * x / 2) * (1 - y * y / 2)
        hessian = np.array([
            [16 - 12 * x + t * y * (x**3 / 4 - 1.5 * x), cross],
            [cross, 8 + 6 * y + t * x * (y**3 / 4 - 1.5 * y)],
        ])
        return float(energy), gradient, hessian


SURFACES = {"cerjan_miller": CerjanMiller, "muller_brown": MullerBrown, "adams": Adams}

# The MB numbers are the actual rounded entries of Doye-Wales2002 Table I.
# Higher-precision values below are computed locally, not attributed to papers.
PAPER_POINTS = {
    "cerjan_miller": [
        {"id": "minimum", "xy": [0.0, 0.0], "index": 0, "source": "banerjee1985"},
        {"id": "left_saddle", "xy": [-1.0, 0.0], "index": 1, "source": "banerjee1985"},
        {"id": "right_saddle", "xy": [1.0, 0.0], "index": 1, "source": "banerjee1985"},
    ],
    "muller_brown": [
        {"id": "SP1", "xy": [-0.558, 1.442], "energy": -146.700, "index": 0},
        {"id": "SP2", "xy": [0.623, 0.028], "energy": -108.167, "index": 0},
        {"id": "SP3", "xy": [-0.050, 0.467], "energy": -80.768, "index": 0},
        {"id": "SP4", "xy": [0.212, 0.293], "energy": -72.249, "index": 1},
        {"id": "SP5", "xy": [-0.822, 0.624], "energy": -40.665, "index": 1},
    ],
}
for _point in PAPER_POINTS["muller_brown"]:
    _point.update(source="doye_wales2002", printed_decimal_places=3)


def independent_references():
    """Solve grad(V)=0 with an analytic Jacobian, independently of P-RFO/NEB."""
    records = []
    seeds = {
        **PAPER_POINTS,
        "adams": [
            {"id": "minimum", "xy": [0, 0], "index": 0},
            {"id": "right_saddle", "xy": [2.2, 0.4], "index": 1},
            {"id": "lower_saddle", "xy": [-0.2, -2.3], "index": 1},
            {"id": "maximum", "xy": [3.8, -4.4], "index": 2},
        ],
    }
    for name, points in seeds.items():
        surface = SURFACES[name]()
        for point in points:
            solved = root(
                lambda xy: surface.raw(xy)[1], point["xy"],
                jac=lambda xy: surface.raw(xy)[2], method="hybr", options={"xtol": 1e-11},
            )
            energy, gradient, hessian = surface.raw(solved.x)
            eigenvalues = np.linalg.eigvalsh(hessian)
            index = int(np.count_nonzero(eigenvalues < -1e-8))
            if np.max(np.abs(gradient)) > 1e-9 or index != point["index"]:
                raise RuntimeError(f"Independent reference did not converge: {name}/{point['id']}")
            record = {
                "surface": name, "id": point["id"], "xy": solved.x.tolist(),
                "energy_unscaled": energy, "index": index,
                "gradient_max_unscaled": float(np.max(np.abs(gradient))),
                "hessian_eigenvalues_unscaled": eigenvalues.tolist(),
                "scipy_success": bool(solved.success), "scipy_message": str(solved.message),
            }
            if "source" in point:
                error = float(np.max(np.abs(solved.x - point["xy"])))
                record["paper"] = {**point, "max_coordinate_difference": error}
                if "energy" in point:
                    record["paper"]["absolute_energy_difference"] = abs(energy - point["energy"])
                    record["paper"]["matches_printed_precision"] = bool(
                        error <= 0.000501 and abs(energy - point["energy"]) <= 0.000501
                    )
                else:
                    record["paper"]["matches_printed_precision"] = error < 1e-10
            records.append(record)
    return records


def summarize_geometry(surface, geometry, reference):
    energy, gradient, hessian = surface.exact(geometry)
    eigenvalues = np.linalg.eigvalsh(hessian)
    xy = np.asarray(geometry)[0, :2]
    return {
        "geometry": np.asarray(geometry).tolist(), "energy_unscaled": energy / surface.scale,
        "reference_coordinate_error": float(np.max(np.abs(xy - reference["xy"]))),
        "reference_energy_error_unscaled": abs(energy / surface.scale - reference["energy_unscaled"]),
        "exact_max_force": float(np.max(np.abs(gradient))),
        "exact_hessian_eigenvalues": eigenvalues.tolist(),
        "exact_index": int(np.count_nonzero(eigenvalues < -1e-8)),
    }


def refinement_case(surface, reference, start, tolerance, *, hessian_step=1e-4):
    before = surface.evaluations
    result = refine_transition_state(
        surface, np.asarray(start, dtype=float).reshape(1, 3),
        gradient_tolerance=tolerance, hessian_step=hessian_step,
    )
    measured = summarize_geometry(surface, result["geometry"], reference)
    return {
        "surface": surface.name, "target": reference["id"], "start": np.asarray(start).tolist(),
        "tolerance": tolerance, "hessian_step": hessian_step, "max_steps": 100,
        "max_step": 0.2, "negative_eigenvalue_tolerance": 1e-8,
        "converged": result["converged"], "iterations": result["iterations"],
        "pes_evaluations": surface.evaluations - before,
        "diagnostics": result["diagnostics"], "history": result["history"], **measured,
        "target_reached": bool(result["converged"] and measured["reference_coordinate_error"] < 0.01),
    }


def neb_case(surface, start, end, reference, images, tolerance, *, bowed=False):
    left = np.array([[*start["xy"], 0.0]])
    right = np.array([[*end["xy"], 0.0]])
    initial = None
    if bowed:
        initial = np.linspace(left, right, images)
        arc = np.sin(np.linspace(0, np.pi, images))
        initial[:, 0, 1] += 0.25 * arc
        initial[:, 0, 2] += 0.1 * arc
    before = surface.evaluations
    evaluations = []

    class RecordingProvider:
        def evaluate(self, q):
            values = surface.evaluate(q)
            evaluations.append((q[0].copy(), float(values.energy[0]), values.forces[0].copy()))
            return values

    record = {
        "surface": surface.name, "start": start["id"], "end": end["id"],
        "target": reference["id"], "images": images, "tolerance": tolerance,
        "initial_path": "bowed" if bowed else "linear", "max_steps": 1000,
        "spring_constant": 0.1, "max_step": 0.1, "climb_after": 20,
    }
    try:
        result = climbing_image_neb(
            RecordingProvider(), left, right, images=images, gradient_tolerance=tolerance,
            initial_path=initial,
        )
        record.update(
            band_converged=result["converged"], iterations=result["iterations"],
            max_band_force=result["max_force"], max_climbing_force=result["max_climbing_force"],
            diagnostics=result["diagnostics"], path=result["images"].tolist(),
            energies_unscaled=(result["energies"] / surface.scale).tolist(),
            band_result=summarize_geometry(surface, result["geometry"], reference),
        )
        if not result["converged"]:
            # NEB evaluates two fixed endpoints, followed by each interior
            # image of each band. Two final curvature-probe evaluations may
            # follow the last band. Copying the provider calls preserves the
            # actual evaluated trajectory without changing the optimizer.
            frames = result["iterations"] + 1
            band_calls = 2 + frames * (images - 2)
            if len(evaluations) not in (band_calls, band_calls + 2):
                raise RuntimeError("Unexpected NEB evaluation order: cannot reconstruct failure trajectory")
            interior = evaluations[2:band_calls]
            positions = np.empty((frames, images, 1, 3))
            energies = np.empty((frames, images))
            forces = np.empty_like(positions)
            for endpoint, entry in ((0, evaluations[0]), (-1, evaluations[1])):
                positions[:, endpoint], energies[:, endpoint], forces[:, endpoint] = entry
            positions[:, 1:-1] = np.asarray([entry[0] for entry in interior]).reshape(frames, images - 2, 1, 3)
            energies[:, 1:-1] = np.asarray([entry[1] for entry in interior]).reshape(frames, images - 2)
            forces[:, 1:-1] = np.asarray([entry[2] for entry in interior]).reshape(frames, images - 2, 1, 3)
            if not np.array_equal(positions[-1], result["images"]):
                raise RuntimeError("Failure trajectory does not end at the returned NEB band")
            record["_failure_trajectory"] = {
                "positions": positions, "energies_unscaled": energies / surface.scale,
                "physical_forces": forces,
            }
        if result["converged"]:
            refined = refine_transition_state(
                surface, result["geometry"], initial_mode=result["tangent"],
                gradient_tolerance=1e-9,
            )
            record["refinement"] = {
                "converged": refined["converged"], "iterations": refined["iterations"],
                **summarize_geometry(surface, refined["geometry"], reference),
            }
            barrier = refined["energy"] / surface.scale - start["energy_unscaled"]
            reference_barrier = reference["energy_unscaled"] - start["energy_unscaled"]
            record["refinement"].update(
                forward_barrier_unscaled=barrier,
                independent_forward_barrier_unscaled=reference_barrier,
                barrier_error_unscaled=abs(barrier - reference_barrier),
            )
            if "energy" in reference.get("paper", {}) and "energy" in start.get("paper", {}):
                published_difference = reference["paper"]["energy"] - start["paper"]["energy"]
                record["refinement"]["rounded_paper_energy_difference"] = published_difference
                record["refinement"]["paper_difference_note"] = (
                    "Derived from rounded Table I energies; not a separately published barrier"
                )
        record["pipeline_success"] = bool(
            record.get("refinement", {}).get("converged")
            and record["refinement"]["reference_coordinate_error"] < 1e-6
            and record["refinement"]["exact_index"] == 1
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        record.update(band_converged=False, pipeline_success=False, exception=str(error))
    record["pes_evaluations"] = surface.evaluations - before
    return record


def ase_crosscheck(surface, start, end, reference, *, images_count=9):
    """Independent optional ASE implementation using the same numeric PES."""
    try:
        import ase
        from ase import Atoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.mep import NEB
        from ase.optimize import FIRE
    except ImportError:
        return {"available": False, "reason": "ASE is not installed"}

    class AnalyticCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            result = surface.evaluate(atoms.positions[None])
            self.results = {"energy": result.energy[0], "forces": result.forces[0]}

    images = []
    for xy in np.linspace(start["xy"], end["xy"], images_count):
        atoms = Atoms("H", positions=[[*xy, 0.0]])
        atoms.calc = AnalyticCalculator()
        images.append(atoms)
    band = NEB(images, k=0.1, climb=False, method="improvedtangent")
    FIRE(band, logfile=None, dt=0.1, maxstep=0.1).run(fmax=1e-6, steps=20)
    band.climb = True
    optimizer = FIRE(band, logfile=None, dt=0.1, maxstep=0.1)
    converged = optimizer.run(fmax=1e-6, steps=1000)
    energies = [atoms.get_potential_energy() for atoms in images]
    highest = 1 + int(np.argmax(energies[1:-1]))
    return {
        "available": True, "ase_version": ase.__version__, "method": "improvedtangent/FIRE/climb",
        "start": start["id"], "end": end["id"], "target": reference["id"],
        "images": images_count, "converged": bool(converged), "climbing_iterations": optimizer.nsteps,
        "gradient_tolerance": 1e-6, "max_climbing_steps": 1000, "warmup_steps": 20,
        "spring_constant": 0.1, "initial_time_step": 0.1, "max_step": 0.1,
        "max_band_force_norm": float(np.max(np.linalg.norm(band.get_forces(), axis=1))),
        "unit_note": "Identical numeric scaled model; ASE energy/length labels are not physical molecular units here",
        **summarize_geometry(surface, images[highest].positions, reference),
    }


def run_benchmark():
    started = time.perf_counter()
    references = independent_references()
    lookup = {(item["surface"], item["id"]): item for item in references}
    local_cases = []
    for reference in references:
        if reference["index"] != 1:
            continue
        for offset in ([0.12, -0.08, 0.05], [-0.10, 0.09, -0.03], [0.25, 0.15, 0.08]):
            for tolerance in (1e-4, 1e-6, 1e-8):
                start = np.r_[reference["xy"], 0.0] + offset
                local_cases.append(refinement_case(SURFACES[reference["surface"]](), reference, start, tolerance))
    hessian_cases = []
    for key in (("cerjan_miller", "right_saddle"), ("muller_brown", "SP5")):
        reference = lookup[key]
        for step in (1e-3, 1e-4, 1e-5):
            hessian_cases.append(refinement_case(
                SURFACES[key[0]](), reference, np.r_[reference["xy"], 0] + [0.12, -0.08, 0.03],
                1e-9, hessian_step=step,
            ))
    stress_cases = []
    for key, start in (
        (("cerjan_miller", "right_saddle"), [0.01, 0.01, 0]),
        (("muller_brown", "SP5"), [-1.169, 0.741, 0]),
        (("muller_brown", "SP5"), [-1.559, 1.543, 0]),
        (("muller_brown", "SP5"), [-0.097, 1.076, 0]),
        (("muller_brown", "SP4"), [-0.995, -0.053, 0]),
        (("adams", "right_saddle"), [0.1, -0.1, 0]),
        (("adams", "lower_saddle"), [-0.1, 0.1, 0]),
    ):
        stress_cases.append(refinement_case(SURFACES[key[0]](), lookup[key], start, 1e-8))
    stationary_rejections = []
    for reference in references:
        if reference["index"] != 1:
            result = refine_transition_state(SURFACES[reference["surface"]](), [[*reference["xy"], 0]])
            stationary_rejections.append({
                "surface": reference["surface"], "id": reference["id"],
                "reference_index": reference["index"], "converged": result["converged"],
                "diagnostics": result["diagnostics"],
            })
    neb_cases = []
    paths = [("SP1", "SP3", "SP5"), ("SP3", "SP2", "SP4"),
             ("SP2", "SP3", "SP4"), ("SP1", "SP2", "SP5")]
    for start_id, end_id, saddle_id in paths:
        start, end, reference = [lookup[("muller_brown", item)] for item in (start_id, end_id, saddle_id)]
        for images in (5, 7, 13, 21):
            for tolerance in (1e-4, 1e-6):
                case = neb_case(MullerBrown(), start, end, reference, images, tolerance)
                neb_cases.append(case)
                print(f"CI-NEB {start_id}->{end_id}, {images} images, {tolerance:g}: {case['pipeline_success']}", flush=True)
        neb_cases.append(neb_case(MullerBrown(), start, end, reference, 13, 1e-6, bowed=True))
    independent_ase = []
    for start_id, end_id, saddle_id in paths[:2]:
        start, end, reference = [lookup[("muller_brown", item)] for item in (start_id, end_id, saddle_id)]
        independent_ase.append(ase_crosscheck(MullerBrown(), start, end, reference))
    start, end, reference = [lookup[("muller_brown", item)] for item in paths[1]]
    for images in (5, 7):
        independent_ase.append(ase_crosscheck(MullerBrown(), start, end, reference, images_count=images))
    for case in stress_cases:
        if not case["converged"]:
            case["attained_stationary_point"] = None
            continue
        candidates = [point for point in references if point["surface"] == case["surface"]]
        attained = min(candidates, key=lambda point: np.linalg.norm(
            np.asarray(case["geometry"])[0, :2] - point["xy"]
        ))
        case["attained_stationary_point"] = attained["id"]
    accepted = [case for case in local_cases + hessian_cases + stress_cases if case["converged"]]
    accepted.extend(
        case["refinement"] for case in neb_cases
        if case.get("refinement", {}).get("converged")
    )
    core_checks = {
        "published_stationary_points_match_printed_precision": all(
            point["paper"]["matches_printed_precision"] for point in references if "paper" in point
        ),
        "all_local_refinements_reach_expected_saddle": all(
            case["target_reached"] and case["exact_max_force"] <= case["tolerance"] * (1 + 1e-6)
            for case in local_cases
        ),
        "all_hessian_step_sweeps_reach_expected_saddle": all(
            case["target_reached"] and case["exact_max_force"] <= case["tolerance"] * (1 + 1e-6)
            for case in hessian_cases
        ),
        "no_accepted_refinement_has_wrong_hessian_index": all(case["exact_index"] == 1 for case in accepted),
        "all_nontransition_stationary_points_rejected": all(not case["converged"] for case in stationary_rejections),
        "all_bands_with_at_least_seven_images_reach_expected_saddle": all(
            case["pipeline_success"] for case in neb_cases if case["images"] >= 7
        ),
    }
    return {
        "schema": "pyrpmd.transition_state_search.analytic.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "code_sha256": {
            path: hashlib.sha256((Path(__file__).resolve().parents[2] / path).read_bytes()).hexdigest()
            for path in (
                "benchmarks/transition_state_search/analytic.py",
                "src/pyrpmd/workflows/prepare.py",
                "src/pyrpmd/workflows/saddle.py",
                "src/pyrpmd/workflows/neb.py",
            )
        },
        "sources": SOURCES,
        "core_checks": core_checks, "core_checks_passed": all(core_checks.values()),
        "scope": {
            "paper_comparison": "Rounded CM coordinates and Muller-Brown coordinates/energies/indices; not paper iteration counts or molecular rates",
            "independent_reference": "scipy.optimize.root(hybr), analytic gradient and Hessian, residual checked below 1e-9 in unscaled units",
            "embedding": "one [x,y,z] atom; V_AU=scale*(V_paper(x,y)+z_curvature*z^2/2), masses=1",
            "scales": {key: {"energy_scale": value().scale, "z_curvature_unscaled": value().z_curvature} for key, value in SURFACES.items()},
            "adams_note": "Displayed formula tested against independent derivatives/roots only; retrieved paper coordinates are inconsistent, and no corrected paper values are inferred",
            "local_limit": "Refinement is local: stress starts may reach another saddle or exhaust; these outcomes are retained",
            "stress_reference_note": "Remote starts have comparison labels, not guaranteed target basins; no endpoint or mode was supplied",
        },
        "references": references, "refinement": local_cases, "hessian_step_sweep": hessian_cases,
        "stress_refinement": stress_cases, "stationary_rejections": stationary_rejections,
        "neb": neb_cases, "independent_ase": independent_ase,
        "summary": {
            "paper_points": sum("paper" in point for point in references),
            "paper_points_match": sum(point.get("paper", {}).get("matches_printed_precision", False) for point in references),
            "local_refinement_cases": len(local_cases), "local_target_success": sum(case["target_reached"] for case in local_cases),
            "hessian_step_cases": len(hessian_cases), "hessian_step_success": sum(case["target_reached"] for case in hessian_cases),
            "stress_cases": len(stress_cases), "stress_target_success": sum(case["target_reached"] for case in stress_cases),
            "stress_converged_to_any_saddle": sum(case["converged"] for case in stress_cases),
            "stress_converged_to_other_saddle": sum(case["converged"] and not case["target_reached"] for case in stress_cases),
            "stress_unconverged": sum(not case["converged"] for case in stress_cases),
            "stationary_rejections": len(stationary_rejections),
            "stationary_false_successes": sum(case["converged"] for case in stationary_rejections),
            "neb_cases": len(neb_cases), "neb_pipeline_success": sum(case["pipeline_success"] for case in neb_cases),
            "accepted_refinement_wrong_indices": sum(case["exact_index"] != 1 for case in accepted),
            "accepted_refinement_max_coordinate_error": max(case["reference_coordinate_error"] for case in local_cases if case["target_reached"]),
            "failure_diagnostics": dict(Counter(case.get("exception", "; ".join(case.get("diagnostics", []))) for case in neb_cases if not case["pipeline_success"])),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectories = {}
    for index, case in enumerate(result["neb"]):
        case["case_id"] = f"neb_{index:03d}"
        trajectory = case.pop("_failure_trajectory", None)
        if trajectory is None:
            continue
        case["failure_trajectory_arrays"] = {}
        for name, array in trajectory.items():
            key = f"{case['case_id']}_{name}"
            trajectories[key] = array
            case["failure_trajectory_arrays"][name] = {"key": key, "shape": list(array.shape)}
    if trajectories:
        archive_path = args.output.with_name(args.output.stem + "-neb-failure-trajectories.npz")
        np.savez_compressed(archive_path, **trajectories)
        result["failure_trajectory_archive"] = {
            "filename": archive_path.name,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "format": "NumPy compressed NPZ; load with allow_pickle=False",
            "axis_order": "iteration, image, atom, Cartesian component (energy arrays: iteration, image)",
            "iteration_note": "Includes the initial band at iteration 0 and every evaluated band through max_steps",
            "units": {"positions": "model coordinates passed numerically as bohr", "energies_unscaled": "original model units", "physical_forces": "scaled hartree/bohr"},
            "endpoint_note": "Endpoint positions, energies and physical forces are fixed and repeated for each iteration",
        }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps({"core_checks_passed": result["core_checks_passed"]}))
    raise SystemExit(0 if result["core_checks_passed"] else 1)


if __name__ == "__main__":
    main()
