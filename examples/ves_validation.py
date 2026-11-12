"""Independent, reproducible VES checks against quadrature and umbrella/MBAR.

The fast reference uses independent *continuous* rejection samples from the
current biased distribution between coefficient updates.  Those samples pass
through VESBias.observe; no exact moments or fitted reference coefficients are
injected.  This separates stochastic optimizer error from molecular-dynamics
mixing.  ``--md`` additionally runs the public FES workflow on an analytic
double well, freezes the learned bias, and compares its reweighted production
histogram with bin-integrated quadrature and independent umbrella/MBAR.

All model quantities are in atomic units; beta=1 is an illustrative toy-model
temperature, not a claim about molecular convergence.  Independent-replica
standard errors in this validation harness are not production convergence
certificates.  The ordinary workflow reports task-specific diagnostics only.

Example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/ves_validation.py \
        --output ves-validation.json --md --md-directory ves-md
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import t as student_t

from pyrpmd.analysis.fes import reconstruct_fes
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.ves import VESBias


BETA = 1.0
RANGE = (-1.5, 1.5)


def model_energy(points):
    """Tilted quartic in 1D; genuinely coupled, positive quadratic in 2D."""
    points = np.asarray(points, dtype=float)
    x = points[..., 0]
    if points.shape[-1] == 1:
        return (x*x - 1.0)**2 + 0.2*x
    y = points[..., 1]
    return 0.5*(x*x + 1.4*y*y) + 0.35*x*y + 0.2*x - 0.1*y


def _model_gradient_bound(dimension):
    # Extrema of the 1D cubic derivative occur at endpoints or +/-1/sqrt(3).
    if dimension == 1:
        x = np.array([-1.5, 1.5, -1/np.sqrt(3), 1/np.sqrt(3)])
        return np.array([np.max(np.abs(4*x*(x*x - 1) + 0.2))])
    return 1.5*np.sum(np.abs([[1., .35], [.35, 1.4]]), axis=1) + [.2, .1]


def _uniform_rejection(rng, energy, lower_bound, dimension, count):
    """IID continuous samples using a mathematically valid energy lower bound."""
    retained = []
    remaining, proposed = count, 0
    while remaining:
        trial = rng.uniform(*RANGE, size=(max(256, 4*remaining), dimension))
        log_acceptance = -BETA*(energy(trial) - lower_bound)
        if np.max(log_acceptance) > 1e-11:
            raise AssertionError("rejection envelope is not a lower energy bound")
        accepted = trial[np.log(rng.random(len(trial))) < log_acceptance][:remaining]
        retained.append(accepted)
        remaining -= len(accepted)
        proposed += len(trial)
    return np.concatenate(retained), proposed


def _biased_draws(rng, bias, orders, count):
    dimension = len(orders)
    # |P'_n(z)| <= n(n+1)/2 for z in [-1,1].  A rectangular grid plus
    # this Lipschitz envelope supplies a rigorous lower energy bound even
    # while the bias is adapting.  No discretized-grid samples are used.
    grid_count = 33 if dimension == 1 else 17
    axes = [np.linspace(*RANGE, grid_count)]*dimension
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
    energy = lambda values: model_energy(values) + bias.physical_energy(values)
    order = np.asarray(orders)
    derivative = (_model_gradient_bound(dimension)
                  + np.sum(np.abs(bias.coefficients))*order*(order + 1)/(RANGE[1] - RANGE[0]))
    half_spacing = (RANGE[1] - RANGE[0])/(2*(grid_count - 1))
    lower = float(np.min(energy(grid)) - half_spacing*np.sum(derivative))
    return _uniform_rejection(rng, energy, lower, dimension, count)


def train_reference(dimension, target, *, seed, iterations=300, block_size=64):
    """Learn a bias only from stochastic, unbiased estimates of its moments."""
    orders = [4] if dimension == 1 else [2, 2]
    cv = CVStack([Cartesian(0, axis) for axis in range(dimension)])
    bias = VESBias(cv, BETA, [RANGE]*dimension, orders,
                   target=target, bias_factor=4., learning_rate=3.,
                   pace=block_size, target_update_interval=10,
                   quadrature_order=32, wall_stiffness=20.)
    rng = np.random.default_rng(seed)
    proposed = 0
    for _ in range(iterations):
        samples, count = _biased_draws(rng, bias, orders, block_size)
        proposed += count
        for value in samples:
            bias.observe(value)
    return bias, proposed


def bin_free_energy(energy, edges, *, order=8):
    """Gauss--Legendre integration of density in each bin, including volume."""
    nodes, weights = leggauss(order)
    centers = np.stack(np.meshgrid(*[(e[:-1] + e[1:])/2 for e in edges], indexing="ij"), axis=-1)
    half_width = np.stack(np.meshgrid(*[np.diff(e)/2 for e in edges], indexing="ij"), axis=-1)
    contributions = []
    for index in product(range(order), repeat=len(edges)):
        values = centers + half_width*nodes[list(index)]
        weight = np.prod(weights[list(index)]/2)
        contributions.append(-BETA*energy(values) + np.log(weight))
    return -logsumexp(np.asarray(contributions), axis=0)/BETA


def umbrella_reference(dimension, edges, *, seed=11903, samples_per_window=6000, bootstrap=24):
    """Independent exact equilibrium umbrella samples and existing MBAR code."""
    centers = (np.array([[-.9], [0.], [.9]]) if dimension == 1 else
               np.array(list(product([-.8, .8], repeat=2))))
    stiffness = 2.5
    rng = np.random.default_rng(seed)
    # The quartic is nonnegative and .2*x >= -.3.  The quadratic form in
    # 2D is positive and .2*x-.1*y >= -.45; the umbrellas are nonnegative.
    lower_bound = -.3 if dimension == 1 else -.45
    samples = []
    for center in centers:
        def energy(values):
            return model_energy(values) + .5*stiffness*np.sum((values - center)**2, axis=-1)
        values, _ = _uniform_rejection(rng, energy, lower_bound, dimension, samples_per_window)
        samples.append(values)
    return reconstruct_fes(samples, centers, stiffness*np.eye(dimension), BETA,
                           bins=edges, ranges=[RANGE]*dimension, min_count=10,
                           bootstrap=bootstrap, seed=seed + 1)


def reference_case(dimension, target, *, iterations=300, replicas=6, block_size=64,
                   samples_per_window=6000, bootstrap=24):
    """Return complete uncertainty-aware stochastic optimizer/MBAR evidence."""
    started = time.perf_counter()
    bins = 20 if dimension == 1 else 10
    edges = [np.linspace(*RANGE, bins + 1) for _ in range(dimension)]
    exact = bin_free_energy(model_energy, edges, order=8)
    check = bin_free_energy(model_energy, edges, order=5)
    integration_error = float(np.max(np.abs(exact - check)))
    umbrella = umbrella_reference(dimension, edges, samples_per_window=samples_per_window,
                                  bootstrap=bootstrap)
    reference = tuple(umbrella["bootstrap_reference_bin"])
    exact -= exact[reference]
    surfaces, states, proposals = [], [], []
    for replica in range(replicas):
        bias, proposed = train_reference(dimension, target,
                                         seed=7321 + 101*dimension + 7919*replica,
                                         iterations=iterations, block_size=block_size)
        free = bin_free_energy(bias.free_energy, edges)
        surfaces.append(free - free[reference])
        states.append({"coefficients": bias.coefficients.tolist(),
                       "observations": bias.observations, "updates": bias.updates})
        proposals.append(proposed)
    surfaces = np.asarray(surfaces)
    mean, sem = surfaces.mean(axis=0), surfaces.std(axis=0, ddof=1)/np.sqrt(replicas)
    mbar = umbrella["free_energy"] - umbrella["free_energy"][reference]
    mbar_sem = umbrella["standard_error"]
    supported = ~umbrella["mask"] & np.isfinite(mbar_sem)
    supported[reference] = False
    difference = mean - exact
    combined_sem = np.sqrt(sem**2 + mbar_sem**2)
    simultaneous_level = .99
    band = float(student_t.ppf(1 - (1 - simultaneous_level)/(2*np.sum(supported)), replicas - 1))
    return {"dimension": dimension, "target": target, "beta": BETA,
            "model": "tilted_quartic" if dimension == 1 else "coupled_quadratic",
            "iterations": iterations, "block_size": block_size, "replicas": replicas,
            "samples_per_replica": iterations*block_size,
            "umbrella_samples_per_window": samples_per_window,
            "bootstrap_replicates": bootstrap, "reference_bin": list(reference),
            "edges": [edge.tolist() for edge in edges],
            "ranges_au": [list(RANGE)]*dimension,
            "quadrature_order_8_vs_5_max_hartree": integration_error,
            "ves_quadrature_rmse_hartree": float(np.sqrt(np.mean(difference[supported]**2))),
            "mbar_quadrature_rmse_hartree": float(np.sqrt(np.mean((mbar[supported] - exact[supported])**2))),
            "ves_mbar_rmse_hartree": float(np.sqrt(np.mean((mean[supported] - mbar[supported])**2))),
            "ves_quadrature_max_sem": float(np.max(np.abs(difference[supported])/sem[supported])),
            "mbar_quadrature_max_sem": float(np.max(np.abs(mbar[supported] - exact[supported])/mbar_sem[supported])),
            "ves_mbar_max_combined_sem": float(np.max(np.abs(mean[supported] - mbar[supported])/combined_sem[supported])),
            "simultaneous_confidence_level": simultaneous_level,
            "bonferroni_student_t_multiplier": band,
            "zero_bias_rmse_hartree": float(np.sqrt(np.mean(exact[supported]**2))),
            "learning_rmse_ratio": float(np.sqrt(np.mean(difference[supported]**2)/np.mean(exact[supported]**2))),
            "supported_bins": int(np.sum(supported)), "connected": bool(umbrella["connected"]),
            "elapsed_seconds": time.perf_counter() - started,
            "reference_free_energy": exact.tolist(), "ves_mean": mean.tolist(), "ves_sem": sem.tolist(),
            "mbar_free_energy": mbar.tolist(), "mbar_sem": mbar_sem.tolist(),
            "support": supported.tolist(), "states": states, "rejection_proposals": proposals,
            
            "scope": "IID continuous sampling of each current VES bias; no MD mixing claim"}


class _ValidationPotential:
    """Same analytic models with harmonic confinement of unobserved axes."""

    metadata = {"provider": "ves_validation", "thread_safety": "reentrant",
                "execution_device": "cpu"}

    def __init__(self, dimension=1):
        self.dimension = int(dimension)
        if self.dimension not in (1, 2):
            raise ValueError("validation potential dimension must be one or two")

    def evaluate(self, q):
        from pyrpmd.potentials.base import PESResult
        q = np.asarray(q)
        if (q.ndim != 3 or q.shape[1:] != (1, 3) or len(q) == 0
                or q.dtype.kind not in "fiu" or not np.isfinite(q).all()):
            raise ValueError("validation potential requires finite [B,1,3] coordinates for one atom")
        q = np.asarray(q, dtype=float)
        points = q[:, 0, :self.dimension]
        energy = model_energy(points) + .5*np.sum(q[:, 0, self.dimension:]**2, axis=1)
        forces = -np.array(q, copy=True)
        x = points[:, 0]
        if self.dimension == 1:
            forces[:, 0, 0] = -4*x*(x*x - 1) - .2
        else:
            y = points[:, 1]
            forces[:, 0, 0] = -x - .35*y - .2
            forces[:, 0, 1] = -1.4*y - .35*x + .1
        return PESResult(energy, forces)


def build_validation_potential(**params):
    """Python PES factory used by the public input/runner path."""
    return _ValidationPotential(**params)


def md_job(dimension, target, *, replicas=6, production_steps=30000, timestep=.04, workers=2):
    """An explicitly reproducible P=1 job; freeze after one third of production."""
    from pyrpmd.constants import KB_AU
    pace = 100
    freeze = (production_steps//3//pace)*pace
    if freeze < pace or production_steps <= freeze + 1000:
        raise ValueError("MD validation requires more than 1500 production steps")
    return {
        "task": "fes", "temperature": 1/(BETA*KB_AU), "beads": 1,
        "seed": 61709 + dimension, "native": False,
        "system": {"symbols": ["H"], "masses": [1.],
                   "geometry": [[-.9, 0., 0.]], "geometry_unit": "bohr"},
        "potential": {"provider": "python",
                      "factory": str(Path(__file__).resolve()) + ":build_validation_potential",
                      "params": {"dimension": dimension}},
        "dynamics": {"timestep": timestep, "loop_backend": "python"},
        "output": {"checkpoint_steps": 5000},
        "parallel": {"backend": "process" if workers > 1 else "serial",
                     "trajectory_workers": workers, "threads_per_rank": 1},
        "fes": {
            "method": "ves",
            "cvs": [{"name": "xyz"[axis], "kind": "cartesian", "atom": 0,
                     "axis": axis, "estimator": "centroid"} for axis in range(dimension)],
            "ves": {"ranges_au": [list(RANGE)]*dimension,
                    "orders": [4] if dimension == 1 else [2, 2],
                    "target": target, "bias_factor": 4., "learning_rate_au": 1.,
                    "pace": pace, "target_update_interval": 10, "quadrature_order": 32,
                    "wall_stiffness_au": [20.]*dimension,
                    "freeze_after_steps": freeze, "freeze_equilibration_steps": 1000},
            "sampling": {"thermostat": "pile-l", "gamma_centroid": 1.,
                         "equilibration_steps": 1000, "production_steps": production_steps,
                         "sample_interval": 5, "replicas": replicas},
            "analysis": {"method": "reweighted_histogram", "bins": 12 if dimension == 1 else 8,
                         "min_count": 10, "bootstrap": 0},
        },
    }


def md_validation(directory, *, replicas=6, production_steps=30000, timestep=.04, workers=2):
    """Actual workflow MD checks on both dimensions with frozen reweighting.

    This is intentionally separate from the IID optimizer test.  Replica SEM
    is computed here from separately normalized histories; the FES artifact's
    own uncertainty remains its documented descriptive replica spread.
    """
    from pyrpmd.analysis.fes_result import FESResult
    from pyrpmd.runner import run, analyze_run, resume_run
    results = []
    for dimension, target in ((1, "uniform"), (2, "well_tempered")):
        started = time.perf_counter()
        job = md_job(dimension, target, replicas=replicas,
                     production_steps=production_steps, timestep=timestep, workers=workers)
        output = Path(directory)/f"{dimension}d-{target}"
        summary = run(job, output=output, resume=output.exists())
        sampled_seconds = time.perf_counter() - started
        result = FESResult.read(output)
        reference = tuple(summary["reference_bin"])
        exact = bin_free_energy(model_energy, result.edges)
        exact -= exact[reference]
        histories = result.extras["free_energy_per_replica"]
        mean = histories.mean(axis=0)
        sem = histories.std(axis=0, ddof=1)/np.sqrt(replicas)
        umbrella = umbrella_reference(dimension, result.edges)
        mbar = umbrella["free_energy"]
        # MBAR's bootstrap errors belong to its own fixed reference bin;
        # rebase VES and exact curves to that same reference for comparison.
        mbar_reference = tuple(umbrella["bootstrap_reference_bin"])
        histories = histories - histories[(slice(None),) + mbar_reference].reshape((-1,) + (1,)*dimension)
        mean = histories.mean(axis=0)
        sem = histories.std(axis=0, ddof=1)/np.sqrt(replicas)
        exact -= exact[mbar_reference]
        mbar = mbar - mbar[mbar_reference]
        mbar_sem = umbrella["standard_error"]
        support = ~result.mask & ~umbrella["mask"] & np.isfinite(mbar_sem) & np.isfinite(sem)
        support[mbar_reference] = False
        combined = np.sqrt(sem**2 + mbar_sem**2)
        band = float(student_t.ppf(1 - .01/(2*np.sum(support)), replicas - 1))
        # Explicit saved-state recomputation and finished-run resume must
        # reproduce the result.
        reanalyzed = analyze_run(output, recompute=True)
        resumed = resume_run(output)
        reread = FESResult.read(output)
        np.testing.assert_array_equal(result.free_energy, reread.free_energy)
        metrics = {
            "dimension": dimension, "target": target, "beta": BETA,
            "replicas": replicas, "production_steps": production_steps, "timestep_au": timestep,
            "trajectory_workers": workers, "threads_per_rank": 1,
            "edges": [edge.tolist() for edge in result.edges],
            "ranges_au": [[float(edge[0]), float(edge[-1])] for edge in result.edges],
            "reference_bin": list(mbar_reference),
            "freeze_after_steps": job["fes"]["ves"]["freeze_after_steps"],
            "retained_samples_per_replica": summary["retained_samples_per_replica"],
            "outside_observations_per_replica": summary["outside_observations_per_replica"],
            "updates_per_replica": summary["updates_per_replica"],
            "supported_bins": int(np.sum(support)),
            "ves_quadrature_rmse_hartree": float(np.sqrt(np.mean((mean[support] - exact[support])**2))),
            "ves_quadrature_max_sem": float(np.max(np.abs(mean[support] - exact[support])/sem[support])),
            "ves_mbar_rmse_hartree": float(np.sqrt(np.mean((mean[support] - mbar[support])**2))),
            "ves_mbar_max_combined_sem": float(np.max(np.abs(mean[support] - mbar[support])/combined[support])),
            "simultaneous_confidence_level": .99,
            "bonferroni_student_t_multiplier": band,
            "elapsed_sampling_seconds": sampled_seconds,
            "elapsed_including_analysis_seconds": time.perf_counter() - started,
            "steps_per_second": replicas*(production_steps + 1000)/sampled_seconds,
            "reference_free_energy": exact.tolist(), "ves_mean": mean.tolist(), "ves_sem": sem.tolist(),
            "mbar_free_energy": mbar.tolist(), "mbar_sem": mbar_sem.tolist(),
            "support": support.tolist(), 
            "finished_resume_and_reanalysis_exact": True,
            "scope": "P=1 MD on toy models; no molecular, bead, basis or timestep convergence claim",
        }
        results.append(metrics)
        print(json.dumps({key: metrics[key] for key in ("dimension", "target", "elapsed_sampling_seconds",
              "ves_quadrature_rmse_hartree", "ves_quadrature_max_sem", "ves_mbar_max_combined_sem")}), flush=True)
    return results


def _metadata():
    import scipy
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "processor": cpu, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ves-validation.json"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--replicas", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--md", action="store_true")
    parser.add_argument("--md-only", action="store_true")
    parser.add_argument("--md-directory", type=Path, default=Path("ves-md"))
    parser.add_argument("--md-steps", type=int, default=30000)
    parser.add_argument("--md-replicas", type=int, default=6)
    parser.add_argument("--md-workers", type=int, default=2)
    args = parser.parse_args()
    report = {"metadata": _metadata(), "reference_cases": []}
    for dimension, target in (() if args.md_only else product((1, 2), ("uniform", "well_tempered"))):
        result = reference_case(dimension, target, iterations=args.iterations,
                                replicas=args.replicas, block_size=args.block_size)
        report["reference_cases"].append(result)
        print(json.dumps({key: result[key] for key in ("dimension", "target", "elapsed_seconds",
              "ves_quadrature_rmse_hartree", "ves_quadrature_max_sem", "ves_mbar_max_combined_sem")}), flush=True)
    if args.md or args.md_only:
        report["molecular_dynamics"] = md_validation(args.md_directory, replicas=args.md_replicas,
                                                      production_steps=args.md_steps, workers=args.md_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from pyrpmd.io import write_report_json
    write_report_json(args.output, report)


if __name__ == "__main__":
    main()
