"""Restartable analytic-reference active learning, with independent error checks.

Run from an installed checkout::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/ml_lifecycle/closed_loop.py --output runs/closed-loop
    python examples/ml_lifecycle/closed_loop.py --output runs/closed-loop --resume

This deliberately small one-dimensional example uses a quartic reference, three
trainable polynomial models and committee-driven Metropolis sampling.  Every
public energy, force and coordinate is in atomic units.  The diagnostic reference
calculations are additional, analytically cheap held-out checks; they are never
provided to the trainer and are not charged against the training label budget.
Committee disagreement is not a calibrated error estimate, and this example does
not demonstrate molecular rate or physical convergence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from pyrpmd.active_learning import ActiveLearningLoop, SamplingBatch
from pyrpmd.io import write_json
from pyrpmd.potentials.base import PESResult, Potential, coordinates
from pyrpmd.potentials.lifecycle import CommitteePotential


# The three models deliberately have different errors.  Their arithmetic mean is
# not the analytic reference.  These fixed priors also define the ridge penalty.
PRIORS = np.array([[0.8, 0.0, -0.4, 0.0, 0.3],
                   [1.2, 0.0, -1.6, 0.0, 0.7],
                   [0.4, 0.0, -0.7, 0.0, 0.4]], dtype=np.float64)


class DoubleWellReference(Potential):
    """Independent analytic V=.5(x²-1)²+.5(y²+z²), for one atom."""

    identity = {"provider": "analytic_double_well", "schema": 1,
                "barrier_hartree": 0.5, "well_position_bohr": 1.0,
                "transverse_force_constant_au": 1.0, "symbols": ["H"]}
    metadata = {"provider": "analytic_double_well", "batch_independent": True,
                "native_batch": True, "execution_device": "cpu"}

    def evaluate(self, positions):
        q = coordinates(positions)
        if q.shape[1] != 1:
            raise ValueError("the double-well example requires exactly one atom")
        x = q[:, 0, 0]
        energy = 0.5 * (x*x - 1.0)**2 + 0.5 * np.sum(q[:, 0, 1:]**2, axis=1)
        forces = -q.copy()
        forces[:, 0, 0] = -2.0*x*(x*x - 1.0)
        return PESResult(energy, forces)


class PolynomialPotential(Potential):
    """Quartic in x and a fixed harmonic transverse potential, in atomic units."""

    metadata = {"provider": "example_quartic", "batch_independent": True,
                "native_batch": True, "execution_device": "cpu"}

    def __init__(self, coefficients):
        value = np.asarray(coefficients)
        if value.shape != (5,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("quartic coefficients must be five finite real numbers")
        self.coefficients = np.array(value, dtype=np.float64, copy=True)

    def evaluate(self, positions):
        q = coordinates(positions)
        if q.shape[1] != 1:
            raise ValueError("the polynomial example requires exactly one atom")
        x = q[:, 0, 0]
        energy = np.polynomial.polynomial.polyval(x, self.coefficients)
        energy += 0.5 * np.sum(q[:, 0, 1:]**2, axis=1)
        forces = -q.copy()
        derivative = np.arange(1, 5) * self.coefficients[1:]
        forces[:, 0, 0] = -np.polynomial.polynomial.polyval(x, derivative)
        return PESResult(energy, forces)


class PolynomialCodec:
    """No pickle: the loop persists and hashes the numerical coefficients."""

    identity = {"codec": "example_quartic_coefficients", "schema": 1,
                "degree": 4, "transverse_force_constant_au": 1.0}

    @staticmethod
    def encode(model):
        if not isinstance(model, PolynomialPotential):
            raise TypeError("PolynomialCodec requires PolynomialPotential models")
        return {"coefficients": model.coefficients.copy()}

    @staticmethod
    def decode(payload):
        if set(payload) != {"coefficients"}:
            raise ValueError("unexpected polynomial model payload")
        if payload["coefficients"].dtype != np.float64:
            raise ValueError("polynomial checkpoint must retain float64 coefficients")
        return PolynomialPotential(payload["coefficients"])


class PolynomialTrainer:
    """Refit all successful labels using energy and force least squares.

    The three different fixed-prior ridge penalties create model diversity.  The
    unnormalised data loss makes their influence decrease as labels accumulate.
    This is an inexpensive demonstrator, not a calibrated posterior ensemble.
    """

    def __init__(self, ridge=(0.1, 0.3, 0.9)):
        values = np.asarray(ridge)
        if values.shape != (3,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("ridge requires three finite positive strengths")
        self.ridge = np.array(values, dtype=np.float64, copy=True)
        self.identity = {"trainer": "example_energy_force_ridge", "schema": 1,
                         "ridge": self.ridge.tolist(), "priors": PRIORS.tolist(),
                         "degree": 4, "energy_weight": 1.0, "force_weight": 1.0,
                         "normalise_by_frame_count": False}

    def __call__(self, dataset, previous_models):
        if len(previous_models) != 3 or not all(isinstance(m, PolynomialPotential) for m in previous_models):
            raise ValueError("the example trainer requires three polynomial predecessors")
        if not len(dataset) or np.any(dataset.status != 0) or dataset.positions.shape[1] != 1:
            raise ValueError("the example trainer requires successful one-atom labels")
        x = dataset.positions[:, 0, 0]
        energy_features = np.vander(x, 5, increasing=True)
        force_features = np.zeros_like(energy_features)
        force_features[:, 1:] = -np.arange(1, 5) * x[:, None]**np.arange(4)
        features = np.concatenate((energy_features, force_features), axis=0)
        energy = dataset.energy - 0.5 * np.sum(dataset.positions[:, 0, 1:]**2, axis=1)
        targets = np.concatenate((energy, dataset.forces[:, 0, 0]))
        # Augmented least squares avoids squaring the condition number of the
        # design matrix, including the first sparse acquisition round.
        models = []
        for strength, prior in zip(self.ridge, PRIORS):
            augmented = np.concatenate((features, np.sqrt(strength)*np.eye(5)))
            rhs = np.concatenate((targets, np.sqrt(strength)*prior))
            coefficients = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
            models.append(PolynomialPotential(coefficients))
        return tuple(models)


class MetropolisSampler:
    """Vectorised independent walkers; every restartable variable is in state."""

    def __init__(self, walkers=128, steps=64, beta=2.0, proposal=0.35, boundary=1.8):
        for value, name in ((walkers, "walkers"), (steps, "steps")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for value, name in ((beta, "beta"), (proposal, "proposal"), (boundary, "boundary")):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.walkers, self.steps = int(walkers), int(steps)
        self.beta, self.proposal, self.boundary = float(beta), float(proposal), float(boundary)
        self.identity = {"sampler": "example_vectorised_metropolis", "schema": 1,
                         "walkers": self.walkers, "steps_per_segment": self.steps,
                         "beta_inverse_hartree": self.beta,
                         "proposal_std_bohr": self.proposal,
                         "box_half_width_bohr": self.boundary,
                         "degrees_of_freedom": "one atom x; y=z=0"}

    def __call__(self, committee, rng, state):
        if state is None:
            q = np.zeros((self.walkers, 1, 3), dtype=np.float64)
            q[:, 0, 0] = rng.uniform(-self.boundary, self.boundary, self.walkers)
            accepted, attempted = 0, 0
        else:
            q = coordinates(state["walkers"]).copy()
            if q.shape != (self.walkers, 1, 3) or np.any(q[:, 0, 1:] != 0) or np.any(np.abs(q[:, 0, 0]) > self.boundary):
                raise ValueError("saved walkers do not match the configured one-dimensional box")
            accepted, attempted = int(state["accepted"]), int(state["attempted"])
        energy = committee.evaluate(q).energy.copy()
        for _ in range(self.steps):
            proposal = q.copy()
            proposal[:, 0, 0] += rng.normal(scale=self.proposal, size=self.walkers)
            proposed_energy = committee.evaluate(proposal).energy
            log_uniform = np.log(rng.random(self.walkers))
            accept = ((np.abs(proposal[:, 0, 0]) <= self.boundary)
                      & (log_uniform < -self.beta*(proposed_energy-energy)))
            q[accept] = proposal[accept]
            energy[accept] = proposed_energy[accept]
            accepted += int(np.sum(accept))
            attempted += self.walkers
        return SamplingBatch(q, {"walkers": q.copy(), "accepted": accepted, "attempted": attempted})


def initial_models():
    return tuple(PolynomialPotential(row) for row in PRIORS)


def reference_holdout(count=1024, beta=2.0, boundary=1.8):
    """Independent deterministic quantiles of the reference Boltzmann density."""
    grid = np.linspace(-boundary, boundary, 65537)
    weights = np.exp(-beta * 0.5 * (grid*grid - 1.0)**2)
    # Trapezoidal integration includes both finite-box endpoints correctly.
    cdf = np.concatenate(([0.0], np.cumsum(0.5*(weights[1:]+weights[:-1])*np.diff(grid))))
    cdf /= cdf[-1]
    positions = np.zeros((count, 1, 3), dtype=np.float64)
    positions[:, 0, 0] = np.interp((np.arange(count)+0.5)/count, cdf, grid)
    return positions


def error_and_disagreement(committee, positions):
    """Compare an independent reference error with the same-unit diagnostic."""
    truth = DoubleWellReference().evaluate(positions)
    prediction = committee.evaluate_uncertainty(positions)
    force_error = np.linalg.norm(prediction.mean.forces-truth.forces, axis=-1)
    force_rmse = float(np.sqrt(np.mean(force_error**2)))
    sigma = float(np.sqrt(np.mean(prediction.force_std**2)))
    return {"frames": len(positions),
            "energy_rmse_hartree": float(np.sqrt(np.mean((prediction.mean.energy-truth.energy)**2))),
            "energy_disagreement_rms_hartree": float(np.sqrt(np.mean(prediction.energy_std**2))),
            "force_vector_rmse_hartree_per_bohr": force_rmse,
            "force_disagreement_rms_hartree_per_bohr": sigma,
            "max_force_disagreement_hartree_per_bohr": float(np.max(prediction.max_force_std)),
            "force_rmse_over_disagreement": force_rmse/sigma if sigma else None,
            "fraction_force_error_above_one_disagreement": float(np.mean(force_error > prediction.force_std))}


def make_loop(directory, *, resume=False, seed=37):
    trainer, sampler = PolynomialTrainer(), MetropolisSampler()
    return ActiveLearningLoop(
        directory, None if resume else initial_models(),
        reference=DoubleWellReference(), reference_identity=DoubleWellReference.identity,
        sampler=sampler, sampler_identity=sampler.identity,
        trainer=trainer, trainer_identity=trainer.identity, model_codec=PolynomialCodec(),
        force_threshold=0.02, max_rounds=8, max_labels=32, labels_per_round=4,
        label_batch_size=2, seed=seed,
    )


def run_demo(directory, *, resume=False, max_steps=None):
    """Run or resume the pilot, then atomically write a diagnostic JSON report."""
    directory = Path(directory)
    started = time.perf_counter()
    with make_loop(directory, resume=resume) as loop:
        outcome = loop.run(max_steps=max_steps)
        elapsed = time.perf_counter()-started
        holdout = reference_holdout()
        sampler_state = loop.sampler_state
        with CommitteePotential(initial_models(), owns_members=True) as original, loop.committee as current:
            comparison = {"independent_reference_boltzmann_quantiles": {
                "before": error_and_disagreement(original, holdout),
                "after": error_and_disagreement(current, holdout)}}
            if sampler_state is not None:
                walkers = sampler_state["walkers"]
                comparison["final_sampled_configurations"] = {
                    "before": error_and_disagreement(original, walkers),
                    "after": error_and_disagreement(current, walkers)}
    report = {
        "example": "analytic-double-well-active-learning", "schema": 1,
        "learning": outcome,
        "true_error_vs_uncertainty": comparison,
        "uncertainty_is_calibrated_error": False,
        "physical_converged": False,
        "scope": "one-dimensional fixed-box classical Metropolis pilot; y=z=0",
        "diagnostic_labels_enter_training": False,
        "diagnostic_reference_frames": 2*(len(holdout)+(0 if sampler_state is None else len(sampler_state["walkers"]))),
        "timing": {"this_invocation_seconds": elapsed,
                   "platform": platform.platform(), "processor": platform.processor() or "unavailable",
                   "logical_cpus": os.cpu_count(),
                   "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                   "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset")},
    }
    write_json(directory / "diagnostics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/closed-loop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, help="stop after this many committed driver transitions")
    args = parser.parse_args()
    if (args.output / "checkpoint.npz").exists() and not args.resume:
        parser.error("this directory already has a checkpoint; pass --resume or choose a new output")
    print(json.dumps(run_demo(args.output, resume=args.resume, max_steps=args.max_steps), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
