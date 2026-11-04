"""T38 statistical comparison on a two-dimensional double well.

This deliberately small campaign exercises the real shared-bias classes with
an overdamped toy integrator.  It is estimator/synchronization evidence, not a
claim of molecular-dynamics or physical convergence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import logsumexp

from pyrpmd.opes import SharedOPESBias
from pyrpmd.shared_metadynamics import (
    AsynchronousMetadynamicsBias, SharedMetadynamicsBias)
from pyrpmd.walker_exchange import make_record


class TwoDimensionalCV:
    dimension = 2
    periods = (None, None)
    units = ("dimensionless", "dimensionless")


def potential(values):
    values = np.asarray(values, dtype=np.float64)
    x, y = values[..., 0], values[..., 1]
    energy = 2.5 * (x * x - 1.0) ** 2 + 0.65 * (y - 0.35 * x) ** 2
    gradient_x = 10.0 * x * (x * x - 1.0) - 0.455 * (y - 0.35 * x)
    gradient_y = 1.3 * (y - 0.35 * x)
    return energy, np.stack((gradient_x, gradient_y), axis=-1)


def grid():
    axes = (np.linspace(-1.6, 1.6, 31), np.linspace(-1.2, 1.2, 25))
    values = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    mask = potential(values)[0] < 4.5
    return values, mask


def shifted(values, mask):
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values[mask])


def metadynamics_run(seed, *, asynchronous, steps=3000, walkers=4,
                     pace=30, poll_steps=23):
    beta, dt, bias_factor = 1.4, 0.0015, 10.0
    sigma, height = (0.24, 0.24), 0.018
    rng = np.random.default_rng(seed)
    coordinates = np.empty((walkers, 2), dtype=np.float64)
    coordinates[:, 0] = (np.where(np.arange(walkers) % 2, 1.0, -1.0)
                         + rng.normal(0.0, 0.1, walkers))
    coordinates[:, 1] = rng.normal(0.0, 0.2, walkers)
    noise = rng.normal(size=(steps, walkers, 2))
    if asynchronous:
        local = [AsynchronousMetadynamicsBias(
            TwoDimensionalCV(), sigma, height, beta, walkers=walkers,
            bias_factor=bias_factor) for _ in range(walkers)]
        records, sequences = [], [0] * walkers
        committed = [0] * walkers
    else:
        bias = SharedMetadynamicsBias(
            TwoDimensionalCV(), sigma, height, beta, walkers=walkers,
            bias_factor=bias_factor)
    for step in range(1, steps + 1):
        _, physical_gradient = potential(coordinates)
        if asynchronous:
            bias_gradient = np.stack([
                local[index].physical_energy_gradient(coordinates[index])[1]
                for index in range(walkers)])
        else:
            bias_gradient = bias.physical_energy_gradient(coordinates)[1]
        coordinates += (-dt * (physical_gradient + bias_gradient)
                        + np.sqrt(2.0 * dt / beta) * noise[step - 1])
        if asynchronous and step % poll_steps == 0:
            for index in range(walkers):
                local[index].merge_records(records[committed[index]:])
                committed[index] = len(records)
        if step % pace == 0:
            if asynchronous:
                # Every writer publishes against the history currently visible
                # to that writer. Own records are visible immediately; remote
                # records become visible on the next poll.
                for index in range(walkers):
                    payload = local[index].prepare_local_payload(
                        coordinates[index])
                    record = make_record(
                        "metadynamics", index, sequences[index], step=step,
                        time=step * dt, payload=payload)
                    sequences[index] += 1
                    records.append(record)
                    local[index].merge_records([record])
            else:
                bias.deposit_walkers(
                    coordinates, step=step, time=step * dt)
    if asynchronous:
        bias = AsynchronousMetadynamicsBias(
            TwoDimensionalCV(), sigma, height, beta, walkers=walkers,
            bias_factor=bias_factor, records=records)
    values, mask = grid()
    estimate = (-bias_factor / (bias_factor - 1.0)
                * bias.physical_energy(values))
    return shifted(estimate, mask)[mask]


def metropolis(seed, count, *, beta=1.4, burn=800, thin=10):
    rng = np.random.default_rng(seed)
    coordinate = rng.normal(size=2)
    energy = float(potential(coordinate)[0])
    samples = []
    for iteration in range(burn + count * thin):
        proposal = coordinate + rng.normal(scale=0.45, size=2)
        proposal_energy = float(potential(proposal)[0])
        if np.log(rng.random()) < -beta * (proposal_energy - energy):
            coordinate, energy = proposal, proposal_energy
        if iteration >= burn and (iteration - burn) % thin == 0:
            samples.append(coordinate.copy())
    return np.asarray(samples)


def analytic_explore_fes(samples, values, *, sigma=0.28,
                         beta=1.4, bias_factor=10.0):
    delta = (values[..., None, :] - np.asarray(samples)) / sigma
    logs = np.sum(-0.5 * delta * delta
                  - np.log(np.sqrt(2.0 * np.pi) * sigma), axis=-1)
    return -bias_factor / beta * (logsumexp(logs, axis=-1)
                                  - np.log(len(samples)))


def shared_opes_fes(samples, *, walkers=4, sigma=0.28,
                    beta=1.4, bias_factor=10.0):
    batches = np.asarray(samples).reshape(-1, walkers, 2)
    bias = SharedOPESBias(
        TwoDimensionalCV(), [sigma, sigma], beta, walkers=walkers,
        barrier=8.0, bias_factor=bias_factor, mode="explore")
    for step, centers in enumerate(batches, 1):
        bias.update_walkers(centers, step=step, time=float(step))
    values, _ = grid()
    return -bias_factor / beta * bias.log_probability(values)


def rms(array):
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))


def run_campaign(replicates=6):
    started = time.perf_counter()
    values, mask = grid()
    sync, asynchronous = [], []
    for index in range(replicates):
        seed = 100 + index
        sync.append(metadynamics_run(seed, asynchronous=False))
        asynchronous.append(metadynamics_run(seed, asynchronous=True))
    sync, asynchronous = np.asarray(sync), np.asarray(asynchronous)
    meta_mean_difference = rms(sync.mean(axis=0) - asynchronous.mean(axis=0))
    meta_sync_spread = rms(np.std(sync, axis=0, ddof=1))
    meta_async_spread = rms(np.std(asynchronous, axis=0, ddof=1))
    meta_combined_sem = float(np.hypot(
        meta_sync_spread, meta_async_spread) / np.sqrt(replicates))

    # Same-center representation check against an independent KDE formula.
    same_samples = metropolis(8123, 160)
    analytic = shifted(analytic_explore_fes(same_samples, values), mask)
    shared_same = shifted(shared_opes_fes(same_samples), mask)
    representation_max_abs = float(np.max(np.abs(
        analytic[mask] - shared_same[mask])))

    single, shared = [], []
    for index in range(8):
        single_samples = metropolis(300 + index, 160)
        shared_samples = np.concatenate([
            metropolis(1300 + index * 4 + walker, 40)
            for walker in range(4)])
        single.append(shifted(
            analytic_explore_fes(single_samples, values), mask)[mask])
        shared.append(shifted(shared_opes_fes(shared_samples), mask)[mask])
    single, shared = np.asarray(single), np.asarray(shared)
    opes_mean_difference = rms(single.mean(axis=0) - shared.mean(axis=0))
    opes_single_spread = rms(np.std(single, axis=0, ddof=1))
    opes_shared_spread = rms(np.std(shared, axis=0, ddof=1))
    opes_combined_sem = float(np.hypot(
        opes_single_spread, opes_shared_spread) / np.sqrt(len(single)))

    return {
        "schema_version": 1,
        "campaign": "two-dimensional double-well multiple-walker estimators",
        "metadynamics": {
            "replicates": replicates,
            "walkers": 4,
            "steps_per_walker": 3000,
            "pace": 30,
            "async_poll_steps": 23,
            "mean_fes_rms_difference": meta_mean_difference,
            "sync_run_to_run_rms_spread": meta_sync_spread,
            "async_run_to_run_rms_spread": meta_async_spread,
            "combined_standard_error": meta_combined_sem,
            "acceptance_limit_2se": 2.0 * meta_combined_sem,
            "within_statistical_error": meta_mean_difference <= 2.0 * meta_combined_sem,
        },
        "opes_explore": {
            "single_kernels": 160,
            "shared_walkers": 4,
            "shared_kernels": 160,
            "independent_replicates": len(single),
            "same_center_formula_max_abs_difference": representation_max_abs,
            "mean_fes_rms_difference": opes_mean_difference,
            "single_run_to_run_rms_spread": opes_single_spread,
            "shared_run_to_run_rms_spread": opes_shared_spread,
            "combined_standard_error": opes_combined_sem,
            "acceptance_limit_2se": 2.0 * opes_combined_sem,
            "within_statistical_error": opes_mean_difference <= 2.0 * opes_combined_sem,
            "compression": "none",
        },
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "note": (
            "Toy overdamped and Markov-chain evidence only; neither estimator "
            "is asserted physically converged."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_campaign(args.replicates)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not (result["metadynamics"]["within_statistical_error"]
            and result["opes_explore"]["within_statistical_error"]
            and result["opes_explore"]["same_center_formula_max_abs_difference"] < 1e-11):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
