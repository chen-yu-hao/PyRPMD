"""Independent quartic rate checks through the ordinary input workflows.

The stationary quadrature reference and raw-observation error calculations
are independent of the workflow's summaries.  A passing finite-chain check
does not certify physical, timestep, bead or interface convergence.
"""

import json
import math
from pathlib import Path
import runpy
import time

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.input import JobSpec
from pyrpmd.runner import run


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "sampling"
REFERENCE = runpy.run_path(str(EXAMPLES / "path_sampling_quartic.py"))


def reference_rate():
    partition, error = quad(
        lambda x: np.exp(-(x * x - 1)**2), -np.inf, np.inf,
        epsabs=1e-12, epsrel=1e-12,
    )
    assert error < 2e-11
    return 2 * np.exp(-1) / (np.sqrt(2 * np.pi) * partition)


def observations(directory):
    with load_checkpoint(
        directory / "samples" / "path_sampling.npz",
        expected_kind="PyRPMD.path_sampling_observations.v1", require_integrity=True,
    ) as data:
        return {key: data[key].copy() for key in data.files if key != "metadata"}


def block_error(values):
    assert len(values) >= 4
    result = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert np.isfinite(result) and result > 0
    return result


@pytest.mark.parametrize("method", ["retis", "repptis"])
def test_toml_quartic_rates_match_equilibrium_flux_with_raw_block_errors(tmp_path, method):
    # These are the documented input files, including relative Python factory
    # resolution and the external flux's first-center convention for REPPTIS.
    job = JobSpec.from_file(EXAMPLES / f"path_sampling_{method}.inp")
    start = time.perf_counter()
    result = run(job, output=tmp_path / method)
    raw = observations(tmp_path / method)
    size = job.data["path_sampling"]["block_size"]
    n = job.data["path_sampling"]["samples"]
    nblocks = n // size
    assert raw["path_frames"].shape == (n, 2)
    if method == "retis":
        crossing = raw["crossing"].reshape(nblocks, size, 2).mean(axis=1)
        durations = (raw["minus_frames"] + raw["path_frames"][:, 0] - 4) * 0.04
        rates = crossing.prod(axis=1) / durations.reshape(nblocks, size).mean(axis=1)
        estimate = raw["crossing"].mean(axis=0).prod() / durations.mean()
        assert result["zero_swap_attempts"] > 0
        field = "rate_standard_error_block_au"
    else:
        starts = raw["starts"].reshape(nblocks, size, 2)
        ends = raw["ends"].reshape(nblocks, size, 2)
        # In this special uphill-only ladder, every right-origin path returns
        # left.  The exact directional first-passage recurrence then reduces
        # to a product of left-origin conditional probabilities.
        assert np.all(ends[starts == 1] == -1)
        left_counts = (starts == -1).sum(axis=1)
        assert np.all(left_counts > 0)
        assert np.all((starts == 1).sum(axis=1) > 0)
        right_exits = ((starts == -1) & (ends == 1)).sum(axis=1)
        flux = job.data["path_sampling"]["external_flux"]["flux_au"]
        rates = flux * (right_exits / left_counts).prod(axis=1)
        estimate = flux * (((raw["starts"] == -1) & (raw["ends"] == 1)).sum(axis=0)
                           / (raw["starts"] == -1).sum(axis=0)).prod()
        assert 0 < result["swap_accepts"][0] < result["swap_attempts"][0]
        assert result["memory_loss_approximation"] is True
        field = "rate_standard_error_block_conditional_on_flux_au"
    error = block_error(rates)
    exact = reference_rate()
    assert result["rate_estimate_au"] == pytest.approx(estimate, rel=2e-14)
    assert result[field] == pytest.approx(error, rel=2e-14)
    assert abs(estimate - exact) <= 5 * error
    assert result["completed"] is True
    assert result["converged"] is False and result["physical_converged"] is False
    assert result["rate_s_inv"] is None
    assert all(item["length_rejections"] == 0 for item in result["interfaces"])
    print("PATH_WORKFLOW_QUARTIC_REFERENCE", json.dumps({
        "method": method, "rate_estimate_au": float(estimate),
        "independent_block_se_au": error, "exact_rate_au": exact,
        "samples": n, "block_size": size, "dt_au": 0.04,
        "wall_seconds": time.perf_counter() - start,
    }, sort_keys=True))


def test_infinite_retis_quartic_rate_uses_only_free_ensemble_observations(tmp_path):
    job = JobSpec.from_file(EXAMPLES / "path_sampling_infinite_retis.inp")
    start = time.perf_counter()
    result = run(job, output=tmp_path / "infinite")
    raw = observations(tmp_path / "infinite")
    counts, crossings, durations = (raw[key] for key in ("counts", "crossings", "durations"))
    n = job.data["path_sampling"]["samples"]
    size = job.data["path_sampling"]["block_size"]
    assert counts.shape == (n, 3)
    assert np.any(~counts)
    assert np.all(crossings[~counts] == 0)
    assert np.all(durations[~counts] == 0)

    def statistic(c, y, t):
        observed = c.sum(axis=0)
        assert np.all(observed > 0)
        probabilities = y[:, 1:].sum(axis=0) / observed[1:]
        cycle = t[:, 0].sum() / observed[0] + t[:, 1].sum() / observed[1]
        return probabilities.prod() / cycle

    estimate = statistic(counts, crossings, durations)
    rates = [statistic(counts[i:i+size], crossings[i:i+size], durations[i:i+size])
             for i in range(0, n, size)]
    error = block_error(rates)
    exact = reference_rate()
    assert result["rate_estimate_au"] == pytest.approx(estimate, rel=2e-14)
    assert result["rate_standard_error_block_au"] == pytest.approx(error, rel=2e-14)
    assert abs(estimate - exact) <= 5 * error
    assert result["completed"] and not result["physical_converged"]
    print("PATH_WORKFLOW_QUARTIC_REFERENCE", json.dumps({
        "method": "infinite-retis", "rate_estimate_au": float(estimate),
        "independent_block_se_au": error, "exact_rate_au": exact,
        "events": n, "free_counts": counts.sum(axis=0).tolist(),
        "block_size": size, "workers": 2, "dt_au": 0.04,
        "wall_seconds": time.perf_counter() - start,
    }, sort_keys=True))


def test_ffs_quartic_rate_matches_reference_across_independent_complete_runs(tmp_path):
    start = time.perf_counter()
    rates = []
    for seed in (981, 1990, 2999, 4008):
        config = REFERENCE["workflow_input"](
            "ffs", seed=seed, flux_steps=6000, trials=160, block_size=100,
        )
        directory = tmp_path / f"ffs-{seed}"
        summary = run(config, output=directory)
        raw = observations(directory)
        assert summary["completed"]
        assert not summary["physical_converged"]
        # Retain complete zero-event runs if they occur; never condition the
        # independent-run average on at least one success.
        flux = raw["flux_events"].sum() / (0.04 * raw["flux_a_history"].sum())
        probabilities = []
        for index in range(2):
            outcomes = raw[f"stage_{index}_outcomes"]
            if not len(outcomes):
                break
            probabilities.append(float(outcomes.mean()))
        estimate = flux * math.prod(probabilities)
        assert summary["rate_estimate_au"] == pytest.approx(estimate, rel=2e-14)
        assert summary["rate_standard_error_au"] is None
        rates.append(estimate)
    error = block_error(rates)
    estimate = float(np.mean(rates))
    exact = reference_rate()
    assert abs(estimate - exact) <= 5 * error
    print("PATH_WORKFLOW_QUARTIC_REFERENCE", json.dumps({
        "method": "ffs-basin-andersen", "rate_estimate_au": estimate,
        "independent_whole_run_se_au": error, "exact_rate_au": exact,
        "replicate_rates_au": rates, "flux_steps_per_run": 6000,
        "trials_per_interface": 160, "dt_au": 0.04,
        "wall_seconds": time.perf_counter() - start,
    }, sort_keys=True))
