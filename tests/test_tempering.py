"""Temperature/Hamiltonian tempering workflow and analytic reference checks."""

from __future__ import annotations

import copy
import itertools
import json

import numpy as np
import pytest
from scipy.special import betainc
from scipy.stats import chisquare

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.replica_exchange import (
    exchange_pairs,
    metropolis_accept,
    round_trip_statistics,
    state_space_diffusion,
    temperature_momentum_scale,
)
from pyrpmd.system import System
from pyrpmd.workflows.tempering import (
    CHECKPOINT_KIND,
    evaluate_thermodynamic_states,
    resolve_tempering,
    run_tempering,
)


def system():
    return System(("H",), np.array([1.0]), np.array([[0.15, -0.1, 0.05]]))


def config(*, backend="serial", total=18, beads=4):
    parallel = {
        "backend": backend,
        "mode": "serial",
        "trajectory_groups": 2 if backend == "mpi" else 1,
        "trajectory_workers": 2 if backend == "process" else 1,
        "bead_workers": 1,
        "threads_per_rank": 1,
        "trajectory_batch_size": 1,
        "scheduling": "static",
        "coordinator_ranks": 0,
        "pes_ranks": 1,
    }
    return {
        "task": "pimd",
        "temperature_K": 300.0,
        "beta": 1.0 / (KB_AU * 300.0),
        "beads": beads,
        "dt_au": 0.025,
        "seed": 41,
        "native": False,
        "potential": {
            "provider": "builtin",
            "name": "harmonic",
            "params": {"k": 0.4},
        },
        "dynamics": {"integrator": "cayley-bcb", "loop_backend": "python"},
        "sampling": {
            "thermostat": "pile-l",
            "integrator": "bcocb",
            "gamma_centroid": 0.7,
            "equilibration_steps": 2,
            "production_steps": total - 2,
            "sample_interval": 2,
            "replicas": 1,
            "andersen_interval": 3,
        },
        "parallel": parallel,
        "output": {"checkpoint_steps": 5},
        "tempering": {
            "temperatures": ["180 K", "540 K"],
            "exchange_interval_steps": 4,
            "pairing": "odd_even",
            "seed": 19,
        },
    }


def mkdirs(path):
    for name in ("checkpoints", "samples", "results"):
        (path / name).mkdir(parents=True, exist_ok=True)


def checkpoint_arrays(path):
    with load_checkpoint(path, expected_kind=CHECKPOINT_KIND, require_integrity=True) as saved:
        return {name: saved[name] for name in saved.files}


def assert_archives_equal(left, right):
    a, b = checkpoint_arrays(left), checkpoint_arrays(right)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_input_schema_keeps_reus_separate_and_resolves_geometric_ladder(tmp_path):
    raw = {
        "task": "pimd",
        "temperature": "300 K",
        "beads": 4,
        "seed": 7,
        "system": {
            "symbols": ["H"],
            "geometry": [[0.0, 0.0, 0.0]],
            "geometry_unit": "bohr",
            "masses": [1.0],
        },
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.4}},
        "dynamics": {"integrator": "cayley-bcb", "timestep": "0.01 fs", "loop_backend": "python"},
        "sampling": {
            "thermostat": "pile-l",
            "gamma_centroid": "0.02 fs^-1",
            "equilibration_steps": 2,
            "production_steps": 8,
            "sample_interval": 2,
            "replicas": 1,
        },
        "tempering": {
            "temperature_min": "150 K",
            "temperature_max": "600 K",
            "replicas": 3,
            "spacing": "geometric",
            "exchange_interval_steps": 2,
        },
        "output": {"directory": str(tmp_path / "run"), "checkpoint_steps": 3},
    }
    resolved, model = JobSpec.from_mapping(raw).resolve()
    options = resolve_tempering(resolved, model)
    np.testing.assert_allclose(options.temperatures, [150.0, 300.0, 600.0])
    assert options.mode == "temperature" and options.nstates == 3
    with pytest.raises(InputError, match="replica_exchange"):
        JobSpec.from_mapping({**raw, "replica_exchange": {"enabled": True}}).resolve()


@pytest.mark.parametrize("target", ["potential", "bias", "split_slow"])
def test_hamiltonian_target_contracts_are_explicit(target):
    cfg = config()
    cfg["tempering"] = {"mode": "hamiltonian", "lambdas": [0.5, 1.0], "target": target}
    if target == "bias":
        with pytest.raises(ValueError, match="requires task='fes'"):
            resolve_tempering(cfg, system())
    elif target == "split_slow":
        # Unsupported provider compositions fail during read-only preflight.
        with pytest.raises(ValueError, match="SplitPotential"):
            resolve_tempering(cfg, system())
    else:
        assert resolve_tempering(cfg, system()).target == "potential"



def test_ladders_reject_boolean_and_negative_hamiltonian_scales():
    cfg = config()
    cfg["tempering"] = {"temperatures": [True, "300 K"]}
    with pytest.raises(ValueError, match="temperature"):
        resolve_tempering(cfg, system())
    cfg["tempering"] = {
        "mode": "hamiltonian", "lambdas": [-0.5, 1.0], "target": "potential"
    }
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_tempering(cfg, system())


def test_mpi_group_default_follows_trajectory_workers():
    cfg = config(backend="mpi")
    cfg["parallel"].pop("trajectory_groups")
    cfg["parallel"]["trajectory_workers"] = 2
    assert resolve_tempering(cfg, system()).nstates == 2


def _finite_p_covariance(beta, beads, mass, stiffness):
    difference = 2 * np.eye(beads) - np.roll(np.eye(beads), 1, axis=0) - np.roll(np.eye(beads), -1, axis=0)
    precision = beads * mass / beta * difference + beta * stiffness / beads * np.eye(beads)
    return np.linalg.inv(precision), difference


def test_harmonic_finite_p_estimators_match_exact_gaussian_trace():
    cfg = config(total=10, beads=5)
    cfg["tempering"]["temperatures"] = ["120 K", "450 K", "900 K"]
    options = resolve_tempering(cfg, system())
    provider = HarmonicPotential(0.4)
    zero = np.zeros((cfg["beads"], 1, 3))
    base = evaluate_thermodynamic_states(
        provider, zero, cfg, system(), options, [None] * options.nstates
    )[1]
    for state, beta in enumerate(options.betas):
        covariance, difference = _finite_p_covariance(beta, cfg["beads"], 1.0, 0.4)
        root = np.linalg.cholesky(covariance)
        reconstructed = base[state].copy()
        for axis in range(3):
            for column in root.T:
                q = np.zeros_like(zero)
                q[:, 0, axis] = column
                value = evaluate_thermodynamic_states(
                    provider, q, cfg, system(), options, [None] * options.nstates
                )[1][state]
                reconstructed += value - base[state]
        p = cfg["beads"]
        trace = np.trace(covariance)
        spring_trace = np.trace(difference @ covariance)
        centroid_projector = np.eye(p) - np.ones((p, p)) / p
        centered_trace = np.trace(centroid_projector @ covariance)
        reference = np.array([
            3 * 0.4 * trace / (2 * p),
            3 * p / (2 * beta) - 3 * p * spring_trace / (2 * beta**2),
            3 / (2 * beta) + 3 * 0.4 * centered_trace / (2 * p),
            3 * 0.4 * trace / (2 * p) + 3 / (2 * beta)
            + 3 * 0.4 * centered_trace / (2 * p),
        ])
        np.testing.assert_allclose(reconstructed, reference, rtol=2e-13, atol=2e-15)


def test_classical_harmonic_exchange_acceptance_matches_beta_distribution():
    beta_cold, beta_hot = 5.0, 1.25
    dimension = 3
    ratio = beta_cold / beta_hot
    expected = 2.0 * betainc(dimension / 2.0, dimension / 2.0, 1.0 / (1.0 + ratio))
    rng = np.random.default_rng(7401)
    samples = 180_000
    cold = rng.chisquare(dimension, size=samples) / (2 * beta_cold)
    hot = rng.chisquare(dimension, size=samples) / (2 * beta_hot)
    log_ratio = (beta_cold - beta_hot) * (cold - hot)
    observed = np.mean(np.minimum(1.0, np.exp(np.minimum(log_ratio, 0.0))))
    assert observed == pytest.approx(expected, abs=2.5e-3)
    # The workflow's rescaling maps the source canonical variance exactly.
    scale = temperature_momentum_scale(beta_cold, beta_hot)
    assert scale**2 / beta_cold == pytest.approx(1.0 / beta_hot)


def test_discrete_permutation_chain_passes_theoretical_chi_square():
    energy = np.array([
        [0.1, 1.2, 2.5],
        [1.6, 0.2, 1.1],
        [2.2, 1.4, 0.3],
    ])
    permutations = list(itertools.permutations(range(3)))
    weight = np.array([np.exp(-sum(energy[w, slot] for w, slot in enumerate(p))) for p in permutations])
    probability = weight / weight.sum()
    slots = np.arange(3, dtype=np.int64)
    rng = np.random.default_rng(902)
    counts = np.zeros(len(permutations), dtype=np.int64)
    burn = 2_000
    sweeps = 120_000
    for sweep in range(burn + sweeps):
        for left, right in exchange_pairs(3, "odd_even", sweep):
            inverse = np.empty(3, dtype=int)
            inverse[slots] = np.arange(3)
            first, second = inverse[left], inverse[right]
            log_ratio = (
                energy[first, left] + energy[second, right]
                - energy[first, right] - energy[second, left]
            )
            if metropolis_accept(log_ratio, rng):
                slots[first], slots[second] = right, left
        if sweep >= burn:
            counts[permutations.index(tuple(slots))] += 1
    statistic = chisquare(counts, sweeps * probability)
    assert statistic.pvalue > 1e-3


def test_two_state_odd_even_proposes_the_only_edge_every_sweep():
    assert exchange_pairs(2, "odd_even", 0) == ((0, 1),)
    assert exchange_pairs(2, "odd_even", 1) == ((0, 1),)


def test_diagnostics_use_walker_to_state_history():
    history = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
    ])
    steps = np.arange(len(history)) * 4
    trips = round_trip_statistics(history, steps)
    assert trips["completed_total"] >= 1
    diffusion = state_space_diffusion(history, steps, max_lag=3)
    assert diffusion["event_lags"] == [1, 2, 3]
    assert all(value >= 0 for value in diffusion["mean_squared_slot_displacement"])


def test_interrupted_committed_generation_resumes_bitwise(tmp_path, monkeypatch):
    import pyrpmd.workflows.tempering as module

    cfg = config(total=18)
    reference = tmp_path / "reference"
    interrupted = tmp_path / "interrupted"
    mkdirs(reference)
    mkdirs(interrupted)
    run_tempering(copy.deepcopy(cfg), system(), reference)
    original = module._propagate_replica

    def fail_after_committed_prefix(task):
        if task["start"] >= 8:
            raise RuntimeError("injected committed-boundary interruption")
        return original(task)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_propagate_replica", fail_after_committed_prefix)
        with pytest.raises(RuntimeError, match="committed-boundary"):
            run_tempering(copy.deepcopy(cfg), system(), interrupted)
    resumed = copy.deepcopy(cfg)
    resumed["resume"] = True
    run_tempering(resumed, system(), interrupted)
    assert_archives_equal(
        reference / "checkpoints" / "tempering.npz",
        interrupted / "checkpoints" / "tempering.npz",
    )
    for left in sorted((reference / "samples").glob("*.npz")):
        right = interrupted / "samples" / left.name
        assert right.exists()
        with load_checkpoint(left, require_integrity=True) as a, load_checkpoint(right, require_integrity=True) as b:
            assert a.files == b.files
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_complete_staged_generation_is_committed_on_resume_bitwise(tmp_path):
    import pyrpmd.workflows.tempering as module

    cfg = config(total=10)
    reference = tmp_path / "reference"
    recovered = tmp_path / "recovered"
    mkdirs(reference)
    mkdirs(recovered)
    run_tempering(copy.deepcopy(cfg), system(), reference)
    options = resolve_tempering(cfg, system())
    module._initial_checkpoint(cfg, system(), recovered, options)
    stop = module._next_boundary(0, options)
    for walker in range(options.nstates):
        module._propagate_replica({
            "config": cfg,
            "system": system(),
            "output": str(recovered),
            "walker": walker,
            "start": 0,
            "stop": stop,
            "generation": 0,
        })
    resumed = copy.deepcopy(cfg)
    resumed["resume"] = True
    run_tempering(resumed, system(), recovered)
    assert_archives_equal(
        reference / "checkpoints" / "tempering.npz",
        recovered / "checkpoints" / "tempering.npz",
    )


def test_partial_replica_generation_is_rejected(tmp_path):
    cfg = config(total=10)
    mkdirs(tmp_path)
    options = resolve_tempering(cfg, system())
    # Initialize a complete ensemble, then leave exactly one uncommitted stage.
    import pyrpmd.workflows.tempering as module

    module._initial_checkpoint(cfg, system(), tmp_path, options)
    module._propagate_replica({
        "config": cfg,
        "system": system(),
        "output": str(tmp_path),
        "walker": 0,
        "start": 0,
        "stop": 2,
        "generation": 0,
    })
    resumed = copy.deepcopy(cfg)
    resumed["resume"] = True
    with pytest.raises(RuntimeError, match="partial replica generation"):
        run_tempering(resumed, system(), tmp_path)


def test_process_and_serial_scheduling_are_bitwise_equal(tmp_path):
    serial, process = tmp_path / "serial", tmp_path / "process"
    mkdirs(serial)
    mkdirs(process)
    run_tempering(config(backend="serial", total=10), system(), serial)
    run_tempering(config(backend="process", total=10), system(), process)
    assert_archives_equal(
        serial / "checkpoints" / "tempering.npz",
        process / "checkpoints" / "tempering.npz",
    )


def test_summary_is_conservative_and_trajectories_are_by_state(tmp_path):
    mkdirs(tmp_path)
    result = run_tempering(config(total=10), system(), tmp_path, return_data=True)
    summary = result["summary"]
    assert summary["converged"] is False
    assert summary["samples_per_state"] == 4
    assert summary["mbar"]["temperature_dependence"].startswith("Each target state")
    assert "state_space_diffusion" in summary
    for state, path in enumerate(summary["files"]["trajectories"]):
        with load_checkpoint(path, require_integrity=True) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            assert metadata["thermodynamic_state"] == state
            assert saved["positions"].shape[1:] == (4, 1, 3)


def test_public_runner_dispatch_and_offline_reanalysis(tmp_path):
    from pyrpmd.runner import analyze_run, run

    mapping = {
        "task": "pimd",
        "temperature": "300 K",
        "beads": 3,
        "seed": 22,
        "native": False,
        "system": {
            "symbols": ["H"], "masses": [1.0],
            "geometry": [[0.1, 0.0, 0.0]], "geometry_unit": "bohr",
        },
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.3}},
        "dynamics": {"integrator": "cayley-bcb", "timestep": 0.02, "loop_backend": "python"},
        "sampling": {
            "thermostat": "pile-l", "gamma_centroid": 0.5,
            "equilibration_steps": 2, "production_steps": 8,
            "sample_interval": 2, "replicas": 1,
        },
        "tempering": {
            "temperatures": ["200 K", "500 K"],
            "exchange_interval_steps": 2,
        },
        "output": {"checkpoint_steps": 3},
    }
    result = run(mapping, output=tmp_path)
    assert result["method"] == "parallel_tempering"
    assert result["output"] == str(tmp_path.resolve())
    before = (tmp_path / "checkpoints/tempering.npz").read_bytes()
    replay = analyze_run(tmp_path, recompute=True)
    assert replay["acceptance"] == result["acceptance"]
    assert (tmp_path / "checkpoints/tempering.npz").read_bytes() == before


def test_split_slow_hremd_executes_contracted_delta_provider(tmp_path):
    cfg = config(total=10, beads=4)
    cfg["potential"] = {
        "provider": "delta",
        "reference": {
            "provider": "builtin", "name": "harmonic", "params": {"k": 0.2}
        },
        "correction": {
            "provider": "builtin", "name": "harmonic", "params": {"k": 0.3}
        },
        "nbeads": 4,
        "contracted_beads": 2,
    }
    cfg["tempering"] = {
        "mode": "hamiltonian",
        "lambdas": [0.0, 1.0],
        "target": "split_slow",
        "exchange_interval_steps": 2,
    }
    mkdirs(tmp_path)
    summary = run_tempering(cfg, system(), tmp_path)
    assert summary["method"] == "hamiltonian_replica_exchange"
    assert summary["hamiltonian_target"] == "split_slow"
    assert summary["lambdas"] == [0.0, 1.0]


def test_fes_umbrella_hremd_and_frozen_metadynamics_whitelist(tmp_path):
    base = config(total=10, beads=3)
    base.update(
        task="fes",
        beta=1.0 / (KB_AU * 300.0),
        temperature_K=300.0,
        cvs=[{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        analysis={"bins": 8, "bootstrap": 0},
        fes={"method": "umbrella"},
        umbrella={"centers": [[0.0]], "stiffness": [[0.8]]},
    )
    base["tempering"] = {
        "mode": "hamiltonian",
        "lambdas": [0.5, 1.0],
        "target": "bias",
        "exchange_interval_steps": 2,
    }
    umbrella = tmp_path / "umbrella"
    mkdirs(umbrella)
    summary = run_tempering(base, system(), umbrella)
    assert summary["method"] == "hamiltonian_replica_exchange"
    assert summary["hamiltonian_target"] == "bias"

    meta = copy.deepcopy(base)
    meta.pop("umbrella")
    meta["fes"] = {"method": "metadynamics"}
    meta["metadynamics"] = {
        "mode": "well_tempered", "sigma": [0.2], "height": 0.01,
        "bias_factor": 4.0, "pace": 2, "freeze_after_steps": 4,
        "freeze_equilibration_steps": 0,
    }
    meta["tempering"] = {
        "temperatures": ["220 K", "480 K"],
        "exchange_interval_steps": 2,
    }
    directory = tmp_path / "metadynamics"
    mkdirs(directory)
    result = run_tempering(meta, system(), directory, return_data=True)
    assert all(len(hills) == 2 for hills in result["checkpoint"]["bias_states"])
    invalid = copy.deepcopy(meta)
    invalid["tempering"] = {
        "mode": "hamiltonian", "lambdas": [0.5, 1.0], "target": "bias"
    }
    with pytest.raises(ValueError, match="evolving metadynamics"):
        resolve_tempering(invalid, system())
