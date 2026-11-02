"""Source-block uncertainty, paired propagation, and exact study resume gates."""

from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[1] / "examples/electrochemistry"
sys.path.insert(0, str(EXAMPLE_DIRECTORY))
_SPEC = importlib.util.spec_from_file_location(
    "adaptive_nrpmd_study", EXAMPLE_DIRECTORY / "adaptive_nrpmd_study.py",
)
study = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = study
_SPEC.loader.exec_module(study)


def gaussian_sources(count=256):
    """Independent samples from restrained, analytically normalizable Gaussians."""
    rng = np.random.default_rng(391)
    centers = np.array([[-0.7], [0.3], [1.2]])
    mean, spring, beta, stiffness = 0.25, 0.8, 2.0, 1.3
    values, chains, windows = [], [], []
    for chain in range(3):
        for window, center in enumerate(centers[:, 0]):
            local_mean = (spring*mean + stiffness*center)/(spring+stiffness)
            sigma = np.sqrt(1/(beta*(spring+stiffness)))
            values.extend(rng.normal(local_mean, sigma, count))
            chains.extend([chain]*count)
            windows.extend([window]*count)
    x = np.asarray(values)
    return dict(cv_values=x[:, None], signs=np.ones(len(x)),
                observables=np.column_stack((x, (x-mean)**2)),
                centers=centers, stiffness=np.array([[stiffness]]), beta=beta,
                chain_ids=np.asarray(chains), window_ids=np.asarray(windows),
                replicates=32, seed=284,
                block_lengths={(c,w): 4 for c in range(3) for w in range(3)})


def test_bootstrap_refits_mbar_and_recovers_independent_gaussian_moments(monkeypatch):
    original = study.solve_mbar
    fits = []

    def record_fit(*args, **kwargs):
        result = original(*args, **kwargs)
        fits.append(result)
        return result

    monkeypatch.setattr(study, "solve_mbar", record_fit)
    arguments = gaussian_sources()
    result = study.source_block_bootstrap(**arguments)
    assert len(fits) == arguments["replicates"]
    assert result["confidence_intervals_available"]
    assert result["valid_replicates"] == 32
    draws = np.asarray(result["valid_draw_estimates_diagnostic"])
    np.testing.assert_allclose(draws.mean(axis=0), [0.25, 1/(2*0.8)], atol=0.06)
    assert np.all(np.asarray(result["standard_error"]) > 0)
    assert study.source_block_bootstrap(**arguments) == result


def test_resampled_blocks_never_cross_original_chain_or_umbrella(monkeypatch):
    arguments = gaussian_sources(count=32)
    chain, window = arguments["chain_ids"], arguments["window_ids"]
    selected = []

    def identify_sources(values, centers, stiffness, beta):
        return np.tile(np.arange(len(values)), (len(centers), 1))

    def record_sources(u, counts, **kwargs):
        selected.append(u[0].astype(int))
        return SimpleNamespace(weights=np.full(u.shape[1], 1/u.shape[1]),
                               converged=True, components=np.zeros(len(counts)))

    monkeypatch.setattr(study, "umbrella_reduced_potentials", identify_sources)
    monkeypatch.setattr(study, "solve_mbar", record_sources)
    study.source_block_bootstrap(**arguments)
    for draw in selected:
        for w in range(3):
            for c in range(3):
                segment = draw[(w*3+c)*32:(w*3+c+1)*32]
                assert np.all(window[segment] == w)
                assert len(np.unique(chain[segment])) == 1
                # Every circular block remains chronological in its own stream.
                blocks = (segment % 32).reshape(-1, 4)
                np.testing.assert_array_equal(np.diff(blocks, axis=1) % 32,
                                              np.ones((8, 3), dtype=int))


def test_bootstrap_refuses_unresolved_signs_and_too_few_source_blocks():
    arguments = gaussian_sources(count=32)
    arguments["block_lengths"] = {(c,w): 32 for c in range(3) for w in range(3)}
    result = study.source_block_bootstrap(**arguments)
    assert not result["confidence_intervals_available"]
    assert "fewer_than_eight_complete_blocks_in_a_source_stream" in result["unavailable_reasons"]
    assert result["percentile_interval"] is None
    arguments = gaussian_sources(count=32)
    arguments["signs"] *= -1
    result = study.source_block_bootstrap(**arguments)
    assert result["valid_replicates"] == 0
    assert len(result["failure_records"]) == arguments["replicates"]
    assert result["standard_error"] is None
    with pytest.raises(ValueError, match="no failure filtering"):
        study.source_block_bootstrap(**arguments, minimum_valid_fraction=0.5)


@pytest.mark.parametrize("field,value", [
    ("chain_ids", "fractional"), ("window_ids", "unknown"),
    ("chain_ids", "negative"), ("block_lengths", "missing"),
])
def test_bootstrap_rejects_invalid_source_identity(field, value):
    arguments = gaussian_sources(count=16)
    if value == "fractional":
        arguments[field] = arguments[field].astype(float)+0.5
    elif value == "unknown":
        arguments[field][0] = 99
    elif value == "negative":
        arguments[field][0] = -1
    else:
        arguments[field].pop((0,0))
    with pytest.raises(ValueError, match="IDs|every source"):
        study.source_block_bootstrap(**arguments)


def test_matched_dynamics_uses_same_complete_parents_and_endpoint():
    potential, _, _ = study.make_model(0.0)
    sampler = study.ThermalNRPMDSampler(
        potential, [1.5], 3.0, np.zeros((2,1,3)), 2, seed=123,
    )
    parents = sampler.sample(3, burn_in=5)
    original = {name: np.array([getattr(p, name) for p in parents.states])
                for name in ("q", "p", "mapping_q", "mapping_p")}
    report, arrays = study.matched_timestep_dynamics(
        parents, potential, [1.5], 3.0, 2,
        dts=(0.02, 0.01, 0.005), duration=0.1,
    )
    assert arrays["time"][-1] == pytest.approx(0.1)
    assert report["same_complete_parents"]
    for name, initial in original.items():
        for d in range(3):
            np.testing.assert_array_equal(arrays[name][d,:,0], initial)
        np.testing.assert_array_equal([getattr(p,name) for p in parents.states], initial)
        errors = report["errors_against_finest_timestep"]
        assert errors[0][name+"_maximum_absolute_error"] > errors[1][name+"_maximum_absolute_error"]
    assert report["bias_applied"] is False
    assert report["rate_constant"] is None


@pytest.mark.parametrize("dts,duration", [
    ((), 0.1), ((0.02,), 0.1), ((0.02, np.nan), 0.1),
    ((0.02, 0.01), -0.1), ((0.02, 0.01), np.inf),
    ((0.02, 0.01), 0.105), ((0.02, 0.007), 0.1),
])
def test_matched_dynamics_rejects_mismatched_endpoint_before_propagation(dts, duration):
    with pytest.raises(ValueError, match="matched timesteps"):
        study.matched_timestep_dynamics(
            None, None, [1.5], 3.0, 2, dts=dts, duration=duration,
        )


def test_case_ids_distinguish_arbitrarily_close_allowed_voltages():
    config = study.StudyConfig(potentials_V=(0.0, 0.00001, np.nextafter(0.0,1.0)))
    ids = [study._point_id(v, 1) for v in config.potentials_V]
    assert len(set(ids)) == len(ids)
    assert all("/" not in case for case in ids)


def test_completed_smoke_case_resumes_only_with_identical_source_and_artifacts(tmp_path, monkeypatch):
    config = study.preset_config("smoke")
    report = study.run_case(asdict(config), 0.0, 1, tmp_path)
    assert report["production_reweighting"]["available"]
    cached = study.run_case(asdict(config), 0.0, 1, tmp_path, resume=True)
    assert cached == report
    with pytest.raises(ValueError, match="completed case exists"):
        study.run_case(asdict(config), 0.0, 1, tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        study.run_case(asdict(replace(config, seed=config.seed+1)), 0.0, 1, tmp_path, resume=True)
    identity = study.source_identity()
    monkeypatch.setattr(study, "source_identity", lambda: {**identity, "sha256": "changed"})
    with pytest.raises(ValueError, match="identity mismatch"):
        study.run_case(asdict(config), 0.0, 1, tmp_path, resume=True)
    monkeypatch.setattr(study, "source_identity", lambda: identity)
    source_path = tmp_path/report["point_id"]/"source.npz"
    with np.load(source_path) as raw:
        assert raw["q"].shape[0] == config.production_chains*config.images*config.production_samples
        assert np.array_equal(np.unique(raw["chain_ids"]), np.arange(config.production_chains))
    source_path.write_bytes(source_path.read_bytes()+b"altered")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        study.run_case(asdict(config), 0.0, 1, tmp_path, resume=True)
