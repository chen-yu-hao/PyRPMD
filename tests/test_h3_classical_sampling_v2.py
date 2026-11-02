"""Preservation and exact restart checks for the separate classical kernel."""
from pathlib import Path

import numpy as np
import pytest

from benchmarks.h3.classical_sampling_v2 import run
from benchmarks.h3.mixing_sampling import run as original_run
from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from benchmarks.h3.classical_sampling_v2 import CHECKPOINT_KIND


@pytest.fixture
def library():
    path = Path(__file__).resolve().parents[1]/"benchmarks/h3/build/libbkmp2.so"
    if not path.is_file():
        pytest.skip("optional pinned BKMP2 library is not built")
    return path


def options(library):
    return dict(library=library, seed=883719, warmup=12, production=60,
                centers=np.array([.1, .8, .98, 1.02]), stiffness=np.array([2., 5., 30., 30.]),
                checkpoint_interval=19, diagnostic_interval=7)


def test_zero_additional_moves_preserves_original_chain_exactly(tmp_path, library):
    old, new = tmp_path/"old", tmp_path/"new"
    original_run(old, beads=1, **options(library))
    run(new, xi_moves=0, **options(library))
    with np.load(old/"checkpoint.npz", allow_pickle=False) as before, \
         np.load(new/"checkpoint.npz", allow_pickle=False) as after:
        for key in ("R", "r", "completed", "samples", "diagnostics", "swaps", "lineage", "rng"):
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        for key in ("accepted", "proposed"):
            np.testing.assert_array_equal(before[key][[0, 1, 4, 5]], after[key][:4])
            assert not after[key][4].any()


@pytest.mark.parametrize("xi_moves", [1, 3])
def test_new_kernel_restart_is_exact(tmp_path, library, xi_moves):
    full, partial = tmp_path/"full", tmp_path/"partial"
    run(full, xi_moves=xi_moves, **options(library))
    run(partial, xi_moves=xi_moves, stop_after=23, **options(library))
    run(partial, xi_moves=xi_moves, resume=True, **options(library))
    with np.load(full/"checkpoint.npz", allow_pickle=False) as a, \
         np.load(partial/"checkpoint.npz", allow_pickle=False) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_array_tamper_is_rejected_before_any_new_sample(tmp_path, library):
    path = tmp_path/"run"
    run(path, xi_moves=1, stop_after=23, **options(library))
    with np.load(path/"checkpoint.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["samples"][0, 0] += .1
    np.savez_compressed(path/"checkpoint.npz", **arrays)
    with pytest.raises(CheckpointError, match="integrity check failed"):
        run(path, xi_moves=1, resume=True, **options(library))


@pytest.mark.parametrize("change", ["fingerprint", "shape"])
def test_valid_envelope_does_not_bypass_configuration_or_shape_guard(tmp_path, library, change):
    path = tmp_path/"run"
    run(path, xi_moves=1, stop_after=23, **options(library))
    with load_checkpoint(path/"checkpoint.npz", require_integrity=True) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if change == "fingerprint":
        arrays["settings_fingerprint"] = np.asarray("different configuration")
    else:
        arrays["samples"] = arrays["samples"][:-1]
    atomic_savez(path/"checkpoint.npz", kind=CHECKPOINT_KIND, **arrays)
    with pytest.raises(CheckpointError, match="fingerprint mismatch|shape mismatch"):
        run(path, xi_moves=1, resume=True, **options(library))


def test_source_mutation_refuses_checkpoint_write(tmp_path, library, monkeypatch):
    from benchmarks.h3 import classical_sampling_v2 as sampler
    actual = sampler.source_identity(library)
    calls = 0
    def changed_source(_library):
        nonlocal calls
        calls += 1
        return actual if calls == 1 else {**actual, "changed_after_start": "test"}
    monkeypatch.setattr(sampler, "source_identity", changed_source)
    with pytest.raises(RuntimeError, match="refusing checkpoint overwrite"):
        sampler.run(tmp_path/"run", xi_moves=1, stop_after=2, **options(library))
    assert not (tmp_path/"run/checkpoint.npz").exists()
