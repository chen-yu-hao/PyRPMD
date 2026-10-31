"""Real rate trajectories retain their identity across authorized production growth.

The deliberately cheap free-particle model tests continuation and saved-result
contracts, not the accuracy or convergence of a molecular reaction rate.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.analysis.rate_io import postprocess_rate
from pyrpmd.analysis.reporting import collect_run
from pyrpmd.checkpoint import ENVELOPE, atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.runner import extend_run, resume_run, run_resolved
from pyrpmd.system import System
from pyrpmd.workflows import fes, rate


SHORT = 31
LONG = 74


def configuration(*, automatic_surface=False):
    system = System(
        ("H", "H", "H"), np.array([2.0, 2.0, 2.0]),
        np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        ((0,), (1, 2)),
    )
    config = {
        "task": "rate", "beta": 5.0, "temperature_K": 300.0,
        "beads": 2, "dt_au": 0.003, "seed": 17, "native": False,
        "potential": {"provider": "free"},
        "sampling": {
            "equilibration_steps": 7, "production_steps": SHORT,
            "sample_interval": 3, "andersen_interval": 7,
        },
        "reaction": {
            "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]], "rinf": 4.0,
        },
        "umbrella": {"centers": np.linspace(0, 1, 5).tolist(), "kappa": 10.0},
        "analysis": {"bins": 12, "bootstrap": 0},
        "parallel": {"backend": "serial"},
        "dynamics": {"block_steps": 11, "loop_backend": "python"},
        "output": {"checkpoint_steps": 13},
        "recrossing": {
            "parent_replicas": 1, "parent_samples": 2,
            "equilibration_steps": 5, "parent_interval_steps": 3,
            "child_pairs": 1, "child_steps": 10, "bootstrap": 0,
        },
    }
    if automatic_surface:
        config["rate"] = {"pmf_search_range": [0.6, 1.0]}
    else:
        config["recrossing"]["dividing_surface"] = 1.0
    return config, system


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rate-extension-baseline")
    config, system = configuration()
    short = directory / "short"
    result = run_resolved(config, system, output=short)
    assert result["pmf"]["ready"] and result["child_trajectories"] == 4
    longer = copy.deepcopy(config)
    longer["sampling"]["production_steps"] = LONG
    continuous = directory / "continuous"
    reference = run_resolved(longer, system, output=continuous)
    assert reference["pmf"]["ready"] and reference["child_trajectories"] == 4
    return {"short": short, "long": continuous, "config": config,
            "system": system, "original": result, "reference": reference}


@pytest.fixture
def saved_run(tmp_path, baseline):
    directory = tmp_path / "run"
    shutil.copytree(baseline["short"], directory)
    return directory


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hashes(directory):
    directory = Path(directory)
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix not in (".lock", ".tmp")
    }


def immutable_original(directory):
    return {
        name: digest for name, digest in file_hashes(directory).items()
        if name != "results/summary.json"
        and not name.startswith("umbrella/checkpoints/")
        and not name.startswith("extensions/")
        and name != "results/extensions.json"
    }


def checkpoint_arrays(path):
    with load_checkpoint(path) as saved:
        arrays = {key: np.array(saved[key], copy=True)
                  for key in saved.files if key != ENVELOPE}
        if "sample_stream" in saved:
            history = fes.read_fes_sample_history(
                saved, dimension=1, chunk_directory=Path(path).with_suffix(".chunks"),
            )
            arrays.update(values=history["values"], energies=history["energies"])
        return arrays


def equal_arrays(first, second, keys=None):
    a, b = checkpoint_arrays(first), checkpoint_arrays(second)
    if keys is None:
        assert a.keys() == b.keys()
        keys = a
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{Path(first).name}: {key}")


def generation(directory, total=LONG):
    return Path(directory) / "extensions" / "rate" / f"production-{total:012d}"


def assert_continuous_result(directory, baseline, result):
    target = generation(directory)
    for checkpoint in sorted((baseline["long"] / "umbrella/checkpoints").glob("*.npz")):
        equal_arrays(checkpoint, directory / "umbrella/checkpoints" / checkpoint.name,
                     ("q", "p", "step", "time", "rng", "values", "energies", "elapsed"))
        arrays = checkpoint_arrays(directory / "umbrella/checkpoints" / checkpoint.name)
        assert int(arrays["elapsed"]) == baseline["config"]["sampling"]["equilibration_steps"] + LONG
        assert len(arrays["values"]) == LONG // baseline["config"]["sampling"]["sample_interval"]
    equal_arrays(baseline["long"] / "rate_pmf.npz", target / "rate_pmf.npz")
    equal_arrays(baseline["long"] / "recrossing.npz", target / "recrossing.npz")
    equal_arrays(baseline["long"] / "checkpoints/parent_0000.npz",
                 target / "checkpoints/parent_0000.npz",
                 ("q", "p", "step", "time", "rng", "snapshots", "elapsed", "xi_star", "pmf_method"))
    for checkpoint in sorted((baseline["long"] / "checkpoints").glob("child_*.npz")):
        equal_arrays(checkpoint, target / "checkpoints" / checkpoint.name,
                     ("numerator", "denominator", "energy_drift",
                      "endpoint_energy_error_hartree_per_bead", "endpoint_energy_error_kbt",
                      "xi_star", "pmf_method"))
    for key in ("kappa", "rate_estimate_cm3_s", "rate_cm3_s", "child_trajectories", "status"):
        assert result[key] == baseline["reference"][key]
    assert result["pmf"] == baseline["reference"]["pmf"]


def test_split_production_matches_continuous_rate_without_reinitializing_windows(saved_run, baseline):
    before = immutable_original(saved_run)
    identities = {path.name: str(checkpoint_arrays(path)["fingerprint"])
                  for path in (saved_run / "umbrella/checkpoints").glob("*.npz")}
    # These boundaries are deliberately incommensurate: the continuation must
    # preserve sample, thermostat, block, and checkpoint clocks.
    with patch.object(fes, "steer_to_center", side_effect=AssertionError("reinitialized a window")):
        result = extend_run(saved_run, production_steps=LONG)
    assert_continuous_result(saved_run, baseline, result)
    assert immutable_original(saved_run) == before
    assert result["original_production_steps"] == SHORT
    assert result["effective_production_steps"] == LONG
    assert result["rate_extension"]["generation"] == str(generation(Path(), LONG))
    assert result["rate_extension"]["production_steps"] == LONG
    assert len(result["rate_extension"]["receipt_sha256"]) == 64
    ledger = read_json(saved_run / "results/extensions.json")
    assert ledger["format"] == "PyRPMD.rate.extensions.v1"
    assert [(request["production_steps"], request["status"])
            for request in ledger["requests"]] == [(LONG, "completed")]
    for path in (saved_run / "umbrella/checkpoints").glob("*.npz"):
        assert str(checkpoint_arrays(path)["fingerprint"]) == identities[path.name]


@pytest.mark.parametrize("factory_kind", ["module", "file"])
def test_worker_restores_local_plugin_imports_after_root_preparation_broadcast(
        tmp_path, monkeypatch, factory_kind):
    from pyrpmd.potentials.python import import_object
    from pyrpmd.workflows import rate_extend

    base = tmp_path / "plugins"
    base.mkdir()
    module = "_extension_plugin_" + hashlib.sha256(str(base).encode()).hexdigest()[:12]
    dependency = module + "_dependency"
    (base / (dependency + ".py")).write_text("VALUE = 'worker imported sibling'\n")
    provider = base / (module + ".py")
    provider.write_text(f"from {dependency} import VALUE\ndef build():\n    return VALUE\n")
    factory = (module if factory_kind == "module" else str(provider)) + ":build"
    config, system = configuration()
    config.update(_base_directory=str(base), potential={"provider": "python", "factory": factory})
    directory = tmp_path / "run"
    prepared = {"completed": False, "config": config, "sampling_config": config,
                "system": system, "generation": str(generation(directory)),
                "preparation": {"ready": True}}
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(base)])
    stages = []

    def broadcast_only(function, *args, **kwargs):
        # A non-root rank receives root results; it must never execute root's
        # preparation or filesystem writes itself merely to obtain sys.path.
        stages.append(function)
        if function is rate_extend._prepare_extension:
            assert str(base) not in sys.path
            return prepared
        assert function in (rate_extend._lock, rate_extend._recover_receipt,
                            rate_extend.write_json, rate_extend._unlock)
        return None

    class ReachedWorkerDispatch(RuntimeError):
        pass

    def sample(sampling_config, *args, **kwargs):
        assert sys.path.count(str(base)) == 1
        assert import_object(sampling_config["potential"]["factory"])() == "worker imported sibling"
        raise ReachedWorkerDispatch()

    monkeypatch.setattr(rate_extend, "root_call", broadcast_only)
    monkeypatch.setattr(fes, "run_fes", sample)
    previous_modules = set(sys.modules)
    try:
        with pytest.raises(ReachedWorkerDispatch):
            rate_extend.extend_rate(directory, production_steps=LONG)
    finally:
        for name in set(sys.modules) - previous_modules:
            file = getattr(sys.modules[name], "__file__", None)
            if isinstance(file, str) and Path(file).is_relative_to(base):
                sys.modules.pop(name)
    assert stages[0] is rate_extend._lock and stages[-1] is rate_extend._unlock
    assert not directory.exists()


def test_rejected_legacy_resume_does_not_migrate_an_extended_manifest(saved_run, baseline):
    from pyrpmd.input import InputError
    from pyrpmd.io import fingerprint, write_json

    config = read_json(saved_run / "input/resolved.json")
    manifest_path = saved_run / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(schema_version=1, physics_fingerprint=fingerprint(config, baseline["system"]))
    manifest.pop("schema", None)
    write_json(manifest_path, manifest)
    original = manifest_path.read_bytes()
    result = extend_run(saved_run, production_steps=LONG)
    assert manifest_path.read_bytes() == original
    before = file_hashes(saved_run)
    with pytest.raises(InputError, match="authorized extension budget"):
        run_resolved(config, baseline["system"], output=saved_run, resume=True)
    assert manifest_path.read_bytes() == original
    assert file_hashes(saved_run) == before
    assert resume_run(saved_run) == result


def test_mbar_extension_matches_continuous_supported_pmf_and_recrossing(tmp_path):
    config, system = configuration()
    config["rate"] = {"pmf_method": "mbar", "dividing_surface": 1.0}
    config["analysis"].update(bins=5, ranges=[[-0.125, 1.125]], min_count=1)
    directory, continuous = tmp_path / "run", tmp_path / "continuous"
    original = run_resolved(config, system, output=directory)
    longer = copy.deepcopy(config)
    longer["sampling"]["production_steps"] = LONG
    reference = run_resolved(longer, system, output=continuous)
    result = extend_run(directory, production_steps=LONG)
    assert original["pmf"]["ready"] and result["pmf"]["ready"]
    assert result["pmf"]["method"] == "mbar"
    assert_continuous_result(directory, {"long": continuous, "config": config,
                                         "reference": reference}, result)


def test_same_budget_and_ordinary_resume_are_read_only(saved_run):
    result = extend_run(saved_run, production_steps=LONG)
    before = file_hashes(saved_run)
    with (
        patch.object(Dynamics, "step", side_effect=AssertionError("propagated an idempotent request")),
        patch.object(Dynamics, "propagate_block", side_effect=AssertionError("propagated an idempotent request")),
    ):
        repeated = extend_run(saved_run, production_steps=LONG)
        resumed = resume_run(saved_run)
    assert repeated == resumed == result
    assert file_hashes(saved_run) == before


def test_direct_original_resume_cannot_silently_use_an_extended_budget(saved_run, baseline):
    extend_run(saved_run, production_steps=LONG)
    before = file_hashes(saved_run)
    with (
        patch.object(Dynamics, "step", side_effect=AssertionError("propagated without extension authorization")),
        patch.object(Dynamics, "propagate_block", side_effect=AssertionError("propagated without extension authorization")),
    ):
        with pytest.raises((ValueError, RuntimeError), match="authorized.*budget"):
            run_resolved(baseline["config"], baseline["system"], output=saved_run, resume=True)
    assert file_hashes(saved_run) == before


def test_larger_second_extension_keeps_prior_generation_immutable(saved_run):
    first = extend_run(saved_run, production_steps=LONG)
    first_files = file_hashes(generation(saved_run))
    second = extend_run(saved_run, production_steps=LONG + 13)
    assert second["effective_production_steps"] == LONG + 13
    assert second["rate_extension"]["generation"] != first["rate_extension"]["generation"]
    assert file_hashes(generation(saved_run)) == first_files
    requests = read_json(saved_run / "results/extensions.json")["requests"]
    assert [(item["production_steps"], item["status"]) for item in requests] == [
        (LONG, "completed"), (LONG + 13, "completed"),
    ]
    before = file_hashes(saved_run)
    with pytest.raises((ValueError, RuntimeError), match="reduce|budget|production"):
        extend_run(saved_run, production_steps=LONG)
    assert file_hashes(saved_run) == before


@pytest.mark.parametrize("total", [True, 0, -1, SHORT, SHORT - 1, 74.5, "74"])
def test_invalid_or_nonincreasing_target_never_changes_run(saved_run, total):
    before = file_hashes(saved_run)
    with patch.object(Dynamics, "step", side_effect=AssertionError("propagated invalid target")):
        with pytest.raises((ValueError, RuntimeError)):
            extend_run(saved_run, production_steps=total)
    assert file_hashes(saved_run) == before


@pytest.mark.parametrize("what", ["temperature", "original_budget", "system", "missing", "corrupt", "integrity", "fingerprint"])
def test_all_original_inputs_and_windows_are_authenticated_before_any_growth(saved_run, what):
    checkpoint = sorted((saved_run / "umbrella/checkpoints").glob("*.npz"))[-1]
    if what in ("temperature", "original_budget"):
        path = saved_run / "input/resolved.json"
        config = read_json(path)
        if what == "temperature":
            config["beta"] *= 2
        else:
            # Setting the original budget to the desired target must not be
            # mistaken for a legitimate extension authorization.
            config["sampling"]["production_steps"] = LONG
        path.write_text(json.dumps(config))
    elif what == "system":
        path = saved_run / "input/system.json"
        system = read_json(path)
        system["masses"][0] *= 2
        path.write_text(json.dumps(system))
    elif what == "missing":
        checkpoint.unlink()
    elif what == "corrupt":
        checkpoint.write_bytes(b"interrupted or corrupt npz")
    elif what == "integrity":
        with np.load(checkpoint, allow_pickle=False) as saved:
            arrays = dict(saved.items())
        arrays["q"][0, 0, 0] += 0.1
        np.savez(checkpoint, **arrays)
    else:
        arrays = checkpoint_arrays(checkpoint)
        arrays["fingerprint"] = np.asarray("0" * 64)
        atomic_savez(checkpoint, **arrays)
    before = file_hashes(saved_run)
    with (
        patch.object(Dynamics, "step", side_effect=AssertionError("advanced before complete preflight")),
        patch.object(Dynamics, "propagate_block", side_effect=AssertionError("advanced before complete preflight")),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            extend_run(saved_run, production_steps=LONG)
    assert file_hashes(saved_run) == before
    assert not (saved_run / "results/extensions.json").exists()


def test_referenced_potential_change_is_rejected_before_growth(tmp_path):
    config, system = configuration()
    provider = tmp_path / "extension_model.py"
    provider.write_text("from pyrpmd.potentials import FreePotential\ndef build():\n    return FreePotential()\n")
    config["potential"] = {"provider": "python", "factory": str(provider) + ":build"}
    directory = tmp_path / "run"
    run_resolved(config, system, output=directory)
    provider.write_text(provider.read_text() + "# modified scientific provider\n")
    before = file_hashes(directory)
    with pytest.raises((ValueError, RuntimeError), match="file|model|changed"):
        extend_run(directory, production_steps=LONG)
    assert file_hashes(directory) == before


def test_interrupted_umbrella_write_resumes_exactly_and_disallows_superseding(saved_run, baseline, tmp_path):
    previous_summary = (saved_run / "results/summary.json").read_bytes()
    immutable = immutable_original(saved_run)
    save = fes._atomic_npz
    cutoff = baseline["config"]["sampling"]["equilibration_steps"] + SHORT

    def persist_then_interrupt(path, **arrays):
        save(path, **arrays)
        if Path(path).parent == saved_run / "umbrella/checkpoints" and int(arrays["elapsed"]) > cutoff:
            raise KeyboardInterrupt("crash after durable continuation checkpoint")

    with patch.object(fes, "_atomic_npz", side_effect=persist_then_interrupt), pytest.raises(KeyboardInterrupt):
        extend_run(saved_run, production_steps=LONG)
    assert read_json(saved_run / "results/extensions.json")["requests"][-1]["status"] == "running"
    assert (saved_run / "results/summary.json").read_bytes() == previous_summary
    assert immutable_original(saved_run) == immutable
    assert any(int(checkpoint_arrays(path)["elapsed"]) > cutoff
               for path in (saved_run / "umbrella/checkpoints").glob("*.npz"))
    # A visible pending directory must not replace the last committed PMF.
    old = postprocess_rate(saved_run, bootstrap=0, output=tmp_path / "pending-analysis.json")
    assert old["rate_estimate_cm3_s"] == pytest.approx(baseline["original"]["rate_estimate_cm3_s"], rel=1e-12)
    before = file_hashes(saved_run)
    with pytest.raises((ValueError, RuntimeError), match="pending|resume|Resume"):
        extend_run(saved_run, production_steps=LONG + 1)
    assert file_hashes(saved_run) == before
    with patch.object(fes, "steer_to_center", side_effect=AssertionError("reinitialized during recovery")):
        result = resume_run(saved_run)
    assert_continuous_result(saved_run, baseline, result)
    assert immutable_original(saved_run) == immutable


def test_interrupted_children_keep_completed_pairs_and_recover(saved_run, baseline):
    previous_summary = (saved_run / "results/summary.json").read_bytes()
    save = rate._atomic_npz

    def persist_then_interrupt(path, **arrays):
        save(path, **arrays)
        if Path(path).name.startswith("child_"):
            raise KeyboardInterrupt("crash after complete child pair")

    with patch.object(rate, "_atomic_npz", side_effect=persist_then_interrupt), pytest.raises(KeyboardInterrupt):
        extend_run(saved_run, production_steps=LONG)
    cached = {path: path.read_bytes() for path in (generation(saved_run) / "checkpoints").glob("child_*.npz")}
    assert len(cached) == 1
    assert (saved_run / "results/summary.json").read_bytes() == previous_summary
    result = resume_run(saved_run)
    assert all(path.read_bytes() == payload for path, payload in cached.items())
    assert_continuous_result(saved_run, baseline, result)


def stop_before_first_parent(directory):
    with patch.object(rate, "_parent_task", side_effect=KeyboardInterrupt("stop before first parent")):
        with pytest.raises(KeyboardInterrupt):
            extend_run(directory, production_steps=LONG)
    assert (generation(directory) / "generation.json").is_file()
    assert not (generation(directory) / "receipt.json").exists()


def test_pending_generation_rejects_changed_source_checkpoint_before_writes(saved_run):
    stop_before_first_parent(saved_run)
    checkpoint = sorted((saved_run / "umbrella/checkpoints").glob("*.npz"))[-1]
    with load_checkpoint(checkpoint) as saved:
        arrays = dict(saved.items())
    # A self-consistent NPZ with unchanged original physics is still not the
    # completed source from which this pending recrossing generation began.
    arrays["q"][0, 0, 0] += 0.01
    atomic_savez(checkpoint, **arrays)
    before = file_hashes(saved_run)
    with patch.object(Dynamics, "propagate_block", side_effect=AssertionError("used a substituted source")):
        with pytest.raises((ValueError, RuntimeError), match="source|generation"):
            resume_run(saved_run)
    assert file_hashes(saved_run) == before


def test_pending_generation_cannot_adopt_old_parent_even_at_identical_surface(saved_run):
    stop_before_first_parent(saved_run)
    folder = generation(saved_run) / "checkpoints"
    folder.mkdir(exist_ok=True)
    shutil.copyfile(saved_run / "checkpoints/parent_0000.npz", folder / "parent_0000.npz")
    previous_summary = (saved_run / "results/summary.json").read_bytes()
    # Old and new parent constraints are both xi=1, but their starting umbrella
    # endpoints differ. A matching surface/physics fingerprint is insufficient.
    with patch.object(rate, "_child_tasks", side_effect=AssertionError("accepted a foreign parent origin")):
        with pytest.raises((ValueError, RuntimeError), match="parent|origin|generation|physics"):
            resume_run(saved_run)
    assert (saved_run / "results/summary.json").read_bytes() == previous_summary


@pytest.mark.parametrize("schema", ["manifest", "ledger", "generation"])
def test_unknown_scientific_or_extension_schemas_are_refused(saved_run, schema):
    if schema == "manifest":
        path = saved_run / "manifest.json"
        value = read_json(path)
        value["schema_version"] = 999
    elif schema == "ledger":
        extend_run(saved_run, production_steps=LONG)
        path = saved_run / "results/extensions.json"
        value = read_json(path)
        value["format"] = "PyRPMD.rate.extensions.v999"
    else:
        stop_before_first_parent(saved_run)
        path = generation(saved_run) / "generation.json"
        value = read_json(path)
        value["format"] = "PyRPMD.rate.generation.v999"
    path.write_text(json.dumps(value))
    before = file_hashes(saved_run)
    with patch.object(Dynamics, "propagate_block", side_effect=AssertionError("used an unknown schema")):
        with pytest.raises((ValueError, RuntimeError)):
            extend_run(saved_run, production_steps=LONG)
    assert file_hashes(saved_run) == before


def test_missing_committed_generation_artifact_is_not_rebuilt_or_replaced_by_old_result(saved_run, tmp_path):
    extend_run(saved_run, production_steps=LONG)
    (generation(saved_run) / "recrossing.npz").unlink()
    before = file_hashes(saved_run)
    with patch.object(Dynamics, "propagate_block", side_effect=AssertionError("rebuilt a corrupted result")):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            resume_run(saved_run)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        postprocess_rate(saved_run, bootstrap=0, output=tmp_path / "analysis.json")
    assert not (tmp_path / "analysis.json").exists()
    assert file_hashes(saved_run) == before


def test_new_unsupported_pmf_withholds_previous_rate_and_launches_no_children(saved_run, baseline, tmp_path):
    reconstruct = fes.reconstruct_fes

    def disconnected_analysis(*args, **kwargs):
        data = reconstruct(*args, **kwargs)
        # Fault injection represents a newly unsupported overlap analysis after
        # real continuation; the actual UI producer and publication path decide
        # how that condition affects the PMF, children, and exported rate.
        data["connected"] = False
        return data

    before = immutable_original(saved_run)
    with (
        patch.object(fes, "reconstruct_fes", side_effect=disconnected_analysis),
        patch.object(rate, "_parent_task", side_effect=AssertionError("sampled unsupported surface")),
        patch.object(rate, "_child_task", side_effect=AssertionError("launched unsupported children")),
    ):
        result = extend_run(saved_run, production_steps=LONG)
    assert baseline["original"]["rate_estimate_cm3_s"] is not None
    assert result["effective_production_steps"] == LONG
    assert result["pmf"]["ready"] is False
    assert result["status"] == "insufficient_overlap"
    assert result["pmf"]["failure_reason"] == "disconnected_overlap"
    assert result["kappa"] is None
    assert result["rate_cm3_s"] is result["rate_estimate_cm3_s"] is None
    assert not (generation(saved_run) / "recrossing.npz").exists()
    assert not list((generation(saved_run) / "checkpoints").glob("child_*.npz"))
    assert immutable_original(saved_run) == before
    analyzed = postprocess_rate(saved_run, bootstrap=0, output=tmp_path / "unsupported.json")
    assert analyzed["status"] == "source_pmf_not_ready"
    assert analyzed["rate_estimate_cm3_s"] is None
    assert resume_run(saved_run) == result


@pytest.mark.parametrize("interrupt_after_commit", [False, True])
def test_interrupted_publication_recovers_without_repropagation(saved_run, baseline, interrupt_after_commit):
    previous_summary = (saved_run / "results/summary.json").read_bytes()
    replace = os.replace
    destination = saved_run / "results/summary.json"

    def interrupt_publication(source, target):
        if Path(target) == destination:
            if interrupt_after_commit:
                replace(source, target)
            raise KeyboardInterrupt("crash at atomic result publication")
        return replace(source, target)

    with patch("pyrpmd.io.os.replace", side_effect=interrupt_publication), pytest.raises(KeyboardInterrupt):
        extend_run(saved_run, production_steps=LONG)
    assert (generation(saved_run) / "receipt.json").is_file()
    if not interrupt_after_commit:
        assert destination.read_bytes() == previous_summary
    else:
        assert read_json(destination)["rate_extension"]["production_steps"] == LONG
    artifacts = file_hashes(generation(saved_run))
    with (
        patch.object(Dynamics, "step", side_effect=AssertionError("repropagated a complete generation")),
        patch.object(Dynamics, "propagate_block", side_effect=AssertionError("repropagated a complete generation")),
    ):
        result = resume_run(saved_run)
    assert file_hashes(generation(saved_run)) == artifacts
    assert read_json(saved_run / "results/extensions.json")["requests"][-1]["status"] == "completed"
    assert_continuous_result(saved_run, baseline, result)


def test_real_automatic_surface_shift_gets_fresh_parents_and_children(tmp_path):
    config, system = configuration(automatic_surface=True)
    directory = tmp_path / "run"
    original = run_resolved(config, system, output=directory)
    old_surface = original["pmf"]["xi_star"]
    old_checkpoints = file_hashes(directory / "checkpoints")
    result = extend_run(directory, production_steps=LONG)
    new_surface = result["pmf"]["xi_star"]
    assert old_surface != new_surface
    assert original["pmf"]["ready"] and result["pmf"]["ready"]
    assert file_hashes(directory / "checkpoints") == old_checkpoints
    assert result["child_trajectories"] == original["child_trajectories"] == 4
    prepared = rate._prepare(config, system, tmp_path / "inspection")
    surface = rate.rate_coordinate(prepared, system).dividing_surface(new_surface)
    parent = checkpoint_arrays(generation(directory) / "checkpoints/parent_0000.npz")
    assert float(parent["xi_star"]) == new_surface
    for q in parent["snapshots"]:
        assert abs(surface.centroid_value_gradient(q.mean(axis=0))[0]) < 1e-8
    children = sorted((generation(directory) / "checkpoints").glob("child_*.npz"))
    assert len(children) == 2
    for child in children:
        assert float(checkpoint_arrays(child)["xi_star"]) == new_surface
        assert str(checkpoint_arrays(child)["fingerprint"]) != str(
            checkpoint_arrays(directory / "checkpoints" / child.name)["fingerprint"])
    binding = read_json(generation(directory) / "generation.json")
    assert binding["xi_star"] == new_surface
    assert binding["production_steps"] == LONG
    assert binding["parent_initial_q_sha256"]
    assert len(binding["source_checkpoints"]) == len(config["umbrella"]["centers"])


def test_offline_readers_follow_committed_generation_and_keep_root_identity(saved_run, baseline, tmp_path):
    result = extend_run(saved_run, production_steps=LONG)
    assert result["rate_estimate_cm3_s"] != baseline["original"]["rate_estimate_cm3_s"]
    before = file_hashes(saved_run)
    with (
        patch.object(Dynamics, "step", side_effect=AssertionError("offline reader ran dynamics")),
        patch.object(Dynamics, "propagate_block", side_effect=AssertionError("offline reader ran dynamics")),
    ):
        analyzed = postprocess_rate(saved_run, bootstrap=0, output=tmp_path / "analysis.json")
        report = collect_run(saved_run, energy_unit="hartree")
    assert analyzed["rate_estimate_cm3_s"] == pytest.approx(result["rate_estimate_cm3_s"], rel=1e-12, abs=0)
    assert analyzed["physics_fingerprint"] == read_json(saved_run / "manifest.json")["physics_fingerprint"]
    provenance = {Path(item["path"]).resolve() for item in analyzed["source_files"]}
    assert generation(saved_run) / "rate_pmf.npz" in provenance
    assert generation(saved_run) / "recrossing.npz" in provenance
    assert saved_run / "input/resolved.json" in provenance
    assert saved_run / "manifest.json" in provenance
    pmf_plot = next(item for item in report["_plots"] if item["id"].endswith("-pmf"))
    pmf = checkpoint_arrays(generation(saved_run) / "rate_pmf.npz")
    np.testing.assert_array_equal(pmf_plot["series"][0]["x"], pmf["xi"])
    np.testing.assert_array_equal(pmf_plot["series"][0]["y"], pmf["free_energy"])
    assert file_hashes(saved_run) == before


def test_corrupt_active_receipt_never_falls_back_to_old_rate(saved_run, tmp_path):
    extend_run(saved_run, production_steps=LONG)
    receipt = generation(saved_run) / "receipt.json"
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises((ValueError, RuntimeError), match="receipt|checksum"):
        postprocess_rate(saved_run, bootstrap=0, output=tmp_path / "analysis.json")
    assert not (tmp_path / "analysis.json").exists()
    try:
        report = collect_run(saved_run, energy_unit="hartree")
    except (ValueError, RuntimeError):
        pass
    else:
        assert not any(item["id"].endswith(("-pmf", "-recrossing")) for item in report["_plots"])
