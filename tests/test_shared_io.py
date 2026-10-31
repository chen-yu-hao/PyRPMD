"""Legacy identities and atomic persistence survive shared-helper refactoring.

The fingerprint literals were produced by the pre-refactor implementations at
c4fa743f552c1d71db35fab791a2e63409bc1c3e. They intentionally distinguish the
manifest and stream protocols; the tests never rebuild the expected hash with
production serialization code or require a second checkout.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Barrier, Lock

import numpy as np
import pytest

from pyrpmd import io
from pyrpmd.checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_metadata_npz,
)
from pyrpmd.system import System
from pyrpmd.workflows import fes, opes


@dataclass
class _SemanticOptions:
    stiffness: float
    cache_generation: int = 99

    def to_dict(self):
        # This cache field is implementation detail, not scientific identity.
        return {"stiffness": self.stiffness}


class _ArrayOptions:
    def to_dict(self):
        return {"weights": np.array([1.0, 2.0]), "nested": {"value": 1.25}}


class _MissingOptions:
    def to_dict(self):
        return {"estimate": float("nan")}


@dataclass
class _PlainOptions:
    stiffness: float
    cache_generation: int = 99


@dataclass
class _NumericalReport:
    values: np.ndarray
    path: Path


def _identity_problem(case):
    system = System(
        ("H", "D"), np.array([1.0, 2.0]),
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        fragments=((0,), (1,)),
    )
    config = {
        "task": "fes", "beta": 2.0, "beads": 4, "dt_au": 0.03,
        "seed": np.int64(42), "native": False,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": np.float64(0.2)}},
        "dynamics": {"integrator": "cayley-bcb", "loop_backend": "python",
                     "block_steps": 16},
        "sampling": {"equilibration_steps": 4, "production_steps": 20,
                     "sample_interval": 2, "andersen_interval": 3, "replicas": 2},
        "cvs": [{"name": "r", "kind": "distance", "atoms": [0, 1]}],
        "umbrella": {"centers_au": [[1.0], [2.0]], "kappa_au": [[2.0]]},
        "analysis": {"bins": 16, "bootstrap": 4, "measure": "cv_density"},
        "parallel": {"backend": "serial", "trajectory_workers": 1},
        "output": {"directory": Path("runs/original"), "checkpoint_steps": 5},
        "resume": False, "_resume": False, "_warnings": ["pilot"],
        "_input_path": "/inputs/original.py", "_base_directory": "/inputs",
    }
    if case == "execution":
        config.update(
            parallel={"backend": "process", "trajectory_workers": 8},
            output={"directory": Path("runs/resumed"), "checkpoint_steps": 17},
            resume=True, _resume=True, _warnings=["resumed"],
            _input_path="/moved/input.py", _base_directory="/moved",
        )
    elif case == "loop_tuning":
        config["dynamics"].update(loop_backend="auto", block_steps=128)
    elif case == "reanalysis":
        config["analysis"].update(bins=32, bootstrap=12)
    elif case == "measure":
        config["analysis"]["measure"] = "rate_flux"
    elif case == "flux_axis":
        config["analysis"].update(measure="rate_flux", flux_axis=0)
    elif case == "geometry":
        system.geometry[1, 0] = 1.6
    elif case in ("periodic", "explicit_periodic"):
        system = System(
            system.symbols, system.masses, system.geometry,
            fragments=system.fragments, cell=np.diag([8.0, 10.0, 12.0]),
            pbc=(True, False, True),
        )
        if case == "explicit_periodic":
            config["cvs"][0]["kind"] = "periodic_distance"
    elif case == "dataclass":
        config["potential"]["params"]["options"] = _SemanticOptions(1.25)
    elif case == "array_to_dict":
        config["potential"]["params"]["options"] = _ArrayOptions()
    elif case == "missing_to_dict":
        config["potential"]["params"]["options"] = _MissingOptions()
    elif case == "plain_dataclass":
        config["potential"]["params"]["options"] = _PlainOptions(1.25)
    elif case != "base":
        raise AssertionError(case)
    return config, system


# Each tuple is (manifest, FES window, OPES replica).
_LEGACY_FINGERPRINTS = {
    'base': (
        'cd3b069a8c4f635e682dc081bb92e5d9f1703ba36e196db3e7f21c13aaf1f95d',
        '06c62f0ad918712536e5269b8ad110b16f8a9d9f288b59fa045e7efd768603e7',
        '04720eec41293a78ff538af7ac11c63a9cba97f3fcf02554903c696c3a0588a4',
    ),
    'execution': (
        'cd3b069a8c4f635e682dc081bb92e5d9f1703ba36e196db3e7f21c13aaf1f95d',
        '06c62f0ad918712536e5269b8ad110b16f8a9d9f288b59fa045e7efd768603e7',
        '04720eec41293a78ff538af7ac11c63a9cba97f3fcf02554903c696c3a0588a4',
    ),
    'loop_tuning': (
        '7404095f7e77f77c3467177dc6494b02e4ec21c5594e8a96ed1a5c633332ddaa',
        '06c62f0ad918712536e5269b8ad110b16f8a9d9f288b59fa045e7efd768603e7',
        '04720eec41293a78ff538af7ac11c63a9cba97f3fcf02554903c696c3a0588a4',
    ),
    'reanalysis': (
        '3f8857f2caa25b176db37782696d01ec8a9f428cfc415e3dcd04ca60a9343c9d',
        '06c62f0ad918712536e5269b8ad110b16f8a9d9f288b59fa045e7efd768603e7',
        '04720eec41293a78ff538af7ac11c63a9cba97f3fcf02554903c696c3a0588a4',
    ),
    'measure': (
        'ff1953debe74924e31753e95f9ed6ffdac99e073e6ae4e1460183a6670cdc457',
        '8137cf3587d52b0ff712d177609e31b6166886d9c2f20a8348da8ce3369576cb',
        '77416f6d134bb152e77284428847e5bf82632e038761c215d46814e108b264d4',
    ),
    'flux_axis': (
        '9c4b4bc2b202046edf58900d4eb8c1e7d7f3a8cfa1f7777e0ad9994aa35faa5c',
        '4bab99f7aadc8a2358515e6b566957b2d8fb12abdd50cfa0ad931ae1d8184049',
        '9a622a0595bdd49aed44afb55b4326686e36ec028df9012e2c7e216ebf3f6fab',
    ),
    'geometry': (
        'ffc716e395a9aa1f03badd4bcccbf54c7d68190873e00de69d5b97bf5ce44314',
        'f3d8cf3cab8c97e7f9179ebb8e9fd77280ce8d8425cb1eaffe9c48443a62b1d9',
        'b66fbfdfb4dcd2c354b4052bb6af55c04a12778fa1415b5ad573578b2997c24a',
    ),
    'periodic': (
        '251bb89b421f0d405e1b35ebd02023677c2eb6d60cff6636ec233de707487238',
        '4c360a2a852a3e019f668a3b06b1a6a5d5ec0c4dde7dcd7251863eb1c5d9a6cb',
        '64f7bfaf5179fbf4e61c6216a6a74c6a812a61c52566cd6a69e756b6220f3ae4',
    ),
    'explicit_periodic': (
        'a177e3915f7f209b251fadf3dd5c75f6cfd42e4ff2013a39f4e6857b6a350104',
        '8d95717f92680bfd5c746322756141f6b7150ff4641150a1bc9ad8388625b3be',
        'dd72c6b68398b44777793a9fecb7b4524d872b5f5bb3385d44d2c8018604e5de',
    ),
    'dataclass': (
        '13f4a826cb7153146b39c3bfe1006baed5579e03856a225bd70ae39569d88cae',
        '06e37f6dc3e211c0693dbc5a0a90b6aff9d201ad2eb48892489f65989fa600ae',
        'a56a3c81078ed88af7e3fbbcf0ba4249f1980d7a6a6c728f0784045295bffcee',
    ),
    'array_to_dict': (
        'cd02c316a476aa49b2137c80e9cb7dd436e881233dce1b03e02e4fd8eaa16899',
        'fa5b42bb9fd92c2e0fd9a6665fa14eb4c5ae6fcfb78ad8fdee9967a8aa211666',
        '0971854883108f9b8d8341dfd506adc46209956314e1c1defd9c60488603f3ad',
    ),
}


@pytest.mark.parametrize("case", _LEGACY_FINGERPRINTS)
def test_legacy_manifest_and_stream_fingerprints_are_byte_compatible(case):
    config, system = _identity_problem(case)
    manifest, window, replica = _LEGACY_FINGERPRINTS[case]
    assert io.fingerprint(config, system) == manifest
    assert fes._fingerprint(config, system, {"window": 2, "replica": 1}) == window
    opes_config = dict(config, task="opes", opes={"sigma_au": [0.3], "pace": 3,
                                                "barrier_au": 2.0, "bias_factor": 5.0})
    opes_config.pop("umbrella")
    assert opes._stream_fingerprint(opes_config, system, 1) == replica


def test_stream_identity_separates_science_from_reanalysis_and_execution():
    identities = {}
    for case in _LEGACY_FINGERPRINTS:
        config, system = _identity_problem(case)
        identities[case] = io.fingerprint(config, system, {"window": 2, "replica": 1})
    assert identities["base"] == identities["execution"]
    assert identities["base"] == identities["loop_tuning"] == identities["reanalysis"]
    for case in ("measure", "flux_axis", "geometry", "periodic", "explicit_periodic"):
        assert identities[case] != identities["base"]
    assert identities["measure"] != identities["flux_axis"]
    assert identities["periodic"] != identities["explicit_periodic"]
    config, system = _identity_problem("base")
    assert io.fingerprint(config, system, {"window": 3, "replica": 1}) != identities["base"]
    # The legacy manifest protocol retains analysis and raw loop options.
    assert _LEGACY_FINGERPRINTS["base"][0] != _LEGACY_FINGERPRINTS["loop_tuning"][0]
    assert _LEGACY_FINGERPRINTS["base"][0] != _LEGACY_FINGERPRINTS["reanalysis"][0]


def test_semantic_to_dict_precedes_dataclass_storage_details():
    value = _SemanticOptions(np.float64(1.25), cache_generation=41)
    assert io.jsonable(value) == {"stiffness": 1.25}
    assert fes.jsonable(value) == {"stiffness": 1.25}
    assert io.jsonable(value, dataclasses="asdict") == {
        "stiffness": 1.25, "cache_generation": 41,
    }


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_strict_metadata_rejects_nested_numpy_nonfinite_values_without_replacing(tmp_path, bad):
    path = tmp_path / "manifest.json"
    path.write_text('{"previous": true}\n', encoding="utf-8")
    before = path.read_bytes()
    nested = {"outer": [np.array([[1.0, bad]]), {"scalar": np.float64(bad)}]}
    with pytest.raises(ValueError):
        io.write_json(path, nested)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("writer", [io.write_report_json, fes.write_json])
def test_numerical_reports_encode_missing_numpy_values_as_json_null(tmp_path, writer):
    path = tmp_path / "report.json"
    report = {
        "nested": [np.array([[1.0, np.nan], [np.inf, -np.inf]]),
                   {"scalar": np.float64(np.nan)}],
        "record": {"values": np.array([2.0, np.nan]), "path": Path("runs/水")},
    }
    writer(path, report)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "nested": [[[1.0, None], [None, None]], {"scalar": None}],
        "record": {"values": [2.0, None], "path": "runs/水"},
    }
    assert "NaN" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("stage", ["fsync", "replace"])
def test_json_write_failure_preserves_destination_and_removes_temporary(tmp_path, monkeypatch, stage):
    path = tmp_path / "state.json"
    io.write_json(path, {"step": 1})
    before = path.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(io.os, stage, interrupted)
    with pytest.raises(OSError, match="persistence failure"):
        io.write_json(path, {"step": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_json_writers_use_independent_temporaries_and_clean_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    io.write_json(path, {"writer": "previous"})
    replace = io.os.replace
    ready, lock, temporary_names = Barrier(2), Lock(), set()

    def coordinated_replace(source, destination):
        value = json.loads(Path(source).read_text(encoding="utf-8"))
        with lock:
            temporary_names.add(str(source))
        ready.wait(timeout=10)
        if value["writer"] == "interrupted":
            raise OSError("simulated interrupted writer")
        replace(source, destination)

    monkeypatch.setattr(io.os, "replace", coordinated_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(io.write_json, path, {"writer": "interrupted", "data": [1] * 40})
        success = pool.submit(io.write_json, path, {"writer": "complete", "data": [2] * 40})
        with pytest.raises(OSError, match="interrupted writer"):
            failed.result(timeout=15)
        success.result(timeout=15)
    assert len(temporary_names) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"writer": "complete", "data": [2] * 40}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("explicit_format", [False, True])
def test_metadata_npz_keeps_legacy_payload_encoding_kind_and_integrity(tmp_path, explicit_format):
    path = tmp_path / "parent.npz"
    metadata = {"step": np.int64(7), "options": _SemanticOptions(np.float64(1.25)),
                "source": Path("initial/state.npz")}
    expected_json = ('{"step": 7, "options": {"stiffness": 1.25, "cache_generation": 99}, '
                     '"source": "initial/state.npz"}')
    kind = "drpmd-arrays"
    if explicit_format:
        kind = "PyRPMD.drpmd.parent.v1"
        metadata["format"] = kind
        expected_json = expected_json[:-1] + ', "format": "PyRPMD.drpmd.parent.v1"}'
    coordinates = np.arange(18.0).reshape(2, 3, 3)
    save_metadata_npz(path, metadata, q=coordinates, step=7)
    with load_checkpoint(path, expected_kind=kind, require_integrity=True) as saved:
        assert saved.envelope["integrity"] == "verified"
        assert str(saved["metadata"].item()) == expected_json
        np.testing.assert_array_equal(saved["q"], coordinates)
        assert saved["step"].item() == 7
    # A valid ZIP with edited metadata still fails before consumers see it.
    with np.load(path, allow_pickle=False) as archive:
        damaged = {name: archive[name] for name in archive.files}
    damaged["metadata"] = np.asarray(expected_json.replace('"step": 7', '"step": 8'))
    np.savez_compressed(path, **damaged)
    with pytest.raises(CheckpointError, match="integrity check failed.*metadata"):
        with load_checkpoint(path, require_integrity=True):
            pytest.fail("tampered metadata reached the workflow")


@pytest.mark.parametrize("invalid,error", [(np.array([np.nan]), ValueError), (_ArrayOptions(), TypeError)])
def test_invalid_metadata_cannot_replace_previous_checkpoint(tmp_path, invalid, error):
    path = tmp_path / "parent.npz"
    save_metadata_npz(path, {"format": "PyRPMD.drpmd.parent.v1", "step": 7}, q=[1.0])
    before = path.read_bytes()
    with pytest.raises(error):
        save_metadata_npz(path, {"format": "PyRPMD.drpmd.parent.v1", "value": invalid}, q=[2.0])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_plain_dataclass_report_requires_explicit_storage_policy(tmp_path):
    path = tmp_path / "report.json"
    report = _NumericalReport(np.array([2.0, np.nan]), Path("runs/水"))
    with pytest.raises(TypeError):
        io.write_report_json(path, report)
    assert not path.exists()
    io.write_report_json(path, report, dataclasses="asdict")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "values": [2.0, None], "path": "runs/水",
    }


def test_plain_dataclass_keeps_legacy_stream_repr_and_manifest_rejection():
    config, system = _identity_problem("plain_dataclass")
    with pytest.raises(TypeError):
        io.fingerprint(config, system)
    assert fes._fingerprint(config, system, {"window": 2, "replica": 1}) == (
        "4e7a9216d89b83b6cdef3728eb994e9e63d7dd1e451ded35808b10b75770f37f"
    )
    opes_config = dict(config, task="opes", opes={"sigma_au": [0.3], "pace": 3,
                                                "barrier_au": 2.0, "bias_factor": 5.0})
    opes_config.pop("umbrella")
    assert opes._stream_fingerprint(opes_config, system, 1) == (
        "aa6d37e3975578362ec937fcf174d04bd53aaca86a87323c0d0b418923fdc18e"
    )


def test_custom_to_dict_retains_writer_specific_recursion(tmp_path):
    path = tmp_path / "record.json"
    value = _ArrayOptions()
    io.write_json(path, value)
    expected = {"weights": [1.0, 2.0], "nested": {"value": 1.25}}
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    before = path.read_bytes()
    # The historical report writer accepted only directly JSON-ready to_dict
    # output. Keeping this boundary also preserves stream fingerprint bytes.
    with pytest.raises(TypeError):
        fes.write_json(path, value)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_custom_to_dict_nonfinite_result_keeps_legacy_stream_identity_only(tmp_path):
    config, system = _identity_problem("missing_to_dict")
    with pytest.raises(ValueError):
        io.fingerprint(config, system)
    assert fes._fingerprint(config, system, {"window": 2, "replica": 1}) == (
        "7f9fd287a8f0eae8374f52d1e227f183984b4e3ef17fe8ec9dd6488063d16aae"
    )
    opes_config = dict(config, task="opes", opes={"sigma_au": [0.3], "pace": 3,
                                                "barrier_au": 2.0, "bias_factor": 5.0})
    opes_config.pop("umbrella")
    assert opes._stream_fingerprint(opes_config, system, 1) == (
        "2248fc674d2cf1d278fcf24cde94025bb8bad7c10b9cd73c0ced96c697abd85f"
    )
    # Supporting an old hash representation must not emit invalid report JSON.
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        fes.write_json(path, _MissingOptions())
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
