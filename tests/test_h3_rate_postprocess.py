"""H3 setup and actual estimator interfaces; expensive PES runs are opt-in."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.cv import CVStack
from pyrpmd.input import InputError, JobSpec
from pyrpmd.workflows.fes import _flux_coordinate, resolve_umbrellas, _write_fes
from pyrpmd.analysis.fes import reconstruct_fes


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("h3_rate_postprocess_benchmark", ROOT / "benchmarks/h3/rate_postprocess.py")
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_h3_joint_configuration_has_real_asymptotic_reference(tmp_path):
    config, rate_config, system = benchmark.configurations(tmp_path)
    cv, _, centers, _ = resolve_umbrellas(config, system)
    assert cv.dimension == 2
    assert config["analysis"]["measure"] == "rate_flux"
    assert config["analysis"]["flux_axis"] == 0
    assert min(centers[:, 0]) < 0 and max(centers[:, 0]) > 1
    assert rate_config["reaction"]["rinf"] == 10.0
    assert len(cv.cvs[0].permutations) == 2
    assert rate_config["reaction"]["symmetry_factor"] == 1
    assert benchmark.reference_rate(rate_config, system) > 0


@pytest.mark.parametrize("axis", [None, 1, -1, True, 0.0])
def test_joint_flux_measure_requires_explicit_integer_first_axis(tmp_path, axis):
    config, _, system = benchmark.configurations(tmp_path)
    bad = copy.deepcopy(config)
    if axis is None:
        bad["analysis"].pop("flux_axis")
    else:
        bad["analysis"]["flux_axis"] = axis
    with pytest.raises(ValueError, match="flux_axis"):
        resolve_umbrellas(bad, system)


def test_joint_flux_measure_rejects_arbitrary_first_cv_and_density(tmp_path):
    config, _, system = benchmark.configurations(tmp_path)
    cv, *_ = resolve_umbrellas(config, system)
    with pytest.raises(ValueError, match="first scalar centroid RPMDrate"):
        _flux_coordinate(config, CVStack(cv.cvs[::-1]))
    config["analysis"]["measure"] = "cv_density"
    with pytest.raises(ValueError, match="requires measure"):
        _flux_coordinate(config, cv)


@pytest.mark.parametrize("axis", [1, True, 0.0])
def test_input_rejects_ambiguous_flux_axis(axis):
    spec = JobSpec.from_file(ROOT / "examples/fes/harmonic2d.inp")
    spec.data["analysis"].update(measure="rate_flux", flux_axis=axis)
    with pytest.raises(InputError, match="flux_axis"):
        spec.resolve()


def test_h3_exit_labels_partition_union_and_permute():
    frames = np.array([
        [[0., 0., 0.], [1.4, 0., 0.], [8., 0., 0.]],
        [[0., 0., 0.], [8., 0., 0.], [1.4, 0., 0.]],
        [[-8., 0., 0.], [0., 0., 0.], [1.4, 0., 0.]],
    ])
    side = np.array([True, True, False])
    indicators = benchmark.channel_indicators(frames, side)
    np.testing.assert_array_equal(indicators, [[True, False], [False, True], [False, False]])
    np.testing.assert_array_equal(indicators.sum(1), side)
    np.testing.assert_array_equal(benchmark.channel_indicators(frames[:, [0, 2, 1]], side), indicators[:, ::-1])


def test_flux_metadata_survives_npz_hdf5_and_summary(tmp_path):
    config, _, system = benchmark.configurations(tmp_path)
    cv, *_ = resolve_umbrellas(config, system)
    rng = np.random.default_rng(41)
    samples = rng.normal(size=(400, 2)) * .1 + [0.5, 1.55]
    stiffness = np.eye(2) * .05
    data = reconstruct_fes([samples], [[.5, 1.55]], stiffness, config["beta"], bins=[2, 2], bootstrap=0)
    data["measure"] = "rate_flux"
    stream = dict(samples=samples, center=np.array([.5, 1.55]), stiffness=stiffness,
                  energies=np.zeros(len(samples)), window=0, replica=0)
    summary = _write_fes(tmp_path, data, [stream], cv, config)
    with np.load(tmp_path / "fes.npz", allow_pickle=False) as saved:
        assert str(saved["measure"]) == "rate_flux"
        assert int(saved["flux_axis"]) == 0
        assert float(saved["beta"]) == config["beta"]
    assert summary["flux_axis"] == 0
    import h5py
    with h5py.File(tmp_path / "fes.h5") as saved:
        assert saved.attrs["measure"] == "rate_flux"
        assert saved.attrs["flux_axis"] == 0


def test_h3_same_sample_oracle_respects_nonuniform_transverse_width():
    xedges = np.array([-.025, .025, .975, 1.025])
    yedges = np.array([0., 1.55, 4.])
    samples = np.array([[0., 1.3], [0., 1.8], [1., 1.3], [1., 1.8]])
    weights = np.array([.5, .3, .08, .12])
    mass, _ = np.histogramdd(samples, bins=[xedges, yedges], weights=weights)
    density = mass / np.diff(xedges)[:, None] / np.diff(yedges)[None, :]
    data = dict(edges=[xedges, yedges], weights=weights, density=density)
    result = benchmark.independent_marginal_check(data, [dict(samples=samples)], beta=200., reference_rate_au=2.)
    assert result["k_tst_au"] == pytest.approx(.5)
    assert result["maximum_density_marginal_error"] < 1e-12
    assert result["transverse_sample_fraction_outside"] == 0


def test_h3_surface_identity_binds_reaction_and_pes_binary(tmp_path):
    config, rate, system = benchmark.configurations(tmp_path)
    library = tmp_path / "bkmp2.so"
    library.write_bytes(b"first checked test binary")
    rate["potential"]["params"]["library"] = str(library)
    source = json.loads((ROOT / "benchmarks/h3/source.json").read_text())
    build = dict(library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                 source=source, wrapper_sha256="test-wrapper")
    library.with_suffix(".build.json").write_text(json.dumps(build))
    original, payload = benchmark.surface_identity(rate, system)
    assert payload["potential"]["pinned_source_sha256"] == source["sha256"]
    changed = copy.deepcopy(rate)
    changed["reaction"]["rinf"] = 12.0
    assert benchmark.surface_identity(changed, system)[0] != original
    changed["reaction"]["rinf"] = rate["reaction"]["rinf"]
    changed["reaction"]["equivalent_paths"] = []
    assert benchmark.surface_identity(changed, system)[0] != original
    library.write_bytes(b"different binary")
    with pytest.raises(ValueError, match="provenance"):
        benchmark.surface_identity(rate, system)
    build["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    library.with_suffix(".build.json").write_text(json.dumps(build))
    assert benchmark.surface_identity(rate, system)[0] != original
