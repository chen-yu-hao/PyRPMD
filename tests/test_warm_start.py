"""Warm start: a new run initialized from an older run's configuration."""
import json

import numpy as np
import pytest

from pyrpmd.bias import MetadynamicsBias
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.cv import CVStack, Cartesian
from pyrpmd.input import InputError
from pyrpmd.runner import resume_run, run
from pyrpmd.warm_start import check_supported, interpolate_beads


def metad_input(*, beads=4, temperature="300 K", mode="well_tempered", bias_factor=6.0,
                sigma="0.2 bohr", estimator="centroid", x=-1.0, neq=20, nprod=60,
                timestep="0.1 fs"):
    factor = f"bias_factor = {bias_factor}\n" if mode == "well_tempered" else ""
    return f"""
task = "fes"
temperature = "{temperature}"
beads = {beads}
seed = 711

[system]
symbols = ["H"]
geometry = [[{x}, 0.0, 0.0]]

[potential]
provider = "builtin"
name = "double_well"
params = {{height = 0.001, width = 1.0, confinement = 0.002}}

[dynamics]
timestep = "{timestep}"
loop_backend = "python"

[sampling]
equilibration_steps = {neq}
production_steps = {nprod}
sample_interval = 1
andersen_interval = 10

[[cvs]]
name = "x"
kind = "cartesian"
atom = 0
axis = "x"
estimator = "{estimator}"
unit = "bohr"

[fes]
method = "metadynamics"

[metadynamics]
cvs = ["x"]
mode = "{mode}"
{factor}sigma = ["{sigma}"]
height = "0.02 kcal/mol"
pace = 10

[analysis]
method = "bias_estimate"
bins = 10
ranges = [[-2.0, 2.0]]
bootstrap = 0

[output]
checkpoint_steps = 20
"""


def write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def parent(tmp_path):
    source = write(tmp_path / "parent.inp", metad_input())
    run(source, output=tmp_path / "parent")
    return tmp_path / "parent"


def test_fourier_bead_interpolation_preserves_centroid_and_round_trips():
    rng = np.random.default_rng(3)
    for old in (1, 2, 3, 4, 5, 8):
        q = rng.normal(size=(old, 2, 3))
        np.testing.assert_array_equal(interpolate_beads(q, old), q)
        for new in (1, 2, 3, 6, 7, 16, 32):
            resampled = interpolate_beads(q, new)
            assert resampled.shape == (new, 2, 3)
            np.testing.assert_allclose(resampled.mean(axis=0), q.mean(axis=0), atol=1e-13)
            if new >= old:
                np.testing.assert_allclose(interpolate_beads(resampled, old), q, atol=1e-12)
    # A band-limited ring is reproduced exactly on the finer grid.
    t = np.arange(4) / 4
    q = np.zeros((4, 1, 3))
    q[:, 0, 0] = 0.3 + np.cos(2 * np.pi * t) + 0.2 * np.sin(2 * np.pi * t)
    fine = np.arange(12) / 12
    expected = 0.3 + np.cos(2 * np.pi * fine) + 0.2 * np.sin(2 * np.pi * fine)
    np.testing.assert_allclose(interpolate_beads(q, 12)[:, 0, 0], expected, atol=1e-13)


def test_initial_hills_are_bias_but_not_own_history():
    cv = CVStack([Cartesian(0, 0)])
    initial = [{"center": [0.0], "height": 0.3}, {"center": [0.5], "height": 0.1}]
    seeded = MetadynamicsBias(cv, [0.4], 0.2, 2.0, bias_factor=4.0, initial_hills=initial)
    reference = MetadynamicsBias(cv, [0.4], 0.2, 2.0, bias_factor=4.0)
    reference._set_initial_hills(None)
    for i, hill in enumerate(initial):
        reference._append({**hill, "step": i + 1, "time": float(i + 1)})
    values = np.linspace(-1, 1, 7)[:, None]
    np.testing.assert_allclose(seeded.physical_energy(values), reference.physical_energy(values))
    assert seeded.hill_count == 0 and seeded.hills == []
    assert seeded.initial_hills == initial
    # Well-tempered tempering sees the imported bias.
    hill = seeded.deposit([0.0], step=1, time=0.1)
    before = 0.3 + 0.1 * np.exp(-0.5 * (0.5 / 0.4) ** 2)
    assert hill["height"] == pytest.approx(0.2 * np.exp(-2.0 * before / 3.0))
    assert seeded.hill_count == 1
    assert len(seeded.native_descriptor()["heights"]) == 3


def test_low_to_high_precision_metadynamics_with_rescaled_bias(tmp_path, parent):
    source = write(tmp_path / "child.inp", metad_input(beads=8, temperature="250 K", bias_factor=10.0,
                                                       timestep="0.05 fs"))
    child = tmp_path / "child"
    summary = run(source, output=child, init_from=parent, init_bias=True)
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    manifest = json.loads((child / "manifest.json").read_text())
    origin = manifest["initialized_from"]
    assert origin["physics_fingerprint"] == parent_manifest["physics_fingerprint"]
    assert origin["checkpoint"] == "checkpoints/metadynamics_0000.npz"
    assert origin["beads"] == 4 and origin["run"] == str(parent.resolve())
    assert manifest["physics_fingerprint"] != parent_manifest["physics_fingerprint"]
    marker = summary["warm_start"]
    assert marker["warm_start"] is True
    assert marker["bead_interpolation"] == "ring_polymer_fourier"
    assert marker["momenta"] == "resampled_maxwell" and marker["equilibration"] == "required"
    bias = marker["initial_bias"]
    assert bias["role"] == "initial_bias" and bias["temperature_changed"] is True
    assert bias["scale"] == pytest.approx((9 / 10) / (5 / 6))
    with load_checkpoint(parent / "checkpoints" / "metadynamics_0000.npz") as data:
        parent_hills = json.loads(str(data["hills"]))
    assert bias["hills"] == len(parent_hills)
    assert summary["initial_bias_hills"] == len(parent_hills)
    # The child's own history contains only its own depositions.
    assert summary["hills_per_replica"] == [6]
    resolved = json.loads((child / "input" / "resolved.json").read_text())
    stored = resolved["warm_start"]["initial_bias"]["hills"]
    np.testing.assert_allclose([h["height"] for h in stored],
                               [h["height"] * bias["scale"] for h in parent_hills])
    again = resume_run(child)
    assert again["warm_start"] == marker
    assert again["hills_per_replica"] == summary["hills_per_replica"]


def test_imported_configuration_replaces_the_input_geometry(tmp_path, parent):
    with load_checkpoint(parent / "checkpoints" / "metadynamics_0000.npz") as data:
        centroid = float(np.mean(data["q"][:, 0, 0]))
    assert centroid < -0.3
    # The new input geometry sits in the other well.
    source = write(tmp_path / "child.inp", metad_input(beads=6, x=1.0, neq=1, nprod=10))
    child = tmp_path / "child"
    run(source, output=child, init_from=parent / "checkpoints" / "metadynamics_0000.npz")
    with load_checkpoint(child / "checkpoints" / "metadynamics_0000.npz") as data:
        first = float(data["samples"][0, 0])
    assert abs(first - centroid) < 0.1
    fresh = tmp_path / "fresh"
    run(source, output=fresh)
    with load_checkpoint(fresh / "checkpoints" / "metadynamics_0000.npz") as data:
        assert float(data["samples"][0, 0]) > 0.8


def test_warm_start_rejections(tmp_path, parent):
    def attempt(name, text, **kwargs):
        source = write(tmp_path / f"{name}.inp", text)
        with pytest.raises(InputError) as error:
            run(source, output=tmp_path / name, init_from=parent, **kwargs)
        assert not (tmp_path / name / "manifest.json").exists()
        return str(error.value)

    assert "equilibration" in attempt("noeq", metad_input(neq=0))
    assert "well-tempered" in attempt("standard", metad_input(mode="standard", temperature="250 K"),
                                      init_bias=True)
    assert "identical metadynamics CVs" in attempt("cv", metad_input(estimator="bead_average"),
                                                   init_bias=True)
    assert "Gaussian widths" in attempt("sigma", metad_input(sigma="0.3 bohr"), init_bias=True)
    with pytest.raises(InputError, match="cannot be combined with resume"):
        run(write(tmp_path / "r.inp", metad_input()), output=tmp_path / "r", resume=True,
            init_from=parent)
    with pytest.raises(InputError, match="new output directory"):
        run(write(tmp_path / "same.inp", metad_input()), output=parent, init_from=parent)
    with pytest.raises(InputError, match="require init_from"):
        run(write(tmp_path / "bias.inp", metad_input()), output=tmp_path / "bias", init_bias=True)
    for task in ("rate", "drpmd", "nonadiabatic"):
        with pytest.raises(InputError, match="not supported"):
            check_supported({"task": task})


def test_same_temperature_standard_bias_import_uses_asymptotic_ratio(tmp_path, parent):
    source = write(tmp_path / "std.inp", metad_input(mode="standard"))
    summary = run(source, output=tmp_path / "std", init_from=parent, init_bias=True)
    assert summary["warm_start"]["initial_bias"]["scale"] == pytest.approx(1 / (5 / 6))
    assert summary["warm_start"]["initial_bias"]["temperature_changed"] is False


def test_umbrella_warm_start_requires_one_checkpoint(tmp_path):
    text = metad_input()
    text = text.replace('[fes]\nmethod = "metadynamics"', '[fes]\nmethod = "umbrella"')
    text = text.split("[metadynamics]")[0] + """
[umbrella]
cvs = ["x"]
centers = [-0.5, 0.5]
stiffness = "0.002 hartree/bohr^2"

[analysis]
bins = 8
ranges = [[-2.0, 2.0]]
bootstrap = 0

[output]
checkpoint_steps = 20
"""
    parent_run = tmp_path / "upar"
    run(write(tmp_path / "u.inp", text), output=parent_run)
    with pytest.raises(InputError, match="several checkpoints"):
        run(write(tmp_path / "u6.inp", text.replace("beads = 4", "beads = 6")),
            output=tmp_path / "u6", init_from=parent_run)
    summary = run(tmp_path / "u6.inp", output=tmp_path / "u6b",
                  init_from=parent_run / "checkpoints" / "window_00001_replica_0000.npz")
    assert summary["warm_start"]["initialized_from"]["checkpoint"] == (
        "checkpoints/window_00001_replica_0000.npz")
    assert summary["warm_start"]["beads"] == 6
