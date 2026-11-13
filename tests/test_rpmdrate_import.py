"""Real importer, parser and workflow tests; no mocked PyRPMD execution."""

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.rpmdrate_import import import_rpmdrate
from pyrpmd.runner import prepare
from pyrpmd.units import parse_quantity


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rpmdrate_import" / "input.py"


def convert(tmp_path, text=None, **options):
    source = tmp_path / "input.py"
    source.write_text(EXAMPLE.read_text() if text is None else text)
    return import_rpmdrate(source, temperature=1000, beads=4,
                          output=tmp_path / "converted", **options)


def assert_modules_restored(before):
    assert sys.modules.keys() == before.keys()
    assert all(sys.modules[key] is value for key, value in before.items())


def test_h3_identity_matches_independent_handwritten_reference(tmp_path):
    report = convert(tmp_path)
    converted, system = JobSpec.from_file(tmp_path / "converted/rate.inp").resolve()
    # Handwritten in the upstream atom order and orientation. No imported
    # reaction fields, geometry or distances are reused to build this oracle.
    reference, expected = JobSpec.from_mapping({
        "schema_version": 2, "task": "rate", "temperature": "1000 K", "beads": 4,
        "system": {"symbols": ["H", "H", "H"],
                   "geometry": [[0, 0, -1.757], [0, 0, 0], [0, 0, 1.757]]},
        "potential": {"provider": "rpmdrate", "module": "h3_bkmp2", "initializer": ""},
        "reaction": {"reactant_fragments": [[0, 1], [2]], "forming_bonds": [[1, 2]],
                     "breaking_bonds": [[0, 1]], "equivalent_paths": [[1, 0, 2]],
                     "rinf": "15.8754 angstrom", "symmetry_factor": 1.0},
        "umbrella": {"centers": [0, .5, 1], "force_constant": .0001},
    }).resolve()
    for field in ("fragments", "forming_bonds", "breaking_bonds", "equivalent_paths", "symmetry_factor"):
        assert converted["reaction"][field] == reference["reaction"][field]
    assert parse_quantity(converted["reaction"]["rinf"], "length") == pytest.approx(
        parse_quantity(reference["reaction"]["rinf"], "length"), rel=1e-15)
    np.testing.assert_array_equal(system.geometry, expected.geometry)
    assert report["unmapped"] == []
    assert len(report["calls"]) == 11
    assert all(record["targets"] for record in report["calls"])
    assert report["potential"]["initializer"] == ""


def test_independent_fortran_bias_and_window_budgets(tmp_path):
    text = EXAMPLE.read_text().replace("kforce=0.0001*T", "kforce=(1+xi)*0.1*T")
    report = convert(tmp_path, text)
    preview = prepare(JobSpec.from_file(tmp_path / "converted/rate.inp"), numerical=False)
    for record, window in zip(report["windows"], preview["umbrella_preparation"]["window_plan"]):
        kforce = (1 + record["xi"]) * 0.1 * 1000
        stiffness = window["stiffness_au"][0][0]
        assert stiffness == pytest.approx(kforce, rel=1e-15)
        # Independently evaluate the pinned Fortran per-bead potential and
        # gradient. Summing P beads is the dynamics' job, not the importer's.
        delta, dxi, beads = .13, -.27, 4
        assert beads * .5 * stiffness * delta**2 == pytest.approx(sum(.5 * kforce * delta**2 for _ in range(beads)))
        assert stiffness * delta * dxi == pytest.approx(kforce * delta * dxi)
        assert (window["chains"], window["equilibration_steps"], window["production_steps"],
                window["sample_stride_steps"]) == (1, 4, 20, 1)
    recrossing = preview["recrossing_preparation"]
    assert (recrossing["planned_configurations"], recrossing["planned_pairs"], recrossing["planned_trajectories"]) == (2, 2, 4)


@pytest.mark.parametrize("unit, factor", [("bohr", 1), ("angstrom", parse_quantity("1 angstrom", "length")),
                                          ("nm", parse_quantity("1 nm", "length"))])
def test_geometry_units_preserve_atom_order(tmp_path, unit, factor):
    report = convert(tmp_path, EXAMPLE.read_text().replace(']], "bohr")', f']], "{unit}")'))
    np.testing.assert_allclose(report["geometry_bohr"], np.array([[0, 0, -1.757], [0, 0, 0], [0, 0, 1.757]]) * factor,
                               rtol=1e-15, atol=0)


@pytest.mark.parametrize("old,new,message", [
    ("reactant2Atoms=[3]", "reactant2Atoms=[4]", "partition"),
    ("reactant2Atoms=[3]", "reactant2Atoms=[2]", "partition"),
    ("formingBonds=[(2, 3)]", "formingBonds=[(2, 4)]", "out-of-range"),
    ("formingBonds=[(2, 3)]", "formingBonds=[(2, 2)]", "self bond"),
    ("formingBonds=[(2, 3)]", "formingBonds=[(2, 3), (3, 2)]", "duplicate bond"),
    ("reactant2Atoms=[3]", "reactant2Atoms=[3.0]", "integer"),
    ("formingBonds=[(2, 3)]", "formingBonds=[(True, 3)]", "integer"),
    ('15.8754, "angstrom"', '15.8754, "furlong"', "unit"),
    (']], "bohr")', ']], "fs")', "dimension"),
    ("breakingBonds=[(2, 1)]", "breakingBonds=[(1, 2)]", "Inconsistent"),
    ("xi in [0.0, 0.5, 1.0]", "xi in [0.0, 0.0, 1.0]", "duplicate umbrella center"),
    ("kforce=0.0001*T", "kforce=-1", "positive"),
    ("childrenPerSampling=2", "childrenPerSampling=3", "even"),
    ("childTrajectories=4", "childTrajectories=5", "divide"),
    ('dt=(0.0001, "ps"), windows=windows', 'dt=(0.0001, "ps"), dt=(0.0001, "ps"), windows=windows', "keyword argument repeated"),
])
def test_invalid_input_creates_no_output(tmp_path, old, new, message):
    with pytest.raises((ValueError, SyntaxError), match=message):
        convert(tmp_path, EXAMPLE.read_text().replace(old, new), allow_unmapped=True)
    assert not (tmp_path / "converted").exists()


@pytest.mark.parametrize("append,message", [
    ('\nequivalentTransitionState(formingBonds=[(1,3)], breakingBonds=[(2,1)])', "Duplicate equivalent"),
    ('\nthermostat("Andersen")', "Duplicate thermostat"),
    ('\nsymmetry_factor=2', "symmetry_factor must be 1"),
    ('\ntask="drpmd"', "dRPMD"),
    ('\nimport rpmdrate', "rpmdrate package"),
    ('\nimport importlib', "Dynamic and wildcard"),
    ('\nfrom PES import *', "Dynamic and wildcard"),
    ('\nget_potential(numpy.zeros((3,3,4)))', "PES call forbidden"),
])
def test_rejected_extensions_restore_modules(tmp_path, append, message):
    before = sys.modules.copy()
    with pytest.raises(ValueError, match=message):
        convert(tmp_path, EXAMPLE.read_text() + append, allow_unmapped=True)
    assert_modules_restored(before)
    assert not (tmp_path / "converted").exists()


def test_collects_all_unmapped_settings_before_writing(tmp_path):
    text = EXAMPLE.read_text().replace('thermostat("Andersen", samplingTime=(0.0002, "ps"))',
                                      'thermostat("GLE", A=("absent.txt", "s^-1"))')
    text += '\ngenerateUmbrellaConfigurations(dt=(.0001,"ps"), evolutionTime=(5,"ps"), xi_list=[0,1], kforce=100)\n'
    text = text.replace("xi_min=0.0, xi_max=1.0", "xi_min=-0.02, xi_max=1.02")
    before = sys.modules.copy()
    with pytest.raises(InputError) as caught:
        convert(tmp_path, text)
    for setting in ("thermostat", "generateUmbrellaConfigurations", "xi_min/xi_max"):
        assert setting in str(caught.value)
    assert_modules_restored(before)
    assert not (tmp_path / "converted").exists()
    report = convert(tmp_path, text, allow_unmapped=True)
    assert not report["all_settings_mapped"]
    assert len(report["unmapped"]) == 3
    assert all(item["reason"] and item["replacement"] for item in report["unmapped"])
    assert (tmp_path / "converted/rate.inp").read_text().count("# UNMAPPED:") == 3
    JobSpec.from_file(tmp_path / "converted/rate.inp").resolve()
    assert_modules_restored(before)


def test_default_andersen_clock_is_not_silently_changed(tmp_path):
    text = EXAMPLE.read_text().replace(', samplingTime=(0.0002, "ps")', '')
    with pytest.raises(InputError, match="floor\\(sqrt"):
        convert(tmp_path, text)
    report = convert(tmp_path, text, allow_unmapped=True)
    assert report["unmapped"][0]["replacement"]["umbrella.sampling.thermostat"]["refresh_every"] == {"steps": 10}


@pytest.mark.parametrize("suffix", ["", "\nraise RuntimeError('input failure')"])
def test_restores_imported_removed_and_replaced_module_bindings(tmp_path, suffix):
    text = EXAMPLE.read_text() + '''
import sys, fractions
sys.modules['t52_added_module'] = fractions
del sys.modules['json']
sys.modules['math'] = fractions
sys.path.append('t52-temporary-path')
''' + suffix
    registry, before, paths = sys.modules, sys.modules.copy(), sys.path[:]
    if suffix:
        with pytest.raises(RuntimeError, match="input failure"):
            convert(tmp_path, text)
    else:
        convert(tmp_path, text)
    assert_modules_restored(before)
    assert sys.modules is registry
    assert sys.path == paths


def test_real_pes_module_is_not_imported_or_called(tmp_path, monkeypatch):
    sentinel = tmp_path / "PES_WAS_IMPORTED"
    pes = tmp_path / "h3_bkmp2.py"
    pes.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\nraise RuntimeError('PES loaded')\n")
    existing = ModuleType("h3_bkmp2")
    existing.get_potential = lambda *args: pytest.fail("PES was called")
    monkeypatch.setitem(sys.modules, "h3_bkmp2", existing)
    before = sys.modules.copy()
    report = convert(tmp_path)
    assert report["potential"]["module"] == str(pes)
    assert not sentinel.exists()
    assert_modules_restored(before)


@pytest.mark.parametrize("statement", ["from PES import get_potential, initialize_potential",
                                       "import PES\nget_potential = PES.get_potential\ninitialize_potential = PES.initialize_potential",
                                       "import package.PES as pes\nget_potential = pes.get_potential\ninitialize_potential = pes.initialize_potential"])
def test_pes_import_styles_and_initializer_are_recorded(tmp_path, statement):
    text = EXAMPLE.read_text().replace("from h3_bkmp2 import get_potential", statement)
    report = convert(tmp_path, text, pes_module=tmp_path / "missing.so")
    assert report["potential"]["initializer"] == "initialize_potential"
    assert report["potential"]["module"] == str(tmp_path / "missing.so")
    assert report["pes_imports"]


def test_initializer_call_is_rejected(tmp_path):
    text = EXAMPLE.read_text().replace("from h3_bkmp2 import get_potential",
                                      "from PES import get_potential, initialize_potential\ninitialize_potential()")
    with pytest.raises(InputError, match="PES call forbidden"):
        convert(tmp_path, text)


@pytest.mark.parametrize("temperature,beads", [(0, 4), (float("nan"), 4), (True, 4),
                                               (1000, 0), (1000, 4.0), (1000, True)])
def test_invalid_command_parameters_precede_execution(tmp_path, temperature, beads):
    source = tmp_path / "input.py"
    source.write_text("raise AssertionError('should not execute')")
    with pytest.raises(ValueError):
        import_rpmdrate(source, temperature=temperature, beads=beads, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_nonempty_destination_is_rejected_before_executing_input(tmp_path):
    output = tmp_path / "converted"
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(InputError, match="empty"):
        convert(tmp_path, "raise AssertionError('should not execute')")
    assert (output / "keep").read_text() == "existing"


def test_empty_existing_destination_stays_empty_after_rejection(tmp_path):
    (tmp_path / "converted").mkdir()
    with pytest.raises(InputError, match="floor\\(sqrt"):
        convert(tmp_path, EXAMPLE.read_text().replace(', samplingTime=(0.0002, "ps")', ''))
    assert list((tmp_path / "converted").iterdir()) == []


def test_real_cli_import_validate_and_workflow_dispatch(tmp_path):
    output = tmp_path / "converted"
    def cli(*args):
        result = subprocess.run([sys.executable, "-m", "pyrpmd", *map(str, args)],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)
    imported = cli("import-rpmdrate", EXAMPLE, "--temperature", 1000, "--beads", 4, "--output", output)
    assert imported["unmapped"] == []
    valid = cli("validate", output / "rate.inp")
    assert valid["valid"] and valid["task"] == "rate" and valid["natoms"] == 3
    preview = cli("prepare", output / "rate.inp")
    assert preview["umbrella_preparation"]["windows"] == 3
    assert preview["recrossing_preparation"]["planned_trajectories"] == 4
    assert not preview["ready"]  # Structural validation does not certify a saddle.
    assert not (output / "runs").exists()


def test_real_cli_unmapped_error_writes_nothing(tmp_path):
    source = tmp_path / "input.py"
    source.write_text(EXAMPLE.read_text().replace(', samplingTime=(0.0002, "ps")', ''))
    result = subprocess.run([sys.executable, "-m", "pyrpmd", "import-rpmdrate", str(source),
                             "--temperature", "1000", "--beads", "4", "--output", str(tmp_path / "out")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "samplingTime" in result.stderr
    assert not (tmp_path / "out").exists()
