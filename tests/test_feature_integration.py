"""End-to-end regressions for the materials/surface integration boundary."""

import copy
import json

import numpy as np
import pytest

from pyrpmd import System, resume_run, run
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.cv import make_cv
from pyrpmd.dynamics import Dynamics
from pyrpmd.potentials import PeriodicCosinePotential, PotentialError, make_potential


@pytest.mark.parametrize("explicit_pbc", [False, True])
def test_nonperiodic_cell_fes_resumes_both_legacy_identities_without_propagation(
    explicit_pbc, tmp_path, monkeypatch
):
    """Exercise the manifest and per-window hashes, not only System roundtrip."""
    saved_system = {
        "symbols": ["Cu", "H"], "masses": [20.0, 2.0],
        "geometry": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        "fragments": [], "charge": 0, "multiplicity": 1,
        "cell": np.diag([5.0, 6.0, 9.0]).tolist(), "fixed_atoms": [0],
    }
    if explicit_pbc:
        saved_system["pbc"] = [False, False, False]
    job = {
        "task": "fes", "beads": 2, "temperature": 1 / (2 * KB_AU),
        "seed": 431, "native": False,
        "system": System.from_dict(saved_system),
        "potential": {"provider": "free"},
        "dynamics": {"timestep": 0.03, "loop_backend": "python"},
        "sampling": {"equilibration_steps": 3, "production_steps": 12,
                     "interval_steps": 2, "andersen_interval": 3},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 1, "axis": 0}],
        "umbrella": {"centers": [-0.1, 0.1], "stiffness": 2.0},
        "analysis": {"bins": 4, "ranges": [[-0.4, 0.4]], "bootstrap": 0},
        "output": {"checkpoint_steps": 3},
    }
    output = tmp_path / "legacy-fes"
    initial = run(job, output=output)
    identity_file = output / "input" / "system.json"
    assert json.loads(identity_file.read_text()) == saved_system
    manifest = (output / "manifest.json").read_bytes()
    checkpoints = sorted((output / "checkpoints").glob("*.npz"))
    assert len(checkpoints) == 2
    payloads = {}
    for path in checkpoints:
        with load_checkpoint(path, require_integrity=True) as payload:
            payloads[path] = {key: payload[key].copy() for key in payload.files}

    def no_new_step(*args, **kwargs):
        raise AssertionError("a completed legacy checkpoint must not propagate")

    monkeypatch.setattr(Dynamics, "step", no_new_step)
    monkeypatch.setattr(Dynamics, "propagate", no_new_step)
    resumed = resume_run(output, execution={"loop_backend": "python", "block_steps": 7})
    assert resumed["task"] == initial["task"] == "fes"
    assert (output / "manifest.json").read_bytes() == manifest
    assert json.loads(identity_file.read_text()) == saved_system
    for path in checkpoints:
        with load_checkpoint(path, require_integrity=True) as payload:
            for key, value in payloads[path].items():
                np.testing.assert_array_equal(payload[key], value)

    # Compatibility must not weaken the scientific identity. Changing either
    # the cell or boundary flags still rejects a completed run before any step.
    for changed_field in ("cell", "pbc"):
        changed = copy.deepcopy(saved_system)
        if changed_field == "cell":
            changed["cell"][0][0] += 0.5
        else:
            changed["pbc"] = [True, False, False]
        identity_file.write_text(json.dumps(changed))
        with pytest.raises(RuntimeError, match="InputError: Cannot resume with changed scientific settings or system"):
            resume_run(output)
        assert (output / "manifest.json").read_bytes() == manifest
    identity_file.write_text(json.dumps(saved_system))


@pytest.mark.parametrize(
    "plain_kind, periodic_kind, atoms",
    [
        ("angle", "periodic_angle", [1, 0, 2]),
        ("bond_difference", "periodic_bond_difference", [1, 0, 2]),
    ],
)
def test_coordinate_image_choice_is_explicit_for_materials_and_fixed_slabs(
    plain_kind, periodic_kind, atoms
):
    """Adding a fixed substrate cannot change the meaning of an existing CV."""
    coordinates = np.array([[0.2, 0.0, 0.0], [3.8, 0.2, 0.0], [0.4, 1.0, 0.0]])
    values = []
    for fixed_atoms in ((), (2,)):
        system = System(
            ("H", "O", "Cu"), [1837.0, 29156.0, 115800.0], coordinates,
            cell=np.diag([4.0, 4.0, 8.0]), pbc=(True, True, False),
            fixed_atoms=fixed_atoms,
        )
        q = np.repeat(coordinates[None], 4, axis=0)
        image = q.copy()
        image[:, 1] += system.cell[0]
        plain = make_cv({"kind": plain_kind, "atoms": atoms}, system)
        periodic = make_cv({"kind": periodic_kind, "atoms": atoms}, system)
        assert not np.allclose(plain.value(q), periodic.value(q))
        assert not np.allclose(plain.value(q), plain.value(image))
        np.testing.assert_allclose(periodic.value(q), periodic.value(image), atol=1e-13)
        values.append((plain.value(q), periodic.value(q)))
        np.testing.assert_array_equal(q, np.repeat(coordinates[None], 4, axis=0))
    np.testing.assert_allclose(values[0], values[1], atol=1e-13)


@pytest.mark.parametrize("changed_field", ["cell", "pbc"])
def test_preconfigured_periodic_pes_rejects_different_system_boundaries(changed_field):
    """A preconfigured lattice model must not keep another System's cell."""
    cell = np.diag([4.0, 5.0, 9.0])
    pbc = [True, True, False]
    provider = PeriodicCosinePotential(cell=cell, pbc=pbc)
    system = System(("H",), [1837.0], [[0.1, 0.2, 0.3]], cell=cell, pbc=pbc)
    assert make_potential(provider, system) is provider
    changed = system.to_dict()
    if changed_field == "cell":
        changed["cell"][0][0] += 0.5
    else:
        changed["pbc"] = [True, False, False]
    with pytest.raises(PotentialError, match="cell/PBC do not match"):
        make_potential(provider, System.from_dict(changed))
