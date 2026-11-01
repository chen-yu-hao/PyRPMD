"""Real LAMMPS unit conversions, atom ordering and reciprocal-space virials."""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU, KCAL_MOL_TO_AU
from pyrpmd.potentials import BackendUnavailableError, PotentialError, check_potential, make_potential
from pyrpmd.potentials.lammps import LAMMPSPotential


def _script(tmp_path, *, units="metal", periodic=True, extra="", charged=False, triclinic=False):
    pytest.importorskip("lammps")
    path = tmp_path / "model.in"
    path.write_text(f"""units {units}
atom_style {'charge' if charged else 'atomic'}
boundary {'p p p' if periodic else 'f f f'}
region box {'prism 0 20 0 20 0 20 1 0.5 0.2' if triclinic else 'block 0 20 0 20 0 20'}
create_box 1 box
create_atoms 1 single 2 2 2
create_atoms 1 single 3.3 2 2
mass 1 1
{"set atom 1 charge 1" if charged else ""}
{"set atom 2 charge -1" if charged else ""}
pair_style {'lj/cut/coul/long' if charged else 'lj/cut'} 5
pair_coeff 1 1 0.2 1
{"kspace_style ewald 1e-8" if charged else ""}
{extra}
""")
    return path


def _system(periodic=True, triclinic=False):
    cell = np.array([[20., 0, 0], [1., 20, 0], [.5, .2, 20]]) if triclinic else np.eye(3)*20
    return {"symbols": ["H", "D"], "cell": cell/BOHR_ANGSTROM, "pbc": [periodic]*3}


def test_missing_lammps_has_actionable_error():
    with patch.dict(sys.modules, {"lammps": None}):
        with pytest.raises(BackendUnavailableError, match="matching lammps"):
            LAMMPSPotential(["H"], "missing.in", type_map=["H"])


@pytest.mark.parametrize("units,energy_factor", [("metal", EV_TO_AU), ("real", KCAL_MOL_TO_AU)])
def test_actual_lammps_lj_units_batch_virial_and_reuse(tmp_path, units, energy_factor):
    path = _script(tmp_path, units=units)
    q = np.array([[[2., 2, 2], [3.3, 2, 2]], [[2., 2, 2], [3.5, 2, 2]]])/BOHR_ANGSTROM
    spec = {"provider": "lammps", "model": str(path), "params": {"type_map": ["H"]}}
    with make_potential(spec, _system()) as pes:
        engine = pes.engine
        result = pes.evaluate(q)
        r = (q[:, 1, 0]-q[:, 0, 0])*BOHR_ANGSTROM
        s6 = r**-6
        expected_force = 24*.2/r*(s6-2*s6*s6)
        np.testing.assert_allclose(result.energy, 4*.2*(s6*s6-s6)*energy_factor, atol=1e-14)
        np.testing.assert_allclose(result.forces[:, 0, 0], expected_force*energy_factor*BOHR_ANGSTROM, atol=1e-14)
        np.testing.assert_allclose(result.forces[:, 1, 0], -expected_force*energy_factor*BOHR_ANGSTROM, atol=1e-14)
        np.testing.assert_allclose(result.virial[:, 0, 0], -r*expected_force*energy_factor, atol=1e-14)
        assert check_potential(pes, q)["max_absolute_force_error"] < 2e-11
        np.testing.assert_allclose(pes.evaluate(q[::-1]).energy, result.energy[::-1], atol=1e-15)
        assert pes.engine is engine
        assert pes.metadata["unit_style"] == units
        assert pes.metadata["kspace_style"] == "none"
    assert pes.engine is None
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(q)


def test_actual_lammps_triclinic_wrapping_does_not_mutate_diffusion_path(tmp_path):
    path = _script(tmp_path, triclinic=True)
    system = _system(triclinic=True)
    q = np.array([[[2., 2, 2], [3.3, 2, 2]]])/BOHR_ANGSTROM
    with LAMMPSPotential(system["symbols"], path, type_map=["H"], cell=system["cell"]) as pes:
        reference = pes.evaluate(q)
        q[:, 0] += 3*system["cell"][0]-system["cell"][2]
        before = q.copy()
        result = pes.evaluate(q)
        np.testing.assert_array_equal(q, before)
        np.testing.assert_allclose(result.energy, reference.energy, atol=1e-14)
        np.testing.assert_allclose(result.forces, reference.forces, atol=1e-14)


def test_actual_lammps_ewald_forces_and_virial_follow_cell_energy_derivative(tmp_path):
    path = _script(tmp_path, charged=True)
    system = _system()
    q = np.array([[[2., 2, 2], [3.3, 2, 2]]])/BOHR_ANGSTROM
    with LAMMPSPotential(system["symbols"], path, type_map=["H"], cell=system["cell"]) as pes:
        result = pes.evaluate(q)
        assert pes.metadata["kspace_style"] == "ewald"
        assert check_potential(pes, q, epsilon=1e-4)["max_absolute_force_error"] < 1e-7
        h = 2e-5
        pes.set_cell(system["cell"]*(1+h))
        ep = pes.evaluate(q*(1+h)).energy.copy()
        pes.set_cell(system["cell"]*(1-h))
        em = pes.evaluate(q*(1-h)).energy.copy()
        np.testing.assert_allclose(-(ep-em)/(2*h), np.trace(result.virial, axis1=1, axis2=2), rtol=2e-5, atol=2e-7)


def test_actual_lammps_reorders_local_atoms_by_global_id(tmp_path):
    path = _script(tmp_path)
    system = _system()
    q = np.array([[[2., 2, 2], [3.3, 2, 2]]])/BOHR_ANGSTROM
    with LAMMPSPotential(system["symbols"], path, type_map=["H"], cell=system["cell"]) as pes:
        before = pes.evaluate(q)
        # Reverse all atom arrays for identical species, mimicking a neighbor
        # sort. The public coordinates/forces must still use atom ID order.
        for name in ("id", "type", "x", "v", "f", "mask", "image"):
            values = pes.engine.numpy.extract_atom(name)
            values[:2] = values[:2][::-1].copy()
        after = pes.evaluate(q)
        np.testing.assert_allclose(after.energy, before.energy, atol=1e-15)
        np.testing.assert_allclose(after.forces, before.forces, atol=1e-15)


def test_actual_lammps_rejects_wrong_species_cell_units_and_fixes(tmp_path):
    system = _system()
    path = _script(tmp_path)
    with pytest.raises(PotentialError, match="symbols/order"):
        LAMMPSPotential(["O", "H"], path, type_map=["H"], cell=system["cell"])
    with pytest.raises(PotentialError, match="System.cell"):
        LAMMPSPotential(system["symbols"], path, type_map=["H"], cell=system["cell"]*1.1)
    with pytest.raises(PotentialError, match="periodicity"):
        LAMMPSPotential(system["symbols"], path, type_map=["H"], pbc=[False]*3)
    path = _script(tmp_path, units="lj")
    with pytest.raises(PotentialError, match="real or metal"):
        LAMMPSPotential(system["symbols"], path, type_map=["H"], cell=system["cell"])
    path = _script(tmp_path, extra="fix integration all nve")
    with pytest.raises(PotentialError, match="cannot contain fixes"):
        LAMMPSPotential(system["symbols"], path, type_map=["H"], cell=system["cell"])


def test_actual_lammps_nonperiodic_box_checks_and_lifecycle(tmp_path):
    path = _script(tmp_path, periodic=False)
    system = _system(periodic=False)
    q = np.array([[[2., 2, 2], [3.3, 2, 2]]])/BOHR_ANGSTROM
    with make_potential({"provider": "lammps", "model": str(path), "params": {"type_map": ["H"]}}, system) as pes:
        assert np.isfinite(pes.evaluate(q).energy).all()
        q[0, 0, 0] = -1
        with pytest.raises(PotentialError, match="nonperiodic box"):
            pes.evaluate(q)
        with pytest.raises(PotentialError, match="full periodicity"):
            pes.set_cell(system["cell"]*1.1)
    pes.close()  # Idempotent; never finalizes MPI.


def test_actual_lammps_refuses_hybrid_and_shrink_wrap_models(tmp_path):
    path = _script(tmp_path)
    text = path.read_text().replace("pair_style lj/cut 5", "pair_style hybrid lj/cut 5")
    path.write_text(text.replace("pair_coeff 1 1 0.2 1", "pair_coeff 1 1 lj/cut 0.2 1"))
    with pytest.raises(PotentialError, match="coordinate-only force families"):
        LAMMPSPotential(["H", "H"], path, type_map=["H"], cell=_system()["cell"])
    path = _script(tmp_path, periodic=False, extra="neighbor 0.3 nsq")
    path.write_text(path.read_text().replace("boundary f f f", "boundary s s s"))
    with LAMMPSPotential(["H", "H"], path, type_map=["H"], pbc=[False]*3) as pes:
        with pytest.raises(PotentialError, match="box changed"):
            pes.evaluate(np.array([[[2., 2, 2], [3.3, 2, 2]]])/BOHR_ANGSTROM)


def test_lammps_assets_are_input_relative_hashed_and_resume_protected(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.io import file_sha256, referenced_files
    from pyrpmd.runner import run, resume_run

    asset = tmp_path / "coefficients.inc"
    asset.write_text("pair_coeff 1 1 0.2 1\n")
    path = _script(tmp_path, extra='include "${pyrpmd_model_dir}/coefficients.inc"')
    mapping = {
        "task": "pimd", "temperature": "10 K", "beads": 1,
        "system": {"symbols": ["H", "H"], "geometry": [[2.,2.,2.], [3.3,2.,2.]],
                   "geometry_unit": "angstrom", "cell": (np.eye(3)*20).tolist(),
                   "cell_unit": "angstrom"},
        "potential": {"provider": "lammps", "model": path.name,
                      "params": {"type_map": ["H"], "asset_files": [asset.name]}},
        "dynamics": {"timestep": "0.01 fs"},
        "sampling": {"equilibration_steps": 0, "production_steps": 4, "interval_steps": 2},
    }
    job = JobSpec.from_mapping(mapping, base_directory=tmp_path)
    config, system = job.resolve()
    assert config["potential"]["params"]["asset_files"] == [str(asset)]
    assert referenced_files(config)[str(asset)] == file_sha256(asset)
    with make_potential(config["potential"], system) as pes:
        assert pes.metadata["asset_sha256"][str(asset)] == file_sha256(asset)
    destination = tmp_path / "run"
    run(job, output=destination)
    resume_run(destination)
    asset.write_text("pair_coeff 1 1 0.4 1\n")
    with pytest.raises(RuntimeError, match="changed or is missing"):
        resume_run(destination)
