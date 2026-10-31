"""Electrode factory/workflow integration and independent review regressions."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import EV_TO_AU
from pyrpmd.electrochemistry import (
    AdiabaticElectrodePotential,
    ElectronReservoir,
    ElectronicPESResult,
)
from pyrpmd.input import JobSpec
from pyrpmd.potentials import PotentialError, check_potential, make_potential
from pyrpmd.runner import run, resume_run


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "electrochemistry"


def reservoir(mu=0.0):
    return ElectronReservoir(0.0, 0.0, mu, "explicit model reference")


class QuarticElectronicModel:
    def __init__(self, quadratic):
        self.quadratic = quadratic

    def evaluate(self, q, numbers):
        return ElectronicPESResult(
            0.25 * numbers**4 + 0.5 * self.quadratic * numbers**2,
            np.zeros_like(q),
            numbers**3 + self.quadratic * numbers,
        )


def test_stationary_local_maximum_is_rejected_despite_valid_endpoint_bracket():
    potential = AdiabaticElectrodePotential(
        QuarticElectronicModel(-1.0), reservoir(), electron_bounds=(-2.0, 2.0),
    )
    # The first secant root is N=0, a maximum; true minima are at N=+-1.
    with pytest.raises(PotentialError, match="local dE/dN"):
        potential.solve(np.zeros((2, 1, 3)))


@pytest.mark.parametrize("quadratic", [0.0, 1e-10, 1e10])
def test_flat_quartic_and_steep_convex_branches_pass_local_guard(quadratic):
    potential = AdiabaticElectrodePotential(
        QuarticElectronicModel(quadratic), reservoir(), electron_bounds=(-2.0, 2.0),
    )
    result = potential.solve(np.zeros((2, 1, 3)))
    np.testing.assert_array_equal(result.electron_numbers, 0.0)
    np.testing.assert_array_equal(result.pes.energy, 0.0)
    np.testing.assert_array_equal(result.chemical_potentials_au, 0.0)


class ReusedElectronicBuffers:
    """A legitimate provider that overwrites all borrowed result arrays."""

    def __init__(self):
        self.result = None

    def evaluate(self, q, numbers):
        if self.result is None:
            self.result = ElectronicPESResult(
                np.zeros(len(q)), np.zeros_like(q), np.zeros(len(q)),
                status=np.zeros(len(q), dtype=np.int32), virial=np.zeros((len(q), 3, 3)),
            )
        x = q[:, 0, 0]
        self.result.energy[:] = 0.5 * x**2 + x * numbers + numbers**2
        self.result.denergy_dnumber[:] = x + 2 * numbers
        self.result.forces[:] = 0.0
        self.result.forces[:, 0, 0] = -x - numbers
        self.result.virial[:] = 0.0
        self.result.virial[:, 0, 0] = (-x - numbers) * x
        return self.result


def test_convexity_probe_preserves_root_results_with_reused_provider_buffers():
    model = ReusedElectronicBuffers()
    potential = AdiabaticElectrodePotential(model, reservoir(), electron_bounds=(-2.0, 2.0))
    q = np.zeros((2, 1, 3))
    q[:, 0, 0] = [0.2, -0.4]
    result = potential.solve(q)
    np.testing.assert_allclose(result.electron_numbers, -0.5 * q[:, 0, 0], atol=1e-15)
    np.testing.assert_allclose(result.pes.energy, 0.25 * q[:, 0, 0] ** 2, atol=1e-15)
    np.testing.assert_allclose(result.pes.forces[:, 0, 0], -0.5 * q[:, 0, 0], atol=1e-15)
    np.testing.assert_allclose(result.pes.virial[:, 0, 0], -0.5 * q[:, 0, 0] ** 2, atol=1e-15)
    np.testing.assert_allclose(result.chemical_potentials_au, 0.0, atol=1e-15)
    # The returned root result remains independent of future provider calls.
    model.evaluate(q, np.zeros(2))
    np.testing.assert_allclose(result.pes.forces[:, 0, 0], -0.5 * q[:, 0, 0], atol=1e-15)
    np.testing.assert_allclose(result.chemical_potentials_au, 0.0, atol=1e-15)


def test_nested_factory_resolves_relative_path_and_preserves_periodic_forces():
    config, system = JobSpec.from_file(EXAMPLES / "fes2d.inp").resolve()
    model_factory = config["potential"]["params"]["model"]["factory"]
    assert model_factory == str(EXAMPLES / "toy_electrode.py") + ":build"
    potential = make_potential(config["potential"], system)
    try:
        q = np.repeat(system.geometry[None], 3, axis=0)
        q[:, 0] += [[0.13, -0.23, 0.12], [0.27, 0.11, -0.17], [-0.31, 0.22, 0.19]]
        result = potential.evaluate(q)
        z = q[:, 0, 2] - q[:, 1:, 2].mean(axis=1) - 3.0
        n = (0.2 * EV_TO_AU + 0.01 * z) / 0.05
        lateral = 0.003 / 2 * np.sum(1 - np.cos(2 * np.pi * q[:, 0, :2] / 6), axis=1)
        expected_energy = lateral + 0.01 * z**2 - 0.025 * n**2
        np.testing.assert_allclose(result.energy, expected_energy, atol=1e-15)
        np.testing.assert_allclose(result.forces[:, 0, 2], -0.02 * z + 0.01 * n, atol=1e-15)
        assert check_potential(potential, q)["max_absolute_force_error"] < 1e-10
        shifted = q.copy()
        shifted[:, 0] += 2 * system.cell[0] - 3 * system.cell[1]
        other = potential.evaluate(shifted)
        np.testing.assert_allclose(other.energy, result.energy, atol=1e-15)
        np.testing.assert_allclose(other.forces, result.forces, atol=1e-15)
        assert potential.metadata["electrode_approximation"] == "independent_frame_adiabatic_minimum"
    finally:
        potential.close()


def test_existing_two_dimensional_fes_workflow_and_resume_use_electrode_factory(tmp_path):
    spec = JobSpec.from_file(EXAMPLES / "fes2d.inp")
    spec.data["native"] = False
    spec.data["sampling"].update(
        equilibration_steps=4, production_steps=20, interval_steps=2, andersen_interval=2,
    )
    spec.data["umbrella"]["grid"] = [
        {"start": "-0.1 bohr", "stop": "0.1 bohr", "count": 2},
        {"start": "2.9 bohr", "stop": "3.1 bohr", "count": 2},
    ]
    spec.data["analysis"]["bins"] = [3, 3]
    spec.data["output"]["checkpoint_steps"] = 4
    directory = tmp_path / "electrode-fes"
    summary = run(spec, output=directory)
    assert summary["task"] == "fes"
    assert summary["dimension"] == 2
    assert summary["streams"] == summary["windows"] == 4
    assert summary["samples"] == 40
    assert summary["measure"] == "cv_density"
    assert summary["energy_unit"] == "hartree"
    assert summary["converged"] is False
    resolved = json.loads((directory / "input" / "resolved.json").read_text())
    assert resolved["potential"]["params"]["reservoir"]["potential_V"] == -0.2
    assert resolved["potential"]["params"]["model"]["factory"].endswith("toy_electrode.py:build")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert str(EXAMPLES / "toy_electrode.py") in manifest["files"]
    checkpoints = sorted((directory / "checkpoints").glob("*.npz"))
    assert len(checkpoints) == 4
    payloads = {}
    _, system = spec.resolve()
    for path in checkpoints:
        with load_checkpoint(path) as saved:
            payloads[path] = {key: np.array(saved[key], copy=True) for key in saved.files}
            expected_fixed = np.repeat(system.geometry[None, 1:, :], 4, axis=0)
            np.testing.assert_array_equal(saved["q"][:, 1:], expected_fixed)
            np.testing.assert_array_equal(saved["p"][:, 1:], 0.0)
            assert np.isfinite(saved["energies"]).all()
    with load_checkpoint(directory / "fes.npz") as saved:
        original_fes = {key: np.array(saved[key], copy=True) for key in saved.files}
    resumed = resume_run(directory)
    assert resumed["samples"] == summary["samples"]
    assert resumed["converged"] is False
    for path, previous in payloads.items():
        with load_checkpoint(path) as saved:
            assert set(saved.files) == set(previous)
            for key in previous:
                np.testing.assert_array_equal(saved[key], previous[key])
    with load_checkpoint(directory / "fes.npz") as saved:
        for key, previous in original_fes.items():
            np.testing.assert_array_equal(saved[key], previous)
