import math

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.constants import AMU_TO_AU, KB_AU
from pyrpmd.potentials import PESResult
from pyrpmd.vtst import (
    MEPOptions,
    MEPPath,
    PartitionSpecies,
    analyze_path_modes,
    canonical_variational_profile,
    compute_vtst,
    rigid_body_basis,
    trace_mep,
    wkb_transmission_probability,
)


def _orthogonal_complement(columns):
    q, _ = np.linalg.qr(columns, mode="complete")
    return q[:, columns.shape[1] :]


def _straight_h3_path(points=81):
    mass = np.full(3, AMU_TO_AU)
    mu = AMU_TO_AU
    base = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tangent = np.zeros(9)
    tangent[[0, 3, 6]] = (1.0, -2.0, 1.0)
    tangent /= np.linalg.norm(tangent)
    s = np.linspace(-1.0, 1.0, points)
    barrier = 0.02
    energy = barrier * (1.0 - s**2) ** 2
    derivative = -4.0 * barrier * s * (1.0 - s**2)
    second = -4.0 * barrier + 12.0 * barrier * s**2
    transverse = np.array([0.004, 0.006, 0.009])
    geometries = np.array([(base.ravel() + coordinate * tangent).reshape(3, 3) for coordinate in s])
    gradients = np.array([(slope * tangent).reshape(3, 3) for slope in derivative])
    hessians = []
    for geometry, reaction_curvature in zip(geometries, second):
        rigid = rigid_body_basis(geometry, mass)
        constraints = np.linalg.qr(np.column_stack((tangent, rigid)))[0][:, : 1 + rigid.shape[1]]
        complement = _orthogonal_complement(constraints)
        assert complement.shape[1] == len(transverse)
        hessian = reaction_curvature * np.outer(tangent, tangent)
        hessian += mu * complement @ np.diag(transverse**2) @ complement.T
        hessians.append(0.5 * (hessian + hessian.T))
    path = MEPPath(
        s_bohr=s,
        geometries_bohr=geometries,
        energies_hartree=energy,
        gradients_hartree_per_bohr=gradients,
        hessians_hartree_per_bohr2=np.asarray(hessians),
        tangent_scaled=np.tile(tangent, (points, 1)),
        side_converged=(True, True),
        saddle_eigenvalues_omega2=np.array([-4.0 * barrier / mu]),
        scaling_mass=mu,
    )
    well_frequency = math.sqrt(8.0 * barrier / mu)
    reactant = PartitionSpecies(
        geometry_bohr=geometries[0],
        masses=mass,
        frequencies_au=np.r_[well_frequency, transverse],
    )
    return path, mass, reactant


class _SeparableBarrier:
    def __init__(self):
        self.mass = np.full(3, AMU_TO_AU)
        self.mu = AMU_TO_AU
        self.base = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.tangent = np.zeros(9)
        self.tangent[[0, 3, 6]] = (1.0, -2.0, 1.0)
        self.tangent /= np.linalg.norm(self.tangent)
        rigid = rigid_body_basis(self.base, self.mass)
        constraints = np.linalg.qr(np.column_stack((self.tangent, rigid)))[0][:, : 1 + rigid.shape[1]]
        self.complement = _orthogonal_complement(constraints)
        self.barrier = 0.02
        self.width = 0.8
        self.omega = np.array([0.004, 0.006, 0.009])

    def _values(self, geometry):
        displacement = geometry.ravel() - self.base.ravel()
        coordinate = float(np.dot(self.tangent, displacement))
        transverse = self.complement.T @ displacement
        energy = self.barrier * (1.0 - (coordinate / self.width) ** 2) ** 2
        energy += 0.5 * self.mu * float(np.dot((self.omega * transverse), (self.omega * transverse)))
        slope = -4.0 * self.barrier * coordinate / self.width**2 * (1.0 - (coordinate / self.width) ** 2)
        gradient = slope * self.tangent
        gradient += self.mu * self.complement @ (self.omega**2 * transverse)
        curvature = -4.0 * self.barrier / self.width**2 + 12.0 * self.barrier * coordinate**2 / self.width**4
        hessian = curvature * np.outer(self.tangent, self.tangent)
        hessian += self.mu * self.complement @ np.diag(self.omega**2) @ self.complement.T
        return energy, gradient.reshape(3, 3), hessian

    def evaluate(self, positions):
        values = [self._values(geometry) for geometry in np.asarray(positions)]
        return PESResult(
            np.array([value[0] for value in values]),
            -np.array([value[1] for value in values]),
        )

    def hessian(self, positions):
        return np.array([self._values(geometry)[2] for geometry in np.asarray(positions)])


def test_bidirectional_mep_reaches_both_analytic_wells():
    potential = _SeparableBarrier()
    path = trace_mep(
        potential,
        potential.base,
        potential.mass,
        options=MEPOptions(
            step_bohr=0.04,
            initial_displacement_bohr=0.01,
            max_steps_per_side=100,
            gradient_tolerance_hartree_per_bohr=1.0e-7,
            hessian_step_bohr=1.0e-4,
        ),
    )
    projected = np.array(
        [np.dot(potential.tangent, geometry.ravel() - potential.base.ravel()) for geometry in path.geometries_bohr]
    )
    assert path.side_converged == (True, True)
    assert sorted(projected[[0, -1]]) == pytest.approx(
        [-potential.width, potential.width], abs=2.0e-5
    )
    assert path.energies_hartree[path.saddle_index] == pytest.approx(potential.barrier)


def test_straight_path_sct_reduces_exactly_to_zct_and_cvt_is_symmetric():
    path, masses, reactant = _straight_h3_path()
    modes = analyze_path_modes(path, masses)
    assert np.max(modes.curvature_bohr_inverse) < 1.0e-11
    assert np.array_equal(modes.effective_mass_ratio_sct, np.ones(len(path.s_bohr)))
    profile = canonical_variational_profile(path, masses, modes, [reactant], 500.0)
    np.testing.assert_allclose(profile.delta_g_hartree, profile.delta_g_hartree[::-1], rtol=2.0e-10, atol=2.0e-12)
    result = compute_vtst(path, masses, [reactant], [300.0, 500.0], tunnelling_quadrature_points=32)
    for row in result.temperatures:
        assert row["zct"]["kappa"] == row["sct"]["kappa"]
        assert row["cvt_zct"]["rate"] == row["cvt_sct"]["rate"]


def test_sct_saddle_mass_uses_nonuniform_linear_interpolation(monkeypatch):
    from types import SimpleNamespace

    import pyrpmd.vtst_thermo as thermo

    s = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    angle = 0.08 * s**2 + 0.03 * s
    tangent = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(s)))

    def fake_modes(_geometry, _masses, _hessian, *, tangent_scaled, **_options):
        normal = np.array([-tangent_scaled[1], tangent_scaled[0], 0.0])
        return SimpleNamespace(
            frequencies_au=np.array([0.01]),
            vectors_scaled=normal[:, None],
            removed_rigid_modes=2,
        )

    monkeypatch.setattr(thermo, "generalized_normal_modes", fake_modes)
    path = MEPPath(
        s_bohr=s,
        geometries_bohr=np.zeros((len(s), 1, 3)),
        energies_hartree=np.zeros(len(s)),
        gradients_hartree_per_bohr=np.zeros((len(s), 1, 3)),
        hessians_hartree_per_bohr2=np.zeros((len(s), 3, 3)),
        tangent_scaled=tangent,
        side_converged=(True, True),
        saddle_eigenvalues_omega2=np.array([-1.0]),
        scaling_mass=AMU_TO_AU,
    )
    analysis = thermo.analyze_path_modes(path, [AMU_TO_AU])
    saddle = path.saddle_index
    expected = np.interp(
        s[saddle],
        s[[saddle - 1, saddle + 1]],
        analysis.effective_mass_ratio_sct[[saddle - 1, saddle + 1]],
    )
    arithmetic_mean = np.mean(
        analysis.effective_mass_ratio_sct[[saddle - 1, saddle + 1]]
    )
    assert analysis.effective_mass_ratio_sct[saddle] == pytest.approx(expected)
    assert analysis.effective_mass_ratio_sct[saddle] != pytest.approx(
        arithmetic_mean, abs=1.0e-10
    )


def test_reactant_partition_matches_low_temperature_harmonic_state():
    from pyrpmd.low_temperature import HarmonicState

    path, masses, reactant = _straight_h3_path()
    beta = 1.0 / (KB_AU * 500.0)
    reference = HarmonicState(
        path.geometries_bohr[0],
        masses,
        path.energies_hartree[0],
        path.hessians_hartree_per_bohr2[0],
        rigid_body=True,
    ).partition_diagnostics(beta)
    actual = reactant.partition_diagnostics(beta)
    assert actual["log_partition"] == pytest.approx(
        reference["log_partition"], rel=2.0e-12, abs=2.0e-12
    )
    assert actual["log_vibrational_partition"] == pytest.approx(
        reference["log_vibrational_partition"], rel=2.0e-12, abs=2.0e-12
    )
    assert actual["log_rotation_partition"] == pytest.approx(
        reference["log_rotation_partition"], rel=2.0e-12, abs=2.0e-12
    )
    assert actual["log_translation_partition_per_bohr3"] == pytest.approx(
        reference["log_translation_partition"], rel=2.0e-12, abs=2.0e-12
    )


def test_eckart_wkb_action_matches_independent_continuous_integral():
    mass = AMU_TO_AU
    barrier = 0.015
    inverse_width = 1.1
    energy = 0.004
    turning = math.acosh(math.sqrt(barrier / energy)) / inverse_width
    expected_action, _ = quad(
        lambda coordinate: math.sqrt(
            2.0 * mass * (barrier / math.cosh(inverse_width * coordinate) ** 2 - energy)
        ),
        -turning,
        turning,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )
    s = np.linspace(-5.0, 5.0, 2001)
    values = barrier / np.cosh(inverse_width * s) ** 2
    top = wkb_transmission_probability(barrier, s, values, scaling_mass=mass)
    assert top["probability"] == pytest.approx(0.5, abs=0.0)
    actual = wkb_transmission_probability(energy, s, values, scaling_mass=mass)
    assert actual["action"] == pytest.approx(expected_action, rel=2.0e-5)
    expected_probability = 1.0 / (1.0 + math.exp(2.0 * expected_action))
    assert actual["probability"] == pytest.approx(expected_probability, rel=4.0e-5)
    reflected = wkb_transmission_probability(
        2.0 * barrier - energy, s, values, scaling_mass=mass
    )
    assert reflected["probability"] == pytest.approx(
        1.0 - expected_probability, rel=4.0e-5
    )


def test_workflow_publishes_versioned_identity_and_complete_bundle(monkeypatch, tmp_path):
    import json

    from pyrpmd.input import InputError
    from pyrpmd.system import System
    import pyrpmd.potentials as potentials
    import pyrpmd.workflows.vtst as workflow

    path, masses, reactant = _straight_h3_path(points=41)
    calculation = compute_vtst(
        path, masses, [reactant], [300.0], tunnelling_quadrature_points=16
    )
    system = System(("H", "H", "H"), masses, path.geometries_bohr[path.saddle_index])
    config = {
        "schema_version": 2,
        "task": "rate",
        "temperature_K": 300.0,
        "system": {"symbols": ["H", "H", "H"]},
        "potential": {"provider": "python", "factory": "unused:build"},
        "reaction": {
            "definition": "explicit",
            "fragments": [[0, 1], [2]],
            "forming_bonds": [[1, 2]],
            "breaking_bonds": [[0, 1]],
            "preparation": {"ts_action": "validate", "check_channel": True},
        },
    }

    class Spec:
        input_path = None

        def resolve(self):
            return config, system

    class Potential:
        closed = False

        def close(self):
            self.closed = True

    calls = {"potential": 0}

    def make_potential(_config, _system):
        calls["potential"] += 1
        return Potential()

    prepared = {
        "ready": True,
        "saddle_valid": True,
        "channel_consistent": True,
        "transition_state": system.geometry.tolist(),
        "reaction": config["reaction"],
        "diagnostics": [],
    }
    monkeypatch.setattr(workflow, "_job_spec", lambda _value: Spec())
    monkeypatch.setattr(potentials, "make_potential", make_potential)
    monkeypatch.setattr(workflow, "prepare_rate", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(workflow, "trace_mep", lambda *_args, **_kwargs: path)
    monkeypatch.setattr(
        workflow,
        "preparation_options",
        lambda _reaction: {"bond_scale": 1.25, "covalent_radii": None},
    )
    monkeypatch.setattr(
        workflow,
        "validate_reaction_path_endpoints",
        lambda *_args, **_kwargs: {
            "checked": True,
            "connected": True,
            "orientation": "left_to_right",
        },
    )
    monkeypatch.setattr(
        workflow,
        "fragment_species_from_path_endpoint",
        lambda *_args, **_kwargs: [reactant],
    )
    monkeypatch.setattr(workflow, "compute_vtst", lambda *_args, **_kwargs: calculation)

    first = tmp_path / "first"
    second = tmp_path / "second"
    summary = workflow.run_vtst("unused.inp", output=first, tunnelling_quadrature_points=16)
    workflow.run_vtst("unused.inp", output=second, tunnelling_quadrature_points=32)
    assert summary["numerical_converged"] is False
    for relative in (
        "manifest.json",
        "input/resolved.json",
        "input/system.json",
        "results/vtst.json",
        "results/vtst_path.csv",
        "results/vtst_rates.csv",
        "results/summary.json",
    ):
        assert (first / relative).is_file()
    manifest_first = json.loads((first / "manifest.json").read_text())
    manifest_second = json.loads((second / "manifest.json").read_text())
    assert manifest_first["schema"] == "pyrpmd.vtst-run"
    assert manifest_first["restartable"] is False
    assert manifest_first["rng"] == "not_used"
    assert manifest_first["physics_fingerprint"] != manifest_second["physics_fingerprint"]
    before = calls["potential"]
    with pytest.raises(InputError, match="already exists"):
        workflow.run_vtst("unused.inp", output=first)
    assert calls["potential"] == before
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_dispatches_vtst_options(monkeypatch, tmp_path, capsys):
    import json

    import pyrpmd.workflows.vtst as workflow
    from pyrpmd.cli import main

    captured = {}

    def fake_run(job, **options):
        captured["job"] = job
        captured.update(options)
        return {"task": "vtst"}

    monkeypatch.setattr(workflow, "run_vtst", fake_run)
    output = tmp_path / "vtst"
    status = main(
        [
            "vtst",
            "job.inp",
            "--output",
            str(output),
            "--temperature",
            "250 K",
            "--temperature",
            "500 K",
            "--mep-step",
            "0.03 bohr",
            "--tunnelling-points",
            "48",
        ]
    )
    assert status == 0
    assert captured["job"] == "job.inp"
    assert captured["output"] == str(output)
    assert captured["temperatures_kelvin"] == ["250 K", "500 K"]
    assert captured["mep_step"] == "0.03 bohr"
    assert captured["tunnelling_quadrature_points"] == 48
