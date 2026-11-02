"""Physical invariants of the approximate action-preserving QCT initializer."""

import json

import numpy as np
import pytest

from pyrpmd.initial_states import prepare_initial_ensemble
from pyrpmd.qct_constraints import constrain_qct_phases


def _prepared(kind, policy=None):
    if kind == "linear":
        geometry = np.array([[0., 0., -2.], [0., 0., 0.], [0., 0., 2.]])
        masses = np.full(3, 2000.)
        modes = np.zeros((4, 3, 3))
        modes[0, :, 2] = np.array([1., 0., -1.]) / np.sqrt(2)
        modes[1, :, 2] = np.array([1., -2., 1.]) / np.sqrt(6)
        modes[2, :, 0] = np.array([1., -2., 1.]) / np.sqrt(6)
        modes[3, :, 1] = np.array([1., -2., 1.]) / np.sqrt(6)
        fragment = {"modes": modes.tolist(), "frequencies": [.01, .02, .03, .03]}
    else:
        geometry = np.array([[0., 0., 0.], [1.8, 0., 0.], [.4, 1.6, 0.], [.3, .3, 1.7]])
        masses = np.array([3000., 4000., 5000., 6000.])
        if kind == "triatom":
            geometry, masses = geometry[:3], masses[:3]
        # A central-force Hessian provides independent analytic rigid zero modes
        # and a positive vibrational block for these triangles/tetrahedra.
        hessian = np.zeros((3 * len(masses), 3 * len(masses)))
        for first in range(len(masses)):
            for second in range(first + 1, len(masses)):
                direction = geometry[first] - geometry[second]
                direction /= np.linalg.norm(direction)
                block = (.4 + .1 * first + .2 * second) * np.outer(direction, direction)
                a, b = slice(3 * first, 3 * first + 3), slice(3 * second, 3 * second + 3)
                hessian[a, a] += block
                hessian[b, b] += block
                hessian[a, b] -= block
                hessian[b, a] -= block
        nv = 3 * len(masses) - 6
        quanta = [0] * nv
        quanta[-1] = 2
        fragment = {"hessian": hessian.tolist(), "vibrational_quanta": quanta}
    system = {"geometry": np.vstack([[0., 0., -15.], geometry]),
              "masses": np.r_[1836., masses],
              "fragments": [[0], list(range(1, len(masses) + 1))]}
    section = {"method": "qct", "fragments": [{}, fragment]}
    if policy is not None:
        section["j0_energy_policy"] = policy
    return prepare_initial_ensemble(section, system, 1, 500.)


def _reference(kind):
    return _prepared(kind).references[1]


def _solve(reference, phase, **kwargs):
    return constrain_qct_phases(phase, reference.frequencies, reference.vibrational_quanta,
                                reference.modes, reference.geometry, reference.masses, **kwargs)


def _cartesian_checks(reference, mode_q, mode_p):
    rows = reference.modes.reshape(reference.nmodes, -1)
    root_mass = np.sqrt(reference.masses)[:, None]
    q = reference.geometry + (mode_q @ rows).reshape(-1, 3) / root_mass
    p = (mode_p @ rows).reshape(-1, 3) * root_mass
    centered = q - np.average(q, axis=0, weights=reference.masses)
    np.testing.assert_allclose(np.sum(np.cross(centered, p), axis=0), 0., atol=2e-10)
    np.testing.assert_allclose(p.sum(axis=0), 0., atol=1e-12)
    projected_q = ((q - reference.geometry) * root_mass).ravel() @ rows.T
    projected_p = (p / root_mass).ravel() @ rows.T
    actual = .5 * (projected_p ** 2 + (reference.frequencies * projected_q) ** 2)
    expected = (np.array(reference.vibrational_quanta) + .5) * reference.frequencies
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("kind", ["triatom", "linear", "tetrahedron"])
def test_joint_j0_and_each_mode_action_for_distinct_phase_draws(kind):
    reference = _reference(kind)
    for seed in range(48):
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, reference.nmodes)
        original = phase.copy()
        mode_q, mode_p = _solve(reference, phase)
        _cartesian_checks(reference, mode_q, mode_p)
        np.testing.assert_array_equal(phase, original)


def test_single_diatomic_stretch_retains_original_phase_and_action():
    masses = np.array([2000., 4000.])
    geometry = np.array([[0., 0., -.7], [0., 0., .7]])
    modes = np.zeros((1, 2, 3))
    modes[0, 0, 2] = -np.sqrt(masses[1] / masses.sum())
    modes[0, 1, 2] = np.sqrt(masses[0] / masses.sum())
    for phase in np.linspace(0, 2 * np.pi, 17):
        q, p = constrain_qct_phases([phase], [.02], [2], modes, geometry, masses,
                                   max_iterations=0)
        np.testing.assert_allclose(q, np.sqrt(5 / .02) * np.cos(phase), rtol=0, atol=0)
        np.testing.assert_allclose(p, np.sqrt(5 * .02) * np.sin(phase), rtol=0, atol=0)
        np.testing.assert_allclose(.5 * (p*p + (.02*q)**2), [.05], rtol=2e-15)


def test_phase_map_is_deterministic_without_consuming_randomness():
    reference = _reference("linear")
    rng = np.random.default_rng(12)
    phase = rng.uniform(0, 2 * np.pi, reference.nmodes)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    first = _solve(reference, phase)
    second = _solve(reference, phase)
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_rotation_translation_and_unequal_masses_leave_phase_solution_invariant():
    reference = _reference("tetrahedron")
    phase = np.random.default_rng(7).uniform(0, 2 * np.pi, reference.nmodes)
    expected = _solve(reference, phase)
    axis = np.array([1., 2., -1.]) / np.sqrt(6)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                      [-axis[1], axis[0], 0.]])
    rotation = np.eye(3) + np.sin(.83) * cross + (1 - np.cos(.83)) * (cross @ cross)
    observed = constrain_qct_phases(
        phase, reference.frequencies, reference.vibrational_quanta,
        reference.modes @ rotation.T, reference.geometry @ rotation.T + [3., -5., 2.],
        reference.masses,
    )
    for a, b in zip(expected, observed):
        np.testing.assert_allclose(a, b, rtol=3e-11, atol=3e-12)


def test_equilibrium_angular_momentum_term_is_included():
    # Deliberately add a tiny reference rotation below the ordinary explicit-mode
    # orthogonality threshold. A C-only formula would return nonzero Cartesian J.
    reference = _reference("triatom")
    root_mass = np.sqrt(reference.masses)[:, None]
    relative = reference.geometry - np.average(reference.geometry, axis=0,
                                               weights=reference.masses)
    rigid = np.cross([0., 0., 1.], relative) * root_mass
    rigid /= np.linalg.norm(rigid)
    modes = reference.modes.copy()
    modes[0] += 2e-9 * rigid
    phase = np.random.default_rng(2).uniform(0, 2 * np.pi, reference.nmodes)
    q, p = constrain_qct_phases(phase, reference.frequencies, reference.vibrational_quanta,
                               modes, reference.geometry, reference.masses)
    rows = modes.reshape(reference.nmodes, -1)
    cartesian_q = reference.geometry + (q @ rows).reshape(-1, 3) / root_mass
    cartesian_p = (p @ rows).reshape(-1, 3) * root_mass
    relative = cartesian_q - np.average(cartesian_q, axis=0, weights=reference.masses)
    np.testing.assert_allclose(np.cross(relative, cartesian_p).sum(axis=0), 0., atol=2e-10)


def test_nonconvergence_is_explicit_and_does_not_return_a_projected_fallback():
    reference = _reference("linear")
    phase = np.random.default_rng(0).uniform(0, 2 * np.pi, reference.nmodes)
    with pytest.raises(RuntimeError, match="did not converge.*no state was returned or resampled"):
        _solve(reference, phase, max_iterations=0)


def test_cartesian_energy_check_rejects_an_inaccurate_reference_basis():
    reference = _reference("triatom")
    modes = reference.modes * (1 + 2e-9)
    with pytest.raises(RuntimeError, match="Cartesian constraint check"):
        constrain_qct_phases(np.zeros(reference.nmodes), reference.frequencies,
                             reference.vibrational_quanta, modes, reference.geometry,
                             reference.masses)


@pytest.mark.parametrize("kind", ["triatom", "linear", "tetrahedron"])
def test_preserve_actions_actual_sample_and_public_preparation_diagnostics(kind):
    prepared = _prepared(kind, "preserve_actions")
    assert prepared.metadata["j0_energy_policy"] == "preserve_actions"
    assert prepared.metadata["phase_sampling"] == "uniform_seed_phases_then_deterministic_j0_constraints_v1"
    for seed in range(24):
        rng = np.random.default_rng(seed)
        original_rng = np.random.default_rng(seed)
        state = prepared.sample(rng)
        for reference in prepared.references:
            original_rng.uniform(0, 2 * np.pi, reference.nmodes)
        assert json.dumps(rng.bit_generator.state) == json.dumps(original_rng.bit_generator.state)
        report = prepared.preparation_diagnostics(state)
        assert report["frame"] == "unrotated_preparation_reference"
        for reference, fragment in zip(prepared.references, report["fragments"]):
            ids = list(reference.atoms)
            root_mass = np.sqrt(reference.masses)[:, None]
            rows = reference.modes.reshape(reference.nmodes, 3 * len(reference.masses))
            q, p = state.q[0, ids], state.p[0, ids]
            mq = ((q - reference.geometry) * root_mass).ravel() @ rows.T
            mp = (p / root_mass).ravel() @ rows.T
            target = (np.array(reference.vibrational_quanta) + .5) * reference.frequencies
            actual = .5 * (mp**2 + (reference.frequencies * mq)**2)
            np.testing.assert_allclose(actual, target, rtol=2e-12, atol=1e-14)
            np.testing.assert_allclose(fragment["reference_mode_energies_au"], actual, atol=1e-15)
            np.testing.assert_allclose(fragment["target_mode_energies_au"], target, atol=1e-15)
            np.testing.assert_allclose(fragment["reference_mode_energy_error_au"], actual-target, atol=1e-15)
            np.testing.assert_allclose(fragment["reference_mode_energy_relative_error"], 0., atol=2e-12)
            np.testing.assert_allclose(fragment["centroid_com_momentum_au"], 0., atol=1e-12)
            np.testing.assert_allclose(fragment["centroid_angular_momentum_au"], 0., atol=2e-10)
            assert abs(fragment["cartesian_harmonic_energy_error_au"]) < 1e-13
        json.dumps(report, allow_nan=False)


def test_preserve_actions_atomic_fragments_and_diatomic_stretch():
    atom_system = {"geometry": [[0., 0., -2.], [0., 0., 2.]], "masses": [2000., 4000.],
                   "fragments": [[0], [1]]}
    atoms = prepare_initial_ensemble({"method": "qct", "j0_energy_policy": "preserve_actions",
                                     "fragments": [{}, {}]}, atom_system, 1, 500.)
    state = atoms.sample(np.random.default_rng(3))
    np.testing.assert_array_equal(state.q[0], atom_system["geometry"])
    np.testing.assert_array_equal(state.p, 0.)
    for fragment in atoms.preparation_diagnostics(state)["fragments"]:
        assert fragment["target_mode_energies_au"] == []
        assert fragment["reference_mode_energy_relative_error"] == []
        assert fragment["cartesian_harmonic_energy_error_au"] == 0.

    system = {"geometry": [[0., 0., -12.], [0., 0., -.7], [0., 0., .7]],
              "masses": [1836., 2000., 4000.], "fragments": [[0], [1, 2]]}
    mode = np.zeros((1, 2, 3))
    mode[0, :, 2] = [-np.sqrt(2/3), np.sqrt(1/3)]
    section = {"method": "qct", "fragments": [{}, {"frequencies": [.02], "modes": mode,
                                                      "vibrational_quanta": [2]}]}
    projected = prepare_initial_ensemble(section, system, 1, 500.)
    constrained = prepare_initial_ensemble(dict(section, j0_energy_policy="preserve_actions"),
                                           system, 1, 500.)
    for seed in range(12):
        first = projected.sample(np.random.default_rng(seed))
        second = constrained.sample(np.random.default_rng(seed))
        np.testing.assert_allclose(first.q, second.q, rtol=0, atol=1e-15)
        np.testing.assert_allclose(first.p, second.p, rtol=0, atol=2e-15)
        report = constrained.preparation_diagnostics(second)["fragments"][1]
        np.testing.assert_allclose(report["reference_mode_energies_au"], [.05], rtol=2e-14)


def test_sample_constraint_failure_propagates_without_legacy_projection_or_random_retry(monkeypatch):
    import pyrpmd.initial_states as initial_states
    import pyrpmd.qct_constraints as constraints

    prepared = _prepared("linear", "preserve_actions")
    original = constraints.constrain_qct_phases
    calls = []

    def fail_molecular(phase, *args, **kwargs):
        calls.append(phase.copy())
        if len(phase):
            raise RuntimeError("intentional phase-constraint failure")
        return original(phase, *args, **kwargs)

    def forbid_legacy(*args, **kwargs):
        pytest.fail("preserve_actions silently fell back to rigid momentum projection")

    monkeypatch.setattr(constraints, "constrain_qct_phases", fail_molecular)
    monkeypatch.setattr(initial_states, "_project_centroid_j0", forbid_legacy)
    rng = np.random.default_rng(19)
    expected_rng = np.random.default_rng(19)
    for reference in prepared.references:
        expected_rng.uniform(0, 2 * np.pi, reference.nmodes)
    with pytest.raises(RuntimeError, match="intentional phase-constraint failure"):
        prepared.sample(rng)
    assert [len(phase) for phase in calls] == [0, 4]
    assert json.dumps(rng.bit_generator.state) == json.dumps(expected_rng.bit_generator.state)


@pytest.mark.parametrize("kind", ["triatom", "linear", "tetrahedron"])
def test_default_project_matches_explicit_policy_without_phase_constraints(kind, monkeypatch):
    import pyrpmd.qct_constraints as constraints

    def forbid_new_solver(*args, **kwargs):
        pytest.fail("the legacy default unexpectedly invokes constrained phase sampling")

    monkeypatch.setattr(constraints, "constrain_qct_phases", forbid_new_solver)
    default = _prepared(kind)
    explicit = _prepared(kind, "project")
    assert default.j0_energy_policy == "project"
    for seed in (0, 8, 42):
        expected = default.sample(np.random.default_rng(seed))
        actual = explicit.sample(np.random.default_rng(seed))
        np.testing.assert_array_equal(actual.q, expected.q)
        np.testing.assert_array_equal(actual.p, expected.p)


def test_preparation_report_distinguishes_projected_mode_sum_from_cartesian_energy_loss(monkeypatch):
    import pyrpmd.initial_states as initial_states

    prepared = _prepared("triatom")
    # Hold the phase draw fixed and isolate only the angular-momentum removal;
    # verify its energy change against the independent rigid-body identity.
    with monkeypatch.context() as context:
        context.setattr(initial_states, "_project_centroid_j0", lambda *args: None)
        before = prepared.sample(np.random.default_rng(42))
    after = prepared.sample(np.random.default_rng(42))
    initial = prepared.preparation_diagnostics(before)["fragments"][1]
    final = prepared.preparation_diagnostics(after)["fragments"][1]
    reference = prepared.references[1]
    masses = reference.masses
    q = before.q[0, list(reference.atoms)]
    p = before.p[0, list(reference.atoms)]
    relative = q - np.average(q, axis=0, weights=masses)
    angular = np.cross(relative, p).sum(axis=0)
    inertia = np.eye(3) * np.sum(masses * np.sum(relative**2, axis=1))
    inertia -= np.einsum("a,ai,aj->ij", masses, relative, relative)
    expected_loss = -.5 * angular @ np.linalg.pinv(inertia, rcond=1e-12) @ angular
    actual_loss = final["cartesian_kinetic_energy_au"] - initial["cartesian_kinetic_energy_au"]
    assert expected_loss < -1e-8
    assert actual_loss == pytest.approx(expected_loss, abs=1e-15)
    assert final["cartesian_harmonic_energy_error_au"] == pytest.approx(expected_loss, abs=1e-15)
    mode_energy_error = sum(final["reference_mode_energy_error_au"])
    assert not np.isclose(mode_energy_error, expected_loss, rtol=1e-4, atol=1e-12)
    assert final["cartesian_kinetic_energy_au"] + final["reference_harmonic_potential_energy_au"] == pytest.approx(
        sum(final["target_mode_energies_au"]) + final["cartesian_harmonic_energy_error_au"], abs=1e-15
    )
