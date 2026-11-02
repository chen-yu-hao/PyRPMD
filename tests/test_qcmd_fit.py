"""Independent IBI/Jacobian references and exact phase-by-phase continuation."""

import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.qcmd import QCMDModel
from pyrpmd.qcmd_fit import (FIT_KIND, _histograms, _ibi_update, _probabilities,
                            fit_fqcmd)


def _options(beads=3):
    geometry = np.array([[0., 0., 0.], [1.7, 0., 0.], [-.5, 1.6, 0.]])
    return dict(potential=HarmonicPotential(.02, center=geometry),
                masses=[1600., 100., 100.], beta=100.,
                initial_q=np.repeat(geometry[None], beads, axis=0),
                molecules=[[0, 1, 2]],
                grids=(np.linspace(.7, 2.9, 17), np.linspace(.4, 2.9, 19)),
                dt=.15, ibi_dt=.2, equilibration_steps=3, sampling_steps=8,
                sample_interval=2, iterations=1, ibi_equilibration_steps=3,
                ibi_sampling_steps=8, gamma_centroid=.07, damping=.5,
                regularization=.02, smoothing=.8, seed=71,
                base_identity="a" * 64, checkpoint_interval=1)


@pytest.mark.parametrize("coordinate", ["bond", "angle"])
def test_exact_boltzmann_inversion_cancels_geometric_jacobian(coordinate):
    """Analytic separable triatomics have r² and sin(theta) measures.

    This checks the IBI sign, temperature, measure cancellation, and use of
    a *correction* rather than adding the entire target PMF a second time.
    """
    x = np.linspace(.8, 2.5, 101)
    beta = 3.1
    base = .5 * .7 * (x - 1.4)**2
    correction = .5 * .35 * (x - 1.8)**2 - .5 * .1 * (x - 1.1)**2
    jacobian = x**2 if coordinate == "bond" else np.sin(x)
    target = jacobian * np.exp(-beta * (base + correction))
    sampled = jacobian * np.exp(-beta * base)
    actual = _ibi_update(np.zeros_like(x), target, sampled, beta=beta,
                         damping=1., regularization=0., smoothing=0.)
    expected = correction - np.dot(target / target.sum(), correction)
    np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=2e-14)
    # The fitted fixed point does not change force or correction gauge.
    fixed = _ibi_update(actual, target, target, beta=beta, damping=1.,
                        regularization=0., smoothing=0.)
    np.testing.assert_allclose(fixed, actual, atol=3e-17)


def test_regularization_has_same_strength_for_different_sample_lengths():
    count = np.array([0., 3., 12., 2., 0.])
    a = _probabilities(count, regularization=.03, smoothing=.6)
    b = _probabilities(100 * count, regularization=.03, smoothing=.6)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
        assert left.sum() == pytest.approx(1.)
    update = _ibi_update(np.zeros(5), count, 100 * count, beta=2., damping=.5,
                         regularization=.03, smoothing=.6)
    np.testing.assert_array_equal(update, np.zeros(5))


def test_zero_bins_without_explicit_regularization_raise():
    with pytest.raises(ValueError, match="zero-density"):
        _probabilities([1, 0, 2, 0], regularization=0., smoothing=0.)


def test_histograms_include_both_endpoints_and_never_drop_domain_overflow():
    grids = (np.linspace(1., 3., 5), np.linspace(.5, 2.5, 5))
    internal = np.array([[1., 3., .5], [2., 2.5, 2.5]])
    counts = _histograms(internal, grids, context="independent endpoint test")
    assert counts[0].sum() == 4
    assert counts[1].sum() == 2
    assert counts[0][0] == 1
    assert counts[0][-1] == 2  # Last bin includes both its lower edge and 3.
    internal[0, 0] = np.nextafter(1., -np.inf)
    with pytest.raises(ValueError, match="outside grid.*endpoint test"):
        _histograms(internal, grids, context="independent endpoint test")


@pytest.mark.parametrize("beads", [1, 3])
def test_fitter_returns_actual_final_validated_model_and_counts(beads, tmp_path):
    options = _options(beads)
    options["checkpoint"] = tmp_path / "fit.npz"
    model = fit_fqcmd(**options)
    assert isinstance(model, QCMDModel)
    assert model.training_beads == beads
    np.testing.assert_equal([c.sum() for c in model.target_counts], [8, 4])
    assert len(model.diagnostics) == options["iterations"] + 1
    final = model.diagnostics[-1]
    assert not final["converged"] and not final["physical_converged"]
    assert final["independent_sample_count"] is None
    for name, correction in zip(("bond", "angle"), model.corrections):
        item = final["coordinates"][name]
        assert item["correction_max_abs_hartree"] == float(np.abs(correction).max())
        assert 0 <= item["raw_total_variation"] <= 1
        assert 0 <= item["regularized_total_variation"] <= 1
    loaded = fit_fqcmd(**options)
    assert loaded.to_dict() == model.to_dict()


@pytest.mark.parametrize("phase,iteration,phase_step", [
    ("target_equilibration", -1, 1), ("target_sampling", -1, 4),
    ("ibi_equilibration", 0, 1), ("ibi_sampling", 0, 4),
    ("ibi_equilibration", 1, 0), ("ibi_sampling", 1, 4),
    ("done", 1, 0),
])
def test_checkpoint_continuation_is_bitwise_exact_in_every_phase(
        tmp_path, monkeypatch, phase, iteration, phase_step):
    import pyrpmd.qcmd_fit as fitting

    class StopTraining(RuntimeError):
        pass

    options = _options()
    reference_path = tmp_path / "reference.npz"
    interrupted_path = tmp_path / "interrupted.npz"
    expected = fit_fqcmd(**options, checkpoint=reference_path)
    original_save = fitting.atomic_savez

    def stop_after_commit(path, **payload):
        original_save(path, **payload)
        meta = json.loads(payload["metadata"])
        if (meta["phase"], meta["iteration"], meta["phase_step"]) == (phase, iteration, phase_step):
            raise StopTraining("simulated process interruption after atomic commit")

    with monkeypatch.context() as scope:
        scope.setattr(fitting, "atomic_savez", stop_after_commit)
        with pytest.raises(StopTraining):
            fit_fqcmd(**options, checkpoint=interrupted_path)
    # Save cadence is operational and can change without altering the walk.
    options["checkpoint_interval"] = 7
    actual = fit_fqcmd(**options, checkpoint=interrupted_path)
    assert actual.to_dict() == expected.to_dict()
    with load_checkpoint(reference_path) as reference, load_checkpoint(interrupted_path) as resumed:
        for field in reference.files:
            np.testing.assert_array_equal(resumed[field], reference[field])


@pytest.mark.parametrize("change", [
    {"seed": 72}, {"gamma_centroid": .05}, {"dt": .12}, {"ibi_dt": .21},
    {"iterations": 2}, {"regularization": .04}, {"smoothing": .7},
    {"damping": .6}, {"sampling_steps": 10}, {"base_identity": "b" * 64},
])
def test_changed_scientific_identity_fails_before_pes_evaluation(tmp_path, change):
    options = _options()
    path = tmp_path / "fit.npz"
    fit_fqcmd(**options, checkpoint=path)

    class ForbiddenPotential:
        def evaluate(self, q):
            raise AssertionError("identity mismatch must be detected before evaluating the PES")

    options.update(change)
    options["potential"] = ForbiddenPotential()
    with pytest.raises(ValueError, match="scientific identity"):
        fit_fqcmd(**options, checkpoint=path)


@pytest.mark.parametrize("field", ["initial_q", "masses", "molecules", "grids"])
def test_geometry_mass_topology_and_grid_enter_training_identity(tmp_path, field):
    options = _options()
    path = tmp_path / "fit.npz"
    fit_fqcmd(**options, checkpoint=path)
    if field == "initial_q":
        options[field] = options[field] + [0., 0., .001]
    elif field == "masses":
        options[field] = [1601., 100., 100.]
    elif field == "molecules":
        options[field] = [[0, 2, 1]]
    else:
        options[field] = (options[field][0] + .01, options[field][1])
    with pytest.raises(ValueError, match="scientific identity"):
        fit_fqcmd(**options, checkpoint=path)


@pytest.mark.parametrize("change", [
    {"seed": True}, {"seed": 1.5}, {"seed": -1}, {"dt": 0},
    {"ibi_dt": np.nan}, {"damping": 1.1}, {"damping": 0},
    {"regularization": 1.}, {"regularization": -.1}, {"smoothing": True},
    {"smoothing": -1.}, {"gamma_centroid": 0}, {"iterations": 1.5},
    {"checkpoint_interval": 0}, {"sample_interval": 9}, {"sampling_steps": True},
    {"base_identity": "unversioned-pes"},
])
def test_invalid_training_options_fail_explicitly(change):
    options = _options()
    options.update(change)
    with pytest.raises(ValueError):
        fit_fqcmd(**options)


def test_target_grid_overflow_fails_before_sampling():
    options = _options()
    options["grids"] = (np.linspace(.7, 1., 10), options["grids"][1])
    with pytest.raises(ValueError, match="outside grid.*initial geometry"):
        fit_fqcmd(**options)


@pytest.mark.parametrize("damage", ["counts", "shape", "clock", "phase", "rng"])
def test_semantically_invalid_integrity_checked_checkpoint_is_rejected(tmp_path, damage):
    options = _options()
    path = tmp_path / "fit.npz"
    fit_fqcmd(**options, checkpoint=path)
    with load_checkpoint(path) as saved:
        payload = {name: saved[name] for name in saved.files}
    meta = json.loads(str(payload["metadata"].item()))
    if damage == "counts":
        payload["target_bond"][0] += 1
    elif damage == "shape":
        payload["q"] = payload["q"][:, :2]
    elif damage == "clock":
        meta["step"] += 1
    elif damage == "phase":
        meta["phase"] = "unrecognized"
    else:
        meta["operators"]["rng"]["bit_generator"] = "not-a-bit-generator"
    payload["metadata"] = json.dumps(meta)
    atomic_savez(path, kind=FIT_KIND, **payload)
    with pytest.raises(ValueError):
        fit_fqcmd(**options, checkpoint=path)


def test_zero_updates_has_final_classical_validation_without_correction():
    options = _options(1)
    options["iterations"] = 0
    model = fit_fqcmd(**options)
    assert len(model.diagnostics) == 1
    for value in model.corrections:
        np.testing.assert_array_equal(value, np.zeros_like(value))


@pytest.mark.parametrize("attributes", [
    {"required_bond_constraints": [(0, 1, 1.7)]},
    {"metadata": {"required_bond_constraints": [(0, 1, 1.7)]}},
    {"pbc": [True, False, False]}, {"cell": np.eye(3)},
    {"metadata": {"boundary_conditions": "periodic"}},
    {"metadata": {"centroid_pmf_beta_au": 100.}},
])
def test_unsupported_base_physics_rejected_before_reference_sampling(attributes):
    class ForbiddenPotential:
        def evaluate(self, q):
            raise AssertionError("unsupported base physics must fail before PIMD")

    potential = ForbiddenPotential()
    for name, value in attributes.items():
        setattr(potential, name, value)
    options = _options()
    options["potential"] = potential
    with pytest.raises(ValueError):
        fit_fqcmd(**options)


@pytest.mark.parametrize("metadata", [{"supports_pbc": True}, {"periodic": True}])
def test_periodic_capability_metadata_does_not_imply_active_periodicity(metadata):
    options = _options(1)
    options["potential"].metadata = metadata
    options["iterations"] = 0
    assert fit_fqcmd(**options).training_beads == 1
