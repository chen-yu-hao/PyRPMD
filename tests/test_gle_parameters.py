"""Synthetic imports and independent analytic checks; no external matrix data."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

from pyrpmd.constants import FS_TO_AU, KB_AU, WAVENUMBER_TO_AU
from pyrpmd.gle_parameters import (
    generate_optimal_gle, load_gle4md, resolve_gle_parameters,
    validate_gle_parameters,
)


def write(tmp_path, text, name="parameters.dat"):
    path = tmp_path / name
    # Exact bytes: text-mode newline translation on Windows would change the
    # file contents that the source-metadata SHA256 is compared against.
    path.write_bytes(text.encode())
    return path


def test_official_raw_grammar_units_and_original_source_metadata(tmp_path):
    text = """# Synthetic canonical parameters, created for this test.
# T=300 K
# Use for simulations with 4 beads, using a RPMD (physical masses) representation
# &omega;max=4000&nbsp;cm^-1
# A MATRIX: (picoseconds^-1)
1.0D+00  2.0
-2.0     1.0
# C MATRIX: (K)
1200 0
0 1200
"""
    source = write(tmp_path, text)
    parameters = load_gle4md(source)
    np.testing.assert_allclose(parameters["A"], np.array([[1., 2.], [-2., 1.]]) / (1000 * FS_TO_AU))
    np.testing.assert_allclose(parameters["C"], np.eye(2), rtol=2e-16, atol=0)
    assert parameters["beta"] == 1 / (300 * KB_AU)
    assert parameters["beads"] == 4
    metadata = parameters["parameter_metadata"]
    assert metadata["source_files"][0]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert "Synthetic canonical parameters, created for this test." in metadata["source_files"][0]["comments"]
    assert metadata["frequency_range_au"] == [None, 4000 * WAVENUMBER_TO_AU]
    assert metadata["residuals"] is None
    assert metadata["fit_converged"] is None
    json.dumps(parameters, allow_nan=False)


def test_separate_headerless_covariance_and_inline_comments(tmp_path):
    a = write(tmp_path, "2 1 # row one\n-1 2\n", "bath.a")
    c = write(tmp_path, "1 0\n0 1\n", "bath.c")
    result = load_gle4md(a, covariance_path=c, drift_unit="autime^-1", covariance_unit="dimensionless")
    np.testing.assert_array_equal(result["A"], [[2, 1], [-1, 2]])
    np.testing.assert_array_equal(result["C"], np.eye(2))
    assert len(result["parameter_metadata"]["source_files"]) == 2
    assert result["parameter_metadata"]["source_files"][0]["comments"] == ["row one"]


@pytest.mark.parametrize("unit,factor", [("femtoseconds^-1", 1 / FS_TO_AU),
                                         ("atomic time units^-1", 1.),
                                         ("cm^-1", WAVENUMBER_TO_AU)])
def test_drift_unit_aliases_are_rates_without_an_extra_two_pi(tmp_path, unit, factor):
    result = load_gle4md(write(tmp_path, f"# A MATRIX: ({unit})\n3\n"))
    assert result["A"][0][0] == pytest.approx(3 * factor, rel=2e-15)


def piglet_text(order=None):
    order = range(4) if order is None else order
    text = "# T=300 K\n# Use for simulations with 4 beads, using a RPMD representation\n"
    text += "# A MATRIX: (atomic time units^-1)\n"
    for k in order:
        text += f"# Matrix for normal mode: {k}\n{k + 1}\n"
    text += "# C MATRIX: (K)\n"
    for k in order:
        text += f"# Matrix for normal mode: {k}\n{1200 * (k + 1)}\n"
    return text


def test_piglet_import_per_mode_kelvin_normalization_and_explicit_reordering(tmp_path):
    source = write(tmp_path, piglet_text())
    real = load_gle4md(source, mode="piglet")
    np.testing.assert_allclose(np.asarray(real["C"])[:, 0, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(real["A"])[:, 0, 0], [1, 2, 3, 4])
    fourier = load_gle4md(source, mode="piglet", mode_order="fourier")
    np.testing.assert_array_equal(np.asarray(fourier["A"])[:, 0, 0], [1, 2, 4, 3])
    np.testing.assert_allclose(np.asarray(fourier["C"])[:, 0, 0], [1, 2, 4, 3])
    assert fourier["parameter_metadata"]["mode_order"] == "real"
    assert fourier["parameter_metadata"]["source_mode_order"] == "fourier"
    # Labels, rather than block arrival order, identify the original modes.
    shuffled = load_gle4md(write(tmp_path, piglet_text([2, 0, 3, 1]), "shuffled.dat"), mode="piglet")
    np.testing.assert_array_equal(shuffled["A"], real["A"])


def test_dimensional_energy_covariance_uses_bead_temperature(tmp_path):
    text = "# A MATRIX: (autime^-1)\n1\n# C MATRIX: (atomic energy units)\n0.5\n"
    result = load_gle4md(write(tmp_path, text), beta=8., beads=4)
    np.testing.assert_array_equal(result["C"], [[1.]])
    # Omitting P is an error; no silent division by the physical temperature.
    with pytest.raises(ValueError, match="beta and beads"):
        load_gle4md(write(tmp_path, text), beta=8.)


@pytest.mark.parametrize("unit,source_factor", [("eV", 11604.506), ("atomic energy units", 315774.67)])
def test_official_historical_energy_serialization_is_explicit(tmp_path, unit, source_factor):
    # Synthetic matrices serialized using the public exporter algorithm, not
    # any third-party parameter table. The website's rounded constants differ
    # from present physical constants, enough to break a strict centroid test.
    text = "# Generated at http://cosmo-epfl.github.io/gle4md\n# T=300 K\n"
    text += "# Use for simulations with 2 beads, using a RPMD representation\n"
    text += "# A MATRIX: (atomic time units^-1)\n"
    text += "# Matrix for normal mode: 0\n2.4188843e-5\n# Matrix for normal mode: 1\n2.4188843e-5\n"
    text += f"# C MATRIX: ({unit})\n# Matrix for normal mode: 0\n{600 / source_factor:.17g}\n"
    text += f"# Matrix for normal mode: 1\n{1200 / source_factor:.17g}\n"
    path = write(tmp_path, text)
    with pytest.raises(ValueError, match="unit_convention='gle4md'"):
        load_gle4md(path, mode="piglet")
    result = load_gle4md(path, mode="piglet", unit_convention="gle4md")
    np.testing.assert_allclose(np.asarray(result["C"])[:, 0, 0], [1., 2.], rtol=2e-15)
    np.testing.assert_allclose(np.asarray(result["A"])[:, 0, 0], 1 / (1000 * FS_TO_AU))
    assert result["parameter_metadata"]["units"]["convention"] == "gle4md"
    assert result["parameter_metadata"]["units"]["source_converters"]["energy_to_kelvin"]["ev"] == 11604.506


def test_inconsistent_header_frequency_range_is_rejected(tmp_path):
    text = "# omega_min=2 autime^-1\n# omega_max=1 autime^-1\n# A MATRIX: (autime^-1)\n1\n"
    with pytest.raises(ValueError, match="frequency range"):
        load_gle4md(write(tmp_path, text))


@pytest.mark.parametrize("text,kwargs,match", [
    ("1\n", {}, "unit is missing"),
    ("# A MATRIX: (fs^-1)\n1\n", {"drift_unit": "ps^-1"}, "conflicts"),
    ("# A MATRIX: (THz)\n1\n", {}, "ambiguous"),
    ("# A MATRIX: (autime^-1)\n1\n# C MATRIX: (K)\n1\n", {}, "beta and beads"),
    ("# A MATRIX: (autime^-1)\n1\n# C MATRIX: (dimensionless)\n", {}, "empty"),
    ("# A MATRIX: (autime^-1)\n", {}, "empty"),
    ("# A MATRIX: (autime^-1)\n1 0\n1\n", {}, "square"),
    ("# A MATRIX: (autime^-1)\nnan\n", {}, "numerical"),
    ("# A MATRIX: (autime^-1)\n1e999\n", {}, "nonfinite"),
    ("# A MATRIX: (autime^-1)\n1+2j\n", {}, "numerical"),
    ("# A MATRIX: (autime^-1)\n# Matrix for normal mode: 1\n1\n", {}, "contiguous"),
    ("# A MATRIX: (autime^-1)\n# Matrix for normal mode: 0\n1\n# Matrix for normal mode: 0\n2\n", {}, "duplicate"),
    ("# A MATRIX: (autime^-1)\n# Matrix for normal mode: 0\n# Matrix for normal mode: 1\n1\n", {}, "empty"),
    ("# A MATRIX: (autime^-1)\n# Matrix for normal mode: -1\n1\n", {}, "header"),
    ("# A MATRIX: (autime^-1)\n1\n# A MATRIX: (autime^-1)\n1\n", {}, "duplicate"),
    ("# A MATRIX: (autime^-1)\n-1\n", {}, "positive real"),
    ("# A MATRIX: (autime^-1)\n1 10\n0 1\n", {}, "diffusion"),
    ("# A MATRIX: (autime^-1)\n1\n# C MATRIX: (dimensionless)\n2\n", {}, "unit physical"),
])
def test_import_rejects_ambiguous_or_unphysical_files(tmp_path, text, kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_gle4md(write(tmp_path, text), **kwargs)


@pytest.mark.parametrize("kwargs", [{"beta": 2.}, {"beads": 3}, {"mode_order": "guess"}])
def test_piglet_calibration_conflicts_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        load_gle4md(write(tmp_path, piglet_text()), mode="piglet", **kwargs)


def test_no_covariance_substitution_for_piglet_or_conflicting_source(tmp_path):
    a = write(tmp_path, "# A MATRIX: (autime^-1)\n1\n")
    with pytest.raises(ValueError, match="explicit C"):
        load_gle4md(a, mode="piglet", beta=2., beads=4)
    combined = write(tmp_path, "# A MATRIX: (autime^-1)\n1\n# C MATRIX: (dimensionless)\n1\n", "combined.dat")
    with pytest.raises(ValueError, match="both"):
        load_gle4md(combined, covariance_path=a)


def test_scalar_optimal_sampling_against_independent_harmonic_lyapunov_integral():
    spec = generate_optimal_gle([.2, 5.], grid_size=17)
    gamma = spec["A"][0][0]
    assert gamma == pytest.approx(1.)
    metadata = spec["parameter_metadata"]
    assert metadata["fit_converged"] is True
    for frequency, reported in zip(metadata["frequency_grid_au"], metadata["residuals"]["tau_V_au"]):
        drift = np.array([[0., -1.], [frequency**2, gamma]])
        covariance = solve_continuous_lyapunov(drift, np.diag([0., 2 * gamma]))
        integral = solve_continuous_lyapunov(drift, np.outer(covariance[:, 0], covariance[:, 0]))
        tau = integral[0, 0] / covariance[0, 0]**2
        assert reported == pytest.approx(tau, rel=2e-12)
    def worst_efficiency(rate):
        return min(2 / (frequency / rate + rate / frequency) for frequency in (.2, 5.))
    assert worst_efficiency(gamma) > worst_efficiency(gamma * 1.001)
    assert worst_efficiency(gamma) > worst_efficiency(gamma / 1.001)


def test_generator_frequency_units_and_covariance_validation():
    wave = generate_optimal_gle([100, 4000], frequency_unit="cm^-1")
    assert wave["A"][0][0] == pytest.approx(math.sqrt(100 * 4000) * WAVENUMBER_TO_AU)
    hz = generate_optimal_gle([1, 4], frequency_unit="THz")
    # The physical band is angular, whereas the imported drift is already a rate.
    assert hz["A"][0][0] == pytest.approx(4 * math.pi / (1000 * FS_TO_AU))
    diagnostics = validate_gle_parameters({"A": [[1., 2.], [-2., 1.]], "C": np.eye(2)})
    assert diagnostics["minimum_diffusion_eigenvalue_au"] == 2.
    with pytest.raises(ValueError, match="positive definite"):
        validate_gle_parameters({"A": np.eye(2), "C": [[1., 0.], [0., 0.]]})


@pytest.mark.parametrize("band", [[0, 1], [-1, 2], [2, 1], [1], [True, False], [True, 2], [1, "2"], [1, np.inf], [1, np.nan], [1j, 2]])
def test_generator_strict_frequency_ranges(band):
    with pytest.raises(ValueError):
        generate_optimal_gle(band)


@pytest.mark.parametrize("settings", [{"naux": True}, {"naux": 4}, {"grid_size": 2}, {"max_nfev": 0}])
def test_generator_strict_counts(settings):
    with pytest.raises(ValueError):
        generate_optimal_gle([1, 2], **settings)


def test_new_parameters_do_not_coerce_mixed_boolean_matrices():
    with pytest.raises(ValueError, match="real numerical"):
        validate_gle_parameters({"A": [[True, 1], [-1, 2]], "C": np.eye(2)})


def test_parameter_metadata_roundtrip_and_legacy_spec_preservation():
    old = {"A": np.array([[1., 2.], [-2., 1.]]), "provenance": "legacy"}
    result = resolve_gle_parameters(old)
    assert set(result) == set(old)
    np.testing.assert_array_equal(result["A"], old["A"])
    result["A"][0, 0] = 123
    assert old["A"][0, 0] == 1
    new = generate_optimal_gle([1, 2])
    before = copy.deepcopy(new)
    resolved = resolve_gle_parameters(json.loads(json.dumps(new)))
    assert resolved == before
    new["parameter_metadata"]["invalid"] = np.nan
    with pytest.raises(ValueError, match="finite JSON"):
        resolve_gle_parameters(new)
    new["parameter_metadata"] = {2: "not a JSON identity key"}
    with pytest.raises(ValueError, match="keys"):
        resolve_gle_parameters(new)


@pytest.mark.parametrize("spec", [
    {"file": "a", "generator": "optimal-sampling"},
    {"generator": "invented"},
    {"generator": "optimal-sampling"},
    {"generator": "optimal-sampling", "frequency_range": [1, 2], "silent_fallback": True},
    {"generator": "optimal-sampling", "frequency_range": [1, 2], "A": [[1]]},
    {"file": "a", "A": [[1]]},
    {"A": [[1]], "parameter_metadata": {}},
])
def test_declaration_failures_are_explicit(spec):
    with pytest.raises(ValueError):
        resolve_gle_parameters(spec)
