"""Opt-in physical consistency checks for the exact EAM4 static reference.

Set PYRPMD_EAM4_STATIC_LIBRARY to a library built by the repository's isolated
build_eam4.py. The default suite does not download or compile third-party code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.materials_reference.build_eam4 import checked_source, suffix
from benchmarks.materials_reference.eam4 import EAM4StaticReference
from benchmarks.materials_reference.verify_eam4 import PROTOCOL, verify


@pytest.fixture(scope="module")
def reference():
    library = os.environ.get("PYRPMD_EAM4_STATIC_LIBRARY")
    if not library:
        pytest.skip("opt-in exact EAM4 static library is not configured")
    return EAM4StaticReference(library)


def test_suffix_excludes_only_the_front_wrapper():
    assert suffix(b"front wrapper\nsubroutine eamnih_forces_umb(a)\nunchanged\n") == b"subroutine eamnih_forces_umb(a)\nunchanged\n"


def test_altered_original_source_rejected(tmp_path):
    candidate = tmp_path / "unverified.f90"
    candidate.write_text("subroutine eamnih_forces_umb(a)\nend\n")
    with pytest.raises(ValueError, match="length/SHA256"):
        checked_source(candidate, None)


def test_predeclared_static_checks(reference):
    report = verify(reference.library_path)
    assert report["all_static_checks_passed"], report
    assert report["paper_reproduction"] is False
    assert len(report["cases"]) == 4
    assert len(report["cutoff_probes"]) == 4


@pytest.mark.parametrize("mobile", [-1, 9, 1.5, True])
def test_invalid_partition_rejected(reference, mobile):
    q = json.loads(PROTOCOL.read_text())["q_angstrom"]
    with pytest.raises(ValueError, match="partition|integer"):
        reference.evaluate(q, mobile_ni=mobile)


def test_coincident_periodic_images_rejected(reference):
    q = [[0., 0., 0.], [24., 0., 0.]]
    with pytest.raises(ValueError, match="coincident"):
        reference.evaluate(q, mobile_ni=1, periodic_xy=True)


def test_undersized_periodic_cell_rejected(reference):
    with pytest.raises(ValueError, match="twice"):
        reference.evaluate([[0., 0., 0.], [2., 0., 0.]], mobile_ni=1,
                           periodic_xy=True, lengths_angstrom=(19., 24., 24.))


def test_energy_only_does_not_return_invented_derivatives(reference):
    q = np.asarray(json.loads(PROTOCOL.read_text())["q_angstrom"])
    energy, gradient = reference.evaluate(q, mobile_ni=8, energy_only=True)
    assert np.isfinite(energy)
    assert np.isnan(gradient).all()
