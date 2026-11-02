"""Independent geometry/unit and optional real-source CBE adapter checks."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.hch4.build_pes import check_source
from benchmarks.hch4.pes import CBE2008Potential
from benchmarks.hch4.verify import paper_geometries, verify
from pyrpmd.constants import BOHR_ANGSTROM
from pyrpmd.potentials import PotentialError


def test_checksum_rejects_same_size_tampering(tmp_path):
    data = b"fixed-format source\r\n"
    record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
    path = tmp_path / "source.f"
    path.write_bytes(data)
    assert check_source(path, record) == path
    path.write_bytes(data.replace(b"fixed", b"FIXED"))
    with pytest.raises(ValueError, match="checksum"):
        check_source(path, record)


def test_table_III_geometries_are_constructed_with_the_right_bonds():
    reactant, saddle = (q*BOHR_ANGSTROM for q in paper_geometries())
    np.testing.assert_allclose(np.linalg.norm(reactant[[0, 2, 3, 4]]-reactant[1], axis=1), 1.0890, atol=1e-14)
    assert np.linalg.norm(saddle[3]-saddle[1]) == pytest.approx(1.3903)
    assert np.linalg.norm(saddle[5]-saddle[3]) == pytest.approx(0.8728)
    cosine = np.dot(saddle[0]-saddle[1], saddle[3]-saddle[1]) / (1.0901*1.3903)
    assert np.rad2deg(np.arccos(cosine)) == pytest.approx(106.82)
    assert np.dot(saddle[1]-saddle[3], saddle[5]-saddle[3]) < 0


@pytest.fixture
def real_library():
    path = os.environ.get("PYRPMD_CBE2008_LIBRARY")
    if not path:
        pytest.skip("Set PYRPMD_CBE2008_LIBRARY to the locally built pinned PES; no implicit download")
    return Path(path)


def test_real_cbe_paper_barrier_and_gradient_rounding(real_library):
    result = verify(real_library)
    assert result["passed"], result["gates"]
    assert len(result["replay"]["points_bohr"]) == 19
    # The raw source is deliberately preserved; it is NOT an exact gradient
    # of its energy conversion. This assertion prevents silent coefficient edits.
    assert result["units"]["expected_gradient_energy_scale_ratio"] == pytest.approx(0.9999979377703975, abs=1e-15)
    detailed = result["finite_difference"]["tests"][2]
    assert detailed["maximum_absolute_error_hartree_per_bohr"] > 5e-8
    assert detailed["maximum_residual_after_known_upstream_scale"] < 3e-9


def test_real_library_hash_receipt_is_required_before_loading(real_library, tmp_path):
    copied = tmp_path / "libcbe2008.so"
    copied.write_bytes(real_library.read_bytes())
    receipt = json.loads(real_library.with_suffix(".build.json").read_text())
    receipt["library_sha256"] = "0"*64
    copied.with_suffix(".build.json").write_text(json.dumps(receipt))
    with pytest.raises(PotentialError, match="provenance"):
        CBE2008Potential(copied)
