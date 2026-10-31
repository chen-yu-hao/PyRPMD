"""Public input/JSON guarantees for complex SOC and the physical EF bath."""
import json

import numpy as np
import pytest

from pyrpmd.input import JobSpec
from pyrpmd.runner import prepare, run_resolved
from test_materials_input_cv import job as materials_job
from test_nrpmd_input import job as collision_job


def test_complex_soc_prepare_is_lossless_json(tmp_path):
    path = tmp_path / "soc.py"
    path.write_text('''import numpy as np
from pyrpmd.nonadiabatic import DiabaticResult
class PES:
 def evaluate(self,q):
  b,n,_=q.shape
  matrix=np.zeros((b,2,2),complex)
  matrix[:,0,1]=.2+1j*q[:,0,0]
  matrix[:,1,0]=matrix[:,0,1].conj()
  gradients=np.zeros((b,2,2,n,3),complex)
  gradients[:,0,1,0,0]=1j
  gradients[:,1,0,0,0]=-1j
  return DiabaticResult(matrix,gradients)
def make(): return PES()
''')
    raw = collision_job()
    raw["potential"]["factory"] = f"{path}:make"
    result = prepare(raw, numerical=True)
    restored = json.loads(json.dumps(result, allow_nan=False))["pes_check"]
    matrix = np.array(restored["matrix_hartree"]) + 1j * np.array(restored["matrix_imag_hartree"])
    np.testing.assert_array_equal(matrix, matrix.swapaxes(1, 2).conj())
    assert matrix[0, 0, 1].real == .2
    assert restored["max_absolute_gradient_error"] < 1e-8


def test_friction_prepare_evaluates_physical_tensor():
    raw = materials_job("diffusion")
    raw["electronic_friction"] = {"gamma": ".01 fs^-1"}
    result = prepare(raw, numerical=True)
    assert result["materials_preparation"]["production_method"] == "centroid_ef_rpmd"
    assert np.array(result["friction_check"]["tensor_atomic_units"]).shape == (1, 3, 3)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("spec", [
    {}, {"method": "bead", "gamma": .1}, {"provider": "unknown"},
    {"gamma": -.1}, {"gamma": .1, "tensor": np.eye(3).tolist()},
    {"provider": "python", "gamma": .1, "factory": "a:b"},
])
def test_invalid_friction_input_rejected_before_output(spec, tmp_path):
    raw = materials_job("diffusion")
    raw["electronic_friction"] = spec
    with pytest.raises(ValueError):
        JobSpec.from_mapping(raw).resolve()


def test_friction_revalidated_on_direct_resolved_calls(tmp_path):
    raw = materials_job("diffusion")
    raw["electronic_friction"] = {"gamma": .1}
    config, system = JobSpec.from_mapping(raw).resolve()
    config["task"] = "pimd"
    with pytest.raises(ValueError, match="task='diffusion'"):
        run_resolved(config, system, output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_pimd_cannot_silently_ignore_electronic_friction():
    raw = materials_job("pimd")
    raw["electronic_friction"] = {"gamma": .1}
    with pytest.raises(ValueError, match="requires task='diffusion'"):
        JobSpec.from_mapping(raw).resolve()
