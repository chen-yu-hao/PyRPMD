"""Optional real training/export/autograd route; no mock learned models."""

import importlib.util
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("method", ["deep-lda", "deep-tica"])
def test_mlcolvar_training_torchscript_centroid_bias_end_to_end(tmp_path, method):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    pytest.importorskip("mlcolvar", minversion="1.2")
    if not version("mlcolvar").startswith("1."):
        pytest.skip("mlcolvar 2.x training APIs are outside this example's declared version range")
    from pyrpmd.advanced_cv import TorchScriptCV
    from pyrpmd.bias import UmbrellaBias

    example = Path(__file__).parents[1] / "examples" / "mlcolvar_cv.py"
    spec = importlib.util.spec_from_file_location("mlcolvar_example", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_dtype = torch.get_default_dtype()
    model, artifact = module.train_and_export(tmp_path / f"{method}.pt", method=method,
                                              n_samples=128, epochs=2, seed=123)
    assert torch.get_default_dtype() == original_dtype
    cv = TorchScriptCV(artifact, dimension=1, units=("dimensionless",),
                       periods=(None,), estimator="centroid", device="cpu")
    q = np.array([[[0.2, -0.1, 0.3]], [[0.4, 0.1, -0.2]], [[0.1, 0.3, 0.1]]])
    result = cv.value_and_jacobian(q)
    assert result.value.shape == (1,)
    assert result.jacobian.shape == (1, 3, 1, 3)
    assert result.value.dtype == result.jacobian.dtype == np.float64
    with torch.no_grad():
        expected = model(torch.as_tensor(q.mean(axis=0)[0, :2][None], dtype=torch.float64)).numpy()[0]
    np.testing.assert_allclose(result.value, expected, rtol=1e-12, atol=1e-12)
    numerical = np.empty_like(result.jacobian)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[(slice(None),) + index] = (cv.value(plus) - cv.value(minus)) / 2e-6
    np.testing.assert_allclose(result.jacobian, numerical, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(result.jacobian[:, 0], result.jacobian[:, 1], atol=1e-14)
    bias = UmbrellaBias(cv, [0.1], 2.)
    _, force = bias.energy_force(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        derivative = (bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / 2e-6
        assert force[index] == pytest.approx(-derivative, rel=2e-5, abs=1e-7)
