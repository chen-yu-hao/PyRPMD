"""Independent fixed-q references and backend equivalence, including SOC."""
import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.electronic_backends import (
    electronic_mapping_flow, electronic_backend_capabilities,
    torch_electronic_mapping_flow, validate_electronic_backend,
)


def inputs(states=3, batch=3, seed=1729, degenerate=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, states, states))+1j*rng.normal(size=(batch, states, states))
    v = 0.5*(x+x.swapaxes(1, 2).conj())
    if degenerate:
        v = np.broadcast_to(np.eye(states)*0.31, v.shape).copy()
    x = rng.normal(size=(batch, states, states, 2, 3))+1j*rng.normal(size=(batch, states, states, 2, 3))
    g = 0.5*(x+x.swapaxes(1, 2).conj())
    z = rng.normal(size=(batch, states))+1j*rng.normal(size=(batch, states))
    return z, v, g


def reference(z, v, g, duration, gamma):
    nodes, weights = np.polynomial.legendre.leggauss(32)
    rotated = np.array([expm(-1j*duration*a)@c for c, a in zip(z, v)])
    delta = np.zeros(g.shape[:1]+g.shape[-2:])
    for node, weight in zip(nodes, weights):
        time = 0.5*duration*(node+1)
        for b in range(len(z)):
            c = expm(-1j*time*v[b])@z[b]
            force = -.5*np.einsum("i,ijac,j->ac", c.conj(), g[b], c).real
            force += .5*gamma*np.einsum("iiac->ac", g[b]).real
            delta[b] += .5*duration*weight*force
    return rotated, delta


@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("states", [1, 2, 3, 6])
@pytest.mark.parametrize("degenerate", [False, True])
@pytest.mark.parametrize("gamma", [1., np.sqrt(3)-1])
def test_flow_matches_independent_exponential_force_quadrature(backend, states, degenerate, gamma):
    if backend == "native" and not electronic_backend_capabilities()["native"]:
        pytest.skip("optional native electronic extension not built")
    z, v, g = inputs(states, degenerate=degenerate)
    old = [a.copy() for a in (z, v, g)]
    actual = electronic_mapping_flow(z, v, g, .41, gamma=gamma, backend=backend)
    expected = reference(z, v, g, .41, gamma)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-13, rtol=2e-13)
    for a, b in zip((z, v, g), old):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_reverse_flow_and_complex_unitary_covariance(backend):
    if backend == "native" and not electronic_backend_capabilities()["native"]:
        pytest.skip("optional native electronic extension not built")
    z, v, g = inputs()
    out, dp = electronic_mapping_flow(z, v, g, .27, backend=backend)
    back, dpback = electronic_mapping_flow(out, v, g, -.27, backend=backend)
    np.testing.assert_allclose(back, z, atol=8e-14)
    np.testing.assert_allclose(dp+dpback, 0., atol=8e-14)
    u = expm(1j*v[0])
    z2 = z@u.conj()
    v2 = u.conj().T@v@u
    g2 = np.einsum("ai,bijnc,jk->baknc", u.conj().T, g, u)
    out2, dp2 = electronic_mapping_flow(z2, v2, g2, .27, backend=backend)
    np.testing.assert_allclose(out2, out@u.conj(), atol=8e-14)
    np.testing.assert_allclose(dp2, dp, atol=8e-14)


def test_leading_trajectory_and_bead_axes_and_scalar_input():
    z, v, g = inputs(batch=8)
    expected = electronic_mapping_flow(z, v, g, .17)
    actual = electronic_mapping_flow(z.reshape(2,4,3), v.reshape(2,4,3,3),
                                     g.reshape(2,4,3,3,2,3), .17)
    np.testing.assert_array_equal(actual[0].reshape(8,3), expected[0])
    np.testing.assert_array_equal(actual[1].reshape(8,2,3), expected[1])
    scalar = electronic_mapping_flow(z[0], v[0], g[0], .17)
    np.testing.assert_allclose(scalar[0], expected[0][0], atol=1e-14)
    np.testing.assert_allclose(scalar[1], expected[1][0], atol=1e-14)


def test_native_thread_count_is_bitwise_deterministic():
    pytest.importorskip("pyrpmd._electronic")
    z, v, g = inputs(batch=19)
    one = electronic_mapping_flow(z, v, g, .2, backend="native", threads=1)
    three = electronic_mapping_flow(z, v, g, .2, backend="native", threads=3)
    for a, b in zip(one, three):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("field,value", [("duration", True), ("gamma", float("nan")),
    ("threads", True), ("threads", 0), ("backend", "gpu"), ("device", "cuda")])
def test_invalid_controls_fail(field, value):
    z, v, g = inputs()
    args = {"duration": .1, field: value}
    with pytest.raises((ValueError, RuntimeError)):
        electronic_mapping_flow(z, v, g, **args)


@pytest.mark.parametrize("which", ["matrix", "gradient", "nonfinite", "shape"])
def test_invalid_physics_inputs_fail(which):
    z, v, g = inputs()
    if which == "matrix": v[0,0,1] += .01j
    if which == "gradient": g[0,0,1,0,0] += .01j
    if which == "nonfinite": z[0,0] = np.nan
    if which == "shape": g = g[..., :2]
    with pytest.raises(ValueError):
        electronic_mapping_flow(z, v, g, .1)


def test_native_rejects_alias_and_wrong_dtype():
    native = pytest.importorskip("pyrpmd._electronic")
    z, v, g = inputs()
    e, u = np.linalg.eigh(v)
    e, u = np.ascontiguousarray(e), np.ascontiguousarray(u)
    out = np.zeros_like(z)
    dp = np.zeros((3,6))
    with pytest.raises(ValueError, match="alias"):
        native.mapping_flow(e,u,z,g.reshape(3,3,3,6),.1,1.,z,dp,1)
    with pytest.raises(ValueError, match="dtype"):
        native.mapping_flow(e.astype(np.float32),u,z,g.reshape(3,3,3,6),.1,1.,out,dp,1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degenerate", [False, True])
def test_torch_device_resident_and_numpy_bridge(device, degenerate):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; CPU path is independently checked")
    z, v, g = inputs(degenerate=degenerate)
    expected = reference(z, v, g, .19, np.sqrt(3)-1)
    tensors = [torch.tensor(a, dtype=torch.complex128, device=device) for a in (z,v,g)]
    actual = torch_electronic_mapping_flow(*tensors, .19, gamma=np.sqrt(3)-1)
    assert all(a.device.type == device for a in actual)
    bridge = electronic_mapping_flow(z,v,g,.19,gamma=np.sqrt(3)-1,backend="torch",device=device)
    for a, b, c in zip(actual, expected, bridge):
        np.testing.assert_allclose(a.cpu().numpy(),b,atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(c,b,atol=2e-12,rtol=2e-12)


def test_requested_cuda_never_silently_runs_on_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            validate_electronic_backend("torch", "cuda")
