"""CPU and CUDA verification for JAXDLPackPotential, JaxPIPPotential, and Torch rate."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU  # noqa: E402
from pyrpmd.input import JobSpec  # noqa: E402
from pyrpmd.potentials import (  # noqa: E402
    BackendUnavailableError,
    JAXDLPackPotential,
    JaxPIPPotential,
    PotentialError,
    check_potential,
    make_potential,
)
from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes  # noqa: E402
from pyrpmd.system import System  # noqa: E402
from pyrpmd.torch_backend import TorchDynamics, TorchState  # noqa: E402
from pyrpmd.workflows.rate import run_rate  # noqa: E402


def _has_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


CUDA_AVAILABLE = _has_cuda()
DEVICES = ["cpu", "cuda:0"] if CUDA_AVAILABLE else ["cpu"]
HAVE_JAXPIP = importlib.util.find_spec("jaxpip") is not None
cuda_only = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not available")
jaxpip_only = pytest.mark.skipif(not HAVE_JAXPIP, reason="optional JaxPIP package not installed")


def _anharmonic_triatomic_energy(q, *, k2=0.35, k4=0.04, r0=1.45):
    """Smooth float64 3-atom pairwise + 3-body anharmonic energy for single frame [N, 3]."""
    d01 = jnp.linalg.norm(q[0] - q[1]) - r0
    d12 = jnp.linalg.norm(q[1] - q[2]) - r0
    d02 = jnp.linalg.norm(q[0] - q[2]) - 2.0 * r0
    bend = jnp.sum((q[0] - 2.0 * q[1] + q[2]) ** 2)
    return (
        0.5 * k2 * (d01**2 + d12**2)
        + 0.25 * k4 * (d01**4 + d12**4)
        + 0.08 * d02**2
        + 0.05 * bend
    )


def _batched_anharmonic_energy(qb):
    """Batched [F, N, 3] -> [F] JAX energy function."""
    d01 = jnp.linalg.norm(qb[:, 0, :] - qb[:, 1, :], axis=-1) - 1.45
    d12 = jnp.linalg.norm(qb[:, 1, :] - qb[:, 2, :], axis=-1) - 1.45
    d02 = jnp.linalg.norm(qb[:, 0, :] - qb[:, 2, :], axis=-1) - 2.90
    bend = jnp.sum((qb[:, 0, :] - 2.0 * qb[:, 1, :] + qb[:, 2, :]) ** 2, axis=-1)
    return 0.5 * 0.35 * (d01**2 + d12**2) + 0.25 * 0.04 * (d01**4 + d12**4) + 0.08 * d02**2 + 0.05 * bend


def _periodic_pair_energy(q, cell, pbc):
    """Periodic minimum-image smooth pair potential supporting affine cell strain."""
    inv_cell = jnp.linalg.inv(cell)
    disp = q[:, None, :] - q[None, :, :]
    frac = disp @ inv_cell
    wrapped_frac = jnp.where(pbc[None, None, :], frac - jnp.round(frac), frac)
    cart = wrapped_frac @ cell
    n = q.shape[0]
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    r2 = jnp.sum(cart**2, axis=-1)
    pair_e = jnp.where(mask, 0.25 * jnp.exp(-0.35 * r2) + 0.03 * r2, 0.0)
    return jnp.sum(pair_e)


def _sample_positions(frames=6, atoms=3, seed=19):
    rng = np.random.default_rng(seed)
    base = np.array([[-1.45, 0.05, -0.02], [0.0, -0.03, 0.04], [1.48, 0.02, -0.01]])
    if atoms != 3:
        base = np.linspace(-1.2, 1.2, atoms * 3).reshape(atoms, 3)
    return base[None] + rng.normal(scale=0.12, size=(frames, atoms, 3))


@pytest.mark.parametrize("device", DEVICES)
def test_jax_dlpack_energy_only_callable_numpy_and_torch_parity(device):
    q = _sample_positions(frames=9, atoms=3, seed=7)
    provider = make_potential(
        {
            "provider": "jax-dlpack",
            "model": _anharmonic_triatomic_energy,
            "device": device,
            "atomic_numbers": [1, 1, 1],
        }
    )
    try:
        assert jax.config.read("jax_enable_x64") is True
        assert provider.metadata["provider"] == "jax-dlpack"
        assert provider.metadata["execution_device"] == device
        assert provider.metadata["dtype"] == "float64"
        assert provider.metadata["batch_independent"] is True

        # Finite-difference check on NumPy contract.
        fd_err = check_potential(provider, q[:2], epsilon=1e-6)["max_absolute_force_error"]
        assert fd_err < 1e-8

        np_res = provider.evaluate(q)
        assert np_res.energy.dtype == np.float64
        assert np_res.forces.dtype == np.float64
        assert np_res.virial is None

        # Direct as_torch_pes() and make_workflow_torch_pes() parity.
        q_t = torch.tensor(q, dtype=torch.float64, device=device)
        direct_pes = provider.as_torch_pes()
        workflow_pes = make_workflow_torch_pes(provider, device=device)
        for adapter in (direct_pes, workflow_pes):
            t_res = adapter.evaluate_tensor(q_t)
            assert t_res.energy.dtype == torch.float64
            assert t_res.forces.dtype == torch.float64
            assert str(t_res.energy.device) == str(torch.device(device))
            assert t_res.virial is None
            np.testing.assert_allclose(
                t_res.energy.cpu().numpy(), np_res.energy, rtol=1e-13, atol=1e-13
            )
            np.testing.assert_allclose(
                t_res.forces.cpu().numpy(), np_res.forces, rtol=1e-13, atol=1e-13
            )
    finally:
        provider.close()


@pytest.mark.parametrize("device", DEVICES)
def test_jax_dlpack_zero_copy_dlpack_pointer_identity(device):
    """Verify DLPack handoff shares exact buffer pointers without host/device copies."""
    observed = {}

    def energy_and_forces(q1):
        e, g = jax.value_and_grad(_anharmonic_triatomic_energy)(q1)
        return e, -g

    provider = JAXDLPackPotential(energy_and_forces, device=device)
    try:
        provider.warmup_torch(4, 3)
        module = provider._torch_module
        orig_eval = provider._evaluate_jax_batch

        def wrapped_eval(q_jax):
            observed["q_jax_ptr"] = q_jax.unsafe_buffer_pointer()
            out = orig_eval(q_jax)
            observed["e_jax_ptr"] = out[0].unsafe_buffer_pointer()
            observed["f_jax_ptr"] = out[1].unsafe_buffer_pointer()
            return out

        # Rebuild module with wrapped_eval to inspect exact pointers at the DLPack boundary.
        provider._torch_module = None
        provider._evaluate_jax_batch = wrapped_eval
        pes = provider.as_torch_pes()
        module = provider._torch_module

        q_t = torch.tensor(_sample_positions(4, 3), dtype=torch.float64, device=device).contiguous()
        e_t, f_t = module(q_t)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        assert q_t.data_ptr() == observed["q_jax_ptr"]
        assert e_t.data_ptr() == observed["e_jax_ptr"]
        assert f_t.data_ptr() == observed["f_jax_ptr"]
        assert pes.evaluate_tensor(q_t).forces.shape == (4, 3, 3)
    finally:
        provider.close()


@pytest.mark.parametrize("device", DEVICES)
def test_jax_dlpack_callable_variants_units_and_chunking(device, tmp_path):
    q = _sample_positions(frames=11, atoms=3, seed=23)
    ref_provider = JAXDLPackPotential(_anharmonic_triatomic_energy, device=device)
    ref = ref_provider.evaluate(q)
    ref_provider.close()

    # 1. derivative='gradient' tuple callable.
    grad_fn = lambda q1: jax.value_and_grad(_anharmonic_triatomic_energy)(q1)  # noqa: E731
    p_grad = JAXDLPackPotential(grad_fn, device=device, derivative="gradient")

    # 2. Batched [F, N, 3] callable with auto detection and max_frames_per_call chunking.
    p_batch = JAXDLPackPotential(
        _batched_anharmonic_energy, device=device, batched="auto", max_frames_per_call=3
    )

    # 3. Units conversion (angstrom / eV) + mapping output + plugin file loading via JobSpec.
    plugin = tmp_path / "angstrom_ev_pes.py"
    plugin.write_text(
        "import jax\n"
        "import jax.numpy as jnp\n"
        f"BOHR_ANG = {BOHR_ANGSTROM!r}\n"
        f"EV_AU = {EV_TO_AU!r}\n"
        "def make_pes(scale=1.0):\n"
        "    def energy_au(q_bohr):\n"
        "        d01 = jnp.linalg.norm(q_bohr[0] - q_bohr[1]) - 1.45\n"
        "        d12 = jnp.linalg.norm(q_bohr[1] - q_bohr[2]) - 1.45\n"
        "        d02 = jnp.linalg.norm(q_bohr[0] - q_bohr[2]) - 2.90\n"
        "        bend = jnp.sum((q_bohr[0] - 2.0 * q_bohr[1] + q_bohr[2]) ** 2)\n"
        "        return scale * (0.5 * 0.35 * (d01**2 + d12**2) + 0.25 * 0.04 * (d01**4 + d12**4) + 0.08 * d02**2 + 0.05 * bend)\n"
        "    def eval_ev_ang(q_ang):\n"
        "        e_au, g_bohr = jax.value_and_grad(energy_au)(q_ang / BOHR_ANG)\n"
        "        e_ev = e_au / EV_AU\n"
        "        f_ev_ang = -(g_bohr / EV_AU) / BOHR_ANG\n"
        "        return {'energy': e_ev, 'forces': f_ev_ang}\n"
        "    return eval_ev_ang\n",
        encoding="utf-8",
    )
    p_units = make_potential(
        {
            "provider": "jax-dlpack",
            "factory": f"{plugin}:make_pes",
            "params": {"scale": 1.0},
            "device": device,
            "position_unit": "angstrom",
            "energy_unit": "eV",
            "max_frames_per_call": 4,
        }
    )

    try:
        q_t = torch.tensor(q, dtype=torch.float64, device=device)
        for prov in (p_grad, p_batch, p_units):
            res_np = prov.evaluate(q)
            res_t = make_workflow_torch_pes(prov, device=device).evaluate_tensor(q_t)
            np.testing.assert_allclose(res_np.energy, ref.energy, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(res_np.forces, ref.forces, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                res_t.energy.cpu().numpy(), ref.energy, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                res_t.forces.cpu().numpy(), ref.forces, rtol=1e-12, atol=1e-12
            )
    finally:
        p_grad.close()
        p_batch.close()
        p_units.close()


@pytest.mark.parametrize("device", DEVICES)
def test_jax_dlpack_periodic_contract_and_strain_virial(device):
    cell = np.array([[7.5, 0.4, -0.2], [0.1, 8.0, 0.3], [-0.1, 0.2, 8.5]], dtype=np.float64)
    pbc = np.array([True, True, True])
    q = _sample_positions(frames=5, atoms=3, seed=31)

    provider = make_potential(
        {
            "provider": "jax-dlpack",
            "model": _periodic_pair_energy,
            "device": device,
            "periodic": True,
            "virial": "auto",
            "max_frames_per_call": 2,
        },
        {"cell": cell, "pbc": pbc, "atomic_numbers": [1, 1, 1]},
    )
    try:
        res_np = provider.evaluate(q)
        assert res_np.virial is not None
        assert res_np.virial.shape == (5, 3, 3)

        # Verify NumPy and TorchPES match on energy, forces, and virial.
        bound = make_workflow_torch_pes(provider, device=device)
        q_t = torch.tensor(q, dtype=torch.float64, device=device)
        res_t = bound.evaluate_tensor(q_t)
        assert res_t.virial is not None
        np.testing.assert_allclose(res_t.energy.cpu().numpy(), res_np.energy, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_t.forces.cpu().numpy(), res_np.forces, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_t.virial.cpu().numpy(), res_np.virial, rtol=1e-12, atol=1e-12)

        # Verify lattice periodic invariance: shifting an atom by integer cell vectors leaves E, F, W unchanged.
        shifted = q.copy()
        shifted[:, 0] += np.array([1.0, -2.0, 1.0]) @ cell
        res_shift = bound.evaluate_tensor(torch.tensor(shifted, dtype=torch.float64, device=device))
        np.testing.assert_allclose(
            res_shift.energy.cpu().numpy(), res_np.energy, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res_shift.forces.cpu().numpy(), res_np.forces, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res_shift.virial.cpu().numpy(), res_np.virial, rtol=1e-12, atol=1e-12
        )

        # Verify strain virial W_ab = -dE / d(eps_ab) against central finite differences.
        eps = 1e-6
        fd_virial = np.zeros_like(res_np.virial)
        for a in range(3):
            for b in range(3):
                strain_p = np.eye(3)
                strain_m = np.eye(3)
                strain_p[a, b] += eps
                strain_m[a, b] -= eps
                provider.set_cell(cell @ strain_p.T)
                ep = provider.evaluate(q @ strain_p.T).energy
                provider.set_cell(cell @ strain_m.T)
                em = provider.evaluate(q @ strain_m.T).energy
                fd_virial[:, a, b] = -(ep - em) / (2.0 * eps)
        provider.set_cell(cell)
        np.testing.assert_allclose(res_np.virial, fd_virial, rtol=1e-7, atol=1e-8)

        # Also check explicit stress -> virial = -stress * |det(cell)| conversion.
        vol = abs(np.linalg.det(cell))

        def stress_callable(q1, c1, p1):
            e, g = jax.value_and_grad(lambda x: _periodic_pair_energy(x, c1, p1))(q1)
            dummy_stress = jnp.eye(3, dtype=jnp.float64) * 0.02
            return {"energy": e, "forces": -g, "stress": dummy_stress}

        p_stress = JAXDLPackPotential(
            stress_callable, device=device, periodic=True, virial="provided", cell=cell, pbc=pbc
        )
        try:
            st_np = p_stress.evaluate(q[:2])
            st_t = p_stress.as_torch_pes().evaluate_tensor(q_t[:2])
            expected_w = np.broadcast_to(-0.02 * vol * np.eye(3), (2, 3, 3))
            np.testing.assert_allclose(st_np.virial, expected_w, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(st_t.virial.cpu().numpy(), expected_w, rtol=1e-13, atol=1e-13)
        finally:
            p_stress.close()
    finally:
        provider.close()


def test_jax_dlpack_strict_precision_and_device_rejection():
    # Reject float32 dtype option.
    with pytest.raises(ValueError, match="float64"):
        JAXDLPackPotential(_anharmonic_triatomic_energy, dtype="float32")

    # Reject float32 callable return value.
    with pytest.raises(ValueError, match="float64"):
        p = JAXDLPackPotential(lambda q: jnp.array(1.0, dtype=jnp.float32), atomic_numbers=[1, 1, 1])
        p.close()

    # Reject float32 captured PyTree leaf parameter.
    bad_leaf = jnp.ones((3, 3), dtype=jnp.float32)

    class Float32Callable:
        def __init__(self, w):
            self.w = w

        def __call__(self, q):
            return jnp.sum(q * self.w.astype(jnp.float64))

    jax.tree_util.register_pytree_node(
        Float32Callable, lambda obj: ((obj.w,), None), lambda _, children: Float32Callable(*children)
    )
    with pytest.raises(ValueError, match="float64 model parameters"):
        JAXDLPackPotential(Float32Callable(bad_leaf))

    # Reject unavailable CUDA device index with BackendUnavailableError (no CPU fallback).
    with pytest.raises(BackendUnavailableError, match="no CPU fallback|unavailable"):
        JAXDLPackPotential(_anharmonic_triatomic_energy, device="cuda:99")

    # Reject float32 Torch tensor, autocast, mismatched device, and closed provider.
    provider = JAXDLPackPotential(_anharmonic_triatomic_energy, device="cpu")
    pes = provider.as_torch_pes()
    q64 = torch.tensor(_sample_positions(2, 3), dtype=torch.float64, device="cpu")
    with pytest.raises(ValueError, match="float64"):
        pes.evaluate_tensor(q64.to(torch.float32))
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(PotentialError, match="autocast"):
            pes.evaluate_tensor(q64)
    if CUDA_AVAILABLE:
        with pytest.raises(ValueError, match="no device conversion"):
            make_workflow_torch_pes(provider, device="cuda:0")
    provider.close()
    with pytest.raises(PotentialError, match="closed"):
        provider.evaluate(_sample_positions(2, 3))
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate_tensor(q64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
def test_jax_dlpack_nve_ring_polymer_energy_conservation(device, integrator):
    """Acceptance test: user-defined float64 JAX energy conserves total RP energy in NVE TorchDynamics."""
    masses = np.array([1837.15, 1837.15, 1837.15], dtype=np.float64)
    beta = 1052.6  # ~300 K
    nbeads = 8
    dt = 2.0  # ~0.048 fs in atomic time units

    provider = make_potential(
        {
            "provider": "jax-dlpack",
            "model": _anharmonic_triatomic_energy,
            "device": device,
            "atomic_numbers": [1, 1, 1],
        }
    )
    try:
        provider.warmup_torch(2 * nbeads, 3)
        pes = make_workflow_torch_pes(provider, device=device)
        dyn = TorchDynamics(
            pes,
            masses,
            beta=beta,
            nbeads=nbeads,
            dt=dt,
            integrator=integrator,
            thermostat=None,
            seed=123,
            normal_mode_layout="rpmdrate" if integrator == "normal-mode-bab" else "rfft",
        )
        q0 = _sample_positions(frames=2 * nbeads, atoms=3, seed=55).reshape(2, nbeads, 3, 3)
        state = TorchState.from_arrays(q0, dyn.sample_momenta(2), device=dyn.device)

        e_init = dyn.energy(state).detach().cpu().numpy()
        energies = [e_init]
        for _ in range(8):
            dyn.propagate(state, 25)
            energies.append(dyn.energy(state).detach().cpu().numpy())
        energies = np.stack(energies, axis=0)
        rel_drift = np.max(np.abs(energies - e_init[None]) / np.abs(e_init[None]))
        assert rel_drift < 2e-4, f"Excessive NVE energy drift {rel_drift:.3e} on {device}/{integrator}"

        # Time-reversibility check.
        state.p.mul_(-1.0)
        dyn.propagate(state, 200)
        np.testing.assert_allclose(state.q.cpu().numpy(), q0, rtol=2e-5, atol=2e-5)
    finally:
        provider.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("provider_kind", ["jax-dlpack", "harmonic"])
def test_torch_rate_workflow_runs_with_jax_dlpack_and_widened_providers(
    device, provider_kind, tmp_path
):
    """End-to-end rate workflow with sampling.engine='torch' on CPU and CUDA."""
    plugin = tmp_path / "rate_pes.py"
    plugin.write_text(
        "import jax.numpy as jnp\n"
        "def energy(q):\n"
        "    d01 = jnp.linalg.norm(q[0] - q[1]) - 1.5\n"
        "    d12 = jnp.linalg.norm(q[1] - q[2]) - 1.5\n"
        "    return 0.02 * (d01**2 + d12**2)\n",
        encoding="utf-8",
    )
    if provider_kind == "jax-dlpack":
        pot_spec = {
            "provider": "jax-dlpack",
            "model": "rate_pes.py:energy",
            "device": device,
        }
    else:
        pot_spec = {"provider": "harmonic", "params": {"k": 0.02}}

    spec_dict = {
        "task": "rate",
        "temperature": "60000 K",
        "beads": 4,
        "seed": 17,
        "system": {
            "symbols": ["H", "H", "H"],
            "masses": [2.0, 2.0, 2.0],
            "geometry": [[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        },
        "potential": pot_spec,
        "dynamics": {"timestep": "0.003 autime", "integrator": "cayley-bcb"},
        "sampling": {
            "engine": "torch",
            "device": device,
            "thermostat": "andersen",
            "equilibration_steps": 20,
            "production_steps": 80,
            "sample_interval": 2,
            "andersen_interval": 10,
        },
        "reaction": {
            "fragments": [[0], [1, 2]],
            "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]],
            "rinf": "4.0 bohr",
        },
        "umbrella": {"centers": np.linspace(0.0, 1.0, 11).tolist(), "kappa": 10.0},
        "analysis": {"bins": 24, "bootstrap": 2},
        "output": {"checkpoint_steps": 40},
        "recrossing": {
            "dividing_surface": 1.0,
            "parent_replicas": 1,
            "parent_samples": 2,
            "equilibration_steps": 12,
            "parent_interval_steps": 4,
            "child_pairs": 2,
            "child_batch_pairs": 2,
            "child_steps": 12,
            "bootstrap": 4,
        },
    }
    resolved, system = JobSpec.from_mapping(spec_dict, base_directory=tmp_path).resolve()
    out_dir = tmp_path / f"run_{provider_kind}_{device.replace(':', '_')}"
    result = run_rate(resolved, system, out_dir)
    assert result["child_trajectories"] == 8
    assert np.isfinite(result["kappa"])
    assert np.isfinite(result["rate_estimate_cm3_s"]) and result["rate_estimate_cm3_s"] >= 0.0
    assert (out_dir / "recrossing.csv").is_file()
    replayed = run_rate(resolved, system, out_dir)
    assert replayed["kappa"] == pytest.approx(result["kappa"], rel=0, abs=1e-15)


@jaxpip_only
@pytest.mark.parametrize("device", DEVICES)
def test_jaxpip_fit_directory_backward_compatibility_and_delegation(device, tmp_path):
    """Acceptance test: JaxPIPPotential retains full pyrpmd.jaxpip-fit/1 compatibility."""
    from pyrpmd.pesgen.jaxpip_data import collect_dataset
    from pyrpmd.pesgen.jaxpip_fit import fit_jaxpip
    from pyrpmd.pesgen.reference import PySCFSettings, ReferencePoint

    def ref_point(r):
        q = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        e = -1.1 + 0.3 * np.exp(-r) + 0.05 * np.exp(-2.0 * r)
        deriv = -0.3 * np.exp(-r) - 0.1 * np.exp(-2.0 * r)
        grad = np.array([[-deriv, 0.0, 0.0], [deriv, 0.0, 0.0]])
        return ReferencePoint(
            ("H", "H"),
            q,
            e,
            grad,
            provenance={
                "engine": "PySCF",
                "version": "synthetic-test-fixture",
                "scf_converged": True,
                "settings": asdict(PySCFSettings(basis="sto-3g")),
                "test_fixture": True,
            },
        )

    splits = {
        "train": np.linspace(1.0, 2.0, 7),
        "validation": [1.1, 1.6],
        "test": [1.2, 1.85],
    }
    data_toml = 'schema="pyrpmd.jaxpip-data/1"\n[references]\n'
    for split, radii in splits.items():
        files = []
        for idx, r in enumerate(radii):
            fname = f"{split}-{idx}.json"
            ref_point(float(r)).save(tmp_path / fname)
            files.append(fname)
        data_toml += f"{split} = {json.dumps(files)}\n"
    (tmp_path / "data.toml").write_text(data_toml, encoding="utf-8")
    data_dir = tmp_path / "dataset"
    collect_dataset(tmp_path / "data.toml", data_dir)

    (tmp_path / "basis.json").write_text(json.dumps([[[0]], [[1]], [[2]], [[3]]]), encoding="utf-8")
    fit_toml = tmp_path / "fit.toml"
    fit_toml.write_text(
        f'schema="pyrpmd.jaxpip-fit/1"\n'
        f"dataset={json.dumps(str(data_dir))}\n"
        f'basis="basis.json"\n'
        f'basis_symbols=["H", "H"]\n'
        f"[fit]\n"
        f'kind="linear"\n'
        f"alpha=0.529177210903\n"
        f'position_unit="angstrom"\n'
        f'energy_unit="eV"\n'
        f"[acceptance]\n"
        f"energy_rmse_max=0.05\n"
        f"force_rmse_max=0.05\n",
        encoding="utf-8",
    )
    fit_dir = tmp_path / "model-fit"
    fit_jaxpip(fit_toml, fit_dir)

    system = System(("H", "H"), np.array([1837.15, 1837.15]), np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]))
    provider = make_potential(
        {
            "provider": "jaxpip",
            "model_directory": str(fit_dir),
            "device": device,
            "max_frames_per_call": 3,
        },
        system,
    )
    try:
        assert isinstance(provider, JaxPIPPotential)
        assert isinstance(provider._bridge, JAXDLPackPotential)
        assert provider.metadata["provider"] == "jaxpip-dlpack"
        assert provider.metadata["execution_device"] == device

        rng = np.random.default_rng(91)
        q = np.repeat(system.geometry[None], 7, axis=0) + rng.normal(scale=0.08, size=(7, 2, 3))
        assert check_potential(provider, q[:2], epsilon=1e-6)["max_absolute_force_error"] < 1e-7

        res_np = provider.evaluate(q)
        provider.warmup_torch(7, 2)
        pes = make_workflow_torch_pes(provider, device=device)
        res_t = pes.evaluate_tensor(torch.tensor(q, dtype=torch.float64, device=device))
        np.testing.assert_allclose(res_t.energy.cpu().numpy(), res_np.energy, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_t.forces.cpu().numpy(), res_np.forces, rtol=1e-12, atol=1e-12)
    finally:
        provider.close()


@cuda_only
@pytest.mark.parametrize("chunk_limit", [None, 5])
def test_cuda_non_default_stream_stress_and_multistream_switching(chunk_limit):
    """Acceptance test: non-default torch.cuda.Stream evaluation has zero races or stale reads."""
    device = torch.device("cuda:0")
    provider = JAXDLPackPotential(
        _anharmonic_triatomic_energy,
        device=device,
        max_frames_per_call=chunk_limit,
        atomic_numbers=(1, 1, 1),
    )
    try:
        provider.warmup_torch(16, 3)
        pes = make_workflow_torch_pes(provider, device=device)

        rng = np.random.default_rng(2026)
        batches_np = [_sample_positions(frames=16, atoms=3, seed=100 + i) for i in range(24)]
        base_tensors = [torch.tensor(b, dtype=torch.float64, device=device) for b in batches_np]

        # Compute synchronized ground-truth references on the default stream.
        torch.cuda.synchronize(device)
        ref_results = []
        for q_base in base_tensors:
            r = pes.evaluate_tensor(q_base)
            torch.cuda.synchronize(device)
            ref_results.append((r.energy.cpu().numpy(), r.forces.cpu().numpy()))

        # Part 1: Single non-default stream with heavy async matmul producer & consumer ops.
        work_stream = torch.cuda.Stream(device=device)
        mat_a = torch.randn((512, 512), dtype=torch.float64, device=device)
        mat_b = torch.randn((512, 512), dtype=torch.float64, device=device)
        outputs_single = []
        with torch.cuda.stream(work_stream):
            for idx, q_base in enumerate(base_tensors):
                # Heavy async GPU work producing a zero-mean perturbation -> exact q_base
                heavy = mat_a @ mat_b
                for _ in range(3):
                    heavy = heavy @ mat_a * 1e-3
                zero_pert = (heavy[0, 0] * 0.0).reshape(1, 1, 1)
                q_async = q_base + zero_pert
                res = pes.evaluate_tensor(q_async)
                # Immediately consume outputs on the non-default stream without host sync.
                e_copy = res.energy + zero_pert.reshape(1)
                f_copy = res.forces + zero_pert
                outputs_single.append((idx, e_copy, f_copy))
        work_stream.synchronize()

        for idx, e_gpu, f_gpu in outputs_single:
            ref_e, ref_f = ref_results[idx]
            np.testing.assert_allclose(e_gpu.cpu().numpy(), ref_e, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(f_gpu.cpu().numpy(), ref_f, rtol=1e-13, atol=1e-13)

        # Part 2: Rapid round-robin switching across 4 non-default streams + default stream.
        streams = [torch.cuda.Stream(device=device) for _ in range(4)] + [
            torch.cuda.default_stream(device=device)
        ]
        multi_outputs = []
        for step in range(60):
            idx = step % len(base_tensors)
            stream = streams[step % len(streams)]
            with torch.cuda.stream(stream):
                busy = (mat_a @ mat_b)[0, 0] * 0.0
                q_in = base_tensors[idx] + busy.reshape(1, 1, 1)
                res = pes.evaluate_tensor(q_in)
                # Downstream read on `stream` before CPU immediately switches to next stream!
                e_out = res.energy * 1.0 + busy.reshape(1)
                f_out = res.forces * 1.0 + busy.reshape(1, 1, 1)
                multi_outputs.append((idx, stream, e_out, f_out))

        for s in streams:
            s.synchronize()

        for idx, _stream, e_out, f_out in multi_outputs:
            ref_e, ref_f = ref_results[idx]
            np.testing.assert_allclose(e_out.cpu().numpy(), ref_e, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(f_out.cpu().numpy(), ref_f, rtol=1e-13, atol=1e-13)

        # Part 3: NVE TorchDynamics propagation entirely on a non-default CUDA stream.
        nve_stream = torch.cuda.Stream(device=device)
        masses = np.array([1837.15, 1837.15, 1837.15], dtype=np.float64)
        with torch.cuda.stream(nve_stream):
            dyn = TorchDynamics(
                pes, masses, beta=1052.6, nbeads=8, dt=2.0, integrator="cayley-bcb", seed=77
            )
            q_init = torch.tensor(
                _sample_positions(frames=16, atoms=3, seed=88).reshape(2, 8, 3, 3),
                dtype=torch.float64,
                device=device,
            )
            state = TorchState(q_init.clone(), dyn.sample_momenta(2))
            e0 = dyn.energy(state)
            dyn.propagate(state, 100)
            e1 = dyn.energy(state)
        nve_stream.synchronize()
        drift = torch.max(torch.abs(e1 - e0) / torch.abs(e0)).item()
        assert drift < 2e-4
    finally:
        provider.close()
