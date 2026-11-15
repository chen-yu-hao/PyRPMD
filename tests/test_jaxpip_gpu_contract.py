"""Preflight input, path-resolution, and resource contracts for JAX DLPack & Torch rate."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd import resources
from pyrpmd.input import JobSpec
from pyrpmd.input_schema import InputError
from pyrpmd.input_tasks import validate_task_sections
from pyrpmd.potentials import PotentialError, validate_periodic_spec
from pyrpmd.system import System
from pyrpmd.workflows.torch_materials import resolve_torch_materials


def _rate(provider="jaxpip", **potential_extra):
    pot = {"provider": provider, **potential_extra}
    if provider == "jaxpip" and not potential_extra:
        pot.update(basis="basis.json", model="model.eqx")
    elif provider == "jax-dlpack" and not potential_extra:
        pot.update(model="pes_mod:energy_fn", device="cuda:0")
    return {
        "task": "rate",
        "sampling": {"engine": "torch", "device": "cuda:0"},
        "potential": pot,
    }


def test_jaxpip_gpu_rate_contract_is_accepted_without_importing_accelerator():
    validate_periodic_spec(_rate()["potential"])
    assert validate_task_sections(_rate()) == (False, False, False)


@pytest.mark.parametrize(
    "provider,extra",
    [
        ("jaxpip", {"basis": "basis.json", "model": "model.eqx"}),
        ("jax-dlpack", {"model": "pkg.pes:energy_fn", "device": "cuda:0"}),
        ("torch", {"model": "model.pt", "device": "cuda:0"}),
        ("mace-batched", {"model": "mace.model", "device": "cuda:0"}),
        ("nequip-batched", {"model": "nequip.pth", "device": "cuda:0"}),
        ("fairchem-batched", {"model": "uma.pt", "device": "cuda:0"}),
        ("deepmd-batched", {"model": "graph.pt", "device": "cuda:0"}),
        ("metatomic", {"model": "mta.pt", "device": "cuda:0"}),
        ("free", {}),
        ("harmonic", {"params": {"k": 0.5}}),
        ("periodic_cosine", {}),
        ("builtin", {"name": "harmonic"}),
    ],
)
def test_torch_rate_widened_provider_validation_accepts_tensor_workflow_providers(
    provider, extra
):
    assert validate_task_sections(_rate(provider, **extra)) == (False, False, False)


@pytest.mark.parametrize(
    "provider,extra",
    [
        ("cabi", {"library": "libpes.so"}),
        ("python", {"function": "mod:pes"}),
        ("ase", {"factory": "mod:calc"}),
        ("lammps", {"model": "in.lammps"}),
        ("openmm", {"model": "system.xml"}),
        ("mace", {"model": "mace.model"}),
        ("morse_pair", {}),
        ("double_well", {}),
        ("builtin", {"name": "morse_pair"}),
    ],
)
def test_torch_rate_rejects_host_only_and_unsupported_providers_at_preflight(
    provider, extra
):
    with pytest.raises(InputError, match="tensor workflow|builtin potential"):
        validate_task_sections(_rate(provider, **extra))


@pytest.mark.parametrize("provider", ["jax-dlpack", "mace-batched", "torch"])
def test_torch_rate_rejects_float32_potential_at_preflight(provider):
    with pytest.raises(InputError, match="float64"):
        validate_task_sections(_rate(provider, model="m", dtype="float32"))
    with pytest.raises(InputError, match="float64"):
        validate_task_sections(
            _rate(provider, model="m", params={"dtype": "float32"})
        )


def test_jaxpip_spec_rejects_periodic_systems_before_model_load():
    with pytest.raises(PotentialError, match="nonperiodic"):
        validate_periodic_spec(
            _rate()["potential"],
            {
                "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
                "pbc": [True, True, True],
            },
        )


@pytest.mark.parametrize("field", ["allow_unvalidated", "max_frames_per_call"])
def test_jaxpip_runtime_guards_are_checked_before_model_load(field):
    config = _rate()["potential"]
    config[field] = "invalid" if field == "allow_unvalidated" else 0
    with pytest.raises(PotentialError, match="JaxPIP"):
        validate_periodic_spec(config)


def test_jax_dlpack_periodic_and_option_preflight_validation():
    base = {"provider": "jax-dlpack", "model": "pkg.mod:energy", "device": "cuda:0"}
    validate_periodic_spec(base)

    with pytest.raises(PotentialError, match="requires potential.model"):
        validate_periodic_spec({"provider": "jax-dlpack"})
    with pytest.raises(PotentialError, match="not both"):
        validate_periodic_spec(
            {"provider": "jax-dlpack", "factory": "a:b", "function": "c:d"}
        )
    with pytest.raises(PotentialError, match="float64"):
        validate_periodic_spec({**base, "dtype": "float32"})
    with pytest.raises(PotentialError, match="not both"):
        validate_periodic_spec({**base, "dtype": "float64", "params": {"dtype": "float64"}})
    with pytest.raises(PotentialError, match="cpu, cuda or cuda:<index>"):
        validate_periodic_spec({**base, "device": "mps"})
    with pytest.raises(PotentialError, match="periodic=true"):
        validate_periodic_spec(
            base,
            {"cell": np.eye(3) * 8.0, "pbc": [True, True, True]},
        )
    validate_periodic_spec(
        {**base, "periodic": True, "virial": "autograd"},
        {"cell": np.eye(3) * 8.0, "pbc": [True, True, True]},
    )
    with pytest.raises(PotentialError, match="virial"):
        validate_periodic_spec({**base, "virial": "bad"})
    with pytest.raises(PotentialError, match="derivative"):
        validate_periodic_spec({**base, "derivative": "hessian"})
    with pytest.raises(PotentialError, match="batched"):
        validate_periodic_spec({**base, "batched": "sometimes"})
    with pytest.raises(PotentialError, match="max_frames_per_call"):
        validate_periodic_spec({**base, "max_frames_per_call": 0})
    with pytest.raises(PotentialError, match="not both"):
        validate_periodic_spec(
            {**base, "units": {"positions": "bohr"}, "position_unit": "bohr"}
        )


def test_jobspec_resolves_jax_dlpack_and_jaxpip_paths_and_callables(tmp_path):
    plugin = tmp_path / "local_pes.py"
    plugin.write_text("def energy(q):\n    return 0.0\n", encoding="utf-8")
    fit_dir = tmp_path / "fit-model"
    fit_dir.mkdir()

    base_job = {
        "task": "rate",
        "temperature": "300 K",
        "beads": 4,
        "system": {
            "symbols": ["H", "H", "H"],
            "masses_amu": [1.008, 1.008, 1.008],
            "geometry": [[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        },
        "reaction": {
            "fragments": [[0], [1, 2]],
            "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]],
            "rinf": "4.0 bohr",
        },
        "umbrella": {"centers": [0.0, 0.5, 1.0], "kappa": 10.0},
        "sampling": {"engine": "torch", "thermostat": "andersen"},
    }

    # 1. Local .py plugin in model="local_pes.py:energy" resolves relative to base_directory.
    resolved_local, _ = JobSpec.from_mapping(
        {
            **base_job,
            "potential": {
                "provider": "jax-dlpack",
                "model": "local_pes.py:energy",
                "device": "cuda:0",
            },
        },
        base_directory=tmp_path,
    ).resolve()
    assert resolved_local["potential"]["model"] == f"{plugin.resolve()}:energy"
    assert resolved_local["sampling"]["device"] == "cuda:0"
    assert resolved_local["sampling"]["normal_mode_layout"] == "rfft"

    # 2. Importable module:object string is preserved without prepending base_directory.
    resolved_import, _ = JobSpec.from_mapping(
        {
            **base_job,
            "potential": {
                "provider": "jax-dlpack",
                "model": "some_pkg.pes:energy_fn",
                "device": "cpu",
            },
        },
        base_directory=tmp_path,
    ).resolve()
    assert resolved_import["potential"]["model"] == "some_pkg.pes:energy_fn"
    assert resolved_import["sampling"]["device"] == "cpu"

    # 3. Direct Python callable in JobSpec is preserved untouched.
    fn = lambda q: 0.0  # noqa: E731
    resolved_callable, _ = JobSpec.from_mapping(
        {
            **base_job,
            "potential": {"provider": "jax-dlpack", "model": fn, "device": "cpu"},
        },
        base_directory=tmp_path,
    ).resolve()
    assert resolved_callable["potential"]["model"] is fn

    # 4. JaxPIP model_directory resolves relative to base_directory and defaults device to cuda.
    resolved_jaxpip, _ = JobSpec.from_mapping(
        {
            **base_job,
            "potential": {"provider": "jaxpip", "model_directory": "fit-model"},
        },
        base_directory=tmp_path,
    ).resolve()
    assert resolved_jaxpip["potential"]["model_directory"] == str(fit_dir.resolve())
    assert resolved_jaxpip["sampling"]["device"] == "cuda"

    # 5. Invalid sampling.device or normal_mode_layout is rejected during resolution.
    with pytest.raises(InputError, match="sampling.device"):
        JobSpec.from_mapping(
            {
                **base_job,
                "potential": {"provider": "harmonic"},
                "sampling": {"engine": "torch", "device": "tpu:0"},
            },
            base_directory=tmp_path,
        ).resolve()
    with pytest.raises(InputError, match="normal_mode_layout"):
        JobSpec.from_mapping(
            {
                **base_job,
                "potential": {"provider": "harmonic"},
                "sampling": {"engine": "torch", "normal_mode_layout": "invalid"},
            },
            base_directory=tmp_path,
        ).resolve()


def test_torch_materials_accepts_jax_dlpack_and_jaxpip_providers():
    sys = System(
        ("H", "He"),
        np.array([2.0, 5.0]),
        np.array([[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]]),
    )
    for provider in ("jax-dlpack", "jaxpip"):
        cfg = {
            "task": "pimd",
            "beta": 2.0,
            "beads": 4,
            "dt_au": 0.04,
            "seed": 11,
            "potential": {"provider": provider, "model": "pkg:fn", "device": "cpu"},
            "sampling": {
                "engine": "torch",
                "device": "cpu",
                "normal_mode_layout": "rfft",
                "integrator": "bcocb",
                "thermostat": "pile-l",
                "gamma_centroid": 0.5,
                "equilibration_steps": 4,
                "production_steps": 8,
                "sample_interval": 2,
                "replicas": 2,
            },
            "dynamics": {"integrator": "cayley-bcb", "block_steps": 4},
            "output": {"checkpoint_steps": 4, "chunk_frames": 2},
        }
        options = resolve_torch_materials(cfg, sys)
        assert options["device"] == "cpu"


def test_resource_report_probes_jax_and_torch_independently(monkeypatch):
    fake_jax = SimpleNamespace(
        devices=lambda platform=None: (
            [SimpleNamespace(id=0, local_hardware_id=0)]
            if platform in (None, "cpu", "gpu")
            else []
        )
    )
    fake_torch = SimpleNamespace(
        device=lambda value: SimpleNamespace(
            type=value.split(":")[0],
            index=int(value.split(":")[1]) if ":" in value else None,
        ),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )

    def fake_import(name):
        if name == "jax":
            return fake_jax
        if name == "torch":
            return fake_torch
        raise ImportError(name)

    monkeypatch.setattr(resources.importlib, "import_module", fake_import)

    # Pure NumPy workflow with jax-dlpack only requires JAX, not Torch.
    rep_numpy = resources.resource_report(
        {"potential": {"provider": "jax-dlpack", "device": "cuda:0"}}
    )
    assert rep_numpy["available"] is True
    assert set(rep_numpy["checks"]) == {"jax"}

    # Torch rate workflow with jax-dlpack requires both JAX and Torch on the device.
    rep_torch = resources.resource_report(_rate("jax-dlpack"))
    assert rep_torch["available"] is True
    assert set(rep_torch["checks"]) == {"jax", "torch"}

    # Out-of-range GPU index is reported unavailable.
    rep_bad = resources.resource_report(
        _rate("jax-dlpack", model="pkg:fn", device="cuda:3")
        | {"sampling": {"engine": "torch", "device": "cuda:3"}}
    )
    assert rep_bad["available"] is False
    assert rep_bad["checks"]["jax"]["available"] is False
    assert rep_bad["checks"]["torch"]["available"] is False
