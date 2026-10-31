#!/usr/bin/env python3
"""End-to-end real JaxPIP export/native propagation validation and timings.

Small deterministic water-symmetry models test the machinery only. They are
not trained chemical potentials, and timings are not production speed claims.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork
from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import CABIPotential, check_potential

BASIS = [[[0, 0, 0]], [[0, 0, 1], [0, 1, 0]], [[1, 0, 0]],
         [[0, 1, 1]], [[1, 0, 1], [1, 1, 0]], [[0, 0, 2], [0, 2, 0]], [[2, 0, 0]]]
WATER = np.array([[0., 0.78383672, 0.44340501], [0., -0.78383672, 0.44340501], [0., 0., -0.11085125]])


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    basis = output / "basis.json"
    basis.write_text(json.dumps(BASIS))
    xyz = WATER[None] + np.random.default_rng(7).normal(0., 0.025, (8, 3, 3))
    geometries = output / "validation.npy"
    np.save(geometries, xyz)
    descriptor = PolynomialDescriptor(BASIS, alpha=1.0, dtype=jnp.float64)
    linear = PolynomialLinearModel(descriptor, coeffs=jnp.linspace(-0.05, 0.05, 7))
    nn = PolynomialNeuralNetwork(descriptor, hidden_layers=[4], key=jax.random.PRNGKey(5), activation="tanh")
    nn = nn.update_scaler(jax.vmap(descriptor)(jnp.asarray(xyz)), jnp.linspace(-0.02, 0.02, len(xyz)))
    library = os.environ["PYRPMD_ONNX_LIBRARY"]
    report = {}
    for kind, model in (("linear", linear), ("nn", nn)):
        checkpoint = output / f"{kind}.eqx"
        model.save(str(checkpoint))
        onnx_file = output / f"{kind}.onnx"
        result = subprocess.run([sys.executable, str(root / "tools/export_jaxpip_onnx.py"),
            "--basis", str(basis), "--checkpoint", str(checkpoint),
            "--geometries", str(geometries), "--output", str(onnx_file), "--kind", kind,
            "--atomic-numbers", "1", "1", "8", "--charge", "0", "--multiplicity", "1",
            "--position-unit", "angstrom", "--energy-unit", "eV"],
            check=False, text=True, capture_output=True)
        (output / f"{kind}-export.log").write_text(result.stdout + result.stderr)
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr, flush=True)
            result.check_returncode()
        # Use a fresh process for conversion, keeping this JAX reference clean.
        @eqx.filter_jit
        def reference(q):
            return jax.vmap(model.get_energy_and_forces)(q)
        start = time.perf_counter()
        pes = CABIPotential(library, {"model": str(onnx_file.resolve()), "device": "cpu"})
        load_seconds = time.perf_counter()-start
        errors, timings = {}, {}
        with pes:
            for batch in (1, 8, 32, 128):
                q = xyz[np.arange(batch) % len(xyz)] / BOHR_ANGSTROM
                e, f = jax.device_get(reference(jnp.asarray(q*BOHR_ANGSTROM)))
                expected_e, expected_f = e*EV_TO_AU, f*EV_TO_AU*BOHR_ANGSTROM
                got = pes.evaluate(q)
                np.testing.assert_allclose(got.energy, expected_e, atol=1e-10, rtol=1e-8)
                np.testing.assert_allclose(got.forces, expected_f, atol=1e-10, rtol=1e-8)
                errors[str(batch)] = {"energy_hartree": float(np.max(np.abs(got.energy-expected_e))),
                                      "forces_hartree_per_bohr": float(np.max(np.abs(got.forces-expected_f)))}
                # Warm both paths; device_get synchronizes JAX. Separate load/JIT.
                pes.evaluate(q)
                jax.device_get(reference(jnp.asarray(q*BOHR_ANGSTROM)))
                repeats = 50
                start = time.perf_counter()
                for _ in range(repeats):
                    pes.evaluate(q)
                onnx_seconds = (time.perf_counter()-start)/repeats
                start = time.perf_counter()
                for _ in range(repeats):
                    jax.device_get(reference(jnp.asarray(q*BOHR_ANGSTROM)))
                jax_seconds = (time.perf_counter()-start)/repeats
                timings[str(batch)] = {"native_abi_from_python_seconds": onnx_seconds,
                                      "jax_jit_from_python_seconds": jax_seconds}
            q = np.repeat(xyz[:1], 8, axis=0) / BOHR_ANGSTROM
            force_check = check_potential(pes, q[:1])
            assert force_check["max_absolute_force_error"] < 1e-7
            dyn = Dynamics(pes, [1800., 1800., 29000.], 220., 8, 0.1)
            p = np.random.default_rng(8).normal(scale=0.1, size=q.shape)
            a, b = State(q.copy(), p.copy()), State(q.copy(), p.copy())
            start = time.perf_counter()
            native = dyn.propagate_block(a, 100, backend="native", record_states=True)
            native_seconds = time.perf_counter()-start
            start = time.perf_counter()
            python = dyn.propagate_block(b, 100, backend="python", record_states=True)
            python_seconds = time.perf_counter()-start
            np.testing.assert_allclose(native.q, python.q, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(native.p, python.p, atol=1e-9, rtol=1e-10)
            drift = float(np.max(np.abs(native.energies-native.energies[0])))
            assert drift < 1e-6
        report[kind] = {"load_seconds": load_seconds, "batch_errors": errors,
                        "force_check": force_check, "timings": timings,
                        "native_block_100_steps_seconds": native_seconds,
                        "python_block_100_steps_seconds": python_seconds,
                        "energy_range_hartree": drift}
        print(json.dumps({kind: report[kind]}, indent=2), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
