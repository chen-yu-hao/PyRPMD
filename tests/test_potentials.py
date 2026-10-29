import json
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
from pyrpmd.potentials import *
from pyrpmd.potentials.optional import TorchPotential
from pyrpmd.potentials.socket import IPIPotential

ROOT = Path(__file__).resolve().parents[1]


class PotentialTests(unittest.TestCase):
    def test_analytic_forces(self):
        x = np.random.default_rng(4).normal(size=(3, 2, 3))
        x[:, 1, 0] += 3
        for pes in (
            HarmonicPotential(0.8),
            MorsePairPotential(),
            DoubleWellPotential(),
        ):
            with self.subTest(pes=type(pes).__name__):
                self.assertLess(
                    check_potential(pes, x)["max_absolute_force_error"], 1e-8
                )

    def test_batch_scalar(self):
        x = np.random.default_rng(2).normal(size=(5, 2, 3))
        p = HarmonicPotential(0.3)
        result = p.evaluate(x)
        np.testing.assert_array_equal(
            result.energy, np.concatenate([p.evaluate(a[None]).energy for a in x])
        )

    def test_units_and_gradient_sign(self):
        from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE

        def gradient(x):
            return 0.5 * np.sum(x * x, axis=(1, 2)), x

        pes = PythonPotential(
            gradient,
            units={"positions": "angstrom", "energy": "eV"},
            derivative="gradient",
        )
        x = np.ones((2, 1, 3))
        r = pes.evaluate(x)
        np.testing.assert_allclose(r.forces, -x * EV_HARTREE * BOHR_ANGSTROM**2)
        self.assertLess(check_potential(pes, x)["max_absolute_force_error"], 1e-11)

    def test_invalid_results(self):
        for result in [
            (np.zeros(2), np.zeros((1, 1, 3))),
            (np.ones(1), np.full((1, 1, 3), np.nan)),
        ]:
            with self.assertRaises(PotentialError):
                validate_result(PESResult(*result), (1, 1, 3))
        with self.assertRaises(PotentialError):
            PESResult(np.ones(1) * 1j, np.zeros((1, 1, 3)))
        result = PESResult([np.nan], np.full((1, 1, 3), np.nan), [2], ["out of domain"])
        validate_result(result, (1, 1, 3), require_success=False)
        with self.assertRaises(PotentialError):
            validate_result(result, (1, 1, 3))

    def test_python_factory_and_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pes.py"
            path.write_text(
                "import numpy as np\ninitialized=False\ndef initialize_potential():\n global initialized\n initialized=True\n return 0\ndef get_potential(q):\n assert initialized\n return .5*np.sum(q*q,axis=(0,1)),q,0\ndef factory():\n return lambda x: (.5*np.sum(x*x,axis=(1,2)),-x)\n"
            )
            x = np.arange(18, dtype=float).reshape(2, 3, 3)
            old = RPMDratePotential(str(path))
            new = make_potential(
                {"provider": "python", "factory": str(path) + ":factory"}
            )
            a = old.evaluate(x)
            b = new.evaluate(x)
            np.testing.assert_allclose(a.energy, b.energy)
            np.testing.assert_allclose(a.forces, b.forces)

    def test_gpu_missing_is_explicit(self):
        try:
            import torch

            if torch.cuda.is_available():
                self.skipTest("GPU available; separate hardware integration test")
        except ImportError:
            pass
        with self.assertRaises(BackendUnavailableError):
            TorchPotential("missing.pt", device="cuda")


@unittest.skipUnless(shutil.which("cc"), "C compiler required for ABI integration")
class CABITests(unittest.TestCase):
    def test_so_and_object_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            obj = base / "harmonic.o"
            lib = base / "harmonic.so"
            source = ROOT / "native/examples/harmonic_pes.c"
            inc = ROOT / "native/include"
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-fPIC",
                    "-I",
                    str(inc),
                    "-c",
                    str(source),
                    "-o",
                    str(obj),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["cc", "-shared", str(obj), "-o", str(lib)],
                check=True,
                capture_output=True,
            )
            system = {
                "symbols": ["H"],
                "atomic_numbers": [1],
                "charge": 0,
                "multiplicity": 1,
            }
            x = np.random.default_rng(5).normal(size=(8, 1, 3))
            with CABIPotential(lib, system=system) as p:
                r = p.evaluate(x)
                np.testing.assert_allclose(r.forces, -x)
                self.assertLess(
                    check_potential(p, x)["max_absolute_force_error"], 1e-10
                )
            with self.assertRaises(PotentialError):
                p.evaluate(x)
            manifest = base / "pes.toml"
            manifest.write_text('compiler="cc"\nobjects=["harmonic.o"]\n')
            linked = build_object(obj, manifest, base / "cache")
            self.assertEqual(linked, build_object(obj, manifest, base / "cache"))
            with CABIPotential(linked, system=system) as p:
                np.testing.assert_allclose(p.evaluate(x).forces, -x)
            with self.assertRaises(PotentialError):
                CABIPotential(lib, system={"symbols": ["O"], "atomic_numbers": [8]})


class SocketTests(unittest.TestCase):
    def test_ipi_protocol(self):
        pes = IPIPotential(port=0, timeout=5)
        errors = []

        def driver():
            try:
                s = socket.create_connection(pes.address)
                state = "NEEDINIT"
                force = None
                energy = None

                def recv(n):
                    out = b""
                    while len(out) < n:
                        chunk = s.recv(n - len(out))
                        if not chunk:
                            raise EOFError()
                        out += chunk
                    return out

                while True:
                    h = recv(12).strip()
                    if h == b"STATUS":
                        s.sendall(state.encode().ljust(12))
                    elif h == b"INIT":
                        _, n = struct.unpack("=ii", recv(8))
                        recv(n)
                        state = "READY"
                    elif h == b"POSDATA":
                        recv(18 * 8)
                        n = struct.unpack("=i", recv(4))[0]
                        x = np.frombuffer(recv(n * 3 * 8), dtype="=f8").reshape(n, 3)
                        force = -x
                        energy = 0.5 * np.sum(x * x)
                        state = "HAVEDATA"
                    elif h == b"GETFORCE":
                        s.sendall(
                            b"FORCEREADY".ljust(12)
                            + struct.pack("=di", energy, len(force))
                            + force.astype("=f8").tobytes()
                            + np.zeros(9, dtype="=f8").tobytes()
                            + struct.pack("=i", 0)
                        )
                        state = "READY"
                    elif h == b"EXIT":
                        break
                    else:
                        raise ValueError(h)
                s.close()
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=driver)
        t.start()
        try:
            x = np.arange(18, dtype=float).reshape(2, 3, 3)
            r = pes.evaluate(x)
            np.testing.assert_allclose(r.forces, -x)
        finally:
            pes.close()
            t.join(6)
        self.assertFalse(t.is_alive())
        self.assertFalse(errors, errors)


class TorchCPUTests(unittest.TestCase):
    """Actual optional-runtime tests, independent of CUDA availability."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("optional PyTorch runtime not installed") from exc
        cls.torch = torch

    def test_torchscript_autograd_and_units(self):
        torch = self.torch

        class Harmonic(torch.nn.Module):
            def forward(self, q):
                return 0.7 * torch.sum(q * q, dim=(1, 2))

        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "harmonic.pt"
            traced = torch.jit.trace(
                Harmonic(), torch.ones((2, 3, 3), dtype=torch.float64)
            )
            traced.save(str(model))
            x = np.random.default_rng(73).normal(size=(4, 3, 3))
            from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE

            with TorchPotential(
                model, device="cpu", units={"positions": "angstrom", "energy": "eV"}
            ) as pes:
                result = pes.evaluate(x)
                scale = EV_HARTREE * BOHR_ANGSTROM**2
                np.testing.assert_allclose(
                    result.energy, 0.7 * np.sum(x * x, axis=(1, 2)) * scale
                )
                np.testing.assert_allclose(result.forces, -1.4 * x * scale)
                # Check a fresh batch after autograd: no graph/gradient accumulation.
                np.testing.assert_allclose(
                    pes.evaluate(2 * x).forces, 2 * result.forces
                )
                self.assertLess(
                    check_potential(pes, x)["max_absolute_force_error"], 2e-10
                )
                self.assertEqual(pes.metadata["execution_device"], "cpu")
                self.assertEqual(pes.metadata["dtype"], "float64")

    def test_factory_direct_forces_and_detached_energy(self):
        with tempfile.TemporaryDirectory() as td:
            factory = Path(td) / "torch_models.py"
            factory.write_text(
                "import torch\n"
                "class Direct(torch.nn.Module):\n"
                " def forward(self,q):\n"
                '  return {"energy":torch.sum(q*q,dim=(1,2)),"forces":-2*q}\n'
                "class Detached(torch.nn.Module):\n"
                " def forward(self,q): return torch.sum(q*q,dim=(1,2)).detach()\n"
                "class Reduced(torch.nn.Module):\n"
                " def forward(self,q): return torch.sum(q*q,dim=(1,2)).float()\n"
                "def direct(device):\n"
                ' assert device=="cpu"\n'
                " return Direct()\n"
                "def detached(device): return Detached()\n"
                "def reduced(device): return Reduced()\n"
            )
            x = np.random.default_rng(14).normal(size=(2, 1, 3))
            with TorchPotential(factory=str(factory) + ":direct", device="cpu") as pes:
                np.testing.assert_allclose(pes.evaluate(x).forces, -2 * x)
            with TorchPotential(
                factory=str(factory) + ":detached", device="cpu"
            ) as pes:
                with self.assertRaisesRegex(PotentialError, "detached energy"):
                    pes.evaluate(x)
            with TorchPotential(factory=str(factory) + ":reduced", device="cpu") as pes:
                with self.assertRaisesRegex(PotentialError, "float64 precision"):
                    pes.evaluate(x)


class ASETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from ase.calculators.lj import LennardJones
        except ImportError as exc:
            raise unittest.SkipTest("optional ASE runtime not installed") from exc

    def test_lennard_jones_batch_atomic_units_and_force_difference(self):
        from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE

        epsilon, sigma, cutoff = 0.03, 1.1, 4.4  # ASE native units: eV and angstrom.
        spec = {
            "provider": "ase",
            "factory": "ase.calculators.lj:LennardJones",
            "params": {"epsilon": epsilon, "sigma": sigma, "rc": cutoff},
        }
        x = np.zeros((3, 2, 3))
        x[:, 1] = (
            np.array([[1.4, 0.0, 0.0], [1.5, 0.2, 0.0], [1.7, 0.1, -0.2]])
            / BOHR_ANGSTROM
        )
        with make_potential(spec, {"symbols": ["Ar", "Ar"]}) as pes:
            result = pes.evaluate(x)
            delta = x[:, 1] * BOHR_ANGSTROM
            r = np.linalg.norm(delta, axis=1)
            ratio6 = (sigma / r) ** 6
            cutoff6 = (sigma / cutoff) ** 6
            expected = (
                4 * epsilon * (ratio6**2 - ratio6 - cutoff6**2 + cutoff6) * EV_HARTREE
            )
            force1 = (
                (24 * epsilon * (2 * ratio6**2 - ratio6) / r**2)[:, None]
                * delta
                * EV_HARTREE
                * BOHR_ANGSTROM
            )
            np.testing.assert_allclose(result.energy, expected, rtol=2e-13, atol=1e-15)
            np.testing.assert_allclose(
                result.forces[:, 1], force1, rtol=2e-13, atol=1e-15
            )
            np.testing.assert_allclose(
                result.forces[:, 0], -force1, rtol=2e-13, atol=1e-15
            )
            scalar = [pes.evaluate(frame[None]) for frame in x]
            np.testing.assert_allclose(
                result.energy, np.concatenate([v.energy for v in scalar])
            )
            np.testing.assert_allclose(
                result.forces, np.concatenate([v.forces for v in scalar])
            )
            self.assertLess(check_potential(pes, x)["max_absolute_force_error"], 1e-9)
            self.assertEqual(pes.metadata["execution_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
