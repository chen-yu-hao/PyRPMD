"""Exercise the public .o build, .so inspect and force-check commands."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pyrpmd.cli import main

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc"), "C compiler required")
class NativeCLITests(unittest.TestCase):
    def test_object_build_output_and_force_check(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            obj = base / "test.o"
            library = base / "libtest.so"
            manifest = base / "pes.toml"
            geometry = base / "atom.xyz"
            subprocess.run(
                [
                    "cc",
                    "-fPIC",
                    "-I",
                    str(ROOT / "native/include"),
                    "-c",
                    str(ROOT / "native/examples/harmonic_pes.c"),
                    "-o",
                    str(obj),
                ],
                check=True,
                capture_output=True,
            )
            manifest.write_text('compiler="cc"\nobjects=["test.o"]\n')
            geometry.write_text("1\ntest atom in Angstrom\nH 0.1 0.2 0.3\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "pes",
                        "build",
                        str(obj),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(library),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(library.is_file())
            self.assertEqual(json.loads(output.getvalue())["library"], str(library))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["pes", "check", str(library), "--geometry", str(geometry)])
            self.assertEqual(code, 0)
            self.assertLess(json.loads(output.getvalue())["max_force_error_au"], 1e-8)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "pes",
                            "build",
                            str(obj),
                            "--manifest",
                            str(manifest),
                            "--output",
                            str(library),
                        ]
                    ),
                    2,
                )
