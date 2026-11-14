"""Validate and execute the Jupyter Notebook examples under examples/notebooks/."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "examples" / "notebooks"
EXPECTED_NOTEBOOKS = (
    "01_pes_potentials.ipynb",
    "02_reaction_rate.ipynb",
    "03_solid_state_materials.ipynb",
    "04_gpu_acceleration.ipynb",
    "05_jaxpip_cpp_cpu.ipynb",
    "06_jaxpip_jax_gpu.ipynb",
)


def _load_notebook(name: str) -> dict:
    path = NOTEBOOK_DIR / name
    assert path.is_file(), f"Missing notebook: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def test_notebook_index_and_schema():
    readme = NOTEBOOK_DIR / "README.md"
    assert readme.is_file()
    readme_text = readme.read_text(encoding="utf-8")
    for name in EXPECTED_NOTEBOOKS:
        assert name in readme_text
        nb = _load_notebook(name)
        assert nb.get("nbformat") == 4
        cells = nb.get("cells", [])
        assert len(cells) >= 6
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        assert len(markdown_cells) >= 3
        assert len(code_cells) >= 3
        for idx, cell in enumerate(code_cells):
            assert cell.get("outputs") == [], f"{name} code cell {idx} has committed outputs"
            assert cell.get("execution_count") is None, f"{name} code cell {idx} has execution_count"
            assert _cell_source(cell).strip(), f"{name} code cell {idx} is empty"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_executes_end_to_end(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if name == "04_gpu_acceleration.ipynb":
        pytest.importorskip("torch")
    if name == "05_jaxpip_cpp_cpu.ipynb":
        pytest.importorskip("fcntl", reason="JaxPIP dataset/export locks require POSIX fcntl")
        for pkg in ("jax", "equinox", "jaxpip", "jax2onnx", "onnx", "onnxruntime"):
            pytest.importorskip(pkg)
    if name == "06_jaxpip_jax_gpu.ipynb":
        for pkg in ("jax", "equinox", "jaxpip", "torch"):
            pytest.importorskip(pkg)
    if name == "01_pes_potentials.ipynb":
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            pytest.skip("C compiler required for C ABI cell in 01_pes_potentials.ipynb")

    nb = _load_notebook(name)
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__main__"}
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        code = _cell_source(cell)
        exec(compile(code, f"{name}#cell-{idx}", "exec"), namespace)
