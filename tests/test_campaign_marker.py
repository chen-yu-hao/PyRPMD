"""Black-box collection checks against small, independently specified suites."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_suite(tmp_path, files, *arguments):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_bytes(Path(__file__).with_name("conftest.py").read_bytes())
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\nmarkers = campaign: benchmark-dependent module\n",
        encoding="utf-8",
    )
    for relative, source in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(tmp_path)}
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--strict-markers", *arguments],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )


CORE = """
def test_core(request):
    assert request.node.get_closest_marker('campaign') is None
"""
CAMPAIGN = """
from benchmarks import value
def test_campaign(request):
    assert value == 42
    assert request.node.get_closest_marker('campaign') is not None
"""


@pytest.mark.parametrize("expression, expected", [(None, "2 passed"), ("campaign", "1 passed, 1 deselected")])
def test_default_keeps_every_test_and_campaign_can_be_selected(tmp_path, expression, expected):
    files = {"tests/test_core.py": CORE, "tests/test_validation.py": CAMPAIGN,
             "benchmarks/__init__.py": "value = 42\n"}
    result = run_suite(tmp_path, files, *([] if expression is None else ["-m", expression]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_fast_selection_never_imports_campaign_or_transitive_helpers(tmp_path, explicit_paths):
    files = {"tests/test_core.py": CORE,
             "tests/test_validation.py": CAMPAIGN,
             "tests/campaign_helper.py": "from benchmarks import value\n",
             "tests/test_transitive.py": "from campaign_helper import value\ndef test_value(): assert value == 42\n",
             "benchmarks/__init__.py": "raise RuntimeError('campaign import must not happen')\n"}
    paths = ["tests/test_core.py", "tests/test_validation.py", "tests/test_transitive.py"] if explicit_paths else []
    result = run_suite(tmp_path, files, "-m", "not campaign", *paths)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "omitted 2 modules before import" in result.stdout


@pytest.mark.parametrize("dependency", [
    "from pathlib import Path\nreference = Path(__file__).parent / 'benchmarks' / 'driver.py'",
    "reference = 'benchmarks/driver.py'",
    "def unused():\n    from benchmarks import value",
    "import importlib\nvalue = importlib.import_module('benchmarks').value",
])
def test_static_path_function_and_dynamic_import_rules(tmp_path, dependency):
    source = dependency + "\nraise RuntimeError('must be classified without importing')\n"
    result = run_suite(tmp_path, {"tests/test_core.py": CORE, "tests/test_validation.py": source},
                       "-m", "not campaign")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "omitted 1 modules before import" in result.stdout


def test_comments_source_strings_and_similar_package_names_are_not_imports(tmp_path):
    source = "# from benchmarks import value\nexample = 'from benchmarks import value'\n" + CORE
    result = run_suite(tmp_path, {"tests/test_core.py": source}, "-m", "not campaign")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "omitted" not in result.stdout


def test_local_helper_cycle_terminates_and_preserves_core_selection(tmp_path):
    result = run_suite(tmp_path, {
        "tests/test_core.py": "import first\n" + CORE,
        "tests/first.py": "import second\n",
        "tests/second.py": "import first\n",
    }, "-m", "not campaign")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
