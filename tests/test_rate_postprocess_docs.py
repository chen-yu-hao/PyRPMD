"""Regression guards for the rate guide's Sphinx inclusion and math syntax.

These checks deliberately need no Sphinx installation in the numerical test
suite. The Documentation workflow additionally builds both HTML variants with
warnings treated as errors.
"""

import ast
from fnmatch import fnmatchcase
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _literal_setting(name):
    tree = ast.parse((DOCS / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Sphinx setting {name!r} is missing")


def test_rate_guide_is_selected_and_linked_in_manual():
    name = "rate-postprocessing.md"
    assert (DOCS / name).is_file()
    assert any(fnmatchcase(name, pattern) for pattern in _literal_setting("include_patterns"))
    assert not any(fnmatchcase(name, pattern) for pattern in _literal_setting("exclude_patterns"))
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    trees = re.findall(r"```\{toctree\}\n(.*?)```", index, flags=re.DOTALL)
    assert any("rate-postprocessing" in block.splitlines() for block in trees)


def test_rate_guide_equations_use_enabled_myst_math_syntax():
    assert "dollarmath" in _literal_setting("myst_enable_extensions")
    source = (DOCS / "rate-postprocessing.md").read_text(encoding="utf-8")
    assert "\\[" not in source and "\\]" not in source
    equations = re.findall(r"^\$\$\n(.+?)^\$\$$", source, flags=re.MULTILINE | re.DOTALL)
    assert len(equations) == 5
    assert all(equation.strip() for equation in equations)
