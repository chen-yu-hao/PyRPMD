"""Regression guards keeping the English and Chinese manuals aligned with code.

These checks run without Sphinx so the standard pytest suite catches CLI,
schema, FES-table, bilingual-chapter and toctree drift before CI builds HTML.
"""

import argparse
import ast
from fnmatch import fnmatchcase
from pathlib import Path
import re

import numpy as np

from pyrpmd.analysis.fes_result import FESResult, write_fes_result
from pyrpmd.cli import parser as build_cli_parser
from pyrpmd.input_fes import _FES_METHODS
from pyrpmd.input_schema import TASKS


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ZH_DOCS = DOCS / "zh"


def _literal_setting(name):
    tree = ast.parse((DOCS / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Sphinx setting {name!r} is missing")


def _toctree_targets(doc_path: Path) -> set[Path]:
    text = doc_path.read_text(encoding="utf-8")
    targets: set[Path] = set()
    for block in re.findall(r"```\{toctree\}\n(.*?)```", text, flags=re.DOTALL):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if "<" in line and line.endswith(">"):
                line = line[line.rfind("<") + 1 : -1].strip()
            if not line.endswith(".md"):
                line += ".md"
            targets.add((doc_path.parent / line).resolve())
    return targets


def _reachable_from_index(index_path: Path) -> set[Path]:
    seen = {index_path.resolve()}
    queue = [index_path.resolve()]
    while queue:
        current = queue.pop()
        if not current.is_file():
            continue
        for target in _toctree_targets(current):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def test_english_and_chinese_manual_chapters_match():
    en_chapters = sorted(p.name for p in (DOCS / "manual").glob("*.md"))
    zh_chapters = sorted(p.name for p in (ZH_DOCS / "manual").glob("*.md"))
    assert en_chapters == zh_chapters


def test_every_sphinx_included_english_doc_is_in_toctree_without_orphan_header():
    include_patterns = _literal_setting("include_patterns")
    exclude_patterns = _literal_setting("exclude_patterns")
    included = sorted(
        p.resolve()
        for p in DOCS.rglob("*.md")
        if not p.relative_to(DOCS).as_posix().startswith("zh/")
        and any(fnmatchcase(p.relative_to(DOCS).as_posix(), pat) for pat in include_patterns)
        and not any(fnmatchcase(p.relative_to(DOCS).as_posix(), pat) for pat in exclude_patterns)
    )
    reachable = _reachable_from_index(DOCS / "index.md")
    missing = sorted(p.relative_to(DOCS).as_posix() for p in included if p not in reachable)
    assert missing == [], f"Sphinx-included docs missing from toctree: {missing}"

    stale_orphans = sorted(
        p.relative_to(DOCS).as_posix()
        for p in included
        if p.read_text(encoding="utf-8").startswith("---\norphan: true\n---")
    )
    assert stale_orphans == [], f"Indexed docs still carry orphan: true: {stale_orphans}"


def test_cli_subcommands_are_documented_in_both_manuals():
    cli_parser = build_cli_parser()
    subcommands = sorted(
        choice
        for action in cli_parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for choice in action.choices
    )
    assert subcommands
    for manual in (DOCS / "manual/input.md", ZH_DOCS / "manual/input.md"):
        text = manual.read_text(encoding="utf-8")
        missing = [cmd for cmd in subcommands if f"`pyrpmd {cmd}" not in text]
        assert missing == [], f"{manual.relative_to(ROOT)} omits CLI subcommands: {missing}"


def test_tasks_and_fes_methods_are_documented_in_both_manuals():
    active_tasks = sorted(set(TASKS) - {"metadynamics", "opes"})
    fes_methods = sorted(_FES_METHODS)
    for manual in (DOCS / "manual/input.md", ZH_DOCS / "manual/input.md"):
        text = manual.read_text(encoding="utf-8")
        missing_tasks = [task for task in active_tasks if f"`{task}`" not in text]
        missing_methods = [
            method
            for method in fes_methods
            if f"`{method}`" not in text and f'"{method}"' not in text
        ]
        assert missing_tasks == [], f"{manual.relative_to(ROOT)} omits tasks: {missing_tasks}"
        assert missing_methods == [], f"{manual.relative_to(ROOT)} omits FES methods: {missing_methods}"


def test_fes_csv_columns_match_results_manuals(tmp_path):
    result = FESResult(
        edges=(np.array([0.0, 1.0, 2.0]),),
        free_energy=np.array([0.0, 0.1]),
        uncertainty=np.array([0.01, 0.02]),
        mask=np.array([False, False]),
        cv_units=("bohr",),
        method="umbrella",
        uncertainty_kind="block_bootstrap",
        metadata={"measure": "cv_density"},
        extras={"raw_count": np.array([10.0, 12.0])},
    )
    write_fes_result(tmp_path, result)
    header = (tmp_path / "fes.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    non_cv_columns = [col for col in header if not col.startswith("cv")]
    for manual in (DOCS / "manual/results.md", ZH_DOCS / "manual/results.md"):
        text = manual.read_text(encoding="utf-8")
        missing = [col for col in non_cv_columns if f"`{col}`" not in text]
        assert missing == [], f"{manual.relative_to(ROOT)} omits fes.csv columns: {missing}"
