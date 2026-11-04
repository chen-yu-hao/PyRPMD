"""The method/paper catalog, its generated pages and the citation metadata stay consistent."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("reference_catalog", ROOT / "tools" / "reference_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_is_internally_consistent():
    tool = _tool()
    assert tool.validate(tool.load()) == []


def test_generated_pages_and_bibtex_are_current():
    tool = _tool()
    cat = tool.load()
    stale = [
        str(path.relative_to(ROOT))
        for path, text in tool.outputs(cat).items()
        if path.read_text(encoding="utf-8") != text
    ]
    assert stale == [], "run `python tools/reference_catalog.py --write`"


def test_every_cited_bibliography_key_is_rendered_once():
    tool = _tool()
    cat = tool.load()
    bib = tool.render_bibtex(cat)
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    assert keys == [r["id"] for r in cat["references"]]
    page = tool.render_references(cat, "en")
    for method in cat["methods"]:
        assert page.count(f"\n### {method['id']}\n") == 1
    for ref in cat["references"]:
        assert page.count(f"\n### {ref['id']}\n") == 1


def test_validation_reports_dangling_references(tmp_path):
    tool = _tool()
    cat = tool.load()
    cat["methods"][0] = dict(cat["methods"][0], references=["no-such-paper"],
                             code=["src/pyrpmd/does_not_exist.py"])
    problems = tool.validate(cat)
    assert any("no-such-paper" in p for p in problems)
    assert any("does_not_exist.py" in p for p in problems)


def test_readmes_quote_current_catalog_size():
    cat = json.loads((ROOT / "docs" / "method-references.json").read_text(encoding="utf-8"))
    methods, refs = len(cat["methods"]), len(cat["references"])
    short = cat["audited_commit"][:7]
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert f"**{methods} feature entries and {refs} bibliographic records**" in english
    assert f"**{methods} 个功能条目与 {refs} 条文献记录**" in chinese
    assert f"`{short}`" in english and f"`{short}`" in chinese


def test_citation_cff_matches_package_metadata():
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    init = (ROOT / "src" / "pyrpmd" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r'__version__ = "([^"]+)"', init).group(1) == version
    assert re.search(r"^version: (\S+)$", cff, re.M).group(1) == version
    assert re.search(r"^cff-version: 1\.2\.0$", cff, re.M)
    assert 'repository-code: "https://github.com/chen-yu-hao/PyRPMD"' in cff
    for readme in ("README.md", "README.zh-CN.md"):
        text = (ROOT / readme).read_text(encoding="utf-8")
        assert f"version = {{{version}}}" in text, readme
