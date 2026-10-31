"""Build the English user manual; historical validation stays in the repository.

The Chinese edition in ``docs/zh`` reuses these settings through ``docs/zh/conf.py``.
Build it into ``<outdir>/zh`` so the navbar language switcher can link both editions.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pyrpmd import __version__

project = "PyRPMD"
author = "PyRPMD contributors"
copyright = "2026, PyRPMD contributors"
release = __version__
version = release
language = "en"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
include_patterns = [
    "index.md", "manual/*.md", "enhanced-sampling.md", "replica-exchange.md", "opes.md",
    "plumed.md", "transition-interface-sampling.md", "rate-postprocessing.md",
    "general-coordinate-rates.md", "rate-studies.md", "instanton.md",
    "quantum-statistics.md", "performance-dynamics.md", "batched-dynamics.md",
    "rpc-mts.md", "trpmd-spectroscopy.md",
    "validation/tier1-2026-10-31/README.md",
    "validation/tier1-2026-10-31/pmf-replay.md",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 4
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_class_signature = "separated"
html_theme = "pydata_sphinx_theme"
html_title = f"PyRPMD {release} User Manual"
html_static_path = ["_static"]
html_css_files = ["manual.css"]
templates_path = ["_templates"]
# Every chapter is a top-level page, so the primary sidebar would stay empty on
# desktop; the navbar and the mobile drawer still list all chapters.
html_sidebars = {"**": []}
html_theme_options = {
    "github_url": "https://github.com/chen-yu-hao/PyRPMD",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "navbar_end": ["language-switcher", "theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "chen-yu-hao",
    "github_repo": "PyRPMD",
    "github_version": "main",
    "doc_path": "docs",
    # English-only extension guides return to the Chinese contents rather than
    # claiming a translation at a nonexistent same-name page.
    "bilingual_pages": ["index", "genindex", "search", "enhanced-sampling", "replica-exchange"] + [
        "manual/" + path.stem
        for path in sorted((ROOT / "docs" / "manual").glob("*.md"))
        if (ROOT / "docs" / "zh" / "manual" / path.name).is_file()
    ],
}
html_show_sourcelink = True

