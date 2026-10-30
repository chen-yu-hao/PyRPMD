"""Build the current user manual; historical validation stays in the repository."""

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
language = "zh_CN"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
include_patterns = ["index.md", "manual/*.md"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 4
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_class_signature = "separated"
html_theme = "pydata_sphinx_theme"
html_title = f"PyRPMD {release} 使用手册"
html_static_path = ["_static"]
html_css_files = ["manual.css"]
html_theme_options = {
    "github_url": "https://github.com/chen-yu-hao/PyRPMD",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "chen-yu-hao",
    "github_repo": "PyRPMD",
    "github_version": "main",
    "doc_path": "docs",
}
html_show_sourcelink = True

