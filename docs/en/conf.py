"""Build the English user manual with the settings of the Chinese edition.

Keep both editions structurally identical: each chapter in ``docs/manual`` has an
English counterpart under ``docs/en/manual`` with the same file name. Build this
edition into ``<outdir>/en`` next to the Chinese one so the language switcher works.
"""

from pathlib import Path
import runpy

_DOCS = Path(__file__).resolve().parents[1]
_shared = runpy.run_path(str(_DOCS / "conf.py"))
globals().update({k: v for k, v in _shared.items() if not k.startswith("_")})

language = "en"
html_title = f"PyRPMD {release} User Manual"
html_static_path = ["../_static"]
templates_path = ["../_templates"]
html_context = {**_shared["html_context"], "doc_path": "docs/en"}
