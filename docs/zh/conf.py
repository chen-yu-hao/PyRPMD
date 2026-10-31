"""Build the Chinese user manual with the settings of the English edition.

Keep both editions structurally identical: each chapter in ``docs/manual`` has a
Chinese counterpart under ``docs/zh/manual`` with the same file name. Build this
edition into ``<outdir>/zh`` next to the English one so the language switcher works.
"""

from pathlib import Path
import runpy

_DOCS = Path(__file__).resolve().parents[1]
_shared = runpy.run_path(str(_DOCS / "conf.py"))
globals().update({k: v for k, v in _shared.items() if not k.startswith("_")})

language = "zh_CN"
html_title = f"PyRPMD {release} 使用手册"
html_static_path = ["../_static"]
templates_path = ["../_templates"]
html_context = {**_shared["html_context"], "doc_path": "docs/zh"}
