"""Check repository-local Markdown links without making network requests.

Only local path existence is checked here; Sphinx -W checks document targets.
External HTTP links are intentionally outside the deterministic CI gate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'(?<!!)\[[^\]]*\]\((<?[^\s)]+>?)')
CODE_FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
AUTOLINK = re.compile(r'<(?:https?://|mailto:)[^>]+>')
TARGETS = [ROOT / "README.md", ROOT / "README.zh-CN.md", *sorted((ROOT / "docs").rglob("*.md"))]


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = CODE_FENCE.match(line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        line = AUTOLINK.sub("", line)
        for match in PATTERN.finditer(line):
            raw = match.group(1).strip("<>")
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            # Absolute /... URLs are site-local, not repository-relative.
            if raw.startswith("/"):
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                errors.append(f"{path.relative_to(ROOT)}:{number}: escapes repository: {raw}")
            elif not target.exists():
                errors.append(f"{path.relative_to(ROOT)}:{number}: missing {raw}")
    return errors


def main() -> int:
    errors = [error for path in TARGETS if path.is_file() for error in check_file(path)]
    for error in errors:
        print(error)
    if errors:
        print(f"{len(errors)} unresolved local Markdown links", file=sys.stderr)
        return 1
    print(f"Checked local links in {sum(p.is_file() for p in TARGETS)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
