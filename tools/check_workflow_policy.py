"""Reject CI changes that could materialize or publish unreviewed code.

The round-four review (docs/roadmap.md) found agent branches that stored
encoded payloads under ``.github/`` together with workflows that decoded them
and pushed commits with ``contents: write``. This checker enforces the task
rule G13 (docs/tasks/README.md) for the whole repository:

* ``.github/`` holds only workflow YAML files and ordinary GitHub metadata;
* no workflow requests a write permission other than ``id-token: write`` in
  the release workflow, which PyPI trusted publishing needs;
* no workflow pushes commits, decodes embedded payloads or executes
  dynamically decoded source.

It uses only the standard library so that it can run before dependencies are
installed. Exit status 1 lists every violation; 0 means the tree is clean.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import sys

ALLOWED_GITHUB_FILES = (
    "CODEOWNERS",
    "PULL_REQUEST_TEMPLATE.md",
    "dependabot.yml",
    "FUNDING.yml",
)
ALLOWED_GITHUB_PREFIXES = ("workflows/", "ISSUE_TEMPLATE/", "PULL_REQUEST_TEMPLATE/")
ID_TOKEN_WORKFLOWS = frozenset({"wheels.yml"})

_WRITE_PERMISSION = re.compile(
    r"^\s*(?P<scope>[a-z][a-z-]*)\s*:\s*write\s*(?:#.*)?$", re.MULTILINE)
_WRITE_ALL = re.compile(r"^\s*permissions\s*:\s*write-all\s*(?:#.*)?$", re.MULTILINE)
_FORBIDDEN_STEPS = (
    (re.compile(r"\bgit\s+push\b"), "pushes commits"),
    (re.compile(r"\bbase64\s+(?:-d|--decode)\b"), "decodes base64 payloads"),
    (re.compile(r"\bb(?:64|85|32)decode\s*\("), "decodes embedded payloads"),
    (re.compile(r"\bexec\s*\(\s*compile\s*\("), "executes dynamically compiled source"),
    (re.compile(r"\bzlib\.decompress\s*\("), "decompresses embedded payloads"),
)


def _github_file_allowed(relative):
    text = relative.as_posix()
    if text in ALLOWED_GITHUB_FILES:
        return True
    if text.startswith("workflows/"):
        return PurePosixPath(text).parent.as_posix() == "workflows" and text.endswith((".yml", ".yaml"))
    return text.startswith(ALLOWED_GITHUB_PREFIXES)


def check_workflow_text(name, text):
    """Return policy violations for one workflow file's text."""
    problems = []
    if _WRITE_ALL.search(text):
        problems.append(f"{name}: requests permissions: write-all")
    for match in _WRITE_PERMISSION.finditer(text):
        scope = match.group("scope")
        if scope == "id-token" and name in ID_TOKEN_WORKFLOWS:
            continue
        problems.append(f"{name}: requests {scope}: write")
    for pattern, description in _FORBIDDEN_STEPS:
        if pattern.search(text):
            problems.append(f"{name}: {description}")
    return problems


def check_repository(root):
    """Return all policy violations below ``root``."""
    root = Path(root)
    github = root / ".github"
    problems = []
    if not github.is_dir():
        return problems
    for path in sorted(p for p in github.rglob("*") if p.is_file()):
        relative = path.relative_to(github)
        if "__pycache__" in relative.parts:
            continue
        if not _github_file_allowed(relative):
            problems.append(f".github/{relative.as_posix()}: only workflows and GitHub metadata belong in .github")
            continue
        if relative.parts[0] == "workflows":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                problems.append(f".github/{relative.as_posix()}: workflow is not UTF-8 text")
                continue
            problems.extend(check_workflow_text(path.name, text))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1],
                        help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    problems = check_repository(args.root)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} workflow policy violation(s); see docs/tasks/README.md rule G13",
              file=sys.stderr)
        return 1
    print("workflow policy: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
