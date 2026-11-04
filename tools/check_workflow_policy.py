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

The textual checks use only the standard library, so the script runs before
dependencies are installed. When PyYAML is available (the CI job installs it),
parsed ``permissions`` values are checked as well, which also resolves YAML
anchors and aliases. Exit status 1 lists every violation; 0 means clean.
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
# Template directories may hold only template/configuration files directly.
TEMPLATE_DIRECTORIES = frozenset({"ISSUE_TEMPLATE", "PULL_REQUEST_TEMPLATE"})
TEMPLATE_SUFFIXES = (".md", ".yml", ".yaml")
ID_TOKEN_WORKFLOWS = frozenset({"wheels.yml"})
# GITHUB_TOKEN permission scopes documented for workflow and job `permissions`.
PERMISSION_SCOPES = (
    "actions", "attestations", "checks", "contents", "deployments", "discussions",
    "id-token", "issues", "models", "packages", "pages", "pull-requests",
    "repository-projects", "security-events", "statuses",
)

# Textual layer: block mappings, flow mappings ({contents: write}) and quoted
# keys or values ("write") all match, at workflow or job scope.
_QUOTE = "[\"']?"
_SCOPE = "|".join(re.escape(scope) for scope in PERMISSION_SCOPES)
_WRITE_PERMISSION = re.compile(
    rf"(?<![\w-]){_QUOTE}(?P<scope>{_SCOPE}){_QUOTE}\s*:\s*{_QUOTE}write{_QUOTE}(?![\w-])")
_WRITE_ALL = re.compile(
    rf"(?<![\w-]){_QUOTE}permissions{_QUOTE}\s*:\s*{_QUOTE}write-all{_QUOTE}(?![\w-])")
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
    path = PurePosixPath(text)
    if text.startswith("workflows/"):
        return path.parent.as_posix() == "workflows" and path.suffix in (".yml", ".yaml")
    return (path.parent.as_posix() in TEMPLATE_DIRECTORIES
            and path.suffix in TEMPLATE_SUFFIXES)


def _parsed_permission_requests(text):
    """Yield (scope, value) pairs from parsed YAML, when PyYAML is installed.

    This second layer resolves anchors, aliases and any other YAML spelling
    that the textual layer cannot see. Without PyYAML it yields nothing and
    the textual layer still applies; CI installs PyYAML for this check.
    """
    try:
        import yaml
    except ImportError:
        return
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        yield ("<document>", "unparseable YAML")
        return
    if not isinstance(document, dict):
        return
    blocks = [document.get("permissions")]
    jobs = document.get("jobs")
    if isinstance(jobs, dict):
        blocks.extend(job.get("permissions") for job in jobs.values() if isinstance(job, dict))
    for block in blocks:
        if isinstance(block, str):
            yield ("<all>", block)
        elif isinstance(block, dict):
            for scope, value in block.items():
                yield (str(scope), value)


def check_workflow_text(name, text):
    """Return policy violations for one workflow file's text."""
    problems = []
    if _WRITE_ALL.search(text):
        problems.append(f"{name}: requests permissions: write-all")
    for match in _WRITE_PERMISSION.finditer(text):
        scope = match.group("scope")
        if scope == "id-token" and name in ID_TOKEN_WORKFLOWS:
            continue
        problem = f"{name}: requests {scope}: write"
        if problem not in problems:
            problems.append(problem)
    for scope, value in _parsed_permission_requests(text):
        if scope == "<document>":
            problems.append(f"{name}: {value}")
            continue
        normalized = str(value).strip().lower()
        if scope == "<all>":
            problem = f"{name}: requests permissions: write-all" if normalized == "write-all" else None
        elif normalized == "write" and not (scope == "id-token" and name in ID_TOKEN_WORKFLOWS):
            problem = f"{name}: requests {scope}: write"
        else:
            problem = None
        if problem and problem not in problems:
            problems.append(problem)
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
