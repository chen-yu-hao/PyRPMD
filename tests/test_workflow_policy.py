"""Workflow policy checker (rule G13): the repository passes, payload patterns fail."""

from pathlib import Path
import importlib.util
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("check_workflow_policy", ROOT / "tools" / "check_workflow_policy.py")
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)


def _workflow(tmp_path, name, body):
    path = tmp_path / ".github" / "workflows" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_repository_satisfies_policy():
    assert policy.check_repository(ROOT) == []
    assert policy.main([str(ROOT)]) == 0


def test_read_only_workflow_passes(tmp_path):
    _workflow(tmp_path, "ci.yml", """
        on: [push]
        permissions:
          contents: read
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - run: python -m pytest -q
    """)
    assert policy.check_repository(tmp_path) == []


@pytest.mark.parametrize("snippet, message", [
    ("permissions:\n  contents: write\n", "contents: write"),
    ("permissions: write-all\n", "write-all"),
    ("permissions:\n  actions: write\n", "actions: write"),
    ("jobs:\n  a:\n    steps:\n      - run: git push origin HEAD:other\n", "pushes commits"),
    ("jobs:\n  a:\n    steps:\n      - run: cat .github/p* | base64 -d > x.tar\n", "base64"),
    ("jobs:\n  a:\n    steps:\n      - run: python -c 'import base64; base64.b85decode(s)'\n", "embedded payloads"),
    ("jobs:\n  a:\n    steps:\n      - run: python -c 'exec(compile(s, \"x\", \"exec\"))'\n", "dynamically compiled"),
])
def test_write_and_materialization_patterns_fail(tmp_path, snippet, message):
    _workflow(tmp_path, "bad.yml", "on: [push]\n" + snippet)
    problems = policy.check_repository(tmp_path)
    assert any(message in problem for problem in problems), problems
    assert policy.main([str(tmp_path)]) == 1


@pytest.mark.parametrize("body, message", [
    ("permissions: {contents: write}\n", "contents: write"),
    ("permissions: {contents: read, pull-requests: write}\n", "pull-requests: write"),
    ("permissions:\n  contents: \"write\"\n", "contents: write"),
    ("permissions:\n  'contents': 'write'\n", "contents: write"),
    ("permissions: \"write-all\"\n", "write-all"),
    ("jobs:\n  a:\n    permissions: {actions: write}\n    runs-on: x\n", "actions: write"),
])
def test_flow_mappings_and_quoted_scalars_fail(tmp_path, body, message):
    _workflow(tmp_path, "bad.yml", "on: [push]\n" + body)
    problems = policy.check_repository(tmp_path)
    assert any(message in problem for problem in problems), problems


def test_parsed_layer_resolves_aliases(tmp_path):
    pytest.importorskip("yaml")
    _workflow(tmp_path, "bad.yml", """
        on: [push]
        x-level: &level write
        permissions:
          contents: *level
    """)
    assert policy.check_repository(tmp_path) == ["bad.yml: requests contents: write"]


def test_id_token_is_allowed_only_for_the_release_workflow(tmp_path):
    body = "on: [push]\npermissions:\n  id-token: write\n  contents: read\n"
    _workflow(tmp_path, "wheels.yml", body)
    assert policy.check_repository(tmp_path) == []
    _workflow(tmp_path, "other.yml", body)
    assert policy.check_repository(tmp_path) == ["other.yml: requests id-token: write"]


@pytest.mark.parametrize("relative", [
    "t30-payload.part00", "t32/payload-0", "workflows/nested/x.yml", "workflows/notes.txt",
    "ISSUE_TEMPLATE/payload.part00", "PULL_REQUEST_TEMPLATE/payload.zip",
    "ISSUE_TEMPLATE/nested/bug.md", "PULL_REQUEST_TEMPLATE/x.md.b64",
    "__pycache__/payload.part00", "workflows/__pycache__/x.cpython-312.pyc",
])
def test_non_metadata_files_under_github_fail(tmp_path, relative):
    path = tmp_path / ".github" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data", encoding="utf-8")
    problems = policy.check_repository(tmp_path)
    assert problems and "only workflows and GitHub metadata" in problems[0]


def test_github_metadata_files_pass(tmp_path):
    for relative in ("CODEOWNERS", "PULL_REQUEST_TEMPLATE.md", "ISSUE_TEMPLATE/bug.md",
                     "ISSUE_TEMPLATE/config.yml", "ISSUE_TEMPLATE/feature.yaml",
                     "PULL_REQUEST_TEMPLATE/release.md"):
        path = tmp_path / ".github" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert policy.check_repository(tmp_path) == []
