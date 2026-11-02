"""Classify benchmark-dependent test modules without importing campaign code."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest


_OMITTED = pytest.StashKey[set[Path]]()


def _benchmark_name(name: str) -> bool:
    return name == "benchmarks" or name.startswith("benchmarks.")


def _benchmark_path(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.startswith(("benchmarks/", "benchmarks\\")))


@lru_cache(maxsize=None)
def _module_dependencies(path: Path) -> tuple[bool, frozenset[Path]]:
    """Read imports and literal benchmark paths, including function bodies.

    Follow local test/helper imports only; do not import packages or inspect the
    installed environment. Comments and embedded Python source are not imports.
    """
    tree = ast.parse(path.read_bytes(), filename=str(path))
    imports: set[str] = set()
    campaign = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imports.update(".".join(filter(None, (node.module, alias.name)))
                           for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            campaign |= _benchmark_path(node.value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # Also recognize ROOT / "benchmarks" / "family" / "driver.py".
            campaign |= any(_benchmark_path(part) or
                            (isinstance(part, ast.Constant) and part.value == "benchmarks")
                            for part in (node.left, node.right))
        elif isinstance(node, ast.Call) and node.args:
            campaign |= any(_benchmark_path(argument) for argument in node.args)
            function = node.func
            name = (function.id if isinstance(function, ast.Name) else
                    function.attr if isinstance(function, ast.Attribute) else "")
            argument = node.args[0]
            if (name in {"import_module", "__import__"}
                    and isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)):
                imports.add(argument.value)
    campaign |= any(_benchmark_name(name) for name in imports)
    dependencies = set()
    for name in imports:
        parts = name.split(".")
        if parts[0] == "tests":
            parts = parts[1:]
        # Imports may name a function/class after the local module name.
        for count in range(1, len(parts) + 1):
            candidate = path.parent.joinpath(*parts[:count]).with_suffix(".py")
            if candidate.is_file():
                dependencies.add(candidate.resolve())
    return campaign, frozenset(dependencies)


def _is_campaign(path: Path) -> bool:
    pending = [path.resolve()]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        campaign, dependencies = _module_dependencies(current)
        if campaign:
            return True
        pending.extend(dependencies - visited)
    return False


class _CampaignModule(pytest.Module):
    def collect(self):
        # Marker deselection normally runs *after* imports. This exact, documented
        # fast entry point must also work without POSIX-only campaign dependencies.
        if self.config.getoption("markexpr").split() == ["not", "campaign"]:
            self.config.stash.setdefault(_OMITTED, set()).add(self.path)
            return []
        return super().collect()


def pytest_pycollect_makemodule(module_path, parent):
    if _is_campaign(module_path):
        module = _CampaignModule.from_parent(parent, path=module_path)
        module.add_marker(pytest.mark.campaign)
        return module
    return None


def pytest_terminal_summary(terminalreporter, config):
    omitted = config.stash.get(_OMITTED, set())
    if omitted:
        terminalreporter.write_line(
            f"campaign: omitted {len(omitted)} modules before import (-m 'not campaign')"
        )
