"""Guard against import-time module replacement and copied namespaces."""

import ast
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'pyrpmd'
# A module loaded from a user-supplied file must be registered under its unique
# generated name before exec_module, so imports and dataclasses can resolve it.
PLUGIN_LOADERS = {'input.py', 'potentials/python.py'}


def _root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def violations(source, filename):
    tree = ast.parse(source, filename=filename)
    imported = set()
    errors = []
    # Imports under try/if guards still execute at module import time. Function
    # and class bodies are separate scopes; ordinary runtime mutation is valid.
    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        if isinstance(node, ast.For) and any(isinstance(part, ast.Name) and part.id in imported
                                            for part in ast.walk(node.iter)):
            imported.update(part.id for part in ast.walk(node.target) if isinstance(part, ast.Name))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.Attribute) and _root(part) in imported:
                        errors.append((node.lineno, 'cross-module attribute assignment'))
                        break
                    if (isinstance(part, ast.Subscript) and isinstance(part.value, ast.Call)
                            and isinstance(part.value.func, ast.Name) and part.value.func.id == 'globals'):
                        errors.append((node.lineno, 'copied global namespace'))
                        break
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == 'setattr'
                    and node.args and _root(node.args[0]) in imported):
                errors.append((node.lineno, 'cross-module setattr'))
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'update'
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == 'globals'):
                errors.append((node.lineno, 'copied global namespace'))
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    # Inspect sys.modules writes in every scope, with narrow plugin exceptions.
    sys_names = {alias.asname or 'sys' for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names if alias.name == 'sys'}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                    and target.value.attr == 'modules' and _root(target.value) in sys_names):
                replaces_self = isinstance(target.slice, ast.Name) and target.slice.id == '__name__'
                if filename not in PLUGIN_LOADERS or replaces_self:
                    errors.append((node.lineno, 'sys.modules replacement'))
    return sorted(set(errors))


def test_package_has_no_import_time_compatibility_patches():
    found = []
    for path in sorted(SOURCE.rglob('*.py')):
        relative = path.relative_to(SOURCE).as_posix()
        found.extend(f'{relative}:{line}: {message}'
                     for line, message in violations(path.read_text(encoding="utf-8"), relative))
    assert not found, '\n'.join(found)


@pytest.mark.parametrize('source', [
    'import sys\nsys.modules[__name__] = replacement',
    'import sys as s\ns.modules[__name__] = replacement',
    'from . import other\nfor name, value in vars(other).items():\n    globals()[name] = value',
    'from . import other\nglobals().update(vars(other))',
    'from . import other\nif True:\n    other.writer = writer',
    'import other\nsetattr(other, "writer", writer)',
    'from . import a, b\nfor module in (a, b):\n    setattr(module, "writer", writer)',
])
def test_structure_guard_rejects_patches(source):
    assert violations(source, 'bad.py')


@pytest.mark.parametrize('source', [
    'from .other import writer\nALIAS = writer',
    'import other\ndef run():\n    other.writer = writer',
    'class Local:\n    def method(self):\n        self.value = 1',
    'local = Local()\nlocal.value = 1',
])
def test_structure_guard_allows_normal_imports_and_local_state(source):
    assert not violations(source, 'good.py')


@pytest.mark.parametrize('filename', sorted(PLUGIN_LOADERS))
def test_plugin_exception_cannot_replace_current_module(filename):
    assert not violations('import sys\ndef load():\n    sys.modules[unique_name] = module', filename)
    assert violations('import sys\nsys.modules[__name__] = module', filename)
