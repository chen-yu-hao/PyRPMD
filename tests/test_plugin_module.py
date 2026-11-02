"""Plugin references split at the attribute colon, never at a drive letter."""

import pytest

from pyrpmd.io import plugin_module


@pytest.mark.parametrize("value,expected", [
    ("/runs/model/pes.py:build", "/runs/model/pes.py"),
    ("C:\\runs\\model\\pes.py:build", "C:\\runs\\model\\pes.py"),
    ("C:/runs/model/pes.py:build", "C:/runs/model/pes.py"),
    ("C:\\runs\\model\\libpes.so", "C:\\runs\\model\\libpes.so"),
    ("relative/pes.py", "relative/pes.py"),
    ("package.module:factory", "package.module"),
    ("package.module", "package.module"),
    ("/data/run:1/pes.py:build", "/data/run:1/pes.py"),
])
def test_plugin_module_ignores_drive_and_directory_colons(value, expected):
    assert plugin_module(value) == expected
