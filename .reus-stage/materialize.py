"""One-shot transport: verify exact reviewed blobs and stage a clean main-based tree.

This helper and its workflow are NOT part of the feature branch or PR.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

stage = Path(__file__).resolve().parents[1]
target = Path(sys.argv[1]).resolve()
manifest = json.loads((stage / '.reus-stage/manifest.json').read_text())

def git(*args):
    return subprocess.check_output(['git', '-C', str(target), *args], text=True).strip()

assert git('rev-parse', 'HEAD') == manifest['base'], 'unexpected baseline'
assert not git('status', '--porcelain'), 'candidate checkout is not clean'
patch = stage / '.reus-stage/changes.patch'
assert hashlib.sha1(b'blob ' + str(patch.stat().st_size).encode() + b'\0' + patch.read_bytes()).hexdigest() == 'dcfb8e6573cf6f537bd2df1c26036fa54dc41ad2'
subprocess.run(['git', '-C', str(target), 'apply', '--index', '--unidiff-zero', str(patch)], check=True)
for name in manifest['new_files']:
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts and name in manifest['blobs']
    destination = target / path
    assert not destination.exists(), name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stage / path, destination)
    subprocess.run(['git', '-C', str(target), 'add', '--', name], check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(manifest['blobs'])
for name, expected in manifest['blobs'].items():
    assert git('hash-object', name) == expected, name
    assert git('rev-parse', ':' + name) == expected, name
assert not git('diff', '--name-only'), 'unstaged change'
subprocess.run(['git', '-C', str(target), 'diff', '--cached', '--check'], check=True)
print(json.dumps({'base': manifest['base'], 'tree': git('write-tree'), 'verified_blobs': manifest['blobs']}, indent=2))
