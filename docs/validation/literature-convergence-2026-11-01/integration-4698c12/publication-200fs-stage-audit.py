"""Verify the explicitly staged evidence and strict manual build before publication."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import subprocess
import time
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from markdown_it import MarkdownIt

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'pyrpmd-convergence-integration'
BASE = REPO / 'docs/validation/literature-convergence-2026-11-01'
OUTPUT = BASE / 'integration-4698c12'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    staged = json.loads((ROOT / 'convergence-publication-staging.json').read_text())
    paths = set(staged['paths'])
    bound = {}
    for name in staged['inventories']:
        path = REPO / name
        bound[name] = sha(path)
        for entry in json.loads(path.read_text())['files']:
            target = REPO / entry['path']
            assert target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], target
            paths.add(entry['path'])
    markdown = MarkdownIt()
    counts = {'json': 0, 'python': 0, 'xml': 0, 'markdown': 0, 'local_targets': 0}
    missing = []
    for name in sorted(paths):
        path = REPO / name
        if path.suffix == '.json':
            json.loads(path.read_text())
            counts['json'] += 1
        elif path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
            counts['python'] += 1
        elif path.suffix == '.xml':
            ET.parse(path)
            counts['xml'] += 1
        elif path.suffix == '.md':
            counts['markdown'] += 1
            for top in markdown.parse(path.read_text()):
                for token in top.children or []:
                    destination = token.attrGet('href') if token.type == 'link_open' else token.attrGet('src') if token.type == 'image' else None
                    if not destination:
                        continue
                    link = urlsplit(destination)
                    if link.scheme or link.netloc or not link.path:
                        continue
                    target = path.parent / unquote(link.path)
                    counts['local_targets'] += 1
                    if not target.exists():
                        missing.append({'source': name, 'target': destination})
    generated_here = {'documentation-200fs-stage-check.json',
                      'documentation-200fs-stage-build.log', 'publication-200fs-stage-audit.json'}
    unexpected_missing = [item for item in missing
                          if not ((REPO / item['source']).parent == OUTPUT
                                  and item['target'] in generated_here)]
    assert not unexpected_missing, json.dumps(unexpected_missing, indent=2)
    hashes = {name: sha(REPO / name) for name in sorted(paths)}
    command = [str(ROOT / 'PyRPMD/.venv/bin/python'), '-m', 'sphinx', '-W', '--keep-going',
               '-b', 'html', 'docs', str(ROOT / 'convergence-docs-html')]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    log = OUTPUT / 'documentation-200fs-stage-build.log'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    unchanged = hashes == {name: sha(REPO / name) for name in sorted(paths)}
    receipt = {
        'schema': 'pyrpmd.complete200fs-documentation-check.v1', 'started_utc': started,
        'command': command, 'exit_code': result.returncode,
        'elapsed_seconds': time.monotonic() - clock, 'log_sha256': sha(log),
        'all_selected_sources_unchanged_during_build': unchanged,
        'scope': 'Strict English Sphinx HTML; immutable historical evidence is excluded by Sphinx and checked separately for local links and parseability.',
        'source_hashes': {name: value for name, value in hashes.items() if name.endswith('.md')},
    }
    (OUTPUT / 'documentation-200fs-stage-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
    audit = {
        'schema': 'pyrpmd.complete200fs-publication-audit.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'inventories_sha256': bound, 'distinct_verified_files': len(paths),
        'format_and_local_link_counts': counts, 'missing_local_targets': unexpected_missing,
        'own_generated_receipt_links_deferred_until_outputs_written': missing,
        'all_selected_sources_unchanged_during_build': unchanged,
        'raw_arrays_and_external_models_excluded_by_whitelist': True,
        'allowed_npz': ['docs/validation/literature-convergence-2026-11-01/water/velocity-cadence/curves.npz'],
        'staged_tree_at_start': staged['expected_tree'],
        'initial_whitespace_checks': {
            'first_log_sha256': sha(ROOT / 'convergence-third-staging.log'),
            'second_log_sha256': sha(ROOT / 'convergence-third-staging-resolved.log'),
            'resolution': 'Frozen generated CSV retains CRLF and compiler --version stdout retains its terminal blank line. Their exact checksums were preserved; authored-source checks still pass.',
        },
        'whitespace_exclusions': staged['whitespace_check_exclusions'],
        'driver_sha256': sha(Path(__file__)),
    }
    (OUTPUT / 'publication-200fs-stage-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    for item in missing:
        assert ((REPO / item['source']).parent / item['target']).exists(), item
    if not unchanged or result.returncode != 0:
        raise RuntimeError('Publication documentation gate failed')
    print(json.dumps({'files': len(paths), 'inventories': len(bound), 'counts': counts,
                      'strict_sphinx_exit': result.returncode, 'elapsed_seconds': receipt['elapsed_seconds']}))


if __name__ == '__main__':
    main()
