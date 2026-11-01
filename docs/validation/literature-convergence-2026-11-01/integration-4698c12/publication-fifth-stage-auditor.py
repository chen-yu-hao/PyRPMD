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
    staged = json.loads((ROOT / 'convergence-fifth-publication-staging.json').read_text())
    paths = set(staged['paths'])
    bound = {}
    supersession_receipt = staged['explicit_test_only_supersession']
    supersession_path = REPO / supersession_receipt['receipt_path']
    assert sha(supersession_path) == supersession_receipt['receipt_sha256'] == 'fcef77c2a0813451e53ba8b890b01e2e434c8a137519b427f2e360e67892b82e'
    supersession = json.loads(supersession_path.read_text())
    replacement = supersession['supersessions'][0]
    assert len(supersession['supersessions']) == 1
    assert replacement == supersession_receipt['replacement']
    assert replacement['current_integration_path'] == 'tests/test_water_production_stability.py'
    assert sha(REPO / replacement['current_integration_path']) == replacement['current_integration_sha256']
    assert sha(REPO / replacement['patch_path']) == replacement['patch_sha256']
    hch4 = staged['additional_test_only_supersession']
    assert sha(REPO / hch4['receipt_path']) == hch4['receipt_sha256'] == '42065fe6c3f8833f0c57929603e40c475317b07e5c27dce745db2f482ba1fead'
    assert sha(REPO / hch4['old_inventory_path']) == hch4['old_inventory_sha256']
    assert sha(REPO / hch4['historical_snapshot_path']) == hch4['historical_sha256']
    assert sha(REPO / hch4['current_path']) == hch4['current_sha256']
    for name in staged['inventories']:
        path = REPO / name
        bound[name] = sha(path)
        for entry in json.loads(path.read_text())['files']:
            target = REPO / entry['path']
            if (name == supersession['old_inventory']['path']
                    and entry['path'] == replacement['original_inventory_entry']):
                target = REPO / replacement['preserved_snapshot_path']
                assert sha(path) == supersession['old_inventory']['sha256']
            if name == hch4['old_inventory_path'] and entry['path'] == hch4['current_path']:
                target = REPO / hch4['historical_snapshot_path']
            assert target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], target
            paths.add(entry['path'])
    markdown = MarkdownIt()
    counts = {'json': 0, 'python': 0, 'xml': 0, 'markdown': 0, 'local_targets': 0}
    missing = []
    context_path = OUTPUT / 'publication-four-pair-stage-historical-link-context.json'
    assert sha(context_path) == '92ec16c45234d3cd23244336a9545c4f9796601bc090249ebf9d937e4d3d5413'
    historical_context = json.loads(context_path.read_text())
    assert sha(REPO / historical_context['source']) == historical_context['source_sha256']
    historical_links = {row['target']: row for row in historical_context['links']}
    rebased_historical_links = []
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
                        if name == historical_context['source'] and destination in historical_links:
                            row = historical_links[destination]
                            restored_target = REPO / row['resolved_from_original_directory']
                            assert restored_target.is_file() and sha(restored_target) == row['target_sha256']
                            rebased_historical_links.append(dict(source=name, **row))
                        else:
                            missing.append({'source': name, 'target': destination})
    assert len(rebased_historical_links) == 7
    generated_here = {'documentation-fifth-stage-check.json',
                      'documentation-fifth-stage-build.log', 'publication-fifth-stage-audit.json'}
    unexpected_missing = [item for item in missing
                          if not ((REPO / item['source']).parent == OUTPUT
                                  and item['target'] in generated_here)]
    assert not unexpected_missing, json.dumps(unexpected_missing, indent=2)
    hashes = {name: sha(REPO / name) for name in sorted(paths)}
    command = [str(ROOT / 'PyRPMD/.venv/bin/python'), '-m', 'sphinx', '-W', '--keep-going',
               '-b', 'html', 'docs', str(ROOT / 'convergence-fifth-docs-html')]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    log = OUTPUT / 'documentation-fifth-stage-build.log'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    unchanged = hashes == {name: sha(REPO / name) for name in sorted(paths)}
    receipt = {
        'schema': 'pyrpmd.complete-cohort-documentation-check.v1', 'started_utc': started,
        'command': command, 'exit_code': result.returncode,
        'elapsed_seconds': time.monotonic() - clock, 'log_sha256': sha(log),
        'all_selected_sources_unchanged_during_build': unchanged,
        'scope': 'Strict English Sphinx HTML; immutable historical evidence is excluded by Sphinx and checked separately for local links and parseability.',
        'source_hashes': {name: value for name, value in hashes.items() if name.endswith('.md')},
    }
    with (OUTPUT / 'documentation-fifth-stage-check.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    audit = {
        'schema': 'pyrpmd.complete-cohort-publication-audit.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'inventories_sha256': bound, 'distinct_verified_files': len(paths),
        'format_and_local_link_counts': counts, 'missing_local_targets': unexpected_missing,
        'relocated_historical_links_checked_at_recorded_original_base': rebased_historical_links,
        'historical_link_context_sha256': sha(context_path),
        'own_generated_receipt_links_deferred_until_outputs_written': missing,
        'all_selected_sources_unchanged_during_build': unchanged,
        'raw_arrays_and_external_models_excluded_by_whitelist': True,
        'allowed_npz': staged['allowed_derived_npz'],
        'staged_tree_at_start': staged['expected_tree'],
        'initial_whitespace_checks': {
            'staging_log_sha256': sha(ROOT / 'convergence-fifth-staging.log'),
            'initial_result': 'Rejected historical CI traceback excerpt whitespace. Preserved all five exact-bound completed log excerpts without normalization; no executed evidence bytes changed.',
            'resolved_staging_log_sha256': sha(ROOT / 'convergence-fifth-staging-preserved-excerpts.log'),
            'policy': 'Byte-exact generated logs, SVG, XML, CSV, archived patches and five explicit CI failure excerpts retain producer formatting. All remaining diagnostics were inspected; no new frozen-source EOF exception was needed. All authored-source checks pass.',
        },
        'frozen_source_EOF_blank_lines_preserved': staged['frozen_source_EOF_blank_lines_preserved'],
        'whitespace_exclusions': staged['whitespace_check_exclusions'],
        'driver_sha256': sha(Path(__file__)),
        'explicit_test_only_supersession': supersession_receipt,
        'additional_test_only_supersession': hch4,
    }
    with (OUTPUT / 'publication-fifth-stage-audit.json').open('x') as stream:
        stream.write(json.dumps(audit, indent=2) + '\n')
    for item in missing:
        assert ((REPO / item['source']).parent / item['target']).exists(), item
    if not unchanged or result.returncode != 0:
        raise RuntimeError('Publication documentation gate failed')
    print(json.dumps({'files': len(paths), 'inventories': len(bound), 'counts': counts,
                      'strict_sphinx_exit': result.returncode, 'elapsed_seconds': receipt['elapsed_seconds']}))


if __name__ == '__main__':
    main()
