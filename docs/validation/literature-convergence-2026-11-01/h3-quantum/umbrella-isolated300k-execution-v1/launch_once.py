"""One exact exec handoff; the frozen owner enforces private worker lifetimes."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'PyRPMD'
OUT = REPO / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-isolated300k-execution-v1'
SOURCE = REPO / 'benchmarks/h3/quantum_isolated_continuation.py'

def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

adoption_path = OUT / 'root-adoption.json'
adoption = json.loads(adoption_path.read_text())
assert adoption['adopted'] and adoption['source_sha256'] == identity(SOURCE)['sha256']
for key in ('actual_resource_receipt', 'independent_review', 'test_gate', 'implementation_inventory'):
    row = adoption[key]
    assert identity(Path(row['path'])) == row
resource_receipt = json.loads(Path(adoption['actual_resource_receipt']['path']).read_text())
free = shutil.disk_usage(BASE).free
assert free >= resource_receipt['joint_required_incremental_bytes']
python = REPO / '.venv/bin/python'
command = [str(python), str(SOURCE), '--authorization', str(adoption_path)]
variables = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}
variables['PYTHONPATH'] = 'src:.'
receipt = {'schema': 'pyrpmd.h3.isolated300k-root-exec-handoff.v1',
    'recorded_immediately_before_exec_utc': datetime.now(timezone.utc).isoformat(),
    'adoption': identity(adoption_path), 'source': identity(SOURCE), 'launcher': identity(Path(__file__)),
    'command': command, 'cwd': str(REPO), 'environment_overrides': variables,
    'free_bytes_immediately_before_exec': free, 'actual_terminal_observation_required_later': True,
    'automatic_retry_or_extension': False, 'scientific_analysis_authorized': False}
with (OUT / 'root-launch.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
with (OUT / 'launch_once.py').open('xb') as stream:
    stream.write(Path(__file__).read_bytes())
print(json.dumps({'launch_receipt': str(OUT / 'root-launch.json'), 'fixed_stage': 300000,
                  'workers': 4, 'seeds': 8, 'status': 'exec handoff; no completion claimed'}), flush=True)
os.chdir(REPO)
os.execve(str(python), command, dict(os.environ, **variables))
