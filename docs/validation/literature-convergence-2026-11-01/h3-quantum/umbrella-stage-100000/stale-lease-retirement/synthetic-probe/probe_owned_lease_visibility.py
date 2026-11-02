"""Observe an owned synthetic directory during a separate rename; no campaign I/O."""
from pathlib import Path
import json
import os
import time

BASE = Path('/workspace/scratch/a5e4820daedd/owned-lease-visibility-probe-20261101')


def main():
    BASE.mkdir(exist_ok=False)
    old = BASE / 'initial-directory'
    old.mkdir()
    owner = old / 'owner.json'
    owner.write_text('{"synthetic":true,"created_by":"probe"}\n')
    with (BASE / 'observations.jsonl').open('x') as output:
        for i in range(25):
            row = {'index': i, 'time_ns': time.time_ns(), 'directory_exists': old.exists(),
                   'owner_exists': owner.exists()}
            if owner.exists():
                row['owner_inode'] = owner.stat().st_ino
            output.write(json.dumps(row) + '\n')
            output.flush()
            os.fsync(output.fileno())
            time.sleep(1)
    print(json.dumps({'normal_exit': True, 'synthetic_only': True, 'no_old_directory_recreation_by_this_script': True}))


if __name__ == '__main__':
    main()
