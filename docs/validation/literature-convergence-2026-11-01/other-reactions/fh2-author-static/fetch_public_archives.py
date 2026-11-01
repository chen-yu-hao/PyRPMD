"""Fetch two explicitly linked public author archives once, without execution."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.error
import urllib.request

OUT = Path('/workspace/scratch/a5e4820daedd/fh2-author-assets')
BASE = 'https://www.itheoc.uni-stuttgart.de/research/werner/research/h2f/potential/'
LIMIT = 16 * 1024 * 1024


def main():
    OUT.mkdir(exist_ok=False)
    records = []
    for name in ['sw1.tar', 'hsw.tar']:
        row = {'url': BASE + name, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'catalog': BASE, 'attempts': 1, 'authentication': False}
        try:
            request = urllib.request.Request(BASE + name, headers={'User-Agent': 'PyRPMD-literature-reproducibility-audit/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(LIMIT + 1)
                if len(data) > LIMIT:
                    raise ValueError('Public archive exceeds the fixed 16 MiB per-file bound')
                row.update(status=response.status, effective_url=response.url,
                           content_type=response.headers.get('Content-Type'))
            path = OUT / name
            with path.open('xb') as stream:
                stream.write(data)
            row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), local_path=str(path))
            with tarfile.open(path, 'r:*') as archive:
                members = []
                for member in archive:
                    if member.isfile():
                        content = archive.extractfile(member).read()
                        members.append({'name':member.name, 'type':'file', 'bytes':len(content),
                                        'sha256':hashlib.sha256(content).hexdigest()})
                    else:
                        members.append({'name':member.name, 'type':str(member.type), 'bytes':member.size})
                row['members'] = members
            row['downloaded'] = True
        except Exception as error:
            row.update(downloaded=False, error=f'{type(error).__name__}: {error}')
        records.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'members'}), flush=True)
    receipt = {'schema':'pyrpmd.fh2.author-public-source-fetch.v1',
               'created_utc':datetime.now(timezone.utc).isoformat(), 'records':records,
               'extracted_to_filesystem':False, 'source_executed':False,
               'corrected_literature_rates_obtained':False, 'raw_uploads':False}
    (OUT / 'fetch-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
