"""Fetch the exact public model, checking archive and extracted-file hashes."""
from pathlib import Path
import hashlib
import io
import tarfile
import urllib.request

MODEL_SHA256 = "64a2a91636d11c091585aaf1bc72af6397e9672a0a3bd02602f95ec1772fb6f4"
ARCHIVE_SHA256 = "3069b69b0f768b907c7d353a355d6be1ab46f638ac674f336b51af3a87debd47"
URL = "https://zenodo.org/api/records/21401971/files/trained_pes.tar.gz/content"


def main():
    target = Path(__file__).resolve().parent / "models" / "REANN_PES_DOUBLE.pt"
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == MODEL_SHA256:
        print(f"Verified existing model: {target}")
        return
    with urllib.request.urlopen(URL, timeout=90) as response:
        archive = response.read(4 * 1024 * 1024)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Published archive hash differs; no model installed")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        members = [m for m in package.getmembers() if m.isfile() and Path(m.name).name == target.name]
        if len(members) != 1 or members[0].size > 4 * 1024 * 1024:
            raise ValueError("Unexpected model archive layout")
        model = package.extractfile(members[0]).read()
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise ValueError("Extracted model hash differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(model)
    temporary.replace(target)
    print(f"Verified model installed: {target}")


if __name__ == "__main__":
    main()
