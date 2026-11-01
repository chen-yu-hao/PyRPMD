"""Make a submission recipe from an actual sdist and explicit maintainers.

The local-source recipe is intentionally not a conda-forge submission: no release
URL, checksum, or maintainer consent is invented. This command records the exact
archive hash and confirms its embedded metadata before replacing the local path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tomllib
from urllib.parse import urlparse


def render(sdist, url, maintainers):
    sdist = Path(sdist)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("source URL must be public HTTPS without credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("use a stable source URL without query parameters or a fragment")
    if not maintainers or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", x)
                              for x in maintainers):
        raise ValueError("provide consenting maintainers' GitHub usernames")
    with tarfile.open(sdist, "r:gz") as archive:
        matches = [m for m in archive.getmembers()
                   if m.isfile() and len(Path(m.name).parts) == 2 and m.name.endswith("/pyproject.toml")]
        if len(matches) != 1:
            raise ValueError("sdist must have one top-level project pyproject.toml")
        metadata = tomllib.loads(archive.extractfile(matches[0]).read().decode("utf-8"))["project"]
    if metadata["name"].lower() != "pyrpmd":
        raise ValueError("archive is not PyRPMD")
    version = metadata["version"]
    if not re.fullmatch(r"[A-Za-z0-9.+!-]+", version):
        raise ValueError("invalid package version")
    text = Path(__file__).with_name("recipe.yaml").read_text(encoding="utf-8")
    text = re.sub(r'  version: "[^"]+"', f'  version: "{version}"', text, count=1)
    digest = hashlib.sha256(sdist.read_bytes()).hexdigest()
    text = text.replace("  path: ../..", f"  url: {json.dumps(url)}\n  sha256: {digest}", 1)
    return text + "extra:\n  recipe-maintainers:\n" + "".join(f"    - {x}\n" for x in maintainers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--url", required=True, help="public URL serving these exact archive bytes")
    parser.add_argument("--maintainer", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new recipe path")
    content = render(args.sdist, args.url, args.maintainer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}. Verify that the public URL serves the same SHA256 before submission.")


if __name__ == "__main__":
    main()
