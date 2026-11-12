#!/usr/bin/env python3
"""Run the compact issue-191 subspace reproduction against a saved rate run."""
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rate_run = args.rate_run.resolve()
    output = args.output.resolve()
    if not rate_run.is_dir():
        parser.error(f"rate-run is not a directory: {rate_run}")
    output.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("subspace.toml").read_text()
    template = re.sub(r'(?m)^rate_run = .*$', "rate_run = " + json.dumps(str(rate_run)), template)
    template = re.sub(r'(?m)^directory = .*$', "directory = " + json.dumps(str(output)), template)
    with tempfile.TemporaryDirectory(prefix="pyrpmd-issue-191-") as work:
        config = Path(work) / "subspace.toml"
        config.write_text(template)
        completed = subprocess.run([sys.executable, "-m", "pyrpmd", "subspace-correct", str(config)])
    return completed.returncode

if __name__ == "__main__":
    raise SystemExit(main())
