"""Compile the local analytic C ABI PES with an existing C compiler."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess


def build(output=None):
    here = Path(__file__).resolve().parent
    target = Path(output).resolve() if output else here / "build" / "libexchange.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(os.environ.get("CC", "cc")) + [
        "-O3", "-std=c11", "-fPIC", "-shared",
        "-I", str(here.parents[2] / "native" / "include"),
        str(here / "exchange.c"), "-lm", "-o", str(target),
    ]
    subprocess.run(command, check=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    print(build(parser.parse_args().output))
