"""Run `pyrpmd run` but raise KeyboardInterrupt after the N-th call of a native entry point.
Usage: python interrupt_cli.py <run_block|advance> <N> <cli args...>"""
import sys
import pyrpmd._native as native
from pyrpmd import cli
name, limit = sys.argv[1], int(sys.argv[2])
original, count = getattr(native, name), [0]
def wrapped(*args):
    result = original(*args)
    count[0] += 1
    if count[0] == limit:
        raise KeyboardInterrupt(f"injected after {name} call {limit}")
    return result
setattr(native, name, wrapped)
try:
    cli.main(sys.argv[3:])
except KeyboardInterrupt as error:
    print("interrupted:", error)
