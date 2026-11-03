"""Require evidence for every exported URL, including Sphinx's ignored 503s."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def failures(results: list[dict], sources: dict[str, list[str]]) -> list[str]:
    by_url = {result["uri"]: result for result in results}
    errors = []
    for url in sorted(sources):
        result = by_url.get(url)
        if result is None:
            detail = "no completed check"
        elif result["status"] not in {"working", "redirected"}:
            detail = f"{result['status']}: {result.get('info', '')}"
        else:
            continue
        errors.append(f"{url}: {detail}\n  referenced by " + ", ".join(sources[url]))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("sources", type=Path)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    results = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines()] if args.results.is_file() else []
    errors = failures(results, sources)
    for error in errors:
        print(error)
    print(f"External URL results: {dict(Counter(result['status'] for result in results))}; {len(errors)} unverified or failed URLs out of {len(sources)}.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
