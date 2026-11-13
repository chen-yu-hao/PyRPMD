"""Render one changelog section to stdout; publication is a separate action."""
import argparse
from pathlib import Path


def render(changelog, version):
    lines = Path(changelog).read_text().splitlines()
    heading = "## " + version
    try:
        start = lines.index(heading) + 1
    except ValueError as error:
        raise ValueError(f"No exact changelog section {version!r}") from error
    stop = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = "\n".join(lines[start:stop]).strip()
    if not body:
        raise ValueError("Release notes cannot be empty")
    return "# " + version + "\n\n" + body + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changelog", type=Path, default=Path(__file__).resolve().parents[1] / "CHANGELOG.md")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(render(args.changelog, args.version), end="")


if __name__ == "__main__":
    main()
