"""Check documentation links without a web server; prepare live URL checks.

Run after both language editions have been built. Markdown links in the two
READMEs are checked against the checkout; rendered HTML links are checked
against the actual files and anchors that Sphinx produced. Links to this
repository's ``main`` branch are also resolved locally, so new pages work in a
pull request before they exist on main. Other HTTP(S) URLs are exported to a
small Sphinx project for the separate, live ``linkcheck`` CI job.

The parser for Markdown is a dependency of the existing ``docs`` extra. No
network request is made by this script. The live check does not ignore failed
requests or disable TLS verification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


REPOSITORY = "chen-yu-hao/PyRPMD"
README_LIMIT = 12_000
MARKDOWN = MarkdownIt("commonmark").enable("table")


@dataclass
class Page:
    anchors: set[str] = field(default_factory=set)
    links: list[tuple[int, str]] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)
    explicit_ids: list[str] = field(default_factory=list)


class HTMLLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = Page()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        identifier = values.get("id") or (values.get("name") if tag == "a" else None)
        if identifier:
            self.page.anchors.add(identifier)
            self.page.explicit_ids.append(identifier)
        for attribute in ("href", "src", "action"):
            if values.get(attribute):
                self.page.links.append((self.getpos()[0], values[attribute]))

    handle_startendtag = handle_starttag


def parse_html(text: str) -> Page:
    parser = HTMLLinks()
    parser.feed(text)
    return parser.page


def github_slug(text: str) -> str:
    """GitHub heading IDs for prose/code headings, including Unicode letters."""
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def parse_markdown(text: str) -> Page:
    page = Page()
    tokens = MARKDOWN.parse(text)
    seen: Counter[str] = Counter()
    heading_ids: set[str] = set()
    for index, token in enumerate(tokens):
        line = token.map[0] + 1 if token.map else 1
        if token.type == "heading_open":
            inline = tokens[index + 1]
            title = "".join(
                child.content for child in inline.children or []
                if child.type in {"text", "code_inline", "image"}
            )
            page.headings.append((int(token.tag[1:]), title))
            slug = github_slug(title)
            number = seen[slug]
            identifier = f"{slug}-{number}" if number else slug
            while identifier in heading_ids:
                number += 1
                identifier = f"{slug}-{number}"
            page.anchors.add(identifier)
            heading_ids.add(identifier)
            seen[slug] = number + 1
        if token.type == "html_block":
            html = parse_html(token.content)
            page.anchors.update(html.anchors)
            page.explicit_ids.extend(html.explicit_ids)
            page.links.extend((line + n - 1, url) for n, url in html.links)
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type in {"link_open", "image"}:
                url = child.attrGet("href") or child.attrGet("src")
                if url:
                    page.links.append((line, url))
            elif child.type == "html_inline":
                html = parse_html(child.content)
                page.anchors.update(html.anchors)
                page.explicit_ids.extend(html.explicit_ids)
                page.links.extend((line, url) for _, url in html.links)
    return page


class LinkChecker:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.pages: dict[Path, Page] = {}
        self.errors: list[str] = []
        self.external: dict[str, set[str]] = {}
        self.links_checked = 0
        self.git_commits: dict[str, bool] = {}
        self.git_targets: dict[tuple[str, str], tuple[str, str] | None] = {}
        self.git_blobs: dict[str, str | None] = {}

    def label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def error(self, path: Path, message: str) -> None:
        self.errors.append(f"{self.label(path)}: {message}")

    def page(self, path: Path) -> Page:
        path = path.resolve()
        if path not in self.pages:
            text = path.read_text(encoding="utf-8")
            self.pages[path] = (
                parse_markdown(text) if path.suffix.lower() == ".md"
                else parse_html(text)
            )
        return self.pages[path]

    def repository_target(self, url: str) -> Path | None:
        """Map canonical main URLs; historical links are checked separately."""
        parsed = urlsplit(url)
        path = unquote(parsed.path)
        prefixes: tuple[str, ...] = ()
        if parsed.netloc.lower() == "github.com":
            prefixes = (f"/{REPOSITORY}/blob/main/", f"/{REPOSITORY}/tree/main/")
        elif parsed.netloc.lower() == "raw.githubusercontent.com":
            prefixes = (f"/{REPOSITORY}/main/",)
        for prefix in prefixes:
            if path.startswith(prefix):
                return (self.root / path[len(prefix):]).resolve()
        return None

    def git(self, *args: str) -> subprocess.CompletedProcess:
        # A partial checkout must never fetch objects during this offline check.
        # Missing history/blobs are left in the external HTTP inventory instead.
        return subprocess.run(
            ["git", "--literal-pathspecs", *args], cwd=self.root,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )

    def check_pinned_link(self, source: Path, line: int, url: str) -> bool:
        """Check exact same-repo 40-SHA objects; never substitute today's file.

        Return False when the ref/blob is unavailable (including source
        archives without .git). Such URLs still receive a live HTTP check.
        A known commit with a missing path/anchor is a definite local failure.
        """
        parsed = urlsplit(url)
        if parsed.netloc.lower() != "github.com" or not (self.root / ".git").exists():
            return False
        match = re.fullmatch(
            rf"/{re.escape(REPOSITORY)}/(blob|tree)/([0-9a-f]{{40}})/(.+)",
            unquote(parsed.path),
        )
        if not match:
            return False
        kind, commit, path = match.groups()
        try:
            if commit not in self.git_commits:
                result = self.git("cat-file", "-t", commit)
                self.git_commits[commit] = result.returncode == 0 and result.stdout.strip() == b"commit"
            if not self.git_commits[commit]:
                return False
            key = commit, path
            if key not in self.git_targets:
                result = self.git("ls-tree", "-z", commit, "--", path)
                if result.returncode:
                    return False
                entry = next((item for item in result.stdout.split(b"\0") if item.partition(b"\t")[2].decode("utf-8") == path), None)
                self.git_targets[key] = tuple(entry.partition(b"\t")[0].decode().split()[1:]) if entry else None
            entry = self.git_targets[key]
            if entry is None or entry[0] != kind:
                self.error(source, f"line {line}: missing {kind} in exact commit {url!r}")
                return True
            fragment = unquote(parsed.fragment).split(":~:text=", 1)[0]
            if not fragment:
                return True
            if kind != "blob":
                return False  # GitHub may render a directory's README anchor.
            object_id = entry[1]
            if object_id not in self.git_blobs:
                result = self.git("cat-file", "blob", object_id)
                try:
                    self.git_blobs[object_id] = result.stdout.decode("utf-8") if result.returncode == 0 else None
                except UnicodeDecodeError:
                    self.git_blobs[object_id] = None
            text = self.git_blobs[object_id]
            if text is None:
                return False
        except FileNotFoundError:  # Git is optional for source-archive builds.
            return False
        line_range = re.fullmatch(r"L([1-9]\d*)(?:-L([1-9]\d*))?", fragment)
        if line_range:
            start = int(line_range[1])
            end = int(line_range[2] or start)
            valid = start <= end <= len(text.splitlines())
        elif Path(path).suffix.lower() == ".md":
            valid = fragment in parse_markdown(text).anchors
        else:
            return False  # Leave viewer-specific anchors to the HTTP inventory.
        if not valid:
            self.error(source, f"line {line}: missing fragment in exact commit {url!r}")
        return True

    def check_link(self, source: Path, line: int, url: str, base: Path) -> Path | None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"", "http", "https"}:
            return None  # mailto:, data:, javascript: are not document paths.
        self.links_checked += 1
        if self.check_pinned_link(source, line, url):
            return None
        target = self.repository_target(url)
        if target is None and (parsed.netloc or parsed.scheme):
            canonical = parsed._replace(fragment="").geturl()
            self.external.setdefault(canonical, set()).add(f"{self.label(source)}:{line}")
            return None
        if target is None:
            path = unquote(parsed.path)
            if not path:
                target = source
            elif path.startswith("/"):
                target = base / path.lstrip("/")
            else:
                target = source.parent / path
            target = target.resolve()
        if not target.exists():
            self.error(source, f"line {line}: missing target {url!r}")
            return None
        if target.is_dir():
            if (target / "index.html").is_file():
                target /= "index.html"
            elif (target / "README.md").is_file():
                target /= "README.md"
            elif parsed.fragment:
                self.error(source, f"line {line}: directory has no anchor target {url!r}")
                return None
        fragment = unquote(parsed.fragment).split(":~:text=", 1)[0]
        if fragment and target.suffix.lower() in {".md", ".html", ".htm", ".svg"}:
            if fragment not in self.page(target).anchors:
                self.error(source, f"line {line}: missing fragment {url!r}")
        return target

    def check_page(self, path: Path, base: Path) -> set[Path]:
        targets = set()
        for line, url in self.page(path).links:
            target = self.check_link(path, line, url, base)
            if target is not None:
                targets.add(target)
        return targets

    def check_readmes(self) -> None:
        paths = [self.root / "README.md", self.root / "README.zh-CN.md"]
        for path in paths:
            if not path.is_file():
                self.error(path, "README is missing")
                return
            size = path.stat().st_size
            if size > README_LIMIT:
                self.error(path, f"{size} bytes exceeds the {README_LIMIT}-byte README budget")
            self.check_page(path, self.root)
        en, zh = (self.page(path) for path in paths)
        if [level for level, _ in en.headings] != [level for level, _ in zh.headings]:
            self.error(paths[1], "heading levels/order must match README.md")
        if not en.explicit_ids or en.explicit_ids != zh.explicit_ids:
            self.error(paths[1], "explicit section IDs/order must match README.md")

    def check_chapters(self) -> None:
        docs = self.root / "docs"
        en = {path.name for path in (docs / "manual").glob("*.md")}
        zh = {path.name for path in (docs / "zh" / "manual").glob("*.md")}
        if en != zh:
            self.error(docs, f"manual chapter mismatch: English-only {sorted(en - zh)}; Chinese-only {sorted(zh - en)}")

    def check_index_parity(self) -> None:
        """Both human-facing indexes must expose the same Markdown guides."""
        docs = self.root / "docs"
        indexes = [docs / "index.md", docs / "zh" / "index.md"]
        if not all(path.is_file() for path in indexes):
            self.error(docs, "both language indexes must exist")
            return
        if [level for level, _ in self.page(indexes[0]).headings] != [level for level, _ in self.page(indexes[1]).headings]:
            self.error(indexes[1], "heading levels/order must match docs/index.md")

        def guides(source: Path) -> set[str]:
            result = set()
            for _, url in self.page(source).links:
                parsed = urlsplit(url)
                path = self.repository_target(url)
                if path is None and not (parsed.netloc or parsed.scheme):
                    path = (source.parent / unquote(parsed.path)).resolve()
                if path is None or path.suffix != ".md" or not path.is_relative_to(docs):
                    continue
                relative = path.relative_to(docs)
                if relative.parts[0] == "zh":
                    relative = Path(*relative.parts[1:])
                if relative.as_posix() != "index.md":
                    result.add(relative.as_posix())
            return result

        en, zh = map(guides, indexes)
        if en != zh:
            self.error(indexes[1], f"guide links differ: English-only {sorted(en - zh)}; Chinese-only {sorted(zh - en)}")

    def check_html(self, directory: Path) -> int:
        directory = directory.resolve()
        if not (directory / "index.html").is_file():
            self.error(directory, "HTML build is missing index.html")
            return 0
        # Themes copy Jinja macro templates into _static; they are not rendered
        # pages. Assets in that directory are still checked when pages link them.
        paths = sorted(
            path for path in directory.rglob("*.html")
            if "_static" not in path.relative_to(directory).parts
        )
        edges = {path: self.check_page(path, directory) for path in paths}
        reachable, pending = set(), [directory / "index.html"]
        while pending:
            path = pending.pop()
            if path in reachable:
                continue
            reachable.add(path)
            pending.extend(edges.get(path, set()) - reachable)
        for path in paths:
            source = self.root / "docs" / path.relative_to(directory).with_suffix(".md")
            if source.is_file() and path not in reachable:
                self.error(path, "built guide is not reachable from index.html")
        return len(paths)

    def write_external_project(self, directory: Path) -> None:
        """Save URLs with provenance; Sphinx performs real network validation."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "conf.py").write_text(
            'project = "PyRPMD external documentation links"\n'
            'root_doc = "index"\n'
            'source_suffix = {".rst": "restructuredtext"}\n'
            '# Local fragments are validated against generated HTML by check_links.py.\n'
            '# Remote anchors often depend on publisher JavaScript; check HTTP targets.\n'
            'linkcheck_anchors = False\n'
            'linkcheck_retries = 2\n'
            'linkcheck_timeout = 20\n'
            '# One worker avoids bursts at DOI registries and publisher sites.\n'
            'linkcheck_workers = 1\n'
            'linkcheck_rate_limit_timeout = 120.0\n'
            'linkcheck_report_timeouts_as_broken = True\n'
            '# DOI content negotiation validates registered citation metadata.\n'
            '# It avoids treating publisher bot/login walls as invalid DOIs.\n'
            'linkcheck_request_headers = {"https://doi.org": {\n'
            '    "Accept": "application/vnd.citationstyles.csl+json"\n'
            '}}\n',
            encoding="utf-8",
        )
        lines = ["External documentation links", "============================", ""]
        for index, url in enumerate(sorted(self.external), 1):
            # Raw URLs are kept as targets, not interpreted as RST inline markup.
            lines.extend((f"* `URL {index} <{url}>`_", ""))
        (directory / "index.rst").write_text("\n".join(lines), encoding="utf-8")
        (directory / "sources.json").write_text(
            json.dumps({url: sorted(refs) for url, refs in sorted(self.external.items())}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (directory / "check_external_results.py").write_text(
            Path(__file__).with_name("check_external_results.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--html", type=Path, action="append", default=[], help="Sphinx build root, including its zh/ edition (repeatable)")
    parser.add_argument("--external-project", type=Path, help="Write a Sphinx project for the separate HTTP linkcheck")
    args = parser.parse_args()
    checker = LinkChecker(args.root)
    checker.check_readmes()
    checker.check_chapters()
    checker.check_index_parity()
    html_count = sum(checker.check_html(directory) for directory in args.html)
    if args.external_project:
        checker.write_external_project(args.external_project)
    for error in checker.errors:
        print(error, file=sys.stderr)
    print(f"Checked 2 READMEs and {html_count} HTML pages: {checker.links_checked} links, {len(checker.external)} external URLs, {len(checker.errors)} errors.")
    return int(bool(checker.errors))


if __name__ == "__main__":
    raise SystemExit(main())
