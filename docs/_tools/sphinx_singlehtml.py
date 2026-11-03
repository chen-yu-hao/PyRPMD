"""Keep Sphinx 8 single-page links unambiguous and usable.

Sphinx 8.2's singlehtml builder can emit ``#document-name#section`` links while
leaving section IDs unqualified. Merely stripping the document name would send
repeated headings (for example, "References") to the wrong chapter. This
build-finished hook instead namespaces IDs inside each included document and
rewrites links to those exact targets. It runs only for singlehtml and fails on
duplicate IDs inside a document or ambiguous unqualified references.

The pydata theme also offers search/index pages which singlehtml does not emit.
Those controls use the searchable sibling edition. Build the normal edition at
``<out>/html`` and the single-page edition at ``<out>/singlehtml`` (Chinese in
each ``zh/``), as in the documentation CI. ``singlehtml_searchable_root`` can
override the relative path when publishing a different directory layout.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag
from sphinx.errors import SphinxError


def rewrite_page(text: str, root_doc: str, searchable_root: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    if soup.head and soup.head.find("meta", attrs={"name": "pyrpmd-singlehtml-links", "content": "1"}):
        return text
    article = soup.select_one("article.bd-article")
    if article is None:
        raise ValueError("singlehtml page has no article.bd-article document container")

    # Sphinx emits a document marker followed by that document's outer section.
    # The root document has no start_of_file node, so add its missing marker.
    scopes: dict[int, str] = {}
    for marker in article.find_all("span", id=lambda value: value and value.startswith("document-")):
        section = marker.find_next_sibling("section")
        if section is None:
            raise ValueError(f"document marker {marker['id']!r} has no section")
        scopes[id(section)] = marker["id"][len("document-"):]
    root_marker = "document-" + root_doc
    if soup.find(id=root_marker) is None:
        article.insert(0, soup.new_tag("span", id=root_marker))

    origins: dict[int, str] = {}
    targets: dict[tuple[str, str], str] = {}
    by_anchor: dict[str, list[str]] = defaultdict(list)

    def qualify(tag: Tag, document: str) -> None:
        document = scopes.get(id(tag), document)
        origins[id(tag)] = document
        original = tag.get("id")
        if original and not original.startswith("document-"):
            key = document, original
            if key in targets:
                raise ValueError(f"duplicate ID {original!r} inside document {document!r}")
            qualified = f"document-{document}--{original}"
            targets[key] = qualified
            by_anchor[original].append(qualified)
            tag["id"] = qualified
        for child in tag.children:
            if isinstance(child, Tag):
                qualify(child, document)

    qualify(article, root_doc)
    theme_anchors = {
        tag["id"] for tag in soup.find_all(id=True)
        if id(tag) not in origins
    }

    def target(fragment: str, document: str, outside_article: bool = False) -> str:
        fragment = unquote(fragment)
        if outside_article and fragment in theme_anchors:
            return fragment
        if fragment.startswith("document-"):
            name, separator, anchor = fragment[len("document-"):].partition("#")
            if separator:
                return targets.get((name, anchor), fragment)
            return fragment
        exact = targets.get((document, fragment))
        if exact:
            return exact
        candidates = by_anchor.get(fragment, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous section link #{fragment} from document {document!r}")
        return fragment  # Theme IDs remain untouched; missing IDs fail check_links.

    for tag in soup.find_all(True):
        document = origins.get(id(tag), root_doc)
        outside_article = id(tag) not in origins
        for attribute in ("href", "action"):
            value = tag.get(attribute)
            if not isinstance(value, str):
                continue
            if value.startswith("#"):
                tag[attribute] = "#" + target(value[1:], document, outside_article)
            elif value in {"search.html", "genindex.html"}:
                tag[attribute] = searchable_root.rstrip("/") + "/" + value
        for attribute in ("for", "headers", "aria-labelledby", "aria-describedby", "aria-controls"):
            values = tag.get(attribute)
            if values:
                values = values if isinstance(values, list) else values.split()
                tag[attribute] = " ".join(target(value, document, outside_article) for value in values)

    # A no-op/incremental builder or another extension may retain this output.
    # Do not reinterpret our namespaced span IDs as fresh document markers.
    if soup.head is None:
        head = soup.new_tag("head")
        (soup.html if soup.html is not None else soup).insert(0, head)
    soup.head.append(soup.new_tag("meta", attrs={"name": "pyrpmd-singlehtml-links", "content": "1"}))
    return str(soup)


def finish(app, exception) -> None:
    if exception is not None or app.builder.name != "singlehtml":
        return
    path = Path(app.outdir) / (app.config.root_doc + ".html")
    searchable_root = app.config.singlehtml_searchable_root
    if searchable_root is None:
        searchable_root = "../../html/zh" if app.config.language.startswith("zh") else "../html"
    try:
        output = rewrite_page(path.read_text(encoding="utf-8"), app.config.root_doc, searchable_root)
    except ValueError as error:
        raise SphinxError(f"singlehtml link correction: {error}") from error
    path.write_text(output, encoding="utf-8")


def setup(app):
    app.add_config_value("singlehtml_searchable_root", None, "html", types=frozenset({str, type(None)}))
    app.connect("build-finished", finish)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
