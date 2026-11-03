"""Regression for duplicate headings and Sphinx's double-fragment links."""

import unittest

from bs4 import BeautifulSoup

from sphinx_singlehtml import rewrite_page


class SingleHTMLLinks(unittest.TestCase):
    def test_duplicate_headings_resolve_to_their_own_documents(self):
        text = '''<html><head><link rel="search" href="search.html"></head>
        <body><a href="#document-a#references">A</a>
        <article class="bd-article"><section id="home">
        <span id="document-a"></span><section id="a"><h2 id="references">A refs</h2>
        <a href="#references">Self A</a><a href="#document-b#references">B refs</a></section>
        <span id="document-b"></span><section id="b"><h2 id="references">B refs</h2>
        <a href="#references">Self B</a><a href="#document-index">Home</a></section>
        </section></article><form action="search.html"></form></body></html>'''
        soup = BeautifulSoup(rewrite_page(text, "index", "../html"), "html.parser")
        ids = [tag["id"] for tag in soup.find_all(id=True)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("document-index", ids)
        self.assertIn("document-a--references", ids)
        self.assertIn("document-b--references", ids)
        for a in soup.find_all("a"):
            self.assertIn(a["href"][1:], ids)
        self.assertEqual(soup.find("a", string="Self A")["href"], "#document-a--references")
        self.assertEqual(soup.find("a", string="Self B")["href"], "#document-b--references")
        self.assertEqual(soup.find("a", string="B refs")["href"], "#document-b--references")
        self.assertEqual(soup.find("form")["action"], "../html/search.html")

    def test_duplicate_ids_within_one_document_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            rewrite_page('<article class="bd-article"><h1 id="same">A</h1><p id="same">B</p></article>', "index", "../html")

    def test_accessibility_references_and_theme_widgets(self):
        text = '''<html><body><div id="control">Theme</div><label for="control">Theme control</label>
        <article class="bd-article"><h1 id="control">Document heading</h1>
        <p aria-labelledby="control">Text</p><table><tr><th id="column">Head</th>
        <td headers="column">Data</td></tr></table></article></body></html>'''
        soup = BeautifulSoup(rewrite_page(text, "index", "../html"), "html.parser")
        self.assertIsNotNone(soup.find("div", id="control"))
        self.assertEqual(soup.find("label")["for"], "control")
        self.assertEqual(soup.find("p")["aria-labelledby"], "document-index--control")
        self.assertEqual(soup.find("td")["headers"], ["document-index--column"])

    def test_ambiguous_unqualified_link_is_rejected(self):
        text = '''<article class="bd-article"><a href="#references">Where?</a>
        <span id="document-a"></span><section><h2 id="references">A</h2></section>
        <span id="document-b"></span><section><h2 id="references">B</h2></section></article>'''
        with self.assertRaisesRegex(ValueError, "ambiguous section link"):
            rewrite_page(text, "index", "../html")

    def test_already_processed_output_is_unchanged(self):
        text = '''<article class="bd-article"><section id="home"><span id="citation">Ref</span>
        <a href="#citation">citation</a></section></article>'''
        once = rewrite_page(text, "index", "../html")
        self.assertEqual(rewrite_page(once, "index", "../html"), once)


if __name__ == "__main__":
    unittest.main()
